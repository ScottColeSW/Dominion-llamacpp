"""The show's Host, as a real agent -- same ScriptedAgent/LLMAgent split
every player already has (see scripted_agent.py/llm_agent.py), applied
to the one role that was, until now, never actually a decision-maker at
all: every host line was a fixed template picked client-side in
web/index.html, disconnected from a second, separate, NEVER-DISPLAYED
fabricated "host said this" quote llm_agent.py's old
_host_announcement_line built purely to give players something to
react to. Those two texts were never the same string. Making the Host a
real agent means there's exactly one host line per beat, generated once,
that's both what's actually spoken/shown AND what players react to.

v1 scope, deliberately narrow: the challenge announcement (once per
duel) and the finale line (once per show) -- the two beats that already
sit next to existing live calls (intro_line_combined x2 per duel), so
adding one more live call here doesn't change the show's pacing
tolerance. Every other host line (stillThinking/intro/postduel/continue
buildup-and-reaction/preduelHostLine's streak-hype) stays exactly as it
is today: scripted, frontend-only, untouched.
"""
from __future__ import annotations

import random

from ..inference.base import InferenceClient
from ..inference.config import settings
from ..engine.models import Player
from .llm_agent import _trim_to_last_sentence

# Scott: "perhaps the Host needs a name to introduce the themselves and
# the show." Rex Domain -- Rex (king/ruler, matching the show's
# kingdoms) + Domain (the trivia category every duel is fought over, and
# the show's own title). Both the scripted templates and the live LLM
# prompts below use this name; web/index.html's static welcome line is
# the actual self-introduction moment ("Good evening, I'm Rex Domain...").
HOST_NAME = "Rex Domain"


class ScriptedHostAgent:
    """Ports the actual template pools from web/index.html's
    hostPreduelQuestion and finaleHostLine into Python -- same content,
    relocated, one shared announcement per duel instead of the two
    near-duplicate per-side questions the old frontend-only version
    asked. This is also LLMHostAgent's fallback on any failure, so it
    has to work standalone, same contract every other agent already has.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    async def announce_challenge(self, challenger: Player, defender: Player,
                                  tested_domain: str) -> str:
        same_domain = defender.origin_domain == tested_domain
        stakes_clause = (
            "and that's exactly the category on the line tonight" if same_domain
            else f"but tonight's category is {tested_domain}, a different subject entirely"
        )
        templates = [
            f"{challenger.kingdom_name} steps up to challenge {defender.kingdom_name} on "
            f"{tested_domain}. {defender.kingdom_name}, that's your ground to defend.",
            f"Here we go -- {challenger.kingdom_name} against {defender.kingdom_name}, "
            f"fighting it out on {tested_domain}. Only one of you walks away from this "
            f"with anything.",
            f"{defender.kingdom_name}, word is you know {defender.origin_domain} well, "
            f"{stakes_clause}. {challenger.kingdom_name} is coming for it anyway.",
        ]
        return self.rng.choice(templates)

    async def announce_finale(self, champion: Player, total_duels: int, prize: int) -> str:
        # Scott: "the stakes are points and bragging rights as well as
        # the $100M (fake dollars) award!" -- the prize alone was
        # carrying the entire finale line; every template now also names
        # the thing that's actually harder to fake: sole ownership of
        # the whole board, the bragging rights that come with it.
        templates = [
            f"Ladies and gentlemen, {champion.kingdom_name} is our champion -- every tile "
            f"on this board, and all the bragging rights that come with it!",
            f"And there it is. {champion.kingdom_name}, sole owner of the board! Nobody's "
            f"got a rebuttal for that.",
            f"What a run! What a show! {champion.kingdom_name} takes it all -- the board, "
            f"the crown, and the prize!",
            f"One board, one champion, and it's {champion.kingdom_name}! That's the whole "
            f"stack, right there. This has been {HOST_NAME} -- goodnight, everybody!",
        ]
        return self.rng.choice(templates)


class LLMHostAgent(ScriptedHostAgent):
    """Same interface as ScriptedHostAgent; overrides both announcements
    to consult a real model via an InferenceClient, falling back to the
    inherited scripted lines whenever the call doesn't yield a usable
    answer -- identical fallback contract to LLMAgent."""

    def __init__(self, rng: random.Random, model: str, client: InferenceClient, **kwargs):
        super().__init__(rng, **kwargs)
        self.model = model
        self.client = client

    async def announce_challenge(self, challenger: Player, defender: Player,
                                  tested_domain: str) -> str:
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show, in front of a "
            f"real studio audience. Announce the next matchup, live, on air, in ONE or TWO "
            f"short punchy sentences a real game show host would actually say out loud. "
            f"You don't need to say your own name in every line -- only if it actually fits "
            f"naturally.\n"
            f"Challenger: {challenger.kingdom_name} ({challenger.profession}).\n"
            f"Defender: {defender.kingdom_name} ({defender.profession}), whose real "
            f"expertise is {defender.origin_domain}.\n"
            f"Domain on the line tonight: {tested_domain} -- this is the DEFENDER's own "
            f"territory being attacked, so name the stakes for {defender.kingdom_name} "
            f"specifically.\n"
            f"Reply with ONLY the announcement itself, no stage directions, no quotation "
            f"marks."
        )
        # num_predict=80, not intro_line_combined's tighter 55 -- this
        # announcement has to cover BOTH players' names/professions plus
        # the domain, objectively more content than one player's own
        # first-person reaction. _trim_to_last_sentence cleans up
        # whatever the cap still cuts off mid-sentence (observed live:
        # without it, a real reply cut off mid-word -- "he'll be driving
        # a different route for a").
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=80)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().announce_challenge(
            challenger, defender, tested_domain)

    async def announce_finale(self, champion: Player, total_duels: int, prize: int) -> str:
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show. The show just "
            f"ended: {champion.kingdom_name} ({champion.profession}) is the sole owner of "
            f"the entire board after {total_duels} duels -- every tile, the bragging rights "
            f"that come with owning all of it, AND a ${prize:,} grand prize. "
            f"Announce this live, on air, in ONE or TWO short triumphant sentences a real "
            f"game show host would actually say -- the board and the bragging rights "
            f"matter here every bit as much as the prize money, so don't reduce this to "
            f"just a dollar figure. This is your last chance to sign off for the night, so "
            f"a closing line in your own name is a nice touch here if it fits naturally, "
            f"but isn't required. Reply with ONLY the announcement itself, no stage "
            f"directions, no quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=70)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().announce_finale(
            champion, total_duels, prize)


def host_model_for_backend(backend: str) -> str:
    """Which model plays the Host -- a fixed, dedicated choice per
    backend (inference/config.py), independent of the contestant model
    pool (TEXT_MODELS/LLAMACPP_MODELS) the same way those two rosters are
    already independently declared from each other. Not exposed in the
    M8 model picker -- a deliberate v1 cut."""
    if backend == "llamacpp":
        return settings.host_model_llamacpp
    return settings.host_model_ollama
