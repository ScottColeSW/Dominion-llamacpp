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

v1 (M9) scope was deliberately narrow: just the challenge announcement
and the finale line. M32 ("less scripted, more dynamic agent
conversation") added three more real beats, chosen specifically for low
call frequency (once per duel or rarer) so the added live-call cost
never approaches attempt_question's per-turn volume:
announce_challenge gained streak/territory context (replacing the
separate, disconnected preduelHostLine); announce_duel_result merges
the win reaction and the loser's goodbye into one call (replacing 4
scripted pools); announce_continue_decision reacts to the
already-live decide_continue outcome. Still scripted, frontend-only,
deliberately untouched: stillThinking (exists specifically to fill dead
air while ALREADY waiting on a live call), the per-turn reaction pools
(REACTION_EXCLAIM/LUCKY_BLURT/PASS_HOST_LINES), hostContinueBuildup
(nothing dynamic to react to yet), and the rare special-event lines
(advantageHostLine, scrambleHostLine, etc.) -- see the
project_m32_dynamic_agent_investigation memory for the full inventory
and why each tier landed where it did.
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
# Scott, from outside research on what makes an AI game host fun to
# watch: "An Opinionated System Prompt: Give the host clear biases or
# quirks." "The host still seems dull... did we glean any tips from the
# .md I gave?" -- genuinely missing before this; every prompt below only
# ever described the ROLE (announcer), never a personality. Builds on the
# king/ruler theme HOST_NAME already leans on (Rex = ruler) rather than
# reaching for something unrelated to this show -- a theatrical royal
# herald, not a neutral TV voice. Threaded into every one of this class's
# prompts below as its own short sentence, so it's a consistent trait
# every time the Host speaks, not a one-off flourish on a single line.
HOST_PERSONA = (
    "You're a theatrical royal herald at heart, and you treat every duel "
    "like a proclamation of conquest -- never miss a chance for a kingly "
    "flourish"
)


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
                                  tested_domain: str, challenger_streak: int = 0,
                                  defender_streak: int = 0, challenger_territory: int = 0,
                                  defender_territory: int = 0, final_duel: bool = False) -> str:
        # M32 Fix 1: ports web/index.html's old preduelHostLine -- a
        # SEPARATE scripted line that used to fire right before this
        # exact announcement, from its own streak/territory-aware
        # template pool. The Host was speaking twice per duel through
        # two disconnected code paths; this folds that tiered logic
        # (hot streak first, then a real size mismatch, then the
        # existing generic pool) directly into this one call instead.
        #
        # Scott caught this live: "no one seemed to know when it was the
        # last duel of the whole game." With exactly two active players
        # left, this next duel's loser is eliminated and the winner is
        # therefore, by construction, the sole owner of the entire board
        # (see game.py's is_final_duel) -- genuinely the deciding duel of
        # the whole show, not just another matchup. Checked FIRST, ahead
        # of even a hot streak or a territory mismatch: knowing this is
        # the last one matters more than either.
        if final_duel:
            return self.rng.choice([
                f"This is it -- whoever wins this one walks away with the ENTIRE board. "
                f"{challenger.kingdom_name} against {defender.kingdom_name}, winner takes all.",
                f"Down to just these two. One duel left, and the whole board goes to whoever "
                f"wins it -- {challenger.kingdom_name} versus {defender.kingdom_name}, on "
                f"{tested_domain}.",
                f"This is the final duel of the night. {challenger.kingdom_name} and "
                f"{defender.kingdom_name}, everything comes down to this one.",
            ])
        hot_streak = max(challenger_streak, defender_streak)
        if hot_streak >= 3:
            hot_player, cool_player = (
                (challenger, defender) if challenger_streak >= defender_streak
                else (defender, challenger)
            )
            return self.rng.choice([
                f"{hot_player.kingdom_name} is riding a real streak into this one -- "
                f"{hot_streak} in a row. Can anybody stop that tonight?",
                f"{hot_player.kingdom_name} has been on fire lately. Let's see if it holds "
                f"up against {cool_player.kingdom_name}.",
            ])
        territory_gap = abs(challenger_territory - defender_territory)
        if territory_gap >= 3:
            big_name, small_name = (
                (challenger.kingdom_name, defender.kingdom_name)
                if challenger_territory > defender_territory
                else (defender.kingdom_name, challenger.kingdom_name)
            )
            return self.rng.choice([
                f"{small_name} against the odds here, taking on {big_name}'s much bigger "
                f"kingdom on {tested_domain}.",
                f"{big_name} has the bigger kingdom, but the board doesn't care about that "
                f"once the clock starts -- {small_name} against {big_name} on {tested_domain}.",
            ])
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

    async def announce_duel_result(self, winner: Player, loser: Player, tested_domain: str,
                                    is_upset: bool, is_close: bool) -> str:
        # M32 Fix 2: a real host blends the win reaction and a genuine
        # goodbye to the loser in one breath -- ported from web/
        # index.html's winHostLine/winCloseHostLine/winUpsetHostLine
        # (the win half) and ELIMINATION_LINES (the goodbye half), which
        # used to fire as 4 separate scripted pools across two different
        # beats (celebrateDuelWinner, eliminationBeat) for what's really
        # one moment. Merging them here halves the added live-call cost
        # vs. two separate new methods, for the same narrative gain.
        if is_upset:
            win_templates = [
                f"What an UPSET! {winner.kingdom_name} just took down {loser.kingdom_name}! "
                f"Nobody saw that coming!",
                f"The little kingdom just took down the giant. {winner.kingdom_name}, over "
                f"{loser.kingdom_name}!",
                f"That is exactly the kind of underdog moment this show is built for, "
                f"{winner.kingdom_name}!",
                f"Hold on, did {winner.kingdom_name} really just beat {loser.kingdom_name}?!",
            ]
        elif is_close:
            win_templates = [
                f"{winner.kingdom_name} wins it, down to the wire! What a finish!",
                f"You could not squeeze a second between them, but {winner.kingdom_name} "
                f"gets there first!",
                f"That's as close as it gets, folks. {winner.kingdom_name} takes "
                f"{tested_domain} by a hair!",
                f"Right down to the final tick of the clock, and {winner.kingdom_name} "
                f"comes out on top!",
            ]
        else:
            win_templates = [
                f"{winner.kingdom_name} takes it! And that's the ballgame!",
                f"{winner.kingdom_name} seals it. What a finish!",
                f"{tested_domain} now belongs to {winner.kingdom_name}. Territory changes "
                f"hands!",
                f"Clean and decisive. {winner.kingdom_name} gets it done.",
                f"That's how it's done, {winner.kingdom_name}!",
                f"{winner.kingdom_name} closes it out. No doubt about that one!",
            ]
        goodbye_templates = [
            f"Goodbye, {loser.kingdom_name}! Thanks for playing tonight.",
            f"That's it for {loser.kingdom_name}. Thanks for playing, and take a bow.",
            f"{loser.kingdom_name} is eliminated. Thanks for playing tonight!",
            f"Goodbye, {loser.kingdom_name}! Your run ends here tonight.",
            f"Thanks for playing, {loser.kingdom_name}! The board says goodbye.",
            f"{loser.kingdom_name}, you're out. Thanks for playing tonight!",
        ]
        return f"{self.rng.choice(win_templates)} {self.rng.choice(goodbye_templates)}"

    async def announce_continue_decision(self, player: Player, keep_going: bool,
                                          reason: str) -> str:
        # M32 Fix 3: ports web/index.html's old hostContinueReaction --
        # reacts to the SAME real decide_continue outcome the player's
        # own interview banner already shows, not a separate invented
        # reaction. reason is accepted for interface parity with
        # LLMHostAgent's live version below (which actually reacts to
        # it); the scripted pool here doesn't reference it, same as
        # hostContinueReaction never did.
        if keep_going:
            templates = [
                f"{player.kingdom_name} pushes on!", f"{player.kingdom_name} is not done yet!",
                f"{player.kingdom_name} stays in it!", f"Onward for {player.kingdom_name}!",
            ]
        else:
            templates = [
                f"{player.kingdom_name} pulls back to defend!",
                f"{player.kingdom_name} plays it safe!", f"{player.kingdom_name} holds the line!",
                f"{player.kingdom_name} calls it there!",
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
                                  tested_domain: str, challenger_streak: int = 0,
                                  defender_streak: int = 0, challenger_territory: int = 0,
                                  defender_territory: int = 0, final_duel: bool = False) -> str:
        # M32 Fix 1: same streak/territory context ScriptedHostAgent's
        # fallback now uses (ported from the old, separate
        # preduelHostLine) -- added to the prompt only when actually
        # notable, same thresholds as the scripted version, so a live
        # model can reference a real streak or size mismatch instead of
        # never hearing about either.
        context_lines = ""
        # Scott: "no one seemed to know when it was the last duel of the
        # whole game." Same final_duel signal ScriptedHostAgent's fallback
        # now leads with (see there for why exactly-2-active-players means
        # this duel decides the whole show) -- told to the model as an
        # unambiguous fact, not just folded in as one more context line,
        # so it can't get lost under a streak/territory mention instead.
        if final_duel:
            context_lines += (
                "\nTHIS IS THE FINAL DUEL OF THE ENTIRE SHOW -- only these two players are "
                "still active, so whoever wins this one becomes the sole owner of the whole "
                "board and wins the grand prize. Make sure the audience knows this is it."
            )
        hot_streak = max(challenger_streak, defender_streak)
        if hot_streak >= 3:
            hot_name = (challenger.kingdom_name if challenger_streak >= defender_streak
                        else defender.kingdom_name)
            context_lines += f"\n{hot_name} is riding a real {hot_streak}-duel win streak tonight."
        territory_gap = abs(challenger_territory - defender_territory)
        if territory_gap >= 3:
            big_name = (challenger.kingdom_name if challenger_territory > defender_territory
                        else defender.kingdom_name)
            context_lines += (
                f"\n{big_name} holds a much bigger kingdom right now -- a real size "
                f"mismatch worth mentioning."
            )
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show, in front of a "
            f"real studio audience. {HOST_PERSONA}. Announce the next matchup, live, on "
            f"air, in ONE or TWO short punchy sentences a real game show host would actually say out loud. "
            f"You don't need to say your own name in every line -- only if it actually fits "
            f"naturally.\n"
            f"Challenger: {challenger.kingdom_name} ({challenger.profession}).\n"
            f"Defender: {defender.kingdom_name} ({defender.profession}), whose real "
            f"expertise is {defender.origin_domain}.\n"
            f"Domain on the line tonight: {tested_domain} -- this is the DEFENDER's own "
            f"territory being attacked, so name the stakes for {defender.kingdom_name} "
            f"specifically.{context_lines}\n"
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
            challenger, defender, tested_domain, challenger_streak=challenger_streak,
            defender_streak=defender_streak, challenger_territory=challenger_territory,
            defender_territory=defender_territory, final_duel=final_duel)

    async def announce_finale(self, champion: Player, total_duels: int, prize: int) -> str:
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show. {HOST_PERSONA}. "
            f"The show just "
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

    async def announce_duel_result(self, winner: Player, loser: Player, tested_domain: str,
                                    is_upset: bool, is_close: bool) -> str:
        # M32 Fix 2: one live call covering both the win reaction and a
        # genuine goodbye to the loser -- see ScriptedHostAgent's own
        # version for the two scripted pools this replaces/merges.
        if is_upset:
            drama_note = (
                f"This is a real UPSET -- {loser.kingdom_name} held meaningfully more "
                f"territory going in, and {winner.kingdom_name} still took it."
            )
        elif is_close:
            drama_note = "This was a nail-biter, decided by only a couple seconds on the clock."
        else:
            drama_note = "A clean, decisive win, nothing especially close or upsetting about it."
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show. {HOST_PERSONA}. "
            f"A duel on "
            f"{tested_domain} just ended: {winner.kingdom_name} won, and {loser.kingdom_name} "
            f"is eliminated -- out of the show for good. {drama_note}\n"
            f"Announce this live, on air, in ONE or TWO short sentences that do BOTH things "
            f"a real host would: react to the win itself, AND give {loser.kingdom_name} a "
            f"genuine, warm send-off by name -- not mocking, this is a real contestant's run "
            f"ending. Reply with ONLY the announcement itself, no stage directions, no "
            f"quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=70)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().announce_duel_result(
            winner, loser, tested_domain, is_upset, is_close)

    async def announce_continue_decision(self, player: Player, keep_going: bool,
                                          reason: str) -> str:
        # M32 Fix 3: reacts to the REAL, already-live decide_continue
        # outcome and reason (M10) -- a cheap, high-value pairing since
        # the content it reacts to is already real, not invented here.
        # hostContinueBuildup (the pre-decision hype line) stays
        # scripted -- there's nothing dynamic to react to before the
        # decision is actually made.
        verdict = "keeps pushing" if keep_going else "pulls back to defend"
        prompt = (
            f"You are {HOST_NAME}, the live host of a TV trivia game show. {HOST_PERSONA}. "
            f"{player.kingdom_name} just made a real decision, live, on air: they said "
            f'"{reason}" and {verdict}.\n'
            f"React to that specific choice and reason, live, on air, in ONE short "
            f"sentence a real game show host would actually say. Reply with ONLY the "
            f"reaction itself, no stage directions, no quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=50)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().announce_continue_decision(
            player, keep_going, reason)


def host_model_for_backend(backend: str) -> str:
    """Which model plays the Host -- a fixed, dedicated choice per
    backend (inference/config.py), independent of the contestant model
    pool (TEXT_MODELS/LLAMACPP_MODELS) the same way those two rosters are
    already independently declared from each other. Not exposed in the
    M8 model picker -- a deliberate v1 cut."""
    if backend == "llamacpp":
        return settings.host_model_llamacpp
    return settings.host_model_ollama
