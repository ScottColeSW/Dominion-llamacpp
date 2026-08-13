"""The show's second AI voice, distinct from the Host (agents/host_agent.py):
a color commentator whose value is specifically that its banter can be
grounded in real, measured cross-show history (engine/history.py's
get_stats()) instead of generic decoration. game.py has no idea SQLite
exists (see history.py's own docstring on that boundary) -- a stats
snapshot is fetched once, up front, by server/app.py's _stream_show and
threaded through run_show(..., stats_snapshot=...) as a plain dict, same
as any other engine input.

Same ScriptedAgent/LLMAgent split every other role already has. Kept to a
modest frequency by design (see game.py's call sites): one comment per
duel (right after the Host's challenge announcement) plus rare/special
events only (advantage_earned, scramble) -- never on every duel_turn,
which would multiply live-call volume far past what a duel currently
pays.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from ..inference.base import InferenceClient
from ..inference.config import settings
from ..engine.models import GameState, Player
from .llm_agent import _history_block, _trim_to_last_sentence


def _stats_for(stats_snapshot: Optional[Dict[str, Any]], backend: str,
                model: str) -> Optional[Dict[str, Any]]:
    if not stats_snapshot:
        return None
    for entry in stats_snapshot.get("models", ()):
        if entry.get("backend") == backend and entry.get("model") == model:
            return entry
    return None


class ScriptedCommentatorAgent:
    """LLMCommentatorAgent's fallback on any failure, and the whole
    commentator in SCRIPTED_ONLY mode -- genuinely decorative, no stats
    needed, same as ScriptedHostAgent's role for the Host."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    async def react_to_matchup(self, challenger: Player, defender: Player,
                                tested_domain: str, game: Optional[GameState] = None) -> str:
        # game accepted but unused here -- genuinely decorative, no duel
        # count or show history needed for a canned line. Matches
        # LLMCommentatorAgent's signature so its super().react_to_matchup(...)
        # fallback call doesn't need special-casing.
        templates = [
            f"{challenger.kingdom_name} really thinks they can take "
            f"{defender.kingdom_name} on {tested_domain} tonight.",
            f"I've seen {defender.kingdom_name} dig in before on their own ground -- "
            f"this won't be easy for {challenger.kingdom_name}.",
            f"Bold move from {challenger.kingdom_name}. We'll see if it pays off.",
        ]
        return self.rng.choice(templates)

    async def react_to_advantage(self, player: Player, streak: int) -> str:
        templates = [
            f"{player.kingdom_name} is putting together a real run tonight -- "
            f"{streak} in a row.",
            f"That's {streak} straight wins for {player.kingdom_name}. Somebody stop them.",
        ]
        return self.rng.choice(templates)

    async def react_to_scramble(self, active_players: int) -> str:
        templates = [
            f"And just like that, the board resets. {active_players} players left standing.",
            "The Scramble! Everything you thought you knew about this board is gone.",
        ]
        return self.rng.choice(templates)

    async def react_to_threepeat_drive(self, player: Player, target: Player) -> str:
        templates = [
            f"{player.kingdom_name} isn't just on a heater -- that's a guaranteed shot clear "
            f"across the board at {target.kingdom_name} now. A real Threepeat Drive!",
            f"Three in a row buys you the right to reach anywhere on this board, and "
            f"{player.kingdom_name} is reaching all the way for {target.kingdom_name}.",
        ]
        return self.rng.choice(templates)


class LLMCommentatorAgent(ScriptedCommentatorAgent):
    """Same interface as ScriptedCommentatorAgent; the matchup reaction
    consults a real model, handed the two duelists' own (backend, model)
    entries from the stats snapshot so it can react to something actually
    true instead of a hallucinated stat. Falls back to the scripted pool
    on any failure -- identical fallback contract to every other agent."""

    def __init__(self, rng: random.Random, model: str, client: InferenceClient, backend: str,
                 stats_snapshot: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(rng, **kwargs)
        self.model = model
        self.client = client
        self.backend = backend
        self.stats_snapshot = stats_snapshot

    async def react_to_matchup(self, challenger: Player, defender: Player,
                                tested_domain: str, game: Optional[GameState] = None) -> str:
        # Scott caught a real gap here live: the commentator kept saying
        # there was "no real recorded history" on a player who very
        # visibly had a real win streak going RIGHT NOW, on screen. The
        # bug was scope, not data: this method only ever looked at
        # get_stats()'s CROSS-SHOW aggregation (won X% of duels across
        # every show ever recorded), which is legitimately sparse/empty
        # early in a project's life, while completely ignoring the two
        # Player objects it's already holding -- push_streak and
        # territory are real, live, THIS-show facts, zero database rows
        # required. A real commentator would obviously mention "riding a
        # 3-duel win streak tonight" even with an empty history table.
        live_lines = []
        if challenger.push_streak >= 2:
            live_lines.append(
                f"{challenger.kingdom_name} is riding a {challenger.push_streak}-duel win "
                f"streak tonight."
            )
        if defender.push_streak >= 2:
            live_lines.append(
                f"{defender.kingdom_name} is riding a {defender.push_streak}-duel win "
                f"streak tonight."
            )
        # Scott caught a second real gap live: with NEITHER explicit
        # streak line above (both under 2), the only remaining live fact
        # was the territory count -- and the model hallucinated a streak
        # anyway, apparently pattern-matching "here's a number about this
        # player" into "that must mean a streak." An explicit, honest
        # NEGATIVE fact (matching stats_block's own "no history yet"
        # fallback below) closes that gap instead of just leaving a
        # silence for the model to fill in on its own.
        if challenger.push_streak < 2 and defender.push_streak < 2:
            live_lines.append(
                "Neither player has a real win streak going tonight yet -- don't invent one."
            )
        # Scott caught a third live gap: even with the streak disclaimer
        # above, the model kept treating raw tile count as if it were
        # itself evidence of skill or "winning" -- e.g. "X has won more
        # tiles tonight than Y, despite their win percentages," an
        # apples-to-oranges comparison that makes no sense (tile count and
        # cross-show win_rate are unrelated numbers) and, worse, fires on
        # the very FIRST duel of the show, when every player's tile count
        # is trivially their own single starting allocation -- 1 tile
        # each, never a meaningful fact at all. Only mention it once a
        # player genuinely holds more than their own single starting
        # tile (real accumulated territory, whatever the cause), and
        # frame it as defending ground, not as a scoreboard entry that
        # could "beat" or "contradict" a win-rate percentage.
        if len(defender.territory) > 1:
            live_lines.append(
                f"{defender.kingdom_name} is currently defending {len(defender.territory)} "
                f"tile(s) of the board tonight."
            )
        live_block = " ".join(live_lines)

        challenger_stats = _stats_for(self.stats_snapshot, self.backend, challenger.model)
        defender_stats = _stats_for(self.stats_snapshot, self.backend, defender.model)
        stats_lines = []
        # Scott caught a live gap here too: the model sometimes said the
        # raw model tag (e.g. "qwen2.5:3b") out loud instead of only ever
        # using the kingdom name -- because the old wording put the tag
        # right in the sentence it was reading. Moved the tag out of the
        # natural-language fact entirely (kept only as an internal
        # backend/model lookup key, never phrased for the model to
        # repeat) and added an explicit instruction below never to say it.
        if challenger_stats and challenger_stats.get("win_rate") is not None:
            stats_lines.append(
                f"{challenger.kingdom_name} has won "
                f"{challenger_stats['win_rate']:.0%} of its recorded duels across every "
                f"show so far."
            )
        if defender_stats and defender_stats.get("win_rate") is not None:
            stats_lines.append(
                f"{defender.kingdom_name} has won "
                f"{defender_stats['win_rate']:.0%} of its recorded duels across every "
                f"show so far."
            )
        stats_block = " ".join(stats_lines) if stats_lines else (
            "No cross-show recorded history on either contestant yet -- this is still early data."
        )
        # Scott: "Commentator is still wrong about duel counts and what has
        # happened in this game so far." Real gap: this call never accepted
        # `game` at all, unlike every other live agent call (choose_target,
        # decide_continue, intro_line_combined) -- it had no duel count and
        # no access to GameState.memory's "SHOW SO FAR" feed to be right
        # OR wrong about; any specific duel count or show-history claim it
        # made was pure invention. game.duel_count counts duels already
        # RESOLVED (incremented after run_duel returns, in game.py) -- this
        # call fires during the challenge announcement, before that
        # increment, so the upcoming duel is duel_count + 1.
        duel_count_line = (
            f" This will be duel #{game.duel_count + 1} of tonight's show." if game else ""
        )
        prompt = (
            "You are the color commentator on a live TV trivia game show, a separate "
            "voice from the host, there to give the audience real insight, not just hype.\n"
            f"Upcoming matchup: {challenger.kingdom_name} challenging {defender.kingdom_name} "
            f"on {tested_domain}.{duel_count_line}\n"
            f"What's actually happening in TONIGHT'S show so far: {live_block}"
            f"{_history_block(game)}\n"
            f"Cross-show history you have on these two contestants (may be sparse -- that's "
            f"fine, the live facts above are real either way): {stats_block}\n"
            "React live, on air, in ONE short sentence that actually uses at least one of "
            "these real facts -- prefer tonight's live facts when they're the more "
            "interesting story, don't just default to 'no history' when a real streak is "
            "sitting right there. Do not invent a stat, a duel count, or a past event that "
            "wasn't given to you above. Tile count "
            "and cross-show win rate are separate, unrelated numbers -- a player's tile "
            "count can grow from board mechanics that have nothing to do with their own "
            "skill (a reshuffle, another player's leftover territory landing on them), so "
            "never say one 'beats' or contradicts the other, or that holding more tiles "
            "means winning more. Refer to both contestants ONLY by their kingdom name -- "
            "never say a model name or version tag out loud, the audience doesn't know or "
            "care what's running under the hood. Reply with ONLY the comment itself, no "
            "stage directions, no quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=70)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().react_to_matchup(
            challenger, defender, tested_domain, game=game)

    async def react_to_advantage(self, player: Player, streak: int) -> str:
        prompt = (
            "You are the color commentator on a live TV trivia game show. "
            f"{player.kingdom_name} ({player.profession}) has just won {streak} duels in a "
            "row, live, on air. React in ONE short excited sentence, as a real commentator "
            "would. Reply with ONLY the comment itself, no stage directions, no quotation "
            "marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=60)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().react_to_advantage(
            player, streak)

    async def react_to_scramble(self, active_players: int) -> str:
        prompt = (
            "You are the color commentator on a live TV trivia game show. The board just "
            f"got scrambled and reshuffled live, on air -- {active_players} players remain. "
            "React in ONE short sentence, as a real commentator would. Reply with ONLY the "
            "comment itself, no stage directions, no quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=60)
        return _trim_to_last_sentence(result.text.strip()) if result.text else await super().react_to_scramble(
            active_players)

    async def react_to_threepeat_drive(self, player: Player, target: Player) -> str:
        prompt = (
            "You are the color commentator on a live TV trivia game show. "
            f"{player.kingdom_name} has just won 3 duels in a row, live, on air, and that "
            f"streak earns a Threepeat Drive: a guaranteed challenge clear across the board "
            f"against {target.kingdom_name}, someone they'd never normally be able to reach. "
            "React in ONE short excited sentence, as a real commentator would, making clear "
            "this is a rare, earned long-range shot, not an ordinary matchup. Reply with ONLY "
            "the comment itself, no stage directions, no quotation marks."
        )
        result = await self.client.generate(self.model, prompt, timeout=settings.generate_timeout,
                                             num_predict=60)
        return _trim_to_last_sentence(result.text.strip()) if result.text else (
            await super().react_to_threepeat_drive(player, target))


def commentator_model_for_backend(backend: str) -> str:
    """A fixed, dedicated model per backend, independent of both the
    contestant model pool and the Host's own dedicated model -- same
    "independently declared roster" pattern host_model_for_backend
    already uses. Not exposed in the M8 model picker for v1."""
    if backend == "llamacpp":
        return settings.commentator_model_llamacpp
    return settings.commentator_model_ollama
