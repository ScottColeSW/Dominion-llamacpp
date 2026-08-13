"""ScriptedCommentatorAgent/LLMCommentatorAgent (agents/commentator_agent.py)
-- the second AI voice, same fallback contract every other agent has, plus
the M11-specific claim: a live matchup reaction actually uses the
stats-snapshot data it's handed, not a hallucinated number."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock

from dominion.agents.commentator_agent import LLMCommentatorAgent, ScriptedCommentatorAgent
from dominion.engine.models import GameState, Player
from dominion.inference.base import GenerationResult


def _player(pid: int, name: str, model: str = "test-model") -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model=model)


def _game(challenger: Player, defender: Player, duel_count: int = 0) -> GameState:
    game = GameState(players={0: challenger, 1: defender}, owner={0: 0, 1: 1},
                      board_adj={0: {1}, 1: {0}}, active_ids={0, 1})
    game.duel_count = duel_count
    return game


class ScriptedCommentatorAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_react_to_matchup_returns_nonempty_text(self) -> None:
        agent = ScriptedCommentatorAgent(random.Random(1))
        challenger, defender = _player(0, "Challenger Kingdom"), _player(1, "Defender Kingdom")
        line = await agent.react_to_matchup(challenger, defender, "Dogs")
        self.assertIsInstance(line, str)
        self.assertTrue(line)

    async def test_react_to_advantage_returns_nonempty_text(self) -> None:
        agent = ScriptedCommentatorAgent(random.Random(1))
        player = _player(0, "Streak Kingdom")
        line = await agent.react_to_advantage(player, streak=3)
        self.assertIn("Streak Kingdom", line)

    async def test_react_to_scramble_returns_nonempty_text(self) -> None:
        agent = ScriptedCommentatorAgent(random.Random(1))
        line = await agent.react_to_scramble(active_players=5)
        self.assertIsInstance(line, str)
        self.assertTrue(line)


class LLMCommentatorAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_scripted_line_on_failure(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        line = await agent.react_to_matchup(challenger, defender, "Dogs")
        self.assertTrue(line)  # the scripted fallback, not a live reply

    async def test_uses_live_text_on_success(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real live comment!", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        line = await agent.react_to_matchup(challenger, defender, "Dogs")
        self.assertEqual(line, "A real live comment!")

    async def test_matchup_prompt_actually_includes_stats_snapshot_data(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting with real data.", think_seconds=0.5, raw_latency_seconds=0.5)
        stats_snapshot = {
            "models": [
                {"backend": "ollama", "model": "alpha-model", "win_rate": 0.75},
                {"backend": "ollama", "model": "beta-model", "win_rate": 0.25},
                {"backend": "llamacpp", "model": "alpha-model", "win_rate": 0.1},
            ]
        }
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama",
                                     stats_snapshot=stats_snapshot)
        challenger = _player(0, "A", model="alpha-model")
        defender = _player(1, "B", model="beta-model")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("75%", prompt)
        self.assertIn("25%", prompt)
        # The llamacpp-backend entry for alpha-model must NOT leak into an
        # ollama-backend show's prompt -- (backend, model) is the real key.
        self.assertNotIn("10%", prompt)

    async def test_matchup_prompt_is_honest_when_no_stats_exist_yet(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Early days.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama",
                                     stats_snapshot=None)
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("No cross-show recorded history", prompt)

    async def test_matchup_prompt_explicitly_denies_a_streak_when_neither_has_one(self) -> None:
        # Scott's live catch: the commentator said a player was "on a
        # streak" who hadn't fought a single duel yet -- with no streak
        # line present at all, the model apparently pattern-matched the
        # territory-count fact into "that must mean a streak." An
        # explicit negative fact closes that gap.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("Neither player has a real win streak", prompt)

    async def test_matchup_prompt_omits_territory_when_its_just_the_starting_tile(self) -> None:
        # Scott's live catch: the commentator said "X has won more tiles
        # tonight than Y, despite their overall win percentages" on the
        # very FIRST duel of the show -- every player's tile count at
        # that point is trivially their own single starting allocation,
        # never a meaningful fact, let alone one worth weighing against a
        # cross-show win rate.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        defender.territory = {5}  # exactly one tile -- their own starting allocation
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertNotIn("tile(s) of the board", prompt)

    async def test_matchup_prompt_includes_territory_once_genuinely_accumulated(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        defender.territory = {5, 6, 7}
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("B is currently defending 3 tile(s) of the board tonight", prompt)

    async def test_matchup_prompt_never_conflates_tile_count_with_win_rate(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("never say one 'beats' or contradicts the other", prompt)

    async def test_matchup_prompt_never_names_the_model_tag(self) -> None:
        # Scott's other live catch: the commentator said the raw model
        # tag out loud (e.g. "qwen2.5:3b") instead of only the kingdom
        # name -- the old wording put the tag directly in the sentence
        # the model was reading.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        stats_snapshot = {
            "models": [{"backend": "ollama", "model": "qwen2.5:3b", "win_rate": 0.5}]
        }
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama",
                                     stats_snapshot=stats_snapshot)
        challenger = _player(0, "Streaky Kingdom", model="qwen2.5:3b")
        defender = _player(1, "B")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertNotIn("qwen2.5:3b", prompt)
        self.assertIn("never say a model name or version tag out loud", prompt)

    async def test_matchup_prompt_includes_a_live_win_streak_even_with_no_stats(self) -> None:
        # Scott's live catch: the commentator kept saying there was no
        # data on a player who very visibly had a real win streak going
        # right now. push_streak is a live fact on the Player object
        # itself -- it should show up in the prompt regardless of
        # whether get_stats() has any cross-show history at all.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="On a real heater tonight.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama",
                                     stats_snapshot=None)
        challenger, defender = _player(0, "A"), _player(1, "B")
        challenger.push_streak = 4
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertIn("A is riding a 4-duel win streak tonight", prompt)

    async def test_matchup_prompt_includes_a_real_duel_count(self) -> None:
        # Scott: "Commentator is still wrong about duel counts and what
        # has happened in this game so far." Real gap: react_to_matchup
        # never accepted `game` at all before this fix, so it had no duel
        # count to be right OR wrong about -- any number it said was
        # invented. game.duel_count counts duels already RESOLVED
        # (incremented after run_duel returns, in game.py), and this call
        # fires during the challenge announcement for the NEXT one, before
        # that increment -- so duel_count=4 means this is duel #5.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        game = _game(challenger, defender, duel_count=4)
        await agent.react_to_matchup(challenger, defender, "Dogs", game=game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("duel #5", prompt)

    async def test_matchup_prompt_includes_what_happened_so_far(self) -> None:
        # Same gap, the other half of Scott's report: GameState.memory
        # (the "SHOW SO FAR" feed every other live agent call already
        # gets via _history_block) never reached the Commentator either.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        game = _game(challenger, defender)
        game.remember("A defeated C in a Cats duel (timeout). C is eliminated and out of the game.")
        await agent.react_to_matchup(challenger, defender, "Dogs", game=game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("A defeated C in a Cats duel", prompt)

    async def test_matchup_prompt_omits_duel_count_and_history_with_no_game(self) -> None:
        # game=None (the default) has to stay a safe no-op -- some callers
        # (this method's own scripted fallback path, or a test that
        # doesn't care about show history) never pass one.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reacting.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.react_to_matchup(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertNotIn("duel #", prompt)

    async def test_advantage_falls_back_and_succeeds(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        player = _player(0, "Streak Kingdom")
        line = await agent.react_to_advantage(player, streak=3)
        self.assertIn("Streak Kingdom", line)

        client.generate.return_value = GenerationResult(
            text="What a run!", think_seconds=0.5, raw_latency_seconds=0.5)
        line = await agent.react_to_advantage(player, streak=3)
        self.assertEqual(line, "What a run!")

    async def test_scramble_falls_back_and_succeeds(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMCommentatorAgent(random.Random(1), model="m", client=client, backend="ollama")
        line = await agent.react_to_scramble(active_players=5)
        self.assertTrue(line)

        client.generate.return_value = GenerationResult(
            text="Board's reshuffled!", think_seconds=0.5, raw_latency_seconds=0.5)
        line = await agent.react_to_scramble(active_players=5)
        self.assertEqual(line, "Board's reshuffled!")


if __name__ == "__main__":
    unittest.main()
