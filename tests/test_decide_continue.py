"""LLMAgent.decide_continue's self-contradiction guards. Scott caught two
distinct ways a live model's free-text reason can disagree with its own
PUSH/RETREAT verdict: claiming to leave the game outright ("stay and hold
ground and retire"), and -- separately -- arguing for the opposite
strategic direction from the verdict actually returned ("take ground" in
the same breath as retreating). Both fall back to a safe canned line
rather than airing a self-contradicting sentence, since decide_continue's
reason is spoken live, on air."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock

from dominion.agents.llm_agent import LLMAgent
from dominion.agents.scripted_agent import ScriptedAgent
from dominion.engine.models import GameState, Player
from dominion.inference.base import GenerationResult


def _player(pid: int, name: str) -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model="test-model")


def _game_with_two_adjacent_players() -> GameState:
    p0, p1 = _player(0, "A"), _player(1, "B")
    return GameState(players={0: p0, 1: p1}, owner={0: 0, 1: 1},
                      board_adj={0: {1}, 1: {0}}, active_ids={0, 1})


class DecideContinueContradictionTest(unittest.IsolatedAsyncioTestCase):
    async def _decide(self, reply_text: str):
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text=reply_text, think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        game = _game_with_two_adjacent_players()
        return await agent.decide_continue(game.players[0], game)

    async def test_consistent_push_is_unaffected(self) -> None:
        keep_going, reason = await self._decide("PUSH: I can feel it, one more!\nMEMO: none")
        self.assertTrue(keep_going)
        self.assertEqual(reason, "I can feel it, one more!")

    async def test_consistent_retreat_is_unaffected(self) -> None:
        keep_going, reason = await self._decide(
            "RETREAT: I'll hold what I've got.\nMEMO: none")
        self.assertFalse(keep_going)
        self.assertEqual(reason, "I'll hold what I've got.")

    async def test_retire_quit_contradiction_falls_back(self) -> None:
        keep_going, reason = await self._decide(
            "RETREAT: I'm going to stay and hold ground and retire.\nMEMO: none")
        self.assertFalse(keep_going)
        self.assertEqual(reason, "retreating to defend.")

    async def test_retreat_with_push_signaling_reason_falls_back(self) -> None:
        # The exact live case Scott reported: verdict says RETREAT, but the
        # reason itself argues for taking more ground -- the opposite
        # strategic direction from what was actually chosen.
        keep_going, reason = await self._decide(
            "RETREAT: I really want to take ground here.\nMEMO: none")
        self.assertFalse(keep_going)
        self.assertEqual(reason, "retreating to defend.")

    async def test_push_with_retreat_signaling_reason_falls_back(self) -> None:
        keep_going, reason = await self._decide(
            "PUSH: Actually I'd rather hold my ground.\nMEMO: none")
        self.assertTrue(keep_going)
        self.assertEqual(reason, "pushing on.")


class TwoPlayerEndgameHeuristicTest(unittest.IsolatedAsyncioTestCase):
    """Scott: "players should not have the strategy of defense when there
    are only 2 players left unless it is a strategic domain knowledge
    play." With exactly two active players, RETREAT doesn't dodge a duel --
    it just flips who challenges whom, and the tested domain always
    belongs to whoever ends up defending. Both ScriptedAgent's heuristic
    and LLMAgent's live prompt should reason about this using real domain
    knowledge, not just temperament/caution."""

    def _two_player_game(self, mover_domain: str, opponent_domain: str) -> GameState:
        mover = Player(id=0, domain=mover_domain, kingdom_name="Mover", profession="tester",
                        origin_domain="Cats", model="test-model")
        opponent = Player(id=1, domain=opponent_domain, kingdom_name="Opponent",
                           profession="tester", origin_domain="Dogs", model="test-model")
        return GameState(players={0: mover, 1: opponent}, owner={0: 0, 1: 1},
                          board_adj={0: {1}, 1: {0}}, active_ids={0, 1})

    async def test_scripted_agent_pushes_when_attacking_is_the_real_edge(self) -> None:
        # Mover's own held ground ("Circus") isn't their expertise; the
        # opponent's held ground ("Cats") IS the mover's real expertise --
        # pushing attacks the opponent's domain, a genuine edge.
        game = self._two_player_game(mover_domain="Circus", opponent_domain="Cats")
        agent = ScriptedAgent(random.Random(1))
        keep_going, reason = await agent.decide_continue(game.players[0], game)
        self.assertTrue(keep_going)
        self.assertIn("coming for it", reason)

    async def test_scripted_agent_retreats_when_defending_is_the_real_edge(self) -> None:
        # Mover's own held ground ("Cats") IS their expertise; the
        # opponent's held ground ("Circus") is not -- retreating means
        # the opponent attacks the mover's own strong domain instead.
        game = self._two_player_game(mover_domain="Cats", opponent_domain="Circus")
        agent = ScriptedAgent(random.Random(1))
        keep_going, reason = await agent.decide_continue(game.players[0], game)
        self.assertFalse(keep_going)
        self.assertIn("come to me", reason)

    async def test_llm_prompt_names_the_real_domain_tradeoff_with_two_players_left(self) -> None:
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text="PUSH: Let's go.\nMEMO: none", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        game = self._two_player_game(mover_domain="Circus", opponent_domain="Cats")
        await agent.decide_continue(game.players[0], game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("down to just the two of you", prompt)
        self.assertIn("Opponent's domain (Cats)", prompt)
        self.assertIn("YOUR domain (Circus)", prompt)


if __name__ == "__main__":
    unittest.main()
