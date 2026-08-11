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


if __name__ == "__main__":
    unittest.main()
