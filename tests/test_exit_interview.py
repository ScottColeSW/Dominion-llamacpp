"""ScriptedAgent.exit_interview / LLMAgent.exit_interview (M12) -- the
just-eliminated player's own last word, same fallback contract every other
agent call already has."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock

from dominion.agents.llm_agent import LLMAgent
from dominion.agents.scripted_agent import ScriptedAgent
from dominion.engine.models import Player
from dominion.inference.base import GenerationResult


def _player(pid: int, name: str, temperament: float = 0.5) -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model="test-model", temperament=temperament)


class ScriptedAgentExitInterviewTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_nonempty_text_mentioning_the_winner(self) -> None:
        agent = ScriptedAgent(random.Random(1))
        loser, winner = _player(0, "Loser Kingdom"), _player(1, "Winner Kingdom")
        line = await agent.exit_interview(loser, winner, "Dogs")
        self.assertIsInstance(line, str)
        self.assertIn("Winner Kingdom", line)

    async def test_varies_by_temperament(self) -> None:
        agent = ScriptedAgent(random.Random(1))
        winner = _player(1, "Winner Kingdom")
        aggressive = _player(0, "Bold One", temperament=0.9)
        cautious = _player(2, "Careful One", temperament=0.1)
        aggressive_line = await agent.exit_interview(aggressive, winner, "Dogs")
        cautious_line = await agent.exit_interview(cautious, winner, "Dogs")
        # The three temperament pools are disjoint strings -- checking each
        # reply lands in ITS OWN pool (not just any nonempty string) is what
        # actually proves temperament shapes the reply, across either of
        # that pool's two templates.
        self.assertTrue(any(marker in aggressive_line.lower() for marker in ("regrets", "swinging")))
        self.assertTrue(any(marker in cautious_line.lower() for marker in ("earned that one", "honest best")))


class LLMAgentExitInterviewTest(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_scripted_line_on_failure(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        loser, winner = _player(0, "A"), _player(1, "B")
        line = await agent.exit_interview(loser, winner, "Dogs")
        self.assertIn("B", line)  # the scripted fallback template, not a live reply

    async def test_uses_live_text_on_success(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="I can't believe it's over.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        loser, winner = _player(0, "A"), _player(1, "B")
        line = await agent.exit_interview(loser, winner, "Dogs")
        self.assertEqual(line, "I can't believe it's over.")


if __name__ == "__main__":
    unittest.main()
