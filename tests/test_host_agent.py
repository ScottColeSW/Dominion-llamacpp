"""ScriptedHostAgent/LLMHostAgent (agents/host_agent.py) -- the Host as a
real agent, same fallback contract every other agent already has."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock

from dominion.agents.host_agent import HOST_NAME, LLMHostAgent, ScriptedHostAgent
from dominion.engine.models import Player
from dominion.inference.base import GenerationResult


def _player(pid: int, name: str, origin: str = "Cats") -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain=origin, model="test-model")


class ScriptedHostAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_announce_challenge_returns_nonempty_text(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        challenger, defender = _player(0, "Challenger Kingdom"), _player(1, "Defender Kingdom")
        line = await agent.announce_challenge(challenger, defender, "Dogs")
        self.assertIsInstance(line, str)
        self.assertTrue(line)
        self.assertIn("Defender Kingdom", line)

    async def test_announce_finale_returns_nonempty_text(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        champion = _player(0, "Champion Kingdom")
        line = await agent.announce_finale(champion, total_duels=12, prize=100_000_000)
        self.assertIsInstance(line, str)
        self.assertIn("Champion Kingdom", line)

    async def test_host_has_a_name(self) -> None:
        # Scott: "perhaps the Host needs a name to introduce the
        # themselves and the show." A fixed persona name, not
        # per-show/per-player generated.
        self.assertEqual(HOST_NAME, "Rex Domain")


class LLMHostAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_prompts_identify_the_host_by_name(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real line.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "A"), _player(1, "B")
        champion = _player(0, "Champion Kingdom")
        await agent.announce_challenge(challenger, defender, "Dogs")
        challenge_prompt = client.generate.call_args.args[1]
        await agent.announce_finale(champion, total_duels=12, prize=100)
        finale_prompt = client.generate.call_args.args[1]
        self.assertIn(HOST_NAME, challenge_prompt)
        self.assertIn(HOST_NAME, finale_prompt)

    async def test_falls_back_to_scripted_line_on_failure(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "A"), _player(1, "B")
        line = await agent.announce_challenge(challenger, defender, "Dogs")
        self.assertIn("B", line)  # the scripted fallback template, not a live reply

    async def test_uses_live_text_on_success(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real live announcement!", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "A"), _player(1, "B")
        line = await agent.announce_challenge(challenger, defender, "Dogs")
        self.assertEqual(line, "A real live announcement!")

    async def test_finale_falls_back_and_succeeds(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        champion = _player(0, "Champion Kingdom")
        line = await agent.announce_finale(champion, total_duels=12, prize=100)
        self.assertIn("Champion Kingdom", line)

        client.generate.return_value = GenerationResult(
            text="What a finale!", think_seconds=0.5, raw_latency_seconds=0.5)
        line = await agent.announce_finale(champion, total_duels=12, prize=100)
        self.assertEqual(line, "What a finale!")


if __name__ == "__main__":
    unittest.main()
