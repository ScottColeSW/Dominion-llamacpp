"""LLMAgent.attempt_question's clock-charging math (agents/llm_agent.py) --
Scott's "wall-clock accurate" fix: the clock must charge real elapsed call
time, not the backend's self-reported pure-generation time (think_seconds),
which stayed near-zero on every warm attempt regardless of how long the
real HTTP round-trip/queueing/contention actually took."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock

from dominion.agents.llm_agent import LLMAgent, MIN_CHARGED_SECONDS
from dominion.engine.content import Domain, Question
from dominion.engine.models import Player
from dominion.inference.base import GenerationResult


def _player(pid: int, name: str) -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model="test-model")


def _question() -> Question:
    return Question(image_prompt="A small animal that barks.", answer="dog")


def _domain() -> Domain:
    return Domain(name="Dogs", questions=[_question()])


class AttemptQuestionClockChargingTest(unittest.IsolatedAsyncioTestCase):
    async def _attempt(self, think_seconds, raw_latency_seconds, reply_text="A"):
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text=reply_text, think_seconds=think_seconds,
            raw_latency_seconds=raw_latency_seconds)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        return await agent.attempt_question(
            _player(0, "A"), _question(), _domain(), distractors=["cat", "bird"])

    async def test_charges_real_elapsed_time_not_tiny_think_seconds(self) -> None:
        # The exact scenario that was invisible to the clock before this
        # fix: a real call that took 22 real seconds (queueing/contention/
        # a cold load), but the backend's own reported pure-generation time
        # was a fraction of a second, since answering with one letter is
        # nearly instant once the model is actually running.
        attempt = await self._attempt(think_seconds=0.05, raw_latency_seconds=22.0)
        self.assertEqual(attempt.seconds_used, 22)

    async def test_min_charged_seconds_floor_still_applies_to_a_fast_call(self) -> None:
        attempt = await self._attempt(think_seconds=0.05, raw_latency_seconds=0.3)
        self.assertEqual(attempt.seconds_used, MIN_CHARGED_SECONDS)

    async def test_charged_seconds_rounds_up_to_a_whole_second(self) -> None:
        attempt = await self._attempt(think_seconds=0.1, raw_latency_seconds=5.2)
        self.assertEqual(attempt.seconds_used, 6)

    async def test_lettered_answer_still_resolves_with_real_elapsed_charge(self) -> None:
        attempt = await self._attempt(think_seconds=0.1, raw_latency_seconds=4.0, reply_text="A")
        # Not asserting correct/incorrect here (choices are shuffled) --
        # just that a real elapsed-based charge coexists correctly with a
        # live, scored outcome rather than falling back to scripted.
        self.assertIn(attempt.outcome, ("correct", "incorrect"))
        self.assertEqual(attempt.seconds_used, 4)
        self.assertTrue(attempt.live)

    async def test_pass_reply_still_resolves_with_real_elapsed_charge(self) -> None:
        attempt = await self._attempt(think_seconds=0.1, raw_latency_seconds=7.0, reply_text="PASS")
        self.assertEqual(attempt.outcome, "passed")
        self.assertEqual(attempt.seconds_used, 7)


if __name__ == "__main__":
    unittest.main()
