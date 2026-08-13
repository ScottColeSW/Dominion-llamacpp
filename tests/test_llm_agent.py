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
from dominion.engine.models import GameState, Player
from dominion.inference.base import GenerationResult
from dominion.inference.config import settings


def _player(pid: int, name: str) -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model="test-model")


def _question() -> Question:
    return Question(image_prompt="A small animal that barks.", answer="dog")


def _domain() -> Domain:
    return Domain(name="Dogs", questions=[_question()])


def _three_player_game() -> GameState:
    p0, p1, p2 = _player(0, "A"), _player(1, "B"), _player(2, "C")
    # 0 adjacent to both 1 and 2, so choose_target sees a real choice
    # instead of falling back to ScriptedAgent (its early-out for <2
    # candidates).
    return GameState(players={0: p0, 1: p1, 2: p2}, owner={0: 0, 1: 1, 2: 2},
                      board_adj={0: {1, 2}, 1: {0}, 2: {0}}, active_ids={0, 1, 2})


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


class OpponentHistoryLineTest(unittest.IsolatedAsyncioTestCase):
    """#47's structured, engine-authored per-opponent memory
    (GameState.opponent_record) actually reaching the three prompts it's
    wired into -- choose_target, choose_tax_target, intro_line_combined --
    and staying out of attempt_question (deliberately, highest-frequency
    call, kept light -- see _domain_familiarity_line's own docstring)."""

    def _client(self, reply_text: str) -> AsyncMock:
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text=reply_text, think_seconds=0.1, raw_latency_seconds=0.1)
        return client

    async def test_choose_target_includes_a_real_head_to_head_record(self) -> None:
        game = _three_player_game()
        game.record_duel_result(winner_id=0, loser_id=1)
        client = self._client("0\nMEMO: none")
        agent = LLMAgent(random.Random(1), model="m", client=client)
        await agent.choose_target(game.players[0], game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("faced B 1 duel before tonight -- won it", prompt)
        # Never faced C yet -- no fabricated record for that candidate.
        self.assertNotIn("faced C", prompt)

    async def test_choose_tax_target_includes_a_real_head_to_head_record(self) -> None:
        game = _three_player_game()
        game.record_duel_result(winner_id=1, loser_id=0)
        client = self._client("0")
        agent = LLMAgent(random.Random(1), model="m", client=client)
        await agent.choose_tax_target(game.players[0], game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("faced B 1 duel before tonight -- lost it", prompt)

    async def test_intro_line_combined_includes_a_real_head_to_head_record(self) -> None:
        game = _three_player_game()
        game.record_duel_result(winner_id=0, loser_id=1)
        game.record_duel_result(winner_id=0, loser_id=1)
        client = self._client("Good to see you again, B!")
        agent = LLMAgent(random.Random(1), model="m", client=client)
        await agent.intro_line_combined(
            game.players[0], "Dogs", game.players[1], host_announcement="Let's go!", game=game)
        prompt = client.generate.call_args.args[1]
        self.assertIn("faced B 2 duels before tonight -- won all 2", prompt)

    async def test_attempt_question_never_mentions_opponent_history(self) -> None:
        # attempt_question has no opponent parameter at all and is the
        # highest-frequency call by far -- confirms this feature stayed
        # scoped to the three decision points it was actually designed for.
        game = _three_player_game()
        game.record_duel_result(winner_id=0, loser_id=1)
        client = self._client("A")
        agent = LLMAgent(random.Random(1), model="m", client=client)
        await agent.attempt_question(
            game.players[0], _question(), _domain(), distractors=["cat", "bird"], game=game)
        prompt = client.generate.call_args.args[1]
        self.assertNotIn("faced B", prompt)


class ChooseTargetInterviewTest(unittest.IsolatedAsyncioTestCase):
    """Scott: "'choosing a target' part of the players role needs to be
    more out loud like an interview and quicker. we give them 7 seconds
    to pick before we choose for them, first adjacent." """

    async def test_returns_the_models_spoken_reason(self) -> None:
        game = _three_player_game()
        options = game.adjacent_opponents(0)
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text="1: They're weaker on defense.\nMEMO: none",
            think_seconds=0.1, raw_latency_seconds=0.1)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        target_id, reason = await agent.choose_target(game.players[0], game)
        self.assertEqual(target_id, options[1])
        self.assertEqual(reason, "They're weaker on defense.")

    async def test_uses_the_tight_target_decision_timeout(self) -> None:
        # Scott: "we give them 7 seconds to pick" -- a much tighter budget
        # than the general-purpose generate_timeout every other call uses.
        game = _three_player_game()
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text="0: Going for it.\nMEMO: none", think_seconds=0.1, raw_latency_seconds=0.1)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        await agent.choose_target(game.players[0], game)
        self.assertEqual(client.generate.call_args.kwargs["timeout"], settings.target_decision_timeout)
        self.assertLess(settings.target_decision_timeout, settings.generate_timeout)

    async def test_falls_back_to_first_adjacent_on_an_unparseable_reply(self) -> None:
        # Scott: "before we choose for them, first adjacent" -- NOT
        # super().choose_target()'s fuller temperament-weighted pick.
        game = _three_player_game()
        client = AsyncMock()
        client.supports_grammar = False
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMAgent(random.Random(1), model="m", client=client)
        target_id, reason = await agent.choose_target(game.players[0], game)
        self.assertEqual(target_id, min(game.adjacent_opponents(0)))
        self.assertIn("closest", reason)


if __name__ == "__main__":
    unittest.main()
