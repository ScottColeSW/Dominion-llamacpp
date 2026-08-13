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

    async def test_announce_challenge_mentions_a_real_hot_streak(self) -> None:
        # M32 Fix 1: ported from the old, separate preduelHostLine --
        # the Host used to say this as a SECOND, disconnected line.
        agent = ScriptedHostAgent(random.Random(1))
        challenger, defender = _player(0, "Streaky Kingdom"), _player(1, "Quiet Kingdom")
        line = await agent.announce_challenge(challenger, defender, "Dogs",
                                                challenger_streak=4, defender_streak=0)
        self.assertIn("Streaky Kingdom", line)
        self.assertIn("4", line)

    async def test_announce_challenge_mentions_a_real_territory_gap(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        challenger, defender = _player(0, "Big Kingdom"), _player(1, "Small Kingdom")
        line = await agent.announce_challenge(challenger, defender, "Dogs",
                                                challenger_territory=10, defender_territory=1)
        self.assertIn("Big Kingdom", line)
        self.assertIn("Small Kingdom", line)

    async def test_announce_challenge_falls_back_to_generic_with_no_notable_context(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        challenger, defender = _player(0, "A Kingdom"), _player(1, "Defender Kingdom")
        line = await agent.announce_challenge(challenger, defender, "Dogs")
        self.assertIn("Defender Kingdom", line)

    async def test_announce_challenge_leads_with_final_duel_over_a_hot_streak(self) -> None:
        # Scott: "no one seemed to know when it was the last duel of the
        # whole game." final_duel takes priority over every other tier,
        # including a real hot streak that would otherwise win the pick.
        agent = ScriptedHostAgent(random.Random(1))
        challenger, defender = _player(0, "Streaky Kingdom"), _player(1, "Quiet Kingdom")
        line = await agent.announce_challenge(challenger, defender, "Dogs",
                                                challenger_streak=4, defender_streak=0,
                                                final_duel=True)
        self.assertIn("Streaky Kingdom", line)
        self.assertIn("Quiet Kingdom", line)
        self.assertNotIn("4", line)

    async def test_announce_duel_result_mentions_winner_and_loser(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        winner, loser = _player(0, "Winner Kingdom"), _player(1, "Loser Kingdom")
        line = await agent.announce_duel_result(winner, loser, "Dogs", is_upset=False, is_close=False)
        self.assertIn("Winner Kingdom", line)
        self.assertIn("Loser Kingdom", line)

    async def test_announce_duel_result_upset_reads_differently_from_plain_win(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        winner, loser = _player(0, "Winner Kingdom"), _player(1, "Loser Kingdom")
        line = await agent.announce_duel_result(winner, loser, "Dogs", is_upset=True, is_close=False)
        markers = ("UPSET", "GIANT", "UNDERDOG", "NOBODY SAW THAT COMING")
        self.assertTrue(any(marker in line.upper() for marker in markers))

    async def test_announce_continue_decision_reflects_keep_going(self) -> None:
        agent = ScriptedHostAgent(random.Random(1))
        player = _player(0, "Pusher Kingdom")
        push_line = await agent.announce_continue_decision(player, True, "riding the momentum")
        retreat_line = await agent.announce_continue_decision(player, False, "playing it safe")
        self.assertIn("Pusher Kingdom", push_line)
        self.assertIn("Pusher Kingdom", retreat_line)


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

    async def test_challenge_prompt_includes_streak_and_territory_context_when_notable(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real line.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "Streaky Kingdom"), _player(1, "Quiet Kingdom")
        await agent.announce_challenge(challenger, defender, "Dogs", challenger_streak=5,
                                        challenger_territory=8, defender_territory=1)
        prompt = client.generate.call_args.args[1]
        self.assertIn("5-duel win streak", prompt)
        self.assertIn("bigger kingdom", prompt)

    async def test_challenge_prompt_flags_the_final_duel_unambiguously(self) -> None:
        # Scott: "no one seemed to know when it was the last duel of the
        # whole game." Told to the model as a plain, unmissable fact.
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real line.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.announce_challenge(challenger, defender, "Dogs", final_duel=True)
        prompt = client.generate.call_args.args[1]
        self.assertIn("FINAL DUEL", prompt)
        self.assertIn("sole owner of the whole board", prompt)

    async def test_challenge_prompt_omits_context_when_nothing_notable(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="A real line.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        challenger, defender = _player(0, "A"), _player(1, "B")
        await agent.announce_challenge(challenger, defender, "Dogs")
        prompt = client.generate.call_args.args[1]
        self.assertNotIn("win streak", prompt)
        self.assertNotIn("bigger kingdom", prompt)

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

    async def test_duel_result_falls_back_and_succeeds(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        winner, loser = _player(0, "Winner Kingdom"), _player(1, "Loser Kingdom")
        line = await agent.announce_duel_result(winner, loser, "Dogs", is_upset=False, is_close=False)
        self.assertIn("Winner Kingdom", line)
        self.assertIn("Loser Kingdom", line)

        client.generate.return_value = GenerationResult(
            text="A real live duel reaction!", think_seconds=0.5, raw_latency_seconds=0.5)
        line = await agent.announce_duel_result(winner, loser, "Dogs", is_upset=True, is_close=False)
        self.assertEqual(line, "A real live duel reaction!")

    async def test_duel_result_prompt_mentions_upset_when_true(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reaction.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        winner, loser = _player(0, "Winner Kingdom"), _player(1, "Loser Kingdom")
        await agent.announce_duel_result(winner, loser, "Dogs", is_upset=True, is_close=False)
        prompt = client.generate.call_args.args[1]
        self.assertIn("UPSET", prompt)

    async def test_continue_decision_falls_back_and_succeeds(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text=None, think_seconds=None, raw_latency_seconds=0.1)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        player = _player(0, "Pusher Kingdom")
        line = await agent.announce_continue_decision(player, True, "riding the momentum")
        self.assertIn("Pusher Kingdom", line)

        client.generate.return_value = GenerationResult(
            text="A real live continue reaction!", think_seconds=0.5, raw_latency_seconds=0.5)
        line = await agent.announce_continue_decision(player, True, "riding the momentum")
        self.assertEqual(line, "A real live continue reaction!")

    async def test_continue_decision_prompt_includes_the_real_reason(self) -> None:
        client = AsyncMock()
        client.generate.return_value = GenerationResult(
            text="Reaction.", think_seconds=0.5, raw_latency_seconds=0.5)
        agent = LLMHostAgent(random.Random(1), model="m", client=client)
        player = _player(0, "Pusher Kingdom")
        await agent.announce_continue_decision(player, True, "I can feel it, one more!")
        prompt = client.generate.call_args.args[1]
        self.assertIn("I can feel it, one more!", prompt)


if __name__ == "__main__":
    unittest.main()
