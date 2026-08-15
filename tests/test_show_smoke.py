"""End-to-end regression: a full scripted-only show, unchanged behavior
guardrail for the src/dominion package restructuring (and, from M2
onward, for the async agent-call boundary). Mirrors
prototype/README.md's original "200 simulated shows, zero failures,
always exactly twelve duels" verification, pinned down as an actual test
instead of a one-off manual run.
"""
from __future__ import annotations

import os
import unittest

os.environ["DOMINION_SCRIPTED_ONLY"] = "1"

from dominion.engine.events import EventLog  # noqa: E402
from dominion.engine.game import PLAYER_COUNT, run_show  # noqa: E402


class ShowSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, seed: int) -> tuple[dict, EventLog]:
        log = EventLog()
        result = await run_show(seed=seed, log=log)
        return result, log

    async def test_exactly_twelve_duels_and_valid_finale(self) -> None:
        # A show always ends with exactly one duel per elimination and
        # PLAYER_COUNT - 1 players eliminated -- see engine/game.py's
        # module docstring and prototype/README.md's "Verified" section.
        for seed in (1, 2, 3, 4, 5):
            with self.subTest(seed=seed):
                result, log = await self._run(seed)
                self.assertEqual(result["total_duels"], PLAYER_COUNT - 1)
                finale_events = [e for e in log if e.type == "finale"]
                self.assertEqual(len(finale_events), 1)
                self.assertEqual(finale_events[0].data["champion_id"], result["champion_id"])

    async def test_deterministic_for_a_fixed_seed(self) -> None:
        result_a, _ = await self._run(42)
        result_b, _ = await self._run(42)
        self.assertEqual(result_a, result_b)

    async def test_many_seeds_run_without_error(self) -> None:
        # A broader sweep than the other tests here -- no crash, and every
        # run's own internal invariants (game.py raises nothing, always
        # reaches a sole owner) hold across a wider range of seeds/rng
        # paths than the few spot-checked above.
        for seed in range(100, 130):
            result, _ = await self._run(seed)
            self.assertEqual(result["total_duels"], PLAYER_COUNT - 1)

    async def test_host_line_events_appear_for_challenge_and_finale(self) -> None:
        # M9: the Host is a real agent now (agents/host_agent.py) --
        # proves the wiring end to end even in scripted-only mode, where
        # host_agent.announce_challenge/announce_finale still run (just
        # via ScriptedHostAgent, not a live call).
        _, log = await self._run(3)
        host_lines = [e for e in log if e.type == "host_line"]
        moments = [e.data["moment"] for e in host_lines]
        self.assertEqual(moments.count("challenge"), PLAYER_COUNT - 1)  # once per duel
        self.assertEqual(moments.count("finale"), 1)
        self.assertTrue(all(not e.data["live"] for e in host_lines))  # SCRIPTED_ONLY
        self.assertTrue(all(e.data["text"] for e in host_lines))

    async def test_target_chosen_events_carry_a_spoken_reason(self) -> None:
        # Scott: "'choosing a target' part of the players role needs to be
        # more out loud like an interview." Proves the wiring end to end
        # in scripted-only mode -- ScriptedAgent.choose_target's new
        # (target_id, reason) return actually reaches a real event, not
        # just LLMAgent's live path. One per duel where a real choice
        # existed at all (a spotlighted player with only one adjacent
        # opponent, or the forced Threepeat Drive path, never emits this
        # -- see game.py's guard), so this is an upper bound, not
        # necessarily exactly PLAYER_COUNT - 1.
        _, log = await self._run(3)
        target_events = [e for e in log if e.type == "target_chosen"]
        self.assertTrue(target_events)
        self.assertTrue(all(e.data["reason"] for e in target_events))
        self.assertTrue(all(e.data["target_id"] is not None for e in target_events))

    async def test_exactly_one_challenge_declared_is_flagged_as_the_final_duel(self) -> None:
        # Scott: "no one seemed to know when it was the last duel of the
        # whole game." Exactly one challenge_declared per show should be
        # flagged final_duel=True -- the one where, going in, only two
        # players were still active -- and it has to be the LAST
        # challenge_declared in the whole log, since the show ends the
        # instant that duel resolves.
        for seed in (1, 2, 3, 4, 5):
            with self.subTest(seed=seed):
                _, log = await self._run(seed)
                declared = [e for e in log if e.type == "challenge_declared"]
                final_flagged = [e for e in declared if e.data["final_duel"]]
                self.assertEqual(len(final_flagged), 1)
                self.assertIs(final_flagged[0], declared[-1])

    async def test_host_line_events_appear_for_m32_duel_result_and_continue_reaction(self) -> None:
        # M32 Fix 2/3: announce_duel_result fires once per duel (every
        # elimination gets the Host's merged win/goodbye reaction, even
        # the final one -- just deferred, see the ordering test below);
        # announce_continue_decision fires once per non-final duel (the
        # loop breaks before reaching it on the show's very last duel).
        _, log = await self._run(3)
        host_lines = [e for e in log if e.type == "host_line"]
        moments = [e.data["moment"] for e in host_lines]
        self.assertEqual(moments.count("duel_result"), PLAYER_COUNT - 1)
        self.assertEqual(moments.count("continue_reaction"), PLAYER_COUNT - 2)

    async def test_duel_result_events_carry_is_upset_and_is_close(self) -> None:
        # M32 Fix 2: computed once, authoritatively, server-side -- the
        # frontend used to recompute an equivalent pair of booleans from
        # client-tracked territory state.
        _, log = await self._run(3)
        duel_results = [e for e in log if e.type == "duel_result"]
        self.assertEqual(len(duel_results), PLAYER_COUNT - 1)
        for e in duel_results:
            self.assertIn("is_upset", e.data)
            self.assertIn("is_close", e.data)
            self.assertIsInstance(e.data["is_upset"], bool)
            self.assertIsInstance(e.data["is_close"], bool)

    async def test_commentator_line_events_appear_for_every_matchup(self) -> None:
        # M11: the Commentator is a second real agent (agents/
        # commentator_agent.py) -- proves the wiring end to end even in
        # scripted-only mode, where it still runs (just via
        # ScriptedCommentatorAgent, not a live call, and with no stats
        # snapshot at all).
        _, log = await self._run(3)
        commentator_lines = [e for e in log if e.type == "commentator_line"]
        moments = [e.data["moment"] for e in commentator_lines]
        self.assertEqual(moments.count("matchup"), PLAYER_COUNT - 1)  # once per duel
        self.assertTrue(all(not e.data["live"] for e in commentator_lines))  # SCRIPTED_ONLY
        self.assertTrue(all(e.data["text"] for e in commentator_lines))

    async def test_exit_interview_events_appear_once_per_eliminated_player(self) -> None:
        # M12: every duel eliminates exactly one player, and each gets one
        # last word (agents/scripted_agent.py's exit_interview) before the
        # show moves on -- proves the wiring end to end even in
        # scripted-only mode.
        _, log = await self._run(3)
        exit_events = [e for e in log if e.type == "exit_interview"]
        self.assertEqual(len(exit_events), PLAYER_COUNT - 1)  # one per elimination
        self.assertTrue(all(not e.data["live"] for e in exit_events))  # SCRIPTED_ONLY
        self.assertTrue(all(e.data["text"] for e in exit_events))
        # Every eliminated player_id is unique -- nobody gets a second exit
        # interview (they're out for good after their first loss).
        eliminated_ids = [e.data["player_id"] for e in exit_events]
        self.assertEqual(len(eliminated_ids), len(set(eliminated_ids)))

    async def test_finale_beats_land_in_order_host_then_exit_then_celebration(self) -> None:
        # Scott's two sequencing catches, both about the show's very last
        # beat: (1) "at the end of a game, the host should speak first,
        # ending the game formally, then the exit interview," and (2)
        # "game music is going after the game is over and plays during
        # post-game interview" -- the finale event's own frontend handler
        # fires the victory jingle/confetti fire-and-forget, so the
        # deferred exit interview has to land BEFORE it, not after, or
        # its spoken line would audibly overlap that music. Every OTHER
        # elimination's exit_interview still fires immediately (right
        # after its own duel_result, unchanged) -- only the very last one
        # is held to this order: host_line(finale) < host_line(duel_result)
        # < exit_interview < finale -- M32 Fix 2 added the middle step,
        # deferred the same way exit_interview already was.
        _, log = await self._run(3)
        finale_host_index = next(
            i for i, e in enumerate(log)
            if e.type == "host_line" and e.data["moment"] == "finale")
        finale_index = next(i for i, e in enumerate(log) if e.type == "finale")
        last_exit_index = max(i for i, e in enumerate(log) if e.type == "exit_interview")
        last_duel_result_host_index = max(
            i for i, e in enumerate(log)
            if e.type == "host_line" and e.data["moment"] == "duel_result")
        self.assertLess(finale_host_index, last_duel_result_host_index)
        self.assertLess(last_duel_result_host_index, last_exit_index)
        self.assertLess(last_exit_index, finale_index)

    async def test_threepeat_drive_forces_a_push_with_no_target_decision(self) -> None:
        # #13: seed 1 is a confirmed, stable repro where a real
        # push_streak%3==0 checkpoint fires with a genuinely non-adjacent
        # candidate available, and the resulting duel's winner/loser end
        # up with no safe reconnect bridge available (see the next test
        # for the case where one IS available) -- the winner instead
        # claims the conquest as a disconnected outpost, no bystander
        # tiles change hands, and the show's fundamental "always exactly
        # PLAYER_COUNT - 1 duels" shape still holds either way.
        #
        # Re-picked AGAIN here (previously seed 1, before that seed 23,
        # before that seed 102 -- see git history for all three): any
        # change to how many rng.* calls happen before this scenario --
        # this time duel.py's BASE_CLOCK dropping from 30 to 20 (Scott,
        # once tonight's retry/circuit-breaker fixes made real turns
        # reliably fast again: "we can decrease the time for each duel
        # from 30 second clocks to 20 second clocks now"), which changes
        # how many turns fit before a timeout and thus how many rng draws
        # happen per duel -- shifts the entire downstream draw sequence
        # for a fixed seed, which is why this exact scenario keeps
        # landing on a different seed number every time -- the scenario
        # itself is unaffected, just where it falls. A real fragility of
        # pinning exact per-seed engine outcomes like this, not a sign
        # anything is actually broken.
        result, log = await self._run(4)
        events = list(log)
        self.assertEqual(result["total_duels"], PLAYER_COUNT - 1)
        drives = [e for e in events if e.type == "threepeat_drive"]
        self.assertEqual(len(drives), 1)
        self.assertEqual(drives[0].data, {"player_id": 9, "target_id": 5, "streak": 3})
        # The forced push right after it has nothing to decide -- no
        # agent_thinking(decision="target") for that turn, and the
        # continues event says so explicitly.
        drive_index = events.index(drives[0])
        continues_index = next(
            i for i in range(drive_index, len(events)) if events[i].type == "continues")
        self.assertTrue(events[continues_index].data.get("forced"))
        between = events[drive_index:continues_index]
        self.assertFalse(any(
            e.type == "agent_thinking" and e.data.get("decision") == "target" for e in between))
        self.assertFalse(any(e.type == "player_eliminated_by_reconnect" for e in events))

    async def test_threepeat_drive_can_resolve_via_a_safe_reconnect_bridge(self) -> None:
        # #13: seed 61 is a confirmed, stable repro of the OTHER branch --
        # a safe bridge (one that doesn't fully strip any bystander) does
        # exist for this particular matchup, so forced_reconnect actually
        # fires instead of falling back to an outpost. Re-picked (again --
        # see the sibling test above) from seed 120, before that 59,
        # before that 44, before that 143 -- this time because duel.py's
        # BASE_CLOCK dropped from 30 to 20, same fragility the sibling
        # test's own comment already documents.
        result, log = await self._run(61)
        self.assertEqual(result["total_duels"], PLAYER_COUNT - 1)
        drives = [e for e in log if e.type == "threepeat_drive"]
        reconnects = [e for e in log if e.type == "forced_reconnect"]
        self.assertEqual(len(drives), 1)
        self.assertEqual(len(reconnects), 1)
        self.assertTrue(reconnects[0].data["tiles"])
        self.assertFalse(any(e.type == "player_eliminated_by_reconnect" for e in log))

    async def test_threepeat_drive_never_causes_a_collateral_elimination(self) -> None:
        # The invariant the outpost fallback + risk-aware bridge search
        # both exist specifically to guarantee: whatever a Drive resolves
        # to, it never eliminates a bystander outside a real duel, and
        # the show-wide "always exactly PLAYER_COUNT - 1 duels" shape
        # (prototype/README.md's original verification) never breaks --
        # swept across a wide seed range, not just the two confirmed
        # single-repro cases above.
        for seed in range(200, 260):
            with self.subTest(seed=seed):
                result, log = await self._run(seed)
                self.assertEqual(result["total_duels"], PLAYER_COUNT - 1)
                self.assertFalse(any(e.type == "player_eliminated_by_reconnect" for e in log))

    async def test_custom_models_list_overrides_the_fixed_roster(self) -> None:
        # The model picker's selection (server/app.py's ?models= query
        # params) should be exactly what players draw from, not just a
        # filter on top of TEXT_MODELS -- even a roster with names that
        # don't match any real TEXT_MODELS/LLAMACPP_MODELS entry.
        log = EventLog()
        custom_models = ["picker-custom-a", "picker-custom-b"]
        await run_show(seed=7, log=log, models=custom_models)
        assigned = {e.data["model"] for e in log if e.type == "draw_assignment"}
        self.assertEqual(assigned, set(custom_models))


if __name__ == "__main__":
    unittest.main()
