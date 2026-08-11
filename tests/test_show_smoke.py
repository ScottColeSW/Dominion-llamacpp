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
