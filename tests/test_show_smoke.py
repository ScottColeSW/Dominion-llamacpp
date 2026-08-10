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
