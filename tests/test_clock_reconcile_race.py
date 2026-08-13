"""Proves the engine-side precondition behind "the clock is ticking down on
the wrong player" (Scott's live report): game.py's fire_commentator runs
the Commentator's matchup reaction as a detached asyncio.create_task, so
its commentator_line event can genuinely be emitted AFTER later, unrelated
events that were logged while that background call was still in flight --
not just theoretically, but reproducibly, on demand.

This is the deterministic-repro half of the fix (the frontend half lives in
web/index.html's reconcile block, verified separately -- no JS test harness
exists in this codebase, only the established node --check syntax pattern
plus live manual testing). What CAN be pinned down here, in Python, is the
actual root cause: that a background-emitted event reaching the wire
out of order with the synchronous events around it is real, reachable
engine behavior, not a hypothetical. The frontend fix (gate the clock
commit on ev.type === 'duel_turn' && ev.data.player_id === liveTick.playerId,
instead of "whatever arrives next") is correct precisely because this
reordering is possible and has to be tolerated, not prevented.

Deliberately does NOT import dominion.engine.game or anything that
transitively imports it: test_show_smoke.py sets
os.environ["DOMINION_SCRIPTED_ONLY"] = "1" at its own import time, and
game.SCRIPTED_ONLY is a plain module-level constant read once at first
import and cached in sys.modules for the whole test session -- whichever
test file some collection order happens to import game.py through first
"wins" that constant for every other test file too. Reproducing the exact
fire_commentator pattern directly (asyncio.create_task wrapping a slow
call, plain synchronous emits happening around it) sidesteps that entirely
and is a faithful match: fire_commentator is a closure over `events`/`emit`
inside _run_show, not something importable in isolation anyway.
"""
from __future__ import annotations

import asyncio
import unittest

from dominion.engine.events import EventLog


class FireCommentatorRaceTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_background_task_can_emit_after_later_synchronous_events(self) -> None:
        log = EventLog()

        async def slow_matchup_reaction() -> str:
            # Stands in for LLMCommentatorAgent.react_to_matchup's real
            # await self.client.generate(...) -- genuinely slow relative to
            # the synchronous game-loop work that keeps running around it.
            await asyncio.sleep(0.05)
            return "Reacting to a matchup that's long since moved on."

        async def fire_commentator(moment: str, coro) -> None:
            # The exact shape of game.py's fire_commentator: schedule, do
            # not await, let it emit whenever it actually resolves.
            async def _run() -> None:
                text = await coro
                log.emit("commentator_line", moment=moment, text=text)
            asyncio.create_task(_run())

        # duel 1's matchup reaction goes out as a background task...
        await fire_commentator("matchup", slow_matchup_reaction())

        # ...but the main game loop doesn't wait for it -- duel 1's own
        # attempt turns, and even duel 2's challenge, keep emitting
        # synchronously in the meantime, exactly like _run_show's real
        # sequential loop does around fire_commentator's call sites.
        log.emit("agent_thinking", player_id=0, decision="attempt")
        log.emit("duel_turn", player_id=0, outcome="correct")
        log.emit("duel_result", winner_id=0, loser_id=1)
        log.emit("challenge_declared", challenger_id=0, defender_id=2)
        log.emit("agent_thinking", player_id=0, decision="attempt")

        # Let the background task actually finish before asserting.
        await asyncio.sleep(0.1)

        events = list(log)
        types_in_order = [e.type for e in events]
        commentator_index = types_in_order.index("commentator_line")
        # The whole point: duel 1's matchup commentary was scheduled FIRST,
        # but landed on the wire well AFTER duel 2's own challenge_declared
        # and its first live attempt -- proving a background-emitted event
        # can interleave with a LATER, logically unrelated duel's events,
        # not just later events within its own duel.
        self.assertGreater(commentator_index, types_in_order.index("challenge_declared"))
        self.assertGreater(commentator_index, types_in_order.index("duel_result"))
        # And every event that was emitted synchronously, in real call
        # order, keeps that exact relative order regardless -- only the
        # background task's own placement is "late," nothing else reorders.
        synchronous_types = [t for t in types_in_order if t != "commentator_line"]
        self.assertEqual(synchronous_types, [
            "agent_thinking", "duel_turn", "duel_result", "challenge_declared", "agent_thinking",
        ])


if __name__ == "__main__":
    unittest.main()
