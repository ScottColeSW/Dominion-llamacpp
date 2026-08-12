#!/usr/bin/env python3
"""#14: an empirical tuning pass -- run a batch of shows headlessly (no
server, no browser) and report the same aggregate stats web/stats.html
shows, so a real balance question (a model's win rate, push/retreat
tendencies, live-vs-fallback reliability, latency) can be answered from a
real sample size instead of watching one show at a time.

Deliberately run AFTER M9-M13 landed (see this repo's own task notes) --
those milestones changed real gameplay balance (the AI Host/Commentator
add live-call latency and cost, the Threepeat Drive mechanic changes how
often a duel is non-adjacent, the wall-clock-accurate clock fix changes
how fast real backend latency drains a player's clock), so a tuning pass
taken before they existed would be measuring a different game.

Reuses the exact same HistoryRecorder/get_stats() pair server/app.py's
_stream_show already wires up for a live show (see run_show's own
docstring) -- this script is not a separate stats system, just a
headless driver for the real one.

Usage:
    python scripts/run_show_batch.py --count 20
    python scripts/run_show_batch.py --count 50 --backend llamacpp
    python scripts/run_show_batch.py --count 200 --scripted-only
    python scripts/run_show_batch.py --count 20 --db-path /tmp/scratch.db
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

REPO_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, required=True, help="how many shows to run")
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default=None,
                         help="defaults to DOMINION_INFERENCE_BACKEND / settings default")
    parser.add_argument("--models", default=None,
                         help="comma-separated model roster override, same as the "
                              "start screen's picker (?models=) -- default: the fixed roster")
    parser.add_argument("--scripted-only", action="store_true",
                         help="skip live model calls entirely -- fast, but the resulting "
                              "stats reflect ScriptedAgent's heuristics, not any real model")
    parser.add_argument("--db-path", default=None,
                         help="defaults to the same DB the live server writes to "
                              "(history.py's DB_PATH) -- override to avoid mixing a "
                              "throwaway batch into real recorded history")
    parser.add_argument("--seed", type=int, default=None,
                         help="base seed for a reproducible batch (seed, seed+1, ...); "
                              "default is a fresh random seed per show, same as a real "
                              "audience never seeing the same draft twice")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    # Must happen BEFORE any dominion import -- engine/game.py's
    # SCRIPTED_ONLY is a module-level constant read once at import time
    # from this exact env var (see test_show_smoke.py's own top-of-file
    # comment for the same reasoning).
    if args.scripted_only:
        os.environ["DOMINION_SCRIPTED_ONLY"] = "1"

    from dominion.engine.events import EventLog  # noqa: E402
    from dominion.engine.game import _default_client_for_backend, run_show  # noqa: E402
    from dominion.engine.history import HistoryRecorder, get_stats, init_db  # noqa: E402
    from dominion.inference.config import settings  # noqa: E402
    from dominion.inference.base import InferenceClient  # noqa: E402

    @asynccontextmanager
    async def maybe_client(scripted_only: bool, backend_name: str) -> AsyncIterator[Optional[InferenceClient]]:
        # A scripted-only batch never makes a single live call (SCRIPTED_ONLY
        # short-circuits run_show before any agent call), so opening a real
        # InferenceClient (a real httpx connection) for the whole run would
        # be pure waste -- yield None and skip it entirely in that case.
        if scripted_only:
            yield None
        else:
            async with _default_client_for_backend(backend_name) as client:
                yield client

    backend = args.backend or settings.inference_backend
    models = args.models.split(",") if args.models else None
    db_path = args.db_path or None  # None lets HistoryRecorder/get_stats use their own DB_PATH default

    # Same schema-creation call server/app.py's own lifespan makes once at
    # startup (CREATE TABLE IF NOT EXISTS -- safe to call every run, real
    # history is never touched if the tables already exist).
    init_db(db_path) if db_path else init_db()

    print(f"Running {args.count} show(s) -- backend={backend}, "
          f"scripted_only={args.scripted_only}, db={db_path or '(default)'}")
    started = time.monotonic()

    async with maybe_client(args.scripted_only, backend) as client:
        for i in range(args.count):
            seed = (args.seed + i) if args.seed is not None else random.randint(0, 2_000_000_000)
            recorder_kwargs = {"db_path": db_path} if db_path else {}
            recorder = HistoryRecorder(seed=seed, scripted_only=args.scripted_only, **recorder_kwargs)
            log = EventLog(on_emit=recorder.on_event)
            t0 = time.monotonic()
            result = await run_show(seed=seed, log=log, client=client, backend=backend, models=models)
            elapsed = time.monotonic() - t0
            print(f"  [{i + 1}/{args.count}] seed={seed} champion={result['champion']['kingdom_name']} "
                  f"({result['total_duels']} duels, {elapsed:.1f}s)")

    total_elapsed = time.monotonic() - started
    print(f"\nDone in {total_elapsed:.1f}s. Aggregate stats "
          f"({'this batch only' if db_path else 'ALL recorded history, including prior shows'}):\n")
    stats = get_stats(db_path) if db_path else get_stats()
    print(f"Shows recorded: {stats['shows_recorded']}  |  Duels recorded: {stats['duels_recorded']}")
    print(f"\n{'backend':<10} {'model':<24} {'games':>6} {'win_rate':>9} "
          f"{'chal_wr':>8} {'def_wr':>7} {'push%':>7} {'fallback%':>10} {'avg_lat_s':>10}")
    for m in stats["models"]:
        def pct(v: float | None) -> str:
            # Plain ASCII, not an em-dash -- Windows terminals default to a
            # codepage (cp1252/cp437) that can't render U+2014 and prints a
            # mangled replacement glyph instead.
            return f"{v * 100:.0f}%" if v is not None else "-"
        print(f"{m['backend']:<10} {m['model']:<24} {m['games_played']:>6} {pct(m['win_rate']):>9} "
              f"{pct(m['challenger_win_rate']):>8} {pct(m['defender_win_rate']):>7} "
              f"{pct(m['push_rate']):>7} {pct(m['fallback_rate']):>10} "
              f"{m['avg_raw_latency_seconds'] if m['avg_raw_latency_seconds'] is not None else '-':>10}")
    if stats["by_reason"]:
        print("\nDuel end reasons (all recorded history):")
        for reason, r in stats["by_reason"].items():
            print(f"  {reason}: {r['total']} ({r['challenger_wins']} challenger wins, "
                  f"{r['defender_wins']} defender wins)")


if __name__ == "__main__":
    asyncio.run(main())
