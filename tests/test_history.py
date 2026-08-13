"""HistoryRecorder + get_stats() against a real temp SQLite file -- the
backend-comparison surface (M5): backend column on shows/player_stats,
per-(backend, model) aggregation, fallback_rate, avg_raw_latency_seconds.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from dominion.engine.history import HistoryRecorder, get_stats, init_db


def _ev(type_: str, **data) -> SimpleNamespace:
    return SimpleNamespace(type=type_, data=data)


class HistoryBackendComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(self.db_path)

    def tearDown(self) -> None:
        os.remove(self.db_path)

    def _record_show(self, backend: str, model: str, seed: int) -> None:
        # player_id 1 deliberately gets a DIFFERENT model ("other-model")
        # than the one under test -- games_played counts player_stats ROWS
        # (player-appearances) for a given (backend, model), not shows, so
        # having both fake players share `model` would silently double-count
        # a single show's contribution to it.
        rec = HistoryRecorder(seed=seed, scripted_only=False, db_path=self.db_path)
        events = [
            _ev("show_start", title="t", players=2, backend=backend),
            _ev("draw_assignment", player_id=0, model=model, kingdom_name="A",
                profession="p", domain="Cats", node=0, on_stage=True, pick_number=1,
                remaining_after=0, temperament=0.5, temperament_label="Balanced"),
            _ev("draw_assignment", player_id=1, model="other-model", kingdom_name="B",
                profession="p", domain="Dogs", node=1, on_stage=False, pick_number=2,
                remaining_after=0, temperament=0.5, temperament_label="Balanced"),
            _ev("challenge_declared", challenger_id=0, defender_id=1, tested_domain="Dogs",
                challenger_using_bonus=False, defender_using_bonus=False, base_clock=25),
            # Both turns are player_id=0's (the model under test) --
            # one live, one fallen back -- so fallback_rate/avg latency
            # below reflect that one model, not split across the two
            # differently-modeled fake players.
            _ev("duel_turn", player_id=0, domain="Dogs", prompt="p", answer="a", guess="b",
                outcome="incorrect", correct=False, seconds_used=3, clock_remaining=22,
                distractors=[], live=False, lucky_blurt=False, raw_latency_seconds=None),
            _ev("duel_turn", player_id=0, domain="Dogs", prompt="p", answer="a", guess="a",
                outcome="correct", correct=True, seconds_used=3, clock_remaining=19,
                distractors=[], live=True, lucky_blurt=False, raw_latency_seconds=0.4),
            _ev("duel_result", winner_id=0, loser_id=1, reason="timeout",
                winner_domain_after="Dogs", questions_seen={0: 1, 1: 1},
                clocks_remaining={0: 22, 1: 0}, turns=2, territory_gained=[1],
                winner_streak=1),
            _ev("finale", champion_id=0, champion_domain="Dogs", champion_kingdom="A",
                champion_profession="p", prize=100, total_duels=1),
        ]
        for ev in events:
            rec.on_event(ev)

    def test_backend_recorded_on_shows_and_player_stats(self) -> None:
        self._record_show("llamacpp", "qwen2.5-3b-instruct", seed=1)
        stats = get_stats(self.db_path)
        self.assertEqual(stats["shows_recorded"], 1)
        models = {(m["backend"], m["model"]): m for m in stats["models"]}
        self.assertIn(("llamacpp", "qwen2.5-3b-instruct"), models)

    def test_same_model_name_different_backends_stay_separate(self) -> None:
        self._record_show("ollama", "shared-name", seed=1)
        self._record_show("llamacpp", "shared-name", seed=2)
        stats = get_stats(self.db_path)
        models = {(m["backend"], m["model"]): m for m in stats["models"]}
        self.assertIn(("ollama", "shared-name"), models)
        self.assertIn(("llamacpp", "shared-name"), models)
        # Each recorded independently -- one game_played each, not merged
        # into a single "shared-name" row that would hide the backend split.
        self.assertEqual(models[("ollama", "shared-name")]["games_played"], 1)
        self.assertEqual(models[("llamacpp", "shared-name")]["games_played"], 1)

    def test_fallback_rate_and_avg_raw_latency(self) -> None:
        self._record_show("llamacpp", "qwen2.5-3b-instruct", seed=1)
        stats = get_stats(self.db_path)
        m = next(m for m in stats["models"] if m["model"] == "qwen2.5-3b-instruct")
        # player_id 0: one live attempt (raw_latency_seconds=0.4).
        # player_id 1: one fallback attempt (raw_latency_seconds=None).
        self.assertEqual(m["live_attempts"], 1)
        self.assertEqual(m["fallback_attempts"], 1)
        self.assertEqual(m["fallback_rate"], 0.5)
        self.assertEqual(m["avg_raw_latency_seconds"], 0.4)

    def test_init_db_is_idempotent(self) -> None:
        init_db(self.db_path)  # setUp already called it once; a second call must not error
        self._record_show("ollama", "m", seed=1)
        stats = get_stats(self.db_path)
        self.assertEqual(stats["shows_recorded"], 1)

    def test_scripted_shows_are_excluded_from_the_real_aggregates(self) -> None:
        # Scott: "all the 'scripted' stats need to be separated from the
        # rest, it skews all the data too much." A scripted-only batch
        # (scripts/run_show_batch.py --scripted-only) writes to the same
        # tables as a real live show -- shows_recorded/duels_recorded/
        # by_reason are meant to describe real model behavior, so a
        # scripted show must not inflate them, even though ScriptedAgent
        # still gets its own row in `models` (it's a real, useful baseline
        # to compare against, just not a "model" to blend into model
        # aggregates).
        self._record_show("ollama", "real-model", seed=1)
        self._record_show("scripted", "scripted", seed=2)
        stats = get_stats(self.db_path)
        self.assertEqual(stats["shows_recorded"], 1)
        self.assertEqual(stats["scripted_shows_recorded"], 1)
        self.assertEqual(stats["duels_recorded"], 1)
        self.assertEqual(sum(r["total"] for r in stats["by_reason"].values()), 1)
        models = {(m["backend"], m["model"]): m for m in stats["models"]}
        self.assertIn(("scripted", "scripted"), models)


if __name__ == "__main__":
    unittest.main()
