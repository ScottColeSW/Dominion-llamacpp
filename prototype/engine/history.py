"""Long-term SQLite capture of every show played, so patterns -- which model
tends to win, whether the clock-fairness fix (see ollama_agent.py) actually
holds, accuracy trends -- can be analyzed across many games instead of just
whichever one is currently on screen.

Purely a passive observer of the same event stream server.py already
streams to the browser: HistoryRecorder.on_event() is meant to be chained
into the same EventLog(on_emit=...) callback used for streaming, right
alongside it. game.py has no idea this exists, matching the project's
existing "engine emits events, consumers subscribe" design (see events.py).

Query examples once you've got a few shows recorded:

    -- win rate by model
    SELECT model, COUNT(*) FILTER (WHERE wins > 0) * 1.0 / COUNT(*) AS win_rate
    FROM player_stats GROUP BY model;

    -- has the challenger/defender bias actually gone away?
    SELECT reason,
           SUM(winner_id = challenger_id) AS challenger_wins,
           SUM(winner_id = defender_id) AS defender_wins
    FROM duels GROUP BY reason;
"""
from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "dominion_history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS shows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    scripted_only INTEGER NOT NULL,
    champion_id INTEGER,
    champion_model TEXT,
    champion_kingdom TEXT,
    champion_profession TEXT,
    champion_domain TEXT,
    total_duels INTEGER,
    prize INTEGER
);

CREATE TABLE IF NOT EXISTS duels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id INTEGER NOT NULL REFERENCES shows(id),
    duel_index INTEGER NOT NULL,
    challenger_id INTEGER,
    challenger_model TEXT,
    defender_id INTEGER,
    defender_model TEXT,
    tested_domain TEXT,
    winner_id INTEGER,
    reason TEXT,
    turns INTEGER,
    challenger_clock_remaining REAL,
    defender_clock_remaining REAL
);

CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    show_id INTEGER NOT NULL REFERENCES shows(id),
    player_id INTEGER,
    model TEXT,
    kingdom_name TEXT,
    profession TEXT,
    starting_domain TEXT,
    wins INTEGER,
    correct INTEGER,
    incorrect INTEGER,
    passed INTEGER,
    total_correct_seconds REAL,
    is_champion INTEGER
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Idempotent; call once at server startup."""
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class HistoryRecorder:
    """Feed every emitted event to on_event(); writes one show + its duels +
    per-player stats to SQLite once the finale event lands. on_event never
    raises -- a storage hiccup (locked/corrupt db file, disk full) should
    never be able to interrupt or crash a live show over what's ultimately
    a nice-to-have."""

    def __init__(self, seed: Optional[int], scripted_only: bool, db_path: str = DB_PATH):
        self.seed = seed
        self.scripted_only = scripted_only
        self.db_path = db_path
        self.started_at = _utcnow()
        self.players: Dict[int, Dict[str, Any]] = {}  # player_id -> draft info + running stats
        self.duel_index = 0
        self.duels: List[Dict[str, Any]] = []
        self._pending_challenge: Optional[Dict[str, Any]] = None

    def on_event(self, ev: Any) -> None:
        try:
            self._handle(ev)
        except Exception:
            pass

    def _handle(self, ev: Any) -> None:
        etype, data = ev.type, ev.data
        if etype == "draw_assignment":
            pid = data["player_id"]
            self.players[pid] = {
                "model": data["model"], "kingdom_name": data["kingdom_name"],
                "profession": data["profession"], "starting_domain": data["domain"],
                "wins": 0, "correct": 0, "incorrect": 0, "passed": 0,
                "total_correct_seconds": 0.0,
            }
        elif etype == "challenge_declared":
            self._pending_challenge = {
                "challenger_id": data["challenger_id"], "defender_id": data["defender_id"],
                "tested_domain": data["tested_domain"],
            }
        elif etype == "duel_turn":
            p = self.players.get(data["player_id"])
            if p is None:
                return
            outcome = data["outcome"]
            if outcome == "correct":
                p["correct"] += 1
                p["total_correct_seconds"] += data["seconds_used"]
            elif outcome == "incorrect":
                p["incorrect"] += 1
            elif outcome == "passed":
                p["passed"] += 1
        elif etype == "duel_result":
            c = self._pending_challenge
            if c is None:
                return
            winner_id = data["winner_id"]
            if winner_id in self.players:
                self.players[winner_id]["wins"] += 1
            clocks = data.get("clocks_remaining") or {}
            self.duel_index += 1
            self.duels.append({
                "duel_index": self.duel_index,
                "challenger_id": c["challenger_id"],
                "challenger_model": self.players.get(c["challenger_id"], {}).get("model"),
                "defender_id": c["defender_id"],
                "defender_model": self.players.get(c["defender_id"], {}).get("model"),
                "tested_domain": c["tested_domain"],
                "winner_id": winner_id,
                "reason": data["reason"],
                "turns": data["turns"],
                "challenger_clock_remaining": clocks.get(c["challenger_id"]),
                "defender_clock_remaining": clocks.get(c["defender_id"]),
            })
            self._pending_challenge = None
        elif etype == "finale":
            self._write(data)

    def _write(self, finale_data: Dict[str, Any]) -> None:
        champ_id = finale_data["champion_id"]
        champ = self.players.get(champ_id, {})
        conn = sqlite3.connect(self.db_path, timeout=5)
        try:
            cur = conn.execute(
                "INSERT INTO shows (seed, started_at, finished_at, scripted_only, "
                "champion_id, champion_model, champion_kingdom, champion_profession, "
                "champion_domain, total_duels, prize) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.seed, self.started_at, _utcnow(), int(self.scripted_only),
                 champ_id, champ.get("model"), finale_data["champion_kingdom"],
                 finale_data["champion_profession"], finale_data["champion_domain"],
                 finale_data["total_duels"], finale_data["prize"]),
            )
            show_id = cur.lastrowid
            for d in self.duels:
                conn.execute(
                    "INSERT INTO duels (show_id, duel_index, challenger_id, challenger_model, "
                    "defender_id, defender_model, tested_domain, winner_id, reason, turns, "
                    "challenger_clock_remaining, defender_clock_remaining) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (show_id, d["duel_index"], d["challenger_id"], d["challenger_model"],
                     d["defender_id"], d["defender_model"], d["tested_domain"], d["winner_id"],
                     d["reason"], d["turns"], d["challenger_clock_remaining"],
                     d["defender_clock_remaining"]),
                )
            for pid, p in self.players.items():
                conn.execute(
                    "INSERT INTO player_stats (show_id, player_id, model, kingdom_name, "
                    "profession, starting_domain, wins, correct, incorrect, passed, "
                    "total_correct_seconds, is_champion) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (show_id, pid, p["model"], p["kingdom_name"], p["profession"],
                     p["starting_domain"], p["wins"], p["correct"], p["incorrect"],
                     p["passed"], round(p["total_correct_seconds"], 1), int(pid == champ_id)),
                )
            conn.commit()
        finally:
            conn.close()
