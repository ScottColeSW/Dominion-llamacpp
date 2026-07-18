"""Live, Ollama-backed agent for the three lower-frequency player decisions:
who to challenge (choose_target), whether to keep pushing (decide_continue),
and who to domain-tax (choose_tax_target).

Trivia answering (attempt_question) deliberately stays on the inherited
ScriptedAgent behavior: on this project's dev hardware a single Ollama call
took 11s warm / 86s cold (CPU-only inference; see prototype/README.md), which
would blow through the 25-second duel clock in one or two turns. The three
decisions above happen only a few times per duel, so real latency there is
tolerable in a way it isn't for the timed trivia exchange.

Every live call degrades to the scripted fallback on any failure -- timeout,
connection error, unparseable reply -- so a slow, hung, or unreachable Ollama
server can never stall or crash the show.
"""
from __future__ import annotations
import json
import random
import re
import urllib.error
import urllib.request
from typing import Optional

from .agents import ScriptedAgent
from .models import Player, GameState

OLLAMA_URL = "http://localhost:11434/api/generate"
# Generous given observed cold-start cost (86s on this dev machine); a slow
# reply still beats a wrong/absent one, and the fallback below covers a
# reply that never comes at all.
OLLAMA_TIMEOUT = 90.0

TEXT_MODELS = ["llama3.2:latest", "gpt-oss:20b", "qwen2.5:3b", "gemma2:2b", "phi3:mini"]


def _ask_ollama(model: str, prompt: str) -> Optional[str]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def _parse_index(reply: Optional[str], count: int) -> Optional[int]:
    if not reply or count <= 0:
        return None
    m = re.search(r"\d+", reply)
    if not m:
        return None
    idx = int(m.group())
    return idx if 0 <= idx < count else None


class OllamaAgent(ScriptedAgent):
    """Same interface as ScriptedAgent; overrides only the three decision
    points to consult this player's assigned local model, falling back to
    the inherited scripted logic whenever the call doesn't yield a usable
    answer."""

    def __init__(self, rng: random.Random, model: str, **kwargs):
        super().__init__(rng, **kwargs)
        self.model = model

    def choose_target(self, player: Player, game: GameState) -> Optional[int]:
        options = game.adjacent_opponents(player.id)
        if not options or len(options) == 1:
            return super().choose_target(player, game)
        lines = [
            f"{i}: {game.players[pid].kingdom_name}, holds {len(game.players[pid].territory)} "
            f"tile(s), currently plays {game.players[pid].domain}"
            for i, pid in enumerate(options)
        ]
        prompt = (
            f"You are {player.kingdom_name}, a {player.profession} on a trivia game show, "
            f"playing style: {player.temperament_label()}. Pick ONE opponent to challenge next.\n"
            + "\n".join(lines)
            + "\nReply with ONLY the number of your choice."
        )
        idx = _parse_index(_ask_ollama(self.model, prompt), len(options))
        return options[idx] if idx is not None else super().choose_target(player, game)

    def decide_continue(self, player: Player, game: GameState) -> bool:
        if not game.adjacent_opponents(player.id):
            return False
        prompt = (
            f"You are {player.kingdom_name}, a {player.profession}, playing style: "
            f"{player.temperament_label()}. You are on a {player.push_streak}-win streak. "
            f"Do you keep pushing for more territory, or retreat to defend what you have?\n"
            f"Reply with ONLY the word PUSH or RETREAT."
        )
        reply = _ask_ollama(self.model, prompt)
        if reply:
            upper = reply.upper()
            if "PUSH" in upper:
                return True
            if "RETREAT" in upper:
                return False
        return super().decide_continue(player, game)

    def choose_tax_target(self, player: Player, game: GameState) -> Optional[int]:
        candidates = [pid for pid in game.active_ids if pid != player.id]
        if not candidates:
            return None
        lines = [
            f"{i}: {game.players[pid].kingdom_name}, currently plays {game.players[pid].domain}"
            for i, pid in enumerate(candidates)
        ]
        prompt = (
            f"You are {player.kingdom_name}. You've earned a Domain Tax: swap domains with one "
            f"opponent of your choice.\n" + "\n".join(lines) + "\nReply with ONLY the number."
        )
        idx = _parse_index(_ask_ollama(self.model, prompt), len(candidates))
        return candidates[idx] if idx is not None else super().choose_tax_target(player, game)
