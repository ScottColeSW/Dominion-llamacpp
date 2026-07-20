"""Live, Ollama-backed agent for every player decision: who to challenge
(choose_target), whether to keep pushing (decide_continue), who to
domain-tax (choose_tax_target), and trivia answering (attempt_question).

With a working GPU (see prototype/README.md), warm calls run well under a
second, which is why trivia answering can be live here -- on CPU-only
inference a single call took 11s warm / 86s cold, which would have blown
through the duel clock in one or two turns. game.py uses a longer 60s clock
for live shows specifically to give real latency room.

attempt_question turns each question into 4-way multiple choice using the
same distractor pool duel.py already generates for the frontend's cosmetic
blurt animation -- far more reliable than matching free-text guesses against
the answer, and the model is never shown which option is correct.

Every live call degrades to the scripted fallback on any failure -- timeout,
connection error, unparseable reply -- so a slow, hung, or unreachable Ollama
server can never stall or crash the show.
"""
from __future__ import annotations
import json
import random
import re
import time
import urllib.error
import urllib.request
from typing import List, Optional

from .agents import AnswerAttempt, ScriptedAgent
from .content import Question, Domain
from .models import Player, GameState

# 127.0.0.1, not "localhost": on this machine "localhost" resolves to the
# IPv6 loopback (::1) first, but Ollama only listens on IPv4 -- that connect
# attempt doesn't fail fast, it hangs in SYN_SENT for minutes (observed
# directly via netstat), silently defeating every timeout value below since
# the hang happens at the TCP handshake, before any of this module's own
# per-call timeout logic gets a real chance to matter. Skip the ambiguity.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Generous given observed cold-start cost (86s on this dev machine); a slow
# reply still beats a wrong/absent one, and the fallback below covers a
# reply that never comes at all.
OLLAMA_TIMEOUT = 90.0

# gpt-oss:20b was dropped from this pool: it hard-errors ("tensor
# ...ffn_down_exps.weight size overflow") on this machine's Ollama version /
# 8GB-VRAM combination, consistently, not a transient failure. Re-add it
# once that's resolved upstream.
TEXT_MODELS = ["llama3.2:latest", "qwen2.5:3b", "gemma2:2b", "phi3:mini"]


def _ask_ollama(model: str, prompt: str, timeout: float = OLLAMA_TIMEOUT) -> Optional[str]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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

    def attempt_question(self, player: Player, question: Question, domain: Domain,
                          miss_streak: int = 0, distractors: Optional[List[str]] = None,
                          time_remaining: Optional[float] = None) -> AnswerAttempt:
        options = list(distractors or [])
        if not options:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors)
        choices = options + [question.answer]
        self.rng.shuffle(choices)
        letters = "ABCD"[:len(choices)]
        lines = [f"{letters[i]}: {choice}" for i, choice in enumerate(choices)]
        prompt = (
            f"Trivia category: {domain.name}\nClue: {question.image_prompt}\n"
            + "\n".join(lines)
            + "\nReply with ONLY the single letter of your answer, or PASS if you don't know."
        )
        # Cap how long we'll wait for a reply to roughly what this player
        # actually has left on their clock (plus a small grace), not the
        # full OLLAMA_TIMEOUT -- a slow or cold-loading call otherwise keeps
        # running well past the moment this player's clock would already
        # have hit zero, and the duel visibly drags on longer than the
        # audience is watching a countdown for. A 1.5s floor still gives a
        # fast local model a fair shot even when time is nearly out.
        call_timeout = OLLAMA_TIMEOUT
        if time_remaining is not None:
            call_timeout = max(1.5, min(OLLAMA_TIMEOUT, time_remaining + 1.5))
        t0 = time.time()
        reply = _ask_ollama(self.model, prompt, timeout=call_timeout)
        elapsed = max(0.2, time.time() - t0)
        if not reply:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors)
        upper = reply.strip().upper()
        if upper.startswith("PASS"):
            return AnswerAttempt(outcome="passed", correct=False, seconds_used=round(elapsed, 1),
                                  guess="", live=True)
        m = re.search(r"[A-D]", upper)
        idx = letters.index(m.group()) if (m and m.group() in letters) else None
        if idx is None:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors)
        chosen = choices[idx]
        correct = (chosen == question.answer)
        return AnswerAttempt(outcome="correct" if correct else "incorrect", correct=correct,
                              seconds_used=round(elapsed, 1), guess=chosen, live=True)
