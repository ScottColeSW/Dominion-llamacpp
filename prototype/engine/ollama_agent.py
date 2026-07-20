"""Live, Ollama-backed agent for every player decision: who to challenge
(choose_target), whether to keep pushing (decide_continue), who to
domain-tax (choose_tax_target), and trivia answering (attempt_question).

With a working GPU (see prototype/README.md), warm calls run well under a
second, which is why trivia answering can be live here -- on CPU-only
inference a single call took 11s warm / 86s cold, which would have blown
through the duel clock in one or two turns. attempt_question caps its own
wait to roughly what's left on the player's clock (see call_timeout below)
and charges only actual thinking time, not model-load overhead, so both
modes share duel.py's original 25s clock -- see game.py and README.md.

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

# A floor under attempt_question's charged clock time, not an addition on
# top of real latency -- Scott's rule: no turn should read as having taken
# less than a beat, since a real contestant (however sharp) still takes a
# moment to read a question and answer, and an instant answer reads as
# implausible on screen regardless of how fast the underlying model is.
MIN_CHARGED_SECONDS = 1.0


def _ask_ollama(model: str, prompt: str, timeout: float = OLLAMA_TIMEOUT) -> "tuple[Optional[str], Optional[float]]":
    """Returns (reply_text, think_seconds). think_seconds is Ollama's own
    total_duration minus load_duration (both nanoseconds in the response) --
    i.e. actual generation time with model-loading overhead excluded -- or
    None if those fields are missing/malformed. A player whose model happens
    to already be resident vs. one that needs a cold load shouldn't get a
    different trivia-clock charge for the exact same quality of answer; see
    attempt_question, the only caller that uses this second value."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("response") or "").strip()
        think_seconds = None
        total_ns, load_ns = data.get("total_duration"), data.get("load_duration")
        if isinstance(total_ns, (int, float)) and isinstance(load_ns, (int, float)) and total_ns >= load_ns:
            think_seconds = (total_ns - load_ns) / 1e9
        return text, think_seconds
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None, None


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
        reply, _ = _ask_ollama(self.model, prompt)
        idx = _parse_index(reply, len(options))
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
        reply, _ = _ask_ollama(self.model, prompt)
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
        reply, _ = _ask_ollama(self.model, prompt)
        idx = _parse_index(reply, len(candidates))
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
        # PASS is framed here as a legitimate, smart play -- not a last
        # resort admission of failure -- specifically so a model doesn't
        # default to guessing blindly just to avoid saying it. Missing the
        # same question repeatedly (miss_streak) was the exact scenario
        # that motivated duel.py's FORCED_PASS_MISS_STREAK backstop: a
        # model that never volunteers PASS on its own can otherwise hammer
        # the same wrong guess forever, since a wrong answer keeps the same
        # image up for another try. That backstop still exists as a hard
        # guarantee, but a model that understands passing is a real option
        # should rarely need it.
        pass_hint = (
            " You've already missed this exact question before -- guessing "
            "again without a real idea isn't a good look; passing to get a "
            "fresh one is often the smarter play."
            if miss_streak > 0 else ""
        )
        prompt = (
            f"Trivia category: {domain.name}\nClue: {question.image_prompt}\n"
            + "\n".join(lines)
            + "\nReply with ONLY the single letter of your answer, or PASS."
            + pass_hint
            + " Passing when you're genuinely unsure is a legitimate, smart "
              "move on this show, not a failure -- a sharp contestant "
              "doesn't guess blindly just to avoid admitting they don't know."
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
        reply, think_seconds = _ask_ollama(self.model, prompt, timeout=call_timeout)
        elapsed = max(0.2, time.time() - t0)
        # Charge the clock for actual thinking time (Ollama's total_duration
        # minus load_duration), not full wall-clock elapsed -- otherwise
        # whichever player's model happens to need a cold load pays for it
        # out of their own 25s clock as if it were slow reasoning, while the
        # OTHER player in the same duel isn't charged a cent for the exact
        # same infrastructure cost just because their model was already
        # resident. Falls back to wall-clock elapsed only if Ollama's
        # response is missing the duration fields entirely.
        raw_seconds = think_seconds if think_seconds is not None else elapsed
        # Floored at MIN_CHARGED_SECONDS -- Scott's rule: a turn should never
        # read as having taken less than a beat, on top of whatever real
        # latency it already cost. Measured live, small local models were
        # answering in well under a tenth of a second, which is why the
        # question cap kept needing readjustment (README/duel.py docstring):
        # the clock barely moved no matter how many attempts happened, so
        # the cap -- not the clock -- ended up deciding almost every duel.
        # This is a floor, not an addition -- a genuinely slow call still
        # gets charged its real time, never less.
        charged_seconds = round(max(MIN_CHARGED_SECONDS, raw_seconds), 1)
        if not reply:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors)
        upper = reply.strip().upper()
        if upper.startswith("PASS"):
            return AnswerAttempt(outcome="passed", correct=False, seconds_used=charged_seconds,
                                  guess="", live=True)
        m = re.search(r"[A-D]", upper)
        idx = letters.index(m.group()) if (m and m.group() in letters) else None
        if idx is None:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors)
        chosen = choices[idx]
        correct = (chosen == question.answer)
        return AnswerAttempt(outcome="correct" if correct else "incorrect", correct=correct,
                              seconds_used=charged_seconds, guess=chosen, live=True)

    def intro_line(self, player: Player, tested_domain: str,
                    opponent_line: Optional[str] = None) -> str:
        # Called for BOTH the challenger and defender right as a duel opens,
        # before run_duel starts -- not just narration (Scott's ask: the
        # host should talk about the domain owned and challenged, for every
        # player, not just one side). It doubles as a fix for a real
        # fairness bug: choose_target already warms the CHALLENGER's model
        # for free (an untimed call right before the duel), but the
        # defender never got an equivalent warm-up, so their first live
        # trivia answer could eat a real cold-load cost their opponent
        # never paid. Calling this for both sides here gives the defender
        # the same head start, symmetrically, before the timed clock starts
        # -- and since it's a real, full generation call (not a truncated
        # one), it's a genuine warm-up, not a token gesture.
        #
        # The prompt below is deliberately fed a fuller character sheet
        # (profession, playing style, territory held, any active streak),
        # not just domain names -- a bare "you hold X, they're testing Y"
        # prompt kept coming back flat/generic. This is the one moment per
        # duel the audience hears a player "in their own words" before the
        # clock starts, so it needs to actually carry some personality.
        #
        # opponent_line (set only for the second speaker -- game.py calls
        # the challenger first, then passes their actual reply in here for
        # the defender) makes this a real two-way exchange instead of two
        # side-by-side monologues that happen to air back to back: the
        # second player is genuinely responding to what the first one said,
        # per Scott's "has to be 2 ways" ask.
        defending = player.domain == tested_domain
        stakes = (
            f"Tonight's duel tests {tested_domain} -- your own strongest ground, "
            f"you're defending home turf."
            if defending else
            f"You hold {player.domain}, but tonight's duel is on {tested_domain} instead."
        )
        streak_note = (
            f" You're riding a {player.push_streak}-duel win streak tonight."
            if player.push_streak >= 2 else ""
        )
        # A stable identity anchor distinct from current holdings (which
        # drift all show via conquest/tax): only surfaced when it's actually
        # different from what they hold now, so it reads as "who they still
        # are underneath" rather than a redundant restatement of `stakes`
        # above. Scott's ask: give the model a quick memory of its own
        # original domain specifically, not just whatever it happens to be
        # standing on right now.
        origin_note = (
            f" You started this show as the {player.origin_domain} expert -- "
            f"that's still core to who you are, whatever you hold now."
            if player.origin_domain and player.origin_domain != player.domain
            else ""
        )
        reaction_note = (
            f' Your opponent just said, live, on air: "{opponent_line}" React to THEM '
            f"directly, not just to the domain -- agree, push back, needle them, whatever "
            f"fits your style."
            if opponent_line else
            ""
        )
        prompt = (
            f"You are {player.kingdom_name}, a {player.profession} competing live "
            f"on a trivia game show. Your playing style is {player.temperament_label()}. "
            f"You currently control {len(player.territory)} tile(s) of the board.{streak_note}"
            f"{origin_note} "
            f"{stakes}{reaction_note}\n"
            f"In ONE short, in-character sentence -- like a real contestant caught by "
            f"a TV camera, not a narrator describing the scene -- react with some actual "
            f"personality."
        )
        reply, _ = _ask_ollama(self.model, prompt)
        return reply.strip() if reply else super().intro_line(player, tested_domain, opponent_line)
