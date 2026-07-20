"""Duel resolution: the chess-clock mechanic from Section 4 of the design
doc, implemented exactly to the corrected Revision 19 rules.

- Both players are tested on the DEFENDER's domain only.
- Each player has an independent clock (25s, +5s if a time bonus is spent).
  Revision 18 shortened the base clock from 60s to 25s specifically to
  burn through duels faster and reduce how many times any single hot
  domain can get re-tested over the course of a whole show.
- A turn ends only on a correct answer; wrong answers and passes keep the
  same player's turn going and keep draining their own clock.
- A wrong (non-pass) answer keeps showing the SAME image again, the player
  gets another crack at the thing they just missed. Only a correct answer
  (control passes to the other player) or an explicit pass (a deliberate
  skip) brings up a new image.
- The only true elimination trigger is a clock reaching zero. The clock
  value recorded for display is clamped at 0: the real internal clock can
  dip slightly negative the instant it crosses zero (a slow final answer
  can burn more time than was left), but elimination and comparisons use
  the real value, only the number shown to the audience is clamped.
- The whole duel is capped at 30 questions seen TOTAL, combined across
  both players (Revision 18, tightened from 25 seen per player, i.e. up
  to 50 total, specifically to shorten duels further). The instant that
  combined total hits 30, the duel resolves immediately by comparing
  remaining time.
- Questions are drawn from a show-wide used_questions set (threaded in by
  the caller) so the same image/answer pair is avoided for the entire
  show, not just this duel, falling back to a repeat only once a domain's
  whole pool is genuinely exhausted.
- on_turn, if given, fires immediately after each turn is computed (not
  just appended to the returned DuelResult.turns_log) -- with a live agent
  (engine/ollama_agent.py), computing a single turn can itself take real
  seconds, so the caller needs per-turn events as they happen to stream
  live, rather than the entire duel resolving silently before anything
  reaches the frontend.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Set, Tuple
import random

from .content import Domain, Question
from .agents import ScriptedAgent
from .models import Player


QUESTION_CAP = 30  # total across BOTH players in one duel, not per-player
BASE_CLOCK = 25.0


@dataclass
class DuelResult:
    winner_id: int
    loser_id: int
    reason: str                      # "timeout" or "question_cap"
    turns_log: List[Dict[str, Any]] = field(default_factory=list)
    clocks_remaining: Dict[int, float] = field(default_factory=dict)
    questions_seen: Dict[int, int] = field(default_factory=dict)


def _draw_question(domain: Domain, used_questions: Set[Tuple[str, str]],
                    rng: random.Random) -> Question:
    available = [q for q in domain.questions
                 if (domain.name, q.image_prompt) not in used_questions]
    if not available:
        # This domain's whole pool has already been shown elsewhere in the
        # show; a repeat here is a last resort, not the normal case.
        available = list(domain.questions)
    question = rng.choice(available)
    used_questions.add((domain.name, question.image_prompt))
    return question


def _pick_distractors(domain: Domain, question: Question, rng: random.Random) -> List[str]:
    """A handful of OTHER real answers from this same domain, for the
    frontend to flash through as quick rapid-fire guesses while the clock
    ticks -- Scott's note that the old static "is thinking..." label read
    as too passive; a real contestant blurts out guesses rather than
    sitting there quietly. Picked once per question, not per attempt, so a
    stubborn wrong-answer loop on the same image keeps cycling the same
    plausible pool instead of reshuffling every single turn. Purely
    cosmetic -- never includes the real answer itself."""
    pool = [q.answer for q in domain.questions if q.answer != question.answer]
    return rng.sample(pool, k=min(3, len(pool))) if pool else []


def run_duel(challenger: Player, defender: Player, domain: Domain,
             agents: Dict[int, ScriptedAgent], rng: random.Random,
             challenger_bonus: bool = False, defender_bonus: bool = False,
             used_questions: Optional[Set[Tuple[str, str]]] = None,
             base_clock: float = BASE_CLOCK,
             on_turn: Optional[Callable[[Dict[str, Any]], None]] = None) -> DuelResult:

    if used_questions is None:
        used_questions = set()

    clocks = {
        challenger.id: base_clock + (5.0 if challenger_bonus else 0.0),
        defender.id: base_clock + (5.0 if defender_bonus else 0.0),
    }
    seen = {challenger.id: 0, defender.id: 0}
    turns_log: List[Dict[str, Any]] = []

    turn_order = [challenger.id, defender.id]
    players_by_id = {challenger.id: challenger, defender.id: defender}
    current = 0  # index into turn_order; challenger acts first
    miss_streak = 0  # consecutive incorrect-or-passed attempts on this turn
    total_seen = 0  # combined attempts across BOTH players this duel

    question = _draw_question(domain, used_questions, rng)
    distractors = _pick_distractors(domain, question, rng)

    while True:
        pid = turn_order[current]
        agent = agents[pid]
        player = players_by_id[pid]

        # time_remaining lets a live agent (OllamaAgent) cap how long it'll
        # wait for a reply to roughly what's actually left on THIS player's
        # clock -- otherwise a slow/cold live call can run well past the
        # moment this player's clock would have hit zero, and the duel
        # visibly drags on past when the audience already expects it to
        # end. ScriptedAgent ignores this; it's never slow enough to matter.
        attempt = agent.attempt_question(player, question, domain, miss_streak=miss_streak,
                                          distractors=distractors, time_remaining=clocks[pid])

        clocks[pid] -= attempt.seconds_used
        seen[pid] += 1
        total_seen += 1
        turns_log.append({
            "player_id": pid, "domain": domain.name, "prompt": question.image_prompt,
            "answer": question.answer, "guess": attempt.guess, "outcome": attempt.outcome,
            "correct": attempt.correct, "seconds_used": attempt.seconds_used,
            "distractors": distractors,
            # Clamped to 0 for display: the real clock can dip slightly
            # negative the instant it crosses zero, but the audience should
            # never see a negative number on screen. Elimination logic
            # below still uses the real, unclamped clocks[pid] value.
            "clock_remaining": round(max(0.0, clocks[pid]), 1),
        })
        # Fire as each turn is actually computed, not just appended to
        # turns_log for the caller to replay after the whole duel finishes.
        # With a live agent, computing one turn can itself take real
        # seconds; without this, game.py's caller would block silently
        # through the entire duel before streaming any of it to the
        # frontend, which is exactly the "plays longer than needed" feel a
        # per-call timeout cap alone doesn't fix.
        if on_turn is not None:
            on_turn(turns_log[-1])

        if clocks[pid] <= 0:
            other = turn_order[1 - current]
            return DuelResult(winner_id=other, loser_id=pid, reason="timeout",
                               turns_log=turns_log, clocks_remaining=clocks, questions_seen=seen)

        if total_seen >= QUESTION_CAP:
            other = turn_order[1 - current]
            if clocks[pid] >= clocks[other]:
                winner, loser = pid, other
            else:
                winner, loser = other, pid
            return DuelResult(winner_id=winner, loser_id=loser, reason="question_cap",
                               turns_log=turns_log, clocks_remaining=clocks, questions_seen=seen)

        if attempt.correct:
            current = 1 - current  # turn passes to the other player
            miss_streak = 0
            question = _draw_question(domain, used_questions, rng)
            distractors = _pick_distractors(domain, question, rng)
        elif attempt.outcome == "passed":
            # Deliberate skip: same player continues, but they get a new
            # image, exactly what passing is for.
            miss_streak += 1
            question = _draw_question(domain, used_questions, rng)
            distractors = _pick_distractors(domain, question, rng)
        else:
            # Plain wrong answer: same player, same image, another crack
            # at the thing they just missed.
            miss_streak += 1
