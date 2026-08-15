"""Duel resolution: the chess-clock mechanic from Section 4 of the design
doc, implemented exactly to the corrected Revision 19 rules.

- Both players are tested on the DEFENDER's currently-held domain only --
  this is the territory actually at stake (only the LOSER's land ever
  transfers to the winner, and the defender is always whoever's land is
  being attacked), so the trivia subject has to be the defender's domain,
  full stop. Revision 20 briefly tried coin-flipping this between the
  challenger's and defender's own domain to address a real, measured
  structural edge, but Scott caught that it broke the actual conquest
  fiction: "it seems the challenges are taking place on the challenger's
  domain. that shouldn't be possible since they want to capture a different
  territory/domain." He's right -- testing the challenger's own land never
  made narrative sense regardless of the fairness math, so Revision 21
  reverted this rule back to exactly what it always was and moved the
  fairness fix to a compensating clock handicap (CHALLENGER_HOME_TURF_HANDICAP)
  instead. Revision 22 then dropped that handicap too -- after watching more
  shows with the new live-vs-fallback answer counter, Scott: "honestly, I
  don't think they need that home-field advantage." The structural edge
  documented in README.md is a real, known, accepted side effect of tying
  domain to territory (see the module comment there), not something this
  engine tries to correct for anymore.
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
- After FORCED_PASS_MISS_STREAK consecutive misses on one question, a pass
  is forced regardless of what the agent says. A live model's pass is
  entirely voluntary (only registers if its reply literally starts with
  "PASS"), so without this a model that never says that word could hammer
  the exact same image indefinitely -- confirmed live: a duel visibly stuck
  alternating between the same two distractor guesses, never passing, never
  reaching a new question.
- The only true elimination trigger is a clock reaching zero. The clock
  value recorded for display is clamped at 0: the real internal clock can
  dip slightly negative the instant it crosses zero (a slow final answer
  can burn more time than was left), but elimination and comparisons use
  the real value, only the number shown to the audience is clamped.
- The whole duel is capped at question_cap questions seen TOTAL, combined
  across both players (default 30, Revision 18, tightened from 25 seen per
  player i.e. up to 50 total, specifically to shorten duels faster). The
  instant that combined total hits the cap, the duel resolves immediately
  by comparing remaining time. That default assumes scripted-pace answers
  (roughly 1-4s each); with live Ollama answers -- especially once
  ollama_agent.py stopped charging model-load overhead against the clock --
  a single question can cost well under a second, so game.py raises this
  cap for live play (LIVE_QUESTION_CAP). It was first raised all the way to
  300 to try to make clock exhaustion the normal ending again, but measured
  live charged time turned out to average under 0.1s/attempt -- nowhere
  close to what would take to actually drain a 25s+ clock even at 300
  combined attempts, so raising the cap further only made real-world duels
  (which pay real Ollama round-trip latency per attempt, independent of how
  little clock time gets charged) take dramatically longer without fixing
  anything. Pulled back down to a smaller multiple of the scripted default
  instead -- see game.py for the current value and reasoning. The forced-
  pass rule above is what actually prevents a duel from feeling stuck; this
  cap is just the final backstop.
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
- on_before_attempt, if given, fires right before the real, potentially
  slow agent.attempt_question call (never for a forced pass or a lucky
  blurt, since neither one actually calls the agent) -- lets the caller
  surface a "this player's live model is thinking right now" indicator for
  the trivia-answering turns specifically. Previously the only
  agent_thinking events game.py ever emitted were for the pre-duel
  interview, target choice, push/retreat, and domain tax decisions -- the
  actual question-answering turns, which are most of a live duel's real
  wall-clock time, never got one at all, so nothing on screen (including
  the Player Roster's own thinking indicator) ever reflected them. Scott:
  "I'd expect it to be 'thinking' during the interviews and game
  challenge" -- this was the "game challenge" half that was missing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Set, Tuple
import random

from .content import Domain, Question
from ..agents.scripted_agent import AnswerAttempt, ScriptedAgent
from .models import Player, GameState


QUESTION_CAP = 30  # total across BOTH players in one duel, not per-player
# The single, authoritative definition of the per-player starting clock --
# every real duel gets its actual value from here (threaded through
# game.py's challenge_declared/show_start emits as base_clock, which the
# frontend always prefers over its own bootstrap default -- see
# web/index.html's BASE_CLOCK). Change gameplay pacing by editing this one
# number; nothing else needs to change to match.
#
# An int, not 30.0 -- every charge against this clock (attempt.seconds_used)
# is now a whole number by construction (Scott's rule), so keeping the
# clock itself an int end to end means clocks_remaining never drifts back
# into "4.0"-looking float output despite every individual charge already
# being a clean integer.
#
# Raised from 25 to 30 earlier (Scott, after seeing the wall-clock-accuracy
# fix's real numbers: ~18-20s per live turn) -- 25 gave a player almost no
# room to survive a single slow real call. Brought back down to 20 now that
# the retry-budget/circuit-breaker fixes (inference/retry.py,
# ollama_client.py/llamacpp_client.py) and the cleaned-up model roster have
# made real turns reliably fast again -- Scott: "nailed it! ... we can
# decrease the time for each duel from 30 second clocks to 20 second
# clocks now." A tighter clock reads as more urgent/exciting on screen as
# long as it isn't starving a genuine live call, which it no longer is.
BASE_CLOCK = 20
# Revision 21 added CHALLENGER_HOME_TURF_HANDICAP here (a flat clock head
# start for the challenger, compensating for domain_record/
# _domain_familiarity_line's real accumulated-experience edge for whoever
# currently holds a piece of land -- see README.md's home-turf section for
# the full mechanism and the measured 56.5%/43.5% split it produces).
# Revision 22 removed it again: after watching more shows with the new
# live-vs-fallback answer counter, Scott: "honestly, I don't think they need
# that home-field advantage." The structural edge is still real and still
# documented, it's just no longer something this engine tries to patch --
# an accepted, known asymmetry rather than a bug to compensate for.
# A live model's "pass" is entirely voluntary -- attempt_question only
# registers one if the model's reply literally starts with "PASS" (see
# ollama_agent.py). A model that never volunteers that word can otherwise
# hammer the exact same image forever (a wrong answer keeps the same
# question up for another try, by design), which reads to a viewer as the
# show being stuck, not just an unlucky streak. After this many consecutive
# misses on one question, force a pass regardless of what the agent says --
# ScriptedAgent's own escalating pass_chance already self-resolves well
# before this in practice, so it's a live-only safety net in effect.
FORCED_PASS_MISS_STREAK = 4
# A real contestant rattling off rapid-fire guesses (the frontend's "blurt"
# animation, web/index.html's startRapidFireGuesses) can genuinely land on
# the right answer by accident, before their actual considered answer would
# even come back -- Scott: "a blurt could be a right answer and counts if it
# is," an explicit real mechanic, not just a cosmetic flourish. Rolled once
# per attempt (same granularity as agent.attempt_question), and only when
# there's an actual blurt to land on (a question with distractors) -- a hit
# skips the real agent call entirely (live or scripted), since the blurt
# already got there first. Applies equally to both players every turn, so
# it doesn't touch the challenger/defender win-rate split (see history.py's
# get_stats and README's "home-turf" note) one way or the other.
LUCKY_BLURT_CHANCE = 0.06


@dataclass
class DuelResult:
    winner_id: int
    loser_id: int
    reason: str                      # "timeout" or "question_cap"
    turns_log: List[Dict[str, Any]] = field(default_factory=list)
    clocks_remaining: Dict[int, int] = field(default_factory=dict)
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
    cosmetic -- never includes the real answer itself.

    Capped at 6, not 3: the frontend shows each one at most once per
    player per turn now (never repeats within one player's own blurt
    burst, Scott's report), and a longer turn's blurt count can run past 3
    -- 6 gives enough unique material that a long, heavily-hesitant turn
    still doesn't have to repeat, while a short domain with fewer real
    answers just yields a shorter, still repeat-free burst.

    Deduplicated by VALUE, not just by question -- different questions in
    the same domain frequently share an answer (two different clues can
    both answer "elephant"), so a plain list comprehension here let
    rng.sample() return the same STRING twice at two different list
    positions, which is a real duplicate as far as the frontend blurt
    burst is concerned even though sample() itself never repeats a
    position. Order preserved (first occurrence in domain.questions, not
    a raw set) so this stays deterministic under a seeded rng -- only
    rng.sample's own selection introduces randomness, the pool it draws
    from is fixed given the same domain/question."""
    seen: Set[str] = set()
    pool: List[str] = []
    for q in domain.questions:
        if q.answer != question.answer and q.answer not in seen:
            seen.add(q.answer)
            pool.append(q.answer)
    return rng.sample(pool, k=min(6, len(pool))) if pool else []


async def run_duel(challenger: Player, defender: Player, domain: Domain,
                    agents: Dict[int, ScriptedAgent], rng: random.Random,
                    challenger_bonus: bool = False, defender_bonus: bool = False,
                    used_questions: Optional[Set[Tuple[str, str]]] = None,
                    base_clock: int = BASE_CLOCK, question_cap: int = QUESTION_CAP,
                    on_turn: Optional[Callable[[Dict[str, Any]], None]] = None,
                    on_before_attempt: Optional[Callable[[int, str], None]] = None,
                    on_question_drawn: Optional[Callable[[Dict[str, Any]], None]] = None,
                    game: Optional[GameState] = None) -> DuelResult:
    # game (added for the "agents get smarter as they play" feature) is
    # passed straight through to agent.attempt_question below so a live
    # LLMAgent can reference this player's accumulated domain_record for
    # the domain being tested right now -- ScriptedAgent's own
    # attempt_question just accepts and ignores it, same as it already does
    # with distractors/time_remaining.
    #
    # async because agent.attempt_question below is the hot, per-turn path
    # to a live model call (see agents/base.py) -- on_turn/on_before_attempt
    # stay plain sync callbacks (they only ever emit an event, no I/O).

    if used_questions is None:
        used_questions = set()

    clocks = {
        challenger.id: base_clock + (5 if challenger_bonus else 0),
        defender.id: base_clock + (5 if defender_bonus else 0),
    }
    seen = {challenger.id: 0, defender.id: 0}
    turns_log: List[Dict[str, Any]] = []

    turn_order = [challenger.id, defender.id]
    players_by_id = {challenger.id: challenger, defender.id: defender}
    current = 0  # index into turn_order; challenger acts first
    miss_streak = 0  # consecutive incorrect-or-passed attempts on this turn (across ANY question)
    # Scoped specifically to consecutive WRONG GUESSES on the CURRENT
    # question -- separate from miss_streak above, which was being used for
    # both purposes and let the two collide: FORCED_PASS_MISS_STREAK was
    # documented as "consecutive misses on ONE question" but miss_streak
    # itself never resets on a pass (only on a correct answer), so a player
    # who passed a few times in a row could hit the forced-pass threshold
    # and then never be offered a real attempt again -- attempt_question
    # stops getting called at all once forced, so there's no path back to a
    # correct answer to reset it, guaranteeing passes until the clock hits
    # zero. Confirmed live: exactly Scott's report of an agent "falling
    # into a pass loop and ending up losing." question_miss_streak resets
    # to 0 on every fresh question (a pass OR a correct answer both draw a
    # new one), so a run of voluntary passes can never itself trigger the
    # forced-pass safety net -- only genuinely repeated wrong guesses AT
    # THE SAME QUESTION can, which is what this was always meant to catch.
    question_miss_streak = 0
    # Wrong guesses already given for the CURRENT question by the CURRENT
    # player -- reset alongside question_miss_streak whenever a fresh
    # question comes up. Scott: "agents are repeating guesses." Neither
    # agent previously had any memory of what it had already tried and
    # gotten wrong on this exact question, so nothing stopped the same
    # wrong answer coming back around on a later attempt.
    previously_wrong: Set[str] = set()
    total_seen = 0  # combined attempts across BOTH players this duel
    # Scott: "the players need to know they got an answer right or wrong
    # too." This player's own most recent attempt ANYWHERE in this duel
    # (not just on the current question -- see attempt_question's own
    # wrong_hint for that narrower, same-question case), handed back to
    # them right before their next one so a live model gets told plainly
    # whether its last guess landed, not just left to infer it from
    # silence. Keyed by player id since the two players' own attempt
    # histories are independent of each other and of turn order.
    last_own_attempt: Dict[int, AnswerAttempt] = {}

    # Scott: "clocks are running before the question is out." Root cause:
    # the question/answer only ever reached the frontend bundled INSIDE
    # the completed duel_turn event -- the same event that reveals the
    # outcome -- so there was never a moment where the audience could see
    # what's being asked before the clock (which correctly starts the
    # instant a player begins thinking, see agent_thinking's own
    # startLiveClockTick call) had already been ticking for a while. This
    # draws and emits the question the instant it's actually decided,
    # BEFORE the next attempt (and its clock) ever starts -- called here
    # for the duel's first question, and again at both re-draw sites
    # below (a correct answer or a pass) so the frontend can reveal the
    # image right as (or fractionally before) the clock for the next
    # attempt begins, not after it's already been ticking.
    def _draw_and_announce() -> Tuple[Question, List[str]]:
        q = _draw_question(domain, used_questions, rng)
        d = _pick_distractors(domain, q, rng)
        if on_question_drawn is not None:
            on_question_drawn({"prompt": q.image_prompt, "answer": q.answer, "domain": domain.name})
        return q, d

    question, distractors = _draw_and_announce()

    while True:
        pid = turn_order[current]
        agent = agents[pid]
        player = players_by_id[pid]

        # Fired for every turn now, regardless of which branch below
        # actually decides it -- Scott: "these are like chess clocks...
        # the counters are not decreasing at least 1 tick per turn." The
        # forced-pass and lucky-blurt branches used to skip this entirely
        # (no live agent call to wait on, so it looked pointless to signal
        # "thinking"), which meant the frontend's liveTick never started
        # for those turns -- clockState just jumped straight from the old
        # value to the new one when duel_turn arrived, with no visible
        # countdown motion. Both branches still charge real seconds_used
        # against the clock, so the audience should still see that time
        # tick away like every other turn, not vanish in a jump.
        if on_before_attempt is not None:
            on_before_attempt(pid, player.model)

        lucky_blurt = False
        if question_miss_streak >= FORCED_PASS_MISS_STREAK:
            # The agent doesn't get asked at all this turn -- forced past
            # the point where a stubborn live model would otherwise keep
            # guessing the same image wrong forever. A flat time cost (a
            # quick on-air "I'll come back to that") rather than a real
            # Ollama call, since the point is to guarantee progress, not to
            # spend more real latency on the question we're skipping -- but
            # still at least a full second, same floor as every other turn
            # (Scott's rule: no turn reads as having taken less than a
            # beat, and every seconds_used is a whole number, never a
            # fraction).
            attempt = AnswerAttempt(outcome="passed", correct=False, seconds_used=1, guess="")
        elif distractors and rng.random() < LUCKY_BLURT_CHANCE:
            # The blurt itself got there first -- see LUCKY_BLURT_CHANCE.
            # Skips the real agent call entirely (no live Ollama round trip
            # spent on a question that's already been won by accident), and
            # charges a short, fast time -- a lucky blurt is a quick beat,
            # not a considered answer.
            lucky_blurt = True
            attempt = AnswerAttempt(outcome="correct", correct=True,
                                     seconds_used=rng.randint(1, 2), guess=question.answer)
        else:
            # time_remaining lets a live agent (LLMAgent) cap how long
            # it'll wait for a reply to roughly what's actually left on THIS
            # player's clock -- otherwise a slow/cold live call can run well
            # past the moment this player's clock would have hit zero, and
            # the duel visibly drags on past when the audience already
            # expects it to end. ScriptedAgent ignores this; it's never
            # slow enough to matter.
            attempt = await agent.attempt_question(
                player, question, domain, miss_streak=miss_streak,
                distractors=distractors, time_remaining=clocks[pid],
                game=game, previously_wrong=previously_wrong,
                last_own_attempt=last_own_attempt.get(pid))

        last_own_attempt[pid] = attempt
        # Scott: "the player 2 interview can include some answers given by
        # player 1." Set on the Player itself (see its own comment) so it
        # outlives this duel -- unlike last_own_attempt above, which is
        # local to this run_duel call and only ever feeds this same
        # player's own next attempt within THIS duel.
        if attempt.outcome == "passed":
            player.last_attempt_note = f"passed on a {domain.name} question"
        elif attempt.correct:
            player.last_attempt_note = f"got a {domain.name} question right (guessed '{attempt.guess}')"
        else:
            player.last_attempt_note = f"got a {domain.name} question wrong (guessed '{attempt.guess}')"
        clocks[pid] -= attempt.seconds_used
        seen[pid] += 1
        total_seen += 1
        turns_log.append({
            "player_id": pid, "domain": domain.name, "prompt": question.image_prompt,
            "answer": question.answer, "guess": attempt.guess, "outcome": attempt.outcome,
            "correct": attempt.correct, "seconds_used": attempt.seconds_used,
            "distractors": distractors, "live": attempt.live, "lucky_blurt": lucky_blurt,
            "raw_latency_seconds": attempt.raw_latency_seconds,
            # Clamped to 0 for display: the real clock can dip negative the
            # instant it crosses zero (an int charge subtracted from an int
            # clock, so still a clean negative int, e.g. -2), but the
            # audience should never see a negative number on screen.
            # Elimination logic below still uses the real, unclamped
            # clocks[pid] value. base_clock and every seconds_used charged
            # against it are ints by construction (Scott's rule), so this
            # was always a whole number -- max(0, ...) just clamps the
            # sign, nothing here rounds or truncates a fraction away.
            "clock_remaining": max(0, clocks[pid]),
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

        if total_seen >= question_cap:
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
            question_miss_streak = 0
            previously_wrong = set()
            question, distractors = _draw_and_announce()
        elif attempt.outcome == "passed":
            # Deliberate skip: same player continues, but they get a new
            # image, exactly what passing is for. question_miss_streak
            # resets here too (a fresh question means zero wrong guesses
            # against it so far) -- this is what stops a run of passes
            # from ever tripping the forced-pass safety net on its own;
            # only repeated wrong guesses AT ONE question can do that now.
            miss_streak += 1
            question_miss_streak = 0
            previously_wrong = set()
            question, distractors = _draw_and_announce()
        else:
            # Plain wrong answer: same player, same image, another crack
            # at the thing they just missed. Remember the guess itself so
            # attempt_question's next call for this exact question can
            # steer away from repeating it -- see previously_wrong above.
            miss_streak += 1
            question_miss_streak += 1
            if attempt.guess:
                previously_wrong.add(attempt.guess)
