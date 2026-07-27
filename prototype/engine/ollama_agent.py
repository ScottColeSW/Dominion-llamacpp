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
import math
import os
import random
import re
import string
import time
import urllib.error
import urllib.request
from typing import List, Optional, Set, Tuple

from .agents import AnswerAttempt, ScriptedAgent
from .content import Question, Domain, same_category
from .models import Player, GameState, PROFESSION_DOMAIN_AFFINITY

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

# decide_continue's push/retreat announcement is a live, on-air moment the
# audience is actively watching and waiting on (Scott: "they have to decide
# in 15 seconds or less") -- nothing like intro_line's long leash before the
# timed clock even starts. A much tighter timeout than OLLAMA_TIMEOUT keeps
# the whole announcement beat (host build-up + this call + the reaction)
# comfortably inside that budget even in the worst case.
CONTINUE_DECISION_TIMEOUT = 6.0

# A floor under attempt_question's charged clock time, not an addition on
# top of real latency -- Scott's rule: no turn should read as having taken
# less than a beat, since a real contestant (however sharp) still takes a
# moment to read a question and answer, and an instant answer reads as
# implausible on screen regardless of how fast the underlying model is.
# Whole seconds only, always rounded UP: 0.6s charges a full 1, not 0.6 --
# see the max(MIN_CHARGED_SECONDS, math.ceil(...)) call site below.
#
# Raised from 1 to 3 -- Scott's report that "both players seem to
# incorrectly get a clock bonus after every turn... the clock speeds up a
# lot but the games go on forever." No code anywhere actually adds time
# back (confirmed: duel.py's clock only ever subtracts), so this wasn't a
# bonus bug -- it's this floor being too low relative to real Ollama
# round-trip latency. Measured live charged time averaged well under
# 0.1s/attempt after model-load overhead stopped counting against the
# clock, which the 1s floor bumped up to "1" regardless -- but a 25s clock
# still only drains by 1 real second per attempt while EACH attempt still
# costs a full live Ollama call's worth of actual wall-clock wait. With
# LIVE_QUESTION_CAP=60 combined attempts, that reads exactly as Scott
# described: the on-screen clock barely moves turn to turn (feels like
# free/bonus time) while the real-world duel drags on for many minutes
# before the large combined-attempt cap, not a genuine timeout, ends it.
# Charging 3s per attempt instead drains a 25s clock in well under a
# combined 60 attempts, so genuine timeout becomes the common case again
# and each duel needs meaningfully fewer real Ollama round trips to
# resolve either way.
MIN_CHARGED_SECONDS = 3


def _ask_ollama(model: str, prompt: str, timeout: float = OLLAMA_TIMEOUT,
                 num_predict: Optional[int] = None) -> "tuple[Optional[str], Optional[float]]":
    """Returns (reply_text, think_seconds). think_seconds is Ollama's own
    total_duration minus load_duration (both nanoseconds in the response) --
    i.e. actual generation time with model-loading overhead excluded -- or
    None if those fields are missing/malformed. A player whose model happens
    to already be resident vs. one that needs a cold load shouldn't get a
    different trivia-clock charge for the exact same quality of answer; see
    attempt_question, the only caller that uses this second value.

    num_predict, when given, caps how many tokens Ollama will generate
    before cutting the reply off outright -- Scott: interview replies "are
    not concise... I kinda expected that." Asking nicely in the prompt text
    ("ONE to TWO short sentences") was never going to reliably work on small
    local models, which follow length instructions loosely at best; nothing
    in this call was capping generation length structurally before this,
    since the request body never sent an `options` dict at all. This is the
    actual enforcement mechanism -- a real ceiling the model physically
    cannot talk past, not another polite request it can ignore.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}
    body = json.dumps(payload).encode("utf-8")
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


def _trim_to_last_sentence(text: str) -> str:
    """Cleans up a reply cut off mid-sentence by num_predict (see
    _ask_ollama above) -- a hard token ceiling stops generation the instant
    it's hit, with no regard for where a sentence actually ends, so a
    truncated reply can trail off mid-word ("...and honestly I think the")
    if left as-is. Backs up to the last '.', '!', or '?' found, if there is
    one; a reply that never reaches a full sentence at all (a very short
    cap, or a model that just doesn't use terminal punctuation) is returned
    unchanged rather than stripped down to nothing."""
    for punct in ".!?":
        idx = text.rfind(punct)
        if idx != -1:
            return text[:idx + 1]
    return text


# A short, sticky-note-length cap on the private note a player can leave
# for their own future turn (see Player.private_notes in models.py) -- this
# is meant to be one memorable line ("Player 4 is weak on Space and
# Planets"), not a second essay smuggled past num_predict/whatever length
# budget the actual decision reply gets.
NOTE_MAX_CHARS = 140


def _extract_note(reply: Optional[str]) -> "tuple[Optional[str], Optional[str]]":
    """Splits a raw model reply into (everything else, the private note or
    None). Looks for a line starting with MEMO: (case-insensitive) anywhere
    in the reply -- models asked for this will usually put it on its own
    line as instructed, but this doesn't assume a fixed line position, just
    presence -- and strips that line out of what it returns as the "main"
    text, so existing decision-parsing logic downstream (choose_target's
    _parse_index, decide_continue's PUSH/RETREAT split) never sees the note
    content mixed into the actual answer. Deliberately tagged MEMO rather
    than NOTE -- decide_continue's own prompt already uses the word "Note:"
    for an unrelated aside (the RETREAT-doesn't-mean-quitting clarification),
    and reusing the same word as this parser's marker would risk the model
    echoing that unrelated sentence back as if it were its private note.
    A reply with no MEMO: line at all (very common -- small local models
    don't reliably follow the two-line format) returns unchanged with a
    None note, so every existing call site's parsing behaves exactly as it
    did before this feature existed.
    """
    if not reply:
        return reply, None
    lines = reply.split("\n")
    kept = []
    note = None
    for line in lines:
        m = re.match(r"\s*MEMO\s*:\s*(.+)", line, re.IGNORECASE)
        if m and note is None:
            note = m.group(1).strip()[:NOTE_MAX_CHARS]
        else:
            kept.append(line)
    return "\n".join(kept).strip(), note


def _note_recall_line(player: Player) -> str:
    """A short "here's what you told yourself last time" prompt fragment --
    empty string if this player has never left a note yet (early in the
    show, or every past call happened to skip the MEMO: line), which is the
    common case and exactly why every prompt below only adds this when
    there's actually something to recall."""
    notes = player.recent_notes_text()
    return f" Private notes you left yourself earlier tonight: {notes}." if notes else ""


def _host_announcement_line(player: Player, opponent: Player, tested_domain: str) -> str:
    """A short, quoted line standing in for what the host actually just said
    live, on air -- Scott: "players do not seem aware they are in a game
    show. they never hear the host." That complaint is literally true of
    the engine, not just a vibe: index.html's hostPreduelQuestion generates
    the ACTUAL spoken host line, but purely in the browser, AFTER this
    Python call already ran -- nothing in this module ever had access to
    host language at all, live or otherwise, only bare facts (domain,
    opponent, streak) it turned into its own prompt wording. This doesn't
    try to match hostPreduelQuestion's exact phrasing (that would need
    moving host-line generation into Python entirely, a much bigger
    rebuild) -- it's a standalone quoted announcement built from the same
    underlying facts, good enough to give the model something concrete to
    have just "heard" and react to, right at the moment it's put on the
    spot -- which also covers Scott's "read the transcript... so they're
    ready for the spotlight pick" idea, since it lands at exactly that
    moment rather than earlier.
    """
    return (
        f' The host just said this live, in front of the crowd: "Time for our '
        f"next matchup! {player.kingdom_name} versus {opponent.kingdom_name}, "
        f'tested on {tested_domain}!" You heard that, live, same as everyone '
        f"watching."
    )


# Where every player's private MEMO note gets logged -- Scott: "people
# don't need to see it, except me, so a log or something." Deliberately a
# plain append-only JSON-lines file, not anything surfaced in the show's
# UI or spoken by the host (see _extract_note's docstring for how the
# MEMO: line is already stripped out of the actual on-air reply before
# this is ever reached) -- this is a private diagnostic trail, not a game
# feature, meant for Scott to open directly and for a future Claude
# session to grep/read when debugging agent behavior.
PLAYER_MEMO_LOG_PATH = os.path.join(os.path.dirname(__file__), "player_memos.log")


def _log_player_memo(player: Player, decision: str, note: str) -> None:
    """Appends one JSON object per line: timestamp, which player, which
    live decision produced the note (choose_target or decide_continue),
    and the note text itself. Best-effort and silent on any failure --
    same "never let an observability side-channel interrupt the actual
    show" pattern as HistoryRecorder.on_event in history.py; a disk
    hiccup writing a debugging log is not a reason to crash or stall a
    live duel."""
    try:
        with open(PLAYER_MEMO_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": time.time(),
                "player_id": player.id,
                "kingdom_name": player.kingdom_name,
                "model": player.model,
                "decision": decision,
                "note": note,
            }) + "\n")
    except OSError:
        pass


def _parse_index(reply: Optional[str], count: int) -> Optional[int]:
    if not reply or count <= 0:
        return None
    m = re.search(r"\d+", reply)
    if not m:
        return None
    idx = int(m.group())
    return idx if 0 <= idx < count else None


def _history_block(game: Optional[GameState]) -> str:
    """A short "SHOW SO FAR" prompt fragment built from GameState.memory
    (see models.py) -- Scott: "everyone, all agents, will be more and more
    informed as the game proceeds; like the players get smarter as they
    really play." Threaded into every live call EXCEPT attempt_question
    (see _domain_familiarity_line below for why that one gets a lighter,
    separate treatment) -- those calls happen once or twice per duel, not
    dozens of times, so the fuller context is cheap in aggregate. Empty
    string early in the show (nothing remembered yet), so those prompts
    stay exactly as short as they always were until there's actually
    something to report.
    """
    if not game:
        return ""
    text = game.recent_history_text()
    return f" What's happened in the show so far: {text}" if text else ""


def _domain_familiarity_line(game: Optional[GameState], player: Player, domain_name: str) -> str:
    """A one-line "you've already seen this domain tonight" fact for
    attempt_question's prompt specifically -- real accumulated in-show
    experience with the SPECIFIC subject being tested right now, which is
    the most literal reading of "the players get smarter as they really
    play." Kept separate from _history_block above (and never includes the
    full show history) since this prompt gets built on every single trivia
    attempt, the highest-frequency call site by far -- a full history
    block there would add real prompt weight (and latency) to every turn
    of every duel, not just once or twice per duel like the other calls.
    """
    if not game:
        return ""
    familiarity = game.domain_familiarity(player.id, domain_name)
    if not familiarity:
        return ""
    correct, total = familiarity
    return (
        f" You've already faced {total} question(s) in {domain_name} earlier tonight "
        f"({correct} correct) -- draw on that experience."
    )


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
            f"playing style: {player.temperament_label()}.{_note_recall_line(player)}"
            f"{_history_block(game)} "
            f"Pick ONE opponent to challenge next.\n"
            + "\n".join(lines)
            + "\nReply on TWO lines: line 1 is ONLY the number of your choice; line 2 "
              "starts with MEMO: followed by a short private strategic note to your "
              "future self for later in the show (not shown to the audience)."
        )
        reply, _ = _ask_ollama(self.model, prompt)
        # Strip any private note out BEFORE parsing the index -- otherwise a
        # note that happens to contain a digit could get picked up by
        # _parse_index's plain regex search instead of the actual chosen
        # number. See _extract_note's docstring for why a missing MEMO: line
        # (the common case on models that ignore the two-line format) leaves
        # this parsing identical to how it worked before this feature.
        reply, note = _extract_note(reply)
        if note:
            player.remember_note(note)
            _log_player_memo(player, "choose_target", note)
        idx = _parse_index(reply, len(options))
        return options[idx] if idx is not None else super().choose_target(player, game)

    def decide_continue(self, player: Player, game: GameState) -> Tuple[bool, str]:
        # Returns (keep_going, reason) now -- the host announces this choice
        # live (Scott: it "is being lost in the action," needs real drama),
        # and the reason is what gets said, not just a silent bool. One
        # combined call/parse, not two, to stay inside the 15-second budget
        # for the whole beat (see CONTINUE_DECISION_TIMEOUT above) -- a
        # separate follow-up call for "now explain yourself" would double
        # the real latency this fast-paced moment can afford.
        if not game.adjacent_opponents(player.id):
            return super().decide_continue(player, game)
        prompt = (
            f"You are {player.kingdom_name}, a {player.profession}, playing style: "
            f"{player.temperament_label()}. You are on a {player.push_streak}-win streak."
            f"{_note_recall_line(player)}{_history_block(game)} "
            f"Do you keep pushing for more territory, or retreat to defend what you have? "
            f"Note: RETREAT means staying in the game and defending your ground, "
            f"NOT quitting or retiring -- you are still an active player either way. "
            f"The host and the live audience are waiting on your answer right now -- "
            f"keep it quick.\n"
            f"Reply on TWO lines. Line 1 in EXACTLY this format: PUSH or RETREAT, then a "
            f"colon, then one short in-character reason that matches your verdict and does "
            f"not mention quitting or retiring. Line 2 starts with MEMO: followed by a "
            f"short private strategic note to your future self (not shown to viewers). "
            f"Example:\nPUSH: I can feel it, one more!\nMEMO: Player 4 folds under pressure, "
            f"target them next."
        )
        reply, _ = _ask_ollama(self.model, prompt, timeout=CONTINUE_DECISION_TIMEOUT)
        if reply:
            # Pull the private note out first -- decide_continue's reason is
            # SPOKEN on air (the host announces it live), so leaving a MEMO:
            # line inside `reply` unstripped would risk the private note
            # bleeding into what gets said out loud, since the reason below
            # is taken as everything after the verdict's first colon in
            # whatever text remains.
            reply, note = _extract_note(reply)
            if note:
                player.remember_note(note)
                _log_player_memo(player, "decide_continue", note)
            upper = reply.upper()
            verdict_part = upper.split(":", 1)[0]
            keep_going = None
            if "PUSH" in verdict_part:
                keep_going = True
            elif "RETREAT" in verdict_part:
                keep_going = False
            if keep_going is not None:
                reason = reply.split(":", 1)[1].strip() if ":" in reply else ""
                # A free-text reason can contradict its own verdict (Scott
                # reported one agent saying it would "stay and hold ground
                # and retire" in the same breath) -- "retire"/"quit"/"give
                # up" never accurately describe either PUSH or RETREAT here,
                # since the player stays active in the game regardless of
                # which choice they make. Fall back to the safe canned line
                # rather than air a self-contradicting sentence.
                contradiction_words = ("retire", "quit", "give up", "drop out", "resign")
                if not reason or any(w in reason.lower() for w in contradiction_words):
                    reason = "pushing on." if keep_going else "retreating to defend."
                return keep_going, reason
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
            f"opponent of your choice.{_history_block(game)}\n"
            + "\n".join(lines) + "\nReply with ONLY the number."
        )
        reply, _ = _ask_ollama(self.model, prompt)
        idx = _parse_index(reply, len(candidates))
        return candidates[idx] if idx is not None else super().choose_tax_target(player, game)

    def attempt_question(self, player: Player, question: Question, domain: Domain,
                          miss_streak: int = 0, distractors: Optional[List[str]] = None,
                          time_remaining: Optional[float] = None,
                          game: Optional[GameState] = None,
                          previously_wrong: Optional[Set[str]] = None) -> AnswerAttempt:
        options = list(distractors or [])
        if not options:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors, game=game,
                                             previously_wrong=previously_wrong)
        choices = options + [question.answer]
        self.rng.shuffle(choices)
        # The actual alphabet, not a project-invented cap: a hardcoded
        # "ABCD" (4 letters) crashed with IndexError the moment
        # distractors grew past 3 (duel.py's _pick_distractors is at 6 now,
        # for the blurt-repeat fix, so choices can reach 7) -- string
        # slicing doesn't pad or error, so letters silently stayed 4
        # characters long past that point and letters[i] blew up three
        # lines later. Replacing one hardcoded number with a bigger one
        # (10) just moves the same bug further out, not closes it --
        # string.ascii_uppercase (26) is the real ceiling for "a single
        # letter," not a number anyone chose for this game, so there's
        # nothing project-specific left to grow out of sync. The assert
        # makes it impossible to repeat this class of bug silently: if
        # choices ever somehow exceeds 26, this fails loudly right here
        # with a clear message, not three lines down as a bare IndexError.
        letters = string.ascii_uppercase[:len(choices)]
        assert len(letters) == len(choices), (
            f"attempt_question has {len(choices)} choices but the alphabet "
            f"only has {len(string.ascii_uppercase)} letters to label them with."
        )
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
        # Scott: "agents are repeating guesses." A live model has no memory
        # between calls, so without this it can easily re-pick a letter it
        # already tried and got wrong on this exact question -- naming the
        # ruled-out options explicitly instead of just raising miss_streak
        # generically (pass_hint above) gives it something concrete to
        # actually avoid.
        wrong_hint = (
            f" Already ruled out and wrong on this exact question: {', '.join(sorted(previously_wrong))}."
            if previously_wrong else ""
        )
        prompt = (
            f"Trivia category: {domain.name}\nClue: {question.image_prompt}\n"
            + "\n".join(lines)
            + "\nReply with ONLY the single letter of your answer, or PASS."
            + pass_hint
            + wrong_hint
            + _domain_familiarity_line(game, player, domain.name)
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
        # Whole seconds only, always rounded UP -- Scott's rule: 0.6s of
        # real thinking time still costs a full second off the clock, never
        # a fraction of one. Measured live, small local models were
        # answering in well under a tenth of a second, which is why the
        # question cap kept needing readjustment (README/duel.py docstring):
        # the clock barely moved no matter how many attempts happened, so
        # the cap -- not the clock -- ended up deciding almost every duel.
        # Rounding up is a floor, not an addition -- a genuinely slow call
        # still gets charged its real time, ceiled, never truncated down.
        charged_seconds = max(MIN_CHARGED_SECONDS, math.ceil(raw_seconds))
        if not reply:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors, game=game,
                                             previously_wrong=previously_wrong)
        upper = reply.strip().upper()
        if upper.startswith("PASS"):
            return AnswerAttempt(outcome="passed", correct=False, seconds_used=charged_seconds,
                                  guess="", live=True)
        # Matched against the SAME letters actually offered this turn, not
        # a hardcoded [A-D] -- that would have silently mis-parsed (or
        # never matched) any reply lettered E or beyond once choices could
        # run past 4.
        m = re.search(f"[{letters}]", upper)
        idx = letters.index(m.group()) if (m and m.group() in letters) else None
        if idx is None:
            return super().attempt_question(player, question, domain, miss_streak=miss_streak,
                                             distractors=distractors, game=game,
                                             previously_wrong=previously_wrong)
        chosen = choices[idx]
        correct = (chosen == question.answer)
        return AnswerAttempt(outcome="correct" if correct else "incorrect", correct=correct,
                              seconds_used=charged_seconds, guess=chosen, live=True)

    def intro_line_combined(self, player: Player, tested_domain: str, opponent: Player,
                             opponent_line: Optional[str] = None,
                             game: Optional[GameState] = None) -> str:
        # The single pre-duel interview call, called for BOTH sides right as
        # a duel opens, before run_duel starts. Replaces the old two-call
        # origin/challenge sequence (later three, with intro_line_opponent)
        # -- Scott found three separate live calls read like disconnected
        # exchanges rather than the model actually answering one real
        # question: "it doesn't seem like the agents are hearing each other
        # in conversation... player 1, I see you are a cautious expert in
        # flowers and gardens, how do you feel about competing with player 2
        # on the subject of Farm Animals. Like that kinda." One call now,
        # answering exactly that shape of loaded question -- own expertise,
        # tonight's actual domain, AND the named opponent, all at once (see
        # index.html's hostPreduelQuestion for the matching host question
        # this replies to). Still doubles as the fairness fix the old
        # rounds gave the defender (choose_target already warms the
        # CHALLENGER's model for free; this call gives the defender an
        # equivalent head start before the timed clock starts), just in one
        # call instead of two or three.
        # Revision 27 -- Scott: agents "aren't making rudimentary
        # connections like zoos domain and animals expertise and seeing
        # the advantage." PROFESSION_DOMAIN_AFFINITY (models.py) already
        # encodes exactly this kind of professional connection (e.g. a
        # veterinary assistant's list includes Zoo Animals, Farm Animals,
        # Cats, Dogs, Birds), but a live model was never told about it --
        # only origin_domain (the single fixed draft result) was ever
        # mentioned, so a genuinely relevant professional tie to tonight's
        # ACTUAL tested domain went completely unmentioned in the prompt,
        # and the model had no way to reason about or voice an edge that,
        # per this data, really does exist.
        # Revision 32 -- Scott's own examples of the reasoning he wants:
        # "I can see how my knowledge of farms could help me in oceans,
        # they both have a lot of animals," and "as a farmer, other
        # reptiles affect farms." Neither is a PROFESSION match -- nothing
        # ties "farmer" to Ocean Animals in PROFESSION_DOMAIN_AFFINITY --
        # it's a connection between the two DOMAINS themselves. same_category
        # (content.py) is the new, coarser layer for exactly that. Checked
        # only after the profession-affinity tier so a genuine professional
        # tie still wins when both apply.
        is_home_turf = player.origin_domain == tested_domain
        profession_affinity = PROFESSION_DOMAIN_AFFINITY.get(player.profession, ())
        is_professionally_related = (not is_home_turf) and tested_domain in profession_affinity
        category = None if (is_home_turf or is_professionally_related) else same_category(
            player.origin_domain, tested_domain)
        if is_home_turf:
            stakes = "this genuinely IS your home turf, the one subject you actually know."
        elif is_professionally_related:
            stakes = (
                f"not your official expertise ({player.origin_domain} is), but as a "
                f"{player.profession}, {tested_domain} is a natural professional fit for "
                f"you too -- you have a real, legitimate edge here even though it's not "
                f"your origin domain. Notice that connection and use it."
            )
        elif category:
            stakes = (
                f"not your official expertise ({player.origin_domain} is), and there's no "
                f"professional tie either -- but {player.origin_domain} and {tested_domain} "
                f"are both '{category}' at heart. Think concretely about what actually "
                f"carries over between them (specific facts, not just a vague feeling), "
                f"and say so if you find a real one."
            )
        else:
            stakes = (
                f"NOT your subject and has no real tie to your work as a "
                f"{player.profession} either. Outside {player.origin_domain}, you're no "
                f"more informed than an average person off the street; a real contestant "
                f"here wouldn't fake confidence they don't have."
            )
        streak_note = (
            f" You're riding a {player.push_streak}-duel win streak tonight."
            if player.push_streak >= 2 else ""
        )
        opponent_reaction = (
            f' They already said something about facing you, live, on air: '
            f'"{opponent_line}" React to that directly if you want -- agree, '
            f"brush it off, fire back, whatever fits your style."
            if opponent_line else
            ""
        )
        # Rebuilt as explicit labeled fields, Revision 22 -- Scott: "the
        # players seem a bit confused about the difference between their own
        # expertise, own name, domain challenged on, domain they hold and the
        # other players info for the same." The old version ran all of this
        # together as one flowing paragraph (name, profession, temperament,
        # origin_domain, tested_domain, opponent's name/profession/
        # territory, all in the same breath) with nothing to visually anchor
        # which fact was which -- an easy thing for a small local model to
        # blur together, especially "your real subject" (origin_domain,
        # fixed identity) vs. "tonight's domain" (tested_domain, always the
        # DEFENDER's currently-held land, changes every duel) vs. this
        # player's own CURRENTLY HELD domain (player.domain -- can differ
        # from origin_domain the moment they've ever won as a challenger and
        # inherited someone else's land; the old prompt never even mentioned
        # this one at all). Also now tells each player the OPPONENT's real
        # expertise, not just their name/profession/territory count -- the
        # old prompt gave full detail about yourself but left "the other
        # player's info for the same" thing incomplete, which is exactly the
        # asymmetry Scott flagged.
        # Revision 33 -- Scott: "some agents think they are NOT the expert
        # they are." This universal line used to ALWAYS describe origin_domain
        # with an "amateur, not a world-class expert" hedge, even on a home-turf
        # night where the stakes sentence right below it says the opposite
        # ("this genuinely IS your home turf"). A small model reading the
        # negation ("not... expert") right next to its own identity, one
        # sentence before being told it IS the expert, is exactly the kind
        # of contradiction that produces "I'm not really an expert" replies
        # on nights it actually should be confident. The hedge is still
        # real and worth keeping -- it's what stops a player from acting
        # like a world authority on a domain they only casually enjoy -- but
        # it only belongs in the sentence when tonight ISN'T home turf; on
        # a home-turf night this describes their expertise cleanly instead.
        expertise_clause = (
            f"you genuinely know this domain better than an average person off the "
            f"street -- this is your real, legitimate area of knowledge, not a guess."
            if is_home_turf else
            f"you're a genuine enthusiast there, an eager amateur, not a world-class "
            f"expert, but it's still the one thing you actually know personally."
        )
        facts = (
            f"YOUR NAME: {player.kingdom_name} ({player.profession}, playing style "
            f"{player.temperament_label()})\n"
            f"YOUR FIXED REAL-LIFE EXPERTISE (never changes): {player.origin_domain} -- "
            f"{expertise_clause}\n"
            f"LAND YOU CURRENTLY HOLD ON THE BOARD: {player.domain}"
            + (" (this happens to BE your real expertise)" if player.domain == player.origin_domain
               else " (NOT your real expertise -- you took this territory over from someone else)")
            + f"\nDOMAIN BEING TESTED IN TONIGHT'S DUEL: {tested_domain} -- {stakes}\n"
            f"YOUR OPPONENT TONIGHT: {opponent.kingdom_name} ({opponent.profession}, "
            f"playing style {opponent.temperament_label()}), whose own fixed real-life "
            f"expertise is {opponent.origin_domain} and who currently controls "
            f"{len(opponent.territory)} tile(s) of the board."
        )
        # Revision 27 -- Scott: agents "don't seem to be getting their role
        # in all this quickly/one-shot." Revision 22 already fixed the
        # WITHIN-facts confusion (own expertise vs. own name vs. tested
        # domain vs. held domain vs. opponent's info) by labeling each
        # field explicitly, but nothing oriented the model on what's about
        # to happen and what it's supposed to DO before dropping it into
        # that whole block of facts -- a plain one-line scene-setter up
        # front, before the dense part, gives a small local model a
        # concrete frame to read the rest of the facts INTO instead of
        # discovering the actual task only after parsing all of them.
        # Revision 33 -- Scott: "players do not seem aware they are in a
        # game show. they never hear the host." See _host_announcement_line's
        # own docstring for why that's an accurate gap, not just a feel
        # issue -- this is the fix, giving the model an actual quoted line
        # it just "heard" right before being put on the spot.
        role_header = (
            "You are a live contestant on a TV trivia game show, in front of a real "
            "studio audience.\n"
            f"{_host_announcement_line(player, opponent, tested_domain)}\n"
            "The host is now about to put you on the spot with one direct question, "
            "live, on air. Read your facts below, then answer AS that contestant.\n"
        )
        # Revision 34 -- Scott caught a live example: "player 2 thought they
        # had player 1's expertise." The two expertise fields ARE already
        # clearly labeled ("YOUR FIXED REAL-LIFE EXPERTISE" vs "YOUR
        # OPPONENT TONIGHT... whose own fixed real-life expertise is"), but
        # they're both just domain names sitting a few lines apart in a
        # fairly dense block, and this prompt has only gotten denser since
        # (the host announcement, the stakes/category reasoning) -- an easy
        # pair for a small model to blur together, especially the further
        # apart they land from the actual answer. This reminder repeats
        # both names explicitly, by OWNER, as the very last thing before the
        # model has to answer -- deliberately redundant with the facts
        # block above it, since what a small model reads LAST tends to
        # carry more weight than what it read several lines earlier.
        identity_reminder = (
            f"Quick reminder before you answer: YOUR expertise is "
            f"{player.origin_domain} -- that one is yours, no one else's. "
            f"{opponent.kingdom_name}'s expertise is {opponent.origin_domain} -- "
            f"that one belongs to THEM, not you. Do not mix the two up."
        )
        prompt = (
            f"{role_header}{facts}{streak_note}{opponent_reaction}"
            f"{_history_block(game)}\n{identity_reminder}\n"
            f"The host just asked you all of this in ONE breath, live, on air: your "
            f"real expertise, tonight's actual domain, and your read on this specific "
            f"opponent. Reply in ONE to TWO short, in-character sentences that actually "
            f"answer what was just asked -- like a real contestant caught by a TV camera "
            f"responding to exactly that question, not a narrator describing the scene. "
            f"Speak AS YOURSELF, first person -- use 'I'/'me'/'my', e.g. 'I built my name "
            f"on...', never 'they built their name on...' or your own name in the third "
            f"person."
        )
        # num_predict=55 is the real enforcement for "ONE to TWO short
        # sentences" above -- Scott: interviews "are not concise... I kinda
        # expected that." ~55 tokens is roughly 35-45 words, generous enough
        # for two real sentences with a kingdom name in them, tight enough
        # that a model that ignores the instruction gets cut off rather than
        # running on.
        reply, _ = _ask_ollama(self.model, prompt, num_predict=55)
        return _trim_to_last_sentence(reply.strip()) if reply else super().intro_line_combined(
            player, tested_domain, opponent, opponent_line, game=game)
