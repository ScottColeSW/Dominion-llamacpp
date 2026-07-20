"""Agent interface. Phase 1 uses a scripted heuristic stand-in (Section 11:
"fast model for players" comes later); this keeps the engine runnable and
free today, and gives real Claude-backed agents a drop-in interface to
replace later without touching the engine itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Optional

from .content import Question, Domain
from .models import Player, GameState


@dataclass
class AnswerAttempt:
    outcome: str        # "correct" | "incorrect" | "passed"
    correct: bool        # convenience flag, True iff outcome == "correct"
    seconds_used: float
    guess: str           # the stated answer; empty when passed
    # True only for a genuine live Ollama reply (OllamaAgent.attempt_question's
    # success path) -- always False here, since a scripted attempt's
    # seconds_used is fake/simulated time, not time that actually elapsed.
    # The frontend uses this to tell the two apart: a live seconds_used is
    # real wall-clock time that already passed server-side before this
    # event ever streamed out, so re-animating it client-side would double
    # the real wait rather than just replaying a simulated one.
    live: bool = False


class ScriptedAgent:
    """A simple, self-contained stand-in for an LLM-backed player.

    Accuracy, passing, and answer speed are randomized within plausible
    bounds so duels have real variance without needing any external model
    calls. When wrong, the guess is a real answer pulled from elsewhere in
    the same domain, a plausible mix-up rather than a blank flag, so the
    audience has something concrete to compare against the image.

    Passing is modeled separately from a wrong guess and gets more likely
    the longer a player's own turn drags on without a correct answer,
    miss_streak counts consecutive incorrect-or-passed attempts within the
    current turn. A frustrated player bailing on an image they can't place
    is exactly the kind of beat that plays well on screen.
    """

    def __init__(self, rng: random.Random, base_accuracy: float = 0.62,
                 base_pass_chance: float = 0.10):
        self.rng = rng
        self.base_accuracy = base_accuracy
        self.base_pass_chance = base_pass_chance

    def choose_target(self, player: Player, game: GameState) -> Optional[int]:
        options = game.adjacent_opponents(player.id)
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        # Temperament shapes WHO gets challenged, not just whether to keep
        # going: aggressive players (temperament -> 1.0) lean toward
        # whoever holds the most territory adjacent to them -- the bigger
        # prize, and a bigger flex -- while cautious players (-> 0.0) lean
        # toward the smallest, safest adjacent target instead. A neutral
        # 0.5 temperament cancels the exponent to 0, giving plain uniform
        # weights -- i.e. the exact old random-choice behavior.
        exponent = (player.temperament - 0.5) * 4  # -2..+2
        weights = [max(0.05, len(game.players[pid].territory) ** exponent) for pid in options]
        return self.rng.choices(options, weights=weights, k=1)[0]

    def attempt_question(self, player: Player, question: Question, domain: Domain,
                          miss_streak: int = 0, distractors: Optional[List[str]] = None,
                          time_remaining: Optional[float] = None) -> AnswerAttempt:
        # distractors/time_remaining are accepted but unused here --
        # ScriptedAgent already "cheats" via question.answer directly, and
        # is never slow enough for time_remaining to matter. They exist on
        # this signature so the shared call site (duel.py) can pass them
        # uniformly to any agent; OllamaAgent (ollama_agent.py) is the one
        # that actually uses them -- distractors as real multiple-choice
        # options for a live model that must NOT be handed the answer, and
        # time_remaining to cap how long it'll wait for a reply.
        # Revision 25: the frontend now animates each turn in genuine real
        # time (one real second per simulated second, per Scott's explicit
        # request), so how long a turn *feels* to watch is a direct,
        # literal function of these ranges, not just a simulation detail
        # anymore. The original 1.0-4.0s (pass) / 2.0-7.5s (answer) ranges
        # read as sluggish once played back honestly. Tightened here so
        # real-time playback stays snappy while keeping real variance, a
        # player still visibly "thinks" for a beat, it's just a shorter one.
        # Temperament nudges pass_chance without touching base_accuracy --
        # aggressive players are less willing to bail on a guess (temperament
        # 1.0 -> -0.06), cautious players pass a bit more readily (0.0 ->
        # +0.06); this is a style difference, not a skill difference.
        temperament_adjust = (0.5 - player.temperament) * 0.12
        pass_chance = min(0.45, max(0.02, self.base_pass_chance + temperament_adjust + 0.07 * miss_streak))
        if self.rng.random() < pass_chance:
            seconds = round(self.rng.uniform(0.6, 2.2), 1)
            return AnswerAttempt(outcome="passed", correct=False, seconds_used=seconds, guess="")

        seconds = round(self.rng.uniform(1.2, 4.5), 1)
        correct = self.rng.random() < self.base_accuracy
        if correct:
            return AnswerAttempt(outcome="correct", correct=True, seconds_used=seconds, guess=question.answer)

        others = [q.answer for q in domain.questions if q.answer != question.answer]
        guess = self.rng.choice(others) if others else question.answer
        return AnswerAttempt(outcome="incorrect", correct=False, seconds_used=seconds, guess=guess)

    def decide_continue(self, player: Player, game: GameState) -> bool:
        # More willing to keep pushing early, more cautious as the streak
        # builds, since more is on the line the longer a chain runs. The
        # base rate and the decay-per-win are both temperament-scaled:
        # aggressive players (-> 1.0) start higher and decay slower, so a
        # real, visible "won't back down" streak is possible; cautious
        # players (-> 0.0) start lower and back off faster on each win.
        # temperament 0.5 reproduces the old fixed 0.75 base / 0.15 decay
        # exactly, so a Balanced player's behavior is unchanged.
        if not game.adjacent_opponents(player.id):
            return False
        base_rate = 0.65 + 0.20 * player.temperament       # 0.5 -> 0.75
        decay_rate = 0.23 - 0.16 * player.temperament      # 0.5 -> 0.15
        floor = 0.15 + 0.20 * player.temperament           # 0.5 -> 0.25
        continue_chance = max(floor, base_rate - decay_rate * player.push_streak)
        # A banked time bonus only ever pays off once it's actually spent on
        # a future duel, which needs this player to be the spotlighted
        # challenger again -- retreating with it unused just leaves it
        # sitting idle for who knows how long. Bias toward pushing on
        # instead of retreating while holding one unspent.
        if player.time_bonus_banked:
            continue_chance = min(0.95, continue_chance + 0.25)
        return self.rng.random() < continue_chance

    def choose_tax_target(self, player: Player, game: GameState) -> Optional[int]:
        candidates = [pid for pid in game.active_ids if pid != player.id]
        return self.rng.choice(candidates) if candidates else None
