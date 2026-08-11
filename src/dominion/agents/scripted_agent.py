"""Agent interface. Phase 1 uses a scripted heuristic stand-in (Section 11:
"fast model for players" comes later); this keeps the engine runnable and
free today, and gives real Claude-backed agents a drop-in interface to
replace later without touching the engine itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import List, Optional, Set, Tuple

from ..engine.content import Question, Domain, same_category
from ..engine.models import Player, GameState, PROFESSION_DOMAIN_AFFINITY


@dataclass
class AnswerAttempt:
    outcome: str        # "correct" | "incorrect" | "passed"
    correct: bool        # convenience flag, True iff outcome == "correct"
    # Always a whole number of seconds, always rounded UP (never down and
    # never fractional) -- Scott's rule: a turn that took 0.6s of real or
    # simulated time still costs a full second off the clock, not a
    # fraction of one. A player who misses, misses, passes, misses, then
    # finally hits can rack up a genuinely large integer here across
    # several attempts on the same question; see duel.py's per-attempt loop.
    seconds_used: int
    guess: str           # the stated answer; empty when passed
    # True only for a genuine live Ollama reply (LLMAgent.attempt_question's
    # success path) -- always False here, since a scripted attempt's
    # seconds_used is fake/simulated time, not time that actually elapsed.
    # The frontend uses this to tell the two apart: a live seconds_used is
    # real wall-clock time that already passed server-side before this
    # event ever streamed out, so re-animating it client-side would double
    # the real wait rather than just replaying a simulated one.
    live: bool = False
    # Real wall-clock seconds the backend call actually took (InferenceClient's
    # GenerationResult.raw_latency_seconds), distinct from seconds_used above
    # (the charged, whole-second, MIN_CHARGED_SECONDS-floored clock cost) --
    # this is the genuine, uncharged latency number the backend comparison
    # (Standings page) needs. None for a scripted attempt (no real backend
    # call happened) or a lucky blurt/forced pass (duel.py never calls the
    # agent for either), same convention as `live` above.
    raw_latency_seconds: Optional[float] = None


def _domain_edge(player: Player, domain_name: str) -> int:
    """A coarse, ranked score for how informed `player` actually is on
    `domain_name` -- same three-tier read intro_line_combined already
    uses to talk about a matchup (home turf > professional/category tie
    > no real connection), reused here for the 2-player endgame's
    push/retreat choice (see decide_continue) rather than duplicating
    a separate ad-hoc heuristic."""
    if player.origin_domain == domain_name:
        return 2
    profession_affinity = PROFESSION_DOMAIN_AFFINITY.get(player.profession, ())
    if domain_name in profession_affinity:
        return 1
    if same_category(player.origin_domain, domain_name):
        return 1
    return 0


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

    async def choose_target(self, player: Player, game: GameState) -> Optional[int]:
        # async for interface uniformity with LLMAgent (agents/base.py) --
        # this body is pure/synchronous and never actually awaits anything.
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

    async def attempt_question(self, player: Player, question: Question, domain: Domain,
                                miss_streak: int = 0, distractors: Optional[List[str]] = None,
                                time_remaining: Optional[float] = None,
                                game: Optional[GameState] = None,
                                previously_wrong: Optional[Set[str]] = None) -> AnswerAttempt:
        # distractors/time_remaining/game are accepted but unused here --
        # ScriptedAgent already "cheats" via question.answer directly, and
        # is never slow enough for time_remaining to matter. They exist on
        # this signature so the shared call site (duel.py) can pass them
        # uniformly to any agent; LLMAgent (ollama_agent.py) is the one
        # that actually uses them -- distractors as real multiple-choice
        # options for a live model that must NOT be handed the answer, and
        # time_remaining to cap how long it'll wait for a reply.
        # Revision 25: the frontend now animates each turn in genuine real
        # time (one real second per simulated second, per Scott's explicit
        # request), so how long a turn *feels* to watch is a direct,
        # literal function of seconds_used below, not just a simulation
        # detail anymore -- see the blurt-count comment further down for
        # how that value is actually derived now.
        # Temperament nudges pass_chance without touching base_accuracy --
        # aggressive players are less willing to bail on a guess (temperament
        # 1.0 -> -0.06), cautious players pass a bit more readily (0.0 ->
        # +0.06); this is a style difference, not a skill difference.
        temperament_adjust = (0.5 - player.temperament) * 0.12
        # A pass costs the SAME player's own clock just like any other
        # attempt (duel.py: "wrong answers and passes keep the same
        # player's turn going and keep draining their own clock") -- it
        # does NOT "save the clock," a framing Scott flagged as backwards.
        # The only real upside to passing is dodging a question you don't
        # know before burning several MORE attempts guessing wrong at that
        # same image, and that tradeoff only makes sense with time to
        # spare: a player already low on their own clock can't afford to
        # gamble on a fresh unknown question when they might not know
        # THAT one either, so time_remaining now genuinely shapes
        # pass_chance instead of only temperament/miss_streak deciding it.
        time_adjust = 0.0
        if time_remaining is not None:
            if time_remaining < 8:
                time_adjust = -0.15  # can't afford to gamble on an unknown fresh question
            elif time_remaining > 15:
                time_adjust = 0.08   # comfortable enough to bail on a bad one
        pass_chance = min(0.45, max(0.02, self.base_pass_chance + temperament_adjust
                                     + time_adjust + 0.07 * miss_streak))
        # Charged time is driven by a blurt count, not a flat random roll --
        # Scott's ask: "add .5 seconds per blurt, so 3 blurts is 2 seconds,
        # etc." (ceil(3 * 0.5) == 2), so the frontend's rapid-fire guess
        # animation (web/index.html's startRapidFireGuesses) actually means
        # something instead of being pure decoration disconnected from the
        # clock. This is also the fix for duels reading as "tit for tat":
        # a flat range gave every turn roughly the same shape regardless of
        # what's actually happening in the duel.
        if self.rng.random() < pass_chance:
            # Giving up is quick -- 1 to 3 rapid tries before bailing.
            blurts = self.rng.randint(1, 3)
            seconds = max(1, math.ceil(blurts * 0.5))
            return AnswerAttempt(outcome="passed", correct=False, seconds_used=seconds, guess="")

        # More blurts possible the longer this exact question has already
        # gone unanswered (miss_streak): a player who's already missed a
        # few times on this image visibly hesitates and second-guesses more
        # on their next try, a real escalating reason to take longer, not a
        # uniform roll every time regardless of history.
        blurts = self.rng.randint(1, min(7, 3 + miss_streak))
        seconds = max(1, math.ceil(blurts * 0.5))
        correct = self.rng.random() < self.base_accuracy
        if correct:
            return AnswerAttempt(outcome="correct", correct=True, seconds_used=seconds, guess=question.answer)

        others = [q.answer for q in domain.questions if q.answer != question.answer]
        # Steer away from a wrong answer already given for this exact
        # question -- Scott: "agents are repeating guesses." Only actually
        # excludes anything if doing so still leaves a real option; a
        # small domain can genuinely run out of fresh wrong answers before
        # FORCED_PASS_MISS_STREAK kicks in, and repeating one then is
        # better than crashing on an empty choice list.
        if previously_wrong:
            fresh = [a for a in others if a not in previously_wrong]
            if fresh:
                others = fresh
        guess = self.rng.choice(others) if others else question.answer
        return AnswerAttempt(outcome="incorrect", correct=False, seconds_used=seconds, guess=guess)

    async def decide_continue(self, player: Player, game: GameState) -> Tuple[bool, str]:
        # More willing to keep pushing early, more cautious as the streak
        # builds, since more is on the line the longer a chain runs. The
        # base rate and the decay-per-win are both temperament-scaled:
        # aggressive players (-> 1.0) start higher and decay slower, so a
        # real, visible "won't back down" streak is possible; cautious
        # players (-> 0.0) start lower and back off faster on each win.
        # temperament 0.5 reproduces the old fixed 0.75 base / 0.15 decay
        # exactly, so a Balanced player's behavior is unchanged.
        #
        # Returns (keep_going, reason) now, not just a bool -- Scott's ask:
        # this choice was "being lost in the action," and the host needs a
        # real line to announce, not just a silent decision. Fallback text
        # here also covers LLMAgent whenever its live reply doesn't
        # parse cleanly, so it has to work standalone, not just as filler.
        if not game.adjacent_opponents(player.id):
            return False, "no one left in range to challenge. Holding position."
        # Scott: "players should not have the strategy of defense when
        # there are only 2 players left unless it is a strategic domain
        # knowledge play." With exactly two active players, RETREAT
        # doesn't actually avoid a duel -- it just hands the spotlight to
        # the only other player, who then has exactly one legal target
        # (this player) and becomes the challenger instead. Since the
        # tested domain is always the DEFENDER's own domain (game.py's
        # duel_result logic), what this choice actually controls is
        # WHICH domain gets tested next: PUSH means attacking the
        # opponent's currently held domain; RETREAT means defending your
        # own. That's a genuine domain-knowledge question, not generic
        # caution -- pure defense with no real edge behind it is exactly
        # the illogical fallback Scott flagged. Only overrides the
        # ordinary temperament roll below when one side's edge is
        # actually stronger; a genuine tie falls through unchanged.
        if len(game.active_ids) == 2:
            opponent_id = game.adjacent_opponents(player.id)[0]
            opponent = game.players[opponent_id]
            my_edge = _domain_edge(player, player.domain)
            their_edge = _domain_edge(player, opponent.domain)
            if their_edge > my_edge:
                return True, "I know their ground better than my own right now. I'm coming for it."
            if my_edge > their_edge:
                return False, "I'd rather they come to me -- this is my strongest ground."
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
        keep_going = self.rng.random() < continue_chance
        if keep_going:
            return True, "riding the momentum. Pushing on for more."
        return False, "content to hold this ground and let someone else take the spotlight."

    async def choose_tax_target(self, player: Player, game: GameState) -> Optional[int]:
        candidates = [pid for pid in game.active_ids if pid != player.id]
        return self.rng.choice(candidates) if candidates else None

    async def intro_line_combined(self, player: Player, tested_domain: str, opponent: Player,
                                   host_announcement: str,
                                   opponent_line: Optional[str] = None,
                                   game: Optional[GameState] = None) -> str:
        """The single pre-duel interview line -- replaces the old
        three-round origin/challenge/opponent sequence (Scott: three
        separate rounds "doesn't seem like the agents are hearing each
        other in conversation," reading as disconnected exchanges rather
        than one real answer; he asked for it condensed into one loaded
        question/answer). Covers, in one line: the player's real
        origin-domain standing (the ONE domain they actually built a name
        on, set once at the draft -- see models.py), an honest read on
        tonight's ACTUAL tested domain (only a genuine expert if it
        happens to match their origin domain -- everywhere else they're
        exactly as informed as anyone off the street, no better), and a
        temperament-flavored read on the specific opponent named.
        host_announcement/opponent_line/game accepted but unused here --
        a canned fallback can't meaningfully react to arbitrary host/
        opponent text or show history the way a live reply can (unlike
        LLMAgent, which is what host_announcement -- the real, single
        host line from agents/host_agent.py -- actually exists for).
        LLMAgent overrides this with a live model reply and falls back to
        this exact method on any failure, so it also has to work
        standalone.
        """
        # Revision 26 -- Scott: "agents might be better if they talk in
        # first person. our labels stay, but the words might be better."
        # The tell that this was actually written in third person all
        # along, not just deliberately elliptical: "Respects X, but still
        # THINKS this is winnable" and "FIGURES X is a real test" use
        # third-person verb conjugation (he/she thinks, figures) with the
        # subject dropped -- read aloud as this player's own live quote
        # (see index.html's interviewEl, which prefixes these with the
        # player's own name and a colon), that grammar mismatch is exactly
        # what makes a scripted line sound like a narrator describing the
        # player instead of the player actually talking.
        # Revision 27 -- Scott: agents "aren't making rudimentary
        # connections like zoos domain and animals expertise and seeing
        # the advantage." PROFESSION_DOMAIN_AFFINITY (models.py) already
        # tracks exactly this kind of professional connection -- e.g. a
        # veterinary assistant's affinity list includes Zoo Animals,
        # Farm Animals, Cats, Dogs, Birds -- but until now it was only
        # ever consulted once, at the draft, to weight which domain a
        # player's origin_domain gets drawn from; nothing surfaced it
        # again afterward, so a player whose ACTUAL tested domain tonight
        # happens to be a different-but-related one to their profession
        # had no way to say so, and read as generically clueless outside
        # their single fixed origin_domain instead of professionally
        # informed. This adds a real middle case between "exactly my home
        # turf" and "outside my wheelhouse."
        # Revision 32 -- Scott's own examples of the reasoning he actually
        # wants: "I can see how my knowledge of farms could help me in
        # oceans, they both have a lot of animals," and "as a farmer, other
        # reptiles affect farms." Neither is a PROFESSION match (nothing in
        # PROFESSION_DOMAIN_AFFINITY ties "farmer" to Ocean Animals) -- it's
        # a connection between the two DOMAINS themselves. same_category
        # (content.py) is the new, coarser layer for exactly that: two
        # different domains that happen to share a broad family (Farm
        # Animals / Ocean Animals: both "Animals"). Checked only after the
        # profession-affinity tier so a genuine professional tie still wins
        # when both apply.
        streak = f", riding a {player.push_streak}-win streak" if player.push_streak >= 2 else ""
        profession_affinity = PROFESSION_DOMAIN_AFFINITY.get(player.profession, ())
        category = same_category(player.origin_domain, tested_domain)
        if player.origin_domain == tested_domain:
            domain_take = (f"I built a name on {tested_domain}{streak}, and tonight's test "
                            f"is exactly my home turf.")
        elif tested_domain in profession_affinity:
            domain_take = (f"I built a name on {player.origin_domain}{streak}, but as a "
                            f"{player.profession}, {tested_domain} is right in my wheelhouse "
                            f"professionally too. I've got a real edge here.")
        elif category:
            domain_take = (f"I built a name on {player.origin_domain}{streak}. {tested_domain} "
                            f"isn't officially my thing, but they're both {category.lower()}, "
                            f"so I think some of that knowledge carries over.")
        else:
            domain_take = (f"I built a name on {player.origin_domain}{streak}, so tonight's "
                            f"{tested_domain} test is outside my wheelhouse. Just giving you "
                            f"my honest guess.")
        if player.temperament > 0.65:
            opponent_take = f"I'm not worried about {opponent.kingdom_name}. I've got this handled."
        elif player.temperament < 0.35:
            opponent_take = f"I respect {opponent.kingdom_name}, but I still think this is winnable."
        else:
            opponent_take = f"I figure {opponent.kingdom_name} is a real test, but I'm ready for it."
        return f"{domain_take} {opponent_take}"

    async def exit_interview(self, player: Player, winner: Player, tested_domain: str) -> str:
        """One short reaction from the just-eliminated player -- structurally
        the same treatment as intro_line_combined: a real fallback/standalone
        pool here, LLMAgent overrides with a live reply and falls back to
        this exact method on any failure. Right now elimination is silent
        past a template goodbye line (web/index.html's ELIMINATION_LINES);
        this is the player's own last word before that send-off plays."""
        if player.temperament > 0.65:
            templates = [
                f"{winner.kingdom_name} got the better of me tonight, but I'd do it again. "
                f"No regrets.",
                f"Credit to {winner.kingdom_name}. I went down swinging, and that's exactly "
                f"how I wanted it.",
            ]
        elif player.temperament < 0.35:
            templates = [
                f"{winner.kingdom_name} earned that one. I knew {tested_domain} wasn't my "
                f"strongest ground going in.",
                f"I gave it my honest best out there. {winner.kingdom_name} was just sharper "
                f"tonight.",
            ]
        else:
            templates = [
                f"That's the game. {winner.kingdom_name} took it fair and square on "
                f"{tested_domain}.",
                f"Tough way to go out, but {winner.kingdom_name} played it well. Good luck "
                f"to the rest of them.",
            ]
        return self.rng.choice(templates)
