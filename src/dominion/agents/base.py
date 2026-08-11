"""The agent interface every player's decision-maker implements. All
methods are async: LLMAgent's are genuinely I/O-bound (a network call to
an InferenceClient backend), and ScriptedAgent's are trivially async
wrappers around its existing synchronous logic -- kept that way so
dominion.engine.game/duel can `await` every player uniformly without a
runtime type check to decide whether a particular call needs awaiting.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Set, Tuple

from ..engine.content import Domain, Question
from ..engine.models import GameState, Player
from .scripted_agent import AnswerAttempt


class Agent(Protocol):
    async def choose_target(self, player: Player, game: GameState) -> Optional[int]: ...

    async def decide_continue(self, player: Player, game: GameState) -> Tuple[bool, str]: ...

    async def choose_tax_target(self, player: Player, game: GameState) -> Optional[int]: ...

    async def attempt_question(
        self,
        player: Player,
        question: Question,
        domain: Domain,
        miss_streak: int = 0,
        distractors: Optional[List[str]] = None,
        time_remaining: Optional[float] = None,
        game: Optional[GameState] = None,
        previously_wrong: Optional[Set[str]] = None,
    ) -> AnswerAttempt: ...

    async def intro_line_combined(
        self,
        player: Player,
        tested_domain: str,
        opponent: Player,
        host_announcement: str,
        opponent_line: Optional[str] = None,
        game: Optional[GameState] = None,
    ) -> str: ...

    async def exit_interview(self, player: Player, winner: Player, tested_domain: str) -> str: ...


class HostAgent(Protocol):
    """The Host's interface -- ScriptedHostAgent/LLMHostAgent
    (host_agent.py). Deliberately separate from Agent above: the Host
    isn't a player (no domain, no elimination, no territory), just two
    announcement points for v1 -- see host_agent.py's module docstring."""

    async def announce_challenge(self, challenger: Player, defender: Player,
                                  tested_domain: str) -> str: ...

    async def announce_finale(self, champion: Player, total_duels: int, prize: int) -> str: ...


class CommentatorAgent(Protocol):
    """The Commentator's interface (commentator_agent.py) -- a second AI
    voice, distinct from the Host above: reacts to the show rather than
    running it, at a deliberately lower frequency (one per duel plus rare
    special events, never every duel_turn)."""

    async def react_to_matchup(self, challenger: Player, defender: Player,
                                tested_domain: str) -> str: ...

    async def react_to_advantage(self, player: Player, streak: int) -> str: ...

    async def react_to_scramble(self, active_players: int) -> str: ...
