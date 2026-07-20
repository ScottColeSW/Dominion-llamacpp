"""Core state model: players and the live game state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List

KINGDOM_NAME_PARTS_A = [
    "Whisker", "Meatloaf", "Drizzle", "Cobblestone", "Marmalade", "Thistle",
    "Foghorn", "Butterscotch", "Gravel", "Lantern", "Pickle", "Rustling",
    "Velvet", "Sawdust", "Moonlit",
]
KINGDOM_NAME_PARTS_B = [
    "Barony", "Dominion", "Hollow", "Reach", "Expanse", "Commons", "Fen",
    "Crossing", "Territory", "Enclave", "Bluff", "Thicket", "Province",
]

PROFESSIONS = [
    "claims adjuster", "night-shift security guard", "dental hygienist",
    "long-haul trucker", "barista", "HVAC technician", "school bus driver",
    "actuary", "landscaper", "pharmacy tech", "insurance underwriter",
    "line cook", "IT help desk analyst", "veterinary assistant",
    "warehouse supervisor",
]


@dataclass
class Player:
    id: int
    domain: str
    kingdom_name: str
    profession: str
    active: bool = True
    territory: Set[int] = field(default_factory=set)
    push_streak: int = 0
    time_bonus_banked: bool = False
    disposition: Dict[str, str] = field(default_factory=dict)  # domain -> "confident"/"uneasy"
    milestone_paid: bool = False  # whether the territory milestone bonus has been awarded yet
    # A fixed-for-the-run playing-style trait, 0.0 (most cautious) to 1.0
    # (most aggressive), assigned randomly at the draft and never changed.
    # Deliberately independent of base_accuracy/skill -- this is a style
    # difference (how boldly someone plays), not a talent difference.
    # Read directly off the Player by ScriptedAgent.choose_target,
    # decide_continue, and attempt_question to give some contestants real,
    # visible aggressive streaks and others a more laid-back run.
    temperament: float = 0.5
    # The local Ollama model backing this player's live decisions (target
    # choice, push/retreat, domain tax) -- see ollama_agent.py. Fixed for the
    # whole run, assigned at the draft, shown on the player's badge.
    model: str = ""
    # The domain this player actually drafted, set once at the draw and
    # never touched again -- domain (above) is CURRENT holdings, which
    # drifts constantly over a show via conquest and domain tax, so it's
    # not a stable enough anchor for a live model's own sense of identity.
    # ollama_agent.py's intro_line prompt uses this as a fixed "this is
    # where you started, this is who you are" reference point, separate
    # from whatever domain happens to be on the line tonight.
    origin_domain: str = ""

    def name_tag(self) -> str:
        return self.domain

    def temperament_label(self) -> str:
        if self.temperament < 0.35:
            return "Cautious"
        if self.temperament > 0.65:
            return "Aggressive"
        return "Balanced"


@dataclass
class GameState:
    players: Dict[int, Player]
    owner: Dict[int, int]                 # region id -> player id
    board_adj: Dict[int, Set[int]]        # region id -> set of adjacent region ids
    active_ids: Set[int]
    spotlight: Optional[int] = None
    excluded_from_pick: Optional[int] = None
    duel_count: int = 0
    scrambled: bool = False
    burst_prizes: List[dict] = field(default_factory=list)

    def sole_owner(self) -> Optional[int]:
        return next(iter(self.active_ids)) if len(self.active_ids) == 1 else None

    def players_adjacent(self, a: int, b: int) -> bool:
        a_regions = [r for r, o in self.owner.items() if o == a]
        b_regions = {r for r, o in self.owner.items() if o == b}
        return any(self.board_adj[r] & b_regions for r in a_regions)

    def adjacent_opponents(self, player_id: int) -> List[int]:
        return [pid for pid in self.active_ids if pid != player_id and self.players_adjacent(player_id, pid)]
