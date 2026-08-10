"""Core state model: players and the live game state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Tuple

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

# A "great fit" domain shortlist per profession, used to weight the
# without-replacement origin_domain draw in game.py's draft loop (Scott:
# "it is very unlikely that a construction worker type is going to choose
# stamp collecting" -- origin_domain was previously a flat, uniform random
# pick across the whole 39-domain library, totally uncoupled from
# profession, which is exactly how that kind of nonsense pairing happened).
# Not an exhaustive or exclusive list -- any domain NOT listed here for a
# given profession still has a real (just much lower) chance of being that
# player's origin_domain, so the pairing stays a genuine draw, not a rigid
# lookup table. Domain names must match DOMAIN_LIBRARY entries in content.py
# exactly.
PROFESSION_DOMAIN_AFFINITY: Dict[str, List[str]] = {
    "claims adjuster": ["Cars and Trucks", "Weather", "Construction and Tools", "Firefighters"],
    "night-shift security guard": ["Fair Grounds", "Grocery Store", "Community Helpers",
                                    "Construction and Tools", "Space and Planets"],
    "dental hygienist": ["Ice Cream", "Bakery and Bread", "Pizza", "Community Helpers"],
    "long-haul trucker": ["Cars and Trucks", "Trains", "Weather", "Camping"],
    "barista": ["Cooking and Kitchen", "Bakery and Bread", "Ice Cream", "Grocery Store"],
    "HVAC technician": ["Winter and Snow", "Weather", "Construction and Tools", "Autumn"],
    "school bus driver": ["School", "Playgrounds", "Cars and Trucks", "Birthdays"],
    "actuary": ["Weather", "Space and Planets", "Board Games", "Cars and Trucks"],
    "landscaper": ["Flowers and Gardens", "Autumn", "Farm Animals", "Camping", "Bugs and Insects"],
    "pharmacy tech": ["Community Helpers", "Grocery Store", "Bakery and Bread"],
    "insurance underwriter": ["Cars and Trucks", "Weather", "Construction and Tools", "Firefighters"],
    "line cook": ["Cooking and Kitchen", "Pizza", "Bakery and Bread", "Grocery Store", "Ice Cream"],
    "IT help desk analyst": ["Space and Planets", "Board Games", "School"],
    "veterinary assistant": ["Cats", "Dogs", "Farm Animals", "Zoo Animals", "Birds",
                              "Reptiles and Amphibians", "Ocean Animals"],
    "warehouse supervisor": ["Grocery Store", "Construction and Tools", "Cars and Trucks", "Bicycles"],
}
# Weight multiplier for a profession's shortlisted domains vs. everything
# else in the pool -- high enough to make a sensible pairing likely, low
# enough that it's still a real draw (an early, well-matched domain can
# still get taken by an earlier pick in the draft order) rather than a
# guarantee.
PROFESSION_DOMAIN_WEIGHT = 6


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
    # This player's personal, amateur expertise -- deliberately independent
    # of domain (above), which is CURRENT board holdings and drifts
    # constantly over a show via conquest and domain tax, so it's not a
    # stable enough anchor for a live model's own sense of identity.
    # ollama_agent.py's intro_line prompt uses this as a fixed "this is
    # who you are" reference point, separate from whatever domain happens
    # to be on the line tonight. Drawn once at the draft from the FULL
    # domain library (not just the 13 seeded on this show's board), so
    # landing on a matching board domain is a real, rare coincidence rather
    # than guaranteed -- but the draw itself is weighted toward whatever
    # plausibly fits this player's profession (PROFESSION_DOMAIN_AFFINITY
    # above) and drawn without replacement across players, so no two
    # players ever share the same personal expertise, the same way no two
    # tiles ever share the same board domain.
    origin_domain: str = ""
    # Revision 29 -- Scott's "leave a note for your future self" idea: each
    # live Ollama call is a fresh, stateless request (see ollama_agent.py's
    # module docstring), so nothing about a player's OWN prior reasoning
    # survives to their next turn unless it's explicitly written down and
    # handed back to them -- like someone with anterograde amnesia relying
    # on a notebook. Deliberately separate from GameState.memory
    # (models.py -- shared across every player, engine-authored, and always
    # factual, e.g. "Player 3 beat Player 7"): this is PRIVATE to this one
    # player, and unlike the engine's log, it's the model's OWN claim about
    # its own strategy, which can be wrong, stale, or overconfident. Capped
    # short -- both in length per note (NOTE_MAX_CHARS in ollama_agent.py)
    # and in count here -- so it stays a sticky note a future turn can
    # glance at, not a growing, eventually-unreliable diary the model has to
    # re-read and could double down on being wrong about.
    private_notes: List[str] = field(default_factory=list)
    PRIVATE_NOTE_LIMIT = 2

    def remember_note(self, note: str) -> None:
        if not note:
            return
        self.private_notes.append(note)
        if len(self.private_notes) > self.PRIVATE_NOTE_LIMIT:
            self.private_notes = self.private_notes[-self.PRIVATE_NOTE_LIMIT:]

    def recent_notes_text(self) -> str:
        """Oldest first, same convention as GameState.recent_history_text --
        empty string if this player has never left themselves a note yet,
        which every call site checks before adding anything to a prompt."""
        return " / ".join(self.private_notes)

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
    # A short, bounded rolling log of plain-English facts about the show so
    # far -- duel outcomes, the Scramble -- threaded into every live
    # LLMAgent prompt that makes a strategic call or speaks in character
    # (choose_target, decide_continue, choose_tax_target, the intro_line_*
    # trio), so real models get progressively more informed about the state
    # of the board as the show goes on, not just about their own personal
    # history. Scott: "everyone, all agents, will be more and more informed
    # as the game proceeds; like the players get smarter as they really
    # play." Capped at MEMORY_LIMIT entries via remember() below (never by
    # truncating the field type itself), so prompt size can't grow without
    # bound no matter how long a show runs -- only the most recent handful
    # of facts are ever in view, which also keeps this a "recent form" read
    # rather than an ever-growing, eventually-unreadable transcript.
    memory: List[str] = field(default_factory=list)
    # player_id -> domain name -> [correct_count, total_count] for THIS
    # show only -- lets attempt_question's prompt say "you've already faced
    # N questions in this domain tonight, M correct," real accumulated
    # in-show experience with a SPECIFIC subject. Deliberately separate
    # from `memory` above (which is show-wide and narrative) since this is
    # per-player and numeric, and it's consulted on the highest-frequency
    # call path (every single trivia attempt) where a full history block
    # would be too much prompt weight to pay every time.
    domain_record: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)

    def sole_owner(self) -> Optional[int]:
        return next(iter(self.active_ids)) if len(self.active_ids) == 1 else None

    def players_adjacent(self, a: int, b: int) -> bool:
        a_regions = [r for r, o in self.owner.items() if o == a]
        b_regions = {r for r, o in self.owner.items() if o == b}
        return any(self.board_adj[r] & b_regions for r in a_regions)

    def adjacent_opponents(self, player_id: int) -> List[int]:
        return [pid for pid in self.active_ids if pid != player_id and self.players_adjacent(player_id, pid)]

    MEMORY_LIMIT = 8

    def remember(self, line: str) -> None:
        self.memory.append(line)
        if len(self.memory) > self.MEMORY_LIMIT:
            self.memory = self.memory[-self.MEMORY_LIMIT:]

    def recent_history_text(self) -> str:
        """A single sentence-joined block of the most recent remembered
        facts, oldest first -- empty string if nothing's happened yet
        (early in the show), which every call site below checks before
        adding a "SHOW SO FAR" section to a prompt at all."""
        return " ".join(self.memory)

    def domain_familiarity(self, player_id: int, domain_name: str) -> Optional[Tuple[int, int]]:
        """(correct, total) this player has faced in domain_name so far
        this show, or None if they haven't seen this domain at all yet --
        the None case matters: a player's very first question in a domain
        should read as genuinely uninformed, not claim "0 correct" as if
        that were meaningful experience."""
        rec = self.domain_record.get(player_id, {}).get(domain_name)
        return (rec[0], rec[1]) if rec else None

    def record_duel_turns(self, turns_log: List[dict]) -> None:
        for turn in turns_log:
            rec = self.domain_record.setdefault(turn["player_id"], {}).setdefault(turn["domain"], [0, 0])
            rec[1] += 1
            if turn["correct"]:
                rec[0] += 1
