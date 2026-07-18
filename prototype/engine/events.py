"""The event log. Every state change emits a structured event here; this
log is the entire game and is fully replayable (design doc, Section 10)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Dict, Any, Optional
import time


@dataclass
class Event:
    seq: int
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLog:
    def __init__(self, on_emit: Optional[Callable[[Event], None]] = None) -> None:
        self._events: List[Event] = []
        # Optional callback invoked synchronously right after each event is
        # appended -- lets a caller (e.g. server.py) stream events out to a
        # client as they're produced, instead of waiting for the whole show
        # to finish. None by default: fully backward compatible with any
        # caller that only wants the final to_list().
        self._on_emit = on_emit

    def emit(self, type_: str, **data: Any) -> Event:
        ev = Event(seq=len(self._events), type=type_, data=data)
        self._events.append(ev)
        if self._on_emit is not None:
            self._on_emit(ev)
        return ev

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)
