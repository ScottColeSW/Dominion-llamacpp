"""The backend-agnostic inference boundary. dominion.agents.llm_agent talks
only to this interface -- never to httpx/urllib directly -- so the same
agent code runs unmodified against Ollama or llama.cpp (see
ollama_client.py / llamacpp_client.py), selected via inference/config.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class GenerationResult:
    """Backend-agnostic reply. think_seconds mirrors what
    dominion.agents.llm_agent has always charged against a player's duel
    clock -- actual generation/reasoning time, with model-load overhead
    excluded (see each client's own docstring for how it derives this from
    its backend's specific response fields). None on any failure -- the
    caller (llm_agent.py) falls back to ScriptedAgent exactly as it always
    has, backend-independent.
    """

    text: Optional[str]
    think_seconds: Optional[float]
    raw_latency_seconds: float


class InferenceClient(Protocol):
    """One method, one capability flag. Every existing prompt-building and
    reply-parsing behavior in llm_agent.py is unchanged by this
    abstraction -- only the transport call underneath it moves behind
    this interface."""

    #: Whether this backend can constrain generation to a GBNF grammar
    #: (see inference/grammars.py). Ollama's public /api/generate has no
    #: such mechanism; llama.cpp's /completion does. llm_agent.py checks
    #: this flag rather than branching on backend name, so a future
    #: backend that also supports grammars needs no agent-side change.
    supports_grammar: bool

    #: Stable label ("ollama", "llamacpp") used as the "backend" tag on
    #: every structured log line and Prometheus metric (observability/
    #: metrics.py) -- and, via game.py's own record of the same string,
    #: what ends up in history.py's shows/player_stats rows. Every layer
    #: of the backend comparison traces back to this one string per client.
    backend_name: str

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        timeout: float,
        num_predict: Optional[int] = None,
        grammar: Optional[str] = None,
    ) -> GenerationResult:
        """Never raises on a network/timeout/parse failure -- returns a
        GenerationResult with text=None instead, same contract the
        original _ask_ollama had (see docs/ENGINE_NOTES.md), so callers
        never need a try/except around a live call."""
        ...

    async def aclose(self) -> None:
        """Releases the underlying connection pool. game.py's
        _default_client_for_backend calls this on any client it created
        itself; server/app.py's lifespan calls it on its two long-lived,
        shared clients at shutdown."""
        ...
