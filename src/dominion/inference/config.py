"""Central, env-driven configuration for the inference layer. Replaces the
scattered module-level constants ollama_agent.py used to define directly
(OLLAMA_URL, OLLAMA_TIMEOUT, CONTINUE_DECISION_TIMEOUT) with one
pydantic-settings model, read once at process startup like
dominion.engine.game.SCRIPTED_ONLY already is.
"""
from __future__ import annotations

from typing import Dict, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# llama.cpp models registry: maps a stable identifier (used the same way
# agents/llm_agent.py's TEXT_MODELS list of Ollama tags is today) to the
# GGUF basename expected under llama-server's --models-dir in router mode
# (see llamacpp_client.py's module docstring and README.md's llama.cpp
# setup section). A plain module constant, not a Settings field -- this is
# a code-level roster analogous to TEXT_MODELS, not a single env-tunable
# value. Nothing here auto-downloads a GGUF; point each entry at whatever
# file you've actually placed in your models directory.
LLAMACPP_MODELS: Dict[str, str] = {
    "llama-3.2-3b-instruct": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "gemma-2-2b-it": "gemma-2-2b-it-Q4_K_M.gguf",
    "phi-3-mini-4k-instruct": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    # Placeholder: muse-glimmer isn't a model this codebase has any real
    # knowledge of (not released as of this writing) -- this entry only
    # reserves the slot and the identifier so the roster/config wiring
    # doesn't need a second pass later. Replace the filename with the real
    # quantized GGUF's actual basename once it's out and you've placed it
    # under your models directory; until then this entry simply won't
    # resolve to a real file and llama-server will report it unavailable,
    # same as any other misconfigured entry.
    "muse-glimmer": "muse-glimmer-Q4_K_M.gguf",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOMINION_")

    # Which InferenceClient backend a show runs against by default -- M4
    # adds a per-request override (the backend selector), this is the
    # fallback when a request doesn't specify one.
    inference_backend: Literal["ollama", "llamacpp"] = "ollama"

    # 127.0.0.1, not "localhost" -- see ollama_client.py's docstring for
    # why (IPv6/IPv4 resolution-order hang, not a stylistic choice).
    ollama_url: str = "http://127.0.0.1:11434"
    llamacpp_url: str = "http://127.0.0.1:8080"
    # Where llama-server's own --models-dir points (README.md's "llama.cpp
    # setup" section) -- the model picker (server/model_catalog.py) scans
    # this same directory for *.gguf files so "installed" means the exact
    # same thing on both backends: actually present on disk, not a fixed
    # list in code.
    llamacpp_models_dir: str = "./models"

    # Generous given observed cold-start cost on CPU-only inference (86s
    # measured on the original dev machine, see docs/ENGINE_NOTES.md) --
    # a slow reply still beats a wrong/absent one, and the ScriptedAgent
    # fallback covers a reply that never comes at all.
    generate_timeout: float = 90.0
    # decide_continue's push/retreat announcement is a live, on-air moment
    # the audience is actively waiting on -- a much tighter budget than
    # generate_timeout so the whole announcement beat (host build-up +
    # this call + the reaction) stays inside what a live show can afford.
    continue_decision_timeout: float = 6.0

    # Bounded retry + circuit breaker (inference/retry.py) shared by every
    # backend client -- a live model call is worth one quick retry on a
    # transient failure (a dropped connection, not a real timeout), but
    # not an unbounded one; after enough consecutive failures the breaker
    # opens and fails fast for a cooldown window instead of continuing to
    # hammer a backend that's genuinely down.
    retry_max_attempts: int = 2
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0


settings = Settings()
