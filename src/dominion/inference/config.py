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
    # Scott's own Ollama library has qwen2.5:7b pulled beyond the curated
    # TEXT_MODELS four -- a real, matched GGUF counterpart, same
    # bartowski Q4_K_M quantization convention as everything above.
    # Bigger than the rest of this roster (~4.7GB vs ~2GB), so
    # llama-server needs --models-max bumped to at least 5 to actually
    # serve it alongside the original four -- see README.md.
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    # Placeholder: muse-glimmer turned out to be real (Meta's "Muse
    # Glimmer," a 30B-parameter multimodal agentic model) but wildly out
    # of scale with everything else here -- its smallest public GGUF
    # quant is 17GB+ vs. ~2-5GB for this roster's other entries, and
    # genuinely needs real VRAM headroom to run at all. Deliberately left
    # unmapped pending an explicit decision (real, quantifiable download/
    # stability cost) rather than silently pulled in -- this entry
    # reserves the slot/identifier only; replace the filename with the
    # real GGUF basename once you've actually decided to add it and
    # placed the file under your models directory.
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
    # Scott: "RAM usage climbing every round during a live show," then
    # separately, "ollama really manages the memory well if we use it
    # with the right parameters." Root cause: ollama_client.py never sent
    # a keep_alive on /api/generate at all, so Ollama fell back to its own
    # default (5 minutes) -- with a real show cycling through TEXT_MODELS'
    # whole ~4-5-model pool well inside any single 5-minute window, every
    # one of them ends up resident in memory AT ONCE for as long as the
    # show keeps running (and can still be resident when the NEXT show
    # starts, since Ollama's model cache lives in the daemon, not this
    # process). A short, explicit keep_alive caps how many models can
    # pile up before an idle one actually unloads -- the real tradeoff is
    # memory vs. cold-load latency (an unloaded model's next call pays a
    # real reload cost, see docs/ENGINE_NOTES.md's 86s-cold measurement)
    # so this is deliberately override-able, not hardcoded, rather than
    # this codebase silently picking a value for a tradeoff Scott may
    # want tuned differently. Ollama's own accepted formats: a duration
    # string ("30s", "5m"), a plain number of seconds, or "-1" (never
    # unload) / "0" (unload immediately after each call).
    ollama_keep_alive: str = "1m"
    llamacpp_url: str = "http://127.0.0.1:8080"
    # Where llama-server's own --models-dir points (README.md's "llama.cpp
    # setup" section) -- the model picker (server/model_catalog.py) scans
    # this same directory for *.gguf files so "installed" means the exact
    # same thing on both backends: actually present on disk, not a fixed
    # list in code.
    llamacpp_models_dir: str = "./models"

    # The Host (agents/host_agent.py, M9) is fixed to one dedicated model
    # per backend, independently declared here the same way TEXT_MODELS/
    # LLAMACPP_MODELS are independent of each other -- not exposed in the
    # M8 model picker for v1, a deliberate cut, not an oversight.
    #
    # Bumped from llama3.2:latest to a bigger model deliberately: the Host
    # speaks maybe 3 times a duel (announce_challenge/announce_duel_result/
    # announce_continue_decision), nowhere near attempt_question's
    # every-single-turn call volume for players, so it can absorb real
    # cold-load cost far more easily than TEXT_MODELS can (measured 37s vs.
    # the fast tier's 9-24s). Also frees llama3.2:latest to be purely a
    # player voice instead of double-duty as Host+player, and a bigger
    # model is more likely to actually hold onto an opinionated persona
    # instruction consistently, which the Host doesn't have yet.
    host_model_ollama: str = "qwen2.5:7b"
    host_model_llamacpp: str = "llama-3.2-3b-instruct"

    # The Commentator (agents/commentator_agent.py, M11) -- a second,
    # independently declared dedicated model, same pattern as the Host's
    # pair above. Deliberately a different default than the Host on
    # Ollama so the two AI voices aren't literally the same weights
    # talking to themselves.
    commentator_model_ollama: str = "qwen2.5:3b"
    commentator_model_llamacpp: str = "qwen2.5-3b-instruct"

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
    # choose_target -- Scott: "needs to be more out loud like an
    # interview and quicker. we give them 7 seconds to pick before we
    # choose for them, first adjacent." Same tight-live-beat reasoning as
    # continue_decision_timeout above, just a hard number Scott named
    # directly rather than one this codebase measured its way to.
    target_decision_timeout: float = 7.0

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
