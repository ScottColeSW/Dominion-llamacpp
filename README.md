# Dominion — Agent vs. Agent

A browser-based game show simulator: thirteen contestants draft trivia
domains, duel head-to-head on a chess clock, and fight to become sole owner
of the board for a $100,000,000 grand prize. Each player is backed by a
local LLM that makes their live in-show decisions and answers trivia; if a
call fails, that player transparently falls back to a scripted stand-in
agent so the show never stalls (see
[`src/dominion/agents/llm_agent.py`](src/dominion/agents/llm_agent.py)).

## Two implementations, one comparison

This repo is a fork of the original Ollama-only prototype, in progress on
porting the inference layer to **llama.cpp** and raising the engineering
around it to a real service architecture, so the two can be run side by
side and compared on real numbers (latency, live-vs-fallback reliability,
accuracy) rather than by feel. See
[`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md) for the original engine's
full design history. Current status: both backends run end to end on the
new async service architecture, including llama.cpp's GBNF
grammar-constrained decisions (see "llama.cpp setup" below) -- pick either
one with the Ollama/llama.cpp toggle on the start screen (falls back to
scripted per-decision on any failure, same as always, regardless of which
one you pick). The comparison view segmenting the Standings page by
backend is the next milestone; until then, recorded history isn't yet
labeled by which backend produced it.

**This is an intentional, explicit dependency tradeoff.** The original
prototype was proud of being zero-pip-dependency, standard-library-only
Python. This fork trades that for production service scaffolding — an
async web framework, structured config, structured logging, metrics,
containerization — because that scaffolding is the actual point of this
fork. The game engine itself (`src/dominion/engine/`) still has zero
third-party imports; every dependency in `pyproject.toml` is
service/transport-layer only.

## Demo

A short demonstration video of the original Ollama-only version is
available on Loom.
[`Dominion live`](https://www.loom.com/share/2d3bfddb3e5a4a4bbf9f38dd51299f70)

## Requirements

- Python 3.10+.
- [Ollama](https://ollama.com/download) — optional. Powers the Ollama
  backend's live agents; without it, every player just uses the scripted
  fallback.
- A llama.cpp `llama-server` binary — optional. Powers the llama.cpp
  backend once you pick it via the start-screen toggle; see "llama.cpp
  setup" below. Without it running (or not picked at all), every player
  just uses the scripted fallback, same as Ollama.

## Setup

```bash
python check_env.py
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

`check_env.py` checks your Python version, and if Ollama is installed,
pulls the models the live agents use. It's a preflight check, not a
package installer — `pip install -e ".[dev]"` is what actually installs
the app and its dependencies (declared in `pyproject.toml`).

## Run it

```bash
uvicorn dominion.server.app:app --port 8765
```

Then open **http://localhost:8765**. Before clicking **Start Show**, the
start screen lists every model actually installed for the selected
backend (real Ollama tags via `/api/tags`, real `.gguf` files under
`DOMINION_LLAMACPP_MODELS_DIR`) with its on-disk size, pre-checked with
sensible defaults so Start stays one click if you don't touch it.
Checking/unchecking a model live-updates a memory budget against this
machine's actual available RAM (a conservative estimate, not a real
per-backend load/swap simulation -- see
[`server/model_catalog.py`](src/dominion/server/model_catalog.py)) and
greys out picks that would no longer fit. Set `DOMINION_SCRIPTED_ONLY=1`
first to skip live model calls entirely and run near-instantly, useful
for quick local iteration.

```bash
DOMINION_SCRIPTED_ONLY=1 uvicorn dominion.server.app:app --port 8765
```

See [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md) for the engine layout,
live-agent details, and environment variables
(`DOMINION_SCRIPTED_ONLY`, `DOMINION_PORT`), and
[`design/Game Show Sim - Design Document.docx`](design/Game%20Show%20Sim%20-%20Design%20Document.docx)
for the full rule set and design rationale.

## llama.cpp setup

The backend selector (the toggle on the start screen, or
`?backend=llamacpp` on `/api/run-show` directly) already picks this
backend for a real show end to end — you just need an actual
`llama-server` running for it to talk to.

1. Build or download `llama-server` from
   [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
2. Create a models directory and put GGUF files in it, named to match
   [`inference/config.py`](src/dominion/inference/config.py)'s
   `LLAMACPP_MODELS` registry (edit that mapping if you'd rather name
   things differently, or to add/remove models — it's a plain dict, not
   generated). Roughly the same four families the Ollama backend uses,
   any Q4_K_M-or-similar quantization:
   - `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
   - `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
   - `gemma-2-2b-it-Q4_K_M.gguf`
   - `Phi-3-mini-4k-instruct-Q4_K_M.gguf`
   - `muse-glimmer-Q4_K_M.gguf` — reserved slot for a model not yet
     released as of this writing; the registry entry is a placeholder
     until a real quantized GGUF exists to point it at.
3. Run `llama-server` in **router mode** (no `-m`, so it serves every
   model in the directory from one process/port, the same way Ollama
   swaps between models on one daemon):
   ```bash
   llama-server --models-dir ./models --models-max 4 --port 8080
   ```
4. `DOMINION_LLAMACPP_URL` defaults to `http://127.0.0.1:8080` — override
   it if you ran `llama-server` on a different host/port.

## Observability

Every inference call (either backend) logs structured JSON to stdout
(model, backend, retries, latency, outcome) and records to Prometheus,
scraped at:

```
GET /metrics
```

`dominion_inference_requests_total{backend,model,outcome}`,
`dominion_inference_latency_seconds{backend,model}`, and
`dominion_inference_fallback_total{backend,model}` (see
[`observability/metrics.py`](src/dominion/observability/metrics.py)) --
no external Prometheus/Grafana setup is required to just look at the
endpoint directly.

## Running in Docker

```bash
docker compose up
```

Brings up the app (port 8765) and a `llama-server` in router mode (port
8080, serving whatever GGUFs you've put in `./models` -- see "llama.cpp
setup" above). Ollama itself isn't a compose service (containerizing it
well needs real GPU passthrough this file isn't trying to own) -- the app
container points at `DOMINION_OLLAMA_URL=http://host.docker.internal:11434`
by default, i.e. an Ollama already running on your host; override that
env var in `docker-compose.yml` if yours lives elsewhere. History persists
across restarts via the `dominion-data` named volume (`DOMINION_DB_PATH`,
set in the `Dockerfile`, points the SQLite file there instead of into the
installed package directory, which the container's non-root user can't
write to).

## Tests

```bash
pytest
```

## Not comfortable with a terminal?

The steps above assume some command-line familiarity. If you just want to
watch a show run without setting any of this up yourself, the
[demo video](https://www.loom.com/share/2d3bfddb3e5a4a4bbf9f38dd51299f70)
above is the easiest way in — running the actual server still needs
Python and (optionally) Ollama installed and the commands above typed
into Terminal (macOS/Linux) or PowerShell (Windows), there's no
one-click installer for either backend yet.

## License

MIT — see [LICENSE](LICENSE).
