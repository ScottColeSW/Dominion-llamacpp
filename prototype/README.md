# Dominion (Agent vs. Agent) — Phase 1 Prototype

A runnable, text-only implementation of the core rules from the design
document: the spotlight loop, chess-clock duels, asymmetric domain
inheritance, the domain tax, the Scramble, and burst prizes.

Trivia answering is still a scripted stand-in (real inference is too slow
relative to the 25-second duel clock on typical hardware — see "Ollama
agents" below). The three lower-frequency player decisions — who to
challenge, whether to push or retreat, who to domain-tax — are live calls to
a local Ollama model, one per player, shown on their badge.

## Run it

Requires Python 3.9+. No external packages, standard library only.

```
python3 server.py
```

Then open **http://localhost:8765** and click **Start Show**. The server
streams the event log to the page as newline-delimited JSON as the show is
actually produced (not all at once), which the page reveals as a scrolling
broadcast transcript, ending in the prize reveal.

## Ollama agents

Each of the 13 players is assigned one of `TEXT_MODELS` in
[`engine/ollama_agent.py`](engine/ollama_agent.py) (currently `llama3.2:latest`,
`gpt-oss:20b`, `qwen2.5:3b`, `gemma2:2b`, `phi3:mini` — pull whichever you
don't already have via `ollama pull <name>`). That model makes the player's
live `choose_target` / `decide_continue` / `choose_tax_target` decisions;
`OllamaAgent` falls back to the scripted heuristic on any timeout, connection
error, or unparseable reply, so a slow, hung, or unreachable Ollama server
never stalls or crashes the show.

**Expect real latency.** On CPU-only inference (no usable GPU offload), a
single call took 11s warm / 86s cold in testing — a full show can run well
past ten minutes once you add ~30 live calls, more if Ollama has to swap
loaded models between players. Set `DOMINION_SCRIPTED_ONLY=1` in the
environment before starting the server to force plain scripted decisions for
every player (near-instant, no live calls at all) when you just want to
iterate quickly.

## Vision-assisted image fetch

[`engine/fetch_images.py`](engine/fetch_images.py) (pre-production, not
imported by the running engine) fetches real Wikimedia Commons photos per
question. Pass `--vision` to additionally verify each candidate with a local
`llava:7b` model (`ollama pull llava:7b`) before accepting it, instead of
trusting Commons' title-text search ranking alone:

```
python3 engine/fetch_images.py --domain Cats --limit 15 --vision
```

## Layout

- `engine/board.py` — the hub-and-ring board formula, sized to any player
  count, verified connected at every size the Scramble can produce.
- `engine/content.py` — the domain library (a working subset of the
  full fifty; same shape, just add more entries to grow it).
- `engine/models.py` — Player and GameState.
- `engine/duel.py` — the chess-clock duel resolver.
- `engine/agents.py` — the scripted Phase 1 agent (trivia answering, and the
  decision fallback behavior).
- `engine/ollama_agent.py` — the live, Ollama-backed agent for the target/
  continue/tax decisions; see "Ollama agents" above.
- `engine/game.py` — orchestrates the full show and emits every event.
- `server.py` — a dependency-free local server streaming the show as
  newline-delimited JSON over HTTP.
- `web/index.html` — the Start button and broadcast-style display.

## Verified

200 simulated shows (scripted agents), zero failures, always exactly twelve
duels, domain inheritance chains correctly across multiple hops, the
Scramble triggers at the documented threshold. See the design document for
the full rule set this implements.
