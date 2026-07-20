# Dominion — Agent vs. Agent

A browser-based game show simulator: thirteen contestants draft trivia
domains, duel head-to-head on a chess clock, and fight to become sole owner
of the board for a $100,000,000 grand prize. Each player is backed by a
local Ollama model that makes their live in-show decisions and answers
trivia; if Ollama isn't installed or a call fails, that player transparently
falls back to a scripted stand-in agent so the show never stalls (see
[`prototype/engine/ollama_agent.py`](prototype/engine/ollama_agent.py)).

## Requirements

- Python 3.9+. No pip packages needed — the whole project is standard
  library only (`requirements.txt` exists but is intentionally empty, for
  tooling that expects one).
- [Ollama](https://ollama.com/download) — optional. Powers the live agents;
  without it, every player just uses the scripted fallback.

## Setup

```
python setup.py
```

Checks your Python version, and if Ollama is installed, pulls the models the
live agents use. Prints the exact command to start the server at the end —
use that command; it accounts for whether your system's Python 3 is called
`python` or `python3`. (If `python setup.py` itself doesn't run, try
`python3 setup.py`.)

## Run it

```
python prototype/server.py
```

Then open **http://localhost:8765** and click **Start Show**. Set
`DOMINION_SCRIPTED_ONLY=1` first to skip live Ollama calls entirely and run
near-instantly, useful for quick local iteration.

See [`prototype/README.md`](prototype/README.md) for the engine layout and
live-agent details, and
[`design/Game Show Sim - Design Document.docx`](design/Game%20Show%20Sim%20-%20Design%20Document.docx)
for the full rule set and design rationale.

## License

MIT — see [LICENSE](LICENSE).
