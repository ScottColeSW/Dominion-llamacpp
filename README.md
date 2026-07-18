# Dominion — Agent vs. Agent

A browser-based game show simulator: thirteen contestants draft trivia
domains, duel head-to-head on a chess clock, and fight to become sole owner
of the board for a $100,000,000 grand prize. Player decisions are currently
made by a scripted stand-in agent (see [`prototype/engine/agents.py`](prototype/engine/agents.py)),
with a real Claude-backed agent planned as a later phase.

## Run it

Requires Python 3.9+, standard library only — no external packages.

```
python prototype/server.py
```

Then open **http://localhost:8765** and click **Start Show**.

See [`prototype/README.md`](prototype/README.md) for the engine layout, and
[`design/Game Show Sim - Design Document.docx`](design/Game%20Show%20Sim%20-%20Design%20Document.docx)
for the full rule set and design rationale.

## License

MIT — see [LICENSE](LICENSE).
