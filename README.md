# Dominion — Agent vs. Agent

A browser-based game show simulator: thirteen contestants draft trivia
domains, duel head-to-head on a chess clock, and fight to become sole owner
of the board for a $100,000,000 grand prize. Each player is backed by a
local Ollama model that makes their live in-show decisions and answers
trivia; if Ollama isn't installed or a call fails, that player transparently
falls back to a scripted stand-in agent so the show never stalls (see
[`prototype/engine/ollama_agent.py`](prototype/engine/ollama_agent.py)).

# Demo
A short demonstration video is available on Loom.
[`Dominion live`](https://www.loom.com/share/2d3bfddb3e5a4a4bbf9f38dd51299f70)

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

See [`prototype/README.md`](prototype/README.md) for the engine layout,
live-agent details, and environment variables
(`DOMINION_SCRIPTED_ONLY`, `DOMINION_PORT`), and
[`design/Game Show Sim - Design Document.docx`](design/Game%20Show%20Sim%20-%20Design%20Document.docx)
for the full rule set and design rationale.

## macOS setup, step by step

Everything above works as-is on a Mac — this section is just the same steps
spelled out in full for anyone who's never used Terminal before.

1. **Get the project onto your Mac.** However you received it (a zipped
   folder, a `git clone`), make sure it ends up somewhere easy to find, like
   your Desktop.
2. **Install Python.** macOS ships a stub `python3` command tied to Xcode's
   developer tools, but it's often old and not meant for running real
   projects. Download the official installer from
   [python.org/downloads](https://www.python.org/downloads/) (get 3.9 or
   newer, the latest release is fine), open the `.pkg`, and click through
   the installer with the default options.
3. **Install Ollama (optional, but it's what makes the players "real").**
   Download the Mac version from
   [ollama.com/download](https://ollama.com/download), open the `.dmg`,
   drag the Ollama app into Applications, then launch it once from
   Applications so it finishes installing its background service. Skipping
   this step is fine too — the show still runs end to end with every
   player on the scripted fallback agent instead of a live model.
4. **Open Terminal.** Press `Cmd+Space`, type `Terminal`, hit Enter. A
   Mac's Terminal is the same idea as Command Prompt/PowerShell on Windows,
   just a different app.
5. **Navigate to the project folder.** Type `cd ` (with a trailing space,
   don't hit Enter yet), then drag the project folder from Finder straight
   into the Terminal window — it'll paste in the full path automatically —
   and press Enter.
6. **Run setup, then start the server:**
   ```
   python3 setup.py
   python3 server.py
   ```
   (On a Mac, `python3` is the right command, not `python`, for most
   installs — `setup.py`'s own output confirms which one to use if that
   guess is ever wrong.)
7. **Open the show.** Go to a browser and visit **http://localhost:8765**,
   then click **Start Show**.

Nothing here needs `sudo` or admin rights beyond the two `.pkg`/`.dmg`
installer clicks — this all runs as a completely ordinary local Python
process.

## License

MIT — see [LICENSE](LICENSE).
