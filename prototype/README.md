# Dominion (Agent vs. Agent) — Phase 1 Prototype

A runnable, text-only implementation of the core rules from the design
document: the spotlight loop, chess-clock duels, asymmetric domain
inheritance, the domain tax, the Scramble, and burst prizes.

Every player is backed by a local Ollama model, one per player, shown on
their badge (see "Ollama agents" below): it makes their live target/
push-retreat/domain-tax decisions *and* answers trivia. Any call that fails
or times out falls back to a scripted stand-in agent
([`engine/agents.py`](engine/agents.py)), so a slow, hung, or missing Ollama
server never stalls the show.

## Run it

Requires Python 3.9+. No external packages, standard library only. First
time here? Run `python setup.py` from the repo root first — see the root
[`README.md`](../README.md).

```
python3 server.py
```

If `python3` isn't right for your system, try `python server.py` instead —
whichever one `python setup.py` told you to use is always correct for your
machine.

Then open **http://localhost:8765** and click **Start Show**. The server
streams the event log to the page as newline-delimited JSON as the show is
actually produced (not all at once), which the page reveals as a scrolling
broadcast transcript, ending in the prize reveal.

## Ollama agents

Each of the 13 players is assigned one of `TEXT_MODELS` in
[`engine/ollama_agent.py`](engine/ollama_agent.py) (currently `llama3.2:latest`,
`qwen2.5:3b`, `gemma2:2b`, `phi3:mini` — pull whichever you don't already
have via `ollama pull <name>`; `gpt-oss:20b` was dropped after it started
hard-erroring on this machine's Ollama version/GPU combination). That model
makes the player's live `choose_target` / `decide_continue` /
`choose_tax_target` decisions *and* answers trivia (as a 4-way multiple
choice against the same distractor pool the frontend already shows).
`OllamaAgent` falls back to the scripted heuristic on any timeout, connection
error, or unparseable reply, so a slow, hung, or unreachable Ollama server
never stalls or crashes the show.

`ollama_agent.py` and `fetch_images.py` both talk to Ollama at `127.0.0.1`,
not `localhost` -- on some machines `localhost` resolves to the IPv6
loopback first, and since Ollama only listens on IPv4 that connection hangs
in `SYN_SENT` for minutes instead of failing fast, silently defeating every
timeout in this file. If you ever see a call hang far longer than its
timeout should allow, check `netstat` for a `SYN_SENT` entry to `[::1]:11434`
before assuming it's just a slow model.

With a working GPU, warm calls are well under a second; without one, expect
the 11s-86s range noted below. Live trivia uses the same 25s duel clock as
scripted play (Revision 18's original call, "shortened from 60s to 25s
specifically to burn through duels faster") -- `attempt_question` caps how
long it'll wait for a live reply to roughly what's left on that player's own
clock, so a slow or cold-loading call can't drag a duel out past when the
clock should already have ended it.

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
- `engine/agents.py` — the scripted fallback agent every player's
  `OllamaAgent` falls back to on any failed/unparseable live call.
- `engine/ollama_agent.py` — the live, Ollama-backed agent: target/continue/
  tax decisions and trivia answering; see "Ollama agents" above.
- `engine/fetch_images.py` — pre-production image fetch/vision-verification
  script; see "Vision-assisted image fetch" below.
- `engine/game.py` — orchestrates the full show and emits every event.
- `server.py` — a dependency-free local server streaming the show as
  newline-delimited JSON over HTTP.
- `web/index.html` — the Start button and broadcast-style display.

## Verified

200 simulated shows (scripted agents), zero failures, always exactly twelve
duels, domain inheritance chains correctly across multiple hops, the
Scramble triggers at the documented threshold. See the design document for
the full rule set this implements.
