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

`AnswerAttempt.live` (`engine/agents.py`) marks whether an answer actually
came from a live model reply, as opposed to the scripted fallback. The
frontend uses this: a live turn's `seconds_used` is real wall-clock time a
model already spent *server-side*, before the event ever streamed out, so
`playDuelTurn` (`web/index.html`) gives it a short fixed flourish instead of
re-animating the clock over that same duration a second time -- which would
otherwise double the real wait and drift the on-screen countdown away from
actual elapsed time. Scripted turns are untouched, still the original
real-time-proportional animation.

**The clock charges thinking time, not loading time.** `_ask_ollama`
(`engine/ollama_agent.py`) reads Ollama's own `total_duration` minus
`load_duration` and charges *that* against a player's duel clock, not the
full wall-clock elapsed. Without this, whichever player's model happened to
need a cold load paid for it out of their own 25s clock as if it were slow
reasoning -- and since `choose_target` already warms the challenger's model
for free (an untimed decision call, right before the duel starts) while the
defender gets no such warm-up, this was a real, structural bias: one live
show measured the challenger winning 8 of 10 cap-resolved duels. Also raised
the question cap for live play only (`LIVE_QUESTION_CAP` in `game.py`) --
duel.py's original 30-question cap assumed scripted-pace answers (1-4s
each); a live answer can cost well under a second, so 30 stopped being a
rare safety valve and became the ordinary way duels ended, usually with most
of both clocks still unspent. Scripted play is unaffected by either change
(still `QUESTION_CAP = 30`, still full wall-clock charged, since scripted
seconds_used was never real elapsed time to begin with).

**Every turn costs at least a beat, never less.** A live re-verification
after the above fixes measured charged think_seconds averaging under 0.1s
per attempt on this machine's small local models -- fast enough that the
clock barely moved no matter how many attempts happened, so the question
cap (not the clock) ended up deciding almost every duel regardless of how
high it was set. `MIN_CHARGED_SECONDS` (`ollama_agent.py`) floors
`attempt_question`'s charged time at 1.0s -- a floor under real latency,
never an addition on top of it, so a genuinely slow call still costs its
real time. `duel.py`'s forced-pass fallback and `agents.py`'s scripted pass
timing were bumped to the same 1.0s floor for consistency: no turn, live or
scripted, forced or genuine, should read as having taken less than a beat.

**Both sides get a warm-up, not just the challenger.** Right as a duel opens
(before either clock starts), the host now gets a one-line in-character
reaction from *both* players about the domain on the line
(`intro_line` in `engine/agents.py`/`engine/ollama_agent.py`, streamed as a
`pre_duel_intro` event per side). This is real narration, not filler: the
challenger-bias fix above only closed the gap in what a cold load *costs*
once charged; it didn't stop `choose_target` from being an untimed call that
warms the challenger's model for free before every duel while the defender
got nothing equivalent. Giving both sides one live call here, symmetrically,
before the timed clock starts, closes that gap too.

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

## Long-running history

Every show is also persisted to a local SQLite database,
`engine/dominion_history.db` (gitignored -- it's a growing local log, not
checked-in data), via `engine/history.py`. `server.py` feeds it the same
event stream the frontend consumes, so this needs no engine changes and
adds no per-request latency the show doesn't already pay for. It's a passive
observer, not a dependency: every write is wrapped so a persistence failure
(disk full, locked file, whatever) can never affect a running show, only
this record of it. Three tables -- `shows` (one row per run: seed,
timestamps, champion, prize), `duels` (one row per duel: participants,
winner, domain, clocks remaining), `player_stats` (one row per player per
show: wins, accuracy, average correct-answer time) -- enough to ask
questions across many shows later (does the challenger-bias fix above
actually hold at scale? does any one model out-perform the others over
hundreds of duels?) without re-running anything.

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
- `engine/history.py` — passive SQLite recorder for cross-show analysis;
  see "Long-running history" above.
- `server.py` — a dependency-free local server streaming the show as
  newline-delimited JSON over HTTP.
- `web/index.html` — the Start button and broadcast-style display.

## Verified

200 simulated shows (scripted agents), zero failures, always exactly twelve
duels, domain inheritance chains correctly across multiple hops, the
Scramble triggers at the documented threshold. See the design document for
the full rule set this implements.
