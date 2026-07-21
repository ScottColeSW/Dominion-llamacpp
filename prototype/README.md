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

**Every turn costs a whole second, always rounded up, never a fraction.** A
live re-verification after the above fixes measured charged think_seconds
averaging under 0.1s per attempt on this machine's small local models --
fast enough that the clock barely moved no matter how many attempts
happened, so the question cap (not the clock) ended up deciding almost
every duel regardless of how high it was set. `attempt_question`
(`ollama_agent.py`) now charges `math.ceil()` of the real thinking time --
0.6s costs a full 1, 3.1s costs 4 -- with `MIN_CHARGED_SECONDS` (1) as a
defensive backstop for the edge case where measured think time is exactly
0.0. `duel.py`'s forced-pass fallback and `agents.py`'s scripted pass/answer
timing were moved to the same whole-integer-seconds scheme for consistency.
`AnswerAttempt.seconds_used` (`agents.py`) is typed `int`, not `float`, and
`duel.py`'s per-turn `clock_remaining` is a plain `int` too -- every value
on the clock's path is a whole number end to end. This also means a player
stuck missing, missing, passing, missing before finally landing an answer
can rack up a genuinely large integer total across those attempts, which
reads as real, escalating tension rather than a suspiciously precise
fraction.

**A blurt can genuinely win the question, not just decorate the wait.** The
rapid-fire "blurt" guesses flashing by on screen while the clock ticks
(`startRapidFireGuesses`, `web/index.html`) used to be purely cosmetic --
real other answers from the same domain, cycled for flavor, but never
connected to the actual outcome. Scott: "a blurt could be a right answer and
counts if it is" -- a real mechanic, not just flavor. `duel.py`'s
`LUCKY_BLURT_CHANCE` (6%) is now rolled once per attempt (skipped during a
forced pass, and only when there's an actual blurt to land on); a hit
resolves the turn correct immediately and skips the real agent call
entirely -- live or scripted -- since the blurt already got there first,
charging a short 1-2s "fast, lucky beat" instead of a considered answer's
timing. The new `lucky_blurt` flag rides along on the `duel_turn` event so
the frontend can make that turn's blurt burst actually land on the true
answer (as the final flick, with its own gold glow -- see the CSS comment
above `.say-text.lucky-hit`) instead of only ever cycling distractors, plus
a dedicated crowd reaction pool (`REACTION_LUCKY_BLURT`) that always fires
for it rather than rolling against the ordinary reaction chance. Applies
identically to both players every turn, so it's neutral with respect to the
challenger/defender split documented under "Long-running history" below.

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

**The defender really does have a structural edge, not just bad luck.**
`get_stats()` (`engine/history.py`) aggregates `player_stats`/`duels` by
model (the only identity that actually persists across shows -- player_id,
kingdom name, and profession are all redrawn fresh every run) for the new
Standings page (`web/stats.html`, served at `/stats.html`, backed by the new
`GET /api/stats` in `server.py`). Across the first 18 recorded shows (216
duels), the defender won 56.5% of the time vs the challenger's 43.5% --
`duel.py` tests "the DEFENDER's domain only" by design (Section 4), so the
defender is answering a domain they already hold while the challenger is
attacking into potentially unfamiliar territory. That's a real home-turf
advantage baked into the rules as written, not an engine bug -- surfaced on
the Standings page itself (a "home-turf" panel with the live split) so it
stays visible as more shows get recorded, rather than something only
discoverable by querying the db by hand.

## Layout

- `engine/board.py` — the hub-and-ring board formula, sized to any player
  count, verified connected at every size the Scramble can produce.
- `engine/test_board_geometry.py` — checks board.py's declared adjacency
  against the actual drawn hex geometry (`web/index.html`'s own layout
  math, mirrored in Python); see "Verified" below.
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
- `engine/history.py` — passive SQLite recorder for cross-show analysis,
  plus `get_stats()`, the aggregation query behind the Standings page; see
  "Long-running history" above.
- `server.py` — a dependency-free local server streaming the show as
  newline-delimited JSON over HTTP, plus `GET /api/stats`.
- `web/index.html` — the Start button and broadcast-style display.
- `web/stats.html` — the Standings page: a leaderboard and one
  "baseball card" per model, aggregated across every recorded show.

## Verified

200 simulated shows (scripted agents), zero failures, always exactly twelve
duels, domain inheritance chains correctly across multiple hops, the
Scramble triggers at the documented threshold. See the design document for
the full rule set this implements.

**The post-Scramble board's logical adjacency now matches what's actually
drawn on screen.** `build_hub_ring` (`engine/board.py`) used to also connect
each ring tile to the one TWO positions away ("offset-2"), on top of its
immediate neighbor -- reasoning it'd give more matchup variety. But
`circularPositions` (`web/index.html`) lays the post-Scramble board out as a
real hex flower (hub at center, ring tiles at exact hex-touching distance
from the hub and from their immediate neighbors, 60 degrees apart), and an
offset-2 pair sits 120 degrees apart -- about 1.7x the true hex-touching
distance, with a visibly different-colored tile actually sitting between
them. A player could legitimately (per the graph) hold both without holding
what's between them, which read as broken, split territory on screen even
though the game considered it one connected piece (Scott's report: "the
collapse game breaking contiguous territory ownership"). Fixed by dropping
the offset-2 ring edges -- verified directly across 100 shows that the
logical ring adjacency now exactly equals true hex-flower geometric
adjacency, and a 300-show contiguity stress test (checking every winner's
and every reassignment recipient's territory forms one connected piece
after every transfer) still comes back clean. The hub alone already
guarantees the whole graph stays connected regardless (it borders every
ring tile), so this costs a little matchup variety, not connectivity.

**That fix is now a permanent test, not a one-off verification.**
[`engine/test_board_geometry.py`](engine/test_board_geometry.py) mirrors
`hexPoints`/`pyramidPositions`/`circularPositions` (`web/index.html`) in
Python, derives "true" adjacency directly from which hexagons' edges
actually coincide (the same fact a viewer would see on screen), and asserts
`board.py`'s declared `board_adj` equals it -- for the pyramid AND for
every ring size the Scramble can produce (2 through 7 active players, not
just the 7 it happens to fire at today). A plain internal-consistency check
on `board_adj` would never have caught the offset-2 bug above, since that
graph was always self-consistent; only the picture disagreed with it. This
test would have failed immediately. Run it with:

```
python -m unittest engine.test_board_geometry -v
```

It already earned its keep once, on the very first run: it caught a
second, latent version of the same class of bug -- the ring-closing logic
treated the occupied slots as always wrapping into a full circle, but
`circularPositions`' six slots only close into one once ALL SIX are in use
(ring_size 6, i.e. exactly 7 active players). For any smaller ring
(reachable if `SCRAMBLE_MAX_ACTIVE` or the Scramble threshold ever change,
not reachable in a show today), the occupied slots are a contiguous arc
with real empty angular space at each end, and the old code drew a phantom
edge connecting them anyway. Not live yet, but exactly the kind of thing
this test exists to catch before it becomes a real bug in the wild.

**The board redraw is now its own mini-intermission.** `rebuildBoardAfterScramble`
(`web/index.html`) used to just fade the board's opacity out and back in.
It now reuses the same curtain assets as the pre-production open
(`curtainOverlay`/`curtainText`/`spawnSparkles`): closes the curtain, shows
"Board Scramble", rebuilds the board entirely behind it (the audience never
sees the swap itself), then reopens with the same flash + crowd swell as
the real open. Verified directly (not through a full throttled show replay
-- see below): called with fabricated `board_size`/`new_owner` data against
a live page, confirmed the curtain classes transition correctly end to end
and the rebuilt board's tile fills matched the supplied ownership exactly.

**A disconnected client no longer crashes the server.** `server.py`'s
`write_event` calls `self.wfile.write(...)`, which raises
`ConnectionAbortedError`/`ConnectionResetError`/`BrokenPipeError` the moment
a browser tab closes or navigates away mid-show -- previously this
propagated all the way up through `emit()` to socketserver's default error
handler, printing a full traceback for a completely ordinary occurrence
(Scott hit this from the terminal). `do_POST` now catches those three
specifically around `run_show(...)` and just stops quietly: nothing to
recover, there's simply nobody left listening.

**A note on testing this in an automated/headless browser:** the Browser
pane used to verify the above kept reporting `document.hidden === true`
even when fronted, and the whole show-playback loop leans on
`setTimeout`-based `sleep()` for its pacing throughout (not just the duel
clock, which was already hardened against exactly this -- see "Both sides
get a warm-up" above) -- Chrome throttles timers hard in hidden/backgrounded
documents, so a full show can appear to hang for tens of seconds at a time
in that environment specifically. Confirmed this wasn't an engine or
frontend bug by hitting `/api/run-show` directly with `curl`: the full
12-duel show, `finale` event included, comes back in well under a second
server-side every time.

**The content pool is deduped and meaningfully bigger.** `engine/content.py`
went from 2050 to 2355 questions across the same 39 domains (Scott: "I do
think we have to expand the domain questions and answers pool"). Along the
way, a real bug turned up: `duel.py`'s `_pick_distractors` dedupes the
blurt/distractor pool by answer VALUE, so nine domains with a repeated
answer (56 collisions total -- Board Games had 60 questions but only 52
distinct answers) were silently drawing from a smaller effective pool than
their question count implied. Every collision was replaced with a fresh
question, not just deleted, so the fix and the growth pass happened
together. Verified with a script asserting zero within-domain answer
duplicates across all 39 domains (was 56, now 0), plus 60 full scripted
shows run end to end with no errors and `engine/test_board_geometry.py`
still green.

**A player's expertise was never actually independent of their board
domain -- it was the same value.** `game.py`'s draft loop used to set
`origin_domain=domain.name`, aliasing a player's real (if amateur) expertise
straight to whichever domain they drafted onto the board -- meaning every
player started every show as a genuine expert in their own held territory,
100% of the time, never the rare lucky coincidence the design called for
(Scott, an earlier session: "the players do not necessarily have expertise
that will come up at all... or the super lucky would land on a matching
domain"), and exactly why the interview badge's "Expertise:" and "Holds:"
lines always showed the identical domain at the top of every show. Now
drawn independently from the full 39-domain library via `rng.choice`.
Verified directly: replaying the exact draft logic across 300 shows, the
match rate dropped from 100% to 2.87% (expected ~2.56% for 1-in-39 odds).

**The board's domain draw was already fair -- confirmed, not assumed.**
Scott's report that "the same domains come up as board territory a lot"
turned out to be sampling variance, not a bug: `pick_domains` draws 13 of
~39 domains (a third of the whole library) every single show, so real
overlap across just a handful of shows is expected by chance alone. A
500-show simulation came back statistically uniform (expected ~166.7 draws
per domain, observed range 145-192, all 39 domains represented). The new
`board_domains` table (`engine/history.py`, populated from `domain_seeded`
events) and its Standings-page panel exist so this stays checkable against
real accumulated data going forward instead of needing a fresh simulation
every time the question comes up again.

**The "just retreat and wait" strategy some Scott observed live isn't a
bug -- it's a real, measured quirk of the local models, not the engine
math.** Elimination here is all-or-nothing (a duel's loser is eliminated
outright, no partial damage), so every additional duel a player fights is
an independent fresh chance to be knocked out -- pushing repeatedly
compounds risk, retreating doesn't. A 150-show scripted simulation found no
temperament bias in who becomes champion (champions averaged temperament
0.510 vs the full pool's 0.495, actually skewing slightly toward
*aggressive*), which means the scripted push/retreat formula itself is
fine. But a direct live test against the actual local models (fixed
temperament/streak inputs, `OllamaAgent.decide_continue`) found `qwen2.5:3b`
-- the current Standings leader -- answered RETREAT in 8 of 9 trials
regardless of the "aggressive" label it was given, while `llama3.2:latest`
answered PUSH in 9 of 9 regardless of "cautious." The models aren't
reliably following the temperament they're assigned; one of them just
happens to default to a genuinely conservative answer, and that bias is
exactly what this ruleset rewards. Worth knowing, not obviously worth
"fixing" -- there's no single correct temperament-compliance fix, and the
mismatch is arguably a fun, honest artifact of using real small local
models rather than a scripted personality.

**Several presentation bugs fixed together, all from the same testing
pass:**
- The players-remaining counter (`#playersRemaining`) used to live inside
  `#heroStage`, which `startShow()` sets to `display:none` for the entire
  run -- it was updating correctly in the DOM the whole time, just never
  visible once the show started. Moved to sit permanently above the board.
- The spotlight toddle-in animation (`.player-face.toddle-in`) used to get
  applied immediately in `openStage()`, well before the same function's own
  `duelStage.scrollIntoView({behavior:'smooth'})` a few lines later --
  scrollIntoView is an uncontrolled async browser scroll with no completion
  callback, so on a page with real scroll distance the whole 0.9s animation
  could finish while the stage was still off-screen. Now applied ~350ms
  *after* the scroll call instead of before it.
- A live turn's clock animation was a flat 400ms regardless of
  `seconds_used` -- at 400ms (further divided by Fast Forward),
  `animateClock`'s own `setInterval(tick, 100)` only fires once or twice,
  reading as a single instant jump followed by a much longer static hold
  through the rest of the turn's guess-reveal sequence, not a countdown.
  Now scales with `seconds_used` (350ms-1200ms) so longer charges visibly
  take longer.
- The Scramble curtain intermission's total hold was ~3.2s, not the 5s
  Scott asked for; redistributed across the closing/text/reopening beats to
  sum to exactly 5s at 1x speed.
- Challenge Stage badge colors used to tint from each player's own
  `PLAYER_COLORS` board tile -- a 13-color palette picked for whole-board
  variety, not contrast against one specific opponent, so two similar hues
  could land on both badges at once and read as barely different (Scott:
  "we are unable to set the challenge stage player badge background colors
  correctly"). Now a fixed, always-distinct left/right pair
  (`STAGE_LEFT_COLOR`/`STAGE_RIGHT_COLOR`, electric blue vs hot magenta --
  not literally red/green, per Scott's own "no, not that") so which side is
  which reads instantly regardless of who's up. Board tiles and territory
  captures still use each player's real, persistent `playerColor()`.

**The "game freezes after a few shows, clicking around fixes it" report is
almost certainly Chrome's own background-tab timer throttling, not a leak.**
Audited every `setInterval`/`setTimeout` in `web/index.html` for a missing
`clearInterval`/`clearTimeout` -- all of them are properly paired. Audited
DOM growth across repeated shows in the same tab -- `#stage` and
`#chatterFeed` are both explicitly cleared at the top of `startShow()`, so
nothing accumulates unbounded across "New Game" replays. This session's own
automated browser testing (see the disconnect-crash fix above) hit the
exact same symptom directly: `document.hidden` read `true` even while
nominally fronted, and the whole show-playback loop is `sleep()`-driven
(`setTimeout`-based); Chrome clamps timers hard in a backgrounded/unfocused
tab, which produces precisely "looks stuck, then a click on the page
un-sticks it" (a click restores focus/visibility, un-throttling every
pending timer at once). Good news either way: `HistoryRecorder` writes
happen entirely server-side as `run_show()` executes (confirmed earlier via
`curl` completing a full 12-duel show in under a second) -- a frozen-looking
frontend has zero effect on whether a show's data actually got captured.

**The push/retreat model-bias finding is now a tracked scorecard stat, not
a one-off test result.** Scott: "it would look informed to add the 'test
whether the real models show a systematic bias' results as part of the
scorecard." `game.py`'s `continues`/`retreats` events now carry the
player's temperament alongside `model`; `history.py` tracks `pushes`/
`retreats` per player per show (migrated onto the existing `player_stats`
table via `ALTER TABLE`, not a schema rebuild, so the accumulated history
wasn't lost) and `get_stats()` computes a `push_rate` per model. Surfaced
on the Standings page as a "Push %" column plus a note explaining what
compliance should look like (temperament is randomized fresh every show, so
a model actually tracking it should land in the middle and vary -- a rate
parked at 0% or 100% regardless of volume is the signal, not the raw
number). This also means the finding keeps getting more solid as more real
shows are recorded, instead of resting on the original 9-trial direct test.

**Two small usability gaps closed:**
- The end-of-show popup (`#newGameOverlay`) had no way to dismiss it
  short of clicking NEW GAME -- added a close button (`#overlayCloseBtn`)
  that just hides the overlay, champion still visible underneath.
- No way to stop a show once started. Added a Quit/Halt control-panel
  button: aborts the in-flight `fetch('/api/run-show')` via
  `AbortController` (which `server.py`'s existing disconnect handling --
  see the crash fix above -- then stops server-side too, instead of
  burning real Ollama calls for a show nobody's watching anymore), then
  reloads the page rather than hand-resetting every piece of mid-show
  state by hand. Verified directly: aborting mid-show produced no server
  traceback (same graceful path as a closed browser tab) and left the page
  in a clean, fully-reset state.
