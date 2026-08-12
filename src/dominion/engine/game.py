"""Full show orchestration: draw ceremony, the spotlight core loop, duels,
domain inheritance, advantages, the Scramble, burst prizes, and the finale.
Emits everything to an EventLog so the frontend never needs game logic of
its own (Section 10 of the design doc).
"""
from __future__ import annotations
import os
import random
from collections import Counter
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from .board import build_hub_ring, build_pyramid_13, connected_components, find_reconnect_path
from .content import pick_domains, DOMAINS_BY_NAME, Domain
from .models import (
    Player, GameState, KINGDOM_NAME_PARTS_A, KINGDOM_NAME_PARTS_B, PROFESSIONS,
    PROFESSION_DOMAIN_AFFINITY, PROFESSION_DOMAIN_WEIGHT,
)
from ..agents.scripted_agent import ScriptedAgent
from ..agents.llm_agent import LLMAgent, TEXT_MODELS
from ..agents.host_agent import LLMHostAgent, ScriptedHostAgent, host_model_for_backend
from ..agents.commentator_agent import (
    LLMCommentatorAgent, ScriptedCommentatorAgent, commentator_model_for_backend,
)
from ..inference.base import InferenceClient
from ..inference.config import LLAMACPP_MODELS, settings
from ..inference.llamacpp_client import LlamaCppInferenceClient
from ..inference.ollama_client import OllamaInferenceClient
from .duel import run_duel, BASE_CLOCK, QUESTION_CAP

# Set DOMINION_SCRIPTED_ONLY=1 to force plain ScriptedAgent for every player,
# skipping all live Ollama calls -- useful for fast local iteration without
# waiting on real model latency (see docs/ENGINE_NOTES.md).
SCRIPTED_ONLY = bool(os.environ.get("DOMINION_SCRIPTED_ONLY"))

# duel.py's default QUESTION_CAP (30) was tuned for scripted-pace answers
# (roughly 1-4s each). A live answer, once ollama_agent.py stopped charging
# model-load time against the clock, can cost well under a second -- 30
# stopped being a rare safety-valve and became the ordinary way live duels
# ended, usually with most of both 25s clocks still unspent (confirmed: 10
# of 12 duels in one live show hit the cap, only 2 hit an actual timeout).
#
# First attempt at a fix raised this all the way to 300 to try to make
# clock exhaustion the normal ending again. It didn't: a live re-verification
# afterward measured charged think_seconds averaging under 0.1s/attempt, so
# even 300 combined attempts barely dented a 25s clock (duels were still
# ending via the cap, just after a much longer grind) -- and every one of
# those 300 attempts still pays real Ollama round-trip latency independent
# of the tiny charged time, so the fix mostly just made live shows take much
# longer in real wall-clock terms (one full show measured at ~24 minutes)
# without changing which reason duels ended for. Pulled back down to a
# smaller, saner multiple of the scripted default -- see duel.py's docstring
# for the reasoning and for FORCED_PASS_MISS_STREAK, the change that
# actually addresses a duel feeling "stuck" (a live model that never
# volunteers the word PASS could otherwise hammer the same question
# forever; that's a real bug this cap alone never fixed).
LIVE_QUESTION_CAP = 60
EFFECTIVE_QUESTION_CAP = QUESTION_CAP if SCRIPTED_ONLY else LIVE_QUESTION_CAP

# Live trivia used a longer 60s clock at first to give real per-turn latency
# room to breathe, but that read as slow-paced/boring to actually watch, and
# it was defensive anyway -- LLMAgent.attempt_question now caps a live
# call's own wait time to roughly what's left on the player's clock (see
# ollama_agent.py), so a slow call can no longer drag a duel out past when
# the audience already expects it to end. Both modes use the same,
# snappier 25s clock (duel.py's BASE_CLOCK, Revision 18's original call:
# "shortened from 60s to 25s specifically to burn through duels faster").

PLAYER_COUNT = 13
GRAND_PRIZE = 100_000_000
SCRAMBLE_MIN_DUELS = 6
SCRAMBLE_MAX_ACTIVE = 7
BURST_CHECKPOINT_EVERY = 3
# The burst-prize checkpoint's own real cash bonus, awarded to whoever's
# leading in territory every BURST_CHECKPOINT_EVERY duels -- a real bug
# found live (Scott: "money awards aren't being allocated or spoken about
# properly as the stakes"): despite this file's own OLDER comment below
# already describing "burst prizes" as something a player is awarded
# ("on top of... burst prizes and the grand prize"), the emit itself never
# actually carried a bonus field at all -- purely a cosmetic "who's ahead
# right now" checkpoint with a misleadingly prize-shaped name and no real
# prize behind it.
BURST_PRIZE_BONUS = 500_000
# A one-time bonus the first time a player's held territory reaches this
# many tiles, on top of (not instead of) burst prizes and the grand prize.
# Ties for the grand prize are proven structurally impossible (Section 5:
# every duel eliminates exactly one player and the board stays connected,
# so the show always ends with a single sole owner), so there is currently
# no reachable path to a tied finale; a tie split, if the win condition
# ever changes to allow one, would be GRAND_PRIZE * 1.25 each.
MILESTONE_TERRITORY = 5
MILESTONE_BONUS = 1_000_000


def _make_kingdom_name(rng: random.Random, a_pool: list, b_pool: list) -> str:
    # Pop from shuffled, per-show pools rather than an independent rng.choice
    # each time -- Scott noticed players ending up with the same "last name"
    # (the B part), which was near-guaranteed since KINGDOM_NAME_PARTS_B has
    # exactly PLAYER_COUNT entries and independent draws hit the birthday
    # paradox hard at that size. Consuming without replacement (same pattern
    # as origin_domain_pool below) makes every B part unique across the 13
    # players, and every A part unique too since that pool is even bigger.
    a = a_pool.pop() if a_pool else rng.choice(KINGDOM_NAME_PARTS_A)
    b = b_pool.pop() if b_pool else rng.choice(KINGDOM_NAME_PARTS_B)
    return f"{a} {b}"


def model_pool_for_backend(backend: str) -> list:
    """Which model roster a show's players draw from -- Ollama tags for
    the ollama backend, llama.cpp model identifiers (inference/config.py's
    LLAMACPP_MODELS keys, matched against llama-server's --models-dir in
    router mode) for llamacpp. Public: server/app.py's /api/models uses
    this too, so the roster panel and the actual draw pool can never
    silently drift apart."""
    if backend == "llamacpp":
        return list(LLAMACPP_MODELS.keys())
    return TEXT_MODELS.copy()


@asynccontextmanager
async def _default_client_for_backend(backend: str) -> AsyncIterator[InferenceClient]:
    """The InferenceClient run_show falls back to when a caller doesn't
    supply one -- convenient for tests/ad-hoc runs, but wasteful if
    you're about to call run_show repeatedly (the server shares one
    long-lived client per backend instead; see server/app.py's lifespan)."""
    if backend == "llamacpp":
        client: InferenceClient = LlamaCppInferenceClient(
            settings.llamacpp_url,
            max_retry_attempts=settings.retry_max_attempts,
            circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
            circuit_breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )
    else:
        client = OllamaInferenceClient(
            settings.ollama_url,
            max_retry_attempts=settings.retry_max_attempts,
            circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
            circuit_breaker_cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )
    try:
        yield client
    finally:
        await client.aclose()


async def run_show(seed: Optional[int] = None, log=None,
                    client: Optional[InferenceClient] = None,
                    backend: Optional[str] = None,
                    models: Optional[List[str]] = None,
                    stats_snapshot: Optional[Dict[str, Any]] = None) -> dict:
    """Public entrypoint. SCRIPTED_ONLY shows never touch client/backend at
    all. Otherwise: pass your own InferenceClient if you're running more
    than one show (the server does, one shared client per backend);
    omitted, a default client for `backend` (falling back to
    settings.inference_backend) is created and cleanly closed for just
    this one call. models, if given, is the exact roster players draw
    from (the pre-show model picker) instead of model_pool_for_backend's
    fixed TEXT_MODELS/LLAMACPP_MODELS list -- see draft loop below.
    stats_snapshot, if given, is history.py's get_stats() dict, fetched
    once up front by the caller (server/app.py's _stream_show) and handed
    to the Commentator (agents/commentator_agent.py, M11) -- this module
    never touches SQLite itself, matching history.py's own "engine emits
    events, consumers subscribe" boundary."""
    backend = backend or settings.inference_backend
    if SCRIPTED_ONLY or client is not None:
        return await _run_show(seed=seed, log=log, client=client, backend=backend, models=models,
                                stats_snapshot=stats_snapshot)
    async with _default_client_for_backend(backend) as default_client:
        return await _run_show(seed=seed, log=log, client=default_client, backend=backend,
                                models=models, stats_snapshot=stats_snapshot)


async def _run_show(seed: Optional[int] = None, log=None,
                     client: Optional[InferenceClient] = None,
                     backend: str = "ollama",
                     models: Optional[List[str]] = None,
                     stats_snapshot: Optional[Dict[str, Any]] = None) -> dict:
    rng = random.Random(seed)
    events = log

    def emit(type_, **data):
        if events is not None:
            events.emit(type_, **data)

    # --- Pre-production already happened; this is the on-air draw. ---
    domains = pick_domains(PLAYER_COUNT, rng)
    board_adj = build_pyramid_13()  # Revision 15: locked pyramid tessellation

    effective_backend = "scripted" if SCRIPTED_ONLY else backend
    # base_clock lets the frontend sync its own BASE_CLOCK constant from
    # duel.py's single authoritative definition instead of hardcoding a
    # second copy of the number -- fired once, right at the top of the
    # show, well before any duel's clock is ever actually displayed.
    emit("show_start", title="Dominion (Agent vs. Agent)", players=PLAYER_COUNT,
         backend=effective_backend, base_clock=BASE_CLOCK)

    # Phase 1: seed the whole board with domains first, independent of who
    # ends up standing where. Which domain lands on which tile is decided
    # and revealed completely before any contestant is placed, Revision 20:
    # the board is set, then players are dropped onto it, rather than a
    # player and a tile being handed out together in lockstep.
    node_ids = list(range(PLAYER_COUNT))
    rng.shuffle(node_ids)  # which tile each drawn domain lands on
    node_domain: Dict[int, Domain] = {}
    for i, node in enumerate(node_ids):
        node_domain[node] = domains[i]
        emit("domain_seeded", node=node, domain=domains[i].name)
    emit("board_seeded")

    # Phase 2, Revision 21: every player drafts, one at a time, in a
    # randomized draft order (who picks first vs. last is itself part of
    # the drama). Each player chooses freely from whatever domains/tiles
    # are still unclaimed at their turn, and that choice removes the
    # domain from the pool for everyone who picks after them, so a player
    # sixth in the draft order is choosing from a list that already has
    # five domains missing. Real interest-driven picking needs a reasoning
    # agent, Phase 2/3 of the design doc; the scripted stand-in agents used
    # here just pick uniformly at random from what remains, which already
    # produces the mismatches (a persona ending up with a domain that has
    # nothing to do with their flavor) that make the draft fun to watch.
    remaining_nodes = list(node_domain.keys())
    draft_order = list(range(PLAYER_COUNT))
    rng.shuffle(draft_order)

    players: Dict[int, Player] = {}
    owner: Dict[int, int] = {}
    professions_pool = PROFESSIONS.copy()
    rng.shuffle(professions_pool)
    # Each player gets a fixed-for-the-run model backing their live
    # decisions (see agents/llm_agent.py). `models` (the pre-show model
    # picker's selection, threaded in from server/app.py's /api/run-show)
    # takes priority when given and non-empty; otherwise falls back to
    # whichever backend's fixed roster is actually active
    # (model_pool_for_backend), so a llamacpp show's players still get
    # llama.cpp model identifiers by default, never Ollama tags that
    # LlamaCppInferenceClient wouldn't recognize. Shuffled once per show
    # so which players land on which model varies game to game.
    model_pool = list(models) if models else model_pool_for_backend(effective_backend)
    rng.shuffle(model_pool)
    # A player's origin_domain (their real, if amateur, expertise) is drawn
    # independently from the FULL library, not just the 13 domains actually
    # seeded on this show's board -- Scott's design note: "the players do
    # not necessarily have expertise that will come up at all... or the
    # super lucky would land on a matching domain." Previously this line
    # just aliased origin_domain to whatever board domain the player drafted
    # (domain.name below), which meant every single player started the show
    # as a genuine expert in their own held territory, always -- never the
    # rare, lucky coincidence the design called for, and directly why
    # Scott's own interview ("Expertise: X" / "Holds: X") always showed the
    # exact same domain on both lines at the top of every show.
    #
    # Drawn WITHOUT replacement across players (Scott: "its a choice without
    # replacement, so we never get expertise overlap, just like the board
    # works") -- pop() below removes each pick from the shared pool as it's
    # taken, same no-duplicates guarantee the board's remaining_nodes list
    # already gives tile domains. Also weighted by profession fit rather
    # than flat-uniform (Scott's biggest-problem report: "it is very
    # unlikely that a construction worker type is going to choose stamp
    # collecting" -- a fully independent, unweighted draw was exactly how
    # that kind of nonsense pairing happened). Still a real draw, not a
    # rigid lookup: an earlier pick in the draft order can take a domain
    # that would have been a later player's ideal fit too, same scarcity
    # dynamic as the board.
    origin_domain_pool = list(DOMAINS_BY_NAME.keys())
    # Shuffled, per-show kingdom-name pools -- see _make_kingdom_name.
    kingdom_a_pool = list(KINGDOM_NAME_PARTS_A)
    kingdom_b_pool = list(KINGDOM_NAME_PARTS_B)
    rng.shuffle(kingdom_a_pool)
    rng.shuffle(kingdom_b_pool)

    for pick_number, pid in enumerate(draft_order, start=1):
        node = rng.choice(remaining_nodes)
        remaining_nodes.remove(node)
        domain = node_domain[node]
        profession = professions_pool[pid % len(professions_pool)]

        affinity = set(PROFESSION_DOMAIN_AFFINITY.get(profession, ()))
        weights = [PROFESSION_DOMAIN_WEIGHT if d in affinity else 1 for d in origin_domain_pool]
        origin_domain = rng.choices(origin_domain_pool, weights=weights, k=1)[0]
        origin_domain_pool.remove(origin_domain)

        player = Player(
            id=pid,
            domain=domain.name,
            origin_domain=origin_domain,
            kingdom_name=_make_kingdom_name(rng, kingdom_a_pool, kingdom_b_pool),
            profession=profession,
            territory={node},
            # 0.1-0.9 rather than the full 0-1 range, so nobody plays as a
            # pure coin-flip robot or an absolute maniac -- everyone still
            # has some real variance run to run, just bounded.
            temperament=rng.uniform(0.1, 0.9),
            model=model_pool[(pick_number - 1) % len(model_pool)],
        )
        players[pid] = player
        owner[node] = pid
        emit("draw_assignment", player_id=pid, domain=domain.name,
             origin_domain=player.origin_domain,
             kingdom_name=player.kingdom_name, profession=player.profession,
             node=node, on_stage=(node == 0), pick_number=pick_number,
             remaining_after=len(remaining_nodes), temperament=player.temperament,
             temperament_label=player.temperament_label(), model=player.model)

    if SCRIPTED_ONLY:
        agents = {pid: ScriptedAgent(rng) for pid in players}
        host_agent = ScriptedHostAgent(rng)
        commentator_agent = ScriptedCommentatorAgent(rng)
    else:
        assert client is not None  # guaranteed by run_show's wrapper above
        agents = {pid: LLMAgent(rng, model=players[pid].model, client=client) for pid in players}
        host_agent = LLMHostAgent(
            rng, model=host_model_for_backend(effective_backend), client=client)
        commentator_agent = LLMCommentatorAgent(
            rng, model=commentator_model_for_backend(effective_backend), client=client,
            backend=effective_backend, stats_snapshot=stats_snapshot)
    game = GameState(players=players, owner=owner, board_adj=board_adj,
                      active_ids=set(players.keys()))
    used_questions: Set[Tuple[str, str]] = set()  # (domain_name, prompt) pairs, shared for the whole show

    emit("draw_complete")

    # --- Main spotlight loop ---
    while game.sole_owner() is None:
        if game.spotlight is None:
            pool = [pid for pid in game.active_ids if pid != game.excluded_from_pick] or list(game.active_ids)
            game.spotlight = rng.choice(pool)
            game.excluded_from_pick = None
            emit("spotlight_chosen", player_id=game.spotlight)

        active_pid = game.spotlight
        active_player = players[active_pid]
        agent = agents[active_pid]

        # #13 Threepeat Drive: a guaranteed non-adjacent target set two
        # steps ago (the push_streak%3==0 checkpoint below) overrides the
        # normal choice entirely -- there's nothing to decide, so no
        # agent_thinking/live call either.
        if game.pending_threepeat_target is not None:
            target_id: Optional[int] = game.pending_threepeat_target
            game.pending_threepeat_target = None
        else:
            emit("agent_thinking", player_id=active_pid, model=active_player.model, decision="target")
            target_id = await agent.choose_target(active_player, game)
        if target_id is None:
            # This spotlighted player has no adjacent opponents right now
            # (can happen once the board fragments into several owners,
            # especially right after the Scramble reshuffles it). The board
            # graph itself stays connected the whole game, so with more than
            # one active player, some active player always has a legal
            # target even when this one doesn't. Hand the spotlight to one
            # of those instead of ending the show early.
            candidates = [pid for pid in game.active_ids if game.adjacent_opponents(pid)]
            if not candidates:
                # Should be unreachable on a connected board with >1 active
                # player; guard here rather than silently loop forever.
                break
            game.spotlight = rng.choice(candidates)
            game.excluded_from_pick = None
            emit("spotlight_chosen", player_id=game.spotlight)
            continue

        defender = players[target_id]
        challenger_bonus = active_player.time_bonus_banked
        # The territory actually at stake in every duel is always the
        # DEFENDER's (territory transfer below only ever moves the LOSER's
        # land to the winner, and only the defender's land is what's being
        # contested here), so the trivia domain has to be the defender's --
        # Revision 20's brief attempt at coin-flipping this instead was
        # reverted (Scott: "it seems the challenges are taking place on the
        # challenger's domain. that shouldn't be possible since they want to
        # capture a different territory/domain"). Revision 21 then tried a
        # compensating clock handicap for the challenger instead (see
        # README.md's home-turf section for the real mechanism behind the
        # measured 56.5%/43.5% split -- domain_record/_domain_familiarity_line,
        # not this domain choice). Revision 22 dropped that handicap too:
        # Scott, after watching more shows with the live-vs-fallback answer
        # counter, "honestly, I don't think they need that home-field
        # advantage." The structural edge is real and documented, just no
        # longer something this engine tries to correct for.
        tested_domain = defender.domain
        emit("challenge_declared", challenger_id=active_pid, defender_id=target_id,
             tested_domain=tested_domain, challenger_using_bonus=challenger_bonus,
             defender_using_bonus=False, base_clock=BASE_CLOCK)
        # M32 Fix 2: snapshotted here, BEFORE run_duel/territory transfer
        # touches anything, so announce_duel_result's is_upset check
        # below can compare each side's territory as it genuinely stood
        # going into this duel -- the same definition celebrateDuelWinner's
        # old client-side isUpset calc used, just computed once,
        # authoritatively, here instead of duplicated in JS.
        challenger_territory_pre = len(active_player.territory)
        defender_territory_pre = len(defender.territory)

        # The Host, as a real agent (M9) -- one announcement, generated
        # once, that's both what's actually spoken/shown on screen (the
        # host_line event below) AND what both players' own
        # intro_line_combined calls react to as host_announcement. Before
        # this, those were two different, disconnected texts -- see
        # agents/host_agent.py's module docstring.
        #
        # M32 Fix 1: streak/territory context passed in here too, folded
        # into this ONE call instead of a separate scripted
        # preduelHostLine firing a second, disconnected line right
        # before this one (the Host was speaking twice per duel through
        # two different code paths -- see host_agent.py's
        # announce_challenge for the ported tiered logic).
        # Scott: "do we have any built in pauses? i see a lot of
        # no-activity then a burst." Every Player decision already gets
        # a visible agent_thinking indicator (banner + roster glow) for
        # however long its live call takes, but the Host's and
        # Commentator's OWN live lines had no equivalent -- a slow call
        # here read as pure dead air right up until the line (and
        # whatever chain of events follows it) landed all at once. This
        # is that same indicator, generalized past just Player decisions.
        emit("host_thinking", role="host")
        host_line = await host_agent.announce_challenge(
            active_player, defender, tested_domain,
            challenger_streak=active_player.push_streak, defender_streak=defender.push_streak,
            challenger_territory=len(active_player.territory),
            defender_territory=len(defender.territory))
        emit("host_line", moment="challenge", text=host_line, live=not SCRIPTED_ONLY)

        # The Commentator (M11) -- a second AI voice, a beat after the
        # Host's own announcement, not a replacement for it. Grounded in
        # real cross-show history when stats_snapshot is available (a
        # live show); ScriptedCommentatorAgent's canned pool otherwise.
        emit("host_thinking", role="commentator")
        commentator_line = await commentator_agent.react_to_matchup(
            active_player, defender, tested_domain)
        emit("commentator_line", moment="matchup", text=commentator_line, live=not SCRIPTED_ONLY)

        # The pre-duel interview: ONE combined round per side now -- used to
        # be three separate rounds (origin domain, tested domain, read on
        # the opponent), per Scott's original ask for "several back and
        # forth between host and player so the models are warm to their own
        # domain and knowledge of the domain they are dueling on," later
        # extended to a third round about the opponent specifically. Scott
        # then found three separate rounds read like disconnected exchanges
        # rather than the model actually answering one real question:
        # "it doesn't seem like the agents are hearing each other in
        # conversation... player 1, I see you are a cautious expert in
        # flowers and gardens, how do you feel about competing with player 2
        # on the subject of Farm Animals. Like that kinda." One combined
        # call now (intro_line_combined) covers all three things at once --
        # own expertise, tonight's actual domain, and the named opponent --
        # matching the host's single loaded question in index.html's
        # hostPreduelQuestion. Two full generation calls per duel now
        # instead of six, one per side, still giving the DEFENDER the same
        # warm-up head start choose_target already gives the CHALLENGER for
        # free (an untimed call) before their own first timed trivia turn.
        #
        # Sequential, not simultaneous, and challenger first then defender,
        # same as before -- reads as one contestant's answer finishing
        # before the next starts. The defender's call still gets fed the
        # challenger's actual reply as opponent_line, so it's a real
        # two-way exchange, not two isolated monologues that happen to air
        # back to back.
        defender_agent = agents[target_id]

        emit("agent_thinking", player_id=active_pid, model=active_player.model, decision="intro")
        challenger_line = await agent.intro_line_combined(
            active_player, tested_domain, defender, host_announcement=host_line, game=game)
        emit("pre_duel_intro", player_id=active_pid, role="challenger",
             model=active_player.model, text=challenger_line)

        emit("agent_thinking", player_id=target_id, model=defender.model, decision="intro")
        defender_line = await defender_agent.intro_line_combined(
            defender, tested_domain, active_player, host_announcement=host_line,
            opponent_line=challenger_line, game=game)
        emit("pre_duel_intro", player_id=target_id, role="defender",
             model=defender.model, text=defender_line)

        # Emitted live, turn by turn, as run_duel computes each one -- not
        # batched up and replayed after the whole duel resolves. With a live
        # agent a single turn can itself take real seconds, so the frontend
        # needs to see each one land as it actually happens, the same way
        # agent_thinking/decision events already do outside a duel.
        turn_count = [0]

        def emit_turn(turn: dict) -> None:
            emit("duel_turn", turn_index=turn_count[0], challenger_id=active_pid,
                 defender_id=target_id, player_id=turn["player_id"],
                 domain=turn["domain"], prompt=turn["prompt"],
                 answer=turn["answer"], guess=turn["guess"], outcome=turn["outcome"],
                 correct=turn["correct"], seconds_used=turn["seconds_used"],
                 clock_remaining=turn["clock_remaining"],
                 distractors=turn["distractors"], live=turn["live"],
                 lucky_blurt=turn["lucky_blurt"],
                 raw_latency_seconds=turn["raw_latency_seconds"])
            turn_count[0] += 1

        result = await run_duel(
            active_player, defender, DOMAINS_BY_NAME[tested_domain],
            agents, rng, challenger_bonus=challenger_bonus,
            used_questions=used_questions, base_clock=BASE_CLOCK,
            question_cap=EFFECTIVE_QUESTION_CAP, on_turn=emit_turn,
            on_before_attempt=lambda pid, model: emit(
                "agent_thinking", player_id=pid, model=model, decision="attempt"),
            game=game)
        if challenger_bonus:
            active_player.time_bonus_banked = False

        # Real accumulated in-show experience per player per domain (Scott:
        # "the players get smarter as they really play") -- feeds
        # LLMAgent.attempt_question's domain-familiarity line on this
        # player's future attempts in the SAME domain later in the show.
        game.record_duel_turns(result.turns_log)

        game.duel_count += 1
        winner_id, loser_id = result.winner_id, result.loser_id
        winner, loser = players[winner_id], players[loser_id]

        # Head-to-head record for this pair, this show -- feeds
        # choose_target/choose_tax_target's per-candidate lines and
        # intro_line_combined's rivalry banter on any future rematch.
        game.record_duel_result(winner_id, loser_id)

        # M32 Fix 2: is_upset/is_close computed here, before territory
        # transfer below touches anything, from the PRE-duel snapshot
        # taken right after challenge_declared -- same definitions
        # celebrateDuelWinner's old client-side JS used (loser held >=2
        # more tiles going in = upset; clocks within 3s of each other =
        # close), just computed once, authoritatively, server-side, and
        # carried on duel_result below so the frontend no longer needs
        # its own parallel calculation.
        winner_territory_pre = (
            challenger_territory_pre if winner_id == active_pid else defender_territory_pre)
        loser_territory_pre = (
            defender_territory_pre if winner_id == active_pid else challenger_territory_pre)
        is_upset = (loser_territory_pre - winner_territory_pre) >= 2
        is_close = not is_upset and abs(
            result.clocks_remaining.get(winner_id, 0) - result.clocks_remaining.get(loser_id, 0)) < 3

        # #13 Threepeat Drive Home: the split/transfer logic just below
        # has always assumed winner and loser were already adjacent
        # (every ordinary duel -- choose_target only ever offers
        # adjacent_opponents), so it never actually checks that
        # assumption; it just ships whichever piece of the loser's
        # territory happens to touch the winner's. A genuinely
        # non-adjacent win (the Drive) would otherwise transfer NOTHING
        # to the winner -- none of the loser's territory would touch
        # theirs, so every piece would scatter to bystanders instead of
        # rewarding the win at all.
        #
        # First choice: forcibly annex the shortest bridge of
        # bystander-owned tiles (find_reconnect_path, board.py) so the
        # conquered land becomes genuinely, physically reconnected to the
        # winner's kingdom -- "reconnect," not just a rule exception.
        # But: EARLY in a show almost every bystander still owns only
        # their single starting tile, since territory consolidation takes
        # many duels -- meaning a genuinely safe bridge (one that doesn't
        # fully strip some uninvolved bystander down to zero tiles, a
        # collateral elimination with no duel, no agency, no drama, and
        # -- worse -- one that breaks this show's fundamental "always
        # exactly PLAYER_COUNT - 1 duels" shape) is very often simply
        # unavailable, not just a rare edge case. Rather than accept that
        # cost, fall back to treating the conquest as a legitimate
        # disconnected outpost instead (outpost=True below bypasses the
        # split loop's normal touching requirement entirely) -- no
        # bystander tiles change hands at all in that case.
        outpost = False
        if not game.players_adjacent(winner_id, loser_id):
            excluded: Set[int] = set()
            bridge_tiles: List[int] = []
            for _ in range(len(game.board_adj)):
                trial_adj = {node: (neighbors - excluded) for node, neighbors in game.board_adj.items()
                             if node not in excluded}
                trial_path = find_reconnect_path(trial_adj, winner.territory, loser.territory)
                if not trial_path:
                    break
                # "Risky" means taking every tile THIS path asks of a
                # given bystander would fully empty them out -- not just
                # a single-tile owner; a 2-tile bystander who'd lose BOTH
                # of their tiles to the same path is exactly as wiped out.
                bystander_contribution = Counter(
                    game.owner[r] for r in trial_path if game.owner[r] not in (winner_id, loser_id))
                newly_risky = {
                    r for r in trial_path if game.owner[r] not in (winner_id, loser_id)
                    and bystander_contribution[game.owner[r]] >= len(players[game.owner[r]].territory)
                }
                if not newly_risky:
                    bridge_tiles = trial_path
                    break
                excluded |= newly_risky
            if not bridge_tiles:
                outpost = True
            else:
                bridged_from_ids = set()
                for r in bridge_tiles:
                    bystander_id = game.owner[r]
                    if bystander_id not in (winner_id, loser_id):
                        players[bystander_id].territory.discard(r)
                        bridged_from_ids.add(bystander_id)
                    winner.territory.add(r)
                    game.owner[r] = winner_id
                emit("forced_reconnect", player_id=winner_id, tiles=bridge_tiles,
                     from_ids=list(bridged_from_ids))
                # A safe bridge only ever excludes single-TILE owners, but
                # a multi-tile bystander who loses one interior tile can
                # still end up split into two pieces -- the same
                # "exclave" problem the loser-transfer loop below already
                # solves for the LOSER, just now possible for a bystander
                # who wasn't even in this duel. Keep their largest piece,
                # redistribute anything smaller the same way (winner if
                # touching, else whichever other active player borders
                # it, else the winner as a last resort). A full wipe-out
                # should no longer be reachable given the exclusion
                # search above, but the check stays as a cheap safety net.
                for bystander_id in bridged_from_ids:
                    bystander = players[bystander_id]
                    if not bystander.territory:
                        bystander.active = False
                        game.active_ids.discard(bystander_id)
                        emit("player_eliminated_by_reconnect", player_id=bystander_id)
                        game.remember(
                            f"{bystander.kingdom_name} lost their last foothold when "
                            f"{winner.kingdom_name}'s Threepeat Drive cut straight through it, "
                            f"and is out of the game."
                        )
                        continue
                    bystander_pieces = connected_components(bystander.territory, game.board_adj)
                    if len(bystander_pieces) <= 1:
                        continue
                    main_piece = max(bystander_pieces, key=len)
                    for piece in bystander_pieces:
                        if piece is main_piece:
                            continue
                        bystander.territory -= piece
                        if any(game.board_adj[r] & winner.territory for r in piece):
                            target_owner_id = winner_id
                        else:
                            piece_candidates = [
                                pid for pid in game.active_ids
                                if pid not in (winner_id, bystander_id)
                                and any(game.board_adj[r] & players[pid].territory for r in piece)
                            ]
                            target_owner_id = rng.choice(piece_candidates) if piece_candidates else winner_id
                        players[target_owner_id].territory |= piece
                        for r in piece:
                            game.owner[r] = target_owner_id

        # Territory transfer -- split rather than blind merge. The loser's
        # territory is split into connected pieces (per the current board
        # graph); only the piece(s) actually touching the winner's own
        # territory transfer to the winner (or, on a Drive outpost win,
        # EVERY piece -- outpost bypasses the touching requirement
        # entirely, since the whole point of that duel was claiming this
        # land as a new, deliberately disconnected foothold). Any OTHER
        # disconnected piece (possible if the loser had themselves picked
        # up territory through an earlier chain of wins) is reassigned to
        # whichever other still-active player borders it instead. A
        # player should never end up holding a disconnected "exclave"
        # just from winning one fight -- except a Drive's own outpost,
        # which is deliberately exactly that.
        loser_components = connected_components(loser.territory, game.board_adj)
        winner_gain = set()
        reassignments = []  # (new_owner_id, tiles) for any leftover piece
        for comp in loser_components:
            if outpost or any(game.board_adj[r] & winner.territory for r in comp):
                winner_gain |= comp
                continue
            other_candidates = [
                pid for pid in game.active_ids
                if pid not in (winner_id, loser_id)
                and any(game.board_adj[r] & players[pid].territory for r in comp)
            ]
            if other_candidates:
                new_owner_id = rng.choice(other_candidates)
                players[new_owner_id].territory |= comp
                for r in comp:
                    game.owner[r] = new_owner_id
                reassignments.append((new_owner_id, list(comp)))
            else:
                # Should be unreachable -- the board stays fully connected
                # and every tile is always owned by an active player, so
                # some other active player must border any leftover piece.
                # Fall back to the winner rather than leaving tiles
                # ownerless if this somehow ever happens.
                winner_gain |= comp

        territory_gained = list(winner_gain)
        winner.territory |= winner_gain
        for r in winner_gain:
            game.owner[r] = winner_id

        was_challenger_win = (winner_id == active_pid)
        if was_challenger_win:
            winner.domain = loser.domain  # asymmetric inheritance: challenger wins -> inherits
            # the domain they just conquered and must now defend it everywhere
            # they hold ground. A winning defender keeps their own domain.

        loser.active = False
        game.active_ids.discard(loser_id)

        # M32 Fix 2: the Host's own reaction to this duel -- win/upset/
        # close framing blended with a genuine goodbye to the loser in
        # one call (agents/host_agent.py's announce_duel_result),
        # replacing 4 separate scripted pools (winHostLine/
        # winCloseHostLine/winUpsetHostLine/ELIMINATION_LINES) that used
        # to fire across two different beats for what's really one
        # moment. Computed here, before the same final_elimination gate
        # M12's exit_interview already uses below.
        emit("host_thinking", role="host")
        duel_result_host_line = await host_agent.announce_duel_result(
            winner, loser, tested_domain, is_upset, is_close)

        # M12: the loser's own last word, live on air, before the send-off
        # plays -- right now elimination is otherwise silent past a
        # scripted goodbye template. Emitted before duel_result below so
        # the frontend can air it while this player is still visibly on
        # stage, ahead of the toddle-off exit animation duel_result's own
        # handler triggers.
        #
        # Scott caught a real sequencing bug here: on the show's very
        # last duel, this fired before the Host's own finale
        # announcement even existed -- the loser's last word landed
        # before the show had formally ended. "At the end of a game, the
        # host should speak first, ending the game formally, then the
        # exit interview." final_elimination checks sole_owner() right
        # here, with loser.active already False above, so it correctly
        # reflects whether THIS elimination is the one that ends the
        # show. Every other (non-final) elimination emits both lines
        # immediately, unchanged; only the very last one holds BOTH
        # back, until after the finale host_line/finale events below --
        # the Host's duel-result line goes first, then the loser's own
        # exit interview, preserving the same "Host formally closes,
        # then the loser's own word" order M12 already established.
        loser_agent = agents[loser_id]
        emit("agent_thinking", player_id=loser_id, model=loser.model, decision="exit_interview")
        exit_line = await loser_agent.exit_interview(loser, winner, tested_domain)
        final_elimination = game.sole_owner() is not None
        if not final_elimination:
            emit("host_line", moment="duel_result", text=duel_result_host_line,
                 live=not SCRIPTED_ONLY)
            emit("exit_interview", player_id=loser_id, text=exit_line, live=not SCRIPTED_ONLY)

        # Streak tracking: every win extends it, regardless of whether the
        # winner was pushing as challenger or successfully defending.
        # Revision 12 originally only counted challenger wins and reset the
        # whole counter on a defensive win -- which meant a real, visible
        # 4-in-a-row streak could silently drop to 0 mid-run the moment one
        # of those wins happened to be defensive, hiding both the streak
        # badge and the advantage bonus even though the audience just
        # watched four straight wins. Counting any win matches what's
        # actually on screen. Moved ahead of the duel_result emit below so
        # that event can carry the winner's up-to-date count as of THIS
        # duel, for the frontend's badge display, rather than the count
        # from before it.
        winner.push_streak += 1

        emit("duel_result", winner_id=winner_id, loser_id=loser_id, reason=result.reason,
             winner_domain_after=winner.domain, questions_seen=result.questions_seen,
             clocks_remaining=result.clocks_remaining, turns=len(result.turns_log),
             territory_gained=territory_gained, winner_streak=winner.push_streak,
             is_upset=is_upset, is_close=is_close)

        # Show-wide memory (Scott: "everyone, all agents, will be more and
        # more informed as the game proceeds") -- a plain-English fact any
        # live model's prompt can reference later via GameState.memory, not
        # just this player's own personal record. tested_domain is read here
        # (not winner.domain, which may have just been reassigned to the
        # loser's old domain on a challenger win) specifically because it's
        # the domain this exact duel was actually fought on -- always the
        # defender's domain, per the revert above -- regardless of who ends
        # up owning it afterward.
        #
        # Every duel loss is a full elimination in this format (single loss
        # and you're out -- see loser.active = False / active_ids.discard
        # above), but "defeated... in a duel" alone doesn't say that
        # outright. Scott: eliminated players seem to still get referenced
        # by other agents as if they're still around -- choose_target
        # already can't literally pick an eliminated player again (it only
        # draws from active_ids), so this was a narrative-clarity gap, not
        # a game-logic bug: a live model reading its own history block has
        # no explicit signal that "defeated" here means gone for good, not
        # just a lost skirmish in a show with many duels. Spelling it out
        # plainly removes the ambiguity.
        game.remember(
            f"{winner.kingdom_name} defeated {loser.kingdom_name} in a {tested_domain} duel "
            f"({result.reason.replace('_', ' ')}). {loser.kingdom_name} is eliminated and out "
            f"of the game."
        )

        for reassigned_id, tiles in reassignments:
            emit("territory_reassigned", to_id=reassigned_id, tiles=tiles)

        # Territory milestone: a one-time bonus the first time a player's
        # held territory reaches MILESTONE_TERRITORY tiles. Checked for the
        # winner AND for anyone who just received a reassigned exclave,
        # since either could be what pushes someone past the threshold.
        for pid in [winner_id] + [rid for rid, _ in reassignments]:
            p = players[pid]
            if not p.milestone_paid and len(p.territory) >= MILESTONE_TERRITORY:
                p.milestone_paid = True
                emit("territory_milestone", player_id=pid, territory=len(p.territory),
                     bonus=MILESTONE_BONUS)

        if winner.push_streak > 0 and winner.push_streak % 3 == 0:
            emit("advantage_earned", player_id=winner_id, streak=winner.push_streak)
            emit("host_thinking", role="commentator")
            advantage_comment = await commentator_agent.react_to_advantage(winner, winner.push_streak)
            emit("commentator_line", moment="advantage", text=advantage_comment, live=not SCRIPTED_ONLY)
            if rng.random() < 0.8:
                emit("agent_thinking", player_id=winner_id, model=winner.model, decision="tax")
                target_pid = await agents[winner_id].choose_tax_target(winner, game)
                if target_pid is not None:
                    target_player = players[target_pid]
                    winner.domain, target_player.domain = target_player.domain, winner.domain
                    emit("domain_tax_applied", from_id=winner_id, to_id=target_pid,
                         winner_new_domain=winner.domain, target_new_domain=target_player.domain)
            else:
                winner.time_bonus_banked = True
                emit("time_bonus_banked", player_id=winner_id)

            # #13 Threepeat Drive: a SEPARATE effect at this same
            # every-3rd-win checkpoint, additive to the tax/time-bonus
            # roll above, not a replacement for it -- a guaranteed
            # challenge against a genuinely non-adjacent opponent (the
            # board is a fixed, real graph with meaningful non-adjacency,
            # engine/board.py's build_pyramid_13). Silently doesn't
            # trigger if no non-adjacent active opponent exists right now
            # (late game, everyone's already touching) -- the tax/bonus
            # roll above still happened either way.
            non_adjacent = [pid for pid in game.active_ids
                             if pid != winner_id and pid not in game.adjacent_opponents(winner_id)]
            if non_adjacent:
                threepeat_target_id = rng.choice(non_adjacent)
                game.pending_threepeat_target = threepeat_target_id
                emit("threepeat_drive", player_id=winner_id, target_id=threepeat_target_id,
                     streak=winner.push_streak)
                emit("host_thinking", role="commentator")
                drive_comment = await commentator_agent.react_to_threepeat_drive(
                    winner, players[threepeat_target_id])
                emit("commentator_line", moment="threepeat_drive", text=drive_comment,
                     live=not SCRIPTED_ONLY)

        # Burst prize checkpoint.
        if game.duel_count % BURST_CHECKPOINT_EVERY == 0 and game.sole_owner() is None:
            leader_id = max(game.active_ids, key=lambda pid: len(players[pid].territory))
            emit("burst_prize", duel_count=game.duel_count, leader_id=leader_id,
                 leader_domain=players[leader_id].domain, territory=len(players[leader_id].territory),
                 bonus=BURST_PRIZE_BONUS)
            game.burst_prizes.append({"duel_count": game.duel_count, "leader_id": leader_id})

        # The Scramble.
        if (not game.scrambled and game.duel_count >= SCRAMBLE_MIN_DUELS
                and len(game.active_ids) <= SCRAMBLE_MAX_ACTIVE and len(game.active_ids) > 1):
            _apply_scramble(game)
            game.scrambled = True
            emit("scramble", active_players=len(game.active_ids),
                 board_size=len(game.owner), new_owner=dict(game.owner))
            emit("host_thinking", role="commentator")
            scramble_comment = await commentator_agent.react_to_scramble(len(game.active_ids))
            emit("commentator_line", moment="scramble", text=scramble_comment, live=not SCRIPTED_ONLY)
            game.remember(f"The board was scrambled -- {len(game.active_ids)} players remain, territory reshuffled.")

        if game.sole_owner() is not None:
            break

        # Winner's choice: continue pushing or retreat. Now carries a
        # spoken reason (Scott: this "is being lost in the action" -- the
        # host announces it live, with the player briefly explaining their
        # choice, a side effect being a bit more model warm-up too) --
        # decide_continue returns (keep_going, reason) for every agent now.
        winner_agent = agents[winner_id]
        # #13 Threepeat Drive: a target was just set above (this same
        # winner, this same checkpoint) -- there's no real push/retreat
        # choice to make, so decide_continue is skipped entirely rather
        # than asked and overridden. push_streak is NOT reset (matches
        # the existing reset-only-on-retreat rule -- they're continuing).
        is_forced_drive = game.pending_threepeat_target is not None
        if is_forced_drive:
            keep_going, continue_reason = True, "answering the call of the Threepeat Drive."
            game.spotlight = winner_id
            emit("continues", player_id=winner_id, model=winner.model, reason=continue_reason,
                 temperament=winner.temperament, temperament_label=winner.temperament_label(),
                 forced=True)
        else:
            emit("agent_thinking", player_id=winner_id, model=winner.model, decision="continue")
            keep_going, continue_reason = await winner_agent.decide_continue(winner, game)
            if keep_going and game.adjacent_opponents(winner_id):
                game.spotlight = winner_id
                emit("continues", player_id=winner_id, model=winner.model, reason=continue_reason,
                     temperament=winner.temperament, temperament_label=winner.temperament_label())
            else:
                winner.push_streak = 0
                game.spotlight = None
                game.excluded_from_pick = winner_id
                emit("retreats", player_id=winner_id, model=winner.model, reason=continue_reason,
                     temperament=winner.temperament, temperament_label=winner.temperament_label())
        # M32 Fix 3: emitted AFTER continues/retreats, not before -- the
        # frontend's own handler for that event plays the buildup
        # question and the player's real stated reason FIRST
        # (hostContinueBuildup, then the interview banner), and the
        # Host's reaction has to land after that setup, not before it.
        # Reacts to the EFFECTIVE outcome (whether the winner actually
        # keeps going), not the raw keep_going flag alone -- keep_going
        # =True with no adjacent opponents left still ends in a retreat
        # on the board, and the Host's line should match what viewers
        # actually see happen.
        # A forced Drive always effectively pushes, by design, regardless
        # of ordinary adjacency (the whole point is the target ISN'T
        # adjacent) -- the normal effective-outcome check below only
        # applies to a real decide_continue call.
        effective_keep_going = is_forced_drive or (keep_going and bool(game.adjacent_opponents(winner_id)))
        emit("host_thinking", role="host")
        continue_reaction_line = await host_agent.announce_continue_decision(
            winner, effective_keep_going, continue_reason)
        emit("host_line", moment="continue_reaction", text=continue_reaction_line,
             live=not SCRIPTED_ONLY)

    champion_id = game.sole_owner()
    # The while loop above only exits once sole_owner() is no longer None
    # -- guaranteed non-None here, mypy just can't trace that across the
    # loop boundary.
    assert champion_id is not None
    champion = players[champion_id]
    emit("host_thinking", role="host")
    finale_host_line = await host_agent.announce_finale(champion, game.duel_count, GRAND_PRIZE)
    emit("host_line", moment="finale", text=finale_host_line, live=not SCRIPTED_ONLY)
    # The show's very last exit interview, held back from its normal spot
    # (see the loop above) until after the Host has formally ended the
    # show -- final_elimination/loser_id/exit_line are always set here:
    # the loop only ever exits via the sole_owner() break, which only
    # happens right after that last elimination sets them. Emitted BEFORE
    # the finale event below, not after: finale's own frontend handler
    # fires the victory jingle/crowd applause/confetti fire-and-forget
    # (not awaited), so an exit_interview emitted after it would have its
    # own spoken line audibly overlapping that music -- exactly the "game
    # music... plays during post-game interview" Scott reported. This
    # order gives a clean beat sequence: Host's formal close, then the
    # loser's last word, then the actual celebration.
    if final_elimination:
        emit("host_line", moment="duel_result", text=duel_result_host_line,
             live=not SCRIPTED_ONLY)
        emit("exit_interview", player_id=loser_id, text=exit_line, live=not SCRIPTED_ONLY)
    emit("finale", champion_id=champion_id, champion_domain=champion.domain,
         champion_kingdom=champion.kingdom_name, champion_profession=champion.profession,
         prize=GRAND_PRIZE, total_duels=game.duel_count)

    return {
        "champion_id": champion_id,
        "champion": {
            "domain": champion.domain, "kingdom_name": champion.kingdom_name,
            "profession": champion.profession,
        },
        "total_duels": game.duel_count,
        "prize": GRAND_PRIZE,
    }


def _apply_scramble(game: GameState) -> None:
    """Redraw the board smaller around current ownership. Phase 1 is
    text-only, so 'nearest new position' is simplified to one consolidated
    node per active player rather than real spatial nearest-neighbor
    placement, which needs the imagery/visual layer from Phase 2."""
    active = list(game.active_ids)
    new_adj = build_hub_ring(len(active))
    new_owner = {}
    for idx, pid in enumerate(active):
        game.players[pid].territory = {idx}
        new_owner[idx] = pid
    game.board_adj = new_adj
    game.owner = new_owner
