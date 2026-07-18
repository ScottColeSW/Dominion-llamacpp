"""Full show orchestration: draw ceremony, the spotlight core loop, duels,
domain inheritance, advantages, the Scramble, burst prizes, and the finale.
Emits everything to an EventLog so the frontend never needs game logic of
its own (Section 10 of the design doc).
"""
from __future__ import annotations
import os
import random
from typing import Dict, List, Optional

from .board import build_hub_ring, build_pyramid_13, connected_components
from .content import pick_domains, DOMAINS_BY_NAME, Domain
from .models import Player, GameState, KINGDOM_NAME_PARTS_A, KINGDOM_NAME_PARTS_B, PROFESSIONS
from .agents import ScriptedAgent
from .ollama_agent import OllamaAgent, TEXT_MODELS
from .duel import run_duel

# Set DOMINION_SCRIPTED_ONLY=1 to force plain ScriptedAgent for every player,
# skipping all live Ollama calls -- useful for fast local iteration without
# waiting on real model latency (see prototype/README.md).
SCRIPTED_ONLY = bool(os.environ.get("DOMINION_SCRIPTED_ONLY"))

PLAYER_COUNT = 13
GRAND_PRIZE = 100_000_000
SCRAMBLE_MIN_DUELS = 6
SCRAMBLE_MAX_ACTIVE = 7
BURST_CHECKPOINT_EVERY = 3
# A one-time bonus the first time a player's held territory reaches this
# many tiles, on top of (not instead of) burst prizes and the grand prize.
# Ties for the grand prize are proven structurally impossible (Section 5:
# every duel eliminates exactly one player and the board stays connected,
# so the show always ends with a single sole owner), so there is currently
# no reachable path to a tied finale; a tie split, if the win condition
# ever changes to allow one, would be GRAND_PRIZE * 1.25 each.
MILESTONE_TERRITORY = 5
MILESTONE_BONUS = 1_000_000


def _make_kingdom_name(rng: random.Random) -> str:
    return f"{rng.choice(KINGDOM_NAME_PARTS_A)} {rng.choice(KINGDOM_NAME_PARTS_B)}"


def run_show(seed: Optional[int] = None, log=None) -> dict:
    rng = random.Random(seed)
    events = log

    def emit(type_, **data):
        if events is not None:
            events.emit(type_, **data)

    # --- Pre-production already happened; this is the on-air draw. ---
    domains = pick_domains(PLAYER_COUNT, rng)
    board_adj = build_pyramid_13()  # Revision 15: locked pyramid tessellation

    emit("show_start", title="Dominion (Agent vs. Agent)", players=PLAYER_COUNT)

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
    # Each player gets a fixed-for-the-run local model backing their live
    # decisions (see ollama_agent.py); shuffled once per show so which
    # players land on which model varies game to game.
    model_pool = TEXT_MODELS.copy()
    rng.shuffle(model_pool)

    for pick_number, pid in enumerate(draft_order, start=1):
        node = rng.choice(remaining_nodes)
        remaining_nodes.remove(node)
        domain = node_domain[node]
        player = Player(
            id=pid,
            domain=domain.name,
            kingdom_name=_make_kingdom_name(rng),
            profession=professions_pool[pid % len(professions_pool)],
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
             kingdom_name=player.kingdom_name, profession=player.profession,
             node=node, on_stage=(node == 0), pick_number=pick_number,
             remaining_after=len(remaining_nodes), temperament=player.temperament,
             temperament_label=player.temperament_label(), model=player.model)

    if SCRIPTED_ONLY:
        agents = {pid: ScriptedAgent(rng) for pid in players}
    else:
        agents = {pid: OllamaAgent(rng, model=players[pid].model) for pid in players}
    game = GameState(players=players, owner=owner, board_adj=board_adj,
                      active_ids=set(players.keys()))
    used_questions = set()  # (domain_name, prompt) pairs, shared for the whole show

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

        emit("agent_thinking", player_id=active_pid, model=active_player.model, decision="target")
        target_id = agent.choose_target(active_player, game)
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
        emit("challenge_declared", challenger_id=active_pid, defender_id=target_id,
             tested_domain=defender.domain, challenger_using_bonus=challenger_bonus,
             defender_using_bonus=False)

        result = run_duel(active_player, defender, DOMAINS_BY_NAME[defender.domain],
                           agents, rng, challenger_bonus=challenger_bonus,
                           used_questions=used_questions)
        if challenger_bonus:
            active_player.time_bonus_banked = False

        for turn_index, turn in enumerate(result.turns_log):
            emit("duel_turn", turn_index=turn_index, challenger_id=active_pid,
                 defender_id=target_id, player_id=turn["player_id"],
                 domain=turn["domain"], prompt=turn["prompt"],
                 answer=turn["answer"], guess=turn["guess"], outcome=turn["outcome"],
                 correct=turn["correct"], seconds_used=turn["seconds_used"],
                 clock_remaining=turn["clock_remaining"],
                 distractors=turn["distractors"])

        game.duel_count += 1
        winner_id, loser_id = result.winner_id, result.loser_id
        winner, loser = players[winner_id], players[loser_id]

        # Territory transfer -- split rather than blind merge. The loser's
        # territory is split into connected pieces (per the current board
        # graph); only the piece(s) actually touching the winner's own
        # territory transfer to the winner. Any OTHER disconnected piece
        # (possible if the loser had themselves picked up territory through
        # an earlier chain of wins) is reassigned to whichever other still-
        # active player borders it instead. A player should never end up
        # holding a disconnected "exclave" just from winning one fight.
        loser_components = connected_components(loser.territory, game.board_adj)
        winner_gain = set()
        reassignments = []  # (new_owner_id, tiles) for any leftover piece
        for comp in loser_components:
            if any(game.board_adj[r] & winner.territory for r in comp):
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
             territory_gained=territory_gained, winner_streak=winner.push_streak)

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
            if rng.random() < 0.8:
                emit("agent_thinking", player_id=winner_id, model=winner.model, decision="tax")
                target_pid = agents[winner_id].choose_tax_target(winner, game)
                if target_pid is not None:
                    target_player = players[target_pid]
                    winner.domain, target_player.domain = target_player.domain, winner.domain
                    emit("domain_tax_applied", from_id=winner_id, to_id=target_pid,
                         winner_new_domain=winner.domain, target_new_domain=target_player.domain)
            else:
                winner.time_bonus_banked = True
                emit("time_bonus_banked", player_id=winner_id)

        # Burst prize checkpoint.
        if game.duel_count % BURST_CHECKPOINT_EVERY == 0 and game.sole_owner() is None:
            leader_id = max(game.active_ids, key=lambda pid: len(players[pid].territory))
            emit("burst_prize", duel_count=game.duel_count, leader_id=leader_id,
                 leader_domain=players[leader_id].domain, territory=len(players[leader_id].territory))
            game.burst_prizes.append({"duel_count": game.duel_count, "leader_id": leader_id})

        # The Scramble.
        if (not game.scrambled and game.duel_count >= SCRAMBLE_MIN_DUELS
                and len(game.active_ids) <= SCRAMBLE_MAX_ACTIVE and len(game.active_ids) > 1):
            _apply_scramble(game)
            game.scrambled = True
            emit("scramble", active_players=len(game.active_ids),
                 board_size=len(game.owner), new_owner=dict(game.owner))

        if game.sole_owner() is not None:
            break

        # Winner's choice: continue pushing or retreat.
        winner_agent = agents[winner_id]
        emit("agent_thinking", player_id=winner_id, model=winner.model, decision="continue")
        keep_going = winner_agent.decide_continue(winner, game)
        if keep_going and game.adjacent_opponents(winner_id):
            game.spotlight = winner_id
            emit("continues", player_id=winner_id)
        else:
            winner.push_streak = 0
            game.spotlight = None
            game.excluded_from_pick = winner_id
            emit("retreats", player_id=winner_id)

    champion_id = game.sole_owner()
    champion = players[champion_id]
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
