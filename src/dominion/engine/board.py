"""Board construction.

Two shapes are used, deliberately, for two different jobs:

- build_pyramid_13(): the locked starting board, Revision 15. A real hex
  tessellation, five rows of 1, 2, 3, 4, 3 tiles, node 0 is the apex and
  is the Challenge Stage. Verified connected via real geometric hex
  adjacency (distance-based, not hand-asserted): avg degree 4.0, min
  degree 2 (the Stage itself, the lowest-reach tile on the board), max
  degree 6 (the three interior tiles). This exact row pattern does not
  generalize safely to arbitrary player counts: growing 1,2,3,...,k and
  dumping whatever is left over as a final row only produces a connected
  board when that leftover row's width lands exactly one less than the
  row above it. Checked computationally at every size from 13 down to 2,
  most sizes fail that condition and come out disconnected. So this
  shape is fixed at 13 and only used for the opening board.

- build_hub_ring(n): the general-purpose formula from earlier revisions,
  verified connected at every size from 13 down to 2. Used for the
  Scramble redraw, which needs to resize safely as players are
  eliminated. The show visually shifts from the pyramid tessellation to
  a compact wheel at the Scramble, which is an intentional beat, not a
  visual inconsistency: the board looking different is how the audience
  feels the redraw actually happened.
"""
from __future__ import annotations
from typing import Dict, Set, List


def build_pyramid_13() -> Dict[int, Set[int]]:
    """13-tile hex pyramid, rows of 1,2,3,4,3. Node 0 is the apex/Stage.

    Node 9's list previously included a bogus "6": node 9 is the rightmost
    tile of row 4 (indices 6,7,8,9) and node 6 is that row's LEFTMOST tile
    -- opposite ends of the same row, nowhere near each other. That one
    stray edge was one-directional (9 listed 6, but 6 never listed 9 back),
    which is exactly how a duel could get declared between two tiles that
    don't actually touch: any check starting from node 9's side saw them
    as adjacent, while a check from node 6's side correctly didn't. Fixed
    by removing that entry; re-derived by hand from the actual row/column
    geometry and cross-checked against the degree distribution this
    docstring already claimed (avg 4.0, needs an even total, which the
    buggy version failed: 53 is odd, 52 is not).
    """
    adj_lists = {
        0: {1, 2},
        1: {0, 2, 3, 4},
        2: {0, 1, 4, 5},
        3: {1, 4, 6, 7},
        4: {1, 2, 3, 5, 7, 8},
        5: {2, 4, 8, 9},
        6: {3, 7, 10},
        7: {3, 4, 6, 8, 10, 11},
        8: {4, 5, 7, 9, 11, 12},
        9: {5, 8, 12},
        10: {6, 7, 11},
        11: {7, 8, 10, 12},
        12: {8, 9, 11},
    }
    return {k: set(v) for k, v in adj_lists.items()}


def build_hub_ring(n: int) -> Dict[int, Set[int]]:
    """Build a hub (node 0) + ring (nodes 1..n-1) adjacency map for n nodes --
    the post-Scramble board (see game.py's _apply_scramble). The frontend
    (web/index.html's circularPositions) lays this out as a real hex flower:
    the hub at center, ring tiles at 60-degree intervals around it, each at
    the exact hex-touching distance from the hub AND from their immediate
    ring neighbors -- so ring node i geometrically touches only i-1 and i+1,
    never anything farther around the ring.

    Only ring OFFSET-1 edges here, not offset-2: an earlier version also
    connected each ring node to the one TWO positions away (offset-2, 120
    degrees apart), reasoning it'd give more matchup variety. It's a real
    geometric gap, not a rendering nuance -- chord distance at 120 degrees
    works out to about 1.7x the true hex-touching distance the 60-degree
    neighbors sit at, so there's a visibly different-colored tile actually
    sitting between two "adjacent" offset-2 tiles on screen. A player could
    legitimately (per the graph) hold both without holding what's between
    them, which reads as broken, split territory to anyone looking at the
    board even though the game considered it one connected piece -- this is
    what Scott's contiguity report turned out to be. The hub alone already
    guarantees the whole graph stays connected (it borders every ring tile),
    so dropping the offset-2 chords costs some matchup variety but not
    connectivity, and the logical graph now means exactly what it looks
    like on screen.

    The ring only wraps into a closed cycle when ALL SIX slots are in use
    (ring_size == 6, i.e. n == 7 -- the only size the Scramble actually
    produces today, game.py's SCRAMBLE_MAX_ACTIVE). circularPositions'
    SLOT_ANGLES_DEG is six FIXED 60-degree positions used in order; for
    fewer active players, the occupied slots are a contiguous ARC of that
    circle, not the whole thing -- the first and last occupied petals sit
    on opposite ends of that arc with real, unused angular space (the
    remaining slots) between them, not touching. Treating the ring as
    always-a-cycle regardless of size (an earlier version of this
    function, and the plain modular-arithmetic way to write the loop) drew
    a phantom edge between those two end petals for any n < 7 -- caught by
    engine/test_board_geometry.py, which mirrors circularPositions in
    Python and checks board_adj against actual hex-edge coincidence rather
    than trusting this docstring's reasoning alone. Not reachable in a
    real show today (the Scramble only ever fires at n == 7), but a latent
    bug all the same, and the exact kind this test exists to catch before
    it becomes a live one."""
    if n <= 1:
        return {0: set()}
    if n == 2:
        return {0: {1}, 1: {0}}

    ring: List[int] = list(range(1, n))
    ring_size = len(ring)
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i in ring:
        adj[0].add(i)
        adj[i].add(0)

    for idx in range(ring_size - 1):
        i, j = ring[idx], ring[idx + 1]
        adj[i].add(j)
        adj[j].add(i)
    if ring_size == 6:
        # Only now does the occupied arc actually close into a full circle
        # -- the first and last petals are true 60-degree neighbors too.
        i, j = ring[0], ring[-1]
        adj[i].add(j)
        adj[j].add(i)

    return adj


def is_connected(adj: Dict[int, Set[int]]) -> bool:
    if not adj:
        return True
    start = next(iter(adj))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(adj)


def connected_components(nodes: Set[int], adj: Dict[int, Set[int]]) -> List[Set[int]]:
    """Split an arbitrary set of nodes into its connected pieces, using adj
    restricted to that set. Used to make sure a player never inherits a
    disconnected "exclave" from a defeated opponent -- only whichever piece
    actually touches the winner should transfer to them; see game.py's
    duel_result handling."""
    remaining = set(nodes)
    components: List[Set[int]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = {start}
        remaining.discard(start)
        while stack:
            node = stack.pop()
            for neighbor in adj.get(node, ()):
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    comp.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)
    return components
