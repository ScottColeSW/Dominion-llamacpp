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
    """Build a hub (node 0) + ring (nodes 1..n-1) adjacency map for n nodes."""
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

    max_offset = min(2, ring_size // 2)
    for idx, i in enumerate(ring):
        for offset in range(1, max_offset + 1):
            j = ring[(idx + offset) % ring_size]
            if j != i:
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
