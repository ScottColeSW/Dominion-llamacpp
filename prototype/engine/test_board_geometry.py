"""Verifies engine/board.py's declared adjacency (board_adj) against what's
actually drawn on screen -- not just internal graph consistency.

This exists because of a real bug: build_hub_ring used to also connect each
ring tile to the one TWO positions away ("offset-2"), reasoning it'd give
more matchup variety. The graph was perfectly self-consistent (symmetric,
fully connected) the whole time; the problem was that it didn't match
circularPositions' actual hex-flower layout in web/index.html, where an
offset-2 pair sits 120 degrees apart -- a real geometric gap, with a
different-colored tile visibly sitting between them. A player could hold
both without holding what's between them: one connected piece per the
graph, visibly broken/split territory on screen. A plain "is board_adj
internally consistent" check would never have caught this, since the graph
never contradicted itself -- only the picture did. So this mirrors the
frontend's own position math in Python and derives "true" adjacency
directly from which hexagons' edges actually coincide, the same way you'd
check it by looking at the board.

Whenever someone changes hexPoints/pyramidPositions/circularPositions in
web/index.html OR the adjacency formulas in board.py, run this file -- if
the numbers below no longer match those functions, update them here too
(deliberately hand-mirrored, not shared code, so a real port needed for
correctness is what actually keeps this test meaningful).
"""
from __future__ import annotations
import math
import unittest
from typing import Dict, List, Set, Tuple

from .board import build_pyramid_13, build_hub_ring

Point = Tuple[float, float]


def _hex_points(cx: float, cy: float, r: float) -> List[Point]:
    """Mirrors web/index.html's hexPoints() exactly: 6 vertices at 60-degree
    steps, rounded to 1 decimal place like the frontend's toFixed(1) -- two
    hexagons share a true edge only if a full vertex PAIR coincides at this
    same precision, not just "close.\""""
    pts = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i)
        pts.append((round(cx + r * math.sin(angle), 1), round(cy - r * math.cos(angle), 1)))
    return pts


def _pyramid_positions() -> Tuple[List[Point], float]:
    """Mirrors web/index.html's pyramidPositions() exactly."""
    size = 76.0
    dx = math.sqrt(3) * size
    dy = 1.5 * size
    rows = [1, 2, 3, 4, 3]
    cx, cy = 350.0, 92.0
    positions = []
    for r, w in enumerate(rows):
        y = cy + r * dy
        for i in range(w):
            x = cx + (i - (w - 1) / 2) * dx
            positions.append((x, y))
    return positions, size


SLOT_ANGLES_DEG = [270, 330, 30, 90, 150, 210]


def _circular_positions(n: int) -> Tuple[List[Point], float]:
    """Mirrors web/index.html's circularPositions() exactly, including the
    max-size-that-fits-the-viewport search -- uniform scaling never changes
    which hexagons touch, but mirroring it exactly leaves no room for a
    "well it's probably fine" gap in what this test actually checks."""
    cx, cy = 350.0, 320.0
    petal_count = max(0, min(6, n - 1))
    slots = SLOT_ANGLES_DEG[:petal_count]

    margin_x, margin_y = 40.0, 30.0
    max_half_w, max_half_h = 350.0 - margin_x, cy - margin_y
    max_size = 150.0
    for deg in slots:
        rad = math.pi / 180 * deg
        ux = math.sin(rad) * math.sqrt(3)
        uy = -math.cos(rad) * math.sqrt(3)
        half_w_needed = abs(ux) + 1
        half_h_needed = abs(uy) + 1
        max_size = min(max_size, max_half_w / half_w_needed, max_half_h / half_h_needed)
    size = max(58.0, math.floor(max_size))

    positions = [(cx, cy)]
    neighbor_dist = math.sqrt(3) * size
    for deg in slots:
        rad = math.pi / 180 * deg
        positions.append((cx + neighbor_dist * math.sin(rad), cy - neighbor_dist * math.cos(rad)))
    return positions, size


def _geometric_adjacency(positions: List[Point], size: float) -> Dict[int, Set[int]]:
    """Derives adjacency directly from the drawn hexagons: two tiles are
    adjacent iff their hex outlines share a real edge (a coinciding vertex
    pair), the same fact a viewer would see on screen -- not a formula
    that's supposed to describe it."""
    hexes = [_hex_points(x, y, size) for x, y in positions]
    edge_sets = []
    for pts in hexes:
        edges = set()
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            edges.add(frozenset((a, b)))
        edge_sets.append(edges)

    adj: Dict[int, Set[int]] = {i: set() for i in range(len(positions))}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if edge_sets[i] & edge_sets[j]:
                adj[i].add(j)
                adj[j].add(i)
    return adj


class TestPyramidGeometry(unittest.TestCase):
    def test_declared_adjacency_matches_drawn_hexagons(self):
        positions, size = _pyramid_positions()
        geometric = _geometric_adjacency(positions, size)
        declared = build_pyramid_13()
        self.assertEqual(len(positions), 13)
        self.assertEqual(declared, geometric,
                          "engine/board.py's build_pyramid_13() no longer matches "
                          "what pyramidPositions() actually draws -- see this file's "
                          "module docstring.")


class TestHubRingGeometry(unittest.TestCase):
    def test_declared_adjacency_matches_drawn_hexagons_at_every_size(self):
        # 2..7: every active-player count the Scramble can actually produce
        # (game.py's SCRAMBLE_MAX_ACTIVE=7, and >1 is required to scramble
        # at all) -- not just the one size it happens to fire at today.
        for n in range(2, 8):
            with self.subTest(n=n):
                positions, size = _circular_positions(n)
                geometric = _geometric_adjacency(positions, size)
                declared = build_hub_ring(n)
                self.assertEqual(len(positions), n)
                self.assertEqual(declared, geometric,
                                  f"engine/board.py's build_hub_ring({n}) no longer "
                                  f"matches what circularPositions({n}) actually draws "
                                  f"-- see this file's module docstring.")

    def test_hub_always_borders_every_ring_tile(self):
        # Not strictly required by the geometry check above, but a cheap,
        # independent sanity check on the specific property the whole
        # graph's connectivity relies on: the hub bridges everyone, so ring
        # edges can be sparse without ever risking a disconnected board.
        for n in range(3, 8):
            adj = build_hub_ring(n)
            for ring_node in range(1, n):
                self.assertIn(0, adj[ring_node], f"n={n}: ring node {ring_node} doesn't border the hub")
                self.assertIn(ring_node, adj[0], f"n={n}: hub doesn't border ring node {ring_node}")


if __name__ == "__main__":
    unittest.main()
