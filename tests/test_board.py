"""engine/board.py's find_reconnect_path -- the BFS bridge-finder #13's
Threepeat Drive needs to fix the territory-transfer paradox for a
non-adjacent win (see game.py's territory-transfer block). Deliberately
not tests/test_board_geometry.py, which is narrowly scoped to verifying
board_adj against the frontend's own on-screen hex layout."""
from __future__ import annotations

import unittest

from dominion.engine.board import build_pyramid_13, find_reconnect_path


class FindReconnectPathTest(unittest.TestCase):
    def test_returns_the_shortest_bridge_between_two_non_adjacent_tiles(self) -> None:
        adj = build_pyramid_13()
        # Node 0 (the apex/Stage, only touches {1, 2}) and node 9 (a
        # far corner) are not adjacent -- the one and only shortest path
        # between them is 0 -> 2 -> 5 -> 9, so the real bridge is [2, 5].
        self.assertEqual(find_reconnect_path(adj, {0}, {9}), [2, 5])

    def test_empty_when_the_two_sets_already_touch_directly(self) -> None:
        adj = build_pyramid_13()
        # 0 and 1 are directly adjacent -- no bystander tiles need
        # annexing at all.
        self.assertEqual(find_reconnect_path(adj, {0}, {1}), [])

    def test_empty_when_either_side_is_empty(self) -> None:
        adj = build_pyramid_13()
        self.assertEqual(find_reconnect_path(adj, set(), {9}), [])
        self.assertEqual(find_reconnect_path(adj, {0}, set()), [])

    def test_multi_tile_territories_still_find_the_true_shortest_bridge(self) -> None:
        adj = build_pyramid_13()
        # A winner already holding {0, 1} and a loser holding {9, 12} --
        # still bridges via 0/1 -> 2 -> 5 -> 9 (or 12), same [2, 5] bridge,
        # since neither side's OWN tiles ever touch each other directly.
        self.assertEqual(find_reconnect_path(adj, {0, 1}, {9, 12}), [2, 5])


if __name__ == "__main__":
    unittest.main()
