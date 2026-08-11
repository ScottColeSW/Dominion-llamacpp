"""GameState.opponent_record/opponent_familiarity/record_duel_result
(engine/models.py) -- #47's structured, engine-authored per-opponent
head-to-head memory, the domain_record-shaped counterpart for rivalries
instead of trivia domains. Deliberately not LLM-dependent: nothing here
touches an agent or a prompt, just the same plain-data bookkeeping
domain_record/record_duel_turns already does."""
from __future__ import annotations

import unittest

from dominion.engine.models import GameState, Player


def _player(pid: int, name: str) -> Player:
    return Player(id=pid, domain="Dogs", kingdom_name=name, profession="tester",
                  origin_domain="Cats", model="test-model")


def _game() -> GameState:
    p0, p1, p2 = _player(0, "A"), _player(1, "B"), _player(2, "C")
    return GameState(players={0: p0, 1: p1, 2: p2}, owner={0: 0, 1: 1, 2: 2},
                      board_adj={0: {1, 2}, 1: {0}, 2: {0}}, active_ids={0, 1, 2})


class OpponentRecordTest(unittest.TestCase):
    def test_unfaced_opponent_returns_none_not_zero(self) -> None:
        game = _game()
        self.assertIsNone(game.opponent_familiarity(0, 1))

    def test_a_single_result_is_recorded_symmetrically_for_both_sides(self) -> None:
        game = _game()
        game.record_duel_result(winner_id=0, loser_id=1)
        self.assertEqual(game.opponent_familiarity(0, 1), (1, 0))
        self.assertEqual(game.opponent_familiarity(1, 0), (0, 1))
        # An unrelated third player's record stays untouched.
        self.assertIsNone(game.opponent_familiarity(2, 0))
        self.assertIsNone(game.opponent_familiarity(0, 2))

    def test_a_rematch_accumulates_onto_the_same_pair(self) -> None:
        game = _game()
        game.record_duel_result(winner_id=0, loser_id=1)
        game.record_duel_result(winner_id=1, loser_id=0)  # B wins the rematch
        self.assertEqual(game.opponent_familiarity(0, 1), (1, 1))
        self.assertEqual(game.opponent_familiarity(1, 0), (1, 1))

    def test_a_third_duel_between_the_same_pair_keeps_accumulating(self) -> None:
        game = _game()
        game.record_duel_result(winner_id=0, loser_id=1)
        game.record_duel_result(winner_id=0, loser_id=1)
        game.record_duel_result(winner_id=0, loser_id=1)
        self.assertEqual(game.opponent_familiarity(0, 1), (3, 0))
        self.assertEqual(game.opponent_familiarity(1, 0), (0, 3))


if __name__ == "__main__":
    unittest.main()
