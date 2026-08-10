"""Grammar builders are pure string generation -- these tests check shape
(valid GBNF-looking output, the right literal alternatives present) not
that llama.cpp itself accepts them; that's covered by manual verification
against a real llama-server (see the plan's verification steps)."""
from __future__ import annotations

import unittest

from dominion.inference.grammars import (
    index_choice_grammar,
    letter_or_pass_grammar,
    push_retreat_grammar,
)


class IndexChoiceGrammarTest(unittest.TestCase):
    def test_rejects_non_positive_count(self) -> None:
        with self.assertRaises(ValueError):
            index_choice_grammar(0)

    def test_alternatives_cover_exact_range(self) -> None:
        grammar = index_choice_grammar(3)
        for i in range(3):
            self.assertIn(f'"{i}"', grammar)
        self.assertNotIn('"3"', grammar)

    def test_double_digit_count(self) -> None:
        grammar = index_choice_grammar(12)
        self.assertIn('"11"', grammar)
        self.assertNotIn('"12"', grammar)

    def test_with_memo_requires_memo_line(self) -> None:
        grammar = index_choice_grammar(2, with_memo=True)
        self.assertIn("MEMO: ", grammar)
        self.assertIn("memo ::=", grammar)

    def test_without_memo_has_no_memo_rule(self) -> None:
        grammar = index_choice_grammar(2, with_memo=False)
        self.assertNotIn("MEMO", grammar)


class PushRetreatGrammarTest(unittest.TestCase):
    def test_has_both_verdicts_and_memo(self) -> None:
        grammar = push_retreat_grammar()
        self.assertIn('"PUSH"', grammar)
        self.assertIn('"RETREAT"', grammar)
        self.assertIn("MEMO: ", grammar)


class LetterOrPassGrammarTest(unittest.TestCase):
    def test_rejects_empty_letters(self) -> None:
        with self.assertRaises(ValueError):
            letter_or_pass_grammar("")

    def test_includes_every_letter_and_pass(self) -> None:
        grammar = letter_or_pass_grammar("ABCDE")
        for letter in "ABCDE":
            self.assertIn(f'"{letter}"', grammar)
        self.assertIn('"PASS"', grammar)

    def test_excludes_letters_not_offered(self) -> None:
        grammar = letter_or_pass_grammar("AB")
        self.assertNotIn('"C"', grammar)


if __name__ == "__main__":
    unittest.main()
