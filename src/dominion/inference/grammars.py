"""GBNF grammar builders for llama.cpp's grammar-constrained generation
(llama-server's /completion accepts a "grammar" field -- see
llamacpp_client.py). One builder per structured decision point in
agents/llm_agent.py; each is built dynamically from the actual option
count/letter set offered THAT call, never a hardcoded range -- the same
discipline attempt_question's letters variable already follows (see its
comment about the string.ascii_uppercase bug this avoids repeating).

Ollama has no equivalent mechanism (InferenceClient.supports_grammar is
False for it), so llm_agent.py only builds these when the active backend
actually supports them; every existing regex-based reply parser in
llm_agent.py stays in place regardless -- grammar constrains what the
model CAN say, not how llm_agent.py reads what it said, and Ollama still
needs that same parser unconditionally.

intro_line_combined has no grammar builder here on purpose: it's
open-ended in-character narration, not a structured decision, and stays
num_predict-capped free text on every backend.
"""
from __future__ import annotations

from typing import List


def _literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _alternation(values: List[str]) -> str:
    return " | ".join(_literal(v) for v in values)


def index_choice_grammar(count: int, *, with_memo: bool = False, with_reason: bool = False) -> str:
    """Constrains the reply to exactly one of "0".."count-1" -- the shape
    choose_target and choose_tax_target's prompts both already ask for
    (see llm_agent.py's _parse_index, which stays the parser for both
    backends). with_memo=True additionally requires the "MEMO: ..." second
    line choose_target's prompt asks for -- under grammar, the note is
    always present (still free text, still capped in practice by
    num_predict), one further reliability improvement over Ollama's
    "usually there, not guaranteed" MEMO: line.

    with_reason=True (choose_target only, Scott: make target selection
    "more out loud like an interview") requires a spoken "IDX: reason"
    first line instead of a bare index -- same "VERDICT: reason" shape
    push_retreat_grammar already constrains decide_continue to, so the
    reason is guaranteed present under grammar, same reliability win
    with_memo already gives the MEMO line.
    """
    if count <= 0:
        raise ValueError("index_choice_grammar requires count > 0")
    idx_alt = _alternation([str(i) for i in range(count)])
    if with_reason:
        idx_part = 'idx ": " reason'
        idx_def = f"idx ::= {idx_alt}\nreason ::= [^\\n]+\n"
    else:
        idx_part = "idx"
        idx_def = f"idx ::= {idx_alt}\n"
    if not with_memo:
        return f"root ::= {idx_part}\n{idx_def}"
    return (
        f'root ::= {idx_part} "\\n" "MEMO: " memo\n'
        f"{idx_def}"
        "memo ::= [^\\n]+\n"
    )


def push_retreat_grammar() -> str:
    """Constrains decide_continue's reply to exactly the two-line
    "PUSH: reason" / "RETREAT: reason" plus "MEMO: ..." shape its prompt
    asks for. Grammar guarantees the verdict token itself is unambiguous
    -- it does NOT guarantee the free-text reason doesn't semantically
    contradict that verdict (e.g. "PUSH: ...I'll retire"), so
    llm_agent.py's contradiction_words safety net on the reason text stays
    in place unconditionally; grammar constrains syntax, not semantics.
    """
    return (
        'root ::= verdict ": " reason "\\n" "MEMO: " memo\n'
        'verdict ::= "PUSH" | "RETREAT"\n'
        "reason ::= [^\\n]+\n"
        "memo ::= [^\\n]+\n"
    )


def letter_or_pass_grammar(letters: str) -> str:
    """Constrains attempt_question's reply to exactly one of the letters
    actually offered this turn (the same `letters` string
    attempt_question already builds from string.ascii_uppercase[:len(choices)],
    never a hardcoded A-D) or the literal "PASS". This is the grammar most
    directly aimed at docs/ENGINE_NOTES.md's documented reliability gaps:
    an unparseable reply or an out-of-range letter can no longer happen on
    this backend at all, closing the exact failure modes that previously
    fell back to ScriptedAgent.
    """
    if not letters:
        raise ValueError("letter_or_pass_grammar requires a non-empty letters string")
    return f"root ::= {_alternation(list(letters) + ['PASS'])}\n"
