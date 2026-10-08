"""Deterministic handling for incomplete next-round advice.

The scientific interpreter must persist a non-empty next-round suggestion.
When the LLM returns an empty or partial suggestion, we still write explicit
placeholder text so the report and frontend have something to show, and we
record the missing fields so the round is marked `incomplete` and cannot
automatically advance.
"""

from __future__ import annotations

from core.unified_schema import (
    ConclusionNextRoundSuggestion,
    missing_next_round_advice_fields,
)

INCOMPLETE_EVIDENCE = "本轮未生成有效的证据摘要，需人工补录后再进入下一轮。"
INCOMPLETE_RESOLVED = "需人工补录本轮已解决的不确定性"
INCOMPLETE_UNRESOLVED = "需人工补录仍需解决的不确定性"
INCOMPLETE_REMAINING = "本轮未生成有效的不确定性分析，需人工补录后再进入下一轮。"
INCOMPLETE_PI = "建议暂不进入下一轮，需先补录下一轮决策建议。"
INCOMPLETE_DESIGN = "建议暂不进入下一轮，需先补录下一轮实验设计建议。"
INCOMPLETE_HYPOTHESIS_SPACE = "暂无可用的下一轮假设空间建议，需人工补录。"
INCOMPLETE_RECOMMENDATION = "adjust"


def ensure_next_round_suggestion(
    suggestion: ConclusionNextRoundSuggestion | None,
) -> tuple[ConclusionNextRoundSuggestion, list[str]]:
    """Fill incomplete advice with explicit placeholders and return missing fields.

    The returned suggestion is never `None`; the second tuple item carries the
    fields that still require human/LLM completion.  Consumers use that list to
    gate round closure instead of re-checking the placeholder text.
    """
    if suggestion is None:
        suggestion = ConclusionNextRoundSuggestion()
    missing = missing_next_round_advice_fields(suggestion)
    if not missing:
        return suggestion, []

    suggestion.evidence_summary = (
        (suggestion.evidence_summary or "").strip() or INCOMPLETE_EVIDENCE
    )
    if not any(str(item).strip() for item in suggestion.resolved_uncertainties_this_round):
        suggestion.resolved_uncertainties_this_round = [INCOMPLETE_RESOLVED]
    if not any(str(item).strip() for item in suggestion.unresolved_uncertainties_todo):
        suggestion.unresolved_uncertainties_todo = [INCOMPLETE_UNRESOLVED]
    suggestion.remaining_uncertainty_analysis = (
        (suggestion.remaining_uncertainty_analysis or "").strip()
        or INCOMPLETE_REMAINING
    )
    suggestion.recommendation = suggestion.recommendation or INCOMPLETE_RECOMMENDATION
    suggestion.pi_decision_advice = (
        (suggestion.pi_decision_advice or "").strip() or INCOMPLETE_PI
    )
    suggestion.experiment_design_advice = (
        (suggestion.experiment_design_advice or "").strip() or INCOMPLETE_DESIGN
    )
    if not (suggestion.hypothesis_space_advice or "").strip():
        suggestion.hypothesis_space_advice = INCOMPLETE_HYPOTHESIS_SPACE
    for field in missing:
        if field not in suggestion.advice_incomplete:
            suggestion.advice_incomplete.append(field)
    note_prefix = "下一轮建议字段缺失，本轮标记为 incomplete，不允许自动进入下一轮："
    if not any(note.startswith(note_prefix) for note in suggestion.notes):
        suggestion.notes.append(note_prefix + "、".join(missing))
    return suggestion, list(missing)
