"""Gold-blind, ranking-compatible wrapper around DPC's one-shot selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass
class StageResult:
    status: str
    runtime_ms: float
    usage: dict[str, Any]
    selected_sql: str | None = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    selector_invoked: bool = True


def _sum_usage(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "selector_invocations": sum(stage["selector_invoked"] for stage in stages)
    }
    for key in (
        "api_calls",
        "api_successful_calls",
        "api_failed_calls",
        "input_tokens",
        "output_tokens",
        "reported_input_tokens",
        "reported_output_tokens",
    ):
        values = [stage["usage"].get(key) for stage in stages]
        usage[key] = sum(values) if all(type(value) is int for value in values) else None
    usage["token_usage_complete"] = all(
        stage["usage"].get("token_usage_complete") is True for stage in stages
    )
    usage["pipeline_fallbacks"] = sum(
        stage["evidence"].get("pipeline_fallback") is True for stage in stages
    )
    return usage


def select_candidates(
    candidates: Sequence[Mapping[str, str]],
    choose: Callable[[Sequence[Mapping[str, str]]], StageResult],
    mode: str,
) -> dict[str, Any]:
    """Return a one-shot selection or an induced total order.

    The callback runs the unchanged DPC selector on the remaining public SQLs.
    Identical SQL strings map to the first remaining public candidate ID.
    """
    if mode not in {"one_shot", "sequential"}:
        raise ValueError(f"Unsupported DPC benchmark mode: {mode}")
    if len(candidates) < 2:
        raise ValueError("DPC benchmark requires at least two candidates")
    remaining = list(candidates)
    ranking: list[str] = []
    stages: list[dict[str, Any]] = []
    calls = 1 if mode == "one_shot" else len(remaining) - 1

    for invocation in range(1, calls + 1):
        stage = choose(tuple(remaining))
        status = stage.status
        selected_id = None
        error = stage.error
        if status == "success":
            selected = next(
                (candidate for candidate in remaining if candidate["sql"] == stage.selected_sql),
                None,
            )
            if selected is None:
                status = "invalid_output"
                error = "DPC selected SQL that is not an exact remaining candidate"
            else:
                selected_id = selected["candidate_id"]

        trace = {
            "invocation": invocation,
            "remaining_candidate_ids": [candidate["candidate_id"] for candidate in remaining],
            "selected_candidate_id": selected_id,
            "status": status,
            "runtime_ms": stage.runtime_ms,
            "usage": stage.usage,
            "evidence": stage.evidence,
            "selector_invoked": stage.selector_invoked,
        }
        if error:
            trace["error"] = {"message": error}
        stages.append(trace)

        if status != "success":
            return {
                "status": status,
                "error": {"message": error or "DPC selection failed"},
                "partial_ranking": ranking,
                "selection_trace": stages,
                "usage": _sum_usage(stages),
            }

        if mode == "one_shot":
            return {
                "status": "success",
                "selected_candidate_id": selected_id,
                "selection_trace": stages,
                "usage": _sum_usage(stages),
            }

        ranking.append(selected_id)
        remaining = [candidate for candidate in remaining if candidate["candidate_id"] != selected_id]

    ranking.append(remaining[0]["candidate_id"])
    return {
        "status": "success",
        "ranking": ranking,
        "selection_trace": stages,
        "usage": _sum_usage(stages),
    }
