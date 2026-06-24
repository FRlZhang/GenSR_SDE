#!/usr/bin/env python3
"""Offline gate-feasibility diagnostic for fixed P2 residual calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_p2_admitted_candidate_scoring import json_ready, parse_target_samples


V0 = "V0_current_rolewise_no_multi_u0"
VARIANTS = [
    "V3_active_without_dim_76",
    "V4_active_without_dims_76_72_82_88",
    "V6_active_and_weak_without_recurring_dims",
]
OBSERVABLE_GATES = [
    "G1_variant_selected_source_is_p2_admitted",
    "G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded",
    "G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable",
    "G4_v0_selected_active_top_dim_is_76",
    "G5_v0_selected_active_top_dim_in_76_72_82_88",
    "G6_v0_selected_active_recurring_dims_share_ge_0_50",
    "G7_v0_selected_active_recurring_dims_share_ge_0_35",
    "G8_v0_selected_weak_top_dim_in_136_137_133",
    "G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted",
]
UPPER_BOUND_GATES = [
    "U1_apply_variant_only_when_V0_is_not_oracle",
    "U2_apply_variant_only_when_variant_is_oracle",
    "U3_apply_variant_only_on_known_P2_newly_recovered_samples",
]
RECURRING_ACTIVE_DIMS = {76, 72, 82, 88}
RECURRING_WEAK_DIMS = {136, 137, 133}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all32-smoke-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--counterfactual-json", type=Path, required=True)
    parser.add_argument("--target-samples", type=str, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def by_sample(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in payload.get("samples", [])}


def candidate_lookup(sidecar: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in sidecar.get("samples", []):
        result[int(sample["sample_index"])] = {
            row["candidate_id"]: row for row in sample.get("candidates", [])
        }
    return result


def residual_feature_by_sample(residual_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    features: dict[int, dict[str, Any]] = {}
    for sample in residual_payload.get("samples", []):
        idx = int(sample["sample_index"])
        active_dims = [
            int(row["dimension_index"])
            for row in sample.get("top_active_residual_advantages_for_selected", [])
            if row.get("dimension_index") is not None
        ]
        weak_dims = [
            int(row["dimension_index"])
            for row in sample.get("top_weak_residual_advantages_for_selected", [])
            if row.get("dimension_index") is not None
        ]
        recurring_count = sum(1 for dim in active_dims if dim in RECURRING_ACTIVE_DIMS)
        features[idx] = {
            "active_top_dim": active_dims[0] if active_dims else None,
            "weak_top_dim": weak_dims[0] if weak_dims else None,
            "active_dims": active_dims,
            "weak_dims": weak_dims,
            "active_recurring_dims_share": (
                recurring_count / len(active_dims) if active_dims else None
            ),
        }
    return features


def token_text(tokens: list[str] | None) -> str:
    return " ".join(tokens or [])


def sequence_for(
    idx: int,
    variant_row: dict[str, Any],
    candidates_by_sample: dict[int, dict[str, dict[str, Any]]],
) -> list[str] | None:
    candidate_id = variant_row.get("selected_candidate_id")
    if candidate_id is None:
        return None
    row = candidates_by_sample.get(idx, {}).get(candidate_id)
    return None if row is None else row.get("full_sequence_tokens")


def gate_is_available(gate: str, residual_features: dict[int, dict[str, Any]], target_samples: list[int]) -> tuple[bool, str | None]:
    if gate in {
        "G1_variant_selected_source_is_p2_admitted",
        "G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded",
        "G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable",
        "G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted",
    }:
        return True, None
    missing = [idx for idx in target_samples if idx not in residual_features]
    if missing:
        return (
            False,
            "all-32 selected residual top-dimension fields are missing; only "
            f"{len(residual_features)}/{len(target_samples)} P2 score-miss samples have residual diagnostics",
        )
    return True, None


def observable_gate_applies(
    gate: str,
    idx: int,
    v0_row: dict[str, Any],
    variant_row: dict[str, Any],
    residual_features: dict[int, dict[str, Any]],
) -> bool | None:
    if gate == "G1_variant_selected_source_is_p2_admitted":
        return variant_row.get("selected_candidate_source") == "p2_admitted"
    if gate == "G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded":
        return (
            variant_row.get("selected_candidate_source") == "p2_admitted"
            and v0_row.get("selected_candidate_source") == "current_expanded"
        )
    if gate == "G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable":
        variant_score = variant_row.get("selected_score")
        v0_score = v0_row.get("selected_score")
        if variant_score is None or v0_score is None:
            return None
        return float(variant_score) < float(v0_score)
    if gate == "G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted":
        return (
            v0_row.get("selected_candidate_source") == "current_expanded"
            and variant_row.get("selected_candidate_source") == "p2_admitted"
        )
    features = residual_features.get(idx)
    if features is None:
        return None
    if gate == "G4_v0_selected_active_top_dim_is_76":
        return features.get("active_top_dim") == 76
    if gate == "G5_v0_selected_active_top_dim_in_76_72_82_88":
        return features.get("active_top_dim") in RECURRING_ACTIVE_DIMS
    if gate == "G6_v0_selected_active_recurring_dims_share_ge_0_50":
        share = features.get("active_recurring_dims_share")
        return None if share is None else share >= 0.50
    if gate == "G7_v0_selected_active_recurring_dims_share_ge_0_35":
        share = features.get("active_recurring_dims_share")
        return None if share is None else share >= 0.35
    if gate == "G8_v0_selected_weak_top_dim_in_136_137_133":
        return features.get("weak_top_dim") in RECURRING_WEAK_DIMS
    raise ValueError(f"unknown observable gate: {gate}")


def upper_bound_gate_applies(
    gate: str,
    idx: int,
    v0_row: dict[str, Any],
    variant_row: dict[str, Any],
    p2_target_samples: set[int],
) -> bool:
    if gate == "U1_apply_variant_only_when_V0_is_not_oracle":
        return not bool(v0_row.get("selected_is_oracle_bool"))
    if gate == "U2_apply_variant_only_when_variant_is_oracle":
        return bool(variant_row.get("selected_is_oracle_bool"))
    if gate == "U3_apply_variant_only_on_known_P2_newly_recovered_samples":
        return idx in p2_target_samples
    raise ValueError(f"unknown upper-bound gate: {gate}")


def change_type(v0_row: dict[str, Any], after_row: dict[str, Any], gate_applies: bool) -> str:
    if not gate_applies or after_row.get("selected_candidate_id") == v0_row.get("selected_candidate_id"):
        return "unchanged"
    if (not v0_row.get("selected_is_oracle_bool")) and after_row.get("selected_is_oracle_bool"):
        return "rescue"
    if v0_row.get("selected_is_oracle_bool") and not after_row.get("selected_is_oracle_bool"):
        return "harm"
    return "changed_nonoracle_to_nonoracle"


def feature_payload(
    gate: str,
    idx: int,
    v0_row: dict[str, Any],
    variant_row: dict[str, Any],
    residual_features: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    features = {
        "v0_selected_source": v0_row.get("selected_candidate_source"),
        "variant_selected_source": variant_row.get("selected_candidate_source"),
        "v0_selected_score": v0_row.get("selected_score"),
        "variant_selected_score": variant_row.get("selected_score"),
    }
    if idx in residual_features:
        features.update(residual_features[idx])
    return features


def evaluate_gate(
    variant: str,
    gate: str,
    gate_type: str,
    samples_by_idx: dict[int, dict[str, Any]],
    target_samples: list[int],
    candidates_by_sample: dict[int, dict[str, dict[str, Any]]],
    residual_features: dict[int, dict[str, Any]],
    p2_target_samples: set[int],
    available: bool = True,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    if not available:
        return {
            "variant": variant,
            "gate": gate,
            "gate_type": "unavailable",
            "unavailable_reason": unavailable_reason,
            "samples_gate_applies_to": [],
            "selected_exact_count_after_gate": None,
            "selected_relaxed_count_after_gate": None,
            "selected_p2_oracle_count_after_gate": None,
            "rescues_vs_V0": None,
            "harms_vs_V0": None,
            "net_exact_delta_vs_V0": None,
            "preserves_all_V0_hits_bool": False,
            "uses_oracle_labels_bool": False,
            "implementable_bool": False,
            "per_sample_changes": [],
        }

    selected_exact = 0
    selected_relaxed = 0
    selected_p2_oracle = 0
    rescues = 0
    harms = 0
    applies_to: list[int] = []
    changes: list[dict[str, Any]] = []
    v0_exact = sum(
        1
        for idx in target_samples
        if samples_by_idx[idx]["variants"][V0].get("selected_exact_bool")
    )
    v0_relaxed = sum(
        1
        for idx in target_samples
        if samples_by_idx[idx]["variants"][V0].get("selected_relaxed_bool")
    )

    for idx in target_samples:
        sample = samples_by_idx[idx]
        v0_row = sample["variants"][V0]
        variant_row = sample["variants"][variant]
        if gate_type == "observable":
            applies = observable_gate_applies(gate, idx, v0_row, variant_row, residual_features)
            if applies is None:
                applies = False
        else:
            applies = upper_bound_gate_applies(gate, idx, v0_row, variant_row, p2_target_samples)
        if applies:
            applies_to.append(idx)
        after = variant_row if applies else v0_row
        selected_exact += int(after.get("selected_exact_bool"))
        selected_relaxed += int(after.get("selected_relaxed_bool"))
        selected_p2_oracle += int(after.get("selected_is_p2_oracle_bool"))
        ctype = change_type(v0_row, after, applies)
        rescues += int(ctype == "rescue")
        harms += int(ctype == "harm")
        if ctype != "unchanged":
            changes.append(
                {
                    "sample_index": idx,
                    "gate_applied_bool": applies,
                    "V0_selected_sequence": sequence_for(idx, v0_row, candidates_by_sample),
                    "V0_is_oracle": bool(v0_row.get("selected_is_oracle_bool")),
                    "variant_selected_sequence": sequence_for(idx, variant_row, candidates_by_sample),
                    "variant_is_oracle": bool(variant_row.get("selected_is_oracle_bool")),
                    "after_gate_sequence": sequence_for(idx, after, candidates_by_sample),
                    "change_type": ctype,
                    "observable_gate_features": feature_payload(
                        gate, idx, v0_row, variant_row, residual_features
                    ),
                }
            )

    return {
        "variant": variant,
        "gate": gate,
        "gate_type": gate_type,
        "unavailable_reason": None,
        "samples_gate_applies_to": applies_to,
        "selected_exact_count_after_gate": selected_exact,
        "selected_relaxed_count_after_gate": selected_relaxed,
        "selected_p2_oracle_count_after_gate": selected_p2_oracle,
        "rescues_vs_V0": rescues,
        "harms_vs_V0": harms,
        "net_exact_delta_vs_V0": selected_exact - v0_exact,
        "net_relaxed_delta_vs_V0": selected_relaxed - v0_relaxed,
        "preserves_all_V0_hits_bool": harms == 0,
        "uses_oracle_labels_bool": gate_type == "oracle_upper_bound",
        "implementable_bool": gate_type == "observable",
        "per_sample_changes": changes,
    }


def best_row(rows: list[dict[str, Any]], require_zero_harm: bool = False) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("gate_type") == "observable"
        and row.get("selected_exact_count_after_gate") is not None
        and (not require_zero_harm or row.get("harms_vs_V0") == 0)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["net_exact_delta_vs_V0"],
            row["selected_exact_count_after_gate"],
            row["selected_p2_oracle_count_after_gate"],
            -row["harms_vs_V0"],
            row["variant"],
            row["gate"],
        ),
    )


def best_upper_bound(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("gate_type") == "oracle_upper_bound"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["net_exact_delta_vs_V0"],
            row["selected_exact_count_after_gate"],
            row["selected_p2_oracle_count_after_gate"],
            -row["harms_vs_V0"],
        ),
    )


def concise_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        key: row.get(key)
        for key in [
            "variant",
            "gate",
            "gate_type",
            "selected_exact_count_after_gate",
            "selected_relaxed_count_after_gate",
            "selected_p2_oracle_count_after_gate",
            "rescues_vs_V0",
            "harms_vs_V0",
            "net_exact_delta_vs_V0",
            "net_relaxed_delta_vs_V0",
            "preserves_all_V0_hits_bool",
            "implementable_bool",
            "uses_oracle_labels_bool",
        ]
    }


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Calibration Gate Feasibility",
        "",
        "## Scope",
        "",
        "Offline helper-only gate feasibility diagnostic over existing JSON artifacts. No model decoding, candidate generation, formal eval, scorer mode, or production default change was run.",
        "",
        "## Input All32 Smoke Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Target samples analyzed | {summary['target_samples_analyzed']} |",
        f"| V0 selected exact/relaxed | {summary['v0_selected_exact_count']} |",
        f"| V0 selected P2 oracle | {summary['v0_selected_p2_oracle_count']} |",
        f"| V0 selected hit samples | {summary['v0_selected_oracle_samples']} |",
        "",
        "## Variants Considered",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant}`")
    lines.extend(
        [
            "",
            "## Observable Gates",
            "",
            "| Gate | Status | Reason |",
            "| --- | --- | --- |",
        ]
    )
    unavailable = result["unavailable_gates"]
    for gate in OBSERVABLE_GATES:
        reason = unavailable.get(gate)
        lines.append(
            f"| `{gate}` | {'unavailable' if reason else 'available'} | {reason or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Best Observable Gates",
            "",
            "| Variant | Gate | Exact | Relaxed | P2 oracle | Rescues | Harms | Net | Zero harm |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    observable_rows = [
        row
        for row in result["gate_results"]
        if row["gate_type"] == "observable" and row["selected_exact_count_after_gate"] is not None
    ]
    for row in sorted(
        observable_rows,
        key=lambda item: (
            -item["net_exact_delta_vs_V0"],
            item["harms_vs_V0"],
            item["variant"],
            item["gate"],
        ),
    )[:12]:
        lines.append(
            "| {variant} | `{gate}` | {selected_exact_count_after_gate} | {selected_relaxed_count_after_gate} | {selected_p2_oracle_count_after_gate} | {rescues_vs_V0} | {harms_vs_V0} | {net_exact_delta_vs_V0} | {preserves_all_V0_hits_bool} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Oracle Upper Bounds",
            "",
            "These gates use oracle labels or known oracle-recovered samples and are not implementable.",
            "",
            "| Variant | Gate | Exact | Relaxed | P2 oracle | Rescues | Harms | Net | Zero harm |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in [item for item in result["gate_results"] if item["gate_type"] == "oracle_upper_bound"]:
        lines.append(
            "| {variant} | `{gate}` | {selected_exact_count_after_gate} | {selected_relaxed_count_after_gate} | {selected_p2_oracle_count_after_gate} | {rescues_vs_V0} | {harms_vs_V0} | {net_exact_delta_vs_V0} | {preserves_all_V0_hits_bool} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval, no rerank mode, and no P2 eval integration unless an observable zero-harm gate exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    all32 = load_json(args.all32_smoke_json)
    sidecar = load_json(args.scorer_ready_json)
    counterfactual = load_json(args.counterfactual_json)
    target_samples = parse_target_samples(args.target_samples)
    samples_by_idx = by_sample(all32)
    candidates_by_sample = candidate_lookup(sidecar)
    residual_path = Path("p2_oracle_score_miss_residual_diagnostics.json")
    residual_features = residual_feature_by_sample(load_json(residual_path)) if residual_path.is_file() else {}
    p2_target_samples = set(int(idx) for idx in all32.get("summary", {}).get("p2_target_samples", []))

    missing_samples = [idx for idx in target_samples if idx not in samples_by_idx]
    if missing_samples:
        raise ValueError(f"missing target samples in all32 smoke JSON: {missing_samples}")

    unavailable: dict[str, str] = {}
    available: dict[str, bool] = {}
    for gate in OBSERVABLE_GATES:
        is_available, reason = gate_is_available(gate, residual_features, target_samples)
        available[gate] = is_available
        if reason:
            unavailable[gate] = reason

    gate_results = []
    for variant in VARIANTS:
        for gate in OBSERVABLE_GATES:
            gate_results.append(
                evaluate_gate(
                    variant,
                    gate,
                    "observable",
                    samples_by_idx,
                    target_samples,
                    candidates_by_sample,
                    residual_features,
                    p2_target_samples,
                    available=available[gate],
                    unavailable_reason=unavailable.get(gate),
                )
            )
        for gate in UPPER_BOUND_GATES:
            gate_results.append(
                evaluate_gate(
                    variant,
                    gate,
                    "oracle_upper_bound",
                    samples_by_idx,
                    target_samples,
                    candidates_by_sample,
                    residual_features,
                    p2_target_samples,
                )
            )

    observable_available_rows = [
        row
        for row in gate_results
        if row["gate_type"] == "observable" and row["selected_exact_count_after_gate"] is not None
    ]
    observable_positive_zero_harm = [
        row
        for row in observable_available_rows
        if row["net_exact_delta_vs_V0"] > 0 and row["harms_vs_V0"] == 0
    ]
    observable_positive = [row for row in observable_available_rows if row["net_exact_delta_vs_V0"] > 0]
    observable_p2_gain = [
        row
        for row in observable_available_rows
        if row["selected_p2_oracle_count_after_gate"]
        > all32["summary"]["variant_summaries"][V0]["selected_is_p2_oracle_count"]
    ]
    best_observable = best_row(gate_results)
    best_observable_zero_harm = best_row(gate_results, require_zero_harm=True)
    best_upper = best_upper_bound(gate_results)

    if observable_positive_zero_harm:
        decision = "A"
        recommendation = "At least one observable gate gives positive selected recovery with zero V0 hit harms; a later tiny implementation smoke may be justified, still no formal eval."
    elif observable_positive or observable_p2_gain:
        decision = "B"
        recommendation = "Observable gates rescue some cases or improve net, but positive gates harm V0 hits or fail the zero-harm criterion. Do not implement; close fixed residual calibration implementation path for now."
    elif best_upper and best_upper["net_exact_delta_vs_V0"] > 0:
        decision = "C"
        recommendation = "Only oracle-label upper-bound gates look useful. This is diagnostic-only and not implementable; close fixed residual calibration implementation path."
    else:
        decision = "B"
        recommendation = "No observable gate provides safe improvement. Close fixed residual calibration implementation path for now."

    v0_summary = all32["summary"]["variant_summaries"][V0]
    result = {
        "metadata": {
            "all32_smoke_json": str(args.all32_smoke_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "counterfactual_json": str(args.counterfactual_json),
            "target_samples": target_samples,
            "model_decoding_run": False,
            "candidate_generation_run": False,
            "formal_eval_run": False,
        },
        "summary": {
            "target_samples_analyzed": len(target_samples),
            "variants_considered": VARIANTS,
            "observable_gates_tested": OBSERVABLE_GATES,
            "oracle_upper_bound_gates_tested": UPPER_BOUND_GATES,
            "unavailable_gates": unavailable,
            "observable_gate_count": len(OBSERVABLE_GATES) - len(unavailable),
            "unavailable_gate_count": len(unavailable),
            "v0_selected_exact_count": v0_summary["selected_exact_count"],
            "v0_selected_relaxed_count": v0_summary["selected_relaxed_no_constants_count"],
            "v0_selected_p2_oracle_count": v0_summary["selected_is_p2_oracle_count"],
            "v0_selected_oracle_samples": v0_summary["selected_oracle_samples"],
            "best_observable_gate_by_net": concise_row(best_observable),
            "best_observable_gate_preserving_all_V0_hits": concise_row(best_observable_zero_harm),
            "best_oracle_upper_bound_gate": concise_row(best_upper),
            "any_observable_gate_positive_and_zero_harm": bool(observable_positive_zero_harm),
            "any_observable_gate_improves_selected_exact": bool(observable_positive),
            "any_observable_gate_increases_p2_oracle_selected": bool(observable_p2_gain),
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
        "unavailable_gates": unavailable,
        "gate_results": gate_results,
    }

    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_report(json_ready(result)))
    print(f"decision={decision}")
    print(f"target_samples_analyzed={len(target_samples)}")
    print(f"observable_gate_count={result['summary']['observable_gate_count']}")
    print(f"unavailable_gate_count={result['summary']['unavailable_gate_count']}")
    print(f"any_observable_gate_positive_and_zero_harm={result['summary']['any_observable_gate_positive_and_zero_harm']}")
    print(f"best_observable_gate_by_net={result['summary']['best_observable_gate_by_net']}")


if __name__ == "__main__":
    main()
