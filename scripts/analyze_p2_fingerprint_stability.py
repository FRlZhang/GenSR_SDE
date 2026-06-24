#!/usr/bin/env python3
"""Offline fingerprint simulation stability diagnostic for V0/V4 cases."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_p2_admitted_candidate_scoring import json_ready, score_seed, target_array
from analyze_p2_calibration_all32_smoke import ACTIVE_SLICE, CONSTANT_VALUES, GLOBAL_ACTIVE_DIMS
from analyze_p2_fingerprint_ambiguity import (
    FINGERPRINT_CONFIG,
    REMOVED_ACTIVE_DIMS,
    candidate_brief,
    candidate_lookup,
    finite,
    l2_without,
    token_text,
)
from sde_validation_probe import score_candidate_system


V0 = "V0_current_rolewise_no_multi_u0"
V4 = "V4_active_without_dims_76_72_82_88"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambiguity-json", type=Path, required=True)
    parser.add_argument("--all32-smoke-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--target-case-types", type=str, default="V4_rescue,V4_harm")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sample_by_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in payload.get("samples", [])}


def parse_case_types(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(finite_values),
        "std": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
    }


def coefficient_of_variation(mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or abs(mean) < EPS:
        return None
    return abs(std / mean)


def score_tokens_against_target(
    tokens: list[str],
    target_y: np.ndarray,
    sample_index: int,
    candidate_index: int,
    repeat: int,
) -> dict[str, Any]:
    details, failure_reason, fingerprint_failures = score_candidate_system(
        tokens,
        target_y,
        FINGERPRINT_CONFIG,
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(sample_index, candidate_index + repeat * 997),
    )
    if details is None:
        return {
            "valid_bool": False,
            "failure_reason": failure_reason,
            "fingerprint_failures": fingerprint_failures,
            "total_score": None,
            "active_score": None,
            "active_score_after_removed_dims": None,
            "weak_score": None,
            "best_drift_constant": None,
            "best_diffusion_constant": None,
            "active_residual": None,
            "removed_dim_residuals": None,
            "removed_dim_norm": None,
        }
    active_residual = details.get("active_kramers_moyal_residual")
    active_removed = l2_without(active_residual, GLOBAL_ACTIVE_DIMS)
    removed_values = removed_dim_residuals(active_residual)
    weak = details.get("gaussian_weak_kernel_distance")
    return {
        "valid_bool": True,
        "failure_reason": None,
        "fingerprint_failures": fingerprint_failures,
        "total_score": details.get("distance"),
        "active_score": details.get("active_kramers_moyal_distance"),
        "active_score_after_removed_dims": active_removed,
        "weak_score": weak,
        "v4_score": None if active_removed is None or weak is None else active_removed + weak,
        "best_drift_constant": details.get("best_drift_constant"),
        "best_diffusion_constant": details.get("best_diffusion_constant"),
        "active_residual": active_residual,
        "removed_dim_residuals": removed_values,
        "removed_dim_norm": removed_norm(removed_values),
    }


def removed_dim_residuals(active_residual: Any) -> dict[int, float] | None:
    if active_residual is None:
        return None
    vec = np.asarray(active_residual, dtype=np.float64)
    values: dict[int, float] = {}
    for dim in REMOVED_ACTIVE_DIMS:
        local = dim - ACTIVE_SLICE.start
        if 0 <= local < len(vec):
            values[dim] = float(vec[local])
    return values


def removed_norm(values: dict[int, float] | None) -> float | None:
    if not values:
        return None
    return float(np.linalg.norm(np.asarray(list(values.values()), dtype=np.float64)))


def score_series(
    sample: dict[str, Any],
    row: dict[str, Any],
    candidate_index: int,
    repeats: int,
) -> list[dict[str, Any]]:
    target_y = target_array(sample)
    tokens = row.get("full_sequence_tokens") or []
    return [
        score_tokens_against_target(tokens, target_y, int(sample["sample_index"]), candidate_index, repeat)
        for repeat in range(repeats)
    ]


def stored_score_payload(ambiguity_sample: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    candidates = [
        ambiguity_sample.get("V0_selected_score_components_under_V0"),
        ambiguity_sample.get("V4_selected_score_components_under_V4"),
        ambiguity_sample.get("oracle_score_components_under_V0"),
        ambiguity_sample.get("oracle_score_components_under_V4"),
        ambiguity_sample.get("nonoracle_score_components_under_V0"),
        ambiguity_sample.get("nonoracle_score_components_under_V4"),
    ]
    for item in candidates:
        if item and item.get("candidate_id") == candidate_id:
            return item
    return None


def summarize_candidate(row: dict[str, Any], series: list[dict[str, Any]], stored: dict[str, Any] | None) -> dict[str, Any]:
    total = mean_std([item.get("v4_score") for item in series])
    active = mean_std([item.get("active_score_after_removed_dims") for item in series])
    weak = mean_std([item.get("weak_score") for item in series])
    first_valid = next((item for item in series if item.get("valid_bool")), {})
    return {
        "candidate": candidate_brief(row),
        "stored_score": stored,
        "baseline_total_score": stored.get("total_score") if stored else None,
        "baseline_active_score": stored.get("active_score") if stored else None,
        "baseline_weak_score": stored.get("weak_score") if stored else None,
        "baseline_best_drift_constant": stored.get("best_drift_constant") if stored else None,
        "baseline_best_diffusion_constant": stored.get("best_diffusion_constant") if stored else None,
        "mean_total_score": total["mean"],
        "std_total_score": total["std"],
        "cv_total_score": coefficient_of_variation(total["mean"], total["std"]),
        "mean_active_score": active["mean"],
        "std_active_score": active["std"],
        "mean_weak_score": weak["mean"],
        "std_weak_score": weak["std"],
        "first_repeat_best_drift_constant": first_valid.get("best_drift_constant"),
        "first_repeat_best_diffusion_constant": first_valid.get("best_diffusion_constant"),
        "valid_repeats": sum(1 for item in series if item.get("valid_bool")),
    }


def sign(value: float | None) -> int:
    if value is None:
        return 0
    if value < -1e-10:
        return -1
    if value > 1e-10:
        return 1
    return 0


def gap_values(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for left_row, right_row in zip(left, right):
        lval = left_row.get(key)
        rval = right_row.get(key)
        values.append(None if not finite(lval) or not finite(rval) else float(lval) - float(rval))
    return values


def segment_noise_driver(active_std: float | None, weak_std: float | None) -> str:
    if active_std is None or weak_std is None:
        return "unknown"
    if active_std >= 2.0 * weak_std:
        return "active"
    if weak_std >= 2.0 * active_std:
        return "weak"
    return "mixed"


def pair_metrics(
    name: str,
    left_label: str,
    right_label: str,
    left_series: list[dict[str, Any]],
    right_series: list[dict[str, Any]],
    baseline_gap: float | None,
    oracle_label: str | None,
) -> dict[str, Any]:
    total_gaps = gap_values(left_series, right_series, "v4_score")
    active_gaps = gap_values(left_series, right_series, "active_score_after_removed_dims")
    weak_gaps = gap_values(left_series, right_series, "weak_score")
    total = mean_std(total_gaps)
    active = mean_std(active_gaps)
    weak = mean_std(weak_gaps)
    signs = [sign(value) for value in total_gaps if value is not None]
    nonzero = [item for item in signs if item != 0]
    stable = bool(nonzero) and len(set(nonzero)) == 1
    sign_flip_count = 0 if not nonzero else len(nonzero) - max(Counter(nonzero).values())
    std = total["std"]
    mean = total["mean"]
    noise_dominated = (
        bool(mean is not None and std is not None and abs(mean) <= 2.0 * std)
    )
    left_wins = sum(1 for value in total_gaps if value is not None and value < 0)
    right_wins = sum(1 for value in total_gaps if value is not None and value > 0)
    ties = sum(1 for value in total_gaps if value is not None and sign(value) == 0)
    oracle_wins = None
    selected_wins = None
    if oracle_label == left_label:
        oracle_wins = left_wins
        selected_wins = right_wins
    elif oracle_label == right_label:
        oracle_wins = right_wins
        selected_wins = left_wins
    active_signs = [sign(value) for value in active_gaps if value is not None]
    weak_signs = [sign(value) for value in weak_gaps if value is not None]
    return {
        "pair_name": name,
        "left_label": left_label,
        "right_label": right_label,
        "baseline_gap": baseline_gap,
        "mean_resim_gap": mean,
        "std_resim_gap": std,
        "gap_z_score": None if mean is None or std is None else abs(mean) / (std + EPS),
        "gap_sign_flip_count": sign_flip_count,
        "gap_sign_flip_fraction": None if not nonzero else sign_flip_count / len(nonzero),
        "ordering_stable_bool": stable,
        "ordering_stable_count": 1 if stable else 0,
        "ordering_stable_fraction": 1.0 if stable else 0.0,
        "oracle_wins_count": oracle_wins,
        "selected_wins_count": selected_wins,
        "left_wins_count": left_wins,
        "right_wins_count": right_wins,
        "tie_or_near_tie_count": ties,
        "active_gap_mean": active["mean"],
        "active_gap_std": active["std"],
        "weak_gap_mean": weak["mean"],
        "weak_gap_std": weak["std"],
        "active_gap_sign_flip_fraction": sign_flip_fraction(active_signs),
        "weak_gap_sign_flip_fraction": sign_flip_fraction(weak_signs),
        "noise_dominated_bool": noise_dominated,
        "segment_noise_driver": segment_noise_driver(active["std"], weak["std"]),
        "stable_clear_bool": stable and not noise_dominated,
    }


def sign_flip_fraction(signs: list[int]) -> float | None:
    nonzero = [item for item in signs if item != 0]
    if not nonzero:
        return None
    return (len(nonzero) - max(Counter(nonzero).values())) / len(nonzero)


def removed_dim_stability(
    selected_series: list[dict[str, Any]],
    oracle_series: list[dict[str, Any]],
) -> dict[str, Any]:
    per_dim: dict[str, dict[str, Any]] = {}
    for dim in REMOVED_ACTIVE_DIMS:
        selected_values = [
            item.get("removed_dim_residuals", {}).get(dim)
            for item in selected_series
            if item.get("removed_dim_residuals")
        ]
        oracle_values = [
            item.get("removed_dim_residuals", {}).get(dim)
            for item in oracle_series
            if item.get("removed_dim_residuals")
        ]
        gaps = [
            None if not finite(oval) or not finite(sval) else float(oval) - float(sval)
            for oval, sval in zip(oracle_values, selected_values)
        ]
        gap_stats = mean_std(gaps)
        per_dim[str(dim)] = {
            "selected_residual_mean": mean_std(selected_values)["mean"],
            "selected_residual_std": mean_std(selected_values)["std"],
            "oracle_residual_mean": mean_std(oracle_values)["mean"],
            "oracle_residual_std": mean_std(oracle_values)["std"],
            "residual_gap_mean": gap_stats["mean"],
            "residual_gap_std": gap_stats["std"],
            "sign_flip_fraction": sign_flip_fraction([sign(value) for value in gaps if value is not None]),
        }
    selected_norms = [item.get("removed_dim_norm") for item in selected_series]
    oracle_norms = [item.get("removed_dim_norm") for item in oracle_series]
    contribution = [
        None if not finite(oval) or not finite(sval) else float(oval) - float(sval)
        for oval, sval in zip(oracle_norms, selected_norms)
    ]
    stats = mean_std(contribution)
    signs = [sign(value) for value in contribution if value is not None]
    return {
        "removed_dims": REMOVED_ACTIVE_DIMS,
        "per_dim": per_dim,
        "removed_dims_contribution_mean": stats["mean"],
        "removed_dims_contribution_std": stats["std"],
        "removed_dims_contribution_sign_flip_fraction": sign_flip_fraction(signs),
        "removed_dims_noise_dominated_bool": (
            stats["mean"] is not None
            and stats["std"] is not None
            and abs(stats["mean"]) <= 2.0 * stats["std"]
        ),
        "removed_dims_stable_explanation_bool": (
            stats["mean"] is not None
            and stats["std"] is not None
            and abs(stats["mean"]) > 2.0 * stats["std"]
            and (sign_flip_fraction(signs) or 0.0) == 0.0
        ),
    }


def dedupe_candidates(rows: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not row:
            continue
        candidate_id = str(row.get("candidate_id"))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(row)
    return result


def candidate_from_brief(brief: dict[str, Any] | None, lookup: dict[str, tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    if not brief:
        return None
    candidate_id = brief.get("candidate_id")
    if candidate_id is None or str(candidate_id) not in lookup:
        return None
    return lookup[str(candidate_id)][1]


def build_sample_stability(
    ambiguity_sample: dict[str, Any],
    sidecar_sample: dict[str, Any],
    smoke_sample: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    idx = int(ambiguity_sample["sample_index"])
    lookup = candidate_lookup(sidecar_sample)
    v0_row = candidate_from_brief(ambiguity_sample.get("V0_selected_candidate"), lookup)
    v4_row = candidate_from_brief(ambiguity_sample.get("V4_selected_candidate"), lookup)
    exact_oracle = candidate_from_brief(ambiguity_sample.get("exact_oracle_candidate"), lookup)
    p2_oracle = candidate_from_brief(ambiguity_sample.get("p2_oracle_candidate"), lookup)
    best_oracle = exact_oracle or p2_oracle or candidate_from_brief(
        ambiguity_sample.get("best_current_expanded_oracle_candidate"), lookup
    )
    candidates = dedupe_candidates([v0_row, v4_row, best_oracle, p2_oracle])
    series: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        candidate_index = lookup[candidate_id][0]
        series[candidate_id] = score_series(sidecar_sample, row, candidate_index, repeats)
        summaries[candidate_id] = summarize_candidate(
            row,
            series[candidate_id],
            stored_score_payload(ambiguity_sample, candidate_id),
        )
    oracle_row = best_oracle
    oracle_id = str(oracle_row["candidate_id"]) if oracle_row else None
    v0_id = str(v0_row["candidate_id"]) if v0_row else None
    v4_id = str(v4_row["candidate_id"]) if v4_row else None
    pairs = []
    gap_summaries = ambiguity_sample.get("gap_summaries", {})
    if v0_id and oracle_id and v0_id != oracle_id:
        pairs.append(
            pair_metrics(
                "V0_selected_vs_oracle",
                "V0_selected",
                "oracle",
                series[v0_id],
                series[oracle_id],
                gap_summaries.get("oracle_vs_nonoracle_gap_under_V0"),
                "oracle",
            )
        )
    if v4_id and oracle_id and v4_id != oracle_id:
        pairs.append(
            pair_metrics(
                "V4_selected_vs_oracle",
                "V4_selected",
                "oracle",
                series[v4_id],
                series[oracle_id],
                gap_summaries.get("oracle_vs_nonoracle_gap_under_V4"),
                "oracle",
            )
        )
    if v0_id and v4_id and v0_id != v4_id:
        pairs.append(
            pair_metrics(
                "V0_selected_vs_V4_selected",
                "V0_selected",
                "V4_selected",
                series[v0_id],
                series[v4_id],
                None,
                "V4_selected" if smoke_sample["variants"][V4].get("selected_is_oracle_bool") else (
                    "V0_selected" if smoke_sample["variants"][V0].get("selected_is_oracle_bool") else None
                ),
            )
        )
    selected_for_removed = v0_row if ambiguity_sample["case_type"] == "V4_rescue" else v4_row
    removed = None
    if selected_for_removed and oracle_id:
        selected_id = str(selected_for_removed["candidate_id"])
        if selected_id != oracle_id:
            removed = removed_dim_stability(series[selected_id], series[oracle_id])
    return {
        "sample_index": idx,
        "case_type": ambiguity_sample["case_type"],
        "truth_sequence": ambiguity_sample.get("truth_sequence"),
        "truth_drift_tokens": ambiguity_sample.get("truth_drift_tokens"),
        "truth_diffusion_tokens": ambiguity_sample.get("truth_diffusion_tokens"),
        "candidates": summaries,
        "comparison_pairs": pairs,
        "removed_dimension_stability": removed,
        "stability_modes": {
            "M0_stored_target_stored_candidate": "available_from_previous_score_fields",
            "M1_stored_target_resim_candidate": "available",
            "M2_resim_target_stored_candidate_if_possible": "unavailable: sidecar does not store candidate fingerprint vectors",
            "M3_paired_resim_target_and_candidate": "unavailable: target raw numeric SDE constants are not stored, only CONSTANT-token templates and target fingerprint vectors",
        },
    }


def aggregate(samples: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    pairs = [pair for sample in samples for pair in sample.get("comparison_pairs", [])]
    primary_pairs = [
        pair for pair in pairs
        if pair["pair_name"] in {"V0_selected_vs_oracle", "V4_selected_vs_oracle"}
    ]
    removed = [
        sample["removed_dimension_stability"]
        for sample in samples
        if sample.get("removed_dimension_stability")
    ]
    driver_counts = Counter(pair.get("segment_noise_driver") for pair in primary_pairs)
    case_by_idx = {sample["sample_index"]: sample["case_type"] for sample in samples}
    rescue_noise = 0
    harm_noise = 0
    rescue_oracle_majority = 0
    harm_oracle_majority = 0
    for pair in primary_pairs:
        case = case_by_idx.get(pair.get("sample_index"))
        # sample_index is attached just below for aggregate convenience.
        if case is None:
            continue
        if pair.get("noise_dominated_bool"):
            if case == "V4_rescue":
                rescue_noise += 1
            elif case == "V4_harm":
                harm_noise += 1
        if pair.get("oracle_wins_count") is not None and pair["oracle_wins_count"] > repeats / 2:
            if case == "V4_rescue":
                rescue_oracle_majority += 1
            elif case == "V4_harm":
                harm_oracle_majority += 1
    m1_stable = sum(1 for pair in primary_pairs if pair.get("ordering_stable_bool"))
    m1_unstable = len(primary_pairs) - m1_stable
    noise_dominated = sum(1 for pair in primary_pairs if pair.get("noise_dominated_bool"))
    stable_clear = sum(1 for pair in primary_pairs if pair.get("stable_clear_bool"))
    removed_noise = sum(1 for row in removed if row.get("removed_dims_noise_dominated_bool"))
    removed_stable = sum(1 for row in removed if row.get("removed_dims_stable_explanation_bool"))
    if m1_unstable > m1_stable or noise_dominated > stable_clear:
        decision = "B"
        recommendation = (
            "Ordering instability is visible with stored target and candidate resimulation. "
            "Diagnose candidate fingerprint variance or path budget before scorer changes."
        )
    else:
        decision = "A"
        recommendation = (
            "Ordering is mostly stable under candidate resimulation. Return to candidate-generation / drift-span diagnostics; scorer changes remain unsupported."
        )
    return {
        "samples_analyzed": len(samples),
        "candidate_pairs_analyzed": len(primary_pairs),
        "all_pair_comparisons_analyzed": len(pairs),
        "repeats": repeats,
        "stability_modes": {
            "M0_stored_target_stored_candidate": "available",
            "M1_stored_target_resim_candidate": "available",
            "M2_resim_target_stored_candidate_if_possible": "unavailable",
            "M3_paired_resim_target_and_candidate": "unavailable",
        },
        "unavailable_modes": {
            "M2_resim_target_stored_candidate_if_possible": "sidecar does not store candidate fingerprint vectors",
            "M3_paired_resim_target_and_candidate": "raw numeric target SDE constants are not stored, so target fingerprint Monte Carlo resimulation is not cleanly reproducible",
        },
        "M1_ordering_stable_count": m1_stable,
        "M1_ordering_unstable_count": m1_unstable,
        "M3_ordering_stable_count": None,
        "M3_ordering_unstable_count": None,
        "noise_dominated_pair_count": noise_dominated,
        "stable_clear_pair_count": stable_clear,
        "active_noise_driver_count": driver_counts["active"],
        "weak_noise_driver_count": driver_counts["weak"],
        "mixed_noise_driver_count": driver_counts["mixed"],
        "unknown_noise_driver_count": driver_counts["unknown"],
        "rescue_cases_noise_dominated_count": rescue_noise,
        "harm_cases_noise_dominated_count": harm_noise,
        "rescue_cases_oracle_recovers_majority_count": rescue_oracle_majority,
        "harm_cases_oracle_recovers_majority_count": harm_oracle_majority,
        "removed_dims_noise_dominated_count": removed_noise,
        "removed_dims_stable_explanation_count": removed_stable,
        "decision": decision,
        "formal_eval_recommended": False,
        "recommendation": recommendation,
        "model_decoding_avoided": True,
        "candidate_generation_avoided": True,
        "sde_validation_probe_avoided": True,
    }


def attach_sample_index(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        for pair in sample.get("comparison_pairs", []):
            pair["sample_index"] = sample["sample_index"]
            pair["case_type"] = sample["case_type"]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Fingerprint Stability Diagnostics",
        "",
        "## Why This Diagnostic Was Run",
        "",
        "The previous V0/V4 fingerprint ambiguity report found stable ordering in only",
        "2/7 small resimulation comparisons. This helper narrows that result to",
        "fingerprint simulation stability for the 5 V4 rescues and 2 V4 harms. It",
        "does not run model decoding, candidate generation, `sde_validation_probe.py`,",
        "formal eval, scorer grids, rerank-mode implementation, or production default",
        "changes.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Samples analyzed | {summary['samples_analyzed']} |",
        f"| Candidate oracle pairs analyzed | {summary['candidate_pairs_analyzed']} |",
        f"| Repeats | {summary['repeats']} |",
        f"| M1 ordering stable | {summary['M1_ordering_stable_count']} |",
        f"| M1 ordering unstable | {summary['M1_ordering_unstable_count']} |",
        f"| Noise-dominated pairs | {summary['noise_dominated_pair_count']} |",
        f"| Stable clear pairs | {summary['stable_clear_pair_count']} |",
        f"| Active noise driver | {summary['active_noise_driver_count']} |",
        f"| Weak noise driver | {summary['weak_noise_driver_count']} |",
        f"| Mixed noise driver | {summary['mixed_noise_driver_count']} |",
        "",
        "## Stability Modes",
        "",
        "| Mode | Status |",
        "| --- | --- |",
    ]
    for mode, status in summary["stability_modes"].items():
        reason = summary.get("unavailable_modes", {}).get(mode)
        lines.append(f"| {mode} | {status if reason is None else status + ': ' + reason} |")
    lines.extend(
        [
            "",
            "## Per-Sample Ordering Stability",
            "",
            "| Sample | Case | Pair | Stable | Oracle wins | Selected wins | Noise dominated | Driver | Mean gap | Std gap | Z |",
            "| ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for sample in result["samples"]:
        for pair in sample.get("comparison_pairs", []):
            if pair["pair_name"] not in {"V0_selected_vs_oracle", "V4_selected_vs_oracle"}:
                continue
            lines.append(
                "| {idx} | {case} | {pair} | {stable} | {oracle} | {selected} | {noise} | {driver} | {mean} | {std} | {z} |".format(
                    idx=sample["sample_index"],
                    case=sample["case_type"],
                    pair=pair["pair_name"],
                    stable=fmt(pair.get("ordering_stable_bool")),
                    oracle=fmt(pair.get("oracle_wins_count")),
                    selected=fmt(pair.get("selected_wins_count")),
                    noise=fmt(pair.get("noise_dominated_bool")),
                    driver=pair.get("segment_noise_driver"),
                    mean=fmt(pair.get("mean_resim_gap")),
                    std=fmt(pair.get("std_resim_gap")),
                    z=fmt(pair.get("gap_z_score")),
                )
            )
    lines.extend(
        [
            "",
            "## Gap Mean/Std/Z Score",
            "",
            "| Sample | Pair | Baseline gap | Active gap mean/std | Weak gap mean/std | Sign flip fraction |",
            "| ---: | --- | ---: | --- | --- | ---: |",
        ]
    )
    for sample in result["samples"]:
        for pair in sample.get("comparison_pairs", []):
            if pair["pair_name"] not in {"V0_selected_vs_oracle", "V4_selected_vs_oracle"}:
                continue
            lines.append(
                "| {idx} | {pair} | {baseline} | {amean}/{astd} | {wmean}/{wstd} | {flip} |".format(
                    idx=sample["sample_index"],
                    pair=pair["pair_name"],
                    baseline=fmt(pair.get("baseline_gap")),
                    amean=fmt(pair.get("active_gap_mean")),
                    astd=fmt(pair.get("active_gap_std")),
                    wmean=fmt(pair.get("weak_gap_mean")),
                    wstd=fmt(pair.get("weak_gap_std")),
                    flip=fmt(pair.get("gap_sign_flip_fraction")),
                )
            )
    lines.extend(
        [
            "",
            "## Removed-Dimension Stability",
            "",
            "| Sample | Case | Contribution mean | Contribution std | Sign flip fraction | Noise dominated | Stable explanation |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for sample in result["samples"]:
        removed = sample.get("removed_dimension_stability")
        if not removed:
            continue
        lines.append(
            "| {idx} | {case} | {mean} | {std} | {flip} | {noise} | {stable} |".format(
                idx=sample["sample_index"],
                case=sample["case_type"],
                mean=fmt(removed.get("removed_dims_contribution_mean")),
                std=fmt(removed.get("removed_dims_contribution_std")),
                flip=fmt(removed.get("removed_dims_contribution_sign_flip_fraction")),
                noise=fmt(removed.get("removed_dims_noise_dominated_bool")),
                stable=fmt(removed.get("removed_dims_stable_explanation_bool")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "M1 is available and isolates candidate fingerprint resimulation with the",
            "stored target fingerprint fixed. M2 is unavailable because candidate",
            "fingerprint vectors are not stored. M3 is unavailable because the sidecar",
            "stores target fingerprint vectors but not the raw numeric target SDE",
            "constants needed to resimulate the target truth process exactly.",
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval, no rerank mode, no P2 eval integration, and no scorer",
            "change are recommended from this diagnostic alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ambiguity_payload = load_json(args.ambiguity_json)
    smoke_payload = load_json(args.all32_smoke_json)
    sidecar_payload = load_json(args.scorer_ready_json)
    target_case_types = parse_case_types(args.target_case_types)
    smoke_by_idx = sample_by_index(smoke_payload)
    sidecar_by_idx = sample_by_index(sidecar_payload)
    samples = []
    for ambiguity_sample in ambiguity_payload.get("samples", []):
        if ambiguity_sample.get("case_type") not in target_case_types:
            continue
        idx = int(ambiguity_sample["sample_index"])
        samples.append(
            build_sample_stability(
                ambiguity_sample,
                sidecar_by_idx[idx],
                smoke_by_idx[idx],
                args.repeats,
            )
        )
    attach_sample_index(samples)
    summary = aggregate(samples, args.repeats)
    result = {
        "metadata": {
            "ambiguity_json": str(args.ambiguity_json),
            "all32_smoke_json": str(args.all32_smoke_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "target_case_types": sorted(target_case_types),
            "fingerprint_config": FINGERPRINT_CONFIG.to_dict(),
            "score_kind": "constant_grid_rolewise_no_multi_u0",
            "active_weak_weights": [1.0, 1.0],
            "constant_values": CONSTANT_VALUES,
            "removed_active_dims": REMOVED_ACTIVE_DIMS,
        },
        "summary": summary,
        "samples": samples,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_markdown(json_ready(result)))
    print(f"decision={summary['decision']}")
    print(f"samples_analyzed={summary['samples_analyzed']}")
    print(f"candidate_pairs_analyzed={summary['candidate_pairs_analyzed']}")
    print(f"repeats={summary['repeats']}")
    print(f"M1_ordering_stable_count={summary['M1_ordering_stable_count']}")
    print(f"M1_ordering_unstable_count={summary['M1_ordering_unstable_count']}")
    print(f"noise_dominated_pair_count={summary['noise_dominated_pair_count']}")
    print(f"stable_clear_pair_count={summary['stable_clear_pair_count']}")


if __name__ == "__main__":
    main()
