#!/usr/bin/env python3
"""Offline active/weak variance anatomy for P2 candidate fingerprints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
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
from analyze_p2_candidate_fingerprint_path_budget import (
    CONSTANT_VALUES,
    build_pair_specs,
    load_json,
    parse_case_types,
    sample_by_index,
    unique_candidate_calls,
)
from analyze_p2_fingerprint_ambiguity import candidate_brief, finite
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system


EPS = 1e-12
MAX_CODEX_SECONDS = 600.0
ACTIVE_SLICE_START = 72
WEAK_SLICE_START = 90
RECURRING_ACTIVE_DIMS = [76, 72, 82, 88]
RECURRING_WEAK_DIMS = [136, 137, 133]


@dataclass(frozen=True)
class BudgetMode:
    name: str
    n_paths: int
    active_paths: int
    n_steps: int
    repeats: int = 3

    @property
    def config(self) -> FingerprintConfig:
        return FingerprintConfig(
            n_paths=self.n_paths,
            active_paths=self.active_paths,
            n_steps=self.n_steps,
        )


AVAILABLE_BUDGETS = {
    "B0_current_budget": BudgetMode("B0_current_budget", n_paths=800, active_paths=800, n_steps=60),
    "B1_active_paths_high": BudgetMode("B1_active_paths_high", n_paths=800, active_paths=1600, n_steps=60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-budget-json", type=Path, required=True)
    parser.add_argument("--stability-json", type=Path, required=True)
    parser.add_argument("--ambiguity-json", type=Path, required=True)
    parser.add_argument("--all32-smoke-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--budget-modes", type=str, default="B0_current_budget,B1_active_paths_high")
    parser.add_argument("--target-case-types", type=str, default="V4_rescue,V4_harm")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def parse_budget_modes(raw: str) -> list[BudgetMode]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    budgets = []
    for name in names:
        if name not in AVAILABLE_BUDGETS:
            raise ValueError(f"unsupported budget mode for this diagnostic: {name}")
        budgets.append(AVAILABLE_BUDGETS[name])
    return budgets


def has_residual_vectors(path_budget_payload: dict[str, Any]) -> bool:
    for pair in path_budget_payload.get("pairs", []):
        for metrics in pair.get("budget_metrics", {}).values():
            if "active_residual" in metrics or "weak_residual" in metrics:
                return True
            if "selected_series" in metrics or "oracle_series" in metrics:
                return True
    return False


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(finite_values),
        "std": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
    }


def sign(value: float | None) -> int:
    if value is None:
        return 0
    if value < -1e-10:
        return -1
    if value > 1e-10:
        return 1
    return 0


def sign_flip_fraction(values: list[float | None]) -> float | None:
    signs = [sign(value) for value in values if value is not None and sign(value) != 0]
    if not signs:
        return None
    return (len(signs) - max(Counter(signs).values())) / len(signs)


def score_candidate_with_residuals(
    sample: dict[str, Any],
    row: dict[str, Any],
    candidate_index: int,
    budget: BudgetMode,
    repeat: int,
) -> dict[str, Any]:
    details, failure_reason, fingerprint_failures = score_candidate_system(
        row.get("full_sequence_tokens") or [],
        target_array(sample),
        budget.config,
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(int(sample["sample_index"]), candidate_index + repeat * 104729),
    )
    if details is None:
        return {
            "valid_bool": False,
            "failure_reason": failure_reason,
            "fingerprint_failures": fingerprint_failures,
            "total_score": None,
            "active_score": None,
            "weak_score": None,
            "active_residual": None,
            "weak_residual": None,
        }
    return {
        "valid_bool": True,
        "failure_reason": None,
        "fingerprint_failures": fingerprint_failures,
        "total_score": details.get("distance"),
        "active_score": details.get("active_kramers_moyal_distance"),
        "weak_score": details.get("gaussian_weak_kernel_distance"),
        "best_drift_constant": details.get("best_drift_constant"),
        "best_diffusion_constant": details.get("best_diffusion_constant"),
        "active_residual": details.get("active_kramers_moyal_residual"),
        "weak_residual": details.get("gaussian_weak_kernel_residual"),
    }


def estimate_runtime(
    pair_specs: list[dict[str, Any]],
    sidecar_payload: dict[str, Any],
    budgets: list[BudgetMode],
) -> dict[str, Any]:
    calls = unique_candidate_calls(pair_specs, sidecar_payload)
    if not calls:
        return {"estimated_seconds": 0.0, "single_call_seconds": 0.0, "score_calls": 0, "unique_candidates": 0}
    sample, row, candidate_index = calls[0]
    largest = max(budgets, key=lambda budget: budget.n_paths + budget.active_paths)
    start = time.monotonic()
    score_candidate_with_residuals(sample, row, candidate_index, largest, 0)
    elapsed = max(time.monotonic() - start, 1e-6)
    score_calls = len(calls) * sum(budget.repeats for budget in budgets)
    return {
        "estimated_seconds": elapsed * score_calls,
        "single_call_seconds": elapsed,
        "score_calls": score_calls,
        "unique_candidates": len(calls),
    }


def score_budget_mode(
    pair_specs: list[dict[str, Any]],
    sidecar_payload: dict[str, Any],
    budget: BudgetMode,
) -> dict[tuple[int, str], dict[str, list[dict[str, Any]]]]:
    sidecar_by_idx = sample_by_index(sidecar_payload)
    cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for spec in pair_specs:
        sample = sidecar_by_idx[int(spec["sample_index"])]
        for row_key, index_key in (("selected_row", "selected_candidate_index"), ("oracle_row", "oracle_candidate_index")):
            row = spec[row_key]
            key = (int(spec["sample_index"]), str(row["candidate_id"]))
            if key not in cache:
                cache[key] = [
                    score_candidate_with_residuals(sample, row, int(spec[index_key]), budget, repeat)
                    for repeat in range(budget.repeats)
                ]
    out = {}
    for spec in pair_specs:
        out[(int(spec["sample_index"]), spec["pair_type"])] = {
            "selected": cache[(int(spec["sample_index"]), str(spec["selected_row"]["candidate_id"]))],
            "oracle": cache[(int(spec["sample_index"]), str(spec["oracle_row"]["candidate_id"]))],
        }
    return out


def gap_values(oracle_series: list[dict[str, Any]], selected_series: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for oracle_row, selected_row in zip(oracle_series, selected_series):
        oval = oracle_row.get(key)
        sval = selected_row.get(key)
        values.append(None if not finite(oval) or not finite(sval) else float(oval) - float(sval))
    return values


def noise_share(active_std: float | None, weak_std: float | None) -> tuple[float | None, float | None]:
    if active_std is None or weak_std is None:
        return None, None
    denom = active_std * active_std + weak_std * weak_std
    if denom < EPS:
        return None, None
    return active_std * active_std / denom, weak_std * weak_std / denom


def segment_driver(active_std: float | None, weak_std: float | None) -> str:
    if active_std is None or weak_std is None:
        return "unknown"
    if active_std >= 2.0 * weak_std:
        return "active"
    if weak_std >= 2.0 * active_std:
        return "weak"
    return "mixed"


def pair_budget_metrics(spec: dict[str, Any], budget: BudgetMode, series: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected = series["selected"]
    oracle = series["oracle"]
    gaps = gap_values(oracle, selected, "total_score")
    active_gaps = gap_values(oracle, selected, "active_score")
    weak_gaps = gap_values(oracle, selected, "weak_score")
    gap_stats = mean_std(gaps)
    active_stats = mean_std(active_gaps)
    weak_stats = mean_std(weak_gaps)
    mean_gap = gap_stats["mean"]
    std_gap = gap_stats["std"]
    active_mean = active_stats["mean"]
    active_std = active_stats["std"]
    weak_mean = weak_stats["mean"]
    weak_std = weak_stats["std"]
    signs = [sign(value) for value in gaps if value is not None and sign(value) != 0]
    oracle_wins = sum(1 for value in gaps if value is not None and value < 0)
    selected_wins = sum(1 for value in gaps if value is not None and value > 0)
    stable = bool(signs) and len(set(signs)) == 1
    active_share, weak_share = noise_share(active_std, weak_std)
    return {
        "budget_mode": budget.name,
        "sample_index": spec["sample_index"],
        "case_type": spec["case_type"],
        "pair_type": spec["pair_type"],
        "selected_candidate_id": spec["selected_row"].get("candidate_id"),
        "oracle_candidate_id": spec["oracle_row"].get("candidate_id"),
        "selected_source": spec.get("selected_source"),
        "oracle_source": spec.get("oracle_source"),
        "total_gap_mean": mean_gap,
        "total_gap_std": std_gap,
        "total_gap_z": None if mean_gap is None or std_gap is None else abs(mean_gap) / (std_gap + EPS),
        "total_gap_sign_flip_fraction": sign_flip_fraction(gaps),
        "noise_dominated_bool": bool(mean_gap is not None and std_gap is not None and abs(mean_gap) <= 2.0 * std_gap),
        "ordering_stable_bool": stable,
        "oracle_wins_count": oracle_wins,
        "selected_wins_count": selected_wins,
        "active_gap_mean": active_mean,
        "active_gap_std": active_std,
        "active_gap_z": None if active_mean is None or active_std is None else abs(active_mean) / (active_std + EPS),
        "active_gap_sign_flip_fraction": sign_flip_fraction(active_gaps),
        "weak_gap_mean": weak_mean,
        "weak_gap_std": weak_std,
        "weak_gap_z": None if weak_mean is None or weak_std is None else abs(weak_mean) / (weak_std + EPS),
        "weak_gap_sign_flip_fraction": sign_flip_fraction(weak_gaps),
        "active_variance_share": active_share,
        "weak_variance_share": weak_share,
        "mixed_variance_bool": segment_driver(active_std, weak_std) == "mixed",
        "dominant_segment": segment_driver(active_std, weak_std),
    }


def vector_from(row: dict[str, Any], key: str, expected_len: int) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.size != expected_len or not np.all(np.isfinite(vec)):
        return None
    return vec


def vector_series(series: list[dict[str, Any]], key: str, expected_len: int) -> list[np.ndarray]:
    vectors = []
    for row in series:
        vec = vector_from(row, key, expected_len)
        if vec is not None:
            vectors.append(vec)
    return vectors


def dim_anatomy_for_pair(
    spec: dict[str, Any],
    budget: BudgetMode,
    series: dict[str, list[dict[str, Any]]],
    segment: str,
) -> list[dict[str, Any]]:
    if segment == "active":
        key = "active_residual"
        expected_len = 18
        global_start = ACTIVE_SLICE_START
    else:
        key = "weak_residual"
        expected_len = 96
        global_start = WEAK_SLICE_START
    selected = vector_series(series["selected"], key, expected_len)
    oracle = vector_series(series["oracle"], key, expected_len)
    count = min(len(selected), len(oracle))
    if count == 0:
        return []
    selected_arr = np.stack(selected[:count], axis=0)
    oracle_arr = np.stack(oracle[:count], axis=0)
    residual_gap = oracle_arr - selected_arr
    squared_contrib = oracle_arr * oracle_arr - selected_arr * selected_arr
    rows = []
    for local_idx in range(expected_len):
        gap_values_local = residual_gap[:, local_idx].astype(float).tolist()
        contrib_values = squared_contrib[:, local_idx].astype(float).tolist()
        rows.append(
            {
                "budget_mode": budget.name,
                "sample_index": spec["sample_index"],
                "case_type": spec["case_type"],
                "pair_type": spec["pair_type"],
                "segment": segment,
                "dimension_index_global": global_start + local_idx,
                "dimension_index_segment": local_idx,
                "mean_selected_residual": float(np.mean(selected_arr[:, local_idx])),
                "std_selected_residual": float(np.std(selected_arr[:, local_idx])),
                "mean_oracle_residual": float(np.mean(oracle_arr[:, local_idx])),
                "std_oracle_residual": float(np.std(oracle_arr[:, local_idx])),
                "mean_residual_gap": float(np.mean(residual_gap[:, local_idx])),
                "std_residual_gap": float(np.std(residual_gap[:, local_idx])),
                "gap_sign_flip_fraction": sign_flip_fraction(gap_values_local),
                "contribution_to_segment_gap_mean": float(np.mean(squared_contrib[:, local_idx])),
                "contribution_to_segment_gap_variance": float(np.var(contrib_values)),
            }
        )
    return rows


def rank_dimension_rows(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["budget_mode"], row["segment"]), []).append(row)
    for group_rows in groups.values():
        variance_sorted = sorted(group_rows, key=lambda row: row["std_residual_gap"], reverse=True)
        abs_gap_sorted = sorted(group_rows, key=lambda row: abs(row["mean_residual_gap"]), reverse=True)
        for rank, row in enumerate(variance_sorted, start=1):
            row["variance_rank"] = rank
        for rank, row in enumerate(abs_gap_sorted, start=1):
            row["mean_abs_gap_rank"] = rank


def aggregate_dimension_rows(rows: list[dict[str, Any]], segment: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["segment"] == segment:
            grouped.setdefault((row["budget_mode"], row["dimension_index_global"]), []).append(row)
    out = []
    for (budget, dim), dim_rows in sorted(grouped.items()):
        local = dim_rows[0]["dimension_index_segment"]
        out.append(
            {
                "budget_mode": budget,
                "segment": segment,
                "dimension_index_global": dim,
                "dimension_index_segment": local,
                "mean_selected_residual": mean_float([row["mean_selected_residual"] for row in dim_rows]),
                "std_selected_residual": mean_float([row["std_selected_residual"] for row in dim_rows]),
                "mean_oracle_residual": mean_float([row["mean_oracle_residual"] for row in dim_rows]),
                "std_oracle_residual": mean_float([row["std_oracle_residual"] for row in dim_rows]),
                "mean_residual_gap": mean_float([row["mean_residual_gap"] for row in dim_rows]),
                "std_residual_gap": mean_float([row["std_residual_gap"] for row in dim_rows]),
                "gap_sign_flip_fraction": mean_optional([row["gap_sign_flip_fraction"] for row in dim_rows]),
                "contribution_to_segment_gap_mean": mean_float([row["contribution_to_segment_gap_mean"] for row in dim_rows]),
                "contribution_to_segment_gap_variance": mean_float([row["contribution_to_segment_gap_variance"] for row in dim_rows]),
            }
        )
    rank_dimension_rows(out)
    by_b0 = {row["dimension_index_global"]: row for row in out if row["budget_mode"] == "B0_current_budget"}
    for row in out:
        base = by_b0.get(row["dimension_index_global"])
        if row["budget_mode"] == "B0_current_budget" or base is None:
            row["improved_std_vs_B0"] = None
            row["worsened_std_vs_B0"] = None
            row["std_delta_vs_B0"] = None
        else:
            delta = base["std_residual_gap"] - row["std_residual_gap"]
            row["std_delta_vs_B0"] = delta
            row["improved_std_vs_B0"] = delta > 0
            row["worsened_std_vs_B0"] = delta < 0
    return out


def mean_float(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def mean_optional(values: list[float | None]) -> float | None:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(statistics.mean(finite_values)) if finite_values else None


def top_dims(rows: list[dict[str, Any]], budget: str, key: str, n: int = 8, reverse: bool = True) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["budget_mode"] == budget]
    return [
        {
            "dimension_index_global": row["dimension_index_global"],
            "dimension_index_segment": row["dimension_index_segment"],
            key: row.get(key),
            "std_residual_gap": row.get("std_residual_gap"),
            "mean_residual_gap": row.get("mean_residual_gap"),
        }
        for row in sorted(selected, key=lambda row: row.get(key) or 0.0, reverse=reverse)[:n]
    ]


def dims_improved(rows: list[dict[str, Any]], segment: str, improved: bool, n: int = 8) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["segment"] == segment and row["budget_mode"] == "B1_active_paths_high" and row.get("std_delta_vs_B0") is not None
    ]
    if improved:
        candidates = [row for row in candidates if row["std_delta_vs_B0"] > 0]
        candidates = sorted(candidates, key=lambda row: row["std_delta_vs_B0"], reverse=True)
    else:
        candidates = [row for row in candidates if row["std_delta_vs_B0"] <= 0]
        candidates = sorted(candidates, key=lambda row: row["std_delta_vs_B0"])
    return [
        {
            "dimension_index_global": row["dimension_index_global"],
            "dimension_index_segment": row["dimension_index_segment"],
            "std_delta_vs_B0": row["std_delta_vs_B0"],
            "std_residual_gap": row["std_residual_gap"],
        }
        for row in candidates[:n]
    ]


def pair_diagnosis(b0: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    b0_majority = majority_label(b0)
    b1_majority = majority_label(b1)
    reduces_gap = less_than(b1.get("total_gap_std"), b0.get("total_gap_std"))
    increases_z = greater_than(b1.get("total_gap_z"), b0.get("total_gap_z"))
    reduces_active = less_than(b1.get("active_gap_std"), b0.get("active_gap_std"))
    reduces_weak = less_than(b1.get("weak_gap_std"), b0.get("weak_gap_std"))
    stabilizes = bool(b1.get("ordering_stable_bool")) and not bool(b0.get("ordering_stable_bool"))
    still_noise = bool(b1.get("noise_dominated_bool"))
    if stabilizes and reduces_active:
        diagnosis = "active_local_noise_reduced"
    elif still_noise and b1.get("dominant_segment") == "weak" and not reduces_weak:
        diagnosis = "weak_noise_persists"
    elif still_noise and b1.get("dominant_segment") == "mixed":
        diagnosis = "mixed_noise_persists"
    elif b1.get("total_gap_mean") is not None and b1.get("total_gap_std") is not None and abs(b1["total_gap_mean"]) < b1["total_gap_std"]:
        diagnosis = "ordering_intrinsically_small_gap"
    elif not reduces_gap or not reduces_active:
        diagnosis = "active_budget_counterproductive"
    else:
        diagnosis = "insufficient_data"
    return {
        "sample_index": b1["sample_index"],
        "case_type": b1["case_type"],
        "pair_type": b1["pair_type"],
        "B1_stabilizes_pair_bool": stabilizes,
        "B1_reduces_gap_std_bool": reduces_gap,
        "B1_increases_gap_z_bool": increases_z,
        "B1_changes_ordering_majority_bool": b0_majority != b1_majority,
        "B1_reduces_active_noise_bool": reduces_active,
        "B1_reduces_weak_noise_bool": reduces_weak,
        "B1_still_noise_dominated_bool": still_noise,
        "B0_majority": b0_majority,
        "B1_majority": b1_majority,
        "diagnosis": diagnosis,
    }


def less_than(left: Any, right: Any) -> bool:
    return finite(left) and finite(right) and float(left) < float(right)


def greater_than(left: Any, right: Any) -> bool:
    return finite(left) and finite(right) and float(left) > float(right)


def majority_label(row: dict[str, Any]) -> str:
    if row.get("oracle_wins_count", 0) > row.get("selected_wins_count", 0):
        return "oracle"
    if row.get("selected_wins_count", 0) > row.get("oracle_wins_count", 0):
        return "selected"
    return "tie"


def aggregate_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stable_count": sum(1 for row in rows if row.get("ordering_stable_bool")),
        "noise_dominated_count": sum(1 for row in rows if row.get("noise_dominated_bool")),
        "active_variance_share_mean": mean_optional([row.get("active_variance_share") for row in rows]),
        "weak_variance_share_mean": mean_optional([row.get("weak_variance_share") for row in rows]),
        "dominant_segment_counts": dict(Counter(row.get("dominant_segment") for row in rows)),
    }


def dimensions_for_pairs(
    pair_diagnoses: list[dict[str, Any]],
    pair_dim_rows: list[dict[str, Any]],
    segment: str,
    predicate,
    n: int = 8,
) -> list[dict[str, Any]]:
    target_keys = {
        (diag["sample_index"], diag["pair_type"])
        for diag in pair_diagnoses
        if predicate(diag)
    }
    rows = [
        row for row in pair_dim_rows
        if row["segment"] == segment
        and row["budget_mode"] == "B1_active_paths_high"
        and (row["sample_index"], row["pair_type"]) in target_keys
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dimension_index_global"], []).append(row)
    out = []
    for dim, dim_rows in grouped.items():
        out.append(
            {
                "dimension_index_global": dim,
                "mean_std_residual_gap": mean_float([row["std_residual_gap"] for row in dim_rows]),
                "pair_count": len(dim_rows),
            }
        )
    return sorted(out, key=lambda row: row["mean_std_residual_gap"], reverse=True)[:n]


def decide(summary: dict[str, Any]) -> tuple[str, str]:
    b1_noise = summary["B1_noise_dominated_count"]
    b1_stable_gain = summary["B1_stable_gain"]
    active_recurs = summary["recurring_active_dims_reappear_bool"]
    mixed_under_b1 = len(summary["pairs_with_mixed_noise_under_B1"])
    if b1_stable_gain >= 4 and b1_noise <= 2 and active_recurs:
        return (
            "A",
            "Noise appears concentrated in recurring active dimensions and B1 clearly reduces those dimensions. Future work should be logging/cache infrastructure only, not scorer changes.",
        )
    if b1_stable_gain > 0 and (b1_noise > 2 or mixed_under_b1 > 0):
        return (
            "B",
            "B1 helps some pairs, but variance remains mixed or spread across dimensions. Keep scorer calibration closed; return to drift-span/candidate-generation unless broader fingerprint infrastructure is explicitly desired.",
        )
    return (
        "C",
        "B1 does not reduce meaningful dimension-level variance. Stop this path-budget/scorer calibration line and return to drift-span/candidate-generation.",
    )


def blocked_result(args: argparse.Namespace, estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata(args, [], False, True),
        "summary": {
            "decision": "D",
            "recommendation": "Estimated B0/B1 variance anatomy recomputation exceeds the Codex budget. Run a standalone local command instead.",
            "estimated_runtime": estimate,
            "formal_eval_recommended": False,
        },
        "pairs": [],
        "pair_diagnoses": [],
        "active_dimension_anatomy": [],
        "weak_dimension_anatomy": [],
    }


def metadata(args: argparse.Namespace, budgets: list[BudgetMode], existing_sufficient: bool, recomputed: bool) -> dict[str, Any]:
    return {
        "path_budget_json": str(args.path_budget_json),
        "stability_json": str(args.stability_json),
        "ambiguity_json": str(args.ambiguity_json),
        "all32_smoke_json": str(args.all32_smoke_json),
        "scorer_ready_json": str(args.scorer_ready_json),
        "target_case_types": sorted(parse_case_types(args.target_case_types)),
        "budget_modes": [budget.name for budget in budgets],
        "budget_configs": {
            budget.name: {
                "n_paths": budget.n_paths,
                "active_paths": budget.active_paths,
                "n_steps": budget.n_steps,
                "repeats": budget.repeats,
            }
            for budget in budgets
        },
        "used_existing_path_budget_data_bool": existing_sufficient,
        "recomputed_bool": recomputed,
        "score_kind": "constant_grid_rolewise_no_multi_u0",
        "active_weak_weights": [1.0, 1.0],
    }


def summarize(
    pair_specs: list[dict[str, Any]],
    budget_metrics: dict[str, list[dict[str, Any]]],
    pair_diagnoses: list[dict[str, Any]],
    active_dims: list[dict[str, Any]],
    weak_dims: list[dict[str, Any]],
    pair_dim_rows: list[dict[str, Any]],
    existing_sufficient: bool,
    recomputed: bool,
    budgets: list[BudgetMode],
) -> dict[str, Any]:
    b0_rows = budget_metrics.get("B0_current_budget", [])
    b1_rows = budget_metrics.get("B1_active_paths_high", [])
    b0_summary = aggregate_budget(b0_rows)
    b1_summary = aggregate_budget(b1_rows)
    top_active_b0 = top_dims(active_dims, "B0_current_budget", "std_residual_gap")
    top_active_b1 = top_dims(active_dims, "B1_active_paths_high", "std_residual_gap")
    top_weak_b0 = top_dims(weak_dims, "B0_current_budget", "std_residual_gap")
    top_weak_b1 = top_dims(weak_dims, "B1_active_paths_high", "std_residual_gap")
    top_active_values = {row["dimension_index_global"] for row in top_active_b0[:8] + top_active_b1[:8]}
    top_weak_values = {row["dimension_index_global"] for row in top_weak_b0[:8] + top_weak_b1[:8]}
    summary = {
        "samples_analyzed": len({spec["sample_index"] for spec in pair_specs}),
        "candidate_pairs_analyzed": len(pair_specs),
        "used_existing_path_budget_data_bool": existing_sufficient,
        "recomputed_bool": recomputed,
        "budget_modes_analyzed": [budget.name for budget in budgets],
        "B0_noise_dominated_count": b0_summary["noise_dominated_count"],
        "B1_noise_dominated_count": b1_summary["noise_dominated_count"],
        "B0_stable_count": b0_summary["stable_count"],
        "B1_stable_count": b1_summary["stable_count"],
        "B1_stable_gain": b1_summary["stable_count"] - b0_summary["stable_count"],
        "active_variance_share_B0_mean": b0_summary["active_variance_share_mean"],
        "active_variance_share_B1_mean": b1_summary["active_variance_share_mean"],
        "weak_variance_share_B0_mean": b0_summary["weak_variance_share_mean"],
        "weak_variance_share_B1_mean": b1_summary["weak_variance_share_mean"],
        "top_active_variance_dims_B0": top_active_b0,
        "top_active_variance_dims_B1": top_active_b1,
        "top_weak_variance_dims_B0": top_weak_b0,
        "top_weak_variance_dims_B1": top_weak_b1,
        "top_active_dims_improved_by_B1": dims_improved(active_dims, "active", True),
        "top_active_dims_not_improved_by_B1": dims_improved(active_dims, "active", False),
        "top_weak_dims_improved_by_B1": dims_improved(weak_dims, "weak", True),
        "top_weak_dims_not_improved_by_B1": dims_improved(weak_dims, "weak", False),
        "recurring_active_dims_checked": RECURRING_ACTIVE_DIMS,
        "recurring_weak_dims_checked": RECURRING_WEAK_DIMS,
        "recurring_active_dims_reappear_bool": any(dim in top_active_values for dim in RECURRING_ACTIVE_DIMS),
        "recurring_weak_dims_reappear_bool": any(dim in top_weak_values for dim in RECURRING_WEAK_DIMS),
        "active_dims_explaining_noise_dominated_pairs": dimensions_for_pairs(
            pair_diagnoses,
            pair_dim_rows,
            "active",
            lambda diag: diag["B1_still_noise_dominated_bool"],
        ),
        "weak_dims_explaining_noise_dominated_pairs": dimensions_for_pairs(
            pair_diagnoses,
            pair_dim_rows,
            "weak",
            lambda diag: diag["B1_still_noise_dominated_bool"],
        ),
        "active_dims_explaining_B1_stable_gain": dimensions_for_pairs(
            pair_diagnoses,
            pair_dim_rows,
            "active",
            lambda diag: diag["B1_stabilizes_pair_bool"],
        ),
        "pairs_stabilized_by_B1": [
            {"sample_index": diag["sample_index"], "pair_type": diag["pair_type"]}
            for diag in pair_diagnoses if diag["B1_stabilizes_pair_bool"]
        ],
        "pairs_still_noise_dominated_under_B1": [
            {"sample_index": diag["sample_index"], "pair_type": diag["pair_type"]}
            for diag in pair_diagnoses if diag["B1_still_noise_dominated_bool"]
        ],
        "pairs_with_mixed_noise_under_B1": [
            {"sample_index": row["sample_index"], "pair_type": row["pair_type"]}
            for row in b1_rows if row.get("dominant_segment") == "mixed"
        ],
        "B0_dominant_segment_counts": b0_summary["dominant_segment_counts"],
        "B1_dominant_segment_counts": b1_summary["dominant_segment_counts"],
        "model_decoding_avoided": True,
        "candidate_generation_avoided": True,
        "sde_validation_probe_avoided": True,
        "formal_eval_recommended": False,
    }
    decision, recommendation = decide(summary)
    summary["decision"] = decision
    summary["recommendation"] = recommendation
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def dim_list_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return ", ".join(str(row["dimension_index_global"]) for row in rows[:8])


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Fingerprint Variance Anatomy Diagnostic",
        "",
        "## Why This Follows Path-Budget",
        "",
        "The path-budget diagnostic showed B1 active-paths-high improves some",
        "V4 rescue/harm oracle-pair orderings, but most pairs remain",
        "noise-dominated. This helper asks whether the remaining variance is",
        "localized to a small set of active/weak dimensions or spread across",
        "segments. No model decoding, candidate generation, `sde_validation_probe.py`,",
        "formal eval, rerank mode, P2 eval integration, or scorer change was run.",
        "",
    ]
    if summary.get("decision") == "D":
        lines.extend(
            [
                "## Blocked",
                "",
                f"Estimated runtime: `{summary.get('estimated_runtime')}`",
                "",
                f"Decision `D`: {summary['recommendation']}",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Data Source",
            "",
            "| Item | Value |",
            "| --- | ---: |",
            f"| Existing path-budget data sufficient | {fmt(summary['used_existing_path_budget_data_bool'])} |",
            f"| Recomputed B0/B1 residuals | {fmt(summary['recomputed_bool'])} |",
            f"| Samples analyzed | {summary['samples_analyzed']} |",
            f"| Candidate pairs analyzed | {summary['candidate_pairs_analyzed']} |",
            f"| Budget modes analyzed | {', '.join(summary['budget_modes_analyzed'])} |",
            "",
            "Note: the anatomy recomputation uses 3 repeats for both B0 and B1 so",
            "per-dimension residual vectors are directly comparable. The previous",
            "path-budget report used 5 repeats for B0 and 3 repeats for B1, so B0",
            "stable/noise counts can differ slightly here.",
            "",
            "## B0 vs B1 Pair Stability",
            "",
            "| Budget | Stable | Noise dominated | Active variance share | Weak variance share |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| B0 | {summary['B0_stable_count']} | {summary['B0_noise_dominated_count']} | {fmt(summary['active_variance_share_B0_mean'])} | {fmt(summary['weak_variance_share_B0_mean'])} |",
            f"| B1 | {summary['B1_stable_count']} | {summary['B1_noise_dominated_count']} | {fmt(summary['active_variance_share_B1_mean'])} | {fmt(summary['weak_variance_share_B1_mean'])} |",
            "",
            "## Active Dimension Variance",
            "",
            "| Budget | Top variance dims | Top improved dims | Top not-improved dims |",
            "| --- | --- | --- | --- |",
            f"| B0 | {dim_list_text(summary['top_active_variance_dims_B0'])} | NA | NA |",
            f"| B1 | {dim_list_text(summary['top_active_variance_dims_B1'])} | {dim_list_text(summary['top_active_dims_improved_by_B1'])} | {dim_list_text(summary['top_active_dims_not_improved_by_B1'])} |",
            "",
            "## Weak Dimension Variance",
            "",
            "| Budget | Top variance dims | Top improved dims | Top not-improved dims |",
            "| --- | --- | --- | --- |",
            f"| B0 | {dim_list_text(summary['top_weak_variance_dims_B0'])} | NA | NA |",
            f"| B1 | {dim_list_text(summary['top_weak_variance_dims_B1'])} | {dim_list_text(summary['top_weak_dims_improved_by_B1'])} | {dim_list_text(summary['top_weak_dims_not_improved_by_B1'])} |",
            "",
            "## Recurring Dimension Check",
            "",
            "| Dimension set | Reappears in top variance dims |",
            "| --- | --- |",
            f"| Active 76/72/82/88 | {fmt(summary['recurring_active_dims_reappear_bool'])} |",
            f"| Weak 136/137/133 | {fmt(summary['recurring_weak_dims_reappear_bool'])} |",
            "",
            "## Pair-Level Diagnosis",
            "",
            "| Sample | Case | Pair | B1 stabilizes | B1 noise dominated | Reduces gap std | Reduces active std | Reduces weak std | Diagnosis |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["pair_diagnoses"]:
        lines.append(
            "| {idx} | {case} | {pair} | {stab} | {noise} | {gap} | {active} | {weak} | {diag} |".format(
                idx=row["sample_index"],
                case=row["case_type"],
                pair=row["pair_type"],
                stab=fmt(row["B1_stabilizes_pair_bool"]),
                noise=fmt(row["B1_still_noise_dominated_bool"]),
                gap=fmt(row["B1_reduces_gap_std_bool"]),
                active=fmt(row["B1_reduces_active_noise_bool"]),
                weak=fmt(row["B1_reduces_weak_noise_bool"]),
                diag=row["diagnosis"],
            )
        )
    lines.extend(
        [
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


def write_standalone_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \\
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_fingerprint_variance_anatomy.py \\
  --path-budget-json p2_candidate_fingerprint_path_budget.json \\
  --stability-json p2_fingerprint_stability_diagnostics.json \\
  --ambiguity-json p2_fingerprint_ambiguity_diagnostics.json \\
  --all32-smoke-json p2_calibration_all32_smoke.json \\
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \\
  --budget-modes B0_current_budget,B1_active_paths_high \\
  --target-case-types V4_rescue,V4_harm \\
  --report p2_fingerprint_variance_anatomy.md \\
  --json-output p2_fingerprint_variance_anatomy.json \\
  > /private/tmp/gensr_sde_p2_fingerprint_variance_anatomy.log 2>&1
""",
    )
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    path_budget_payload = load_json(args.path_budget_json)
    stability_payload = load_json(args.stability_json)
    ambiguity_payload = load_json(args.ambiguity_json)
    sidecar_payload = load_json(args.scorer_ready_json)
    load_json(args.all32_smoke_json)
    budgets = parse_budget_modes(args.budget_modes)
    existing_sufficient = has_residual_vectors(path_budget_payload)
    pair_specs = build_pair_specs(
        stability_payload,
        ambiguity_payload,
        sidecar_payload,
        parse_case_types(args.target_case_types),
    )
    if existing_sufficient:
        # Current path-budget JSON intentionally lacks these vectors, but keep
        # this branch explicit so future richer logs can report why recompute
        # was avoided.
        raise NotImplementedError("existing residual-vector parsing is not needed for current artifacts")

    estimate = estimate_runtime(pair_specs, sidecar_payload, budgets)
    if estimate["estimated_seconds"] > MAX_CODEX_SECONDS:
        write_standalone_script(Path("scripts/run_p2_fingerprint_variance_anatomy.sh"))
        result = blocked_result(args, estimate)
        args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
        args.report.write_text(build_markdown(json_ready(result)))
        print("decision=D")
        print(f"estimated_seconds={estimate['estimated_seconds']:.3f}")
        return

    metrics_by_budget: dict[str, list[dict[str, Any]]] = {}
    pair_dim_rows: list[dict[str, Any]] = []
    pair_payloads = []
    for budget in budgets:
        series_by_pair = score_budget_mode(pair_specs, sidecar_payload, budget)
        metrics_by_budget[budget.name] = []
        for spec in pair_specs:
            key = (int(spec["sample_index"]), spec["pair_type"])
            series = series_by_pair[key]
            metrics_by_budget[budget.name].append(pair_budget_metrics(spec, budget, series))
            pair_dim_rows.extend(dim_anatomy_for_pair(spec, budget, series, "active"))
            pair_dim_rows.extend(dim_anatomy_for_pair(spec, budget, series, "weak"))
    rank_dimension_rows(pair_dim_rows)

    active_dims = aggregate_dimension_rows(pair_dim_rows, "active")
    weak_dims = aggregate_dimension_rows(pair_dim_rows, "weak")
    for row in active_dims:
        row["segment"] = "active"
    for row in weak_dims:
        row["segment"] = "weak"
    all_dim_rows = active_dims + weak_dims

    b0_lookup = {(row["sample_index"], row["pair_type"]): row for row in metrics_by_budget["B0_current_budget"]}
    b1_lookup = {(row["sample_index"], row["pair_type"]): row for row in metrics_by_budget["B1_active_paths_high"]}
    pair_diagnoses = [
        pair_diagnosis(b0_lookup[key], b1_lookup[key])
        for key in sorted(b0_lookup)
        if key in b1_lookup
    ]
    summary = summarize(
        pair_specs,
        metrics_by_budget,
        pair_diagnoses,
        active_dims,
        weak_dims,
        pair_dim_rows,
        existing_sufficient,
        True,
        budgets,
    )
    for spec in pair_specs:
        pair_payloads.append(
            {
                "sample_index": spec["sample_index"],
                "case_type": spec["case_type"],
                "pair_type": spec["pair_type"],
                "selected_candidate": candidate_brief(spec["selected_row"]),
                "oracle_candidate": candidate_brief(spec["oracle_row"]),
                "budget_metrics": {
                    budget.name: next(
                        row for row in metrics_by_budget[budget.name]
                        if row["sample_index"] == spec["sample_index"] and row["pair_type"] == spec["pair_type"]
                    )
                    for budget in budgets
                },
            }
        )
    result = {
        "metadata": metadata(args, budgets, existing_sufficient, True) | {"runtime_estimate": estimate},
        "summary": summary,
        "pairs": pair_payloads,
        "pair_diagnoses": pair_diagnoses,
        "active_dimension_anatomy": [row for row in all_dim_rows if row["segment"] == "active"],
        "weak_dimension_anatomy": [row for row in all_dim_rows if row["segment"] == "weak"],
        "pair_dimension_rows": pair_dim_rows,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_markdown(json_ready(result)))
    print(f"decision={summary['decision']}")
    print(f"samples_analyzed={summary['samples_analyzed']}")
    print(f"candidate_pairs_analyzed={summary['candidate_pairs_analyzed']}")
    print(f"B0_stable={summary['B0_stable_count']} B0_noise={summary['B0_noise_dominated_count']}")
    print(f"B1_stable={summary['B1_stable_count']} B1_noise={summary['B1_noise_dominated_count']}")
    print(f"B1_stable_gain={summary['B1_stable_gain']}")
    print(f"top_active_dims_B1={dim_list_text(summary['top_active_variance_dims_B1'])}")
    print(f"top_weak_dims_B1={dim_list_text(summary['top_weak_variance_dims_B1'])}")


if __name__ == "__main__":
    main()
