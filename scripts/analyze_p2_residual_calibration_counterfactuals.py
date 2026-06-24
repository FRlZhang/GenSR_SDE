#!/usr/bin/env python3
"""Offline counterfactual calibration diagnostics for P2 score misses."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_p2_admitted_candidate_scoring import json_ready, parse_target_samples, score_seed, target_array
from analyze_p2_oracle_score_miss_residuals import (
    ACTIVE_SLICE,
    WEAK_SLICE,
    candidate_by_id,
    candidate_index_by_id,
    score_candidate,
    summary,
)
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import candidate_tokens_to_system, fingerprint_distance_details, score_candidate_system
from simulator_sde import solve_fingerprint


CONSTANT_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
GLOBAL_ACTIVE_DIMS = [76, 72, 82, 88]
GLOBAL_WEAK_DIMS = [136, 137, 133]
VARIANTS = [
    "V0_current_recomputed",
    "V1_weak_only_diagnostic",
    "V2_active_only_diagnostic",
    "V3_active_without_dim_76",
    "V4_active_without_dims_76_72_82_88",
    "V5_weak_without_dims_136_137_133",
    "V6_active_and_weak_without_recurring_dims",
    "V7_top1_active_advantage_removed_per_sample",
    "V8_top2_active_advantages_removed_per_sample",
    "V9_selected_constants_applied_to_oracle",
    "V10_oracle_constants_applied_to_selected",
    "V11_shared_constants_only_best_of_grid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-json", type=Path, required=True)
    parser.add_argument("--semantic-scoring-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--target-samples", type=str, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def l2_without(residual: list[float] | np.ndarray, remove_global: list[int], offset: int) -> float:
    vec = np.asarray(residual, dtype=np.float64)
    remove_local = {dim - offset for dim in remove_global if 0 <= dim - offset < len(vec)}
    keep = [idx for idx in range(len(vec)) if idx not in remove_local]
    if not keep:
        return 0.0
    return float(np.linalg.norm(vec[keep]))


def active_score(row: dict[str, Any], remove_global: list[int] | None = None) -> float:
    return l2_without(row["active_residual"], remove_global or [], ACTIVE_SLICE.start)


def weak_score(row: dict[str, Any], remove_global: list[int] | None = None) -> float:
    return l2_without(row["weak_residual"], remove_global or [], WEAK_SLICE.start)


def top_selected_active_dims(residual_sample: dict[str, Any], limit: int) -> list[int]:
    return [
        item["dimension_index"]
        for item in residual_sample.get("top_active_residual_advantages_for_selected", [])[:limit]
    ]


def fixed_constant_score(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    candidate_index: int,
    drift_constant: float,
    diffusion_constant: float,
) -> dict[str, Any] | None:
    system = candidate_tokens_to_system(
        candidate["full_sequence_tokens"],
        drift_constant_value=drift_constant,
        diffusion_constant_value=diffusion_constant,
    )
    if system is None:
        return None
    _, y_data, meta = solve_fingerprint(
        system,
        FingerprintConfig(n_paths=800, n_steps=60, active_paths=800),
        score_seed(int(sample["sample_index"]), candidate_index),
    )
    if y_data is None or not np.all(np.isfinite(y_data)):
        return None
    details = fingerprint_distance_details(
        target_array(sample),
        y_data,
        "rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
    )
    return {
        "score": details["distance"],
        "active_score": details["active_kramers_moyal_distance"],
        "weak_score": details["gaussian_weak_kernel_distance"],
        "best_drift_constant": drift_constant,
        "best_diffusion_constant": diffusion_constant,
        "meta": meta,
    }


def shared_best_score(sample: dict[str, Any], candidate: dict[str, Any], candidate_index: int) -> dict[str, Any] | None:
    details, _failure_reason, _fingerprint_failures = score_candidate_system(
        candidate["full_sequence_tokens"],
        target_array(sample),
        FingerprintConfig(n_paths=800, n_steps=60, active_paths=800),
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(int(sample["sample_index"]), candidate_index),
    )
    if details is None:
        return None
    return {
        "score": details["shared_constant_distance"],
        "active_score": details["shared_constant_active_kramers_moyal_distance"],
        "weak_score": details["shared_constant_gaussian_weak_kernel_distance"],
        "best_constant": details["shared_best_constant"],
    }


def variant_row(
    name: str,
    selected_score: float | None,
    oracle_score: float | None,
    baseline_gap: float,
    notes: str = "",
) -> dict[str, Any]:
    if selected_score is None or oracle_score is None:
        gap = None
        oracle_wins = False
        delta = None
    else:
        gap = oracle_score - selected_score
        oracle_wins = gap < 0.0
        delta = gap - baseline_gap
    return {
        "variant": name,
        "selected_variant_score": selected_score,
        "oracle_variant_score": oracle_score,
        "variant_gap": gap,
        "oracle_wins_bool": oracle_wins,
        "gap_delta_vs_baseline": delta,
        "notes": notes,
    }


def evaluate_variants(
    sidecar_sample: dict[str, Any],
    residual_sample: dict[str, Any],
    semantic_sample: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_id = candidate_by_id(sidecar_sample)
    indices = candidate_index_by_id(sidecar_sample)
    selected_id = semantic_sample["combined"]["selected_candidate"]["candidate_id"]
    oracle_id = semantic_sample["p2_oracle"]["candidate_id"]
    selected_sidecar = by_id[selected_id]
    oracle_sidecar = by_id[oracle_id]
    selected = score_candidate(sidecar_sample, selected_sidecar, indices[selected_id])
    oracle = score_candidate(sidecar_sample, oracle_sidecar, indices[oracle_id])
    baseline_gap = oracle["score"] - selected["score"]
    top1 = top_selected_active_dims(residual_sample, 1)
    top2 = top_selected_active_dims(residual_sample, 2)

    variants = [
        variant_row("V0_current_recomputed", selected["score"], oracle["score"], baseline_gap),
        variant_row("V1_weak_only_diagnostic", selected["weak_score"], oracle["weak_score"], baseline_gap),
        variant_row("V2_active_only_diagnostic", selected["active_score"], oracle["active_score"], baseline_gap),
        variant_row(
            "V3_active_without_dim_76",
            active_score(selected, [76]) + weak_score(selected),
            active_score(oracle, [76]) + weak_score(oracle),
            baseline_gap,
        ),
        variant_row(
            "V4_active_without_dims_76_72_82_88",
            active_score(selected, GLOBAL_ACTIVE_DIMS) + weak_score(selected),
            active_score(oracle, GLOBAL_ACTIVE_DIMS) + weak_score(oracle),
            baseline_gap,
        ),
        variant_row(
            "V5_weak_without_dims_136_137_133",
            active_score(selected) + weak_score(selected, GLOBAL_WEAK_DIMS),
            active_score(oracle) + weak_score(oracle, GLOBAL_WEAK_DIMS),
            baseline_gap,
        ),
        variant_row(
            "V6_active_and_weak_without_recurring_dims",
            active_score(selected, GLOBAL_ACTIVE_DIMS) + weak_score(selected, GLOBAL_WEAK_DIMS),
            active_score(oracle, GLOBAL_ACTIVE_DIMS) + weak_score(oracle, GLOBAL_WEAK_DIMS),
            baseline_gap,
        ),
        variant_row(
            "V7_top1_active_advantage_removed_per_sample",
            active_score(selected, top1) + weak_score(selected),
            active_score(oracle, top1) + weak_score(oracle),
            baseline_gap,
            notes=f"removed_active_dims={top1}",
        ),
        variant_row(
            "V8_top2_active_advantages_removed_per_sample",
            active_score(selected, top2) + weak_score(selected),
            active_score(oracle, top2) + weak_score(oracle),
            baseline_gap,
            notes=f"removed_active_dims={top2}",
        ),
    ]

    v9 = fixed_constant_score(
        sidecar_sample,
        oracle_sidecar,
        indices[oracle_id],
        selected["best_drift_constant"],
        selected["best_diffusion_constant"],
    )
    variants.append(
        variant_row(
            "V9_selected_constants_applied_to_oracle",
            selected["score"],
            None if v9 is None else v9["score"],
            baseline_gap,
            notes="oracle rescored with selected best drift/diffusion constants",
        )
    )
    v10 = fixed_constant_score(
        sidecar_sample,
        selected_sidecar,
        indices[selected_id],
        oracle["best_drift_constant"],
        oracle["best_diffusion_constant"],
    )
    variants.append(
        variant_row(
            "V10_oracle_constants_applied_to_selected",
            None if v10 is None else v10["score"],
            oracle["score"],
            baseline_gap,
            notes="selected rescored with oracle best drift/diffusion constants",
        )
    )
    selected_shared = shared_best_score(sidecar_sample, selected_sidecar, indices[selected_id])
    oracle_shared = shared_best_score(sidecar_sample, oracle_sidecar, indices[oracle_id])
    variants.append(
        variant_row(
            "V11_shared_constants_only_best_of_grid",
            None if selected_shared is None else selected_shared["score"],
            None if oracle_shared is None else oracle_shared["score"],
            baseline_gap,
            notes="both candidates rescored with best shared drift=diffusion constant",
        )
    )
    return variants, selected, oracle


def best_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in variants if row["variant_gap"] is not None]
    return min(available, key=lambda row: row["variant_gap"])


def summarize_values(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {"median": None, "max": None}
    return {"median": statistics.median(finite), "max": max(finite)}


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Residual Calibration Counterfactuals",
        "",
        "## Scope",
        "",
        "Offline fixed counterfactuals on the 7 P2 oracle score misses. Lower score is better. No formal eval, rerank-mode implementation, scorer grid, or production default change was run.",
        "",
        "## Variants",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Item | Value |",
            "| --- | ---: |",
            f"| Samples analyzed | {summary['samples_analyzed']} |",
            f"| Baseline oracle wins | {summary['baseline_oracle_wins']} |",
            f"| Samples flipped by any variant | {summary['samples_flipped_by_any_variant']} |",
            f"| Samples flipped by global recurring dims | {summary['samples_flipped_by_global_recurring_dims']} |",
            f"| Samples flipped by per-sample top dims | {summary['samples_flipped_by_per_sample_top_dims']} |",
            f"| Samples flipped by constant counterfactuals | {summary['samples_flipped_by_constant_counterfactual']} |",
            f"| Near-tie samples flipped | {summary['near_tie_samples_flipped']} |",
            f"| Active-trap samples flipped | {summary['active_trap_samples_flipped']} |",
            f"| Weak-trap samples flipped | {summary['weak_trap_samples_flipped']} |",
            "",
            f"Variant oracle wins: `{json.dumps(summary['variant_oracle_wins_by_variant'], sort_keys=True)}`",
            "",
            "## Per-Sample Best Counterfactual",
            "",
            "| Sample | Classification | Baseline gap | Best variant | Best gap | Any flip |",
            "| ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for sample in result["samples"]:
        lines.append(
            f"| {sample['sample_index']} | {sample['classification_from_previous_report']} | "
            f"{sample['baseline_gap']:.6f} | {sample['best_counterfactual_variant']} | "
            f"{sample['best_counterfactual_gap']:.6f} | {sample['does_any_fixed_variant_flip_to_oracle']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval or rerank-mode implementation is recommended unless a later smoke improves selected behavior safely.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    residual = load_json(args.residual_json)
    semantic = load_json(args.semantic_scoring_json)
    sidecar = load_json(args.scorer_ready_json)
    target_samples = parse_target_samples(args.target_samples)
    residual_by_idx = {int(sample["sample_index"]): sample for sample in residual["samples"]}
    semantic_by_idx = {int(sample["sample_index"]): sample for sample in semantic["samples"]}
    sidecar_by_idx = {int(sample["sample_index"]): sample for sample in sidecar["samples"]}

    sample_rows = []
    variant_wins: Counter[str] = Counter()
    median_deltas: dict[str, list[float]] = defaultdict(list)
    global_variants = {
        "V3_active_without_dim_76",
        "V4_active_without_dims_76_72_82_88",
        "V5_weak_without_dims_136_137_133",
        "V6_active_and_weak_without_recurring_dims",
    }
    per_sample_variants = {
        "V7_top1_active_advantage_removed_per_sample",
        "V8_top2_active_advantages_removed_per_sample",
    }
    constant_variants = {
        "V9_selected_constants_applied_to_oracle",
        "V10_oracle_constants_applied_to_selected",
        "V11_shared_constants_only_best_of_grid",
    }
    flips_by_group = {
        "global": set(),
        "per_sample": set(),
        "constant": set(),
        "near_tie": set(),
        "active_trap": set(),
        "weak_trap": set(),
        "current_candidate_trap": set(),
    }

    for idx in target_samples:
        residual_sample = residual_by_idx[idx]
        variants, selected, oracle = evaluate_variants(
            sidecar_by_idx[idx],
            residual_sample,
            semantic_by_idx[idx],
        )
        baseline_gap = residual_sample["score_gap"]
        best = best_variant(variants)
        flipped = [row for row in variants if row["oracle_wins_bool"]]
        for row in variants:
            if row["oracle_wins_bool"]:
                variant_wins[row["variant"]] += 1
            if row["gap_delta_vs_baseline"] is not None:
                median_deltas[row["variant"]].append(row["gap_delta_vs_baseline"])
        flipped_names = {row["variant"] for row in flipped}
        if flipped_names & global_variants:
            flips_by_group["global"].add(idx)
        if flipped_names & per_sample_variants:
            flips_by_group["per_sample"].add(idx)
        if flipped_names & constant_variants:
            flips_by_group["constant"].add(idx)
        classification = residual_sample["classification"]
        if flipped and classification in flips_by_group:
            flips_by_group[classification].add(idx)
        sample_rows.append(
            {
                "sample_index": idx,
                "classification_from_previous_report": classification,
                "selected_source": residual_sample["selected_source"],
                "selected_sequence": residual_sample["selected_sequence"],
                "p2_oracle_sequence": residual_sample["p2_oracle_sequence"],
                "baseline_selected_score": residual_sample["selected_score"],
                "baseline_p2_oracle_score": residual_sample["p2_oracle_score"],
                "baseline_gap": baseline_gap,
                "variants": variants,
                "best_counterfactual_variant": best["variant"],
                "best_counterfactual_gap": best["variant_gap"],
                "does_any_fixed_variant_flip_to_oracle": bool(flipped),
                "does_near_tie_become_oracle": bool(flipped) and classification == "near_tie",
                "is_flip_caused_by_recurring_global_dims": bool(flipped_names & global_variants),
                "is_flip_caused_by_per_sample_top_dims": bool(flipped_names & per_sample_variants),
                "is_flip_caused_by_constant_counterfactual": bool(flipped_names & constant_variants),
            }
        )

    baseline_wins = sum(1 for sample in sample_rows if sample["baseline_gap"] < 0)
    any_flipped = [sample for sample in sample_rows if sample["does_any_fixed_variant_flip_to_oracle"]]
    variant_win_payload = {variant: variant_wins.get(variant, 0) for variant in VARIANTS}
    median_delta_payload = {variant: summarize_values(median_deltas.get(variant, [])) for variant in VARIANTS}
    stable_flips = sum(variant_win_payload[name] for name in global_variants)
    shared_constant_flips = variant_win_payload["V11_shared_constants_only_best_of_grid"]
    if (stable_flips >= 3 and len(flips_by_group["global"]) >= 3) or shared_constant_flips >= 3:
        decision = "A"
        recommendation = "A stable fixed counterfactual flips several P2 misses; one future small smoke may be justified, but no formal eval."
    elif any_flipped:
        decision = "B"
        recommendation = "Flips are limited to aggressive/per-sample or inconsistent counterfactuals; continue diagnostic residual/fingerprint analysis."
    else:
        decision = "C"
        recommendation = "No meaningful fixed counterfactual flips P2 oracle candidates; scorer/fingerprint mismatch remains."

    result = {
        "metadata": {
            "residual_json": str(args.residual_json),
            "semantic_scoring_json": str(args.semantic_scoring_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "target_samples": target_samples,
            "baseline_scorer": "constant_grid_rolewise_no_multi_u0",
            "variants_tested": VARIANTS,
            "formal_eval_run": False,
        },
        "summary": {
            "samples_analyzed": len(sample_rows),
            "baseline_oracle_wins": baseline_wins,
            "variant_oracle_wins_by_variant": variant_win_payload,
            "samples_flipped_by_any_variant": len(any_flipped),
            "samples_flipped_by_global_recurring_dims": len(flips_by_group["global"]),
            "samples_flipped_by_per_sample_top_dims": len(flips_by_group["per_sample"]),
            "samples_flipped_by_constant_counterfactual": len(flips_by_group["constant"]),
            "near_tie_samples_flipped": len(flips_by_group["near_tie"]),
            "active_trap_samples_flipped": len(flips_by_group["active_trap"]),
            "weak_trap_samples_flipped": len(flips_by_group["weak_trap"]),
            "current_candidate_trap_samples_flipped": len(flips_by_group["current_candidate_trap"]),
            "median_gap_delta_by_variant": {
                variant: values["median"] for variant, values in median_delta_payload.items()
            },
            "max_gap_delta_by_variant": {
                variant: values["max"] for variant, values in median_delta_payload.items()
            },
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
        "samples": sample_rows,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_report(json_ready(result)))
    print(f"decision={decision}")
    print(f"samples_analyzed={len(sample_rows)}")
    print(f"samples_flipped_by_any_variant={len(any_flipped)}")
    print(f"variant_oracle_wins_by_variant={dict(variant_win_payload)}")


if __name__ == "__main__":
    main()
