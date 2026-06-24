#!/usr/bin/env python3
"""Residual/segment diagnostics for P2 oracle score misses."""

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

from analyze_p2_admitted_candidate_scoring import json_ready, parse_target_samples, score_seed, target_array
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system


ACTIVE_SLICE = slice(72, 90)
WEAK_SLICE = slice(90, 186)
CONSTANT_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def summary(values: list[float | None]) -> dict[str, float | None]:
    finite = finite_values(values)
    if not finite:
        return {"min": None, "median": None, "max": None}
    return {"min": min(finite), "median": statistics.median(finite), "max": max(finite)}


def candidate_index_by_id(sample: dict[str, Any]) -> dict[str, int]:
    return {
        row["candidate_id"]: index
        for index, row in enumerate(sample.get("candidates", []), start=1)
    }


def candidate_by_id(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in sample.get("candidates", [])}


def score_candidate(sample: dict[str, Any], candidate: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    sample_index = int(sample["sample_index"])
    details, failure_reason, fingerprint_failures = score_candidate_system(
        candidate["full_sequence_tokens"],
        target_array(sample),
        FingerprintConfig(n_paths=800, n_steps=60, active_paths=800),
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(sample_index, candidate_index),
    )
    if details is None:
        return {
            "candidate_id": candidate["candidate_id"],
            "sequence": candidate.get("full_sequence_tokens"),
            "source": candidate.get("candidate_source"),
            "score": None,
            "active_score": None,
            "weak_score": None,
            "best_drift_constant": None,
            "best_diffusion_constant": None,
            "active_residual": None,
            "weak_residual": None,
            "failure_reason": failure_reason,
            "fingerprint_failures": fingerprint_failures,
        }
    return {
        "candidate_id": candidate["candidate_id"],
        "sequence": candidate.get("full_sequence_tokens"),
        "source": candidate.get("candidate_source"),
        "drift_source": candidate.get("drift_source"),
        "diffusion_source": candidate.get("diffusion_source"),
        "drift_rank": candidate.get("drift_rank"),
        "diffusion_rank": candidate.get("diffusion_rank"),
        "pair_rank": candidate.get("pair_rank"),
        "score": details["distance"],
        "active_score": details["active_kramers_moyal_distance"],
        "weak_score": details["gaussian_weak_kernel_distance"],
        "full_score": details["full_distance"],
        "multi_u0_score": details["multi_u0_moments_distance"],
        "best_drift_constant": details["best_drift_constant"],
        "best_diffusion_constant": details["best_diffusion_constant"],
        "active_residual": details["active_kramers_moyal_residual"],
        "weak_residual": details["gaussian_weak_kernel_residual"],
        "failure_reason": failure_reason,
        "fingerprint_failures": fingerprint_failures,
    }


def residual_advantages(
    selected_residual: list[float] | np.ndarray | None,
    oracle_residual: list[float] | np.ndarray | None,
    segment_offset: int,
    limit: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if selected_residual is None or oracle_residual is None:
        return [], []
    selected_abs = np.abs(np.asarray(selected_residual, dtype=np.float64))
    oracle_abs = np.abs(np.asarray(oracle_residual, dtype=np.float64))
    selected_adv = oracle_abs - selected_abs
    oracle_adv = selected_abs - oracle_abs

    def rows(values: np.ndarray, key: str) -> list[dict[str, Any]]:
        ranked = sorted(
            [
                (
                    float(values[idx]),
                    idx,
                    float(selected_abs[idx]),
                    float(oracle_abs[idx]),
                )
                for idx in range(len(values))
                if values[idx] > 0
            ],
            reverse=True,
        )
        return [
            {
                "dimension_index": segment_offset + idx,
                "segment_local_index": idx,
                "selected_abs_residual": selected_value,
                "oracle_abs_residual": oracle_value,
                key: value,
            }
            for value, idx, selected_value, oracle_value in ranked[:limit]
        ]

    return rows(selected_adv, "selected_advantage"), rows(oracle_adv, "oracle_advantage")


def dominant_gap(score_gap: float, active_gap: float, weak_gap: float) -> str:
    if score_gap <= 0.10:
        return "near_tie"
    positive_active = active_gap if active_gap > 0 else 0.0
    positive_weak = weak_gap if weak_gap > 0 else 0.0
    if positive_active > 0 and positive_weak > 0 and abs(positive_active - positive_weak) <= 0.05:
        return "mixed"
    if positive_active >= positive_weak and positive_active > 0:
        return "active"
    if positive_weak > positive_active:
        return "weak"
    return "mixed"


def classify(score_gap: float, active_gap: float, weak_gap: float, selected_source: str) -> str:
    if score_gap <= 0.10:
        return "near_tie"
    segment = dominant_gap(score_gap, active_gap, weak_gap)
    if segment == "active":
        return "active_trap"
    if segment == "weak":
        return "weak_trap"
    if selected_source == "p2_admitted":
        return "p2_wrong_candidate_trap"
    if selected_source == "current_expanded":
        return "current_candidate_trap"
    return "broad_score_miss"


def analyze_sample(
    semantic_sample: dict[str, Any],
    sidecar_sample: dict[str, Any],
) -> dict[str, Any]:
    by_id = candidate_by_id(sidecar_sample)
    indices = candidate_index_by_id(sidecar_sample)
    selected_id = semantic_sample["combined"]["selected_candidate"]["candidate_id"]
    oracle_id = semantic_sample["p2_oracle"]["candidate_id"]
    selected_sidecar = by_id[selected_id]
    oracle_sidecar = by_id[oracle_id]
    selected = score_candidate(sidecar_sample, selected_sidecar, indices[selected_id])
    oracle = score_candidate(sidecar_sample, oracle_sidecar, indices[oracle_id])
    score_gap = oracle["score"] - selected["score"]
    active_gap = oracle["active_score"] - selected["active_score"]
    weak_gap = oracle["weak_score"] - selected["weak_score"]
    active_selected_adv, active_oracle_adv = residual_advantages(
        selected["active_residual"],
        oracle["active_residual"],
        ACTIVE_SLICE.start,
    )
    weak_selected_adv, weak_oracle_adv = residual_advantages(
        selected["weak_residual"],
        oracle["weak_residual"],
        WEAK_SLICE.start,
    )
    constants_differ = (
        selected["best_drift_constant"] != oracle["best_drift_constant"]
        or selected["best_diffusion_constant"] != oracle["best_diffusion_constant"]
    )
    classification = classify(score_gap, active_gap, weak_gap, selected["source"])
    return {
        "sample_index": sidecar_sample["sample_index"],
        "truth_sequence": sidecar_sample["truth_sequence"],
        "selected_sequence": selected["sequence"],
        "selected_source": selected["source"],
        "selected_score": selected["score"],
        "selected_active_score": selected["active_score"],
        "selected_weak_score": selected["weak_score"],
        "selected_best_drift_constant": selected["best_drift_constant"],
        "selected_best_diffusion_constant": selected["best_diffusion_constant"],
        "p2_oracle_sequence": oracle["sequence"],
        "p2_oracle_rank": semantic_sample["p2_oracle"]["combined_rank"],
        "p2_oracle_score": oracle["score"],
        "p2_oracle_active_score": oracle["active_score"],
        "p2_oracle_weak_score": oracle["weak_score"],
        "p2_oracle_best_drift_constant": oracle["best_drift_constant"],
        "p2_oracle_best_diffusion_constant": oracle["best_diffusion_constant"],
        "score_gap": score_gap,
        "active_gap": active_gap,
        "weak_gap": weak_gap,
        "dominant_gap_segment": dominant_gap(score_gap, active_gap, weak_gap),
        "top_active_residual_advantages_for_selected": active_selected_adv,
        "top_weak_residual_advantages_for_selected": weak_selected_adv,
        "top_active_residual_advantages_for_oracle": active_oracle_adv,
        "top_weak_residual_advantages_for_oracle": weak_oracle_adv,
        "classification": classification,
        "selected_is_p2_nonoracle": selected["source"] == "p2_admitted" and selected_id != oracle_id,
        "selected_is_current_candidate": selected["source"] == "current_expanded",
        "constant_choice_differs": constants_differ,
    }


def frequency(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for item in row.get(key, [])[:2]:
            counts[str(item["dimension_index"])] += 1
    return dict(counts.most_common())


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Oracle Score Miss Residual Diagnostics",
        "",
        "## Scorer",
        "",
        "`constant_grid_rolewise_no_multi_u0`, active:weak `1:1`, constants `0.25,0.5,1.0,2.0,4.0`; active slice `[72,90)`, weak slice `[90,186)`.",
        "",
        "No formal eval or P2 eval integration was run.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Samples analyzed | {summary['samples_analyzed']} |",
        f"| Constant mismatch count | {summary['constant_mismatch_count']} |",
        "",
        f"Selected source counts: `{json.dumps(summary['selected_source_counts'], sort_keys=True)}`",
        f"Dominant gap segment counts: `{json.dumps(summary['dominant_gap_segment_counts'], sort_keys=True)}`",
        f"Classification counts: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
        f"Near-gap counts: `{json.dumps(summary['near_gap_counts'], sort_keys=True)}`",
        f"Score gap summary: `{json.dumps(summary['score_gap_summary'], sort_keys=True)}`",
        f"Active gap summary: `{json.dumps(summary['active_gap_summary'], sort_keys=True)}`",
        f"Weak gap summary: `{json.dumps(summary['weak_gap_summary'], sort_keys=True)}`",
        "",
        "## Per-Sample Comparison",
        "",
        "| Sample | Selected source | P2 rank | Score gap | Active gap | Weak gap | Dominant | Classification |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for sample in result["samples"]:
        lines.append(
            f"| {sample['sample_index']} | {sample['selected_source']} | {sample['p2_oracle_rank']} | "
            f"{sample['score_gap']:.6f} | {sample['active_gap']:.6f} | {sample['weak_gap']:.6f} | "
            f"{sample['dominant_gap_segment']} | {sample['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Top Residual Dimensions",
            "",
            f"- Active selected-advantage frequency: `{json.dumps(summary['top_active_dimensions_frequency'], sort_keys=True)}`",
            f"- Weak selected-advantage frequency: `{json.dumps(summary['top_weak_dimensions_frequency'], sort_keys=True)}`",
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval or P2 eval integration is recommended unless residual diagnostics later support it.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    semantic = load_json(args.semantic_scoring_json)
    sidecar = load_json(args.scorer_ready_json)
    target_samples = parse_target_samples(args.target_samples)
    semantic_by_idx = {int(sample["sample_index"]): sample for sample in semantic["samples"]}
    sidecar_by_idx = {int(sample["sample_index"]): sample for sample in sidecar["samples"]}
    samples = [
        analyze_sample(semantic_by_idx[idx], sidecar_by_idx[idx])
        for idx in target_samples
    ]
    selected_source_counts = Counter(sample["selected_source"] for sample in samples)
    dominant_counts = Counter(sample["dominant_gap_segment"] for sample in samples)
    classification_counts = Counter(sample["classification"] for sample in samples)
    score_gaps = [sample["score_gap"] for sample in samples]
    active_gaps = [sample["active_gap"] for sample in samples]
    weak_gaps = [sample["weak_gap"] for sample in samples]
    near_gap_counts = {
        str(threshold): sum(1 for value in score_gaps if value <= threshold)
        for threshold in (0.005, 0.01, 0.05, 0.10)
    }
    if classification_counts.get("active_trap", 0) + classification_counts.get("weak_trap", 0) >= 4:
        decision = "B"
        recommendation = "Mostly active/weak residual traps; do not integrate P2 yet, diagnose residual behavior on these cases only."
    elif classification_counts.get("near_tie", 0) + classification_counts.get("constant_choice_trap", 0) >= 4:
        decision = "A"
        recommendation = "Mostly near-tie or constant-choice misses; a tiny constant-handling smoke may be justified, but no formal eval."
    else:
        decision = "B"
        recommendation = "Mixed score misses; inspect residual behavior before any P2 eval integration."
    result = {
        "metadata": {
            "semantic_scoring_json": str(args.semantic_scoring_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "target_samples": target_samples,
            "scorer": "constant_grid_rolewise_no_multi_u0",
            "active_slice": [ACTIVE_SLICE.start, ACTIVE_SLICE.stop],
            "weak_slice": [WEAK_SLICE.start, WEAK_SLICE.stop],
            "formal_eval_run": False,
        },
        "summary": {
            "samples_analyzed": len(samples),
            "selected_source_counts": dict(selected_source_counts),
            "dominant_gap_segment_counts": dict(dominant_counts),
            "classification_counts": dict(classification_counts),
            "near_gap_counts": near_gap_counts,
            "active_gap_summary": summary(active_gaps),
            "weak_gap_summary": summary(weak_gaps),
            "score_gap_summary": summary(score_gaps),
            "top_active_dimensions_frequency": frequency(samples, "top_active_residual_advantages_for_selected"),
            "top_weak_dimensions_frequency": frequency(samples, "top_weak_residual_advantages_for_selected"),
            "constant_mismatch_count": sum(1 for sample in samples if sample["constant_choice_differs"]),
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
        "samples": samples,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_report(json_ready(result)))
    print(f"decision={decision}")
    print(f"samples_analyzed={len(samples)}")
    print(f"selected_source_counts={dict(selected_source_counts)}")
    print(f"classification_counts={dict(classification_counts)}")


if __name__ == "__main__":
    main()
