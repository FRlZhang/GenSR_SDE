#!/usr/bin/env python3
"""Offline all-32 shared-constant / fixed residual calibration smoke."""

from __future__ import annotations

import argparse
import json
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
from analyze_p2_residual_calibration_counterfactuals import GLOBAL_ACTIVE_DIMS, GLOBAL_WEAK_DIMS
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system, template_tokens


CONSTANT_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
ACTIVE_SLICE = slice(72, 90)
WEAK_SLICE = slice(90, 186)
VARIANTS = [
    "V0_current_rolewise_no_multi_u0",
    "V11_shared_constants_only_best_of_grid",
    "V3_active_without_dim_76",
    "V4_active_without_dims_76_72_82_88",
    "V6_active_and_weak_without_recurring_dims",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--p2-counterfactual-json", type=Path, required=True)
    parser.add_argument("--baseline-coverage-json", type=Path, required=True)
    parser.add_argument("--p2-coverage-json", type=Path, required=True)
    parser.add_argument("--target-samples", type=str, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def l2_without(residual: list[float] | np.ndarray | None, remove_global: list[int], offset: int) -> float | None:
    if residual is None:
        return None
    vec = np.asarray(residual, dtype=np.float64)
    remove_local = {dim - offset for dim in remove_global if 0 <= dim - offset < len(vec)}
    keep = [idx for idx in range(len(vec)) if idx not in remove_local]
    if not keep:
        return 0.0
    return float(np.linalg.norm(vec[keep]))


def selected_order(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not row.get("valid_bool", False),
            row.get(score_key) if row.get(score_key) is not None else float("inf"),
            row.get("candidate_source") or "",
            row.get("candidate_id") or "",
        ),
    )


def score_candidate(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
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
    out = {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_source": candidate.get("candidate_source"),
        "full_sequence_tokens": candidate.get("full_sequence_tokens"),
        "drift_source": candidate.get("drift_source"),
        "diffusion_source": candidate.get("diffusion_source"),
        "drift_rank": candidate.get("drift_rank"),
        "diffusion_rank": candidate.get("diffusion_rank"),
        "pair_rank": candidate.get("pair_rank"),
        "is_truth_drift_canonical": bool(candidate.get("is_truth_drift_canonical")),
        "is_truth_diffusion_exact": bool(candidate.get("is_truth_diffusion_exact")),
        "is_p2_canonical_oracle": bool(candidate.get("is_p2_canonical_oracle")),
        "truth_sequence": sample.get("truth_sequence"),
        "truth_drift_tokens": sample.get("truth_drift_tokens"),
        "truth_diffusion_tokens": sample.get("truth_diffusion_tokens"),
        "failure_reason": failure_reason,
        "fingerprint_failures": fingerprint_failures,
        "valid_bool": False,
        "selected_exact_bool": False,
        "selected_relaxed_bool": False,
        "is_oracle_bool": False,
        "score": None,
        "shared_score": None,
        "active_score": None,
        "weak_score": None,
        "best_constant": None,
        "best_drift_constant": None,
        "best_diffusion_constant": None,
        "active_residual": None,
        "weak_residual": None,
    }
    if details is None:
        return out
    out.update(
        {
            "valid_bool": True,
            "score": details["distance"],
            "shared_score": details["shared_constant_distance"],
            "active_score": details["active_kramers_moyal_distance"],
            "weak_score": details["gaussian_weak_kernel_distance"],
            "best_constant": details["best_constant"],
            "best_drift_constant": details["best_drift_constant"],
            "best_diffusion_constant": details["best_diffusion_constant"],
            "active_residual": details["active_kramers_moyal_residual"],
            "weak_residual": details["gaussian_weak_kernel_residual"],
        }
    )
    truth_sequence = sample.get("truth_sequence") or []
    out["selected_exact_bool"] = candidate.get("full_sequence_tokens") == truth_sequence
    out["selected_relaxed_bool"] = template_tokens(candidate.get("full_sequence_tokens") or []) == template_tokens(
        truth_sequence
    )
    out["is_oracle_bool"] = out["selected_exact_bool"]
    return out


def variant_score(row: dict[str, Any], variant: str) -> float | None:
    if not row.get("valid_bool"):
        return None
    if variant == "V0_current_rolewise_no_multi_u0":
        return row["score"]
    if variant == "V11_shared_constants_only_best_of_grid":
        return row["shared_score"]
    if variant == "V3_active_without_dim_76":
        active = l2_without(row["active_residual"], [76], ACTIVE_SLICE.start)
        weak = row["weak_score"]
        return None if active is None or weak is None else active + weak
    if variant == "V4_active_without_dims_76_72_82_88":
        active = l2_without(row["active_residual"], GLOBAL_ACTIVE_DIMS, ACTIVE_SLICE.start)
        weak = row["weak_score"]
        return None if active is None or weak is None else active + weak
    if variant == "V6_active_and_weak_without_recurring_dims":
        active = l2_without(row["active_residual"], GLOBAL_ACTIVE_DIMS, ACTIVE_SLICE.start)
        weak = l2_without(row["weak_residual"], GLOBAL_WEAK_DIMS, WEAK_SLICE.start)
        return None if active is None or weak is None else active + weak
    raise ValueError(f"unknown variant: {variant}")


def best_row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    ranked = selected_order(
        [
            {**row, "variant_score": variant_score(row, variant)}
            for row in rows
        ],
        "variant_score",
    )
    return ranked[0] if ranked and ranked[0].get("variant_score") is not None else None


def exact_match(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("selected_exact_bool"))


def relaxed_match(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("selected_relaxed_bool"))


def sample_rows_by_index(samples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in samples}


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Calibration All32 Smoke",
        "",
        "## Scope",
        "",
        "Offline all-32 shared-constant / fixed residual calibration smoke on the scorer-ready sidecar. No formal eval, rerank-mode implementation, or production scoring change was run.",
        "",
        "## Inputs",
        "",
        f"- Sidecar: `{summary['scorer_ready_json']}`",
        f"- P2 counterfactuals: `{summary['p2_counterfactual_json']}`",
        f"- Baseline coverage: `{summary['baseline_coverage_json']}`",
        f"- P2 coverage: `{summary['p2_coverage_json']}`",
        f"- Target samples: `{','.join(str(x) for x in summary['target_samples'])}`",
        "",
        "## Baseline Coverage Context",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Current expanded full oracle | {summary['current_expanded_full_oracle']} / 32 |",
        f"| P2 normalized admission coverage | {summary['p2_normalized_admission_coverage']} / 32 |",
        f"| Baseline full oracle | {summary['baseline_full_oracle']} / 32 |",
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
            f"| Candidate rows analyzed | {summary['candidate_rows_analyzed']} |",
            f"| Valid candidates | {summary['valid_candidate_count']} |",
            f"| Parse failures | {summary['parse_failures']} |",
            f"| Fingerprint failures | {summary['fingerprint_failures']} |",
            f"| V0 selected exact | {summary['variant_summaries']['V0_current_rolewise_no_multi_u0']['selected_exact_count']} |",
            f"| V0 selected relaxed | {summary['variant_summaries']['V0_current_rolewise_no_multi_u0']['selected_relaxed_no_constants_count']} |",
            f"| V11 selected exact | {summary['variant_summaries']['V11_shared_constants_only_best_of_grid']['selected_exact_count']} |",
            f"| V11 selected relaxed | {summary['variant_summaries']['V11_shared_constants_only_best_of_grid']['selected_relaxed_no_constants_count']} |",
            f"| Best variant by selected exact | {summary['best_variant_by_selected_exact']} |",
            f"| Best variant by selected relaxed | {summary['best_variant_by_selected_relaxed']} |",
            f"| V0 selected oracle samples | {summary['variant_summaries']['V0_current_rolewise_no_multi_u0']['selected_oracle_samples']} |",
            f"| V0 selected P2 oracle samples | {summary['variant_summaries']['V0_current_rolewise_no_multi_u0']['selected_p2_oracle_samples']} |",
            "",
            "## Variant Comparison",
            "",
            "| Variant | Selected exact | Selected relaxed | Selected source current | Selected source P2 | Selected oracle | Selected P2 oracle | Rescues vs V0 | Harms vs V0 | Net exact delta | Net relaxed delta | V0-hit preserved | V0-hit harmed | Near-tie rescues |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANTS:
        row = summary["variant_summaries"][variant]
        lines.append(
            "| {variant} | {selected_exact_count} | {selected_relaxed_no_constants_count} | {selected_from_current_expanded_count} | {selected_from_p2_admitted_count} | {selected_is_oracle_count} | {selected_is_p2_oracle_count} | {rescues_vs_V0_count} | {harms_vs_V0_count} | {net_selected_exact_delta_vs_V0} | {net_relaxed_delta_vs_V0} | {variant_preserved_V0_hits} | {variant_harmed_V0_hits} | {near_tie_rescues} |".format(
                variant=variant,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## P2 Subset",
            "",
            f"Samples: `{','.join(str(x) for x in summary['p2_target_samples'])}`",
            "",
            "| Variant | Oracle on P2 subset | P2 oracle on P2 subset |",
            "| --- | ---: | ---: |",
        ]
    )
    for variant in VARIANTS:
        row = summary["variant_summaries"][variant]
        lines.append(
            f"| {variant} | {row['selected_oracle_count_on_p2_subset']} | {row['selected_p2_oracle_count_on_p2_subset']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval or rerank-mode implementation unless this smoke gives net selected improvement with no V0 hit harms.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    scorer_ready = load_json(args.scorer_ready_json)
    counterfactual = load_json(args.p2_counterfactual_json)
    baseline_cov = load_json(args.baseline_coverage_json)
    p2_cov = load_json(args.p2_coverage_json)

    target_samples = parse_target_samples(args.target_samples)
    scorer_by_idx = sample_rows_by_index(scorer_ready["samples"])
    counterfactual_by_idx = {int(sample["sample_index"]): sample for sample in counterfactual["samples"]}
    baseline_by_idx = {int(sample["sample_index"]): sample for sample in baseline_cov["samples"]}
    p2_by_idx = {int(sample["sample_index"]): sample for sample in p2_cov["samples"]}

    sample_results = []
    total_candidate_rows = 0
    total_valid_candidates = 0
    total_parse_failures = 0
    total_fingerprint_failures = 0
    variant_metrics: dict[str, dict[str, Any]] = {
        variant: {
            "selected_exact_count": 0,
            "selected_relaxed_no_constants_count": 0,
            "selected_from_current_expanded_count": 0,
            "selected_from_p2_admitted_count": 0,
            "selected_is_oracle_count": 0,
            "selected_is_p2_oracle_count": 0,
            "rescues_vs_V0_count": 0,
            "harms_vs_V0_count": 0,
            "net_selected_exact_delta_vs_V0": 0,
            "net_relaxed_delta_vs_V0": 0,
            "selected_oracle_samples": [],
            "selected_p2_oracle_samples": [],
            "variant_preserved_V0_hits": 0,
            "variant_harmed_V0_hits": 0,
            "near_tie_rescues": 0,
            "selected_oracle_count_on_p2_subset": 0,
            "selected_p2_oracle_count_on_p2_subset": 0,
        }
        for variant in VARIANTS
    }

    variant_rows_per_sample: dict[int, dict[str, dict[str, Any]]] = {}
    v0_selected_oracle_samples: list[int] = []
    v0_selected_p2_oracle_samples: list[int] = []

    for idx in target_samples:
        sample = scorer_by_idx[idx]
        candidates = sample.get("candidates", [])
        scored_rows = [
            score_candidate(sample, candidate, candidate_idx)
            for candidate_idx, candidate in enumerate(candidates, start=1)
        ]
        valid_rows = [row for row in scored_rows if row.get("valid_bool")]
        total_candidate_rows += len(scored_rows)
        total_valid_candidates += len(valid_rows)
        total_parse_failures += sum(1 for row in scored_rows if row.get("failure_reason") == "parse_failed")
        total_fingerprint_failures += sum(
            1
            for row in scored_rows
            if row.get("failure_reason")
            and row.get("failure_reason") != "parse_failed"
            and not row.get("valid_bool")
        )
        selected_by_variant: dict[str, dict[str, Any] | None] = {}
        for variant in VARIANTS:
            ranked = selected_order(
                [{**row, "variant_score": variant_score(row, variant)} for row in valid_rows],
                "variant_score",
            )
            selected = ranked[0] if ranked and ranked[0].get("variant_score") is not None else None
            selected_by_variant[variant] = selected

        truth_sequence = sample.get("truth_sequence") or []
        truth_template = template_tokens(truth_sequence)
        exact_oracle_present = bool(sample.get("current_expanded_oracle_present_bool"))
        p2_oracle_present = bool(sample.get("p2_oracle_present_bool"))
        near_tie = bool(counterfactual_by_idx.get(idx, {}).get("classification_from_previous_report") == "near_tie")

        row_out = {
            "sample_index": idx,
            "truth_sequence": truth_sequence,
            "truth_drift_tokens": sample.get("truth_drift_tokens"),
            "truth_diffusion_tokens": sample.get("truth_diffusion_tokens"),
            "baseline_current_expanded_oracle_present": exact_oracle_present,
            "baseline_p2_oracle_present": p2_oracle_present,
            "baseline_current_expanded_hit": bool(baseline_by_idx.get(idx, {}).get("has_full")),
            "p2_normalized_hit": bool(p2_by_idx.get(idx, {}).get("has_full")),
            "near_tie_from_counterfactual": near_tie,
            "variants": {},
        }

        for variant in VARIANTS:
            selected = selected_by_variant[variant]
            if selected is None:
                variant_row = {
                    "selected_candidate_id": None,
                    "selected_candidate_source": None,
                    "selected_score": None,
                    "selected_exact_bool": False,
                    "selected_relaxed_bool": False,
                    "selected_is_oracle_bool": False,
                    "selected_is_p2_oracle_bool": False,
                    "selected_from_current_expanded_bool": False,
                    "selected_from_p2_admitted_bool": False,
                }
            else:
                variant_row = {
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_candidate_source": selected["candidate_source"],
                    "selected_score": selected["variant_score"],
                    "selected_exact_bool": bool(selected["selected_exact_bool"]),
                    "selected_relaxed_bool": bool(selected["selected_relaxed_bool"]),
                    "selected_is_oracle_bool": bool(selected["is_oracle_bool"]),
                    "selected_is_p2_oracle_bool": bool(selected["is_p2_canonical_oracle"]),
                    "selected_from_current_expanded_bool": selected["candidate_source"] == "current_expanded",
                    "selected_from_p2_admitted_bool": selected["candidate_source"] == "p2_admitted",
                }
            row_out["variants"][variant] = variant_row

            metrics = variant_metrics[variant]
            metrics["selected_exact_count"] += int(variant_row["selected_exact_bool"])
            metrics["selected_relaxed_no_constants_count"] += int(variant_row["selected_relaxed_bool"])
            metrics["selected_from_current_expanded_count"] += int(variant_row["selected_from_current_expanded_bool"])
            metrics["selected_from_p2_admitted_count"] += int(variant_row["selected_from_p2_admitted_bool"])
            metrics["selected_is_oracle_count"] += int(variant_row["selected_is_oracle_bool"])
            metrics["selected_is_p2_oracle_count"] += int(variant_row["selected_is_p2_oracle_bool"])
            metrics["selected_oracle_count_on_p2_subset"] += int(
                idx in counterfactual_by_idx and variant_row["selected_is_oracle_bool"]
            )
            metrics["selected_p2_oracle_count_on_p2_subset"] += int(
                idx in counterfactual_by_idx and variant_row["selected_is_p2_oracle_bool"]
            )
            if near_tie and variant_row["selected_is_oracle_bool"]:
                metrics["near_tie_rescues"] += 1

        v0 = row_out["variants"]["V0_current_rolewise_no_multi_u0"]
        v0_oracle = v0["selected_is_oracle_bool"]
        if v0_oracle:
            v0_selected_oracle_samples.append(idx)
        if v0["selected_is_p2_oracle_bool"]:
            v0_selected_p2_oracle_samples.append(idx)

        for variant in VARIANTS[1:]:
            row = row_out["variants"][variant]
            metrics = variant_metrics[variant]
            metrics["rescues_vs_V0_count"] += int((not v0_oracle) and row["selected_is_oracle_bool"])
            metrics["harms_vs_V0_count"] += int(v0_oracle and not row["selected_is_oracle_bool"])
            metrics["variant_preserved_V0_hits"] += int(v0_oracle and row["selected_is_oracle_bool"])
            metrics["variant_harmed_V0_hits"] += int(v0_oracle and not row["selected_is_oracle_bool"])
            metrics["net_selected_exact_delta_vs_V0"] += int(row["selected_exact_bool"]) - int(v0["selected_exact_bool"])
            metrics["net_relaxed_delta_vs_V0"] += int(row["selected_relaxed_bool"]) - int(v0["selected_relaxed_bool"])

        sample_results.append(row_out)

    baseline_full_oracle = sum(1 for sample in baseline_cov["samples"] if sample.get("has_full"))
    p2_admission = p2_cov.get("p2_normalized_admission", {})
    current_expanded_full_oracle = p2_admission.get(
        "current_expanded_full_oracle",
        sum(1 for sample in scorer_ready["samples"] if sample.get("current_expanded_oracle_present_bool")),
    )
    p2_normalized_admission_coverage = p2_admission.get(
        "p2_normalized_admission_coverage",
        sum(1 for sample in scorer_ready["samples"] if sample.get("p2_oracle_present_bool")),
    )

    variant_summaries = {}
    for variant in VARIANTS:
        metrics = variant_metrics[variant]
        variant_summaries[variant] = {
            **metrics,
            "selected_oracle_samples": [idx for idx in v0_selected_oracle_samples if True]
            if variant == "V0_current_rolewise_no_multi_u0"
            else [
                row["sample_index"]
                for row in sample_results
                if row["variants"][variant]["selected_is_oracle_bool"]
            ],
            "selected_p2_oracle_samples": [
                row["sample_index"]
                for row in sample_results
                if row["variants"][variant]["selected_is_p2_oracle_bool"]
            ],
        }

    best_variant_by_selected_exact = max(
        VARIANTS,
        key=lambda variant: (
            variant_summaries[variant]["selected_exact_count"],
            variant_summaries[variant]["selected_relaxed_no_constants_count"],
            -VARIANTS.index(variant),
        ),
    )
    best_variant_by_selected_relaxed = max(
        VARIANTS,
        key=lambda variant: (
            variant_summaries[variant]["selected_relaxed_no_constants_count"],
            variant_summaries[variant]["selected_exact_count"],
            -VARIANTS.index(variant),
        ),
    )

    variant11 = variant_summaries["V11_shared_constants_only_best_of_grid"]
    v0_summary = variant_summaries["V0_current_rolewise_no_multi_u0"]
    practical_variants = VARIANTS[1:]
    safe_improvements = [
        variant
        for variant in practical_variants
        if variant_summaries[variant]["variant_harmed_V0_hits"] == 0
        and (
            variant_summaries[variant]["selected_exact_count"] > v0_summary["selected_exact_count"]
            or variant_summaries[variant]["selected_p2_oracle_count_on_p2_subset"] > v0_summary[
                "selected_p2_oracle_count_on_p2_subset"
            ]
            or (
                variant_summaries[variant]["selected_exact_count"] == v0_summary["selected_exact_count"]
                and variant_summaries[variant]["selected_relaxed_no_constants_count"]
                > v0_summary["selected_relaxed_no_constants_count"]
            )
        )
    ]
    variants_with_p2_rescue = [
        variant
        for variant in practical_variants
        if variant_summaries[variant]["selected_p2_oracle_count_on_p2_subset"]
        > v0_summary["selected_p2_oracle_count_on_p2_subset"]
    ]
    variants_with_net_improvement = [
        variant
        for variant in practical_variants
        if variant_summaries[variant]["selected_exact_count"] > v0_summary["selected_exact_count"]
        or variant_summaries[variant]["selected_relaxed_no_constants_count"]
        > v0_summary["selected_relaxed_no_constants_count"]
    ]
    variants_with_harms = [
        variant for variant in practical_variants if variant_summaries[variant]["variant_harmed_V0_hits"] > 0
    ]
    if safe_improvements:
        decision = "A"
        recommendation = "A fixed calibration variant is net-positive on the all-32 sidecar with no V0 hit harms; one later implementation smoke may be justified, but no formal eval."
    elif variants_with_p2_rescue or variants_with_net_improvement or variants_with_harms:
        decision = "B"
        recommendation = "The variant rescues some P2 cases but either harms V0 hits or does not improve net selected recovery; use this as scorer diagnostics only."
    else:
        decision = "C"
        recommendation = "No fixed variant improves beyond the counterfactual signal; drop this calibration direction for now."

    result = {
        "metadata": {
            "scorer_ready_json": str(args.scorer_ready_json),
            "p2_counterfactual_json": str(args.p2_counterfactual_json),
            "baseline_coverage_json": str(args.baseline_coverage_json),
            "p2_coverage_json": str(args.p2_coverage_json),
            "target_samples": target_samples,
            "formal_eval_run": False,
            "variants_tested": VARIANTS,
        },
        "summary": {
            "scorer_ready_json": str(args.scorer_ready_json),
            "p2_counterfactual_json": str(args.p2_counterfactual_json),
            "baseline_coverage_json": str(args.baseline_coverage_json),
            "p2_coverage_json": str(args.p2_coverage_json),
            "target_samples": target_samples,
            "samples_analyzed": len(sample_results),
            "candidate_rows_analyzed": total_candidate_rows,
            "valid_candidate_count": total_valid_candidates,
            "parse_failures": total_parse_failures,
            "fingerprint_failures": total_fingerprint_failures,
            "baseline_full_oracle": baseline_full_oracle,
            "current_expanded_full_oracle": current_expanded_full_oracle,
            "p2_normalized_admission_coverage": p2_normalized_admission_coverage,
            "oracle_present_count": current_expanded_full_oracle,
            "p2_oracle_present_count": p2_normalized_admission_coverage,
            "p2_target_samples": [int(sample["sample_index"]) for sample in counterfactual["samples"]],
            "v0_selected_oracle_samples": v0_selected_oracle_samples,
            "variant_summaries": variant_summaries,
            "best_variant_by_selected_exact": best_variant_by_selected_exact,
            "best_variant_by_selected_relaxed": best_variant_by_selected_relaxed,
            "safe_improvement_variants": safe_improvements,
            "variants_with_p2_rescue": variants_with_p2_rescue,
            "variants_with_net_improvement": variants_with_net_improvement,
            "variants_with_v0_hit_harms": variants_with_harms,
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
        "samples": sample_results,
    }

    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_report(json_ready(result)))
    summary = result["summary"]
    print(f"decision={decision}")
    print(f"samples_analyzed={len(sample_results)}")
    print(f"candidate_rows_analyzed={summary['candidate_rows_analyzed']}")
    print(f"valid_candidate_count={summary['valid_candidate_count']}")
    print(f"v0_selected_exact={variant_summaries['V0_current_rolewise_no_multi_u0']['selected_exact_count']}")
    print(f"v11_selected_exact={variant_summaries['V11_shared_constants_only_best_of_grid']['selected_exact_count']}")


if __name__ == "__main__":
    main()
