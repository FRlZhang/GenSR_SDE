#!/usr/bin/env python3
"""Run the P2 normalized drift admission coverage hook over existing JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from normalized_drift_admission import (
    build_p2_normalized_admission_pool,
    canonical_drift_set,
    canonicalize_constant_chains,
    classify_drift_family,
    collision_summary,
    current_expanded_drift_rows,
    drift_only_candidate_rows,
    safety_checks,
    select_p2_one_per_family,
    token_text,
)


EXPECTED = {
    "current_expanded_full_oracle": 15,
    "canonicalized_expanded_pool_coverage": 23,
    "p2_coverage": 30,
    "p2_pair_count": 340,
    "newly_recovered": [0, 2, 6, 7, 15, 25, 29],
    "remaining_missing": [16, 28],
}


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-coverage-json", type=Path, default=Path("candidate_coverage_pair_expanded.json"))
    parser.add_argument("--drift-only-json", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument("--report", type=Path, default=Path("candidate_coverage_p2_normalized_admission.md"))
    parser.add_argument("--json-output", type=Path, default=Path("candidate_coverage_p2_normalized_admission.json"))
    parser.add_argument("--admission-policy", choices=["one_per_family"], default="one_per_family")
    parser.add_argument("--admission-source", choices=["beam", "all"], default="beam")
    parser.add_argument("--include-sampling", type=parse_bool, default=False)
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def source_breakdown(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for sample in samples:
        truth_key = tuple(sample["canonical_truth_drift_tokens"] or [])
        sources = {
            row.get("source")
            for row in sample["admitted_candidates"]
            if tuple(row.get("canonical_drift_tokens", [])) == truth_key and row.get("source")
        }
        if sources:
            counts["+".join(sorted(sources))] += 1
        elif sample["current_canonical_full_oracle"]:
            counts["current_expanded"] += 1
    return dict(counts)


def full_normalized_pair_count(expanded: dict[str, Any], drift_only: dict[str, Any]) -> tuple[int, int]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    pair_count = 0
    coverage_hits = 0
    for drift_sample in drift_only["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        current_keys = canonical_drift_set(current_expanded_drift_rows(expanded_sample))
        all_tail_keys = canonical_drift_set(drift_only_candidate_rows(drift_sample, include_sampling=True))
        truth_key = tuple(canonicalize_constant_chains(expanded_sample["truth_drift"]) or [])
        combined = current_keys | all_tail_keys
        pair_count += len(combined) * int(expanded_sample.get("unique_diffusions", 0))
        if expanded_sample.get("has_diffusion") and truth_key in combined:
            coverage_hits += 1
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    return current_full + coverage_hits, pair_count


def truth_upper_bound_pair_count(expanded: dict[str, Any], drift_only: dict[str, Any]) -> tuple[int, int]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    pair_count = 0
    coverage_hits = 0
    for drift_sample in drift_only["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        current_keys = canonical_drift_set(current_expanded_drift_rows(expanded_sample))
        truth_key = tuple(canonicalize_constant_chains(expanded_sample["truth_drift"]) or [])
        tail_keys = canonical_drift_set(drift_only_candidate_rows(drift_sample, include_sampling=True))
        admitted = {truth_key} if truth_key in tail_keys else set()
        combined = current_keys | admitted
        pair_count += len(combined) * int(expanded_sample.get("unique_diffusions", 0))
        if expanded_sample.get("has_diffusion") and truth_key in combined:
            coverage_hits += 1
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    return current_full + coverage_hits, pair_count


def sampling_recovery_count(samples: list[dict[str, Any]]) -> int:
    count = 0
    for sample in samples:
        truth_key = tuple(sample["canonical_truth_drift_tokens"] or [])
        if any(
            row.get("source") == "sampling" and tuple(row.get("canonical_drift_tokens", [])) == truth_key
            for row in sample["admitted_candidates"]
        ):
            count += 1
    return count


def run_self_checks(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = payload["safety"]["self_checks"]
    if not all(checks.values()):
        failures.append(f"unsafe canonicalizer checks failed: {checks}")
    summary = payload["summary"]
    if summary["current_expanded_full_oracle"] != EXPECTED["current_expanded_full_oracle"]:
        failures.append("current expanded coverage mismatch")
    if summary["canonicalized_expanded_pool_coverage"] != EXPECTED["canonicalized_expanded_pool_coverage"]:
        failures.append("canonicalized expanded-pool coverage mismatch")
    if summary["p2_normalized_admission_coverage"] != EXPECTED["p2_coverage"]:
        failures.append("P2 coverage mismatch")
    if summary["p2_pair_count"] != EXPECTED["p2_pair_count"]:
        failures.append("P2 pair count mismatch")
    if summary["newly_recovered_beyond_current"] != EXPECTED["newly_recovered"]:
        failures.append("newly recovered sample mismatch")
    if summary["remaining_missing_samples"] != EXPECTED["remaining_missing"]:
        failures.append("remaining missing sample mismatch")
    if summary["sampling_recovered_canonical_drift_count"] != 0:
        failures.append("sampling recovered canonical drift unexpectedly")
    if payload["safety"]["unsafe_collision_flag"]:
        failures.append("unsafe collision flag is true")
    return failures


def analyze(expanded: dict[str, Any], drift_only: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pool = build_p2_normalized_admission_pool(
        expanded,
        drift_only,
        admission_source=args.admission_source,
        include_sampling=args.include_sampling,
    )
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    target_samples = pool["samples"]
    current_canonical_hits = [sample for sample in target_samples if sample["current_canonical_full_oracle"]]
    p2_hits = [sample for sample in target_samples if sample["p2_normalized_full_oracle"]]
    newly = [
        sample["sample_index"]
        for sample in p2_hits
        if not sample["current_canonical_full_oracle"]
    ]
    remaining = [sample["sample_index"] for sample in target_samples if not sample["p2_normalized_full_oracle"]]
    p2_pair_count = sum(sample["pair_count"] for sample in target_samples)
    pre_pair_count = sum(
        int(sample.get("unique_drifts", 0)) * int(sample.get("unique_diffusions", 0))
        for sample in expanded["samples"]
        if sample["sample_index"] in {row["sample_index"] for row in target_samples}
    )
    full_coverage, full_pair_count = full_normalized_pair_count(expanded, drift_only)
    upper_coverage, upper_pair_count = truth_upper_bound_pair_count(expanded, drift_only)
    family_counts = Counter(sample["drift_family"] for sample in p2_hits)
    admitted_by_sample = {
        sample["sample_index"]: [
            {
                "raw_drift_tokens": row["raw_drift_tokens"],
                "canonical_drift_tokens": row["canonical_drift_tokens"],
                "source": row.get("source"),
                "rank": row.get("source_rank"),
                "family": row.get("family"),
                "admitted": row.get("admitted"),
                "filter_reason": row.get("filter_reason"),
            }
            for row in sample["admitted_candidates"]
        ]
        for sample in target_samples
    }
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    all_rows = []
    polynomial_truths = []
    for drift_sample in drift_only["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        if drift_sample.get("drift_family") == "polynomial-like drift":
            polynomial_truths.append(expanded_sample["truth_drift"])
        all_rows.extend(current_expanded_drift_rows(expanded_sample))
        all_rows.extend(drift_only_candidate_rows(drift_sample, include_sampling=True))
    safety = collision_summary(all_rows, polynomial_truths)
    safety["self_checks"] = safety_checks()
    payload = {
        "metadata": {
            "admission_policy": args.admission_policy,
            "admission_source": args.admission_source,
            "include_sampling": args.include_sampling,
            "expected_reference": EXPECTED,
        },
        "summary": {
            "total_samples_analyzed": len(expanded["samples"]),
            "target_oracle_absent_samples_analyzed": len(target_samples),
            "current_expanded_full_oracle": current_full,
            "canonicalized_expanded_pool_coverage": current_full + len(current_canonical_hits),
            "p2_normalized_admission_coverage": current_full + len(p2_hits),
            "full_normalized_admission_coverage": full_coverage,
            "truth_aware_upper_bound_coverage": upper_coverage,
            "newly_recovered_beyond_current": newly,
            "remaining_missing_samples": remaining,
            "p2_pair_count": p2_pair_count,
            "pre_admission_pair_count": pre_pair_count,
            "full_normalized_admission_pair_count": full_pair_count,
            "truth_aware_upper_bound_pair_count": upper_pair_count,
            "pair_pressure_reduction_vs_full": full_pair_count - p2_pair_count,
            "pair_pressure_reduction_vs_full_percent": 0.0 if full_pair_count == 0 else 100.0 * (full_pair_count - p2_pair_count) / full_pair_count,
            "sampling_excluded": not args.include_sampling,
            "sampling_recovered_canonical_drift_count": sampling_recovery_count(target_samples),
        },
        "admitted_candidates_by_sample": admitted_by_sample,
        "source_breakdown": source_breakdown(target_samples),
        "family_breakdown": {family: family_counts.get(family, 0) for family in [
            "linear drift",
            "sin drift",
            "nested-mul linear drift",
            "polynomial-like drift",
            "other",
        ]},
        "safety": safety,
        "samples": target_samples,
    }
    payload["self_check_failures"] = run_self_checks(payload)
    return payload


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    safety = payload["safety"]
    lines = [
        "# P2 Normalized Drift Admission Coverage Hook",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: coverage-only runner over existing JSON artifacts. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.",
        "",
        "## Summary",
        "",
        f"- Total samples analyzed: `{summary['total_samples_analyzed']}`.",
        f"- Target oracle-absent samples analyzed: `{summary['target_oracle_absent_samples_analyzed']}`.",
        f"- Current expanded full oracle coverage: `{summary['current_expanded_full_oracle']}/32`.",
        f"- Canonicalized expanded-pool coverage: `{summary['canonicalized_expanded_pool_coverage']}/32`.",
        f"- P2 normalized admission coverage: `{summary['p2_normalized_admission_coverage']}/32`.",
        f"- Full normalized admission coverage: `{summary['full_normalized_admission_coverage']}/32`.",
        f"- Newly recovered samples: `{','.join(str(i) for i in summary['newly_recovered_beyond_current'])}`.",
        f"- Remaining missing samples: `{','.join(str(i) for i in summary['remaining_missing_samples'])}`.",
        f"- Sampling excluded: `{summary['sampling_excluded']}`; sampling recovered canonical drifts: `{summary['sampling_recovered_canonical_drift_count']}`.",
        "",
        "## Pair Pressure",
        "",
        f"- Pre-admission target pair count: `{summary['pre_admission_pair_count']}`.",
        f"- P2 pair count: `{summary['p2_pair_count']}`.",
        f"- Full normalized admission pair count: `{summary['full_normalized_admission_pair_count']}`.",
        f"- Pair-pressure reduction versus full: `{summary['pair_pressure_reduction_vs_full']}` (`{summary['pair_pressure_reduction_vs_full_percent']:.1f}%`).",
        f"- Truth-aware upper bound pair count: `{summary['truth_aware_upper_bound_pair_count']}`.",
        "",
        "## Breakdowns",
        "",
        f"- Source breakdown: `{json.dumps(payload['source_breakdown'], sort_keys=True)}`.",
        f"- Family breakdown: `{json.dumps(payload['family_breakdown'], sort_keys=True)}`.",
        "",
        "## Admitted Candidates",
        "",
        "| Sample | Family | Admitted representatives |",
        "| ---: | --- | --- |",
    ]
    by_sample = payload["admitted_candidates_by_sample"]
    sample_meta = {sample["sample_index"]: sample for sample in payload["samples"]}
    for sample_idx in sorted(by_sample):
        reps = "<br>".join(
            f"`{token_text(row['canonical_drift_tokens'])}` ({row['family']}, {row['source']}:{row['rank']})"
            for row in by_sample[sample_idx]
        )
        lines.append(f"| {sample_idx} | {sample_meta[sample_idx]['drift_family']} | {reps} |")
    lines.extend(
        [
            "",
            "## Safety And Self-Checks",
            "",
            f"- Unsafe collision flag: `{safety['unsafe_collision_flag']}`.",
            f"- Nonconstant structural merge detected: `{safety['nonconstant_structural_merge_detected']}`.",
            f"- Polynomial-like truth collides with linear-like: `{safety['polynomial_like_truth_collides_with_linear_like']}`.",
            f"- Canonicalizer self-checks: `{json.dumps(safety['self_checks'], sort_keys=True)}`.",
            f"- Collision groups: `{safety['collision_group_count']}`.",
            f"- Hook self-check failures: `{payload['self_check_failures']}`.",
            "",
            "This is token-structure-only validation, not semantic fingerprint/rerank validation.",
            "",
            "No formal eval is recommended.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expanded = load_json(args.expanded_coverage_json)
    drift_only = load_json(args.drift_only_json)
    payload = analyze(expanded, drift_only, args)
    if payload["self_check_failures"] and not args.allow_mismatch:
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        write_report(args.report, payload)
        raise SystemExit("self-check failed: " + "; ".join(payload["self_check_failures"]))
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    summary = payload["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"current_expanded_full_oracle={summary['current_expanded_full_oracle']}")
    print(f"canonicalized_expanded_pool_coverage={summary['canonicalized_expanded_pool_coverage']}")
    print(f"p2_normalized_admission_coverage={summary['p2_normalized_admission_coverage']}")
    print(f"p2_pair_count={summary['p2_pair_count']}")
    print(f"self_check_failures={len(payload['self_check_failures'])}")


if __name__ == "__main__":
    main()
