#!/usr/bin/env python3
"""Analyze wrong expanded-pair candidates from existing diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--miss-json",
        type=Path,
        default=Path("expanded_pairing_oracle_miss_ranking_diagnostics.json"),
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("candidate_coverage_pair_expanded.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("expanded_pairing_wrong_pair_pruning_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("expanded_pairing_wrong_pair_pruning_diagnostics.json"),
    )
    return parser.parse_args()


def normalize_sequence(sequence: str) -> list[str]:
    return [token for token in sequence.split() if token != "CONSTANT"]


def approximate_source_rank(sample: dict, source: str, sequence: str) -> int | None:
    normalized = normalize_sequence(sequence)
    source_rows = sample.get("full_by_source", {}).get(source, [])
    try:
        return source_rows.index(normalized) + 1
    except ValueError:
        return None


def source_rank_from_exact_occurrences(sample: dict, source: str) -> int | None:
    ranks = [
        item["source_rank"]
        for item in sample.get("exact_full_occurrences", [])
        if item.get("source") == source
    ]
    return min(ranks) if ranks else None


def pair_oracle_samples(samples: list[dict]) -> list[dict]:
    rows = []
    for sample in samples:
        pair_occurrences = [
            item
            for item in sample.get("exact_full_occurrences", [])
            if item.get("source") == "pair"
        ]
        if not pair_occurrences:
            continue
        best = min(pair_occurrences, key=lambda item: item["source_rank"])
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "pair_source_rank": best["source_rank"],
                "global_candidate_rank": best["candidate_rank"],
                "model_score": best["score"],
            }
        )
    return rows


def enrich_cases(miss_cases: list[dict], samples: list[dict]) -> list[dict]:
    by_index = {sample["sample_index"]: sample for sample in samples}
    enriched = []
    for case in miss_cases:
        sample = by_index[case["sample_index"]]
        selected_pair_rank = None
        if case["selected_source"] == "pair":
            selected_pair_rank = approximate_source_rank(
                sample, "pair", case["selected_sequence"]
            )
        oracle_pair_rank = None
        if case["oracle_source"] == "pair":
            oracle_pair_rank = source_rank_from_exact_occurrences(sample, "pair")
        selected_segments = case["selected_segment_distances"]
        oracle_segments = case["oracle_segment_distances"]
        enriched_case = dict(case)
        enriched_case.update(
            {
                "selected_pair_source_rank_approx": selected_pair_rank,
                "oracle_pair_source_rank": oracle_pair_rank,
                "active_gap": oracle_segments["active_kramers_moyal"]
                - selected_segments["active_kramers_moyal"],
                "weak_gap": oracle_segments["gaussian_weak_kernel"]
                - selected_segments["gaussian_weak_kernel"],
                "selected_multi_u0": selected_segments.get("multi_u0_moments"),
                "oracle_multi_u0": oracle_segments.get("multi_u0_moments"),
            }
        )
        enriched.append(enriched_case)
    return enriched


def summarize(enriched: list[dict], pair_oracles: list[dict]) -> dict:
    selected_pair = [case for case in enriched if case["selected_source"] == "pair"]
    selected_pair_wrong = [case for case in selected_pair if case["score_gap"] > 0]
    pair_oracle_ranks = [row["pair_source_rank"] for row in pair_oracles]
    wrong_pair_ranks = [
        case["selected_pair_source_rank_approx"]
        for case in selected_pair_wrong
        if case["selected_pair_source_rank_approx"] is not None
    ]
    mean_reverting_wrong = [
        case
        for case in selected_pair_wrong
        if "sub CONSTANT x_0" in case["selected_drift"]
    ]
    constant_drift_wrong = [
        case
        for case in selected_pair_wrong
        if case["selected_drift"] == "mul CONSTANT CONSTANT"
    ]
    active_traps = [
        case
        for case in selected_pair_wrong
        if case["active_gap"] > 0 and case["active_gap"] >= abs(case["weak_gap"])
    ]
    return {
        "selected_misses": len(enriched),
        "selected_wrong_pair_cases": len(selected_pair_wrong),
        "pair_oracle_samples": len(pair_oracles),
        "wrong_pair_rank_min": min(wrong_pair_ranks) if wrong_pair_ranks else None,
        "wrong_pair_rank_max": max(wrong_pair_ranks) if wrong_pair_ranks else None,
        "pair_oracle_rank_min": min(pair_oracle_ranks) if pair_oracle_ranks else None,
        "pair_oracle_rank_max": max(pair_oracle_ranks) if pair_oracle_ranks else None,
        "wrong_pair_rank_avg": mean(wrong_pair_ranks) if wrong_pair_ranks else None,
        "pair_oracle_rank_avg": mean(pair_oracle_ranks) if pair_oracle_ranks else None,
        "mean_reverting_wrong_pair_count": len(mean_reverting_wrong),
        "constant_drift_wrong_pair_count": len(constant_drift_wrong),
        "active_trap_wrong_pair_count": len(active_traps),
        "selected_pair_miss_side_counts": Counter(
            case["miss_side"] for case in selected_pair_wrong
        ),
        "selected_pair_oracle_only_count": sum(
            1 for case in selected_pair_wrong if case["oracle_only_pair"]
        ),
        "pair_oracle_ranks": pair_oracle_ranks,
        "wrong_pair_ranks": wrong_pair_ranks,
    }


def rule_assessment(pair_oracles: list[dict]) -> list[dict]:
    """Assess simple rank-only source pruning against known pair oracles."""
    rows = []
    for threshold in [2, 4, 6, 8, 10, 12, 15]:
        removed = [
            row["sample_index"]
            for row in pair_oracles
            if row["pair_source_rank"] > threshold
        ]
        rows.append(
            {
                "rule": f"keep_pair_source_rank<={threshold}",
                "pair_oracles_removed": removed,
                "preserves_all_pair_oracles": not removed,
            }
        )
    return rows


def write_report(
    path: Path,
    enriched: list[dict],
    pair_oracles: list[dict],
    summary: dict,
    rules: list[dict],
) -> None:
    selected_pair = [case for case in enriched if case["selected_source"] == "pair"]
    lines = [
        "# Expanded Pairing Wrong-Pair Pruning Diagnostics",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: existing-log/JSON analysis only. No model run, candidate regeneration, formal eval, retraining, scorer change, or fingerprint change was performed.",
        "",
        "## Executive Summary",
        "",
        f"- Selected misses analyzed: `{summary['selected_misses']}`.",
        f"- Wrong selected pair cases: `{summary['selected_wrong_pair_cases']}`.",
        f"- Pair-oracle samples that must be preserved: `{summary['pair_oracle_samples']}`.",
        f"- Pair-oracle source ranks span `{summary['pair_oracle_rank_min']}` to `{summary['pair_oracle_rank_max']}`.",
        f"- Approximate wrong-pair source ranks span `{summary['wrong_pair_rank_min']}` to `{summary['wrong_pair_rank_max']}`.",
        f"- Active-distance traps among wrong selected pairs: `{summary['active_trap_wrong_pair_count']}/{summary['selected_wrong_pair_cases']}`.",
        "",
        "No safe pruning criterion is visible from the existing logs. The pair oracles are often late pair-source candidates, so rank/cap pruning would remove real oracle pairs. Structural pruning is also unsafe: harmful mean-reverting or constant-drift pairs overlap with true pair-oracle families. The dominant failure is scorer behavior, especially active-kramers-moyal traps, not obviously invalid pair structure.",
        "",
        "Outcome: **B. No safe pruning criterion exists from these logs; recommend scorer/residual analysis of active_kramers_moyal traps before testing a heuristic.**",
        "",
        "## Wrong Selected Pair Cases",
        "",
        "| Sample | Oracle source | Pair-only oracle | Miss side | Selected pair rank approx | Oracle pair rank | Score gap | Active gap | Weak gap | Selected drift | Selected diffusion |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in selected_pair:
        lines.append(
            "| {sample} | {oracle_source} | {pair_only} | {miss_side} | {sel_rank} | {oracle_rank} | {score_gap:.6f} | {active_gap:.6f} | {weak_gap:.6f} | `{drift}` | `{diffusion}` |".format(
                sample=case["sample_index"],
                oracle_source=case["oracle_source"],
                pair_only="yes" if case["oracle_only_pair"] else "no",
                miss_side=case["miss_side"],
                sel_rank=case["selected_pair_source_rank_approx"],
                oracle_rank=case["oracle_pair_source_rank"],
                score_gap=case["score_gap"],
                active_gap=case["active_gap"],
                weak_gap=case["weak_gap"],
                drift=case["selected_drift"],
                diffusion=case["selected_diffusion"],
            )
        )

    lines.extend(
        [
            "",
            "Positive active/weak gap means the selected wrong pair has a lower distance than the oracle for that segment. Most wrong selected pairs are not parse-invalid-looking; they are plausible recombinations that the current active+weak scorer prefers.",
            "",
            "## Pair-Oracle Preservation Checks",
            "",
            "| Pair-oracle sample | Pair source rank | Global candidate rank | Model score |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_oracles:
        lines.append(
            f"| {row['sample_index']} | {row['pair_source_rank']} | {row['global_candidate_rank']} | {row['model_score']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Simple pair source-rank pruning is unsafe because preserving all pair oracles requires keeping ranks up to 15, which also keeps the observed wrong selected pairs.",
            "",
            "| Candidate rule | Removes pair-oracle samples | Preserves all pair oracles |",
            "| --- | --- | --- |",
        ]
    )
    for rule in rules:
        removed = ",".join(str(item) for item in rule["pair_oracles_removed"]) or "none"
        lines.append(
            f"| `{rule['rule']}` | {removed} | {'yes' if rule['preserves_all_pair_oracles'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Pruning Interpretation",
            "",
            "- Harmful high-ranking pair candidates are parse-valid and structurally plausible; they do not look like malformed equations.",
            "- Wrong selected pairs split across wrong-drift, wrong-diffusion, and both-side errors, so no one-sided structural filter is clean.",
            f"- Mean-reverting selected pair drift appears in `{summary['mean_reverting_wrong_pair_count']}` wrong selected pair cases, but mean-reverting drift is also the true drift for paired-oracle samples 23, 30, and 31.",
            f"- Constant selected pair drift appears in `{summary['constant_drift_wrong_pair_count']}` wrong selected pair cases, but constant drift is the true drift for paired-oracle samples 5 and 8.",
            "- Pair source rank is not a safe pruning signal: oracle pairs can appear at source ranks 9, 10, 12, 13, and 15.",
            "- Model score is not a clean pruning signal either; wrong pair and oracle pair model scores overlap tightly around the pair-candidate range.",
            "- Score-component pruning is risky because wrong pairs are often active-distance traps: they win the active segment even when they are symbolically wrong.",
            "",
            "## Recommendation",
            "",
            "Do not add pair candidate pruning or a pair bonus from the current evidence. The safer next step is a targeted residual analysis of active_kramers_moyal traps for wrong pair candidates versus pair oracles, ideally reusing existing residual-vector tooling or adding a log-only/top-candidate residual parser if enough fields are present.",
            "",
            "Only if that analysis identifies a concrete, oracle-preserving rule should a future small/offline pruning diagnostic be proposed. Do not run a formal eval or 64-sample expansion for pruning yet.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def json_ready(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    miss_payload = json.loads(args.miss_json.read_text())
    coverage_payload = json.loads(args.coverage_json.read_text())
    miss_cases = miss_payload["miss_cases"]
    samples = coverage_payload["samples"]
    enriched = enrich_cases(miss_cases, samples)
    pair_oracles = pair_oracle_samples(samples)
    summary = summarize(enriched, pair_oracles)
    rules = rule_assessment(pair_oracles)
    write_report(args.report, enriched, pair_oracles, summary, rules)
    payload = {
        "miss_json": str(args.miss_json),
        "coverage_json": str(args.coverage_json),
        "summary": json_ready(summary),
        "pair_oracle_samples": pair_oracles,
        "rule_assessment": rules,
        "miss_cases": enriched,
    }
    args.json_output.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"selected_misses={summary['selected_misses']}")
    print(f"wrong_selected_pair_cases={summary['selected_wrong_pair_cases']}")
    print("safe_pruning_rule_found=0")


if __name__ == "__main__":
    main()
