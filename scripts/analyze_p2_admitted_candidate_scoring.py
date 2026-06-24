#!/usr/bin/env python3
"""P2 admitted-candidate scoring readiness diagnostic.

This helper is intentionally JSON-only. It checks whether the existing P2
coverage artifacts contain enough information to run the current semantic
fingerprint scorer on P2-admitted candidates. If not, it writes a cleanly
blocked report instead of reconstructing a non-equivalent target.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from normalized_drift_admission import canonicalize_constant_chains, token_text


TARGET_FINGERPRINT_ALIASES = {
    "target_y",
    "y_to_fit",
    "target_fingerprint",
    "fingerprint_target",
    "target_fingerprint_y",
}

SCORER_SCORE_ALIASES = {
    "fingerprint_distance",
    "semantic_score",
    "rerank_distance",
    "distance_details",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-coverage-json", type=Path, required=True)
    parser.add_argument("--expanded-coverage-json", type=Path, required=True)
    parser.add_argument("--baseline-coverage-json", type=Path, required=True)
    parser.add_argument("--drift-only-json", type=Path, required=True)
    parser.add_argument("--target-samples", type=str, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def parse_target_samples(raw: str) -> list[int]:
    samples = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            samples.append(int(part))
    return samples


def by_sample(samples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in samples}


def nested_key_exists(obj: Any, names: set[str]) -> bool:
    if isinstance(obj, dict):
        return any(key in names for key in obj) or any(
            nested_key_exists(value, names) for value in obj.values()
        )
    if isinstance(obj, list):
        return any(nested_key_exists(value, names) for value in obj)
    return False


def find_nested_keys(obj: Any, names: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names:
                found.add(key)
            found.update(find_nested_keys(value, names))
    elif isinstance(obj, list):
        for value in obj:
            found.update(find_nested_keys(value, names))
    return found


def first_rank(rows: list[dict[str, Any]], canonical_truth: tuple[str, ...]) -> int | None:
    ranks = [
        row.get("rank")
        for row in rows
        if tuple(row.get("canonical_drift_tokens", [])) == canonical_truth and row.get("rank") is not None
    ]
    return min(ranks) if ranks else None


def source_bucket(rows: list[dict[str, Any]], canonical_truth: tuple[str, ...]) -> str:
    sources = {
        row.get("source")
        for row in rows
        if tuple(row.get("canonical_drift_tokens", [])) == canonical_truth and row.get("source")
    }
    if not sources:
        return "neither"
    if sources == {"beam"}:
        return "beam"
    if sources == {"sampling"}:
        return "sampling"
    if "beam" in sources and "sampling" in sources:
        return "both"
    return "+".join(sorted(sources))


def format_optional(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def p2_oracle_row(
    idx: int,
    p2_sample: dict[str, Any],
    expanded_sample: dict[str, Any],
    admitted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_drift = p2_sample.get("truth_drift_tokens") or expanded_sample.get("truth_drift", [])
    truth_diffusion = p2_sample.get("truth_diffusion_tokens") or expanded_sample.get("truth_diffusion", [])
    canonical_truth = p2_sample.get("canonical_truth_drift_tokens")
    if canonical_truth is None:
        canonical_truth = canonicalize_constant_chains(truth_drift)
    canonical_truth_tuple = tuple(canonical_truth or [])
    p2_oracle_present = bool(
        p2_sample.get("p2_normalized_full_oracle")
        or (
            p2_sample.get("exact_diffusion_present")
            and any(tuple(row.get("canonical_drift_tokens", [])) == canonical_truth_tuple for row in admitted_rows)
        )
    )
    rank = first_rank(admitted_rows, canonical_truth_tuple)
    source = source_bucket(admitted_rows, canonical_truth_tuple)
    return {
        "sample_index": idx,
        "truth_drift_tokens": truth_drift,
        "truth_diffusion_tokens": truth_diffusion,
        "canonical_truth_drift_tokens": list(canonical_truth_tuple),
        "drift_family": p2_sample.get("drift_family") or expanded_sample.get("truth_drift_family"),
        "exact_diffusion_present": bool(p2_sample.get("exact_diffusion_present")),
        "current_expanded_full_oracle": bool(p2_sample.get("current_expanded_full_oracle")),
        "current_canonical_full_oracle": bool(p2_sample.get("current_canonical_full_oracle")),
        "p2_normalized_full_oracle": bool(p2_sample.get("p2_normalized_full_oracle")),
        "p2_oracle_present": p2_oracle_present,
        "p2_oracle_source": source,
        "p2_oracle_source_rank": rank,
        "admitted_candidate_count": len(admitted_rows),
        "current_candidate_sources": sorted((expanded_sample.get("full_by_source") or {}).keys()),
        "current_full_candidate_count": sum(
            len(rows) for rows in (expanded_sample.get("full_by_source") or {}).values()
        ),
        "semantic_scoring": {
            "best_current_expanded_candidate": None,
            "best_current_expanded_score": None,
            "best_p2_admitted_candidate": None,
            "best_p2_admitted_score": None,
            "p2_oracle_candidate": None if not p2_oracle_present else {
                "drift_tokens": list(canonical_truth_tuple),
                "diffusion_tokens": truth_diffusion,
                "source": source,
                "source_rank": rank,
            },
            "p2_oracle_score": None,
            "p2_oracle_rank": None,
            "p2_oracle_selected": None,
            "score_gap_selected_minus_p2_oracle": None,
        },
        "failure_reason": "insufficient_schema",
    }


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    schema = result["schema_readiness"]
    lines = [
        "# P2 Admitted Candidate Scoring Diagnostics",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Target samples analyzed | {summary['target_samples_analyzed']} |",
        f"| P2 oracle present in coverage JSON | {summary['p2_oracle_present_count']} |",
        f"| P2 oracle selected by current scorer | {format_optional(summary['p2_oracle_selected_count'])} |",
        f"| P2 oracle score misses | {format_optional(summary['p2_oracle_score_miss_count'])} |",
        f"| Parse failures | {format_optional(summary['parse_failures'])} |",
        f"| Fingerprint failures | {format_optional(summary['fingerprint_failures'])} |",
        f"| Median oracle rank | {format_optional(summary['median_oracle_rank'])} |",
        f"| Max oracle rank | {format_optional(summary['max_oracle_rank'])} |",
        "",
        "## Schema Readiness",
        "",
        "| Field group | Status |",
        "| --- | --- |",
        f"| Target fingerprint / `y_to_fit` | {schema['target_fingerprint_status']} |",
        f"| Current expanded candidates | {schema['current_candidates_status']} |",
        f"| P2 admitted candidates | {schema['p2_admitted_candidates_status']} |",
        f"| Paired diffusion candidates | {schema['paired_diffusion_status']} |",
        f"| Candidate source labels | {schema['source_labels_status']} |",
        f"| Canonicalized candidate tokens | {schema['canonical_tokens_status']} |",
        f"| Raw candidate tokens | {schema['raw_tokens_status']} |",
        f"| Existing semantic scores | {schema['semantic_scores_status']} |",
        "",
        "Semantic scoring is blocked because the coverage JSONs do not contain the target",
        "`y_to_fit` / fingerprint vector, nor enough raw eval-sample information with",
        "numeric constants to reconstruct the exact target fingerprint used by",
        "`constant_grid_rolewise_no_multi_u0`. The available `score` fields are",
        "candidate/model scores from coverage generation, not fingerprint distances.",
        "",
        "Missing fields needed for this diagnostic:",
        "",
    ]
    for field in schema["missing_critical_fields"]:
        lines.append(f"- {field}")
    lines.extend(
        [
            "",
            "## Per-Sample Readiness",
            "",
            "| Sample | Family | Truth drift | Truth diffusion | Exact diffusion | P2 oracle present | P2 source | P2 rank | Current candidates | Failure reason |",
            "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in result["samples"]:
        lines.append(
            "| {sample_index} | {family} | `{truth_drift}` | `{truth_diffusion}` | {diffusion} | {present} | {source} | {rank} | {current_count} | {reason} |".format(
                sample_index=row["sample_index"],
                family=row.get("drift_family") or "unknown",
                truth_drift=token_text(row.get("truth_drift_tokens")),
                truth_diffusion=token_text(row.get("truth_diffusion_tokens")),
                diffusion="yes" if row.get("exact_diffusion_present") else "no",
                present="yes" if row.get("p2_oracle_present") else "no",
                source=row.get("p2_oracle_source") or "-",
                rank=format_optional(row.get("p2_oracle_source_rank")),
                current_count=row.get("current_full_candidate_count"),
                reason=row.get("failure_reason"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The P2 admission artifacts are coverage-ready for the seven newly recovered",
            "samples, and all seven have exact diffusion present plus a P2 canonical",
            "oracle drift from beam. They are not scoring-ready from JSON alone.",
            "Running semantic rerank-readiness would require logging target fingerprints",
            "or exact normalized eval samples alongside candidate rows. No formal eval is",
            "recommended.",
            "",
            "## Decision",
            "",
            "**D. Logs are still insufficient.** Add minimal target-fingerprint /",
            "candidate semantic-score logging before any scoring or eval integration",
            "decision. Do not run a model-backed smoke unless explicitly requested.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    p2_coverage = load_json(args.p2_coverage_json)
    expanded = load_json(args.expanded_coverage_json)
    baseline = load_json(args.baseline_coverage_json)
    drift_only = load_json(args.drift_only_json)
    target_indices = parse_target_samples(args.target_samples)

    expanded_by_idx = by_sample(expanded.get("samples", []))
    p2 = p2_coverage.get("p2_normalized_admission", {})
    p2_by_idx = by_sample(p2.get("samples", []))
    admitted_by_sample = p2.get("admitted_candidates_by_sample", {})

    rows = []
    missing_sample_indices = []
    for idx in target_indices:
        p2_sample = p2_by_idx.get(idx)
        expanded_sample = expanded_by_idx.get(idx)
        if p2_sample is None or expanded_sample is None:
            missing_sample_indices.append(idx)
            continue
        admitted_rows = admitted_by_sample.get(str(idx), [])
        rows.append(p2_oracle_row(idx, p2_sample, expanded_sample, admitted_rows))

    target_found = nested_key_exists(p2_coverage, TARGET_FINGERPRINT_ALIASES) or nested_key_exists(
        expanded, TARGET_FINGERPRINT_ALIASES
    )
    semantic_score_keys = find_nested_keys(p2_coverage, SCORER_SCORE_ALIASES) | find_nested_keys(
        expanded, SCORER_SCORE_ALIASES
    )
    p2_oracle_present_count = sum(1 for row in rows if row["p2_oracle_present"])
    ranks = [
        row["p2_oracle_source_rank"]
        for row in rows
        if row.get("p2_oracle_source_rank") is not None
    ]
    family_breakdown = Counter(row.get("drift_family") or "unknown" for row in rows)
    source_breakdown = Counter(row.get("p2_oracle_source") or "neither" for row in rows)

    result = {
        "metadata": {
            "p2_coverage_json": str(args.p2_coverage_json),
            "expanded_coverage_json": str(args.expanded_coverage_json),
            "baseline_coverage_json": str(args.baseline_coverage_json),
            "drift_only_json": str(args.drift_only_json),
            "target_samples": target_indices,
            "score_kind_requested": "constant_grid_rolewise_no_multi_u0",
            "component_weights_requested": [1.0, 2.0, 1.0],
            "effective_active_weak_weights": [1.0, 1.0],
            "constant_values_requested": [0.25, 0.5, 1.0, 2.0, 4.0],
            "json_only": True,
        },
        "schema_readiness": {
            "enough_for_semantic_scoring": False,
            "target_fingerprint_status": "missing",
            "current_candidates_status": "available_as_template_tokens",
            "p2_admitted_candidates_status": "available",
            "paired_diffusion_status": "available_as_exact_diffusion_presence_and_tokens",
            "source_labels_status": "available",
            "canonical_tokens_status": "available_for_p2_admitted_candidates",
            "raw_tokens_status": "available_for_p2_admitted_drift_candidates",
            "semantic_scores_status": "missing"
            if not semantic_score_keys
            else f"non-scorer fields only: {', '.join(sorted(semantic_score_keys))}",
            "target_fingerprint_fields_found": sorted(find_nested_keys(p2_coverage, TARGET_FINGERPRINT_ALIASES)),
            "target_fingerprint_available": target_found,
            "missing_critical_fields": [
                "per-sample target `y_to_fit` or target fingerprint vector used by `score_candidate_system`",
                "or raw normalized eval sample plus numeric constants sufficient to reconstruct the exact target fingerprint",
                "semantic fingerprint distances for current expanded candidates, if scores are to be reused instead of recomputed",
                "full P2 paired candidate rows with scorer-ready full tokens, source, source_rank, and candidate_rank",
            ],
            "missing_sample_indices": missing_sample_indices,
        },
        "summary": {
            "target_samples_analyzed": len(rows),
            "p2_oracle_present_count": p2_oracle_present_count,
            "p2_oracle_selected_count": None,
            "p2_oracle_score_miss_count": None,
            "parse_failures": None,
            "fingerprint_failures": None,
            "median_oracle_rank": statistics.median(ranks) if ranks else None,
            "max_oracle_rank": max(ranks) if ranks else None,
            "score_gap_summary": None,
            "family_breakdown": dict(family_breakdown),
            "source_breakdown": dict(source_breakdown),
            "decision": "D",
            "recommendation": (
                "Add minimal target-fingerprint / scorer-distance logging before "
                "P2 admitted-candidate semantic scoring or eval integration."
            ),
            "formal_eval_recommended": False,
        },
        "inputs_headline": {
            "baseline_samples": len(baseline.get("samples", [])),
            "expanded_samples": len(expanded.get("samples", [])),
            "drift_only_samples": len(drift_only.get("samples", [])),
            "p2_current_expanded_full_oracle": p2.get("current_expanded_full_oracle"),
            "p2_canonicalized_expanded_pool_coverage": p2.get("canonicalized_expanded_pool_coverage"),
            "p2_normalized_admission_coverage": p2.get("p2_normalized_admission_coverage"),
            "p2_newly_recovered": p2.get("newly_recovered"),
            "p2_remaining_missing": p2.get("remaining_missing"),
        },
        "samples": rows,
    }

    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_markdown(result))
    print(
        "P2 admitted candidate scoring diagnostic blocked: "
        "missing target fingerprint / y_to_fit fields for semantic scoring."
    )


if __name__ == "__main__":
    main()
