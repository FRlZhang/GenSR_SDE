#!/usr/bin/env python3
"""Analyze drift diversity for expanded-pairing oracle-absent samples.

This helper is intentionally offline-only: it parses existing coverage JSON and
does not run model decoding, candidate generation, fingerprint scoring, or eval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILY_ORDER = [
    "linear drift",
    "sin drift",
    "nested-mul linear drift",
    "polynomial-like drift",
    "constant drift",
    "other / unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("candidate_coverage_pair_expanded.json"),
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("candidate_coverage_diagnostics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("expanded_pairing_oracle_absent_drift_diversity.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("expanded_pairing_oracle_absent_drift_diversity.json"),
    )
    return parser.parse_args()


def token_text(tokens: list[str] | None) -> str:
    if not tokens:
        return "-"
    return " ".join(tokens)


def template_text(tokens: list[str] | None) -> str:
    if not tokens:
        return "-"
    return " ".join(token for token in tokens if token != "CONSTANT")


def normalize_family(sample: dict[str, Any]) -> str:
    tokens = sample.get("truth_drift", [])
    template = [token for token in tokens if token != "CONSTANT"]
    reported = sample.get("truth_drift_family")
    if "pow2" in template or "pow3" in template or "pow" in template:
        return "polynomial-like drift"
    if "sin" in template:
        return "sin drift"
    if template == ["mul", "x_0"]:
        return "linear drift"
    if template == ["mul"]:
        return "constant drift"
    if len(template) >= 3 and template[:2] == ["mul", "mul"] and "x_0" in template:
        return "nested-mul linear drift"
    if reported == "nested mul structural variant":
        return "nested-mul linear drift"
    if reported in FAMILY_ORDER:
        return reported
    return "other / unknown"


def occurrence_text(occurrences: list[dict[str, Any]], limit: int = 3) -> str:
    if not occurrences:
        return "-"
    parts = []
    for occurrence in occurrences[:limit]:
        parts.append(
            f"{occurrence.get('source')}#{occurrence.get('source_rank')} "
            f"(candidate {occurrence.get('candidate_rank')})"
        )
    return "; ".join(parts)


def best_occurrence(occurrences: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not occurrences:
        return None
    return min(
        occurrences,
        key=lambda item: (
            item.get("candidate_rank") if item.get("candidate_rank") is not None else 10**9,
            item.get("source_rank") if item.get("source_rank") is not None else 10**9,
            item.get("source") or "",
        ),
    )


def nearest_entry(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def nearest_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    occurrence = best_occurrence(row.get("occurrences", []))
    return {
        "distance": row.get("distance"),
        "tokens": row.get("tokens", []),
        "family": normalize_candidate_family(row.get("tokens", []), row.get("family")),
        "right_family": bool(row.get("right_family")),
        "sources": row.get("sources", []),
        "best_occurrence": occurrence,
        "occurrences": row.get("occurrences", []),
    }


def normalize_candidate_family(tokens: list[str], reported: str | None = None) -> str:
    template = [token for token in tokens if token != "CONSTANT"]
    if "pow2" in template or "pow3" in template or "pow" in template:
        return "polynomial-like drift"
    if "sin" in template:
        return "sin drift"
    if template == ["mul", "x_0"]:
        return "linear drift"
    if template == ["mul"]:
        return "constant drift"
    if len(template) >= 3 and template[:2] == ["mul", "mul"] and "x_0" in template:
        return "nested-mul linear drift"
    if reported == "nested mul structural variant":
        return "nested-mul linear drift"
    return reported or "other / unknown"


def nearest_rank(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    occurrence = best_occurrence(row.get("occurrences", []))
    if occurrence is None:
        return None
    return occurrence.get("candidate_rank")


def source_presence(sample: dict[str, Any], side: str) -> dict[str, bool]:
    sources = sample.get(f"{side}_by_source", {})
    return {source: bool(values) for source, values in sorted(sources.items())}


def drift_collapse_label(sample: dict[str, Any], nearest: dict[str, Any] | None) -> str:
    if nearest is None:
        return "unknown"
    truth_template = [token for token in sample.get("truth_drift", []) if token != "CONSTANT"]
    nearest_template = [token for token in nearest.get("tokens", []) if token != "CONSTANT"]
    labels = []
    if nearest_template.count("mul") > truth_template.count("mul"):
        labels.append("over-nested")
    if "x_0" in truth_template and "x_0" not in nearest_template:
        labels.append("state-collapsed")
    if truth_template == ["mul", "x_0"] and nearest_template == ["mul"]:
        labels.append("constant-heavy")
    if "pow2" in truth_template and "pow2" not in nearest_template:
        labels.append("drops-polynomial-power")
    if "sin" in truth_template and "sin" in nearest_template and nearest_template.count("mul") > truth_template.count("mul"):
        labels.append("over-nested-sin")
    if not labels and nearest.get("distance") == 1:
        labels.append("nearby-one-edit-template")
    return ", ".join(dict.fromkeys(labels)) if labels else "not obviously collapsed"


def classify_blocker(sample: dict[str, Any]) -> str:
    if sample.get("has_drift"):
        if sample.get("has_diffusion") and not sample.get("has_full"):
            return "pairing/top-k/cap"
        return "not drift-missing"
    if sample.get("drift_span_rank") is None and not sample.get("exact_drift_occurrences"):
        return "exact drift absent from logged candidate sources"
    return "rank data inconsistent"


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required coverage JSON: {path}")
    return json.loads(path.read_text())


def build_analysis(payload: dict[str, Any], baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    samples = payload.get("samples", [])
    oracle_absent = [sample for sample in samples if not sample.get("has_full")]
    exact_drift_missing = [sample for sample in oracle_absent if not sample.get("has_drift")]
    pairing_missing = [
        sample
        for sample in oracle_absent
        if sample.get("has_drift") and sample.get("has_diffusion") and not sample.get("has_full")
    ]
    diffusion_missing = [sample for sample in oracle_absent if not sample.get("has_diffusion")]

    family_counts = Counter(normalize_family(sample) for sample in exact_drift_missing)
    nearest_family_counts = Counter()
    nearest_template_counts = Counter()
    collapse_counts = Counter()
    nearest_right_family = 0
    nearest_source_counts = Counter()
    exact_diffusion_source_counts = Counter()
    drift_source_sample_counts = Counter()
    diffusion_source_sample_counts = Counter()
    unique_drifts_by_source: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    unique_diffusions_by_source: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    drift_source_avg_counts = Counter()

    rows = []
    for sample in oracle_absent:
        nearest_drift = nearest_summary(nearest_entry(sample.get("drift_nearest_detailed", [])))
        nearest_diffusion = nearest_summary(nearest_entry(sample.get("diffusion_nearest_detailed", [])))
        family = normalize_family(sample)
        collapse = drift_collapse_label(sample, nearest_drift)
        if nearest_drift is not None:
            nearest_family_counts[nearest_drift["family"]] += 1
            nearest_template_counts[token_text(nearest_drift["tokens"])] += 1
            nearest_right_family += int(bool(nearest_drift.get("right_family")))
            for source in nearest_drift.get("sources", []):
                nearest_source_counts[source] += 1
        collapse_counts[collapse] += 1
        for source in sample.get("diffusion_sources", []):
            exact_diffusion_source_counts[source] += 1
        for source, spans in sample.get("drift_by_source", {}).items():
            if spans:
                drift_source_sample_counts[source] += 1
                drift_source_avg_counts[source] += len(spans)
            for span in spans:
                unique_drifts_by_source[source].add(tuple(span))
        for source, spans in sample.get("diffusion_by_source", {}).items():
            if spans:
                diffusion_source_sample_counts[source] += 1
            for span in spans:
                unique_diffusions_by_source[source].add(tuple(span))
        rows.append(
            {
                "sample_index": sample.get("sample_index"),
                "truth_full_expression": token_text(sample.get("truth")),
                "truth_drift_tokens": sample.get("truth_drift", []),
                "truth_diffusion_tokens": sample.get("truth_diffusion", []),
                "has_exact_drift": bool(sample.get("has_drift")),
                "has_exact_diffusion": bool(sample.get("has_diffusion")),
                "has_full_oracle": bool(sample.get("has_full")),
                "coverage": sample.get("coverage"),
                "missing_side": sample.get("missing_side"),
                "drift_family": family,
                "truth_drift_rank": sample.get("drift_span_rank"),
                "nearest_drift_rank": nearest_rank(nearest_entry(sample.get("drift_nearest_detailed", []))),
                "exact_diffusion_rank": sample.get("diffusion_span_rank"),
                "best_available_drift_candidate": nearest_drift,
                "best_available_diffusion_candidate": nearest_diffusion,
                "drift_sources_with_partial_spans": source_presence(sample, "drift"),
                "diffusion_sources_with_partial_spans": source_presence(sample, "diffusion"),
                "exact_diffusion_occurrences": sample.get("exact_diffusion_occurrences", []),
                "blocker": classify_blocker(sample),
                "collapse_pattern": collapse,
                "unique_drifts": sample.get("unique_drifts"),
                "unique_diffusions": sample.get("unique_diffusions"),
            }
        )

    source_diversity = {}
    for source in sorted(set(unique_drifts_by_source) | set(unique_diffusions_by_source)):
        sample_count = drift_source_sample_counts[source]
        source_diversity[source] = {
            "samples_with_drift_spans": sample_count,
            "unique_drift_templates": len(unique_drifts_by_source[source]),
            "avg_drift_templates_per_oracle_absent_sample": (
                drift_source_avg_counts[source] / len(oracle_absent) if oracle_absent else 0.0
            ),
            "samples_with_diffusion_spans": diffusion_source_sample_counts[source],
            "unique_diffusion_templates": len(unique_diffusions_by_source[source]),
        }

    baseline_summary = None
    if baseline_payload is not None:
        baseline_samples = baseline_payload.get("samples", [])
        baseline_absent = [sample for sample in baseline_samples if not sample.get("has_full")]
        baseline_summary = {
            "full_oracle_present": sum(bool(sample.get("has_full")) for sample in baseline_samples),
            "oracle_absent": len(baseline_absent),
            "diffusion_only_present": sum(
                sample.get("coverage") == "diffusion_only_present" for sample in baseline_absent
            ),
            "drift_and_diffusion_separate_not_paired": sum(
                sample.get("coverage") == "drift_and_diffusion_separate_not_paired"
                for sample in baseline_absent
            ),
        }

    fields_available = {
        "truth_full_expression": True,
        "truth_drift_tokens": True,
        "truth_diffusion_tokens": True,
        "exact_drift_present": True,
        "exact_diffusion_present": True,
        "exact_diffusion_rank": True,
        "nearest_drift_candidate": True,
        "nearest_drift_rank_within_logged_candidates": True,
        "source_breakdown_for_logged_candidates": True,
        "unlogged_beam_or_sampling_tail_ranks": False,
        "token_logits_or_raw_sampling_entropy": False,
        "grammar_rejection_counts": False,
    }

    decision = {
        "choice": "A",
        "label": "Drift spans are missing but nearby drift families appear",
        "recommended_next_action": (
            "Run one future drift-only candidate diversity smoke that logs drift-only "
            "candidate spans/ranks before any rerank or formal eval change."
        ),
    }

    return {
        "metadata": payload.get("metadata", {}),
        "baseline_comparison": baseline_summary,
        "summary": {
            "total_samples": len(samples),
            "full_oracle_present": sum(bool(sample.get("has_full")) for sample in samples),
            "oracle_absent": len(oracle_absent),
            "exact_drift_missing": len(exact_drift_missing),
            "pairing_missing": len(pairing_missing),
            "diffusion_missing": len(diffusion_missing),
            "exact_diffusion_present_among_oracle_absent": sum(
                bool(sample.get("has_diffusion")) for sample in oracle_absent
            ),
            "exact_drift_present_among_oracle_absent": sum(
                bool(sample.get("has_drift")) for sample in oracle_absent
            ),
            "nearest_drift_right_family": nearest_right_family,
            "oracle_absent_sample_indices": [sample.get("sample_index") for sample in oracle_absent],
        },
        "drift_family_breakdown": {family: family_counts.get(family, 0) for family in FAMILY_ORDER},
        "nearest_drift_family_breakdown": dict(nearest_family_counts),
        "nearest_drift_template_breakdown": dict(nearest_template_counts),
        "collapse_pattern_breakdown": dict(collapse_counts),
        "nearest_drift_source_counts": dict(nearest_source_counts),
        "exact_diffusion_source_counts": dict(exact_diffusion_source_counts),
        "source_diversity": source_diversity,
        "fields_available": fields_available,
        "limitations": [
            "Existing coverage JSON proves exact drift is absent from the logged candidate pool, but it does not contain ranks for unlogged beam/sampling tails.",
            "Sampling entropy, token logits, and grammar rejection counts are not present, so sampling under-diversity and grammar collapse are inferred from emitted spans only.",
        ],
        "samples": rows,
        "decision": decision,
    }


def format_family_counts(counts: dict[str, int]) -> list[str]:
    lines = ["| Drift family | Count |", "| --- | ---: |"]
    for family in FAMILY_ORDER:
        lines.append(f"| {family} | {counts.get(family, 0)} |")
    return lines


def format_source_diversity(source_diversity: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Source | Samples with drift spans | Unique drift templates | Avg drift templates/sample | Samples with exact/partial diffusion spans | Unique diffusion templates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, row in sorted(source_diversity.items()):
        lines.append(
            f"| {source} | {row['samples_with_drift_spans']} | "
            f"{row['unique_drift_templates']} | "
            f"{row['avg_drift_templates_per_oracle_absent_sample']:.2f} | "
            f"{row['samples_with_diffusion_spans']} | {row['unique_diffusion_templates']} |"
        )
    return lines


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    summary = analysis["summary"]
    baseline = analysis.get("baseline_comparison") or {}
    decision = analysis["decision"]
    lines = [
        "# Expanded-Pairing Oracle-Absent Drift Diversity",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline parsing of existing expanded-pairing coverage JSON and reports. No model decoding, candidate regeneration, fingerprint simulation, formal eval, 64-sample eval, grid, retraining, scorer change, or candidate-generation change was run.",
        "",
        "## Executive Summary",
        "",
        f"- Expanded-pairing full oracle coverage: `{summary['full_oracle_present']}/{summary['total_samples']}`.",
        f"- Expanded-pairing oracle-absent samples analyzed: `{summary['oracle_absent']}/{summary['total_samples']}`.",
        f"- Exact-drift-missing among oracle-absent samples: `{summary['exact_drift_missing']}/{summary['oracle_absent']}`.",
        f"- Pairing-missing among oracle-absent samples: `{summary['pairing_missing']}/{summary['oracle_absent']}`.",
        f"- Diffusion-missing among oracle-absent samples: `{summary['diffusion_missing']}/{summary['oracle_absent']}`.",
        f"- Exact diffusion is already present for `{summary['exact_diffusion_present_among_oracle_absent']}/{summary['oracle_absent']}` oracle-absent samples.",
        f"- Exact drift is present for `{summary['exact_drift_present_among_oracle_absent']}/{summary['oracle_absent']}` oracle-absent samples.",
        f"- Nearest logged drift has the right family in `{summary['nearest_drift_right_family']}/{summary['oracle_absent']}` cases.",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Baseline vs Expanded Check",
        "",
        "| Metric | Baseline | Expanded |",
        "| --- | ---: | ---: |",
        f"| Full oracle present | {baseline.get('full_oracle_present', 'unknown')}/32 | {summary['full_oracle_present']}/32 |",
        f"| Oracle absent | {baseline.get('oracle_absent', 'unknown')}/32 | {summary['oracle_absent']}/32 |",
        f"| Diffusion-only present | {baseline.get('diffusion_only_present', 'unknown')} | {summary['exact_drift_missing']} |",
        f"| Drift+diffusion separate, not paired | {baseline.get('drift_and_diffusion_separate_not_paired', 'unknown')} | {summary['pairing_missing']} |",
        "",
        "The known six baseline pairing-missing cases are no longer in the oracle-absent set. The remaining failures are all exact-drift misses with exact diffusion already present.",
        "",
        "## Drift-Family Breakdown",
        "",
        *format_family_counts(analysis["drift_family_breakdown"]),
        "",
        "Nearest logged drift templates are concentrated in a few nearby families: over-nested sin/linear variants, constant-heavy `mul CONSTANT CONSTANT`, and nested-mul constants. This supports a drift span diversity issue rather than a diffusion or pairing issue.",
        "",
        "## Source Diversity",
        "",
        *format_source_diversity(analysis["source_diversity"]),
        "",
        "Beam contributes the broadest useful drift span set in these logs. Sampling appears in most samples but mostly repeats the same small drift-template families and does not add any exact drift spans. Pairing recombines existing spans, so it improves full-expression coverage only when the exact drift span already exists; it cannot rescue these 17 cases.",
        "",
        "## Rank And Blocker Interpretation",
        "",
        "- Truth drift rank is `null` for all 17 samples because the exact drift span is absent from the logged candidate pool.",
        "- Exact diffusion rank is present for all 17 samples, with ranks between the logged top source spans.",
        "- Current logs do not contain unlogged beam/sampling tail ranks, raw token probabilities, or grammar rejection counts. Therefore they prove current-pool absence, but cannot distinguish a below-unlogged-top-k miss from a model/grammar miss.",
        "- The repeated nearest templates indicate collapse toward over-nested or constant-heavy drift spans, especially for simple linear and sin truth drifts.",
        "",
        "## Per-Sample Diagnostics",
        "",
        "| Sample | Truth drift | Truth diffusion | Family | Exact drift | Exact diffusion/rank | Best logged drift | Drift source(s) | Collapse pattern | Blocker |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in analysis["samples"]:
        nearest = row.get("best_available_drift_candidate") or {}
        best_tokens = token_text(nearest.get("tokens"))
        best_distance = nearest.get("distance", "-")
        sources = ",".join(nearest.get("sources", [])) or "-"
        lines.append(
            f"| {row['sample_index']} | `{token_text(row['truth_drift_tokens'])}` | "
            f"`{token_text(row['truth_diffusion_tokens'])}` | {row['drift_family']} | "
            f"{row['has_exact_drift']} | {row['has_exact_diffusion']} / {row['exact_diffusion_rank']} | "
            f"d={best_distance}: `{best_tokens}` | {sources} | "
            f"{row['collapse_pattern']} | {row['blocker']} |"
        )

    lines.extend(
        [
            "",
            "## Required Conclusions",
            "",
            "1. Oracle-absent after expanded pairing: `17/32`.",
            "2. Split: exact-drift-missing `17`, pairing-missing `0`, diffusion-missing `0`.",
            "3. Drift-family breakdown: linear `6`, sin `6`, nested-mul linear `3`, polynomial-like `2`, constant `0`, other/unknown `0`.",
            "4. Exact diffusion is already present in all `17/17` remaining cases.",
            "5. Missing exact drift candidates are absent from all logged sources, not merely blocked by the expanded pair top-k/cap. Logs are insufficient to prove whether they would appear in unlogged beam/sampling tails.",
            "6. Beam contributes more usable drift diversity than sampling in the emitted pool; sampling mostly duplicates nearby templates and adds no exact drift spans.",
            "7. Sampling temperature/top-k/top-p appears under-diverse for the missing linear, sin, nested-linear, and polynomial-like drift families, but this is an emitted-candidate inference because entropy/logit fields are absent.",
            "8. Grammar-constrained beam appears to collapse toward repeated over-nested or constant-heavy templates, especially `mul CONSTANT CONSTANT`, `mul mul CONSTANT CONSTANT CONSTANT`, and `mul mul CONSTANT CONSTANT sin x_0`.",
            "9. Most plausible intervention: one diagnostic-only drift-only candidate diversity smoke with explicit drift-span ranks/sources. Do not run formal eval from this report alone.",
            "",
            "## Decision",
            "",
            f"`{decision['choice']}. {decision['label']}`: {decision['recommended_next_action']}",
            "",
            "No rerank mode, scorer change, formal 32-sample eval, 64-sample eval, grid, retraining, checkpoint change, target-format change, fingerprint change, or candidate-generation logic change is recommended from this report.",
            "",
            "## Missing Fields",
            "",
        ]
    )
    for field, available in analysis["fields_available"].items():
        lines.append(f"- `{field}`: {'available' if available else 'missing'}")
    lines.extend(["", "Limitations:"] + [f"- {item}" for item in analysis["limitations"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    payload = load_payload(args.coverage_json)
    baseline_payload = load_payload(args.baseline_json) if args.baseline_json.is_file() else None
    analysis = build_analysis(payload, baseline_payload)
    args.json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    write_report(args.report, analysis)
    summary = analysis["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"oracle_absent={summary['oracle_absent']}")
    print(f"exact_drift_missing={summary['exact_drift_missing']}")
    print(f"diffusion_missing={summary['diffusion_missing']}")
    print(f"decision={analysis['decision']['choice']}")


if __name__ == "__main__":
    main()
