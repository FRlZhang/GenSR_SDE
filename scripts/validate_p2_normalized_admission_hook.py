#!/usr/bin/env python3
"""Cross-validate the reusable P2 normalized drift admission hook.

This helper parses existing JSON artifacts only. It imports the reusable
coverage-only hook from ``normalized_drift_admission.py`` and checks that the
hook output agrees with prior helper-only diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from normalized_drift_admission import (
    build_p2_normalized_admission_pool,
    canonicalize_constant_chains,
    safety_checks,
)


EXPECTED = {
    "current_expanded_full_oracle": 15,
    "canonicalized_expanded_pool_coverage": 23,
    "full_normalized_admission_coverage": 30,
    "p2_normalized_admission_coverage": 30,
    "p2_pair_count": 340,
    "full_normalized_admission_pair_count": 1105,
    "truth_aware_upper_bound_pair_count": 320,
    "newly_recovered": [0, 2, 6, 7, 15, 25, 29],
    "remaining_missing": [16, 28],
    "p2_source_breakdown": {"beam": 15},
    "sampling_recovered_canonical_drift_count": 0,
    "sampling_can_be_dropped": True,
    "unsafe_collision_flag": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-json", type=Path, default=Path("candidate_coverage_p2_normalized_admission.json"))
    parser.add_argument("--capped-json", type=Path, default=Path("capped_normalized_drift_admission_diagnostics.json"))
    parser.add_argument(
        "--normalized-admission-json",
        type=Path,
        default=Path("normalized_drift_only_admission_coverage_diagnostics.json"),
    )
    parser.add_argument(
        "--normalization-json",
        type=Path,
        default=Path("candidate_normalization_constant_folding_diagnostics.json"),
    )
    parser.add_argument("--drift-only-json", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument("--expanded-coverage-json", type=Path, default=Path("candidate_coverage_pair_expanded.json"))
    parser.add_argument("--report", type=Path, default=Path("p2_normalized_admission_hook_validation.md"))
    parser.add_argument("--json-output", type=Path, default=Path("p2_normalized_admission_hook_validation.json"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def at(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def sorted_list(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value)
    return value


def source_breakdown_from_pool(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for sample in samples:
        if not sample["p2_normalized_full_oracle"]:
            continue
        truth_key = tuple(sample["canonical_truth_drift_tokens"] or [])
        sources = {
            row.get("source")
            for row in sample["admitted_candidates"]
            if tuple(row.get("canonical_drift_tokens", [])) == truth_key and row.get("source")
        }
        if sources:
            counts["+".join(sorted(sources))] += 1
    return dict(counts)


def recompute_p2(expanded: dict[str, Any], drift_only: dict[str, Any]) -> dict[str, Any]:
    pool = build_p2_normalized_admission_pool(expanded, drift_only, admission_source="beam", include_sampling=False)
    samples = pool["samples"]
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    current_canonical_hits = [sample for sample in samples if sample["current_canonical_full_oracle"]]
    p2_hits = [sample for sample in samples if sample["p2_normalized_full_oracle"]]
    newly = [sample["sample_index"] for sample in p2_hits if not sample["current_canonical_full_oracle"]]
    remaining = [sample["sample_index"] for sample in samples if not sample["p2_normalized_full_oracle"]]
    family_counts = Counter(sample["drift_family"] for sample in p2_hits)
    return {
        "current_expanded_full_oracle": current_full,
        "canonicalized_expanded_pool_coverage": current_full + len(current_canonical_hits),
        "p2_normalized_admission_coverage": current_full + len(p2_hits),
        "p2_pair_count": sum(sample["pair_count"] for sample in samples),
        "newly_recovered": newly,
        "remaining_missing": remaining,
        "p2_source_breakdown": source_breakdown_from_pool(samples),
        "p2_family_breakdown": {
            "linear drift": family_counts.get("linear drift", 0),
            "sin drift": family_counts.get("sin drift", 0),
            "nested-mul linear drift": family_counts.get("nested-mul linear drift", 0),
            "polynomial-like drift": family_counts.get("polynomial-like drift", 0),
            "other": family_counts.get("other", 0),
        },
        "samples": samples,
    }


def add_metric(metrics: dict[str, Any], name: str, expected: Any, sources: list[tuple[str, str, Any]]) -> None:
    normalized_expected = sorted_list(expected)
    entries = []
    mismatches = []
    for report, field, value in sources:
        normalized_value = sorted_list(value)
        ok = normalized_value == normalized_expected
        entries.append({"report": report, "field": field, "value": value, "ok": ok})
        if not ok:
            mismatches.append({"report": report, "field": field, "value": value, "expected": expected})
    metrics[name] = {"expected": expected, "sources": entries, "ok": not mismatches, "mismatches": mismatches}


def schema_report(name: str, obj: dict[str, Any], checks: list[tuple[str, str, str | None]]) -> dict[str, Any]:
    missing = []
    aliases = []
    for label, path, derived in checks:
        if path and at(obj, path, None) is not None:
            aliases.append({"field": label, "path": path, "derived": False})
        elif derived:
            aliases.append({"field": label, "path": derived, "derived": True})
        else:
            missing.append(label)
    status = "ok" if not missing else "partial" if aliases else "missing"
    return {"report": name, "schema_status": status, "missing_fields": missing, "field_aliases_used": aliases}


def validate_schema(
    p2: dict[str, Any],
    capped: dict[str, Any],
    normalized: dict[str, Any],
    normalization: dict[str, Any],
    drift_only: dict[str, Any],
    expanded: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        schema_report(
            "candidate_coverage_p2_normalized_admission.json",
            p2,
            [
                ("total sample count", "summary.total_samples_analyzed", None),
                ("target oracle-absent sample indices", "samples", "samples[].sample_index"),
                ("current expanded oracle coverage", "summary.current_expanded_full_oracle", None),
                ("canonicalized expanded coverage", "summary.canonicalized_expanded_pool_coverage", None),
                ("full normalized admission coverage", "summary.full_normalized_admission_coverage", None),
                ("P2 coverage", "summary.p2_normalized_admission_coverage", None),
                ("P2 admitted candidates by sample", "admitted_candidates_by_sample", None),
                ("P2 family/source/rank metadata", "admitted_candidates_by_sample", "admitted rows include family/source/rank"),
                ("exact diffusion presence", "samples", "samples[].exact_diffusion_present"),
                ("pair-count estimates", "summary.p2_pair_count", None),
                ("safety self-checks", "safety.self_checks", None),
            ],
        ),
        schema_report(
            "capped_normalized_drift_admission_diagnostics.json",
            capped,
            [
                ("total sample count", "summary.total_samples_analyzed", None),
                ("target oracle-absent sample indices", "samples", "samples[].sample_index"),
                ("current expanded oracle coverage", "metadata.baseline.current_expanded_full_oracle_coverage", None),
                ("canonicalized expanded coverage", "metadata.baseline.canonicalized_expanded_pool_coverage", None),
                ("full normalized admission coverage", "metadata.baseline.full_normalized_drift_only_admission_coverage", None),
                ("P2 coverage", "policies.P2_one_per_family.projected_full_oracle_coverage", None),
                ("P2 admitted candidates by sample", "", None),
                ("P2 family/source/rank metadata", "policies.P2_one_per_family.source_breakdown", "policy summary only"),
                ("exact diffusion presence", "samples", "samples[].exact_diffusion_present"),
                ("pair-count estimates", "policies.P2_one_per_family.pair_count_after", None),
                ("safety self-checks", "collision_safety.unsafe_checks", None),
            ],
        ),
        schema_report(
            "normalized_drift_only_admission_coverage_diagnostics.json",
            normalized,
            [
                ("total sample count", "summary.total_samples_analyzed", None),
                ("target oracle-absent sample indices", "samples", "samples[].sample_index"),
                ("current expanded oracle coverage", "summary.current_expanded_full_oracle_coverage", None),
                ("canonicalized expanded coverage", "summary.canonicalized_expanded_pool_coverage", None),
                ("full normalized admission coverage", "summary.normalized_drift_only_admission_coverage", None),
                ("P2 coverage", "", None),
                ("P2 admitted candidates by sample", "", None),
                ("P2 family/source/rank metadata", "", None),
                ("exact diffusion presence", "samples", "samples[].exact_diffusion_present"),
                ("pair-count estimates", "pair_pressure.pair_count_after_canonical_sum", None),
                ("safety self-checks", "collision_safety.unsafe_checks", None),
            ],
        ),
        schema_report(
            "candidate_normalization_constant_folding_diagnostics.json",
            normalization,
            [
                ("total sample count", "summary.total_samples_analyzed", None),
                ("target oracle-absent sample indices", "metadata.target_indices", None),
                ("current expanded oracle coverage", "summary.current_expanded_full_oracle_coverage", None),
                ("canonicalized expanded coverage", "summary.canonicalized_expanded_pool_full_oracle_coverage", None),
                ("full normalized admission coverage", "summary.canonicalized_drift_only_admission_full_oracle_coverage", None),
                ("P2 coverage", "", None),
                ("P2 admitted candidates by sample", "", None),
                ("P2 family/source/rank metadata", "", None),
                ("exact diffusion presence", "samples", "samples[].exact_diffusion_present"),
                ("pair-count estimates", "", None),
                ("safety self-checks", "collision_risk.unsafe_checks", None),
            ],
        ),
        schema_report(
            "drift_only_candidate_diversity_smoke.json",
            drift_only,
            [
                ("total sample count", "", None),
                ("target oracle-absent sample indices", "metadata.target_indices", None),
                ("current expanded oracle coverage", "", None),
                ("canonicalized expanded coverage", "", None),
                ("full normalized admission coverage", "", None),
                ("P2 coverage", "", None),
                ("P2 admitted candidates by sample", "", None),
                ("P2 family/source/rank metadata", "samples", "beam/sampling candidate source_rank fields"),
                ("exact diffusion presence", "samples", "samples[].exact_diffusion_present_previous_coverage"),
                ("pair-count estimates", "", None),
                ("safety self-checks", "", None),
            ],
        ),
        schema_report(
            "candidate_coverage_pair_expanded.json",
            expanded,
            [
                ("total sample count", "samples", "len(samples)"),
                ("target oracle-absent sample indices", "", "samples where has_full is false"),
                ("current expanded oracle coverage", "", "sum(samples[].has_full)"),
                ("canonicalized expanded coverage", "", None),
                ("full normalized admission coverage", "", None),
                ("P2 coverage", "", None),
                ("P2 admitted candidates by sample", "", None),
                ("P2 family/source/rank metadata", "", None),
                ("exact diffusion presence", "samples", "samples[].has_diffusion"),
                ("pair-count estimates", "samples", "unique_drifts * unique_diffusions"),
                ("safety self-checks", "", None),
            ],
        ),
    ]


def canonicalizer_checks() -> dict[str, bool]:
    return {
        "pow2_x0_not_x0": canonicalize_constant_chains("pow2 x_0") != canonicalize_constant_chains("x_0"),
        "const_pow2_not_const_x0": canonicalize_constant_chains("mul CONSTANT pow2 x_0")
        != canonicalize_constant_chains("mul CONSTANT x_0"),
        "sin_x0_not_x0": canonicalize_constant_chains("sin x_0") != canonicalize_constant_chains("x_0"),
        "const_chain_linear_matches": canonicalize_constant_chains("mul mul CONSTANT CONSTANT x_0")
        == canonicalize_constant_chains("mul CONSTANT x_0"),
        "const_chain_sin_matches": canonicalize_constant_chains("mul mul CONSTANT CONSTANT sin x_0")
        == canonicalize_constant_chains("mul CONSTANT sin x_0"),
    }


def analyze(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p2 = inputs["p2"]
    capped = inputs["capped"]
    normalized = inputs["normalized_admission"]
    normalization = inputs["normalization"]
    drift_only = inputs["drift_only"]
    expanded = inputs["expanded"]
    recomputed = recompute_p2(expanded, drift_only)

    metrics: dict[str, Any] = {}
    add_metric(
        metrics,
        "current_expanded_full_oracle",
        EXPECTED["current_expanded_full_oracle"],
        [
            ("p2", "summary.current_expanded_full_oracle", at(p2, "summary.current_expanded_full_oracle")),
            ("capped", "metadata.baseline.current_expanded_full_oracle_coverage", at(capped, "metadata.baseline.current_expanded_full_oracle_coverage")),
            ("normalized_admission", "summary.current_expanded_full_oracle_coverage", at(normalized, "summary.current_expanded_full_oracle_coverage")),
            ("normalization", "summary.current_expanded_full_oracle_coverage", at(normalization, "summary.current_expanded_full_oracle_coverage")),
            ("expanded_coverage", "derived sum(samples[].has_full)", sum(1 for sample in expanded["samples"] if sample.get("has_full"))),
            ("recomputed_hook", "derived", recomputed["current_expanded_full_oracle"]),
        ],
    )
    add_metric(
        metrics,
        "canonicalized_expanded_pool_coverage",
        EXPECTED["canonicalized_expanded_pool_coverage"],
        [
            ("p2", "summary.canonicalized_expanded_pool_coverage", at(p2, "summary.canonicalized_expanded_pool_coverage")),
            ("capped", "metadata.baseline.canonicalized_expanded_pool_coverage", at(capped, "metadata.baseline.canonicalized_expanded_pool_coverage")),
            ("normalized_admission", "summary.canonicalized_expanded_pool_coverage", at(normalized, "summary.canonicalized_expanded_pool_coverage")),
            ("normalization", "summary.canonicalized_expanded_pool_full_oracle_coverage", at(normalization, "summary.canonicalized_expanded_pool_full_oracle_coverage")),
            ("recomputed_hook", "derived", recomputed["canonicalized_expanded_pool_coverage"]),
        ],
    )
    add_metric(
        metrics,
        "full_normalized_admission_coverage",
        EXPECTED["full_normalized_admission_coverage"],
        [
            ("p2", "summary.full_normalized_admission_coverage", at(p2, "summary.full_normalized_admission_coverage")),
            ("capped", "metadata.baseline.full_normalized_drift_only_admission_coverage", at(capped, "metadata.baseline.full_normalized_drift_only_admission_coverage")),
            ("normalized_admission", "summary.normalized_drift_only_admission_coverage", at(normalized, "summary.normalized_drift_only_admission_coverage")),
            ("normalization", "summary.canonicalized_drift_only_admission_full_oracle_coverage", at(normalization, "summary.canonicalized_drift_only_admission_full_oracle_coverage")),
        ],
    )
    add_metric(
        metrics,
        "p2_normalized_admission_coverage",
        EXPECTED["p2_normalized_admission_coverage"],
        [
            ("p2", "summary.p2_normalized_admission_coverage", at(p2, "summary.p2_normalized_admission_coverage")),
            ("capped", "policies.P2_one_per_family.projected_full_oracle_coverage", at(capped, "policies.P2_one_per_family.projected_full_oracle_coverage")),
            ("recomputed_hook", "derived", recomputed["p2_normalized_admission_coverage"]),
        ],
    )
    add_metric(
        metrics,
        "p2_pair_count",
        EXPECTED["p2_pair_count"],
        [
            ("p2", "summary.p2_pair_count", at(p2, "summary.p2_pair_count")),
            ("capped", "policies.P2_one_per_family.pair_count_after", at(capped, "policies.P2_one_per_family.pair_count_after")),
            ("recomputed_hook", "derived", recomputed["p2_pair_count"]),
        ],
    )
    add_metric(
        metrics,
        "full_normalized_admission_pair_count",
        EXPECTED["full_normalized_admission_pair_count"],
        [
            ("p2", "summary.full_normalized_admission_pair_count", at(p2, "summary.full_normalized_admission_pair_count")),
            ("capped", "metadata.baseline.full_normalized_admission_target_pair_count", at(capped, "metadata.baseline.full_normalized_admission_target_pair_count")),
            ("normalized_admission", "pair_pressure.pair_count_after_canonical_sum", at(normalized, "pair_pressure.pair_count_after_canonical_sum")),
        ],
    )
    add_metric(
        metrics,
        "truth_aware_upper_bound_pair_count",
        EXPECTED["truth_aware_upper_bound_pair_count"],
        [
            ("p2", "summary.truth_aware_upper_bound_pair_count", at(p2, "summary.truth_aware_upper_bound_pair_count")),
            ("capped", "truth_aware_upper_bound.pair_count_after", at(capped, "truth_aware_upper_bound.pair_count_after")),
            ("capped", "policies.P4_recovery_targeted_oracle_upper_bound.pair_count_after", at(capped, "policies.P4_recovery_targeted_oracle_upper_bound.pair_count_after")),
        ],
    )
    add_metric(
        metrics,
        "newly_recovered",
        EXPECTED["newly_recovered"],
        [
            ("p2", "summary.newly_recovered_beyond_current", at(p2, "summary.newly_recovered_beyond_current")),
            ("capped", "policies.P2_one_per_family.newly_recovered_beyond_current_expanded_canonical_pool", at(capped, "policies.P2_one_per_family.newly_recovered_beyond_current_expanded_canonical_pool")),
            ("normalized_admission", "summary.newly_recovered_beyond_current_expanded_canonical_pool", at(normalized, "summary.newly_recovered_beyond_current_expanded_canonical_pool")),
            ("recomputed_hook", "derived", recomputed["newly_recovered"]),
        ],
    )
    add_metric(
        metrics,
        "remaining_missing",
        EXPECTED["remaining_missing"],
        [
            ("p2", "summary.remaining_missing_samples", at(p2, "summary.remaining_missing_samples")),
            ("capped", "policies.P2_one_per_family.remaining_missing_samples", at(capped, "policies.P2_one_per_family.remaining_missing_samples")),
            ("normalized_admission", "summary.remaining_missing_samples", at(normalized, "summary.remaining_missing_samples")),
            ("recomputed_hook", "derived", recomputed["remaining_missing"]),
        ],
    )
    add_metric(
        metrics,
        "p2_source_breakdown",
        EXPECTED["p2_source_breakdown"],
        [
            ("p2", "source_breakdown", p2.get("source_breakdown")),
            ("capped", "policies.P2_one_per_family.source_breakdown", at(capped, "policies.P2_one_per_family.source_breakdown")),
            ("normalized_admission", "source_breakdown.drift_only_tail_normalized", at(normalized, "source_breakdown.drift_only_tail_normalized")),
            ("recomputed_hook", "derived", recomputed["p2_source_breakdown"]),
        ],
    )
    add_metric(
        metrics,
        "sampling_recovered_canonical_drift_count",
        EXPECTED["sampling_recovered_canonical_drift_count"],
        [
            ("p2", "summary.sampling_recovered_canonical_drift_count", at(p2, "summary.sampling_recovered_canonical_drift_count")),
        ],
    )
    add_metric(
        metrics,
        "sampling_can_be_dropped",
        EXPECTED["sampling_can_be_dropped"],
        [
            ("p2", "summary.sampling_excluded", at(p2, "summary.sampling_excluded")),
            ("capped", "summary.sampling_can_be_dropped_for_all_policies", at(capped, "summary.sampling_can_be_dropped_for_all_policies")),
            ("capped", "policies.P2_one_per_family.sampling_can_be_dropped", at(capped, "policies.P2_one_per_family.sampling_can_be_dropped")),
            ("normalized_admission", "not summary.sampling_contributes_recovered_canonical_drift", not at(normalized, "summary.sampling_contributes_recovered_canonical_drift")),
        ],
    )
    add_metric(
        metrics,
        "unsafe_collision_flag",
        EXPECTED["unsafe_collision_flag"],
        [
            ("p2", "safety.unsafe_collision_flag", at(p2, "safety.unsafe_collision_flag")),
            ("capped", "collision_safety.unsafe_collision_flag", at(capped, "collision_safety.unsafe_collision_flag")),
            ("normalized_admission", "collision_safety.unsafe_collision_flag", at(normalized, "collision_safety.unsafe_collision_flag")),
            ("normalization", "collision_risk.unsafe", at(normalization, "collision_risk.unsafe")),
        ],
    )

    schema = validate_schema(p2, capped, normalized, normalization, drift_only, expanded)
    checks = canonicalizer_checks()
    mismatches = [mismatch for metric in metrics.values() for mismatch in metric["mismatches"]]
    schema_status = "ok" if all(row["schema_status"] == "ok" for row in schema[:1]) and not any(row["schema_status"] == "missing" for row in schema) else "partial"
    if any(row["schema_status"] == "missing" for row in schema):
        schema_status = "missing"
    if mismatches:
        decision = {
            "choice": "C",
            "label": "Hook disagrees with prior diagnostics",
            "recommended_next_action": "Resolve the exact metric mismatches before integration. No formal eval.",
        }
    elif schema_status == "ok":
        decision = {
            "choice": "A",
            "label": "Hook is internally consistent",
            "recommended_next_action": "Add a future coverage-only candidate-coverage pipeline flag. No formal eval.",
        }
    elif schema_status == "partial":
        decision = {
            "choice": "B",
            "label": "Hook works but schema is brittle",
            "recommended_next_action": "Clean up minimal JSON/schema aliases before integration. No formal eval.",
        }
    else:
        decision = {
            "choice": "D",
            "label": "Insufficient fields",
            "recommended_next_action": "Add minimal JSON logging/report fields before model-backed smoke. No formal eval.",
        }
    return {
        "summary": {
            "json_reports_validated": 6,
            "metric_cross_check_status": "ok" if not mismatches else "mismatch",
            "schema_status": schema_status,
            "canonicalizer_safety_status": "ok" if all(checks.values()) else "failed",
            "mismatch_count": len(mismatches),
        },
        "metrics": metrics,
        "schema": schema,
        "canonicalizer_checks": checks,
        "recomputed_p2": {
            key: value for key, value in recomputed.items() if key != "samples"
        },
        "mismatches": mismatches,
        "decision": decision,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# P2 Normalized Admission Hook Validation",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline JSON-only cross-validation of the reusable P2 normalized drift admission hook against prior helper diagnostics. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.",
        "",
        "## Summary",
        "",
        f"- JSON reports validated: `{payload['summary']['json_reports_validated']}`.",
        f"- Metric cross-check status: `{payload['summary']['metric_cross_check_status']}`.",
        f"- Schema status: `{payload['summary']['schema_status']}`.",
        f"- Canonicalizer safety status: `{payload['summary']['canonicalizer_safety_status']}`.",
        f"- Mismatch count: `{payload['summary']['mismatch_count']}`.",
        f"- Decision: `{payload['decision']['choice']}` - {payload['decision']['label']}.",
        "",
        "## Metric Cross-Checks",
        "",
        "| Metric | Expected | Status | Source fields |",
        "| --- | --- | --- | --- |",
    ]
    for name, metric in payload["metrics"].items():
        fields = "<br>".join(
            f"`{entry['report']}:{entry['field']}={entry['value']}`"
            for entry in metric["sources"]
        )
        lines.append(f"| `{name}` | `{metric['expected']}` | {'ok' if metric['ok'] else 'mismatch'} | {fields} |")
    lines.extend(
        [
            "",
            "## Schema Status",
            "",
            "| Report | Status | Missing fields | Aliases / derived fields |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["schema"]:
        missing = ", ".join(row["missing_fields"]) if row["missing_fields"] else "-"
        aliases = "<br>".join(
            f"`{item['field']} <- {item['path']}{' (derived)' if item['derived'] else ''}`"
            for item in row["field_aliases_used"]
        )
        lines.append(f"| `{row['report']}` | {row['schema_status']} | {missing} | {aliases} |")
    lines.extend(
        [
            "",
            "## Canonicalizer Safety",
            "",
        ]
    )
    for name, ok in payload["canonicalizer_checks"].items():
        lines.append(f"- `{name}`: `{ok}`")
    lines.extend(
        [
            "",
            "## Mismatches",
            "",
        ]
    )
    if payload["mismatches"]:
        for mismatch in payload["mismatches"]:
            lines.append(
                f"- `{mismatch['report']}:{mismatch['field']}` = `{mismatch['value']}`, expected `{mismatch['expected']}`."
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"{payload['decision']['recommended_next_action']}",
            "",
            "No formal eval is recommended.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    inputs = {
        "p2": load_json(args.p2_json),
        "capped": load_json(args.capped_json),
        "normalized_admission": load_json(args.normalized_admission_json),
        "normalization": load_json(args.normalization_json),
        "drift_only": load_json(args.drift_only_json),
        "expanded": load_json(args.expanded_coverage_json),
    }
    payload = analyze(inputs)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"metric_cross_check_status={payload['summary']['metric_cross_check_status']}")
    print(f"schema_status={payload['summary']['schema_status']}")
    print(f"canonicalizer_safety_status={payload['summary']['canonicalizer_safety_status']}")
    print(f"decision={payload['decision']['choice']}")


if __name__ == "__main__":
    main()
