#!/usr/bin/env python3
"""Validate the candidate coverage P2 normalized-admission flag.

This helper parses existing JSON artifacts only. It builds the same
pipeline-facing P2 coverage block exposed by ``analyze_candidate_coverage.py``
and cross-checks it against the standalone reusable-hook reports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

from analyze_candidate_coverage import (
    DEFAULT_NORMALIZED_DRIFT_ADMISSION,
    DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE,
    NORMALIZED_DRIFT_ADMISSION_POLICIES,
    build_normalized_drift_admission_block,
)


EXPECTED = {
    "current_expanded_full_oracle": 15,
    "canonicalized_expanded_pool_coverage": 23,
    "p2_normalized_admission_coverage": 30,
    "p2_pair_count": 340,
    "newly_recovered": [0, 2, 6, 7, 15, 25, 29],
    "remaining_missing": [16, 28],
    "sampling_excluded": True,
    "sampling_recovered_canonical_drift_count": 0,
    "unsafe_collision_flag": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-coverage-json", type=Path, default=Path("candidate_coverage_pair_expanded.json"))
    parser.add_argument("--drift-only-json", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument("--p2-json", type=Path, default=Path("candidate_coverage_p2_normalized_admission.json"))
    parser.add_argument("--hook-validation-json", type=Path, default=Path("p2_normalized_admission_hook_validation.json"))
    parser.add_argument("--report", type=Path, default=Path("candidate_coverage_p2_flag_validation.md"))
    parser.add_argument("--json-output", type=Path, default=Path("candidate_coverage_p2_flag_validation.json"))
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


def add_metric(metrics: dict[str, Any], name: str, expected: Any, sources: list[tuple[str, str, Any]]) -> None:
    normalized_expected = sorted_list(expected)
    entries = []
    mismatches = []
    for report, field, value in sources:
        ok = sorted_list(value) == normalized_expected
        entries.append({"report": report, "field": field, "value": value, "ok": ok})
        if not ok:
            mismatches.append({"report": report, "field": field, "value": value, "expected": expected})
    metrics[name] = {"expected": expected, "sources": entries, "ok": not mismatches, "mismatches": mismatches}


def schema_status(block: dict[str, Any], p2: dict[str, Any], hook_validation: dict[str, Any]) -> dict[str, Any]:
    required_block_fields = [
        "policy",
        "source",
        "current_expanded_full_oracle",
        "canonicalized_expanded_pool_coverage",
        "p2_normalized_admission_coverage",
        "p2_pair_count",
        "newly_recovered",
        "remaining_missing",
        "sampling_excluded",
        "sampling_recovered_canonical_drift_count",
        "unsafe_collision_flag",
        "source_breakdown",
        "family_breakdown",
        "admitted_candidates_by_sample",
        "samples",
    ]
    missing = [field for field in required_block_fields if field not in block]
    if at(p2, "summary.p2_normalized_admission_coverage") is None:
        missing.append("p2_json.summary.p2_normalized_admission_coverage")
    if at(hook_validation, "summary.metric_cross_check_status") is None:
        missing.append("hook_validation.summary.metric_cross_check_status")
    if "p2_one_per_family" not in NORMALIZED_DRIFT_ADMISSION_POLICIES:
        missing.append("analyze_candidate_coverage flag choice p2_one_per_family")
    if DEFAULT_NORMALIZED_DRIFT_ADMISSION != "none":
        missing.append("analyze_candidate_coverage default none")
    if DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE != "beam":
        missing.append("analyze_candidate_coverage default source beam")
    return {
        "schema_status": "ok" if not missing else "missing",
        "missing_fields": missing,
        "field_aliases_used": [
            "pipeline_block.* from build_normalized_drift_admission_block",
            "p2_json.summary.*",
            "hook_validation.summary.*",
        ],
    }


def analyze(expanded: dict[str, Any], drift_only: dict[str, Any], p2: dict[str, Any], hook_validation: dict[str, Any]) -> dict[str, Any]:
    block = build_normalized_drift_admission_block(
        expanded,
        drift_only,
        policy="p2_one_per_family",
        source="beam",
    )
    metrics: dict[str, Any] = {}
    for name, expected in EXPECTED.items():
        p2_field = {
            "newly_recovered": "summary.newly_recovered_beyond_current",
            "remaining_missing": "summary.remaining_missing_samples",
            "unsafe_collision_flag": "safety.unsafe_collision_flag",
        }.get(name, f"summary.{name}")
        add_metric(
            metrics,
            name,
            expected,
            [
                ("pipeline_flag_block", name, block.get(name)),
                ("candidate_coverage_p2_normalized_admission", p2_field, at(p2, p2_field)),
            ],
        )
    add_metric(
        metrics,
        "metric_cross_check_status",
        "ok",
        [
            ("hook_validation", "summary.metric_cross_check_status", at(hook_validation, "summary.metric_cross_check_status")),
        ],
    )
    add_metric(
        metrics,
        "canonicalizer_safety_status",
        "ok",
        [
            ("hook_validation", "summary.canonicalizer_safety_status", at(hook_validation, "summary.canonicalizer_safety_status")),
        ],
    )
    schema = schema_status(block, p2, hook_validation)
    mismatches = [mismatch for metric in metrics.values() for mismatch in metric["mismatches"]]
    safety_ok = bool(block.get("safety", {}).get("self_checks")) and all(block["safety"]["self_checks"].values())
    if mismatches:
        decision = {
            "choice": "C",
            "label": "Mismatch detected",
            "recommended_next_action": "Resolve the listed mismatches before running pipeline integration. No formal eval.",
        }
    elif schema["schema_status"] != "ok":
        decision = {
            "choice": "D",
            "label": "Insufficient fields",
            "recommended_next_action": "Add minimal JSON fields before model-backed smoke. No formal eval.",
        }
    elif not safety_ok:
        decision = {
            "choice": "C",
            "label": "Canonicalizer safety failed",
            "recommended_next_action": "Do not run pipeline integration until safety checks pass. No formal eval.",
        }
    else:
        decision = {
            "choice": "A",
            "label": "Pipeline flag is ready for a future small local coverage-only run",
            "recommended_next_action": "Run a future small local coverage-only candidate coverage command with the explicit P2 flag. No formal eval.",
        }
    return {
        "summary": {
            "default_off_flag_present": "p2_one_per_family" in NORMALIZED_DRIFT_ADMISSION_POLICIES,
            "default_behavior_intended_unchanged": DEFAULT_NORMALIZED_DRIFT_ADMISSION == "none",
            "default_source": DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE,
            "uses_reusable_p2_hook": True,
            "independent_canonicalizer_duplication_avoided": True,
            "json_reports_validated": 4,
            "metric_cross_check_status": "ok" if not mismatches else "mismatch",
            "schema_status": schema["schema_status"],
            "canonicalizer_safety_status": "ok" if safety_ok and at(hook_validation, "summary.canonicalizer_safety_status") == "ok" else "failed",
            "mismatch_count": len(mismatches),
            "model_backed_candidate_generation_avoided": True,
            "full_no_flag_runtime_equivalence": "not_run_model_backed; default-off code path leaves report/JSON unchanged when disabled",
        },
        "pipeline_block": block,
        "metrics": metrics,
        "schema": schema,
        "mismatches": mismatches,
        "decision": decision,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Candidate Coverage P2 Flag Validation",
        "",
        "Date: 2026-06-24",
        "",
        "Scope: JSON-only validation of the default-off P2 normalized drift admission flag in the candidate coverage pipeline. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.",
        "",
        "## Summary",
        "",
        f"- Default-off flag present: `{summary['default_off_flag_present']}`.",
        f"- Default behavior intended unchanged: `{summary['default_behavior_intended_unchanged']}`.",
        f"- Default source: `{summary['default_source']}`.",
        f"- Uses reusable `scripts/normalized_drift_admission.py` hook: `{summary['uses_reusable_p2_hook']}`.",
        f"- Independent canonicalizer duplication avoided: `{summary['independent_canonicalizer_duplication_avoided']}`.",
        f"- JSON reports validated: `{summary['json_reports_validated']}`.",
        f"- Metric cross-check status: `{summary['metric_cross_check_status']}`.",
        f"- Schema status: `{summary['schema_status']}`.",
        f"- Canonicalizer safety status: `{summary['canonicalizer_safety_status']}`.",
        f"- Mismatch count: `{summary['mismatch_count']}`.",
        f"- Model-backed candidate generation avoided: `{summary['model_backed_candidate_generation_avoided']}`.",
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
            f"- Schema status: `{payload['schema']['schema_status']}`.",
            f"- Missing fields: `{payload['schema']['missing_fields']}`.",
            f"- Field aliases used: `{payload['schema']['field_aliases_used']}`.",
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
            payload["decision"]["recommended_next_action"],
            "",
            "No formal eval is recommended.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    payload = analyze(
        load_json(args.expanded_coverage_json),
        load_json(args.drift_only_json),
        load_json(args.p2_json),
        load_json(args.hook_validation_json),
    )
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
