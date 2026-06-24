#!/usr/bin/env python3
"""Validate P2 scorer-ready coverage sidecar logging."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TARGETS = [0, 2, 6, 7, 15, 25, 29]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def has_vector_payload(sample: dict[str, Any]) -> bool:
    payload = sample.get("target_y_to_fit") or sample.get("target_fingerprint")
    return bool(
        isinstance(payload, dict)
        and isinstance(payload.get("shape"), list)
        and isinstance(payload.get("flat"), list)
        and payload["flat"]
    )


def vector_length(sample: dict[str, Any]) -> int | None:
    payload = sample.get("target_y_to_fit") or sample.get("target_fingerprint")
    if not isinstance(payload, dict) or not isinstance(payload.get("flat"), list):
        return None
    return len(payload["flat"])


def p2_oracle_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in sample.get("candidates", [])
        if row.get("candidate_source") == "p2_admitted" and row.get("is_p2_canonical_oracle")
    ]


def build_result(scorer_ready: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    samples = scorer_ready.get("samples", [])
    sample_indices = [sample.get("sample_index") for sample in samples]
    lengths = [vector_length(sample) for sample in samples]
    present_vectors = [idx for idx, sample in zip(sample_indices, samples) if has_vector_payload(sample)]
    p2_rows_by_sample = {sample.get("sample_index"): p2_oracle_rows(sample) for sample in samples}
    p2_oracle_present = [
        idx
        for idx, sample in zip(sample_indices, samples)
        if sample.get("p2_oracle_present_bool") and p2_rows_by_sample.get(idx)
    ]
    exact_diffusion_present = [
        idx for idx, sample in zip(sample_indices, samples) if sample.get("exact_diffusion_present_bool")
    ]
    p2_oracle_source_ok = []
    p2_oracle_rank_ok = []
    full_sequence_ok = []
    parse_ready_ok = []
    for sample in samples:
        idx = sample.get("sample_index")
        rows = p2_rows_by_sample.get(idx, [])
        if rows and all(row.get("drift_source") == "beam" for row in rows):
            p2_oracle_source_ok.append(idx)
        if rows and all(row.get("drift_rank") == 5 for row in rows):
            p2_oracle_rank_ok.append(idx)
        if all(row.get("full_sequence_tokens") and row.get("full_sequence_string") for row in sample.get("candidates", [])):
            full_sequence_ok.append(idx)
        if all("parse_ready_bool" in row and row.get("parse_ready_bool") for row in sample.get("candidates", [])):
            parse_ready_ok.append(idx)

    semantic_score_keys = {
        key
        for sample in samples
        for row in sample.get("candidates", [])
        for key in row
        if key in {"semantic_score", "fingerprint_distance", "rerank_distance", "distance_details"}
    }
    consistency = {
        "sidecar_produced": bool(samples),
        "target_samples_expected": EXPECTED_TARGETS,
        "target_samples_included": sample_indices,
        "target_samples_match": sample_indices == EXPECTED_TARGETS,
        "sample_count": len(samples),
        "target_vector_present_count": len(present_vectors),
        "target_vector_present_samples": present_vectors,
        "target_fingerprint_lengths": sorted({length for length in lengths if length is not None}),
        "target_fingerprint_shapes": sorted(
            {
                tuple((sample.get("target_y_to_fit") or {}).get("shape", []))
                for sample in samples
            }
        ),
        "p2_oracle_row_count": sum(len(rows) for rows in p2_rows_by_sample.values()),
        "p2_oracle_present_count": len(p2_oracle_present),
        "p2_oracle_present_samples": p2_oracle_present,
        "exact_diffusion_present_count": len(exact_diffusion_present),
        "exact_diffusion_present_samples": exact_diffusion_present,
        "p2_oracle_source_beam_samples": p2_oracle_source_ok,
        "p2_oracle_rank5_samples": p2_oracle_rank_ok,
        "full_sequence_fields_present_samples": full_sequence_ok,
        "parse_ready_present_samples": parse_ready_ok,
        "semantic_score_keys_present": sorted(semantic_score_keys),
        "semantic_scoring_run": bool(semantic_score_keys),
        "readiness_previous_decision": readiness.get("summary", {}).get("decision"),
    }
    all_good = (
        consistency["target_samples_match"]
        and consistency["sample_count"] == 7
        and consistency["target_vector_present_count"] == 7
        and len(consistency["target_fingerprint_lengths"]) == 1
        and consistency["p2_oracle_present_count"] == 7
        and consistency["p2_oracle_row_count"] == 7
        and consistency["exact_diffusion_present_count"] == 7
        and len(consistency["p2_oracle_source_beam_samples"]) == 7
        and len(consistency["p2_oracle_rank5_samples"]) == 7
        and len(consistency["full_sequence_fields_present_samples"]) == 7
        and len(consistency["parse_ready_present_samples"]) == 7
        and not consistency["semantic_scoring_run"]
    )
    if all_good:
        decision = "A"
        recommendation = "Next step can be offline semantic scoring of P2 admitted candidates."
    elif consistency["target_vector_present_count"] < 7:
        decision = "D"
        recommendation = "Do not score; target fingerprint / y_to_fit is still missing."
    elif consistency["p2_oracle_present_count"] < 7 or consistency["p2_oracle_row_count"] < 7:
        decision = "C"
        recommendation = "Fix candidate row logging before scoring."
    else:
        decision = "B"
        recommendation = "Clean brittle schema details before scoring."

    source_counts = Counter()
    for sample in samples:
        for row in sample.get("candidates", []):
            source_counts[row.get("candidate_source", "unknown")] += 1

    return {
        "metadata": {
            "scorer_ready_json": str(scorer_ready.get("metadata", {}).get("source_coverage_json", "")),
            "semantic_scoring_intentionally_not_run": True,
        },
        "summary": {
            **consistency,
            "candidate_source_counts": dict(source_counts),
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
    }


def build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Scorer-Ready Logging Validation",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Sidecar JSON produced | {summary['sidecar_produced']} |",
        f"| Target samples included | {','.join(str(i) for i in summary['target_samples_included'])} |",
        f"| Sample count | {summary['sample_count']} |",
        f"| Target vector present | {summary['target_vector_present_count']}/7 |",
        f"| Fingerprint length(s) | {','.join(str(i) for i in summary['target_fingerprint_lengths'])} |",
        f"| P2 oracle rows | {summary['p2_oracle_row_count']} |",
        f"| P2 oracle present | {summary['p2_oracle_present_count']}/7 |",
        f"| Exact diffusion present | {summary['exact_diffusion_present_count']}/7 |",
        f"| P2 oracle source beam | {len(summary['p2_oracle_source_beam_samples'])}/7 |",
        f"| P2 oracle rank 5 | {len(summary['p2_oracle_rank5_samples'])}/7 |",
        f"| Full sequence fields present | {len(summary['full_sequence_fields_present_samples'])}/7 |",
        f"| Parse-ready rows present | {len(summary['parse_ready_present_samples'])}/7 |",
        f"| Semantic scoring run | {summary['semantic_scoring_run']} |",
        "",
        "Candidate source rows:",
        "",
    ]
    for source, count in sorted(summary["candidate_source_counts"].items()):
        lines.append(f"- `{source}`: {count}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No semantic scoring or formal eval was run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    scorer_ready = load_json(args.scorer_ready_json)
    readiness = load_json(args.readiness_json)
    result = build_result(scorer_ready, readiness)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_report(result))
    summary = result["summary"]
    print(f"decision={summary['decision']}")
    print(f"sample_count={summary['sample_count']}")
    print(f"target_vector_present_count={summary['target_vector_present_count']}")
    print(f"p2_oracle_row_count={summary['p2_oracle_row_count']}")


if __name__ == "__main__":
    main()
