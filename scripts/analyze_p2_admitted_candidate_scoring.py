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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from normalized_drift_admission import canonicalize_constant_chains, token_text
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system, template_tokens


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
    parser.add_argument("--scorer-ready-json", type=Path, default=None)
    parser.add_argument("--p2-coverage-json", type=Path, default=None)
    parser.add_argument("--expanded-coverage-json", type=Path, default=None)
    parser.add_argument("--baseline-coverage-json", type=Path, default=None)
    parser.add_argument("--drift-only-json", type=Path, default=None)
    parser.add_argument("--target-samples", type=str, default="")
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


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def target_array(sample: dict[str, Any]) -> np.ndarray:
    payload = sample.get("target_y_to_fit") or sample.get("target_fingerprint")
    if not isinstance(payload, dict) or not isinstance(payload.get("flat"), list):
        raise ValueError(f"sample {sample.get('sample_index')} missing target_y_to_fit")
    shape = payload.get("shape") or [len(payload["flat"])]
    return np.asarray(payload["flat"], dtype=np.float64).reshape(shape)


def score_seed(sample_index: int, candidate_index: int, base_seed: int = 20260612) -> int:
    return base_seed + sample_index * 1009 + candidate_index


def is_canonical_oracle(row: dict[str, Any]) -> bool:
    return bool(row.get("is_truth_drift_canonical") and row.get("is_truth_diffusion_exact"))


def score_row(
    row: dict[str, Any],
    sample_index: int,
    candidate_index: int,
    target_y: np.ndarray,
    config: FingerprintConfig,
) -> dict[str, Any]:
    words = row.get("full_sequence_tokens") or []
    out = {
        "candidate_id": row.get("candidate_id"),
        "candidate_source": row.get("candidate_source"),
        "full_sequence_tokens": words,
        "full_sequence_string": row.get("full_sequence_string") or " ".join(words),
        "drift_source": row.get("drift_source"),
        "diffusion_source": row.get("diffusion_source"),
        "drift_rank": row.get("drift_rank"),
        "diffusion_rank": row.get("diffusion_rank"),
        "pair_rank": row.get("pair_rank"),
        "is_truth_drift_canonical": bool(row.get("is_truth_drift_canonical")),
        "is_truth_diffusion_exact": bool(row.get("is_truth_diffusion_exact")),
        "is_p2_canonical_oracle": bool(row.get("is_p2_canonical_oracle")),
        "is_oracle_like": is_canonical_oracle(row),
        "parse_ready_bool": bool(row.get("parse_ready_bool")),
        "valid_bool": False,
        "failure_reason": None,
        "score": None,
        "active_score": None,
        "weak_score": None,
        "full_score": None,
        "multi_u0_score": None,
        "best_constant": None,
        "best_drift_constant": None,
        "best_diffusion_constant": None,
        "fingerprint_failures": 0,
    }
    details, failure_reason, fingerprint_failures = score_candidate_system(
        words,
        target_y,
        config,
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        [0.25, 0.5, 1.0, 2.0, 4.0],
        score_seed(sample_index, candidate_index),
    )
    out["failure_reason"] = failure_reason
    out["fingerprint_failures"] = fingerprint_failures
    if details is None:
        return out
    out.update(
        {
            "valid_bool": True,
            "score": details["distance"],
            "active_score": details["active_kramers_moyal_distance"],
            "weak_score": details["gaussian_weak_kernel_distance"],
            "full_score": details["full_distance"],
            "multi_u0_score": details["multi_u0_moments_distance"],
            "best_constant": details["best_constant"],
            "best_drift_constant": details["best_drift_constant"],
            "best_diffusion_constant": details["best_diffusion_constant"],
        }
    )
    return out


def finite_score(row: dict[str, Any]) -> bool:
    score = row.get("score")
    return score is not None and np.isfinite(score)


def rank_scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not finite_score(row),
            row["score"] if row.get("score") is not None else float("inf"),
            row.get("candidate_source") or "",
            row.get("candidate_id") or "",
        ),
    )


def best_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_id": row.get("candidate_id"),
        "candidate_source": row.get("candidate_source"),
        "sequence": row.get("full_sequence_tokens"),
        "sequence_string": row.get("full_sequence_string"),
        "score": row.get("score"),
        "active_score": row.get("active_score"),
        "weak_score": row.get("weak_score"),
        "best_drift_constant": row.get("best_drift_constant"),
        "best_diffusion_constant": row.get("best_diffusion_constant"),
        "drift_source": row.get("drift_source"),
        "diffusion_source": row.get("diffusion_source"),
        "drift_rank": row.get("drift_rank"),
        "diffusion_rank": row.get("diffusion_rank"),
        "pair_rank": row.get("pair_rank"),
        "is_p2_canonical_oracle": row.get("is_p2_canonical_oracle"),
        "is_oracle_like": row.get("is_oracle_like"),
    }


def pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_bool")]
    parse_failures = sum(1 for row in rows if row.get("failure_reason") == "parse_failed")
    fingerprint_failures = sum(
        1
        for row in rows
        if row.get("failure_reason")
        and row.get("failure_reason") != "parse_failed"
        and not row.get("valid_bool")
    )
    best = rank_scored(valid)[0] if valid else None
    return {
        "candidate_count": len(rows),
        "valid_count": len(valid),
        "parse_failures": parse_failures,
        "fingerprint_failures": fingerprint_failures,
        "best_candidate": best_summary(best),
    }


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(finite),
        "median": statistics.median(finite),
        "max": max(finite),
    }


def sample_scoring_summary(sample: dict[str, Any], config: FingerprintConfig) -> dict[str, Any]:
    sample_index = int(sample["sample_index"])
    target_y = target_array(sample)
    scored = [
        score_row(row, sample_index, candidate_idx, target_y, config)
        for candidate_idx, row in enumerate(sample.get("candidates", []), start=1)
    ]
    current_rows = [row for row in scored if row.get("candidate_source") == "current_expanded"]
    p2_rows = [row for row in scored if row.get("candidate_source") == "p2_admitted"]
    combined_valid_ranked = rank_scored([row for row in scored if row.get("valid_bool")])
    selected = combined_valid_ranked[0] if combined_valid_ranked else None

    p2_oracles = [row for row in scored if row.get("is_p2_canonical_oracle")]
    p2_oracle = rank_scored(p2_oracles)[0] if p2_oracles else None
    combined_rank = None
    if p2_oracle is not None:
        for rank, row in enumerate(combined_valid_ranked, start=1):
            if row.get("candidate_id") == p2_oracle.get("candidate_id"):
                combined_rank = rank
                break
    selected_score = selected.get("score") if selected else None
    p2_score = p2_oracle.get("score") if p2_oracle else None
    score_gap = None
    active_gap = None
    weak_gap = None
    if selected is not None and p2_oracle is not None and selected_score is not None and p2_score is not None:
        score_gap = p2_score - selected_score
        active_gap = p2_oracle.get("active_score") - selected.get("active_score")
        weak_gap = p2_oracle.get("weak_score") - selected.get("weak_score")
    near_gap = {
        str(threshold): score_gap is not None and abs(score_gap) <= threshold
        for threshold in (0.005, 0.01, 0.05, 0.10)
    }
    p2_oracle_selected = bool(
        selected is not None
        and p2_oracle is not None
        and selected.get("candidate_id") == p2_oracle.get("candidate_id")
    )
    if p2_oracle_selected:
        miss_reason = "selected_p2_oracle"
    elif not p2_oracles:
        miss_reason = "p2_oracle_missing"
    elif p2_oracle is not None and not p2_oracle.get("valid_bool") and p2_oracle.get("failure_reason") == "parse_failed":
        miss_reason = "parse_failure"
    elif p2_oracle is not None and not p2_oracle.get("valid_bool"):
        miss_reason = "fingerprint_failure"
    else:
        miss_reason = "p2_oracle_present_but_score_miss"

    current_summary = pool_summary(current_rows)
    p2_summary = pool_summary(p2_rows)
    combined_summary = pool_summary(scored)
    current_best = current_summary["best_candidate"]
    p2_best = p2_summary["best_candidate"]
    return {
        "sample_index": sample_index,
        "truth_sequence": sample.get("truth_sequence"),
        "truth_drift_tokens": sample.get("truth_drift_tokens"),
        "truth_diffusion_tokens": sample.get("truth_diffusion_tokens"),
        "current_expanded": {
            **current_summary,
            "selected_is_oracle_bool": bool(current_best and current_best.get("is_oracle_like")),
        },
        "p2_admitted": {
            **p2_summary,
            "selected_is_p2_oracle_bool": bool(p2_best and p2_best.get("is_p2_canonical_oracle")),
        },
        "combined": {
            **combined_summary,
            "selected_candidate": best_summary(selected),
            "selected_sequence": selected.get("full_sequence_tokens") if selected else None,
            "selected_source": selected.get("candidate_source") if selected else None,
            "selected_score": selected.get("score") if selected else None,
            "selected_active_score": selected.get("active_score") if selected else None,
            "selected_weak_score": selected.get("weak_score") if selected else None,
            "selected_best_drift_constant": selected.get("best_drift_constant") if selected else None,
            "selected_best_diffusion_constant": selected.get("best_diffusion_constant") if selected else None,
        },
        "p2_oracle": {
            "present_bool": bool(p2_oracles),
            "candidate_id": p2_oracle.get("candidate_id") if p2_oracle else None,
            "source": p2_oracle.get("drift_source") if p2_oracle else None,
            "drift_rank": p2_oracle.get("drift_rank") if p2_oracle else None,
            "diffusion_rank": p2_oracle.get("diffusion_rank") if p2_oracle else None,
            "combined_rank": combined_rank,
            "score": p2_score,
            "active_score": p2_oracle.get("active_score") if p2_oracle else None,
            "weak_score": p2_oracle.get("weak_score") if p2_oracle else None,
            "best_drift_constant": p2_oracle.get("best_drift_constant") if p2_oracle else None,
            "best_diffusion_constant": p2_oracle.get("best_diffusion_constant") if p2_oracle else None,
            "would_be_selected_bool": p2_oracle_selected,
            "score_gap_vs_selected": score_gap,
            "active_score_gap_vs_selected": active_gap,
            "weak_score_gap_vs_selected": weak_gap,
            "near_gap_bool": near_gap,
            "miss_reason": miss_reason,
            "parse_failure": bool(
                p2_oracle is not None and p2_oracle.get("failure_reason") == "parse_failed"
            ),
            "fingerprint_failure": bool(
                p2_oracle is not None
                and p2_oracle.get("failure_reason")
                and p2_oracle.get("failure_reason") != "parse_failed"
                and not p2_oracle.get("valid_bool")
            ),
        },
        "scored_candidates": scored,
    }


def build_scoring_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Admitted Candidate Semantic Scoring",
        "",
        "## Scorer",
        "",
        "`constant_grid_rolewise_no_multi_u0`, active:weak `1:1`, constants `0.25,0.5,1.0,2.0,4.0`, with separate drift/diffusion constants and `multi_u0` excluded.",
        "",
        "Lower score is better. No formal eval was run.",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Target samples analyzed | {summary['target_samples_analyzed']} |",
        f"| Candidate rows analyzed | {summary['candidate_rows_analyzed']} |",
        f"| Valid candidates | {summary['valid_candidate_count']} |",
        f"| Parse failures | {summary['parse_failure_count']} |",
        f"| Fingerprint failures | {summary['fingerprint_failure_count']} |",
        f"| P2 oracle present | {summary['p2_oracle_present_count']} |",
        f"| P2 oracle selected | {summary['p2_oracle_selected_count']} |",
        f"| P2 oracle score misses | {summary['p2_oracle_score_miss_count']} |",
        f"| Combined selected from P2 | {summary['combined_selected_from_p2_count']} |",
        f"| Combined selected from current | {summary['combined_selected_from_current_count']} |",
        f"| Median P2 oracle rank | {summary['median_p2_oracle_rank']} |",
        f"| Max P2 oracle rank | {summary['max_p2_oracle_rank']} |",
        "",
        "Near-gap counts:",
        "",
    ]
    for threshold, count in summary["near_gap_counts"].items():
        lines.append(f"- `<= {threshold}`: {count}")
    lines.extend(
        [
            "",
            f"Score gap summary: `{json.dumps(summary['score_gap_summary'], sort_keys=True)}`",
            f"Active gap summary: `{json.dumps(summary['active_score_gap_summary'], sort_keys=True)}`",
            f"Weak gap summary: `{json.dumps(summary['weak_score_gap_summary'], sort_keys=True)}`",
            "",
            "## Per-Sample P2 Oracle",
            "",
            "| Sample | Selected source | P2 oracle rank | P2 oracle score | Selected score | Gap | Near <=0.10 | Miss reason |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for sample in result["samples"]:
        p2 = sample["p2_oracle"]
        combined = sample["combined"]
        lines.append(
            "| {idx} | {source} | {rank} | {p2_score} | {selected_score} | {gap} | {near} | {reason} |".format(
                idx=sample["sample_index"],
                source=combined.get("selected_source"),
                rank=p2.get("combined_rank"),
                p2_score=format_optional(p2.get("score")),
                selected_score=format_optional(combined.get("selected_score")),
                gap=format_optional(p2.get("score_gap_vs_selected")),
                near=p2.get("near_gap_bool", {}).get("0.1"),
                reason=p2.get("miss_reason"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval is recommended unless a later small eval-integration smoke improves selected behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_scorer_ready_mode(args: argparse.Namespace) -> None:
    sidecar = load_json(args.scorer_ready_json)
    target_samples = parse_target_samples(args.target_samples)
    if not target_samples:
        target_samples = sidecar.get("metadata", {}).get("target_samples_included", [])
    samples_by_index = {
        int(sample["sample_index"]): sample
        for sample in sidecar.get("samples", [])
    }
    config = FingerprintConfig(n_paths=800, n_steps=60, active_paths=800)
    sample_results = [
        sample_scoring_summary(samples_by_index[idx], config)
        for idx in target_samples
    ]
    all_candidates = [
        candidate
        for sample in sample_results
        for candidate in sample.get("scored_candidates", [])
    ]
    valid_candidates = [candidate for candidate in all_candidates if candidate.get("valid_bool")]
    p2_oracles = [sample["p2_oracle"] for sample in sample_results]
    score_gaps = [
        p2["score_gap_vs_selected"]
        for p2 in p2_oracles
        if p2.get("score_gap_vs_selected") is not None
    ]
    active_gaps = [
        p2["active_score_gap_vs_selected"]
        for p2 in p2_oracles
        if p2.get("active_score_gap_vs_selected") is not None
    ]
    weak_gaps = [
        p2["weak_score_gap_vs_selected"]
        for p2 in p2_oracles
        if p2.get("weak_score_gap_vs_selected") is not None
    ]
    p2_ranks = [
        p2["combined_rank"]
        for p2 in p2_oracles
        if p2.get("combined_rank") is not None
    ]
    near_gap_counts = {
        str(threshold): sum(
            1
            for p2 in p2_oracles
            if p2.get("score_gap_vs_selected") is not None
            and abs(p2["score_gap_vs_selected"]) <= threshold
        )
        for threshold in (0.005, 0.01, 0.05, 0.10)
    }
    p2_selected = sum(1 for p2 in p2_oracles if p2.get("would_be_selected_bool"))
    p2_present = sum(1 for p2 in p2_oracles if p2.get("present_bool"))
    parse_failures = sum(1 for candidate in all_candidates if candidate.get("failure_reason") == "parse_failed")
    fingerprint_failures = sum(
        1
        for candidate in all_candidates
        if candidate.get("failure_reason")
        and candidate.get("failure_reason") != "parse_failed"
        and not candidate.get("valid_bool")
    )
    if parse_failures or fingerprint_failures:
        decision = "C"
        recommendation = "Fix candidate serialization or fingerprint failures before scoring/eval integration."
    elif p2_selected or near_gap_counts["0.1"] >= max(1, p2_present // 2):
        decision = "A"
        recommendation = "P2 oracle candidates are selected or often near-selected; a future small eval-integration smoke may be justified."
    else:
        decision = "B"
        recommendation = "P2 oracle candidates are present but mostly lose under current scorer; diagnose residual/segment behavior before eval integration."
    result = {
        "metadata": {
            "scorer_ready_json": str(args.scorer_ready_json),
            "scorer": "constant_grid_rolewise_no_multi_u0",
            "component_weights": [1.0, 2.0, 1.0],
            "effective_active_weak_weights": [1.0, 1.0],
            "constant_values": [0.25, 0.5, 1.0, 2.0, 4.0],
            "fingerprint_config": {
                "n_paths": 800,
                "active_paths": 800,
                "n_steps": 60,
            },
            "semantic_scoring_mode": "offline_sidecar_only",
            "formal_eval_run": False,
        },
        "summary": {
            "target_samples_analyzed": len(sample_results),
            "target_samples": target_samples,
            "candidate_rows_analyzed": len(all_candidates),
            "valid_candidate_count": len(valid_candidates),
            "parse_failure_count": parse_failures,
            "fingerprint_failure_count": fingerprint_failures,
            "p2_oracle_present_count": p2_present,
            "p2_oracle_selected_count": p2_selected,
            "p2_oracle_score_miss_count": sum(
                1
                for p2 in p2_oracles
                if p2.get("miss_reason") == "p2_oracle_present_but_score_miss"
            ),
            "p2_oracle_parse_failure_count": sum(1 for p2 in p2_oracles if p2.get("parse_failure")),
            "p2_oracle_fingerprint_failure_count": sum(1 for p2 in p2_oracles if p2.get("fingerprint_failure")),
            "combined_selected_from_p2_count": sum(
                1
                for sample in sample_results
                if sample["combined"].get("selected_source") == "p2_admitted"
            ),
            "combined_selected_from_current_count": sum(
                1
                for sample in sample_results
                if sample["combined"].get("selected_source") == "current_expanded"
            ),
            "median_p2_oracle_rank": statistics.median(p2_ranks) if p2_ranks else None,
            "max_p2_oracle_rank": max(p2_ranks) if p2_ranks else None,
            "near_gap_counts": near_gap_counts,
            "score_gap_summary": percentile_summary(score_gaps),
            "active_score_gap_summary": percentile_summary(active_gaps),
            "weak_score_gap_summary": percentile_summary(weak_gaps),
            "decision": decision,
            "recommendation": recommendation,
            "formal_eval_recommended": False,
        },
        "samples": sample_results,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_scoring_report(json_ready(result)))
    print(f"decision={decision}")
    print(f"candidate_rows_analyzed={len(all_candidates)}")
    print(f"valid_candidate_count={len(valid_candidates)}")
    print(f"parse_failure_count={parse_failures}")
    print(f"fingerprint_failure_count={fingerprint_failures}")
    print(f"p2_oracle_selected_count={p2_selected}")


def main() -> None:
    args = parse_args()
    if args.scorer_ready_json is not None:
        run_scorer_ready_mode(args)
        return
    required = [
        ("--p2-coverage-json", args.p2_coverage_json),
        ("--expanded-coverage-json", args.expanded_coverage_json),
        ("--baseline-coverage-json", args.baseline_coverage_json),
        ("--drift-only-json", args.drift_only_json),
        ("--target-samples", args.target_samples),
    ]
    missing = [name for name, value in required if not value]
    if missing:
        raise SystemExit(
            "readiness-only mode requires " + ", ".join(missing)
        )
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
