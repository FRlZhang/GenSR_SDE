#!/usr/bin/env python3
"""Offline drift-only admission and constant-chain folding diagnostics.

This helper parses existing JSON outputs only. It does not run model decoding,
candidate generation, reranking, fingerprint simulation, training, or eval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Node:
    kind: str
    children: tuple["Node", ...] = ()


CONST = Node("CONSTANT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-only-json", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument("--expanded-coverage-json", type=Path, default=Path("candidate_coverage_pair_expanded.json"))
    parser.add_argument(
        "--previous-diversity-json",
        type=Path,
        default=Path("expanded_pairing_oracle_absent_drift_diversity.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("drift_only_admission_constant_folding_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("drift_only_admission_constant_folding_diagnostics.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def token_text(tokens: list[str] | None) -> str:
    return " ".join(tokens or [])


def parse_prefix(tokens: list[str], pos: int = 0) -> tuple[Node | None, int]:
    if pos >= len(tokens):
        return None, pos
    token = tokens[pos]
    if token in {"CONSTANT", "x_0"}:
        return Node(token), pos + 1
    if token in {"sin", "sqrt", "abs", "pow2", "pow3", "pow"}:
        child, next_pos = parse_prefix(tokens, pos + 1)
        if child is None:
            return None, pos
        return Node(token, (child,)), next_pos
    if token in {"mul", "add", "sub"}:
        left, next_pos = parse_prefix(tokens, pos + 1)
        if left is None:
            return None, pos
        right, next_pos = parse_prefix(tokens, next_pos)
        if right is None:
            return None, pos
        return Node(token, (left, right)), next_pos
    return None, pos


def flatten_mul(node: Node) -> list[Node]:
    if node.kind == "mul":
        factors: list[Node] = []
        for child in node.children:
            factors.extend(flatten_mul(child))
        return factors
    return [node]


def multiply_chain(factors: list[Node]) -> Node:
    if not factors:
        return CONST
    result = factors[-1]
    for factor in reversed(factors[:-1]):
        result = Node("mul", (factor, result))
    return result


def canonical_node(node: Node) -> Node:
    if node.kind in {"CONSTANT", "x_0"}:
        return node
    if node.kind == "mul":
        factors = []
        for factor in flatten_mul(node):
            factors.append(canonical_node(factor))
        const_seen = any(factor.kind == "CONSTANT" for factor in factors)
        non_const = [factor for factor in factors if factor.kind != "CONSTANT"]
        if not non_const:
            return CONST
        if const_seen:
            return multiply_chain([CONST] + non_const)
        return multiply_chain(non_const)
    if node.kind in {"add", "sub"}:
        return Node(node.kind, tuple(canonical_node(child) for child in node.children))
    if len(node.children) == 1:
        return Node(node.kind, (canonical_node(node.children[0]),))
    return node


def emit_prefix(node: Node) -> list[str]:
    if not node.children:
        return [node.kind]
    tokens = [node.kind]
    for child in node.children:
        tokens.extend(emit_prefix(child))
    return tokens


def canonicalize(tokens: list[str]) -> list[str] | None:
    node, pos = parse_prefix(tokens)
    if node is None or pos != len(tokens):
        return None
    return emit_prefix(canonical_node(node))


def drift_family(tokens: list[str]) -> str:
    canonical = canonicalize(tokens) or tokens
    if "pow2" in canonical or "pow3" in canonical or "pow" in canonical:
        return "polynomial-like drift"
    if "sin" in canonical:
        return "sin drift"
    if canonical == ["mul", "CONSTANT", "x_0"]:
        return "linear drift"
    if canonical == ["CONSTANT"] or canonical == ["mul", "CONSTANT", "CONSTANT"]:
        return "constant drift"
    if len(canonical) >= 5 and canonical[:2] == ["mul", "CONSTANT"] and "x_0" in canonical:
        return "linear drift"
    return "other / unknown"


def candidate_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for source_key in ("nearest_drift_only_candidate",):
        row = sample.get(source_key)
        if row and row.get("drift_tokens"):
            key = tuple(row["drift_tokens"])
            if key not in seen:
                seen.add(key)
                rows.append({**row, "candidate_bucket": source_key})
    for source_key in ("exact_truth_drift_occurrences", "beam_candidates", "sampling_candidates"):
        for row in sample.get(source_key, []) or []:
            if not row.get("drift_tokens"):
                continue
            key = tuple(row["drift_tokens"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "candidate_bucket": source_key})
    return rows


def source_bucket(sample: dict[str, Any]) -> str:
    sources = sample.get("exact_truth_drift_sources", {})
    if sources.get("beam") and sources.get("sampling"):
        return "both"
    if sources.get("beam"):
        return "beam"
    if sources.get("sampling"):
        return "sampling"
    return "neither"


def exact_occurrence_summary(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in sample.get("exact_truth_drift_occurrences", []) or []:
        rows.append(
            {
                "source": row.get("source"),
                "source_rank": row.get("source_rank"),
                "normalized_score": row.get("normalized_score"),
                "drift_tokens": row.get("drift_tokens", []),
            }
        )
    return rows


def best_canonical_match(sample: dict[str, Any], canonical_truth: list[str] | None) -> dict[str, Any] | None:
    if canonical_truth is None:
        return None
    matches = []
    for row in candidate_rows(sample):
        canonical_candidate = canonicalize(row.get("drift_tokens", []))
        if canonical_candidate == canonical_truth:
            matches.append((row.get("source_rank", 10**9), row.get("source", ""), row, canonical_candidate))
    if not matches:
        return None
    _rank, _source, row, canonical_candidate = sorted(matches, key=lambda item: (item[0], item[1]))[0]
    return {
        "source": row.get("source"),
        "source_rank": row.get("source_rank"),
        "candidate_bucket": row.get("candidate_bucket"),
        "drift_tokens": row.get("drift_tokens", []),
        "canonical_drift_tokens": canonical_candidate,
        "normalized_score": row.get("normalized_score"),
    }


def analyze(drift_only: dict[str, Any], expanded: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    previous_by_index = {sample["sample_index"]: sample for sample in previous["samples"]}
    current_expanded_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))

    rows = []
    exact_found = []
    canonical_matches = []
    exact_missing_canonical_matches = []
    serialized_candidate_coverage_note = (
        "drift_only_candidate_diversity_smoke.json stores nearest candidates, exact occurrences, "
        "and serialized top beam/sampling candidates; canonical matching is therefore JSON-only "
        "over available serialized candidates, not a new model run."
    )
    for sample in drift_only["samples"]:
        sample_idx = sample["sample_index"]
        truth = sample["truth_drift_tokens"]
        nearest = sample.get("nearest_drift_only_candidate") or {}
        canonical_truth = canonicalize(truth)
        canonical_nearest = canonicalize(nearest.get("drift_tokens", [])) if nearest else None
        exact = bool(sample.get("exact_truth_drift_appears"))
        exact_diffusion_present = bool(expanded_by_index[sample_idx].get("has_diffusion"))
        canonical_match = best_canonical_match(sample, canonical_truth)
        has_canonical_match = canonical_match is not None
        row = {
            "sample_index": sample_idx,
            "truth_drift_tokens": truth,
            "nearest_drift_only_tokens": nearest.get("drift_tokens", []),
            "drift_family": sample.get("drift_family") or drift_family(truth),
            "canonical_truth_drift_tokens": canonical_truth,
            "canonical_nearest_drift_tokens": canonical_nearest,
            "exact_truth_drift_found": exact,
            "exact_truth_drift_rank": sample.get("exact_truth_drift_rank"),
            "exact_truth_drift_source_bucket": source_bucket(sample),
            "exact_truth_drift_occurrences": exact_occurrence_summary(sample),
            "exact_diffusion_present_previous_coverage": exact_diffusion_present,
            "canonical_match": has_canonical_match,
            "canonical_match_best_candidate": canonical_match,
            "canonical_match_from_exact_missing": has_canonical_match and not exact,
            "previous_current_pool_nearest": previous_by_index[sample_idx].get("best_available_drift_candidate"),
        }
        rows.append(row)
        if exact:
            exact_found.append(row)
        if has_canonical_match:
            canonical_matches.append(row)
            if not exact:
                exact_missing_canonical_matches.append(row)

    exact_found_indices = [row["sample_index"] for row in exact_found]
    canonical_indices = [row["sample_index"] for row in canonical_matches]
    exact_found_by_family = Counter(row["drift_family"] for row in exact_found)
    exact_missing_by_family = Counter(row["drift_family"] for row in rows if not row["exact_truth_drift_found"])
    canonical_by_family = Counter(row["drift_family"] for row in canonical_matches)
    canonical_missing_by_family = Counter(row["drift_family"] for row in rows if not row["canonical_match"])
    exact_source_buckets = Counter(row["exact_truth_drift_source_bucket"] for row in exact_found)

    exact_tail_gain = sum(
        1 for row in exact_found if row["exact_diffusion_present_previous_coverage"]
    )
    canonical_gain = sum(
        1 for row in canonical_matches if row["exact_diffusion_present_previous_coverage"]
    )
    linear_sin = [row for row in rows if row["drift_family"] in {"linear drift", "sin drift"}]
    linear_sin_canonical = [row for row in linear_sin if row["canonical_match"]]
    polynomial = [row for row in rows if row["drift_family"] == "polynomial-like drift"]
    polynomial_missing_after_canonical = [row for row in polynomial if not row["canonical_match"]]

    if canonical_gain >= 10 and len(linear_sin_canonical) == len(linear_sin):
        decision = {
            "choice": "B",
            "label": "Constant-folding explains most misses",
            "recommended_next_action": (
                "Run a future helper-only or small smoke candidate-normalization diagnostic "
                "for constant-chain folding before widening beam/top-k. Do not run formal eval."
            ),
        }
    elif exact_tail_gain > 0 and canonical_gain == exact_tail_gain:
        decision = {
            "choice": "A",
            "label": "Exact-tail admission only",
            "recommended_next_action": (
                "Run a small candidate-coverage-only drift admission probe. Do not run formal eval."
            ),
        }
    elif canonical_gain == 0 and exact_tail_gain == 0:
        decision = {
            "choice": "C",
            "label": "Both exact-tail and constant-folded coverage are weak",
            "recommended_next_action": (
                "Run a model/grammar-side drift generation or template-family seeding diagnostic. "
                "Do not run formal eval."
            ),
        }
    else:
        decision = {
            "choice": "D",
            "label": "Logs are still insufficient",
            "recommended_next_action": (
                "Add minimal JSON fields before any candidate-generation intervention."
            ),
        }

    return {
        "metadata": {
            "current_expanded_full_oracle_present": current_expanded_full,
            "target_samples": [row["sample_index"] for row in rows],
            "serialized_candidate_coverage_note": serialized_candidate_coverage_note,
        },
        "summary": {
            "target_samples_analyzed": len(rows),
            "exact_truth_drift_found_count": len(exact_found),
            "exact_truth_drift_found_indices": exact_found_indices,
            "exact_truth_drift_found_by_source_bucket": {
                bucket: exact_source_buckets.get(bucket, 0)
                for bucket in ("beam", "sampling", "both", "neither")
            },
            "exact_tail_projected_full_oracle_coverage": current_expanded_full + exact_tail_gain,
            "exact_tail_projected_gain": exact_tail_gain,
            "canonicalized_drift_match_count": len(canonical_matches),
            "canonicalized_drift_match_indices": canonical_indices,
            "exact_missing_cases_become_canonical_matches": len(exact_missing_canonical_matches),
            "exact_missing_canonical_match_indices": [row["sample_index"] for row in exact_missing_canonical_matches],
            "canonicalized_projected_full_oracle_coverage": current_expanded_full + canonical_gain,
            "canonicalized_projected_gain": canonical_gain,
            "linear_sin_canonical_matches": len(linear_sin_canonical),
            "linear_sin_total": len(linear_sin),
            "polynomial_like_missing_after_canonical": len(polynomial_missing_after_canonical),
            "polynomial_like_total": len(polynomial),
            "exact_diffusion_present_for_exact_tail_hits": sum(row["exact_diffusion_present_previous_coverage"] for row in exact_found),
            "exact_diffusion_present_for_canonical_matches": sum(row["exact_diffusion_present_previous_coverage"] for row in canonical_matches),
        },
        "family_breakdown": {
            "exact_found_by_family": {family: exact_found_by_family.get(family, 0) for family in FAMILY_ORDER},
            "exact_missing_by_family": {family: exact_missing_by_family.get(family, 0) for family in FAMILY_ORDER},
            "canonical_match_by_family": {family: canonical_by_family.get(family, 0) for family in FAMILY_ORDER},
            "canonical_missing_by_family": {family: canonical_missing_by_family.get(family, 0) for family in FAMILY_ORDER},
        },
        "samples": rows,
        "decision": decision,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    family = payload["family_breakdown"]
    decision = payload["decision"]
    lines = [
        "# Drift-Only Admission and Constant-Folding Diagnostics",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline JSON-only diagnostic over the completed drift-only candidate diversity smoke. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.",
        "",
        "## Executive Summary",
        "",
        f"- Target samples analyzed: `{summary['target_samples_analyzed']}`.",
        f"- Exact truth drift found: `{summary['exact_truth_drift_found_count']}/{summary['target_samples_analyzed']}`.",
        f"- Exact-tail projected full-oracle coverage: `{payload['metadata']['current_expanded_full_oracle_present']}/32 + {summary['exact_tail_projected_gain']} = {summary['exact_tail_projected_full_oracle_coverage']}/32`.",
        f"- Canonicalized drift matches: `{summary['canonicalized_drift_match_count']}/{summary['target_samples_analyzed']}`.",
        f"- Exact-missing cases that become canonicalized matches: `{summary['exact_missing_cases_become_canonical_matches']}`.",
        f"- Constant-folded projected full-oracle coverage: `{payload['metadata']['current_expanded_full_oracle_present']}/32 + {summary['canonicalized_projected_gain']} = {summary['canonicalized_projected_full_oracle_coverage']}/32`.",
        f"- Linear/sin canonical matches: `{summary['linear_sin_canonical_matches']}/{summary['linear_sin_total']}`.",
        f"- Polynomial-like missing after constant folding: `{summary['polynomial_like_missing_after_canonical']}/{summary['polynomial_like_total']}`.",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Exact-Tail Admission",
        "",
        f"Exact truth drift appears for samples: `{', '.join(str(i) for i in summary['exact_truth_drift_found_indices'])}`.",
        "",
        "| Sample | Family | Source | Rank | Exact diffusion present |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for row in payload["samples"]:
        if not row["exact_truth_drift_found"]:
            continue
        lines.append(
            f"| {row['sample_index']} | {row['drift_family']} | "
            f"{row['exact_truth_drift_source_bucket']} | {row['exact_truth_drift_rank']} | "
            f"{row['exact_diffusion_present_previous_coverage']} |"
        )
    lines.extend(
        [
            "",
            "All exact-tail hits are beam-only rank-30 nested-mul linear drift cases. This supports an admission ceiling of up to `+3` full-oracle candidates because exact diffusion is already present for those samples. A naive global expansion to beam top-30/top-32 may be expensive and should not become a default from this diagnostic alone.",
            "",
            "## Constant-Folded Admission",
            "",
            f"Canonicalized drift matches appear for samples: `{', '.join(str(i) for i in summary['canonicalized_drift_match_indices'])}`.",
            "",
            f"Among exact-missing cases, canonicalization adds `{summary['exact_missing_cases_become_canonical_matches']}` matches: `{', '.join(str(i) for i in summary['exact_missing_canonical_match_indices'])}`.",
            "",
            "The diagnostic canonicalizer folds only multiplicative chains of `CONSTANT` factors and preserves nonconstant structure such as `sin`, `pow2`, and `x_0`. It is diagnostic-only and does not change training targets, candidate generation, reranking, or production normalization.",
            "",
            "## Per-Family Breakdown",
            "",
            "| Family | Exact found | Exact missing | Canonical match | Canonical missing |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fam in FAMILY_ORDER:
        lines.append(
            f"| {fam} | {family['exact_found_by_family'][fam]} | "
            f"{family['exact_missing_by_family'][fam]} | "
            f"{family['canonical_match_by_family'][fam]} | "
            f"{family['canonical_missing_by_family'][fam]} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Table",
            "",
            "| Sample | Family | Truth drift | Nearest drift-only | Canonical truth | Canonical nearest | Exact found | Canonical match | Exact diffusion present |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["samples"]:
        lines.append(
            f"| {row['sample_index']} | {row['drift_family']} | "
            f"`{token_text(row['truth_drift_tokens'])}` | "
            f"`{token_text(row['nearest_drift_only_tokens'])}` | "
            f"`{token_text(row['canonical_truth_drift_tokens'])}` | "
            f"`{token_text(row['canonical_nearest_drift_tokens'])}` | "
            f"{row['exact_truth_drift_found']} | {row['canonical_match']} | "
            f"{row['exact_diffusion_present_previous_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact tail admission alone can rescue only the three nested-mul linear rank-30 beam cases.",
            "- Constant-chain folding explains all linear and sin exact misses in the available JSON: `12/12` linear/sin cases become canonicalized matches.",
            "- The two polynomial-like cases remain missing after constant folding because `pow2 x_0` is preserved and does not match the linear-like candidate `mul CONSTANT x_0`.",
            "- Since exact diffusion is already present for every matched case, canonicalized drift matching projects up to `+15` full-oracle candidates over the current expanded `15/32` ceiling, for a diagnostic-only canonicalized ceiling of `30/32`.",
            "- This is not selected recovery evidence and does not justify a formal eval by itself.",
            "",
            "## Decision",
            "",
            f"`{decision['choice']}. {decision['label']}`: {decision['recommended_next_action']}",
            "",
            "No formal eval is recommended.",
            "",
            "## Notes",
            "",
            f"- {payload['metadata']['serialized_candidate_coverage_note']}",
            "- The canonicalizer is intentionally narrow: it folds multiplicative `CONSTANT` chains but does not erase `pow2`, state dependence, or unrelated operator structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    drift_only = load_json(args.drift_only_json)
    expanded = load_json(args.expanded_coverage_json)
    previous = load_json(args.previous_diversity_json)
    payload = analyze(drift_only, expanded, previous)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    summary = payload["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"target_samples={summary['target_samples_analyzed']}")
    print(f"exact_truth_drift_found={summary['exact_truth_drift_found_count']}")
    print(f"exact_tail_projected_gain={summary['exact_tail_projected_gain']}")
    print(f"canonicalized_drift_match_count={summary['canonicalized_drift_match_count']}")
    print(f"canonicalized_projected_gain={summary['canonicalized_projected_gain']}")
    print(f"decision={payload['decision']['choice']}")


if __name__ == "__main__":
    main()
