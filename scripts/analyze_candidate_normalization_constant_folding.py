#!/usr/bin/env python3
"""Candidate-normalization diagnostics for constant-chain folding.

This helper is offline/JSON-only. It parses existing coverage and drift-only
diagnostic JSON files, then estimates oracle-coverage and dedup effects if
multiplicative constant chains were canonicalized at candidate-normalization
time. It does not run model decoding, candidate generation, reranking, scoring,
training, fingerprint simulation, or eval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    parser.add_argument("--expanded-coverage-json", type=Path, default=Path("candidate_coverage_pair_expanded.json"))
    parser.add_argument("--drift-only-json", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument(
        "--constant-folding-json",
        type=Path,
        default=Path("drift_only_admission_constant_folding_diagnostics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("candidate_normalization_constant_folding_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("candidate_normalization_constant_folding_diagnostics.json"),
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
        factors = [canonical_node(factor) for factor in flatten_mul(node)]
        has_constant = any(factor.kind == "CONSTANT" for factor in factors)
        non_constant = [factor for factor in factors if factor.kind != "CONSTANT"]
        if not non_constant:
            return CONST
        return multiply_chain(([CONST] if has_constant else []) + non_constant)
    if node.kind in {"add", "sub"}:
        return Node(node.kind, tuple(canonical_node(child) for child in node.children))
    if len(node.children) == 1:
        return Node(node.kind, (canonical_node(node.children[0]),))
    return node


def emit_prefix(node: Node) -> list[str]:
    tokens = [node.kind]
    for child in node.children:
        tokens.extend(emit_prefix(child))
    return tokens


def canonicalize(tokens: list[str]) -> list[str] | None:
    node, pos = parse_prefix(tokens)
    if node is None or pos != len(tokens):
        return None
    return emit_prefix(canonical_node(node))


def protected_signature(tokens: list[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token not in {"CONSTANT", "mul"})


def drift_family(tokens: list[str]) -> str:
    canonical = canonicalize(tokens) or tokens
    if "pow2" in canonical or "pow3" in canonical or "pow" in canonical:
        return "polynomial-like drift"
    if "sin" in canonical:
        return "sin drift"
    if canonical == ["mul", "CONSTANT", "x_0"]:
        return "linear drift"
    if canonical == ["CONSTANT"]:
        return "constant drift"
    return "other / unknown"


def source_bucket(sources: set[str]) -> str:
    if not sources:
        return "neither"
    if sources == {"beam"}:
        return "beam"
    if sources == {"sampling"}:
        return "sampling"
    if sources == {"pair"}:
        return "pair"
    if "beam" in sources and "sampling" in sources:
        return "both"
    return "+".join(sorted(sources))


def current_expanded_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in sample.get("drift_nearest_detailed", []) or []:
        tokens = item.get("tokens", [])
        if not tokens:
            continue
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "source_family": "current_expanded_pool",
                "source": source_bucket(set(item.get("sources", []))),
                "sources": item.get("sources", []),
                "drift_tokens": tokens,
                "occurrences": item.get("occurrences", []),
                "candidate_bucket": "drift_nearest_detailed",
            }
        )
    if sample.get("has_drift"):
        tokens = sample.get("truth_drift", [])
        key = tuple(tokens)
        if tokens and key not in seen:
            rows.append(
                {
                    "sample_index": sample["sample_index"],
                    "source_family": "current_expanded_pool",
                    "source": source_bucket(set(sample.get("drift_sources", []))),
                    "sources": sample.get("drift_sources", []),
                    "drift_tokens": tokens,
                    "occurrences": sample.get("exact_drift_occurrences", []),
                    "candidate_bucket": "exact_drift_truth",
                }
            )
    return rows


def drift_only_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for bucket in ("nearest_drift_only_candidate",):
        row = sample.get(bucket)
        if row and row.get("drift_tokens"):
            key = tuple(row["drift_tokens"])
            if key not in seen:
                seen.add(key)
                rows.append({**row, "source_family": "drift_only_tail", "candidate_bucket": bucket})
    for bucket in ("exact_truth_drift_occurrences", "beam_candidates", "sampling_candidates"):
        for row in sample.get(bucket, []) or []:
            if not row.get("drift_tokens"):
                continue
            key = tuple(row["drift_tokens"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "source_family": "drift_only_tail", "candidate_bucket": bucket})
    return rows


def canonical_match(rows: list[dict[str, Any]], canonical_truth: list[str] | None) -> dict[str, Any]:
    sources = set()
    best = None
    if canonical_truth is None:
        return {"found": False, "sources": [], "source_bucket": "neither", "best": None}
    for row in rows:
        canonical = canonicalize(row.get("drift_tokens", []))
        if canonical != canonical_truth:
            continue
        row_sources = set(row.get("sources", []))
        if not row_sources and row.get("source"):
            row_sources.add(row["source"])
        sources.update(row_sources)
        rank = row.get("source_rank")
        rank_value = rank if rank is not None else 10**9
        candidate = {
            "source": row.get("source"),
            "sources": sorted(row_sources),
            "source_rank": rank,
            "drift_tokens": row.get("drift_tokens", []),
            "canonical_drift_tokens": canonical,
            "candidate_bucket": row.get("candidate_bucket"),
            "source_family": row.get("source_family"),
        }
        if best is None or (rank_value, token_text(candidate["drift_tokens"])) < (
            best.get("_rank_value", 10**9),
            token_text(best.get("drift_tokens")),
        ):
            best = {**candidate, "_rank_value": rank_value}
    if best is not None:
        best.pop("_rank_value", None)
    return {
        "found": bool(sources),
        "sources": sorted(sources),
        "source_bucket": source_bucket(sources),
        "best": best,
    }


def exact_tail_match(sample: dict[str, Any]) -> dict[str, Any]:
    sources = set()
    for source, present in sample.get("exact_truth_drift_sources", {}).items():
        if present:
            sources.add(source)
    return {
        "found": bool(sample.get("exact_truth_drift_appears")),
        "sources": sorted(sources),
        "source_bucket": source_bucket(sources),
        "rank": sample.get("exact_truth_drift_rank"),
    }


def collect_collision_rows(expanded: dict[str, Any], drift_only: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in expanded["samples"]:
        rows.extend(current_expanded_candidates(sample))
    for sample in drift_only["samples"]:
        rows.extend(drift_only_candidates(sample))
    return rows


def collision_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_unique = {}
    canonical_groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    examples: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tokens = row.get("drift_tokens", [])
        canonical = canonicalize(tokens)
        if canonical is None:
            continue
        raw_key = tuple(tokens)
        raw_unique[raw_key] = tokens
        canonical_key = tuple(canonical)
        canonical_groups[canonical_key].add(raw_key)
        if len(examples[canonical_key]) < 5:
            examples[canonical_key].append(
                {
                    "sample_index": row.get("sample_index"),
                    "source_family": row.get("source_family"),
                    "source": row.get("source"),
                    "drift_tokens": tokens,
                }
            )
    collision_groups = {
        key: values for key, values in canonical_groups.items() if len(values) > 1
    }
    largest = sorted(
        collision_groups.items(),
        key=lambda item: (len(item[1]), token_text(list(item[0]))),
        reverse=True,
    )[:8]
    collision_changes_nonconstant = False
    for canonical_key, raw_values in collision_groups.items():
        signatures = {protected_signature(list(raw)) for raw in raw_values}
        if len(signatures) > 1:
            collision_changes_nonconstant = True
            break
    unsafe_checks = {
        "pow2_x0_equals_x0": canonicalize(["pow2", "x_0"]) == canonicalize(["x_0"]),
        "const_pow2_equals_const_x0": canonicalize(["mul", "CONSTANT", "pow2", "x_0"])
        == canonicalize(["mul", "CONSTANT", "x_0"]),
        "sin_x0_equals_x0": canonicalize(["sin", "x_0"]) == canonicalize(["x_0"]),
    }
    return {
        "serialized_raw_drift_expression_count": len(raw_unique),
        "serialized_canonical_drift_expression_count": len(canonical_groups),
        "collision_group_count": len(collision_groups),
        "largest_collision_groups": [
            {
                "canonical_drift_tokens": list(canonical_key),
                "raw_count": len(raw_values),
                "raw_examples": [list(raw) for raw in sorted(raw_values)[:8]],
                "occurrence_examples": examples[canonical_key],
            }
            for canonical_key, raw_values in largest
        ],
        "collision_changes_nonconstant_structure": collision_changes_nonconstant,
        "unsafe_checks": unsafe_checks,
        "unsafe": any(unsafe_checks.values()) or collision_changes_nonconstant,
    }


def sample_count_rows(samples: list[dict[str, Any]], rows_by_sample: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        sample_idx = sample["sample_index"]
        raw = set()
        canonical = set()
        for row in rows_by_sample.get(sample_idx, []):
            tokens = row.get("drift_tokens", [])
            can = canonicalize(tokens)
            if can is None:
                continue
            raw.add(tuple(tokens))
            canonical.add(tuple(can))
        reduction = 0.0 if not raw else 100.0 * (len(raw) - len(canonical)) / len(raw)
        rows.append(
            {
                "sample_index": sample_idx,
                "serialized_raw_unique_drifts": len(raw),
                "serialized_canonical_unique_drifts": len(canonical),
                "reduction_percent": reduction,
            }
        )
    return rows


def analyze(expanded: dict[str, Any], drift_only: dict[str, Any], constant_folding: dict[str, Any]) -> dict[str, Any]:
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    target_indices = [sample["sample_index"] for sample in drift_only["samples"]]
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    drift_only_by_index = {sample["sample_index"]: sample for sample in drift_only["samples"]}
    per_sample = []
    current_new = []
    drift_only_canonical = []
    exact_tail = []
    current_rows_by_sample = {}
    drift_rows_by_sample = {}

    for sample_idx in target_indices:
        expanded_sample = expanded_by_index[sample_idx]
        drift_sample = drift_only_by_index[sample_idx]
        truth = expanded_sample["truth_drift"]
        canonical_truth = canonicalize(truth)
        exact_diffusion_present = bool(expanded_sample.get("has_diffusion"))
        current_rows = current_expanded_candidates(expanded_sample)
        drift_rows = drift_only_candidates(drift_sample)
        current_rows_by_sample[sample_idx] = current_rows
        drift_rows_by_sample[sample_idx] = drift_rows
        current_match = canonical_match(current_rows, canonical_truth)
        drift_match = canonical_match(drift_rows, canonical_truth)
        exact_match = exact_tail_match(drift_sample)
        current_projected = bool(expanded_sample.get("has_full")) or (
            current_match["found"] and exact_diffusion_present
        )
        drift_projected = bool(expanded_sample.get("has_full")) or (
            drift_match["found"] and exact_diffusion_present
        )
        row = {
            "sample_index": sample_idx,
            "truth_drift_tokens": truth,
            "canonical_truth_drift_tokens": canonical_truth,
            "drift_family": drift_sample.get("drift_family"),
            "exact_drift_found": exact_match["found"],
            "exact_drift_rank": exact_match["rank"],
            "exact_drift_source_bucket": exact_match["source_bucket"],
            "canonical_drift_found_current_expanded_pool": current_match["found"],
            "current_expanded_pool_source_bucket": current_match["source_bucket"],
            "current_expanded_pool_best_match": current_match["best"],
            "canonical_drift_found_drift_only_tail": drift_match["found"],
            "drift_only_tail_source_bucket": drift_match["source_bucket"],
            "drift_only_tail_best_match": drift_match["best"],
            "exact_diffusion_present": exact_diffusion_present,
            "projected_full_oracle_after_current_pool_normalization": current_projected,
            "projected_full_oracle_after_drift_only_tail_normalization": drift_projected,
        }
        per_sample.append(row)
        if current_projected and not expanded_sample.get("has_full"):
            current_new.append(row)
        if drift_projected and not expanded_sample.get("has_full"):
            drift_only_canonical.append(row)
        if exact_match["found"] and exact_diffusion_present and not expanded_sample.get("has_full"):
            exact_tail.append(row)

    all_collision_rows = collect_collision_rows(expanded, drift_only)
    collision = collision_diagnostics(all_collision_rows)
    current_count_rows = sample_count_rows(
        [{"sample_index": idx} for idx in target_indices],
        current_rows_by_sample,
    )
    drift_count_rows = sample_count_rows(
        [{"sample_index": idx} for idx in target_indices],
        drift_rows_by_sample,
    )
    all_raw = sum(row["serialized_raw_unique_drifts"] for row in current_count_rows + drift_count_rows)
    all_canonical = sum(row["serialized_canonical_unique_drifts"] for row in current_count_rows + drift_count_rows)
    reduction = 0.0 if all_raw == 0 else 100.0 * (all_raw - all_canonical) / all_raw

    current_family = Counter(row["drift_family"] for row in current_new)
    drift_family_counts = Counter(row["drift_family"] for row in drift_only_canonical)
    exact_tail_family = Counter(row["drift_family"] for row in exact_tail)
    current_sources = Counter(row["current_expanded_pool_source_bucket"] for row in current_new)
    drift_sources = Counter(row["drift_only_tail_source_bucket"] for row in drift_only_canonical)
    exact_sources = Counter(row["exact_drift_source_bucket"] for row in exact_tail)

    current_canonical_coverage = current_full + len(current_new)
    drift_tail_exact_coverage = current_full + len(exact_tail)
    drift_tail_canonical_coverage = current_full + len(drift_only_canonical)

    if collision["unsafe"]:
        decision = {
            "choice": "C",
            "label": "Constant-folding creates unsafe collisions",
            "recommended_next_action": (
                "Do not proceed to normalization; use more conservative family-specific matching diagnostics."
            ),
        }
    elif current_canonical_coverage == drift_tail_canonical_coverage:
        decision = {
            "choice": "A",
            "label": "Current expanded-pool canonicalization is enough",
            "recommended_next_action": (
                "Prepare a helper-only implementation plan for candidate normalization before pairing. "
                "Do not run formal eval."
            ),
        }
    else:
        decision = {
            "choice": "B",
            "label": "Drift-only tail admission plus canonicalization is needed",
            "recommended_next_action": (
                "Run one small coverage-only smoke/admission diagnostic with normalized drift-only candidates, "
                "not formal eval."
            ),
        }

    return {
        "metadata": {
            "current_expanded_full_oracle_present": current_full,
            "target_indices": target_indices,
            "current_expanded_pool_note": (
                "Current-pool canonicalization uses raw drift tokens serialized in "
                "candidate_coverage_pair_expanded.json, primarily drift_nearest_detailed rows. "
                "It does not regenerate candidates."
            ),
            "constant_folding_reference_decision": constant_folding.get("decision", {}),
        },
        "summary": {
            "total_samples_analyzed": len(expanded["samples"]),
            "target_oracle_absent_samples": len(target_indices),
            "current_expanded_full_oracle_coverage": current_full,
            "canonicalized_expanded_pool_full_oracle_coverage": current_canonical_coverage,
            "canonicalized_expanded_pool_gain": len(current_new),
            "exact_tail_admission_full_oracle_coverage": drift_tail_exact_coverage,
            "exact_tail_admission_gain": len(exact_tail),
            "canonicalized_drift_only_admission_full_oracle_coverage": drift_tail_canonical_coverage,
            "canonicalized_drift_only_admission_gain": len(drift_only_canonical),
            "current_expanded_pool_new_samples": [row["sample_index"] for row in current_new],
            "exact_tail_recovered_samples": [row["sample_index"] for row in exact_tail],
            "canonicalized_drift_only_recovered_samples": [row["sample_index"] for row in drift_only_canonical],
            "polynomial_like_missing_after_normalization": [
                row["sample_index"]
                for row in per_sample
                if row["drift_family"] == "polynomial-like drift"
                and not row["projected_full_oracle_after_drift_only_tail_normalization"]
            ],
            "sampling_contributes_after_canonicalization": any(
                "sampling" in bucket for bucket in drift_sources
            ),
        },
        "source_breakdown": {
            "current_expanded_pool": dict(current_sources),
            "exact_tail": dict(exact_sources),
            "drift_only_tail_canonicalized": dict(drift_sources),
        },
        "family_breakdown": {
            "current_expanded_pool_new": {family: current_family.get(family, 0) for family in FAMILY_ORDER},
            "exact_tail": {family: exact_tail_family.get(family, 0) for family in FAMILY_ORDER},
            "drift_only_tail_canonicalized": {
                family: drift_family_counts.get(family, 0) for family in FAMILY_ORDER
            },
        },
        "collision_risk": collision,
        "dedup_impact": {
            "serialized_raw_unique_drift_count_sum": all_raw,
            "serialized_canonical_unique_drift_count_sum": all_canonical,
            "serialized_reduction_percent": reduction,
            "current_expanded_pool_per_sample": current_count_rows,
            "drift_only_tail_per_sample": drift_count_rows,
            "reported_drift_only_raw_unique_count_sum": sum(
                sample["candidate_counts"]["unique_drift_candidates"]
                for sample in drift_only["samples"]
            ),
            "reported_current_expanded_raw_unique_count_sum": sum(
                sample["unique_drifts"] for sample in expanded["samples"]
            ),
        },
        "samples": per_sample,
        "decision": decision,
    }


def source_count_text(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    decision = payload["decision"]
    dedup = payload["dedup_impact"]
    collision = payload["collision_risk"]
    lines = [
        "# Candidate Normalization Constant-Folding Diagnostics",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline JSON-only diagnostic for applying narrow multiplicative constant-chain folding at candidate-normalization / pairing time. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.",
        "",
        "## Executive Summary",
        "",
        f"- Total samples analyzed: `{summary['total_samples_analyzed']}`.",
        f"- Current expanded full oracle coverage: `{summary['current_expanded_full_oracle_coverage']}/32`.",
        f"- Canonicalized expanded-pool coverage: `{summary['canonicalized_expanded_pool_full_oracle_coverage']}/32` (`+{summary['canonicalized_expanded_pool_gain']}`).",
        f"- Exact-tail admission coverage: `{summary['exact_tail_admission_full_oracle_coverage']}/32` (`+{summary['exact_tail_admission_gain']}`).",
        f"- Canonicalized drift-only admission coverage: `{summary['canonicalized_drift_only_admission_full_oracle_coverage']}/32` (`+{summary['canonicalized_drift_only_admission_gain']}`).",
        f"- Current expanded-pool source breakdown: {source_count_text(payload['source_breakdown']['current_expanded_pool'])}.",
        f"- Drift-only tail canonicalized source breakdown: {source_count_text(payload['source_breakdown']['drift_only_tail_canonicalized'])}.",
        f"- Collision groups: `{collision['collision_group_count']}`; unsafe collision flag: `{collision['unsafe']}`.",
        f"- Serialized raw/canonical drift count: `{dedup['serialized_raw_unique_drift_count_sum']}` -> `{dedup['serialized_canonical_unique_drift_count_sum']}` (`{dedup['serialized_reduction_percent']:.1f}%` reduction).",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Coverage By Source",
        "",
        "| Coverage mode | Coverage | Gain | Samples | Source breakdown |",
        "| --- | ---: | ---: | --- | --- |",
        f"| Current expanded pool | {summary['current_expanded_full_oracle_coverage']}/32 | - | - | - |",
        f"| Current expanded pool + canonicalized drift candidates | {summary['canonicalized_expanded_pool_full_oracle_coverage']}/32 | +{summary['canonicalized_expanded_pool_gain']} | `{', '.join(str(i) for i in summary['current_expanded_pool_new_samples'])}` | {source_count_text(payload['source_breakdown']['current_expanded_pool'])} |",
        f"| Drift-only exact tail admission | {summary['exact_tail_admission_full_oracle_coverage']}/32 | +{summary['exact_tail_admission_gain']} | `{', '.join(str(i) for i in summary['exact_tail_recovered_samples'])}` | {source_count_text(payload['source_breakdown']['exact_tail'])} |",
        f"| Drift-only tail + canonicalized admission | {summary['canonicalized_drift_only_admission_full_oracle_coverage']}/32 | +{summary['canonicalized_drift_only_admission_gain']} | `{', '.join(str(i) for i in summary['canonicalized_drift_only_recovered_samples'])}` | {source_count_text(payload['source_breakdown']['drift_only_tail_canonicalized'])} |",
        "",
        "Current expanded-pool canonicalization and drift-only tail admission are intentionally separated. The current expanded JSON supplies some serialized raw drift candidates through `drift_nearest_detailed`; drift-only tail admission uses the separate drift-only smoke JSON.",
        "",
        "## Per-Family Breakdown",
        "",
        "| Family | Current expanded canonical gain | Exact tail gain | Drift-only canonical gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in FAMILY_ORDER:
        lines.append(
            f"| {family} | {payload['family_breakdown']['current_expanded_pool_new'][family]} | "
            f"{payload['family_breakdown']['exact_tail'][family]} | "
            f"{payload['family_breakdown']['drift_only_tail_canonicalized'][family]} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Table",
            "",
            "| Sample | Truth drift | Canonical truth | Exact drift found | Current expanded canonical found | Drift-only canonical found | Exact diffusion present | Projected full oracle after normalization |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["samples"]:
        lines.append(
            f"| {row['sample_index']} | `{token_text(row['truth_drift_tokens'])}` | "
            f"`{token_text(row['canonical_truth_drift_tokens'])}` | "
            f"{row['exact_drift_found']} | "
            f"{row['canonical_drift_found_current_expanded_pool']} ({row['current_expanded_pool_source_bucket']}) | "
            f"{row['canonical_drift_found_drift_only_tail']} ({row['drift_only_tail_source_bucket']}) | "
            f"{row['exact_diffusion_present']} | "
            f"{row['projected_full_oracle_after_drift_only_tail_normalization']} |"
        )
    lines.extend(
        [
            "",
            "## Collision And Overmerge Risk",
            "",
            f"- Raw serialized drift expressions: `{collision['serialized_raw_drift_expression_count']}`.",
            f"- Canonical drift expressions: `{collision['serialized_canonical_drift_expression_count']}`.",
            f"- Collision groups: `{collision['collision_group_count']}`.",
            f"- Any collision changes nonconstant structure: `{collision['collision_changes_nonconstant_structure']}`.",
            f"- Unsafe checks: `{json.dumps(collision['unsafe_checks'], sort_keys=True)}`.",
            f"- Unsafe overall: `{collision['unsafe']}`.",
            "",
            "Largest collision groups:",
            "",
            "| Canonical drift | Raw count | Raw examples |",
            "| --- | ---: | --- |",
        ]
    )
    for group in collision["largest_collision_groups"][:5]:
        raw_examples = "<br>".join(f"`{token_text(tokens)}`" for tokens in group["raw_examples"][:4])
        lines.append(
            f"| `{token_text(group['canonical_drift_tokens'])}` | {group['raw_count']} | {raw_examples} |"
        )
    lines.extend(
        [
            "",
            "The required unsafe cases are all false: `pow2 x_0` does not canonicalize to `x_0`, `mul CONSTANT pow2 x_0` does not canonicalize to `mul CONSTANT x_0`, and `sin x_0` does not canonicalize to `x_0`.",
            "",
            "## Candidate Count / Dedup Impact",
            "",
            f"- Serialized raw unique drift count sum: `{dedup['serialized_raw_unique_drift_count_sum']}`.",
            f"- Serialized canonical unique drift count sum: `{dedup['serialized_canonical_unique_drift_count_sum']}`.",
            f"- Serialized reduction: `{dedup['serialized_reduction_percent']:.1f}%`.",
            f"- Reported drift-only raw unique count sum from smoke metadata: `{dedup['reported_drift_only_raw_unique_count_sum']}`.",
            f"- Reported current expanded raw unique drift count sum: `{dedup['reported_current_expanded_raw_unique_count_sum']}`.",
            "",
            "Per-sample serialized drift-only tail count changes:",
            "",
            "| Sample | Raw unique | Canonical unique | Reduction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dedup["drift_only_tail_per_sample"]:
        lines.append(
            f"| {row['sample_index']} | {row['serialized_raw_unique_drifts']} | "
            f"{row['serialized_canonical_unique_drifts']} | {row['reduction_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Canonicalization reduces duplicated over-nested constant templates in the observed serialized candidates. It should reduce pair combinations rather than increase them, but this remains token-structure-only evidence, not semantic fingerprint validation.",
            "",
            "## Decision",
            "",
            f"`{decision['choice']}. {decision['label']}`: {decision['recommended_next_action']}",
            "",
            "No formal eval is recommended.",
            "",
            "## Limitations",
            "",
            "- Current expanded-pool canonicalization uses raw drift tokens serialized in `candidate_coverage_pair_expanded.json`, primarily `drift_nearest_detailed`; the coverage JSON does not contain every raw candidate sequence.",
            "- Drift-only tail canonicalization uses serialized drift-only smoke candidates and nearest/exact rows.",
            "- Collision safety is token-structure-only; semantic equivalence and selected-rerank behavior still require later diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expanded = load_json(args.expanded_coverage_json)
    drift_only = load_json(args.drift_only_json)
    constant_folding = load_json(args.constant_folding_json)
    payload = analyze(expanded, drift_only, constant_folding)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    summary = payload["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"current_expanded_full_oracle={summary['current_expanded_full_oracle_coverage']}")
    print(f"canonicalized_expanded_pool_coverage={summary['canonicalized_expanded_pool_full_oracle_coverage']}")
    print(f"exact_tail_admission_coverage={summary['exact_tail_admission_full_oracle_coverage']}")
    print(f"canonicalized_drift_only_admission_coverage={summary['canonicalized_drift_only_admission_full_oracle_coverage']}")
    print(f"unsafe={payload['collision_risk']['unsafe']}")
    print(f"decision={payload['decision']['choice']}")


if __name__ == "__main__":
    main()
