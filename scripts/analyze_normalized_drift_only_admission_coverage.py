#!/usr/bin/env python3
"""Coverage-only diagnostic for normalized drift-only admission.

This helper is offline/JSON-only. It simulates a diagnostic pool consisting of
the current expanded-pairing coverage plus normalized drift-only tail candidates
for the 17 oracle-absent samples. It does not run model decoding, candidate
generation, fingerprint simulation, reranking, scoring, formal eval, or
training.
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
        "--normalization-json",
        type=Path,
        default=Path("candidate_normalization_constant_folding_diagnostics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("normalized_drift_only_admission_coverage_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("normalized_drift_only_admission_coverage_diagnostics.json"),
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
    if node.kind != "mul":
        return [node]
    factors: list[Node] = []
    for child in node.children:
        factors.extend(flatten_mul(child))
    return factors


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


def current_expanded_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
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


def drift_only_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for bucket in ("nearest_drift_only_candidate",):
        row = sample.get(bucket)
        if row and row.get("drift_tokens"):
            key = tuple(row["drift_tokens"])
            if key not in seen:
                seen.add(key)
                rows.append({**row, "source_family": "drift_only_tail", "candidate_bucket": bucket})
    for bucket in ("exact_truth_drift_occurrences", "beam_candidates", "sampling_candidates"):
        for row in sample.get(bucket, []) or []:
            tokens = row.get("drift_tokens", [])
            if not tokens:
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "source_family": "drift_only_tail", "candidate_bucket": bucket})
    return rows


def canonical_set(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for row in rows:
        canonical = canonicalize(row.get("drift_tokens", []))
        if canonical is not None:
            result.add(tuple(canonical))
    return result


def raw_set(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {tuple(row["drift_tokens"]) for row in rows if row.get("drift_tokens")}


def canonical_match(rows: list[dict[str, Any]], canonical_truth: list[str] | None) -> dict[str, Any]:
    if canonical_truth is None:
        return {"found": False, "sources": [], "source_bucket": "neither", "best": None}
    truth_key = tuple(canonical_truth)
    sources: set[str] = set()
    best: dict[str, Any] | None = None
    for row in rows:
        canonical = canonicalize(row.get("drift_tokens", []))
        if canonical is None or tuple(canonical) != truth_key:
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
            "candidate_bucket": row.get("candidate_bucket"),
            "drift_tokens": row.get("drift_tokens", []),
            "canonical_drift_tokens": canonical,
            "_rank_value": rank_value,
        }
        if best is None or (rank_value, token_text(candidate["drift_tokens"])) < (
            best.get("_rank_value", 10**9),
            token_text(best.get("drift_tokens")),
        ):
            best = candidate
    if best is not None:
        best.pop("_rank_value", None)
    return {
        "found": bool(sources),
        "sources": sorted(sources),
        "source_bucket": source_bucket(sources),
        "best": best,
    }


def exact_tail_match(sample: dict[str, Any]) -> dict[str, Any]:
    sources = {source for source, present in sample.get("exact_truth_drift_sources", {}).items() if present}
    return {
        "found": bool(sample.get("exact_truth_drift_appears")),
        "sources": sorted(sources),
        "source_bucket": source_bucket(sources),
        "rank": sample.get("exact_truth_drift_rank"),
    }


def collision_diagnostics(rows: list[dict[str, Any]], polynomial_truths: list[list[str]]) -> dict[str, Any]:
    raw_unique: dict[tuple[str, ...], list[str]] = {}
    canonical_groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    examples: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tokens = row.get("drift_tokens", [])
        canonical = canonicalize(tokens)
        if canonical is None:
            continue
        raw_key = tuple(tokens)
        canonical_key = tuple(canonical)
        raw_unique[raw_key] = tokens
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
    collision_groups = {key: values for key, values in canonical_groups.items() if len(values) > 1}
    largest = sorted(collision_groups.items(), key=lambda item: (len(item[1]), token_text(list(item[0]))), reverse=True)[:8]
    nonconstant_merge = False
    for raw_values in collision_groups.values():
        signatures = {protected_signature(list(raw)) for raw in raw_values}
        if len(signatures) > 1:
            nonconstant_merge = True
            break
    unsafe_checks = {
        "pow2_x0_not_x0": canonicalize(["pow2", "x_0"]) != canonicalize(["x_0"]),
        "const_pow2_not_const_x0": canonicalize(["mul", "CONSTANT", "pow2", "x_0"])
        != canonicalize(["mul", "CONSTANT", "x_0"]),
        "sin_x0_not_x0": canonicalize(["sin", "x_0"]) != canonicalize(["x_0"]),
    }
    unsafe = (not all(unsafe_checks.values())) or nonconstant_merge
    linear_like = tuple(canonicalize(["mul", "CONSTANT", "x_0"]) or [])
    polynomial_collision = False
    for truth in polynomial_truths:
        canonical_truth = canonicalize(truth)
        if canonical_truth is not None and tuple(canonical_truth) == linear_like:
            polynomial_collision = True
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
        "nonconstant_structural_merge_detected": nonconstant_merge,
        "polynomial_like_truth_collides_with_linear_like": polynomial_collision,
        "unsafe_checks": unsafe_checks,
        "unsafe_collision_flag": unsafe,
    }


def summarize_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(row[field] for row in rows))


def analyze(expanded: dict[str, Any], drift_only: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    drift_by_index = {sample["sample_index"]: sample for sample in drift_only["samples"]}
    target_indices = [sample["sample_index"] for sample in drift_only["samples"]]
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    per_sample: list[dict[str, Any]] = []
    current_recovered: list[dict[str, Any]] = []
    exact_tail_recovered: list[dict[str, Any]] = []
    normalized_recovered: list[dict[str, Any]] = []
    all_rows_for_collision: list[dict[str, Any]] = []
    polynomial_truths: list[list[str]] = []

    for sample_idx in target_indices:
        expanded_sample = expanded_by_index[sample_idx]
        drift_sample = drift_by_index[sample_idx]
        truth_drift = expanded_sample["truth_drift"]
        truth_diffusion = expanded_sample["truth_diffusion"]
        canonical_truth = canonicalize(truth_drift)
        if drift_sample.get("drift_family") == "polynomial-like drift":
            polynomial_truths.append(truth_drift)
        exact_diffusion_present = bool(expanded_sample.get("has_diffusion"))
        current_rows = current_expanded_rows(expanded_sample)
        tail_rows = drift_only_rows(drift_sample)
        combined_rows = current_rows + tail_rows
        all_rows_for_collision.extend(combined_rows)

        current_match = canonical_match(current_rows, canonical_truth)
        tail_match = canonical_match(tail_rows, canonical_truth)
        combined_match = canonical_match(combined_rows, canonical_truth)
        exact_match = exact_tail_match(drift_sample)

        current_projected = bool(expanded_sample.get("has_full")) or (current_match["found"] and exact_diffusion_present)
        exact_tail_projected = bool(expanded_sample.get("has_full")) or (exact_match["found"] and exact_diffusion_present)
        normalized_projected = bool(expanded_sample.get("has_full")) or (combined_match["found"] and exact_diffusion_present)

        current_raw_count = int(expanded_sample.get("unique_drifts", 0))
        diffusion_raw_count = int(expanded_sample.get("unique_diffusions", 0))
        current_canonical_count = len(canonical_set(current_rows))
        tail_raw_count = len(raw_set(tail_rows))
        tail_canonical_count = len(canonical_set(tail_rows))
        combined_raw_estimate = current_raw_count + len(raw_set(tail_rows) - raw_set(current_rows))
        combined_canonical_count = len(canonical_set(combined_rows))
        pair_before_raw = current_raw_count * diffusion_raw_count
        pair_before_canonical = current_canonical_count * diffusion_raw_count
        pair_after_raw_estimate = combined_raw_estimate * diffusion_raw_count
        pair_after_canonical = combined_canonical_count * diffusion_raw_count

        row = {
            "sample_index": sample_idx,
            "truth_drift_tokens": truth_drift,
            "canonical_truth_drift_tokens": canonical_truth,
            "truth_diffusion_tokens": truth_diffusion,
            "drift_family": drift_sample.get("drift_family"),
            "exact_diffusion_present": exact_diffusion_present,
            "current_expanded_full_oracle": bool(expanded_sample.get("has_full")),
            "canonical_expanded_full_oracle": current_projected,
            "exact_tail_full_oracle": exact_tail_projected,
            "normalized_drift_only_admitted_full_oracle": normalized_projected,
            "current_expanded_source_bucket": current_match["source_bucket"],
            "drift_only_source_bucket": tail_match["source_bucket"],
            "combined_source_bucket": combined_match["source_bucket"],
            "drift_only_best_match": tail_match["best"],
            "current_raw_drift_count": current_raw_count,
            "current_canonical_drift_count_serialized": current_canonical_count,
            "drift_only_raw_admitted_count_serialized": tail_raw_count,
            "drift_only_canonical_admitted_count": tail_canonical_count,
            "combined_raw_drift_count_estimate": combined_raw_estimate,
            "combined_canonical_drift_count": combined_canonical_count,
            "diffusion_raw_count": diffusion_raw_count,
            "pair_count_before_raw": pair_before_raw,
            "pair_count_before_canonical_serialized": pair_before_canonical,
            "pair_count_after_raw_estimate": pair_after_raw_estimate,
            "pair_count_after_canonical": pair_after_canonical,
            "pair_count_canonical_increase_vs_raw_before": pair_after_canonical - pair_before_raw,
        }
        per_sample.append(row)
        if current_projected and not expanded_sample.get("has_full"):
            current_recovered.append(row)
        if exact_tail_projected and not expanded_sample.get("has_full"):
            exact_tail_recovered.append(row)
        if normalized_projected and not expanded_sample.get("has_full"):
            normalized_recovered.append(row)

    current_recovered_indices = {row["sample_index"] for row in current_recovered}
    normalized_recovered_indices = {row["sample_index"] for row in normalized_recovered}
    newly_beyond_current = [row for row in normalized_recovered if row["sample_index"] not in current_recovered_indices]
    remaining_missing = [row for row in per_sample if not row["normalized_drift_only_admitted_full_oracle"]]

    all_target_rows = []
    for sample_idx in target_indices:
        all_target_rows.extend(current_expanded_rows(expanded_by_index[sample_idx]))
        all_target_rows.extend(drift_only_rows(drift_by_index[sample_idx]))
    collision = collision_diagnostics(all_target_rows, polynomial_truths)

    before_pair_raw_sum = sum(row["pair_count_before_raw"] for row in per_sample)
    after_pair_canonical_sum = sum(row["pair_count_after_canonical"] for row in per_sample)
    after_pair_raw_sum = sum(row["pair_count_after_raw_estimate"] for row in per_sample)
    pair_canonical_increase = after_pair_canonical_sum - before_pair_raw_sum
    pair_canonical_increase_pct = 0.0 if before_pair_raw_sum == 0 else 100.0 * pair_canonical_increase / before_pair_raw_sum
    raw_tail_sum = sum(row["drift_only_raw_admitted_count_serialized"] for row in per_sample)
    canonical_tail_sum = sum(row["drift_only_canonical_admitted_count"] for row in per_sample)
    tail_dedup_pct = 0.0 if raw_tail_sum == 0 else 100.0 * (raw_tail_sum - canonical_tail_sum) / raw_tail_sum

    if collision["unsafe_collision_flag"]:
        decision = {
            "choice": "C",
            "label": "Unsafe collision risk",
            "recommended_next_action": (
                "Do not proceed; run more conservative family-specific constant-folding diagnostics."
            ),
        }
    elif pair_canonical_increase_pct > 100.0:
        decision = {
            "choice": "B",
            "label": "Normalized admission is coverage-positive but pair pressure is high",
            "recommended_next_action": (
                "Use a smaller coverage-only admission rule, such as one canonical representative per "
                "drift family/source/rank bucket or a per-sample cap. No formal eval."
            ),
        }
    else:
        decision = {
            "choice": "A",
            "label": "Normalized admission is coverage-positive and low-pressure",
            "recommended_next_action": (
                "Implement a small diagnostic hook for normalized candidate admission in candidate coverage only. "
                "No formal eval."
            ),
        }

    return {
        "metadata": {
            "normalization_reference_summary": normalization.get("summary", {}),
            "note": (
                "Current expanded-pool canonical counts use serialized raw drift rows available in the coverage JSON, "
                "primarily drift_nearest_detailed. Raw expanded counts use unique_drifts metadata."
            ),
        },
        "summary": {
            "total_samples_analyzed": len(expanded["samples"]),
            "target_oracle_absent_samples_analyzed": len(target_indices),
            "current_expanded_full_oracle_coverage": current_full,
            "canonicalized_expanded_pool_coverage": current_full + len(current_recovered),
            "canonicalized_expanded_pool_gain": len(current_recovered),
            "exact_tail_admission_coverage": current_full + len(exact_tail_recovered),
            "exact_tail_admission_gain": len(exact_tail_recovered),
            "normalized_drift_only_admission_coverage": current_full + len(normalized_recovered),
            "normalized_drift_only_admission_gain": len(normalized_recovered),
            "newly_recovered_beyond_current_expanded_canonical_pool": [
                row["sample_index"] for row in newly_beyond_current
            ],
            "remaining_missing_samples": [row["sample_index"] for row in remaining_missing],
            "exact_diffusion_present_for_recovered_samples": sum(
                1 for row in normalized_recovered if row["exact_diffusion_present"]
            ),
            "sampling_contributes_recovered_canonical_drift": any(
                "sampling" in row["drift_only_source_bucket"] for row in normalized_recovered
            ),
        },
        "source_breakdown": {
            "current_expanded_canonical": summarize_counts(current_recovered, "current_expanded_source_bucket"),
            "drift_only_tail_normalized": summarize_counts(normalized_recovered, "drift_only_source_bucket"),
            "newly_recovered_beyond_current": summarize_counts(newly_beyond_current, "drift_only_source_bucket"),
        },
        "family_breakdown": {
            "normalized_recovered": {family: Counter(row["drift_family"] for row in normalized_recovered).get(family, 0) for family in FAMILY_ORDER},
            "newly_recovered_beyond_current": {family: Counter(row["drift_family"] for row in newly_beyond_current).get(family, 0) for family in FAMILY_ORDER},
            "remaining_missing": {family: Counter(row["drift_family"] for row in remaining_missing).get(family, 0) for family in FAMILY_ORDER},
        },
        "pair_pressure": {
            "raw_expanded_drift_count_sum_targets": sum(row["current_raw_drift_count"] for row in per_sample),
            "canonical_expanded_drift_count_sum_serialized_targets": sum(
                row["current_canonical_drift_count_serialized"] for row in per_sample
            ),
            "raw_drift_only_admitted_count_sum_serialized": raw_tail_sum,
            "canonical_drift_only_admitted_count_sum": canonical_tail_sum,
            "combined_raw_drift_count_sum_estimate": sum(row["combined_raw_drift_count_estimate"] for row in per_sample),
            "combined_canonical_drift_count_sum": sum(row["combined_canonical_drift_count"] for row in per_sample),
            "pair_count_before_raw_sum": before_pair_raw_sum,
            "pair_count_after_raw_estimate_sum": after_pair_raw_sum,
            "pair_count_after_canonical_sum": after_pair_canonical_sum,
            "pair_count_canonical_increase": pair_canonical_increase,
            "pair_count_canonical_increase_percent": pair_canonical_increase_pct,
            "tail_canonical_dedup_reduction_percent": tail_dedup_pct,
            "top_pair_count_increases": sorted(
                [
                    {
                        "sample_index": row["sample_index"],
                        "increase": row["pair_count_canonical_increase_vs_raw_before"],
                        "before_raw": row["pair_count_before_raw"],
                        "after_canonical": row["pair_count_after_canonical"],
                    }
                    for row in per_sample
                ],
                key=lambda item: item["increase"],
                reverse=True,
            )[:8],
        },
        "collision_safety": collision,
        "samples": per_sample,
        "decision": decision,
    }


def count_text(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    pressure = payload["pair_pressure"]
    collision = payload["collision_safety"]
    decision = payload["decision"]
    lines = [
        "# Normalized Drift-Only Admission Coverage Diagnostics",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline JSON-only coverage diagnostic for a candidate pool equal to the current expanded-pairing pool plus normalized drift-only tail candidates for the 17 oracle-absent samples. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.",
        "",
        "## Executive Summary",
        "",
        f"- Total samples analyzed: `{summary['total_samples_analyzed']}`.",
        f"- Target oracle-absent samples analyzed: `{summary['target_oracle_absent_samples_analyzed']}`.",
        f"- Current expanded full oracle coverage: `{summary['current_expanded_full_oracle_coverage']}/32`.",
        f"- Canonicalized expanded-pool coverage: `{summary['canonicalized_expanded_pool_coverage']}/32` (`+{summary['canonicalized_expanded_pool_gain']}`).",
        f"- Exact-tail admission coverage: `{summary['exact_tail_admission_coverage']}/32` (`+{summary['exact_tail_admission_gain']}`).",
        f"- Normalized drift-only admission coverage: `{summary['normalized_drift_only_admission_coverage']}/32` (`+{summary['normalized_drift_only_admission_gain']}`).",
        f"- Newly recovered beyond current expanded canonical pool: `{', '.join(str(i) for i in summary['newly_recovered_beyond_current_expanded_canonical_pool'])}`.",
        f"- Remaining missing samples: `{', '.join(str(i) for i in summary['remaining_missing_samples'])}`.",
        f"- Drift-only normalized source breakdown: {count_text(payload['source_breakdown']['drift_only_tail_normalized'])}.",
        f"- Pair count estimate, raw current before vs canonical after admission: `{pressure['pair_count_before_raw_sum']}` -> `{pressure['pair_count_after_canonical_sum']}` (`+{pressure['pair_count_canonical_increase']}`, `{pressure['pair_count_canonical_increase_percent']:.1f}%`).",
        f"- Collision unsafe flag: `{collision['unsafe_collision_flag']}`.",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Coverage",
        "",
        "| Mode | Coverage | Gain | Samples | Source breakdown |",
        "| --- | ---: | ---: | --- | --- |",
        f"| Current expanded pool | {summary['current_expanded_full_oracle_coverage']}/32 | - | - | - |",
        f"| Current expanded pool + canonicalized drift | {summary['canonicalized_expanded_pool_coverage']}/32 | +{summary['canonicalized_expanded_pool_gain']} | - | {count_text(payload['source_breakdown']['current_expanded_canonical'])} |",
        f"| Exact drift-only tail admission | {summary['exact_tail_admission_coverage']}/32 | +{summary['exact_tail_admission_gain']} | - | beam=3 |",
        f"| Normalized drift-only admission | {summary['normalized_drift_only_admission_coverage']}/32 | +{summary['normalized_drift_only_admission_gain']} | `{', '.join(str(row['sample_index']) for row in payload['samples'] if row['normalized_drift_only_admitted_full_oracle'])}` | {count_text(payload['source_breakdown']['drift_only_tail_normalized'])} |",
        "",
        "All recovered normalized-admission samples already have exact diffusion present in the expanded coverage JSON.",
        "",
        "## Family Breakdown",
        "",
        "| Family | Normalized recovered | Newly beyond current canonical | Remaining missing |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in FAMILY_ORDER:
        lines.append(
            f"| {family} | {payload['family_breakdown']['normalized_recovered'][family]} | "
            f"{payload['family_breakdown']['newly_recovered_beyond_current'][family]} | "
            f"{payload['family_breakdown']['remaining_missing'][family]} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Table",
            "",
            "| Sample | Truth drift | Canonical truth | Exact diffusion | Current full | Current canonical full | Normalized admitted full | Drift-only source | Drift counts before/after | Pair counts before/after |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["samples"]:
        lines.append(
            f"| {row['sample_index']} | `{token_text(row['truth_drift_tokens'])}` | "
            f"`{token_text(row['canonical_truth_drift_tokens'])}` | "
            f"{row['exact_diffusion_present']} | {row['current_expanded_full_oracle']} | "
            f"{row['canonical_expanded_full_oracle']} | {row['normalized_drift_only_admitted_full_oracle']} | "
            f"{row['drift_only_source_bucket']} | "
            f"{row['current_raw_drift_count']}/{row['current_canonical_drift_count_serialized']} -> "
            f"{row['combined_raw_drift_count_estimate']}/{row['combined_canonical_drift_count']} | "
            f"{row['pair_count_before_raw']} -> {row['pair_count_after_canonical']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate / Pair Pressure",
            "",
            f"- Raw expanded drift count, target sum: `{pressure['raw_expanded_drift_count_sum_targets']}`.",
            f"- Canonical expanded drift count from serialized current rows, target sum: `{pressure['canonical_expanded_drift_count_sum_serialized_targets']}`.",
            f"- Raw drift-only admitted count from serialized tail rows: `{pressure['raw_drift_only_admitted_count_sum_serialized']}`.",
            f"- Canonical drift-only admitted count: `{pressure['canonical_drift_only_admitted_count_sum']}`.",
            f"- Combined raw drift count estimate: `{pressure['combined_raw_drift_count_sum_estimate']}`.",
            f"- Combined canonical drift count: `{pressure['combined_canonical_drift_count_sum']}`.",
            f"- Pair count before, raw expanded drift x raw diffusion: `{pressure['pair_count_before_raw_sum']}`.",
            f"- Pair count after, raw estimate: `{pressure['pair_count_after_raw_estimate_sum']}`.",
            f"- Pair count after, canonical drift x raw diffusion: `{pressure['pair_count_after_canonical_sum']}`.",
            f"- Drift-only tail canonical dedup reduction: `{pressure['tail_canonical_dedup_reduction_percent']:.1f}%`.",
            "",
            "Top samples by canonical pair-count increase:",
            "",
            "| Sample | Before raw pairs | After canonical pairs | Increase |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pressure["top_pair_count_increases"]:
        lines.append(f"| {row['sample_index']} | {row['before_raw']} | {row['after_canonical']} | {row['increase']} |")
    lines.extend(
        [
            "",
            "Canonicalization deduplicates many over-nested templates, but normalized admission still increases pair pressure substantially relative to the current expanded pool.",
            "",
            "## Collision / Overmerge Safety",
            "",
            f"- Unsafe collision flag: `{collision['unsafe_collision_flag']}`.",
            f"- Collision groups: `{collision['collision_group_count']}`.",
            f"- Any nonconstant structural merge detected: `{collision['nonconstant_structural_merge_detected']}`.",
            f"- Polynomial-like truth collides with linear-like candidates: `{collision['polynomial_like_truth_collides_with_linear_like']}`.",
            f"- Required safety checks: `{json.dumps(collision['unsafe_checks'], sort_keys=True)}`.",
            "",
            "Largest collision groups:",
            "",
            "| Canonical drift | Raw count | Examples |",
            "| --- | ---: | --- |",
        ]
    )
    for group in collision["largest_collision_groups"][:5]:
        examples = "<br>".join(f"`{token_text(tokens)}`" for tokens in group["raw_examples"][:4])
        lines.append(f"| `{token_text(group['canonical_drift_tokens'])}` | {group['raw_count']} | {examples} |")
    lines.extend(
        [
            "",
            "Unresolved risk: this is token-structure-only validation, not semantic fingerprint or selected-rerank validation.",
            "",
            "## Decision",
            "",
            f"`{decision['choice']}. {decision['label']}`: {decision['recommended_next_action']}",
            "",
            "No formal eval is recommended.",
            "",
            "## Limitations",
            "",
            "- Current expanded-pool canonical counts use serialized raw drift rows available in `candidate_coverage_pair_expanded.json`, primarily `drift_nearest_detailed`; the full raw expanded candidate list is not serialized.",
            "- Diffusion spans are not canonicalized; pair estimates use the raw diffusion count from the expanded coverage JSON.",
            "- The diagnostic simulates oracle coverage/admission only and does not evaluate ranking or selected recovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expanded = load_json(args.expanded_coverage_json)
    drift_only = load_json(args.drift_only_json)
    normalization = load_json(args.normalization_json)
    payload = analyze(expanded, drift_only, normalization)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    summary = payload["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"current_expanded_full_oracle={summary['current_expanded_full_oracle_coverage']}")
    print(f"canonicalized_expanded_pool_coverage={summary['canonicalized_expanded_pool_coverage']}")
    print(f"exact_tail_admission_coverage={summary['exact_tail_admission_coverage']}")
    print(f"normalized_drift_only_admission_coverage={summary['normalized_drift_only_admission_coverage']}")
    print(f"unsafe={payload['collision_safety']['unsafe_collision_flag']}")
    print(f"decision={payload['decision']['choice']}")


if __name__ == "__main__":
    main()
