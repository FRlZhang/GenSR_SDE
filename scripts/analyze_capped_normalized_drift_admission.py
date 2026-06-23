#!/usr/bin/env python3
"""Capped normalized drift-only admission coverage diagnostic.

This helper is offline/JSON-only. It compares a small fixed set of capped
admission policies over existing drift-only tail JSON and expanded-pairing
coverage JSON. It does not run model decoding, candidate generation, fingerprint
simulation, reranking, scoring, formal eval, or training.
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
    "other",
]

POLICY_ORDER = [
    "P0_full_admission",
    "P1_one_per_canonical_form",
    "P2_one_per_family",
    "P3_top_1_canonical_per_sample",
    "P3_top_2_canonical_per_sample",
    "P3_top_3_canonical_per_sample",
    "P3_top_5_canonical_per_sample",
    "P4_recovery_targeted_oracle_upper_bound",
    "P5_rank_bucket_representative",
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
        "--normalized-admission-json",
        type=Path,
        default=Path("normalized_drift_only_admission_coverage_diagnostics.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("capped_normalized_drift_admission_diagnostics.md"))
    parser.add_argument("--json-output", type=Path, default=Path("capped_normalized_drift_admission_diagnostics.json"))
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


def source_order(row: dict[str, Any]) -> tuple[int, int, str]:
    source = row.get("source", "")
    source_rank = row.get("source_rank")
    rank = source_rank if source_rank is not None else 10**9
    source_priority = 0 if source == "beam" else 1 if source == "sampling" else 2
    return (source_priority, rank, token_text(row.get("drift_tokens")))


def source_bucket(sources: set[str]) -> str:
    if not sources:
        return "neither"
    if sources == {"beam"}:
        return "beam"
    if sources == {"sampling"}:
        return "sampling"
    if "beam" in sources and "sampling" in sources:
        return "both"
    return "+".join(sorted(sources))


def candidate_family(canonical_tokens: list[str]) -> str:
    if any(token in {"pow2", "pow3", "pow"} for token in canonical_tokens):
        return "polynomial-like drift"
    if "sin" in canonical_tokens:
        return "sin drift"
    if canonical_tokens == ["mul", "CONSTANT", "x_0"] or canonical_tokens == ["x_0"]:
        return "linear drift"
    return "other"


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
                "source": source_bucket(set(item.get("sources", []))),
                "sources": item.get("sources", []),
                "drift_tokens": tokens,
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
                    "source": source_bucket(set(sample.get("drift_sources", []))),
                    "sources": sample.get("drift_sources", []),
                    "drift_tokens": tokens,
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
                rows.append({**row, "candidate_bucket": bucket})
    for bucket in ("exact_truth_drift_occurrences", "beam_candidates", "sampling_candidates"):
        for row in sample.get(bucket, []) or []:
            tokens = row.get("drift_tokens", [])
            if not tokens:
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            rows.append({**row, "candidate_bucket": bucket})
    enriched: list[dict[str, Any]] = []
    for row in rows:
        canonical = canonicalize(row.get("drift_tokens", []))
        if canonical is None:
            continue
        enriched.append(
            {
                **row,
                "canonical_drift_tokens": canonical,
                "canonical_key": tuple(canonical),
                "candidate_family": candidate_family(canonical),
            }
        )
    return enriched


def best_by_canonical(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = row["canonical_key"]
        if key not in best or source_order(row) < source_order(best[key]):
            best[key] = row
    return best


def canonical_set(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for row in rows:
        canonical = row.get("canonical_drift_tokens") or canonicalize(row.get("drift_tokens", []))
        if canonical is not None:
            result.add(tuple(canonical))
    return result


def select_policy_rows(policy: str, rows: list[dict[str, Any]], truth_key: tuple[str, ...]) -> list[dict[str, Any]]:
    by_canonical = best_by_canonical(rows)
    representatives = sorted(by_canonical.values(), key=source_order)
    if policy == "P0_full_admission":
        return rows
    if policy == "P1_one_per_canonical_form":
        return representatives
    if policy == "P2_one_per_family":
        best_family: dict[str, dict[str, Any]] = {}
        for row in representatives:
            family = row["candidate_family"]
            if family not in best_family or source_order(row) < source_order(best_family[family]):
                best_family[family] = row
        return sorted(best_family.values(), key=source_order)
    if policy.startswith("P3_top_"):
        k = int(policy.split("_")[2])
        return representatives[:k]
    if policy == "P4_recovery_targeted_oracle_upper_bound":
        row = by_canonical.get(truth_key)
        return [row] if row is not None else []
    if policy == "P5_rank_bucket_representative":
        buckets = [(1, 8), (9, 16), (17, 24), (25, 32)]
        selected: list[dict[str, Any]] = []
        beam_reps = [row for row in representatives if row.get("source") == "beam" and row.get("source_rank") is not None]
        for low, high in buckets:
            bucket_rows = [row for row in beam_reps if low <= int(row["source_rank"]) <= high]
            if bucket_rows:
                selected.append(sorted(bucket_rows, key=source_order)[0])
        return selected
    raise ValueError(f"unknown policy: {policy}")


def current_canonical_projected(sample: dict[str, Any], current_rows: list[dict[str, Any]], truth_key: tuple[str, ...]) -> bool:
    if sample.get("has_full"):
        return True
    return bool(sample.get("has_diffusion")) and truth_key in canonical_set(current_rows)


def policy_source_for_truth(selected_rows: list[dict[str, Any]], truth_key: tuple[str, ...]) -> str:
    sources = {row.get("source", "") for row in selected_rows if row.get("canonical_key") == truth_key and row.get("source")}
    return source_bucket(sources)


def collision_safety(all_rows: list[dict[str, Any]], polynomial_truths: list[list[str]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    examples: dict[tuple[str, ...], list[list[str]]] = defaultdict(list)
    for row in all_rows:
        tokens = row.get("drift_tokens", [])
        canonical = canonicalize(tokens)
        if canonical is None:
            continue
        key = tuple(canonical)
        raw = tuple(tokens)
        groups[key].add(raw)
        if len(examples[key]) < 5:
            examples[key].append(tokens)
    collision_groups = {key: values for key, values in groups.items() if len(values) > 1}
    nonconstant_merge = any(
        len({protected_signature(list(raw)) for raw in values}) > 1
        for values in collision_groups.values()
    )
    linear_key = tuple(canonicalize(["mul", "CONSTANT", "x_0"]) or [])
    polynomial_collision = any(tuple(canonicalize(tokens) or []) == linear_key for tokens in polynomial_truths)
    unsafe_checks = {
        "pow2_x0_not_x0": canonicalize(["pow2", "x_0"]) != canonicalize(["x_0"]),
        "const_pow2_not_const_x0": canonicalize(["mul", "CONSTANT", "pow2", "x_0"])
        != canonicalize(["mul", "CONSTANT", "x_0"]),
        "sin_x0_not_x0": canonicalize(["sin", "x_0"]) != canonicalize(["x_0"]),
    }
    largest = sorted(collision_groups.items(), key=lambda item: (len(item[1]), token_text(list(item[0]))), reverse=True)[:8]
    return {
        "unsafe_collision_flag": (not all(unsafe_checks.values())) or nonconstant_merge or polynomial_collision,
        "nonconstant_structural_merge_detected": nonconstant_merge,
        "polynomial_like_truth_collides_with_linear_like": polynomial_collision,
        "unsafe_checks": unsafe_checks,
        "collision_group_count": len(collision_groups),
        "largest_collision_groups": [
            {
                "canonical_drift_tokens": list(key),
                "raw_count": len(values),
                "raw_examples": [list(raw) for raw in sorted(values)[:8]],
            }
            for key, values in largest
        ],
    }


def evaluate_policy(
    policy: str,
    samples: list[dict[str, Any]],
    current_full: int,
    full_pair_count: int,
) -> dict[str, Any]:
    recovered: list[dict[str, Any]] = []
    newly: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    per_sample: dict[int, dict[str, Any]] = {}
    source_counts = Counter()
    raw_admitted = 0
    canonical_admitted = 0
    pair_after = 0

    for sample in samples:
        selected_rows = select_policy_rows(policy, sample["tail_rows"], sample["truth_key"])
        selected_keys = canonical_set(selected_rows)
        current_keys = sample["current_keys"]
        combined_keys = current_keys | selected_keys
        tail_match = sample["truth_key"] in selected_keys
        projected = sample["current_canonical_full_oracle"] or (tail_match and sample["exact_diffusion_present"])
        pair_count = len(combined_keys) * sample["diffusion_raw_count"]
        pair_after += pair_count
        raw_admitted += len(selected_rows)
        canonical_admitted += len(selected_keys)
        source_bucket_value = policy_source_for_truth(selected_rows, sample["truth_key"])
        if projected:
            recovered.append(sample)
            if source_bucket_value != "neither":
                source_counts[source_bucket_value] += 1
            elif sample["current_canonical_full_oracle"]:
                source_counts["current_expanded"] += 1
        else:
            missing.append(sample)
        if projected and not sample["current_canonical_full_oracle"]:
            newly.append(sample)
        per_sample[sample["sample_index"]] = {
            "admitted_raw_count": len(selected_rows),
            "admitted_canonical_count": len(selected_keys),
            "pair_count": pair_count,
            "projected_full_oracle": projected,
            "source_bucket": source_bucket_value,
        }

    coverage = current_full + len(recovered)
    pair_reduction = full_pair_count - pair_after
    pair_reduction_pct = 0.0 if full_pair_count == 0 else 100.0 * pair_reduction / full_pair_count
    retained_pct = 0.0 if full_pair_count == 0 else 100.0 * coverage / 30.0
    recovered_count_pct = 0.0 if not recovered else 100.0 * len(recovered) / 15.0
    return {
        "policy": policy,
        "projected_full_oracle_coverage": coverage,
        "recovered_target_samples": [sample["sample_index"] for sample in recovered],
        "newly_recovered_beyond_current_expanded_canonical_pool": [sample["sample_index"] for sample in newly],
        "remaining_missing_samples": [sample["sample_index"] for sample in missing],
        "family_breakdown_recovered": {family: Counter(sample["drift_family"] for sample in recovered).get(family, 0) for family in FAMILY_ORDER},
        "source_breakdown": dict(source_counts),
        "admitted_raw_drift_count": raw_admitted,
        "admitted_canonical_drift_count": canonical_admitted,
        "pair_count_before": 370,
        "pair_count_after": pair_after,
        "pair_count_increase": pair_after - 370,
        "pair_count_increase_percent": 0.0 if 370 == 0 else 100.0 * (pair_after - 370) / 370.0,
        "pair_pressure_reduction_vs_full_admission": pair_reduction,
        "pair_pressure_reduction_vs_full_admission_percent": pair_reduction_pct,
        "coverage_retained_vs_full_admission_percent": retained_pct,
        "target_recovery_retained_vs_full_admission_percent": recovered_count_pct,
        "sampling_can_be_dropped": all(
            sample["truth_key"] not in {
                row.get("canonical_key")
                for row in select_policy_rows(policy, sample["tail_rows"], sample["truth_key"])
                if row.get("source") == "sampling"
            }
            for sample in samples
        ),
        "samples_16_28_remain_missing": all(idx in [sample["sample_index"] for sample in missing] for idx in (16, 28)),
        "per_sample": per_sample,
    }


def analyze(expanded: dict[str, Any], drift_only: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded["samples"]}
    target_indices = [sample["sample_index"] for sample in drift_only["samples"]]
    normalized_summary = normalized["summary"]
    normalized_recovered = set(
        normalized_summary.get(
            "canonicalized_drift_only_recovered_samples",
            [
                sample_idx
                for sample_idx in target_indices
                if sample_idx not in set(normalized_summary.get("remaining_missing_samples", []))
            ],
        )
    )
    current_full = sum(1 for sample in expanded["samples"] if sample.get("has_full"))
    prepared_samples: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    polynomial_truths: list[list[str]] = []

    for drift_sample in drift_only["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        truth = expanded_sample["truth_drift"]
        canonical_truth = canonicalize(truth)
        if canonical_truth is None:
            raise ValueError(f"could not canonicalize truth drift for sample {sample_idx}")
        current_rows = current_expanded_rows(expanded_sample)
        tail_rows = drift_only_rows(drift_sample)
        current_keys = canonical_set(current_rows)
        current_projected = current_canonical_projected(expanded_sample, current_rows, tuple(canonical_truth))
        if drift_sample.get("drift_family") == "polynomial-like drift":
            polynomial_truths.append(truth)
        all_rows.extend(current_rows)
        all_rows.extend(tail_rows)
        prepared_samples.append(
            {
                "sample_index": sample_idx,
                "truth_drift_tokens": truth,
                "canonical_truth_drift_tokens": canonical_truth,
                "truth_key": tuple(canonical_truth),
                "drift_family": drift_sample.get("drift_family"),
                "exact_diffusion_present": bool(expanded_sample.get("has_diffusion")),
                "current_expanded_full_oracle": bool(expanded_sample.get("has_full")),
                "current_canonical_full_oracle": current_projected,
                "full_normalized_admission_full_oracle": sample_idx in normalized_recovered,
                "current_rows": current_rows,
                "tail_rows": tail_rows,
                "current_keys": current_keys,
                "diffusion_raw_count": int(expanded_sample.get("unique_diffusions", 0)),
            }
        )

    full_pair_count = int(normalized["pair_pressure"]["pair_count_after_canonical_sum"])
    policies = {policy: evaluate_policy(policy, prepared_samples, current_full, full_pair_count) for policy in POLICY_ORDER}
    practical = [policies[key] for key in POLICY_ORDER if key != "P4_recovery_targeted_oracle_upper_bound"]
    practical_sorted = sorted(
        practical,
        key=lambda item: (
            -item["projected_full_oracle_coverage"],
            item["pair_count_after"],
            item["policy"],
        ),
    )
    best = practical_sorted[0]
    upper = policies["P4_recovery_targeted_oracle_upper_bound"]
    safety = collision_safety(all_rows, polynomial_truths)
    if safety["unsafe_collision_flag"]:
        decision = {
            "choice": "C",
            "label": "Unsafe collision risk",
            "recommended_next_action": (
                "Do not proceed; use more conservative family-specific constant folding diagnostics. "
                "No formal eval."
            ),
        }
    elif best["projected_full_oracle_coverage"] >= 29 and best["pair_count_after"] <= 650:
        decision = {
            "choice": "A",
            "label": "A capped practical rule preserves nearly all coverage with much lower pair pressure",
            "recommended_next_action": (
                f"Use `{best['policy']}` for a future small coverage-only implementation hook. "
                "No formal eval."
            ),
        }
    elif upper["pair_count_after"] < best["pair_count_after"] * 0.75 and best["projected_full_oracle_coverage"] < 29:
        decision = {
            "choice": "B",
            "label": "Only truth-aware upper bound is efficient",
            "recommended_next_action": "Find a non-truth-aware proxy for the upper bound. No formal eval.",
        }
    else:
        decision = {
            "choice": "C",
            "label": "All practical caps lose too much coverage",
            "recommended_next_action": (
                "Do not proceed to implementation; investigate why recovered drifts require broad admission. "
                "No formal eval."
            ),
        }

    per_sample_rows = []
    for sample in prepared_samples:
        row = {
            "sample_index": sample["sample_index"],
            "truth_drift_tokens": sample["truth_drift_tokens"],
            "canonical_truth_drift_tokens": sample["canonical_truth_drift_tokens"],
            "drift_family": sample["drift_family"],
            "exact_diffusion_present": sample["exact_diffusion_present"],
            "current_expanded_canonical_full_oracle": sample["current_canonical_full_oracle"],
            "full_normalized_admission_full_oracle": sample["full_normalized_admission_full_oracle"],
            "policy_status": {},
        }
        for policy in POLICY_ORDER:
            row["policy_status"][policy] = policies[policy]["per_sample"][sample["sample_index"]]
        per_sample_rows.append(row)

    return {
        "metadata": {
            "policy_order": POLICY_ORDER,
            "baseline": {
                "current_expanded_full_oracle_coverage": current_full,
                "canonicalized_expanded_pool_coverage": normalized["summary"]["canonicalized_expanded_pool_coverage"],
                "full_normalized_drift_only_admission_coverage": normalized["summary"]["normalized_drift_only_admission_coverage"],
                "full_normalized_admission_target_pair_count": full_pair_count,
                "pre_admission_target_pair_count": normalized["pair_pressure"]["pair_count_before_raw_sum"],
            },
        },
        "summary": {
            "total_samples_analyzed": len(expanded["samples"]),
            "target_oracle_absent_samples_analyzed": len(target_indices),
            "best_practical_policy": best["policy"],
            "truth_aware_upper_bound_policy": "P4_recovery_targeted_oracle_upper_bound",
            "sampling_can_be_dropped_for_all_policies": all(policy["sampling_can_be_dropped"] for policy in policies.values()),
        },
        "policies": policies,
        "best_practical_policy": best,
        "truth_aware_upper_bound": upper,
        "collision_safety": safety,
        "samples": per_sample_rows,
        "decision": decision,
    }


def counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload["metadata"]["baseline"]
    best = payload["best_practical_policy"]
    upper = payload["truth_aware_upper_bound"]
    decision = payload["decision"]
    safety = payload["collision_safety"]
    lines = [
        "# Capped Normalized Drift Admission Diagnostics",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: offline JSON-only comparison of fixed capped normalized drift-only admission policies. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.",
        "",
        "## Executive Summary",
        "",
        f"- Total samples analyzed: `{payload['summary']['total_samples_analyzed']}`.",
        f"- Target oracle-absent samples analyzed: `{payload['summary']['target_oracle_absent_samples_analyzed']}`.",
        f"- Baseline current expanded full oracle coverage: `{baseline['current_expanded_full_oracle_coverage']}/32`.",
        f"- Baseline canonicalized expanded-pool coverage: `{baseline['canonicalized_expanded_pool_coverage']}/32`.",
        f"- Baseline full normalized drift-only admission coverage: `{baseline['full_normalized_drift_only_admission_coverage']}/32`.",
        f"- Baseline full normalized admission target pair count: `{baseline['full_normalized_admission_target_pair_count']}`; pre-admission target pair count: `{baseline['pre_admission_target_pair_count']}`.",
        f"- Best practical policy: `{best['policy']}` with coverage `{best['projected_full_oracle_coverage']}/32` and pair count `{best['pair_count_after']}`.",
        f"- Truth-aware upper bound: coverage `{upper['projected_full_oracle_coverage']}/32` and pair count `{upper['pair_count_after']}`.",
        f"- Sampling can be dropped in this diagnostic: `{payload['summary']['sampling_can_be_dropped_for_all_policies']}`.",
        f"- Collision unsafe flag: `{safety['unsafe_collision_flag']}`.",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Coverage | Recovered targets | Newly beyond current canonical | Remaining missing | Admitted raw/canonical | Pair count | Pair reduction vs full | Coverage retained vs full | Sampling drop |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy in POLICY_ORDER:
        data = payload["policies"][policy]
        lines.append(
            f"| `{policy}` | {data['projected_full_oracle_coverage']}/32 | "
            f"`{','.join(str(i) for i in data['recovered_target_samples'])}` | "
            f"`{','.join(str(i) for i in data['newly_recovered_beyond_current_expanded_canonical_pool'])}` | "
            f"`{','.join(str(i) for i in data['remaining_missing_samples'])}` | "
            f"{data['admitted_raw_drift_count']}/{data['admitted_canonical_drift_count']} | "
            f"{data['pair_count_after']} | "
            f"{data['pair_pressure_reduction_vs_full_admission']} ({data['pair_pressure_reduction_vs_full_admission_percent']:.1f}%) | "
            f"{data['coverage_retained_vs_full_admission_percent']:.1f}% | "
            f"{data['sampling_can_be_dropped']} |"
        )
    lines.extend(
        [
            "",
            "## Best Practical Policy",
            "",
            f"`{best['policy']}` is the best practical policy under the priority order: it preserves `{best['projected_full_oracle_coverage']}/32` coverage, admits `{best['admitted_canonical_drift_count']}` canonical drift representatives, and reduces pair pressure by `{best['pair_pressure_reduction_vs_full_admission']}` pairs versus full normalized admission.",
            "",
            "Truth-aware upper bound comparison:",
            "",
            f"- Upper-bound policy: `{upper['policy']}`.",
            f"- Upper-bound coverage: `{upper['projected_full_oracle_coverage']}/32`.",
            f"- Upper-bound pair count: `{upper['pair_count_after']}`.",
            f"- Best practical extra pair pressure over upper bound: `{best['pair_count_after'] - upper['pair_count_after']}`.",
            "",
            "## Family And Source Breakdown",
            "",
            "| Policy | Source breakdown | Linear | Sin | Nested-mul linear | Polynomial-like | Other |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICY_ORDER:
        data = payload["policies"][policy]
        fam = data["family_breakdown_recovered"]
        lines.append(
            f"| `{policy}` | {counts_text(data['source_breakdown'])} | "
            f"{fam['linear drift']} | {fam['sin drift']} | {fam['nested-mul linear drift']} | "
            f"{fam['polynomial-like drift']} | {fam['other']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Table",
            "",
            "| Sample | Truth drift | Canonical truth | Exact diffusion | Current canonical full | Full normalized full | P1 | P2 | P3 k=1/2/3/5 | P4 upper | P5 | Pair counts P1/P2/P3-5/P4/P5 |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["samples"]:
        status = row["policy_status"]
        p3 = "/".join(
            "Y" if status[f"P3_top_{k}_canonical_per_sample"]["projected_full_oracle"] else "N"
            for k in (1, 2, 3, 5)
        )
        pair_counts = (
            f"{status['P1_one_per_canonical_form']['pair_count']}/"
            f"{status['P2_one_per_family']['pair_count']}/"
            f"{status['P3_top_5_canonical_per_sample']['pair_count']}/"
            f"{status['P4_recovery_targeted_oracle_upper_bound']['pair_count']}/"
            f"{status['P5_rank_bucket_representative']['pair_count']}"
        )
        lines.append(
            f"| {row['sample_index']} | `{token_text(row['truth_drift_tokens'])}` | "
            f"`{token_text(row['canonical_truth_drift_tokens'])}` | "
            f"{row['exact_diffusion_present']} | {row['current_expanded_canonical_full_oracle']} | "
            f"{row['full_normalized_admission_full_oracle']} | "
            f"{'Y' if status['P1_one_per_canonical_form']['projected_full_oracle'] else 'N'} | "
            f"{'Y' if status['P2_one_per_family']['projected_full_oracle'] else 'N'} | "
            f"{p3} | "
            f"{'Y' if status['P4_recovery_targeted_oracle_upper_bound']['projected_full_oracle'] else 'N'} | "
            f"{'Y' if status['P5_rank_bucket_representative']['projected_full_oracle'] else 'N'} | "
            f"{pair_counts} |"
        )
    lines.extend(
        [
            "",
            "## Collision / Overmerge Safety",
            "",
            f"- Unsafe collision flag: `{safety['unsafe_collision_flag']}`.",
            f"- Nonconstant structural merge detected: `{safety['nonconstant_structural_merge_detected']}`.",
            f"- Polynomial-like truth collides with linear-like candidates: `{safety['polynomial_like_truth_collides_with_linear_like']}`.",
            f"- Required safety checks: `{json.dumps(safety['unsafe_checks'], sort_keys=True)}`.",
            f"- Collision groups: `{safety['collision_group_count']}`.",
            "",
            "| Canonical drift | Raw count | Examples |",
            "| --- | ---: | --- |",
        ]
    )
    for group in safety["largest_collision_groups"][:5]:
        examples = "<br>".join(f"`{token_text(tokens)}`" for tokens in group["raw_examples"][:4])
        lines.append(f"| `{token_text(group['canonical_drift_tokens'])}` | {group['raw_count']} | {examples} |")
    lines.extend(
        [
            "",
            "This is token-structure-only validation, not semantic fingerprint or selected-rerank validation.",
            "",
            "## Decision",
            "",
            f"`{decision['choice']}. {decision['label']}`: {decision['recommended_next_action']}",
            "",
            "No formal eval is recommended.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expanded = load_json(args.expanded_coverage_json)
    drift_only = load_json(args.drift_only_json)
    normalized = load_json(args.normalized_admission_json)
    payload = analyze(expanded, drift_only, normalized)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    best = payload["best_practical_policy"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"best_practical_policy={best['policy']}")
    print(f"best_policy_coverage={best['projected_full_oracle_coverage']}")
    print(f"best_policy_pair_count={best['pair_count_after']}")
    print(f"truth_upper_pair_count={payload['truth_aware_upper_bound']['pair_count_after']}")
    print(f"unsafe={payload['collision_safety']['unsafe_collision_flag']}")
    print(f"decision={payload['decision']['choice']}")


if __name__ == "__main__":
    main()
