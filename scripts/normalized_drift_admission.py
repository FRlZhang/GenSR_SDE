"""Reusable coverage-only helpers for normalized drift admission.

The functions in this module are diagnostic-only. They canonicalize narrow
multiplicative constant chains and select a capped set of drift-only candidates
for coverage analysis; they do not run decoding, reranking, scoring, or change
production candidate-generation defaults.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


DRIFT_FAMILIES = [
    "linear drift",
    "sin drift",
    "nested-mul linear drift",
    "polynomial-like drift",
    "other",
]


@dataclass(frozen=True)
class Node:
    kind: str
    children: tuple["Node", ...] = ()


CONST = Node("CONSTANT")


def _tokens(tokens_or_string: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if tokens_or_string is None:
        return []
    if isinstance(tokens_or_string, str):
        return tokens_or_string.split()
    return list(tokens_or_string)


def token_text(tokens: list[str] | tuple[str, ...] | None) -> str:
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


def canonicalize_constant_chains(tokens_or_string: list[str] | tuple[str, ...] | str | None) -> list[str] | None:
    """Fold only multiplicative chains of CONSTANT factors."""

    tokens = _tokens(tokens_or_string)
    node, pos = parse_prefix(tokens)
    if node is None or pos != len(tokens):
        return None
    return emit_prefix(canonical_node(node))


def classify_drift_family(canonical_tokens_or_string: list[str] | tuple[str, ...] | str | None) -> str:
    tokens = _tokens(canonical_tokens_or_string)
    if any(token in {"pow2", "pow3", "pow"} for token in tokens):
        return "polynomial-like drift"
    if "sin" in tokens:
        return "sin drift"
    if tokens in (["mul", "CONSTANT", "x_0"], ["x_0"]):
        return "linear drift"
    return "other"


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
    if sources == {"pair"}:
        return "pair"
    if "beam" in sources and "sampling" in sources:
        return "both"
    return "+".join(sorted(sources))


def current_expanded_drift_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
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


def canonical_drift_set(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for row in rows:
        canonical = row.get("canonical_drift_tokens") or canonicalize_constant_chains(row.get("drift_tokens", []))
        if canonical is not None:
            result.add(tuple(canonical))
    return result


def drift_only_candidate_rows(sample: dict[str, Any], include_sampling: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    buckets = ["nearest_drift_only_candidate", "exact_truth_drift_occurrences", "beam_candidates"]
    if include_sampling:
        buckets.append("sampling_candidates")
    for bucket in buckets:
        source_rows = sample.get(bucket, []) if bucket != "nearest_drift_only_candidate" else [sample.get(bucket)]
        for row in source_rows or []:
            if not row or not row.get("drift_tokens"):
                continue
            if row.get("source") == "sampling" and not include_sampling:
                continue
            key = tuple(row["drift_tokens"])
            if key in seen:
                continue
            seen.add(key)
            canonical = canonicalize_constant_chains(row["drift_tokens"])
            if canonical is None:
                continue
            rows.append(
                {
                    **row,
                    "sample_index": sample["sample_index"],
                    "candidate_bucket": bucket,
                    "raw_drift_tokens": row["drift_tokens"],
                    "canonical_drift_tokens": canonical,
                    "canonical_key": tuple(canonical),
                    "family": classify_drift_family(canonical),
                    "admitted": False,
                    "filter_reason": None,
                }
            )
    return rows


def select_p2_one_per_family(
    drift_candidates: list[dict[str, Any]],
    admission_source: str = "beam",
    include_sampling: bool = False,
) -> dict[str, Any]:
    """Select one best canonical representative per drift family.

    Returns both admitted and filtered rows with diagnostic metadata.
    """

    annotated = []
    best_by_family: dict[str, dict[str, Any]] = {}
    for candidate in drift_candidates:
        row = dict(candidate)
        source = row.get("source")
        if source == "sampling" and not include_sampling:
            row["filter_reason"] = "sampling_excluded"
            annotated.append(row)
            continue
        if admission_source and admission_source != "all" and source != admission_source:
            row["filter_reason"] = f"source_not_{admission_source}"
            annotated.append(row)
            continue
        row["filter_reason"] = "eligible"
        annotated.append(row)
        family = row["family"]
        if family not in best_by_family or source_order(row) < source_order(best_by_family[family]):
            best_by_family[family] = row

    admitted_keys = {id(row) for row in best_by_family.values()}
    admitted = []
    filtered = []
    for row in annotated:
        out = dict(row)
        if id(row) in admitted_keys:
            out["admitted"] = True
            out["filter_reason"] = "admitted_best_family_representative"
            admitted.append(out)
        elif out["filter_reason"] == "eligible":
            out["filter_reason"] = "lower_rank_same_family"
            filtered.append(out)
        else:
            filtered.append(out)
    admitted.sort(key=source_order)
    filtered.sort(key=lambda row: (row.get("sample_index", -1), row.get("family", ""), source_order(row)))
    return {"admitted": admitted, "filtered": filtered, "all": admitted + filtered}


def build_p2_normalized_admission_pool(
    expanded_coverage_json: dict[str, Any],
    drift_only_json: dict[str, Any],
    admission_source: str = "beam",
    include_sampling: bool = False,
) -> dict[str, Any]:
    expanded_by_index = {sample["sample_index"]: sample for sample in expanded_coverage_json["samples"]}
    samples = []
    for drift_sample in drift_only_json["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        current_rows = current_expanded_drift_rows(expanded_sample)
        drift_rows = drift_only_candidate_rows(drift_sample, include_sampling=include_sampling)
        selected = select_p2_one_per_family(
            drift_rows,
            admission_source=admission_source,
            include_sampling=include_sampling,
        )
        current_keys = canonical_drift_set(current_rows)
        admitted_keys = canonical_drift_set(selected["admitted"])
        truth_canonical = canonicalize_constant_chains(expanded_sample["truth_drift"])
        truth_key = tuple(truth_canonical or [])
        current_canonical_full = bool(expanded_sample.get("has_full")) or (
            bool(expanded_sample.get("has_diffusion")) and truth_key in current_keys
        )
        p2_full = current_canonical_full or (bool(expanded_sample.get("has_diffusion")) and truth_key in admitted_keys)
        samples.append(
            {
                "sample_index": sample_idx,
                "truth_drift_tokens": expanded_sample["truth_drift"],
                "canonical_truth_drift_tokens": truth_canonical,
                "truth_diffusion_tokens": expanded_sample["truth_diffusion"],
                "drift_family": drift_sample.get("drift_family") or expanded_sample.get("truth_drift_family"),
                "exact_diffusion_present": bool(expanded_sample.get("has_diffusion")),
                "current_expanded_full_oracle": bool(expanded_sample.get("has_full")),
                "current_canonical_full_oracle": current_canonical_full,
                "p2_normalized_full_oracle": p2_full,
                "current_canonical_drift_count": len(current_keys),
                "p2_admitted_canonical_drift_count": len(admitted_keys),
                "combined_canonical_drift_count": len(current_keys | admitted_keys),
                "diffusion_count": int(expanded_sample.get("unique_diffusions", 0)),
                "pair_count": len(current_keys | admitted_keys) * int(expanded_sample.get("unique_diffusions", 0)),
                "admitted_candidates": selected["admitted"],
                "filtered_candidates": selected["filtered"],
            }
        )
    return {"samples": samples}


def safety_checks() -> dict[str, bool]:
    return {
        "pow2_x0_not_x0": canonicalize_constant_chains("pow2 x_0") != canonicalize_constant_chains("x_0"),
        "const_pow2_not_const_x0": canonicalize_constant_chains("mul CONSTANT pow2 x_0")
        != canonicalize_constant_chains("mul CONSTANT x_0"),
        "sin_x0_not_x0": canonicalize_constant_chains("sin x_0") != canonicalize_constant_chains("x_0"),
    }


def collision_summary(rows: list[dict[str, Any]], polynomial_truths: list[list[str]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        tokens = row.get("drift_tokens") or row.get("raw_drift_tokens") or []
        canonical = canonicalize_constant_chains(tokens)
        if canonical is None:
            continue
        groups[tuple(canonical)].add(tuple(tokens))
    collision_groups = {key: values for key, values in groups.items() if len(values) > 1}
    nonconstant_merge = any(
        len({tuple(token for token in raw if token not in {"CONSTANT", "mul"}) for raw in values}) > 1
        for values in collision_groups.values()
    )
    linear_key = tuple(canonicalize_constant_chains("mul CONSTANT x_0") or [])
    polynomial_collision = any(tuple(canonicalize_constant_chains(tokens) or []) == linear_key for tokens in polynomial_truths)
    checks = safety_checks()
    largest = sorted(collision_groups.items(), key=lambda item: (len(item[1]), token_text(list(item[0]))), reverse=True)[:8]
    return {
        "unsafe_collision_flag": (not all(checks.values())) or nonconstant_merge or polynomial_collision,
        "nonconstant_structural_merge_detected": nonconstant_merge,
        "polynomial_like_truth_collides_with_linear_like": polynomial_collision,
        "safety_checks": checks,
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
