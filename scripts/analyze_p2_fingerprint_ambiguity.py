#!/usr/bin/env python3
"""Offline fingerprint ambiguity diagnostic for V0/V4 changed cases."""

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

from analyze_p2_admitted_candidate_scoring import json_ready, score_seed, target_array
from analyze_p2_calibration_all32_smoke import (
    ACTIVE_SLICE,
    CONSTANT_VALUES,
    GLOBAL_ACTIVE_DIMS,
    score_candidate,
)
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system, split_sde_tokens


V0 = "V0_current_rolewise_no_multi_u0"
V4 = "V4_active_without_dims_76_72_82_88"
REMOVED_ACTIVE_DIMS = [76, 72, 82, 88]
FINGERPRINT_CONFIG = FingerprintConfig(n_paths=800, n_steps=60, active_paths=800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--all32-smoke-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--p2-coverage-json", type=Path, required=True)
    parser.add_argument("--baseline-coverage-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def token_text(tokens: list[str] | None) -> str:
    return " ".join(tokens or [])


def sample_by_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in payload.get("samples", [])}


def candidate_lookup(sample: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    return {
        str(row["candidate_id"]): (idx, row)
        for idx, row in enumerate(sample.get("candidates", []))
    }


def l2_without(residual: list[float] | np.ndarray | None, remove_global: list[int]) -> float | None:
    if residual is None:
        return None
    vec = np.asarray(residual, dtype=np.float64)
    remove_local = {
        dim - ACTIVE_SLICE.start
        for dim in remove_global
        if 0 <= dim - ACTIVE_SLICE.start < len(vec)
    }
    keep = [idx for idx in range(len(vec)) if idx not in remove_local]
    if not keep:
        return 0.0
    return float(np.linalg.norm(vec[keep]))


def finite(value: Any) -> bool:
    return value is not None and np.isfinite(float(value))


def safe_sub(left: Any, right: Any) -> float | None:
    if not finite(left) or not finite(right):
        return None
    return float(left) - float(right)


def safe_div(num: Any, den: Any) -> float | None:
    if not finite(num) or not finite(den) or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def v4_total(scored: dict[str, Any]) -> float | None:
    active = scored.get("active_score_after_removed_dims")
    weak = scored.get("weak_score")
    return None if active is None or weak is None else float(active) + float(weak)


def score_candidate_row(
    sample: dict[str, Any],
    row: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    scored = score_candidate(sample, row, candidate_index)
    scored["active_score_after_removed_dims"] = l2_without(
        scored.get("active_residual"),
        GLOBAL_ACTIVE_DIMS,
    )
    scored["v4_score"] = v4_total(scored)
    return scored


def score_summary(scored: dict[str, Any] | None, variant: str) -> dict[str, Any] | None:
    if scored is None:
        return None
    if variant == V0:
        total = scored.get("score")
        active = scored.get("active_score")
    elif variant == V4:
        total = scored.get("v4_score")
        active = scored.get("active_score_after_removed_dims")
    else:
        raise ValueError(f"unknown variant: {variant}")
    return {
        "candidate_id": scored.get("candidate_id"),
        "candidate_source": scored.get("candidate_source"),
        "total_score": total,
        "active_score": active,
        "weak_score": scored.get("weak_score"),
        "best_drift_constant": scored.get("best_drift_constant"),
        "best_diffusion_constant": scored.get("best_diffusion_constant"),
        "valid_bool": scored.get("valid_bool"),
        "failure_reason": scored.get("failure_reason"),
    }


def candidate_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_id": row.get("candidate_id"),
        "candidate_source": row.get("candidate_source"),
        "sequence": row.get("full_sequence_tokens"),
        "drift_tokens": row.get("drift_tokens_canonical") or row.get("drift_tokens_raw"),
        "diffusion_tokens": row.get("diffusion_tokens_canonical") or row.get("diffusion_tokens_raw"),
        "drift_source": row.get("drift_source"),
        "diffusion_source": row.get("diffusion_source"),
        "drift_rank": row.get("drift_rank"),
        "diffusion_rank": row.get("diffusion_rank"),
        "pair_rank": row.get("pair_rank"),
        "is_current_expanded_candidate": row.get("is_current_expanded_candidate"),
        "is_p2_canonical_oracle": row.get("is_p2_canonical_oracle"),
        "is_truth_drift_canonical": row.get("is_truth_drift_canonical"),
        "is_truth_diffusion_exact": row.get("is_truth_diffusion_exact"),
    }


def exact_oracle_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    truth = sample.get("truth_sequence") or []
    return [
        row for row in sample.get("candidates", [])
        if row.get("full_sequence_tokens") == truth
    ]


def p2_oracle_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in sample.get("candidates", [])
        if row.get("is_p2_canonical_oracle")
        or (row.get("is_truth_drift_canonical") and row.get("is_truth_diffusion_exact"))
    ]


def current_expanded_oracle_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    truth = sample.get("truth_sequence") or []
    return [
        row for row in sample.get("candidates", [])
        if row.get("is_current_expanded_candidate")
        and (
            row.get("full_sequence_tokens") == truth
            or (row.get("is_truth_drift_canonical") and row.get("is_truth_diffusion_exact"))
        )
    ]


def best_scored_row(
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    lookup: dict[str, tuple[int, dict[str, Any]]],
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    best_row = None
    best_score = None
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in cache:
            candidate_index = lookup[candidate_id][0]
            cache[candidate_id] = score_candidate_row(sample, row, candidate_index)
        scored = cache[candidate_id]
        score = scored.get("score")
        if finite(score) and (best_score is None or float(score) < best_score):
            best_score = float(score)
            best_row = row
    return best_row, cache.get(str(best_row.get("candidate_id"))) if best_row else None


def split_roles(tokens: list[str] | None) -> tuple[list[str], list[str]]:
    split = split_sde_tokens(tokens or [])
    if split is None:
        return [], []
    return split


def drift_family(tokens: list[str] | None) -> str:
    toks = tokens or []
    text = token_text(toks)
    if not toks:
        return "unknown"
    if "sin" in toks:
        return "sin"
    if "pow2" in toks or ("mul" in toks and toks.count("x_0") >= 2):
        return "polynomial-like"
    if toks == ["CONSTANT"] or text.startswith("mul CONSTANT CONSTANT"):
        return "constant"
    if "x_0" in toks:
        if toks.count("mul") >= 2:
            return "nested-mul linear"
        return "linear"
    return "other"


def semantic_relation(candidate: dict[str, Any] | None, truth_drift: list[str], truth_diffusion: list[str]) -> dict[str, Any]:
    if candidate is None:
        return {
            "same_drift_as_truth": None,
            "same_diffusion_as_truth": None,
            "drift_family_truth": drift_family(truth_drift),
            "drift_family_selected": None,
        }
    drift_tokens = candidate.get("drift_tokens_canonical") or candidate.get("drift_tokens_raw") or []
    diffusion_tokens = candidate.get("diffusion_tokens_canonical") or candidate.get("diffusion_tokens_raw") or []
    return {
        "same_drift_as_truth": bool(candidate.get("is_truth_drift_canonical") or drift_tokens == truth_drift),
        "same_diffusion_as_truth": bool(candidate.get("is_truth_diffusion_exact") or diffusion_tokens == truth_diffusion),
        "drift_family_truth": drift_family(truth_drift),
        "drift_family_selected": drift_family(drift_tokens),
    }


def classify_semantic_relation(relation: dict[str, Any]) -> str:
    same_drift = bool(relation.get("same_drift_as_truth"))
    same_diffusion = bool(relation.get("same_diffusion_as_truth"))
    if same_drift and not same_diffusion:
        return "same_drift_wrong_diffusion"
    if same_diffusion and not same_drift:
        return "same_diffusion_wrong_drift"
    if not same_drift and not same_diffusion:
        return "both_sides_wrong"
    return "constant_only_or_near_constant"


def classify_ambiguity(active_gap: float | None, weak_gap: float | None) -> str:
    if active_gap is None or weak_gap is None:
        return "unknown"
    a = abs(active_gap)
    w = abs(weak_gap)
    if a >= 2.0 * w:
        return "active"
    if w >= 2.0 * a:
        return "weak"
    return "mixed"


def sign_crosses(case_type: str, gap_v0: float | None, gap_v4: float | None) -> bool:
    if gap_v0 is None or gap_v4 is None:
        return False
    if case_type == "V4_rescue":
        return gap_v0 > 0 and gap_v4 < 0
    if case_type == "V4_harm":
        return gap_v0 < 0 and gap_v4 > 0
    return (gap_v0 > 0 > gap_v4) or (gap_v0 < 0 < gap_v4)


def repeat_score(tokens: list[str], target_y: np.ndarray, sample_index: int, repeat: int) -> dict[str, Any]:
    details, failure_reason, fingerprint_failures = score_candidate_system(
        tokens,
        target_y,
        FINGERPRINT_CONFIG,
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(sample_index, 9000 + repeat * 101),
    )
    if details is None:
        return {
            "valid_bool": False,
            "failure_reason": failure_reason,
            "fingerprint_failures": fingerprint_failures,
            "v0_score": None,
            "v4_score": None,
            "active_score": None,
            "active_score_after_removed_dims": None,
            "weak_score": None,
        }
    active_removed = l2_without(details.get("active_kramers_moyal_residual"), GLOBAL_ACTIVE_DIMS)
    weak = details.get("gaussian_weak_kernel_distance")
    return {
        "valid_bool": True,
        "failure_reason": None,
        "fingerprint_failures": fingerprint_failures,
        "v0_score": details.get("distance"),
        "v4_score": None if active_removed is None or weak is None else active_removed + weak,
        "active_score": details.get("active_kramers_moyal_distance"),
        "active_score_after_removed_dims": active_removed,
        "weak_score": weak,
    }


def mean_std(values: list[float]) -> dict[str, float | None]:
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(finite_values),
        "std": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
    }


def resimulation_check(
    sample: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    nonoracle_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if oracle_row is None or nonoracle_row is None:
        return {"available_bool": False, "reason": "missing comparison row"}
    target_y = target_array(sample)
    oracle_repeats = [
        repeat_score(oracle_row.get("full_sequence_tokens") or [], target_y, int(sample["sample_index"]), repeat)
        for repeat in range(3)
    ]
    nonoracle_repeats = [
        repeat_score(nonoracle_row.get("full_sequence_tokens") or [], target_y, int(sample["sample_index"]), repeat + 31)
        for repeat in range(3)
    ]
    valid_pairs = [
        (oracle, nonoracle)
        for oracle, nonoracle in zip(oracle_repeats, nonoracle_repeats)
        if oracle.get("valid_bool") and nonoracle.get("valid_bool")
    ]
    oracle_beats = [
        bool(oracle.get("v4_score") < nonoracle.get("v4_score"))
        for oracle, nonoracle in valid_pairs
        if oracle.get("v4_score") is not None and nonoracle.get("v4_score") is not None
    ]
    return {
        "available_bool": bool(valid_pairs),
        "repeats": len(valid_pairs),
        "oracle_total_score": mean_std([row.get("v4_score") for row in oracle_repeats]),
        "nonoracle_total_score": mean_std([row.get("v4_score") for row in nonoracle_repeats]),
        "oracle_active_score": mean_std([row.get("active_score_after_removed_dims") for row in oracle_repeats]),
        "nonoracle_active_score": mean_std([row.get("active_score_after_removed_dims") for row in nonoracle_repeats]),
        "oracle_weak_score": mean_std([row.get("weak_score") for row in oracle_repeats]),
        "nonoracle_weak_score": mean_std([row.get("weak_score") for row in nonoracle_repeats]),
        "ordering_stable_bool": len(set(oracle_beats)) == 1 if oracle_beats else None,
        "oracle_beats_selected_in_any_repeat_bool": any(oracle_beats) if oracle_beats else None,
        "oracle_beats_selected_in_majority_bool": (sum(oracle_beats) >= 2) if oracle_beats else None,
    }


def build_case(
    idx: int,
    case_type: str,
    smoke_sample: dict[str, Any],
    sidecar_sample: dict[str, Any],
    coverage_sample: dict[str, Any] | None,
) -> dict[str, Any]:
    lookup = candidate_lookup(sidecar_sample)
    cache: dict[str, dict[str, Any]] = {}
    v0_row_meta = smoke_sample["variants"][V0]
    v4_row_meta = smoke_sample["variants"][V4]
    v0_row = lookup[str(v0_row_meta["selected_candidate_id"])][1]
    v4_row = lookup[str(v4_row_meta["selected_candidate_id"])][1]
    for row in (v0_row, v4_row):
        candidate_id = str(row["candidate_id"])
        cache[candidate_id] = score_candidate_row(sidecar_sample, row, lookup[candidate_id][0])

    exact_row, exact_scored = best_scored_row(
        sidecar_sample,
        exact_oracle_rows(sidecar_sample),
        lookup,
        cache,
    )
    p2_row, p2_scored = best_scored_row(
        sidecar_sample,
        p2_oracle_rows(sidecar_sample),
        lookup,
        cache,
    )
    current_row, current_scored = best_scored_row(
        sidecar_sample,
        current_expanded_oracle_rows(sidecar_sample),
        lookup,
        cache,
    )
    oracle_row = exact_row or p2_row or current_row
    oracle_scored = exact_scored or p2_scored or current_scored

    if case_type == "V4_rescue":
        nonoracle_row = v0_row
        nonoracle_scored = cache[str(v0_row["candidate_id"])]
    elif case_type == "V4_harm":
        nonoracle_row = v4_row
        nonoracle_scored = cache[str(v4_row["candidate_id"])]
    else:
        nonoracle_row = v4_row if v4_row.get("candidate_id") != v0_row.get("candidate_id") else v0_row
        nonoracle_scored = cache[str(nonoracle_row["candidate_id"])]

    v0_selected_scored = cache[str(v0_row["candidate_id"])]
    v4_selected_scored = cache[str(v4_row["candidate_id"])]
    truth_drift = sidecar_sample.get("truth_drift_tokens") or []
    truth_diffusion = sidecar_sample.get("truth_diffusion_tokens") or []
    selected_relation = semantic_relation(nonoracle_row, truth_drift, truth_diffusion)
    oracle_relation = semantic_relation(oracle_row, truth_drift, truth_diffusion)

    oracle_v0 = score_summary(oracle_scored, V0)
    oracle_v4 = score_summary(oracle_scored, V4)
    v0_selected_v0 = score_summary(v0_selected_scored, V0)
    v4_selected_v4 = score_summary(v4_selected_scored, V4)
    nonoracle_v0 = score_summary(nonoracle_scored, V0)
    nonoracle_v4 = score_summary(nonoracle_scored, V4)

    pair_gap_v0 = safe_sub(
        oracle_v0.get("total_score") if oracle_v0 else None,
        nonoracle_v0.get("total_score") if nonoracle_v0 else None,
    )
    pair_gap_v4 = safe_sub(
        oracle_v4.get("total_score") if oracle_v4 else None,
        nonoracle_v4.get("total_score") if nonoracle_v4 else None,
    )
    active_gap_v0 = safe_sub(
        oracle_v0.get("active_score") if oracle_v0 else None,
        nonoracle_v0.get("active_score") if nonoracle_v0 else None,
    )
    active_gap_v4 = safe_sub(
        oracle_v4.get("active_score") if oracle_v4 else None,
        nonoracle_v4.get("active_score") if nonoracle_v4 else None,
    )
    weak_gap = safe_sub(
        oracle_v4.get("weak_score") if oracle_v4 else None,
        nonoracle_v4.get("weak_score") if nonoracle_v4 else None,
    )
    removed_contribution = safe_sub(active_gap_v4, active_gap_v0)
    removed_fraction = safe_div(abs(removed_contribution), abs(active_gap_v0)) if active_gap_v0 is not None and removed_contribution is not None else None
    explains_flip = sign_crosses(case_type, pair_gap_v0, pair_gap_v4)

    return {
        "sample_index": idx,
        "case_type": case_type,
        "truth_sequence": sidecar_sample.get("truth_sequence"),
        "truth_drift_tokens": truth_drift,
        "truth_diffusion_tokens": truth_diffusion,
        "V0_selected_candidate": candidate_brief(v0_row),
        "V4_selected_candidate": candidate_brief(v4_row),
        "exact_oracle_candidate": candidate_brief(exact_row),
        "p2_oracle_candidate": candidate_brief(p2_row),
        "best_current_expanded_oracle_candidate": candidate_brief(current_row),
        "coverage_context": {
            "baseline_full_oracle": bool((coverage_sample or {}).get("has_full")),
            "current_expanded_oracle_present": bool(sidecar_sample.get("current_expanded_oracle_present_bool")),
            "canonicalized_expanded_oracle_present": bool(sidecar_sample.get("canonicalized_expanded_oracle_present_bool")),
            "p2_oracle_present": bool(sidecar_sample.get("p2_oracle_present_bool")),
            "exact_diffusion_present": bool(sidecar_sample.get("exact_diffusion_present_bool")),
        },
        "V0_selected_score_components_under_V0": v0_selected_v0,
        "V4_selected_score_components_under_V4": v4_selected_v4,
        "oracle_score_components_under_V0": oracle_v0,
        "oracle_score_components_under_V4": oracle_v4,
        "nonoracle_comparison_candidate": candidate_brief(nonoracle_row),
        "nonoracle_score_components_under_V0": nonoracle_v0,
        "nonoracle_score_components_under_V4": nonoracle_v4,
        "gap_summaries": {
            "V0_oracle_gap": safe_sub(
                oracle_v0.get("total_score") if oracle_v0 else None,
                v0_selected_v0.get("total_score") if v0_selected_v0 else None,
            ),
            "V4_oracle_gap": safe_sub(
                oracle_v4.get("total_score") if oracle_v4 else None,
                v4_selected_v4.get("total_score") if v4_selected_v4 else None,
            ),
            "gap_delta": safe_sub(
                safe_sub(
                    oracle_v4.get("total_score") if oracle_v4 else None,
                    v4_selected_v4.get("total_score") if v4_selected_v4 else None,
                ),
                safe_sub(
                    oracle_v0.get("total_score") if oracle_v0 else None,
                    v0_selected_v0.get("total_score") if v0_selected_v0 else None,
                ),
            ),
            "oracle_vs_nonoracle_gap_under_V0": pair_gap_v0,
            "oracle_vs_nonoracle_gap_under_V4": pair_gap_v4,
        },
        "segment_level_ambiguity": {
            "active_gap_under_V0": active_gap_v0,
            "active_gap_after_removed_dims": active_gap_v4,
            "weak_gap": weak_gap,
            "removed_dims_contribution_to_active_gap": removed_contribution,
            "removed_dims_fraction_of_active_gap": removed_fraction,
            "removed_dims_explain_flip": explains_flip,
            "ambiguity_classification": classify_ambiguity(active_gap_v4, weak_gap),
            "removed_active_dims": REMOVED_ACTIVE_DIMS,
        },
        "semantic_relation": {
            **selected_relation,
            "same_drift_as_selected": bool(
                oracle_relation.get("same_drift_as_truth")
                and selected_relation.get("same_drift_as_truth")
            ),
            "same_diffusion_as_selected": bool(
                oracle_relation.get("same_diffusion_as_truth")
                and selected_relation.get("same_diffusion_as_truth")
            ),
            "drift_family_oracle": oracle_relation.get("drift_family_selected"),
            "selected_error_type": classify_semantic_relation(selected_relation),
        },
        "resimulation_stability": resimulation_check(sidecar_sample, oracle_row, nonoracle_row)
        if case_type in {"V4_rescue", "V4_harm"}
        else {"available_bool": False, "reason": "not run for compact preserved/nonoracle cases"},
    }


def classify_cases(smoke_payload: dict[str, Any]) -> dict[str, list[int]]:
    cases = {
        "V4_rescue": [],
        "V4_harm": [],
        "changed_nonoracle_to_nonoracle": [],
        "V0_preserved_hit": [],
    }
    for sample in smoke_payload.get("samples", []):
        idx = int(sample["sample_index"])
        variants = sample.get("variants", {})
        v0 = variants.get(V0, {})
        v4 = variants.get(V4, {})
        v0_oracle = bool(v0.get("selected_is_oracle_bool"))
        v4_oracle = bool(v4.get("selected_is_oracle_bool"))
        changed = v0.get("selected_candidate_id") != v4.get("selected_candidate_id")
        if not v0_oracle and v4_oracle:
            cases["V4_rescue"].append(idx)
        elif v0_oracle and not v4_oracle:
            cases["V4_harm"].append(idx)
        elif not v0_oracle and not v4_oracle and changed:
            cases["changed_nonoracle_to_nonoracle"].append(idx)
        elif v0_oracle and v4_oracle:
            cases["V0_preserved_hit"].append(idx)
    return cases


def compact_preserved_case(
    idx: int,
    smoke_sample: dict[str, Any],
    sidecar_sample: dict[str, Any],
) -> dict[str, Any]:
    lookup = candidate_lookup(sidecar_sample)
    v0_meta = smoke_sample["variants"][V0]
    v4_meta = smoke_sample["variants"][V4]
    v0_row = lookup[str(v0_meta["selected_candidate_id"])][1]
    v4_row = lookup[str(v4_meta["selected_candidate_id"])][1]
    return {
        "sample_index": idx,
        "case_type": "V0_preserved_hit",
        "truth_sequence": sidecar_sample.get("truth_sequence"),
        "truth_drift_tokens": sidecar_sample.get("truth_drift_tokens"),
        "truth_diffusion_tokens": sidecar_sample.get("truth_diffusion_tokens"),
        "V0_selected_candidate": candidate_brief(v0_row),
        "V4_selected_candidate": candidate_brief(v4_row),
        "V4_would_change_it": v0_row.get("candidate_id") != v4_row.get("candidate_id"),
        "V4_preserves_or_harms": "preserves",
    }


def count_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    case_counts = Counter(sample["case_type"] for sample in samples)
    detailed = [sample for sample in samples if sample["case_type"] in {"V4_rescue", "V4_harm"}]
    ambiguity_counts = Counter(
        sample.get("segment_level_ambiguity", {}).get("ambiguity_classification", "unknown")
        for sample in detailed
    )
    relation_counts = Counter(
        sample.get("semantic_relation", {}).get("selected_error_type", "unknown")
        for sample in detailed
    )
    removed_explain = sum(
        1
        for sample in detailed
        if sample.get("segment_level_ambiguity", {}).get("removed_dims_explain_flip")
    )
    removed_explain_harm = sum(
        1
        for sample in detailed
        if sample["case_type"] == "V4_harm"
        and sample.get("segment_level_ambiguity", {}).get("removed_dims_explain_flip")
    )
    resim = [
        sample.get("resimulation_stability", {})
        for sample in detailed
        if sample.get("resimulation_stability", {}).get("available_bool")
    ]
    stable = sum(1 for row in resim if row.get("ordering_stable_bool"))
    oracle_recovers = sum(1 for row in resim if row.get("oracle_beats_selected_in_majority_bool"))
    if resim and stable < len(resim):
        decision = "C"
        recommendation = (
            "Some small resimulation orderings changed. Diagnose fingerprint simulation stability before scorer changes."
        )
    elif removed_explain and len(set(relation_counts)) > 1:
        decision = "A"
        recommendation = (
            "Harms and rescues show distinct semantic/fingerprint ambiguity patterns. "
            "A future diagnostic may log these observables more broadly, but no rerank mode or formal eval is recommended."
        )
    else:
        decision = "B"
        recommendation = (
            "Rescues and harms are not separated by a safe observable pattern in this diagnostic. "
            "Keep scorer calibration closed and move to drift span / candidate-generation diagnostics."
        )
    return {
        "samples_analyzed": len(samples),
        "V4_rescue_count": case_counts["V4_rescue"],
        "V4_harm_count": case_counts["V4_harm"],
        "changed_nonoracle_to_nonoracle_count": case_counts["changed_nonoracle_to_nonoracle"],
        "V0_preserved_hit_count": case_counts["V0_preserved_hit"],
        "removed_dims_explain_flip_count": removed_explain,
        "removed_dims_explain_harm_count": removed_explain_harm,
        "active_dominated_ambiguity_count": ambiguity_counts["active"],
        "weak_dominated_ambiguity_count": ambiguity_counts["weak"],
        "mixed_ambiguity_count": ambiguity_counts["mixed"],
        "unknown_ambiguity_count": ambiguity_counts["unknown"],
        "same_diffusion_wrong_drift_count": relation_counts["same_diffusion_wrong_drift"],
        "same_drift_wrong_diffusion_count": relation_counts["same_drift_wrong_diffusion"],
        "both_sides_wrong_count": relation_counts["both_sides_wrong"],
        "constant_only_or_near_constant_count": relation_counts["constant_only_or_near_constant"],
        "resimulation_available_bool": bool(resim),
        "resimulation_ordering_stable_count": stable,
        "resimulation_oracle_recovers_count": oracle_recovers,
        "decision": decision,
        "formal_eval_recommended": False,
        "recommendation": recommendation,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Fingerprint Ambiguity Diagnostics",
        "",
        "## Why This Diagnostic Was Run",
        "",
        "The previous gate-feasibility diagnostic closed the fixed residual calibration",
        "implementation path because the best observable gate (`V4 + G3`) improved net",
        "selected recovery but still harmed V0 hits. This helper inspects the V0/V4",
        "changed cases offline to see whether rescues and harms have separable",
        "fingerprint ambiguity patterns. No model decoding, candidate generation,",
        "`sde_validation_probe.py`, formal eval, rerank-mode implementation, or",
        "production default change was run.",
        "",
        "## Aggregate Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Samples analyzed | {summary['samples_analyzed']} |",
        f"| V4 rescues vs V0 | {summary['V4_rescue_count']} |",
        f"| V4 harms vs V0 | {summary['V4_harm_count']} |",
        f"| Changed nonoracle to nonoracle | {summary['changed_nonoracle_to_nonoracle_count']} |",
        f"| V0 preserved hits | {summary['V0_preserved_hit_count']} |",
        f"| Removed dims explain flips | {summary['removed_dims_explain_flip_count']} |",
        f"| Removed dims explain harms | {summary['removed_dims_explain_harm_count']} |",
        f"| Active dominated ambiguity | {summary['active_dominated_ambiguity_count']} |",
        f"| Weak dominated ambiguity | {summary['weak_dominated_ambiguity_count']} |",
        f"| Mixed ambiguity | {summary['mixed_ambiguity_count']} |",
        "",
        "## V4 Rescue/Harm Samples",
        "",
        "| Sample | Type | Truth drift | Truth diffusion | V0 selected | V4 selected | V0 oracle gap | V4 oracle gap | Removed dims explain flip | Ambiguity | Selected error |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    detailed = [
        sample for sample in result["samples"]
        if sample["case_type"] in {"V4_rescue", "V4_harm"}
    ]
    for sample in detailed:
        gaps = sample["gap_summaries"]
        seg = sample["segment_level_ambiguity"]
        rel = sample["semantic_relation"]
        lines.append(
            "| {idx} | {case} | `{truth_drift}` | `{truth_diffusion}` | `{v0}` | `{v4}` | {v0_gap} | {v4_gap} | {explain} | {ambiguity} | {error} |".format(
                idx=sample["sample_index"],
                case=sample["case_type"],
                truth_drift=token_text(sample.get("truth_drift_tokens")),
                truth_diffusion=token_text(sample.get("truth_diffusion_tokens")),
                v0=token_text(sample["V0_selected_candidate"]["sequence"]),
                v4=token_text(sample["V4_selected_candidate"]["sequence"]),
                v0_gap=fmt(gaps.get("V0_oracle_gap")),
                v4_gap=fmt(gaps.get("V4_oracle_gap")),
                explain=fmt(seg.get("removed_dims_explain_flip")),
                ambiguity=seg.get("ambiguity_classification"),
                error=rel.get("selected_error_type"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Sample Fingerprint Gap Explanation",
            "",
            "| Sample | Pair gap V0 | Pair gap V4 | Active gap V0 | Active gap after removal | Weak gap | Removed contribution | Removed fraction |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sample in detailed:
        gaps = sample["gap_summaries"]
        seg = sample["segment_level_ambiguity"]
        lines.append(
            "| {idx} | {pair_v0} | {pair_v4} | {active_v0} | {active_v4} | {weak} | {removed} | {fraction} |".format(
                idx=sample["sample_index"],
                pair_v0=fmt(gaps.get("oracle_vs_nonoracle_gap_under_V0")),
                pair_v4=fmt(gaps.get("oracle_vs_nonoracle_gap_under_V4")),
                active_v0=fmt(seg.get("active_gap_under_V0")),
                active_v4=fmt(seg.get("active_gap_after_removed_dims")),
                weak=fmt(seg.get("weak_gap")),
                removed=fmt(seg.get("removed_dims_contribution_to_active_gap")),
                fraction=fmt(seg.get("removed_dims_fraction_of_active_gap")),
            )
        )
    lines.extend(
        [
            "",
            "## V0 Preserved Hits",
            "",
            "| Sample | Truth | V0 selected | V4 selected | V4 changes candidate | Outcome |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in result["preserved_hit_samples"]:
        lines.append(
            "| {idx} | `{truth}` | `{v0}` | `{v4}` | {changed} | {outcome} |".format(
                idx=sample["sample_index"],
                truth=token_text(sample.get("truth_sequence")),
                v0=token_text(sample["V0_selected_candidate"]["sequence"]),
                v4=token_text(sample["V4_selected_candidate"]["sequence"]),
                changed=fmt(sample["V4_would_change_it"]),
                outcome=sample["V4_preserves_or_harms"],
            )
        )
    lines.extend(
        [
            "",
            "## Resimulation Stability",
            "",
            f"Available: {fmt(summary['resimulation_available_bool'])}.",
            "",
            "| Sample | Ordering stable | Oracle beats nonoracle in any repeat | Oracle beats nonoracle in majority | Oracle V4 mean/std | Nonoracle V4 mean/std |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in detailed:
        resim = sample["resimulation_stability"]
        oracle_score = resim.get("oracle_total_score", {})
        nonoracle_score = resim.get("nonoracle_total_score", {})
        lines.append(
            "| {idx} | {stable} | {anywin} | {majority} | {omean}/{ostd} | {nmean}/{nstd} |".format(
                idx=sample["sample_index"],
                stable=fmt(resim.get("ordering_stable_bool")),
                anywin=fmt(resim.get("oracle_beats_selected_in_any_repeat_bool")),
                majority=fmt(resim.get("oracle_beats_selected_in_majority_bool")),
                omean=fmt(oracle_score.get("mean")),
                ostd=fmt(oracle_score.get("std")),
                nmean=fmt(nonoracle_score.get("mean")),
                nstd=fmt(nonoracle_score.get("std")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval, no rerank mode, and no P2 eval integration are recommended",
            "from this diagnostic alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    gate_payload = load_json(args.gate_json)
    smoke_payload = load_json(args.all32_smoke_json)
    sidecar_payload = load_json(args.scorer_ready_json)
    p2_coverage_payload = load_json(args.p2_coverage_json)
    baseline_coverage_payload = load_json(args.baseline_coverage_json)

    smoke_by_idx = sample_by_index(smoke_payload)
    sidecar_by_idx = sample_by_index(sidecar_payload)
    p2_coverage_by_idx = sample_by_index(p2_coverage_payload)
    baseline_coverage_by_idx = sample_by_index(baseline_coverage_payload)
    cases = classify_cases(smoke_payload)

    detailed_samples: list[dict[str, Any]] = []
    for case_type in ("V4_rescue", "V4_harm", "changed_nonoracle_to_nonoracle"):
        for idx in cases[case_type]:
            detailed_samples.append(
                build_case(
                    idx,
                    case_type,
                    smoke_by_idx[idx],
                    sidecar_by_idx[idx],
                    p2_coverage_by_idx.get(idx) or baseline_coverage_by_idx.get(idx),
                )
            )
    preserved = [
        compact_preserved_case(idx, smoke_by_idx[idx], sidecar_by_idx[idx])
        for idx in cases["V0_preserved_hit"]
    ]
    result_samples = detailed_samples + preserved
    summary = count_summary(result_samples)
    summary.update(
        {
            "gate_decision": gate_payload.get("summary", {}).get("decision"),
            "V4_rescue_samples": cases["V4_rescue"],
            "V4_harm_samples": cases["V4_harm"],
            "changed_nonoracle_to_nonoracle_samples": cases["changed_nonoracle_to_nonoracle"],
            "V0_preserved_hit_samples": cases["V0_preserved_hit"],
            "model_decoding_avoided": True,
            "candidate_generation_avoided": True,
            "sde_validation_probe_avoided": True,
        }
    )
    result = {
        "metadata": {
            "gate_json": str(args.gate_json),
            "all32_smoke_json": str(args.all32_smoke_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "p2_coverage_json": str(args.p2_coverage_json),
            "baseline_coverage_json": str(args.baseline_coverage_json),
            "removed_active_dims": REMOVED_ACTIVE_DIMS,
            "resimulation_repeats": 3,
            "fingerprint_config": FINGERPRINT_CONFIG.to_dict(),
        },
        "summary": summary,
        "samples": result_samples,
        "preserved_hit_samples": preserved,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_markdown(json_ready(result)))
    print(f"decision={summary['decision']}")
    print(f"samples_analyzed={summary['samples_analyzed']}")
    print(f"V4_rescue_count={summary['V4_rescue_count']}")
    print(f"V4_harm_count={summary['V4_harm_count']}")
    print(f"changed_nonoracle_to_nonoracle_count={summary['changed_nonoracle_to_nonoracle_count']}")
    print(f"removed_dims_explain_flip_count={summary['removed_dims_explain_flip_count']}")
    print(f"resimulation_available_bool={summary['resimulation_available_bool']}")


if __name__ == "__main__":
    main()
