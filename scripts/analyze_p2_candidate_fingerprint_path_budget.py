#!/usr/bin/env python3
"""Offline path-budget stability diagnostic for P2 candidate fingerprints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
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
from analyze_p2_fingerprint_ambiguity import candidate_brief, candidate_lookup, finite, token_text
from sde_fingerprint import FingerprintConfig
from sde_validation_probe import score_candidate_system


CONSTANT_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
EPS = 1e-12
MAX_CODEX_SECONDS = 600.0


@dataclass(frozen=True)
class BudgetMode:
    name: str
    n_paths: int
    active_paths: int
    n_steps: int
    repeats: int

    @property
    def config(self) -> FingerprintConfig:
        return FingerprintConfig(
            n_paths=self.n_paths,
            active_paths=self.active_paths,
            n_steps=self.n_steps,
        )


BUDGETS = [
    BudgetMode("B0_current_budget", n_paths=800, active_paths=800, n_steps=60, repeats=5),
    BudgetMode("B1_active_paths_high", n_paths=800, active_paths=1600, n_steps=60, repeats=3),
    BudgetMode("B2_weak_paths_high", n_paths=1600, active_paths=800, n_steps=60, repeats=3),
    BudgetMode("B3_both_paths_high", n_paths=1600, active_paths=1600, n_steps=60, repeats=3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-json", type=Path, required=True)
    parser.add_argument("--ambiguity-json", type=Path, required=True)
    parser.add_argument("--all32-smoke-json", type=Path, required=True)
    parser.add_argument("--scorer-ready-json", type=Path, required=True)
    parser.add_argument("--target-case-types", type=str, default="V4_rescue,V4_harm")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sample_by_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(sample["sample_index"]): sample for sample in payload.get("samples", [])}


def parse_case_types(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(finite_values),
        "std": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
    }


def sign(value: float | None) -> int:
    if value is None:
        return 0
    if value < -1e-10:
        return -1
    if value > 1e-10:
        return 1
    return 0


def sign_flip_fraction(values: list[float | None]) -> float | None:
    signs = [sign(value) for value in values if value is not None and sign(value) != 0]
    if not signs:
        return None
    return (len(signs) - max(Counter(signs).values())) / len(signs)


def segment_noise_driver(active_std: float | None, weak_std: float | None) -> str:
    if active_std is None or weak_std is None:
        return "unknown"
    if active_std >= 2.0 * weak_std:
        return "active"
    if weak_std >= 2.0 * active_std:
        return "weak"
    return "mixed"


def score_candidate_budget(
    sample: dict[str, Any],
    row: dict[str, Any],
    candidate_index: int,
    budget: BudgetMode,
    repeat: int,
) -> dict[str, Any]:
    details, failure_reason, fingerprint_failures = score_candidate_system(
        row.get("full_sequence_tokens") or [],
        target_array(sample),
        budget.config,
        "constant_grid_rolewise_no_multi_u0",
        (1.0, 2.0, 1.0),
        CONSTANT_VALUES,
        score_seed(int(sample["sample_index"]), candidate_index + repeat * 104729),
    )
    if details is None:
        return {
            "valid_bool": False,
            "failure_reason": failure_reason,
            "fingerprint_failures": fingerprint_failures,
            "total_score": None,
            "active_score": None,
            "weak_score": None,
            "best_drift_constant": None,
            "best_diffusion_constant": None,
        }
    return {
        "valid_bool": True,
        "failure_reason": None,
        "fingerprint_failures": fingerprint_failures,
        "total_score": details.get("distance"),
        "active_score": details.get("active_kramers_moyal_distance"),
        "weak_score": details.get("gaussian_weak_kernel_distance"),
        "best_drift_constant": details.get("best_drift_constant"),
        "best_diffusion_constant": details.get("best_diffusion_constant"),
    }


def candidate_from_brief(brief: dict[str, Any] | None, lookup: dict[str, tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    if not brief:
        return None
    candidate_id = brief.get("candidate_id")
    if candidate_id is None:
        return None
    item = lookup.get(str(candidate_id))
    return None if item is None else item[1]


def oracle_row_for(ambiguity_sample: dict[str, Any], lookup: dict[str, tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    for key in ("exact_oracle_candidate", "p2_oracle_candidate", "best_current_expanded_oracle_candidate"):
        row = candidate_from_brief(ambiguity_sample.get(key), lookup)
        if row is not None:
            return row
    return None


def prior_pair_names(stability_sample: dict[str, Any]) -> set[str]:
    return {
        pair.get("pair_name")
        for pair in stability_sample.get("comparison_pairs", [])
        if pair.get("pair_name") in {"V0_selected_vs_oracle", "V4_selected_vs_oracle"}
    }


def baseline_gap_payload(ambiguity_sample: dict[str, Any], pair_name: str) -> dict[str, Any]:
    gaps = ambiguity_sample.get("gap_summaries", {})
    segment = ambiguity_sample.get("segment_level_ambiguity", {})
    if pair_name == "V0_selected_vs_oracle":
        return {
            "baseline_stored_gap": gaps.get("oracle_vs_nonoracle_gap_under_V0"),
            "baseline_stored_active_gap": segment.get("active_gap_under_V0"),
            "baseline_stored_weak_gap": segment.get("weak_gap"),
        }
    return {
        "baseline_stored_gap": gaps.get("oracle_vs_nonoracle_gap_under_V4"),
        "baseline_stored_active_gap": segment.get("active_gap_after_removed_dims"),
        "baseline_stored_weak_gap": segment.get("weak_gap"),
    }


def build_pair_specs(
    stability_payload: dict[str, Any],
    ambiguity_payload: dict[str, Any],
    sidecar_payload: dict[str, Any],
    target_case_types: set[str],
) -> list[dict[str, Any]]:
    stability_by_idx = sample_by_index(stability_payload)
    ambiguity_by_idx = sample_by_index(ambiguity_payload)
    sidecar_by_idx = sample_by_index(sidecar_payload)
    specs: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str]] = set()
    for idx, ambiguity_sample in sorted(ambiguity_by_idx.items()):
        if ambiguity_sample.get("case_type") not in target_case_types:
            continue
        stability_sample = stability_by_idx.get(idx)
        if not stability_sample:
            continue
        lookup = candidate_lookup(sidecar_by_idx[idx])
        oracle = oracle_row_for(ambiguity_sample, lookup)
        if oracle is None:
            continue
        for pair_name in sorted(prior_pair_names(stability_sample)):
            selected = candidate_from_brief(
                ambiguity_sample.get("V0_selected_candidate" if pair_name.startswith("V0") else "V4_selected_candidate"),
                lookup,
            )
            if selected is None:
                continue
            key = (idx, pair_name, str(selected["candidate_id"]), str(oracle["candidate_id"]))
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                {
                    "sample_index": idx,
                    "case_type": ambiguity_sample.get("case_type"),
                    "pair_type": pair_name,
                    "selected_row": selected,
                    "oracle_row": oracle,
                    "selected_candidate_index": lookup[str(selected["candidate_id"])][0],
                    "oracle_candidate_index": lookup[str(oracle["candidate_id"])][0],
                    "selected_sequence": selected.get("full_sequence_tokens"),
                    "oracle_sequence": oracle.get("full_sequence_tokens"),
                    "selected_source": selected.get("candidate_source"),
                    "oracle_source": oracle.get("candidate_source"),
                    **baseline_gap_payload(ambiguity_sample, pair_name),
                }
            )
    return specs


def unique_candidate_calls(pair_specs: list[dict[str, Any]], sidecar_payload: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    sidecar_by_idx = sample_by_index(sidecar_payload)
    calls: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    seen: set[tuple[int, str]] = set()
    for spec in pair_specs:
        sample = sidecar_by_idx[int(spec["sample_index"])]
        for row_key, index_key in (("selected_row", "selected_candidate_index"), ("oracle_row", "oracle_candidate_index")):
            row = spec[row_key]
            key = (int(spec["sample_index"]), str(row["candidate_id"]))
            if key in seen:
                continue
            seen.add(key)
            calls.append((sample, row, int(spec[index_key])))
    return calls


def estimate_runtime(pair_specs: list[dict[str, Any]], sidecar_payload: dict[str, Any]) -> dict[str, Any]:
    calls = unique_candidate_calls(pair_specs, sidecar_payload)
    if not calls:
        return {"estimated_seconds": 0.0, "single_call_seconds": 0.0, "score_calls": 0}
    sample, row, candidate_index = calls[0]
    start = time.monotonic()
    score_candidate_budget(sample, row, candidate_index, BUDGETS[-1], 0)
    elapsed = max(time.monotonic() - start, 1e-6)
    total_score_calls = sum(len(calls) * budget.repeats for budget in BUDGETS)
    # The probe call used the largest path budget, so this is intentionally conservative.
    return {
        "estimated_seconds": elapsed * total_score_calls,
        "single_call_seconds": elapsed,
        "score_calls": total_score_calls,
        "unique_candidates": len(calls),
    }


def score_budget_mode(
    pair_specs: list[dict[str, Any]],
    sidecar_payload: dict[str, Any],
    budget: BudgetMode,
) -> dict[tuple[int, str], dict[str, list[dict[str, Any]]]]:
    sidecar_by_idx = sample_by_index(sidecar_payload)
    cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for spec in pair_specs:
        sample = sidecar_by_idx[int(spec["sample_index"])]
        for role, row_key, index_key in (
            ("selected", "selected_row", "selected_candidate_index"),
            ("oracle", "oracle_row", "oracle_candidate_index"),
        ):
            row = spec[row_key]
            key = (int(spec["sample_index"]), str(row["candidate_id"]))
            if key not in cache:
                cache[key] = [
                    score_candidate_budget(sample, row, int(spec[index_key]), budget, repeat)
                    for repeat in range(budget.repeats)
                ]
    out: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = {}
    for spec in pair_specs:
        out[(int(spec["sample_index"]), spec["pair_type"])] = {
            "selected": cache[(int(spec["sample_index"]), str(spec["selected_row"]["candidate_id"]))],
            "oracle": cache[(int(spec["sample_index"]), str(spec["oracle_row"]["candidate_id"]))],
        }
    return out


def gap_values(oracle_series: list[dict[str, Any]], selected_series: list[dict[str, Any]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for oracle_row, selected_row in zip(oracle_series, selected_series):
        oval = oracle_row.get(key)
        sval = selected_row.get(key)
        values.append(None if not finite(oval) or not finite(sval) else float(oval) - float(sval))
    return values


def noise_share(active_std: float | None, weak_std: float | None) -> tuple[float | None, float | None]:
    if active_std is None or weak_std is None:
        return None, None
    denom = active_std * active_std + weak_std * weak_std
    if denom < EPS:
        return None, None
    return active_std * active_std / denom, weak_std * weak_std / denom


def segment_driver(active_std: float | None, weak_std: float | None) -> str:
    if active_std is None or weak_std is None:
        return "unknown"
    if active_std >= 2.0 * weak_std:
        return "active"
    if weak_std >= 2.0 * active_std:
        return "weak"
    return "mixed"


def pair_budget_metrics(spec: dict[str, Any], budget: BudgetMode, series: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected = series["selected"]
    oracle = series["oracle"]
    gaps = gap_values(oracle, selected, "total_score")
    active_gaps = gap_values(oracle, selected, "active_score")
    weak_gaps = gap_values(oracle, selected, "weak_score")
    gap_stats = mean_std(gaps)
    active_stats = mean_std(active_gaps)
    weak_stats = mean_std(weak_gaps)
    signs = [sign(value) for value in gaps if value is not None and sign(value) != 0]
    sign_flips = 0 if not signs else len(signs) - max(Counter(signs).values())
    oracle_wins = sum(1 for value in gaps if value is not None and value < 0)
    selected_wins = sum(1 for value in gaps if value is not None and value > 0)
    stable = bool(signs) and len(set(signs)) == 1
    std_gap = gap_stats["std"]
    mean_gap = gap_stats["mean"]
    noise_dominated = bool(mean_gap is not None and std_gap is not None and abs(mean_gap) <= 2.0 * std_gap)
    active_share, weak_share = noise_share(active_stats["std"], weak_stats["std"])
    selected_total = mean_std([row.get("total_score") for row in selected])
    oracle_total = mean_std([row.get("total_score") for row in oracle])
    return {
        "budget_mode": budget.name,
        "sample_index": spec["sample_index"],
        "case_type": spec["case_type"],
        "pair_type": spec["pair_type"],
        "selected_candidate_id": spec["selected_row"].get("candidate_id"),
        "oracle_candidate_id": spec["oracle_row"].get("candidate_id"),
        "selected_sequence": spec["selected_sequence"],
        "oracle_sequence": spec["oracle_sequence"],
        "selected_source": spec["selected_source"],
        "oracle_source": spec["oracle_source"],
        "baseline_stored_gap": spec.get("baseline_stored_gap"),
        "baseline_stored_active_gap": spec.get("baseline_stored_active_gap"),
        "baseline_stored_weak_gap": spec.get("baseline_stored_weak_gap"),
        "mean_selected_total_score": selected_total["mean"],
        "std_selected_total_score": selected_total["std"],
        "mean_oracle_total_score": oracle_total["mean"],
        "std_oracle_total_score": oracle_total["std"],
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "gap_z_score": None if mean_gap is None or std_gap is None else abs(mean_gap) / (std_gap + EPS),
        "gap_sign_flip_fraction": None if not signs else sign_flips / len(signs),
        "noise_dominated_bool": noise_dominated,
        "oracle_wins_count": oracle_wins,
        "selected_wins_count": selected_wins,
        "ordering_stable_bool": stable,
        "ordering_stable_fraction": 1.0 if stable else 0.0,
        "mean_active_gap": active_stats["mean"],
        "std_active_gap": active_stats["std"],
        "active_gap_sign_flip_fraction": sign_flip_fraction(active_gaps),
        "mean_weak_gap": weak_stats["mean"],
        "std_weak_gap": weak_stats["std"],
        "weak_gap_sign_flip_fraction": sign_flip_fraction(weak_gaps),
        "active_noise_share": active_share,
        "weak_noise_share": weak_share,
        "segment_noise_driver": segment_driver(active_stats["std"], weak_stats["std"]),
        "stable_clear_bool": stable and not noise_dominated,
    }


def add_improvements(metrics_by_budget: dict[str, list[dict[str, Any]]]) -> None:
    b0 = {
        (row["sample_index"], row["pair_type"]): row
        for row in metrics_by_budget.get("B0_current_budget", [])
    }
    for budget_name, rows in metrics_by_budget.items():
        for row in rows:
            base = b0.get((row["sample_index"], row["pair_type"]))
            if base is None or budget_name == "B0_current_budget":
                row.update(
                    {
                        "std_gap_reduction_vs_B0": None,
                        "gap_z_score_delta_vs_B0": None,
                        "ordering_stability_delta_vs_B0": None,
                        "noise_dominated_to_stable_clear_bool": False,
                        "driver_changed_bool": False,
                    }
                )
                continue
            row.update(
                {
                    "std_gap_reduction_vs_B0": (
                        None if row.get("std_gap") is None or base.get("std_gap") is None else base["std_gap"] - row["std_gap"]
                    ),
                    "gap_z_score_delta_vs_B0": (
                        None if row.get("gap_z_score") is None or base.get("gap_z_score") is None else row["gap_z_score"] - base["gap_z_score"]
                    ),
                    "ordering_stability_delta_vs_B0": int(bool(row.get("ordering_stable_bool"))) - int(bool(base.get("ordering_stable_bool"))),
                    "noise_dominated_to_stable_clear_bool": bool(base.get("noise_dominated_bool")) and bool(row.get("stable_clear_bool")),
                    "driver_changed_bool": row.get("segment_noise_driver") != base.get("segment_noise_driver"),
                }
            )


def aggregate_budget(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    driver_counts = Counter(row.get("segment_noise_driver") for row in rows)
    return {
        "ordering_stable_count": sum(1 for row in rows if row.get("ordering_stable_bool")),
        "ordering_unstable_count": sum(1 for row in rows if not row.get("ordering_stable_bool")),
        "noise_dominated_pair_count": sum(1 for row in rows if row.get("noise_dominated_bool")),
        "stable_clear_pair_count": sum(1 for row in rows if row.get("stable_clear_bool")),
        "active_noise_driver_count": driver_counts["active"],
        "weak_noise_driver_count": driver_counts["weak"],
        "mixed_noise_driver_count": driver_counts["mixed"],
        "unknown_noise_driver_count": driver_counts["unknown"],
        "rescue_noise_dominated_count": sum(1 for row in rows if row.get("case_type") == "V4_rescue" and row.get("noise_dominated_bool")),
        "harm_noise_dominated_count": sum(1 for row in rows if row.get("case_type") == "V4_harm" and row.get("noise_dominated_bool")),
        "oracle_recovers_majority_count": sum(1 for row in rows if row.get("oracle_wins_count", 0) > repeats / 2),
        "mean_gap_z_score": statistics.mean(
            [row["gap_z_score"] for row in rows if row.get("gap_z_score") is not None]
        ) if any(row.get("gap_z_score") is not None for row in rows) else None,
    }


def decide(summary: dict[str, Any], pair_count: int) -> tuple[str, str]:
    b0 = summary["budget_summaries"]["B0_current_budget"]
    best_name = summary["path_budget_improvement"]["best_budget_mode"]
    best = summary["budget_summaries"][best_name]
    stable_gain = best["ordering_stable_count"] - b0["ordering_stable_count"]
    noise_reduction = b0["noise_dominated_pair_count"] - best["noise_dominated_pair_count"]
    if best["ordering_stable_count"] >= max(1, int(np.ceil(0.75 * pair_count))) and best["noise_dominated_pair_count"] <= int(np.floor(0.25 * pair_count)):
        return (
            "A",
            "Increasing path budget clearly stabilizes most pair orderings. Recommend a future fingerprint-caching/logging infrastructure diagnostic, not a scorer change.",
        )
    if stable_gain > 0 or noise_reduction > 0:
        return (
            "B",
            "Higher path budget improves stability somewhat but not enough. Keep scorer calibration closed and consider targeted active/weak fingerprint variance reduction diagnostics.",
        )
    return (
        "C",
        "Higher path budget does not improve stability. Candidate fingerprint noise appears structural or segment-design dominated; return to drift-span/candidate-generation diagnostics before scorer changes.",
    )


def summarize_result(pair_specs: list[dict[str, Any]], metrics_by_budget: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    repeats_by_budget = {budget.name: budget.repeats for budget in BUDGETS}
    budget_summaries = {
        budget.name: aggregate_budget(metrics_by_budget[budget.name], budget.repeats)
        for budget in BUDGETS
    }
    b0 = budget_summaries["B0_current_budget"]
    high_budget_names = [budget.name for budget in BUDGETS if budget.name != "B0_current_budget"]
    best_budget = max(
        high_budget_names,
        key=lambda name: (
            budget_summaries[name]["ordering_stable_count"],
            -budget_summaries[name]["noise_dominated_pair_count"],
            budget_summaries[name]["mean_gap_z_score"] or 0.0,
        ),
    )
    improvement = {}
    for name in high_budget_names:
        improvement[f"{name[:2]}_vs_B0_stable_gain"] = (
            budget_summaries[name]["ordering_stable_count"] - b0["ordering_stable_count"]
        )
        improvement[f"{name[:2]}_vs_B0_noise_dominated_reduction"] = (
            b0["noise_dominated_pair_count"] - budget_summaries[name]["noise_dominated_pair_count"]
        )
    improvement.update(
        {
            "best_budget_mode": best_budget,
            "best_budget_stable_count": budget_summaries[best_budget]["ordering_stable_count"],
            "best_budget_noise_dominated_count": budget_summaries[best_budget]["noise_dominated_pair_count"],
            "best_budget_mean_gap_z_score": budget_summaries[best_budget]["mean_gap_z_score"],
        }
    )
    summary = {
        "samples_analyzed": len({spec["sample_index"] for spec in pair_specs}),
        "candidate_pairs_analyzed": len(pair_specs),
        "budget_modes_available": [budget.name for budget in BUDGETS],
        "budget_modes_unavailable": [],
        "budget_summaries": budget_summaries,
        "path_budget_improvement": improvement,
        "repeats_by_budget": repeats_by_budget,
        "model_decoding_avoided": True,
        "candidate_generation_avoided": True,
        "sde_validation_probe_avoided": True,
        "formal_eval_recommended": False,
    }
    decision, recommendation = decide(summary, len(pair_specs))
    summary["decision"] = decision
    summary["recommendation"] = recommendation
    return summary


def blocked_result(args: argparse.Namespace, estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {
            "stability_json": str(args.stability_json),
            "ambiguity_json": str(args.ambiguity_json),
            "all32_smoke_json": str(args.all32_smoke_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "target_case_types": sorted(parse_case_types(args.target_case_types)),
        },
        "summary": {
            "decision": "D",
            "recommendation": "Estimated path-budget diagnostic runtime exceeds the Codex budget. Run the standalone local script instead.",
            "estimated_runtime": estimate,
            "formal_eval_recommended": False,
        },
        "pairs": [],
    }


def build_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# P2 Candidate Fingerprint Path-Budget Diagnostic",
        "",
        "## Why This Diagnostic Was Run",
        "",
        "The previous fingerprint stability diagnostic found that candidate",
        "resimulation alone made V4 rescue/harm oracle-pair ordering unstable. This",
        "helper checks whether a small fixed increase in candidate fingerprint path",
        "budget stabilizes those same pairs. No model decoding, candidate generation,",
        "`sde_validation_probe.py`, formal eval, scorer mode, or production default",
        "change was run.",
        "",
    ]
    if summary.get("decision") == "D":
        lines.extend(
            [
                "## Blocked",
                "",
                f"Estimated runtime: `{summary.get('estimated_runtime')}`",
                "",
                f"Decision `D`: {summary['recommendation']}",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Summary",
            "",
            "| Item | Value |",
            "| --- | ---: |",
            f"| Samples analyzed | {summary['samples_analyzed']} |",
            f"| Candidate pairs analyzed | {summary['candidate_pairs_analyzed']} |",
            f"| Budget modes available | {', '.join(summary['budget_modes_available'])} |",
            f"| Budget modes unavailable | {', '.join(summary['budget_modes_unavailable']) or 'none'} |",
            "",
            "## Per-Budget Ordering Stability",
            "",
            "| Budget | Stable | Unstable | Noise dominated | Stable clear | Active driver | Weak driver | Mixed driver | Mean gap z |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for budget in BUDGETS:
        row = summary["budget_summaries"][budget.name]
        lines.append(
            "| {name} | {stable} | {unstable} | {noise} | {clear} | {active} | {weak} | {mixed} | {z} |".format(
                name=budget.name,
                stable=row["ordering_stable_count"],
                unstable=row["ordering_unstable_count"],
                noise=row["noise_dominated_pair_count"],
                clear=row["stable_clear_pair_count"],
                active=row["active_noise_driver_count"],
                weak=row["weak_noise_driver_count"],
                mixed=row["mixed_noise_driver_count"],
                z=fmt(row.get("mean_gap_z_score")),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Pair Path-Budget Metrics",
            "",
            "| Sample | Case | Pair | Budget | Gap mean | Gap std | Z | Stable | Noise dominated | Driver |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for pair in result["pairs"]:
        for budget in BUDGETS:
            metrics = pair["budget_metrics"][budget.name]
            lines.append(
                "| {idx} | {case} | {pair_type} | {budget} | {mean} | {std} | {z} | {stable} | {noise} | {driver} |".format(
                    idx=pair["sample_index"],
                    case=pair["case_type"],
                    pair_type=pair["pair_type"],
                    budget=budget.name,
                    mean=fmt(metrics.get("mean_gap")),
                    std=fmt(metrics.get("std_gap")),
                    z=fmt(metrics.get("gap_z_score")),
                    stable=fmt(metrics.get("ordering_stable_bool")),
                    noise=fmt(metrics.get("noise_dominated_bool")),
                    driver=metrics.get("segment_noise_driver"),
                )
            )
    improvement = summary["path_budget_improvement"]
    lines.extend(
        [
            "",
            "## Path-Budget Improvement",
            "",
            "| Comparison | Stable gain | Noise-dominated reduction |",
            "| --- | ---: | ---: |",
            f"| B1 vs B0 | {improvement['B1_vs_B0_stable_gain']} | {improvement['B1_vs_B0_noise_dominated_reduction']} |",
            f"| B2 vs B0 | {improvement['B2_vs_B0_stable_gain']} | {improvement['B2_vs_B0_noise_dominated_reduction']} |",
            f"| B3 vs B0 | {improvement['B3_vs_B0_stable_gain']} | {improvement['B3_vs_B0_noise_dominated_reduction']} |",
            "",
            f"Best budget mode: `{improvement['best_budget_mode']}`.",
            "",
            "## Decision",
            "",
            f"Decision `{summary['decision']}`: {summary['recommendation']}",
            "",
            "No formal eval, no rerank mode, no P2 eval integration, and no scorer",
            "change are recommended from this diagnostic alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    stability_payload = load_json(args.stability_json)
    ambiguity_payload = load_json(args.ambiguity_json)
    sidecar_payload = load_json(args.scorer_ready_json)
    load_json(args.all32_smoke_json)
    pair_specs = build_pair_specs(
        stability_payload,
        ambiguity_payload,
        sidecar_payload,
        parse_case_types(args.target_case_types),
    )
    estimate = estimate_runtime(pair_specs, sidecar_payload)
    if estimate["estimated_seconds"] > MAX_CODEX_SECONDS:
        result = blocked_result(args, estimate)
        args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
        args.report.write_text(build_markdown(json_ready(result)))
        print("decision=D")
        print(f"estimated_seconds={estimate['estimated_seconds']:.3f}")
        return

    metrics_by_budget: dict[str, list[dict[str, Any]]] = {}
    for budget in BUDGETS:
        series_by_pair = score_budget_mode(pair_specs, sidecar_payload, budget)
        metrics_by_budget[budget.name] = [
            pair_budget_metrics(spec, budget, series_by_pair[(int(spec["sample_index"]), spec["pair_type"])])
            for spec in pair_specs
        ]
    add_improvements(metrics_by_budget)
    summary = summarize_result(pair_specs, metrics_by_budget)
    pairs = []
    for spec in pair_specs:
        pairs.append(
            {
                "sample_index": spec["sample_index"],
                "case_type": spec["case_type"],
                "pair_type": spec["pair_type"],
                "selected_candidate": candidate_brief(spec["selected_row"]),
                "oracle_candidate": candidate_brief(spec["oracle_row"]),
                "selected_sequence": spec["selected_sequence"],
                "oracle_sequence": spec["oracle_sequence"],
                "selected_source": spec["selected_source"],
                "oracle_source": spec["oracle_source"],
                "baseline_stored_gap": spec.get("baseline_stored_gap"),
                "baseline_stored_active_gap": spec.get("baseline_stored_active_gap"),
                "baseline_stored_weak_gap": spec.get("baseline_stored_weak_gap"),
                "budget_metrics": {
                    budget.name: next(
                        row for row in metrics_by_budget[budget.name]
                        if row["sample_index"] == spec["sample_index"] and row["pair_type"] == spec["pair_type"]
                    )
                    for budget in BUDGETS
                },
            }
        )
    result = {
        "metadata": {
            "stability_json": str(args.stability_json),
            "ambiguity_json": str(args.ambiguity_json),
            "all32_smoke_json": str(args.all32_smoke_json),
            "scorer_ready_json": str(args.scorer_ready_json),
            "target_case_types": sorted(parse_case_types(args.target_case_types)),
            "runtime_estimate": estimate,
            "score_kind": "constant_grid_rolewise_no_multi_u0",
            "active_weak_weights": [1.0, 1.0],
            "constant_values": CONSTANT_VALUES,
        },
        "summary": summary,
        "pairs": pairs,
    }
    args.json_output.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    args.report.write_text(build_markdown(json_ready(result)))
    print(f"decision={summary['decision']}")
    print(f"samples_analyzed={summary['samples_analyzed']}")
    print(f"candidate_pairs_analyzed={summary['candidate_pairs_analyzed']}")
    for budget in BUDGETS:
        row = summary["budget_summaries"][budget.name]
        print(
            f"{budget.name}: stable={row['ordering_stable_count']} "
            f"noise_dominated={row['noise_dominated_pair_count']} "
            f"stable_clear={row['stable_clear_pair_count']}"
        )
    print(f"best_budget_mode={summary['path_budget_improvement']['best_budget_mode']}")


if __name__ == "__main__":
    main()
