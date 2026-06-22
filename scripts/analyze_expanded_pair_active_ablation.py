#!/usr/bin/env python3
"""Offline active-residual score ablations for expanded-pairing misses."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sde_fingerprint import FingerprintConfig

from scripts.analyze_rolewise_residuals import (
    parse_constant_values,
    reproduce_eval_targets,
    score_combo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--residual-json",
        type=Path,
        default=Path("expanded_pairing_active_residual_trap_diagnostics.json"),
    )
    parser.add_argument(
        "--miss-json",
        type=Path,
        default=Path("expanded_pairing_oracle_miss_ranking_diagnostics.json"),
    )
    parser.add_argument(
        "--pruning-json",
        type=Path,
        default=Path("expanded_pairing_wrong_pair_pruning_diagnostics.json"),
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("candidate_coverage_pair_expanded.json"),
    )
    parser.add_argument(
        "--source-log",
        type=Path,
        default=Path("/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("expanded_pairing_active_residual_ablation.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("expanded_pairing_active_residual_ablation.json"),
    )
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=800)
    parser.add_argument("--active-paths", type=int, default=800)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--constant-values", type=str, default="0.25,0.5,1.0,2.0,4.0")
    return parser.parse_args()


def l2(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def clipped_l2(values: np.ndarray, cap: float) -> float:
    return float(np.sqrt(np.sum(np.minimum(values * values, cap * cap))))


def huber_norm(values: np.ndarray, delta: float) -> float:
    abs_values = np.abs(values)
    loss = np.where(
        abs_values <= delta,
        0.5 * values * values,
        delta * (abs_values - 0.5 * delta),
    )
    return float(np.sqrt(max(0.0, 2.0 * float(np.sum(loss)))))


def topk_removed_l2(values: np.ndarray, k: int) -> float:
    squared = values * values
    if k <= 0:
        return float(np.sqrt(np.sum(squared)))
    if k >= squared.size:
        return 0.0
    kept = np.sort(squared)[: squared.size - k]
    return float(np.sqrt(np.sum(kept)))


def combo_arrays(combo: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(combo["active_residual"], dtype=np.float64),
        np.asarray(combo["weak_residual"], dtype=np.float64),
    )


def robust_cap(left_active: np.ndarray, right_active: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.abs(np.concatenate([left_active, right_active])), percentile))


def normalize_active(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return values / np.maximum(scales, 1e-8)


def score_variants(left: dict, right: dict, active_scales: np.ndarray) -> dict[str, tuple[float, float, str]]:
    left_active, left_weak = combo_arrays(left)
    right_active, right_weak = combo_arrays(right)
    p90 = robust_cap(left_active, right_active, 90)
    p95 = robust_cap(left_active, right_active, 95)
    weak_left = l2(left_weak)
    weak_right = l2(right_weak)
    return {
        "current_recomputed": (
            float(left["total"]),
            float(right["total"]),
            "recomputed active L2 + weak L2",
        ),
        "active_clipped_l2_p95_plus_weak": (
            clipped_l2(left_active, p95) + weak_left,
            clipped_l2(right_active, p95) + weak_right,
            f"active residuals capped at per-case p95={p95:.4g}; weak unchanged",
        ),
        "active_clipped_l2_p90_plus_weak": (
            clipped_l2(left_active, p90) + weak_left,
            clipped_l2(right_active, p90) + weak_right,
            f"active residuals capped at per-case p90={p90:.4g}; weak unchanged",
        ),
        "active_huber_p90_plus_weak": (
            huber_norm(left_active, p90) + weak_left,
            huber_norm(right_active, p90) + weak_right,
            f"active Huber norm with delta=p90={p90:.4g}; weak unchanged",
        ),
        "active_top1_removed_plus_weak": (
            topk_removed_l2(left_active, 1) + weak_left,
            topk_removed_l2(right_active, 1) + weak_right,
            "remove largest active residual dimension; weak unchanged",
        ),
        "active_top2_removed_plus_weak": (
            topk_removed_l2(left_active, 2) + weak_left,
            topk_removed_l2(right_active, 2) + weak_right,
            "remove largest 2 active residual dimensions; weak unchanged",
        ),
        "per_active_dim_norm_plus_weak": (
            l2(normalize_active(left_active, active_scales)) + weak_left,
            l2(normalize_active(right_active, active_scales)) + weak_right,
            "normalize active dimensions by std over selected/oracle miss residuals; weak unchanged",
        ),
        "weak_only_diagnostic": (
            weak_left,
            weak_right,
            "weak segment only; diagnostic, not a proposed scorer",
        ),
    }


def parse_debug_hit_cases(path: Path) -> list[dict]:
    text = path.read_text()
    if "rerank_debug:" not in text:
        return []
    section = text.split("rerank_debug:", 1)[1]
    starts = list(re.finditer(r"^sample_index=(\d+)$", section, flags=re.MULTILINE))
    hits = []
    for pos, match in enumerate(starts):
        sample_idx = int(match.group(1))
        end = starts[pos + 1].start() if pos + 1 < len(starts) else len(section)
        block = section[match.start() : end]
        if "oracle_rank=1" not in block:
            continue
        rank_rows = parse_candidate_rank_rows(block)
        rank1 = rank_rows.get(1)
        rank2 = rank_rows.get(2)
        if rank1 is None or rank2 is None or not rank1["exact"] or rank2["exact"]:
            continue
        truth_match = re.search(r"^truth: (.+)$", block, flags=re.MULTILINE)
        if truth_match is None:
            continue
        hits.append(
            {
                "sample_index": sample_idx,
                "truth": truth_match.group(1).strip(),
                "oracle": rank1,
                "challenger": rank2,
            }
        )
    return hits


def parse_candidate_rank_rows(block: str) -> dict[int, dict]:
    lines = block.splitlines()
    rows: dict[int, dict] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("candidate_rank="):
            idx += 1
            continue
        values = parse_key_values(line)
        rank = int(values["candidate_rank"])
        drift = None
        diffusion = None
        if idx + 1 < len(lines) and lines[idx + 1].startswith("drift_tokens: "):
            drift = lines[idx + 1].split(": ", 1)[1].strip()
        if idx + 2 < len(lines) and lines[idx + 2].startswith("diffusion_tokens: "):
            diffusion = lines[idx + 2].split(": ", 1)[1].strip()
        if drift is not None and diffusion is not None:
            rows[rank] = {
                "rank": rank,
                "source": values.get("source", "unknown"),
                "distance": float(values["distance"]),
                "exact": values.get("exact") == "1",
                "relaxed": values.get("relaxed_no_constants") == "1",
                "drift": drift.split(),
                "diffusion": diffusion.split(),
            }
        idx += 1
    return rows


def parse_key_values(text: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", text))


def active_scales_from_cases(cases: list[dict]) -> np.ndarray:
    rows = []
    for case in cases:
        diag = case["diagnostics"]
        rows.append(np.asarray(diag["selected_score"]["active_residual"], dtype=np.float64))
        rows.append(np.asarray(diag["oracle_score"]["active_residual"], dtype=np.float64))
    matrix = np.vstack(rows)
    scales = np.std(matrix, axis=0)
    fallback = float(np.median(scales[scales > 0])) if np.any(scales > 0) else 1.0
    return np.where(scales > 1e-8, scales, fallback)


def miss_variant_rows(cases: list[dict], active_scales: np.ndarray) -> list[dict]:
    rows = []
    for case in cases:
        diag = case["diagnostics"]
        selected = diag["selected_score"]
        oracle = diag["oracle_score"]
        variants = score_variants(selected, oracle, active_scales)
        variant_rows = [
            {
                "name": "current_logged_rolewise",
                "selected": float(case["selected_score"]),
                "oracle": float(case["oracle_score"]),
                "oracle_wins": float(case["oracle_score"]) < float(case["selected_score"]),
                "margin": float(case["selected_score"]) - float(case["oracle_score"]),
                "note": "formal log role-wise score; baseline selected wins by construction for these misses",
            }
        ]
        for name, (selected_score, oracle_score, note) in variants.items():
            oracle_wins = oracle_score < selected_score
            variant_rows.append(
                {
                    "name": name,
                    "selected": selected_score,
                    "oracle": oracle_score,
                    "oracle_wins": oracle_wins,
                    "margin": selected_score - oracle_score,
                    "note": note,
                }
            )
        enriched = {
            "sample_index": case["sample_index"],
            "selected_source": case["selected_source"],
            "oracle_source": case["oracle_source"],
            "oracle_only_pair": bool(case["oracle_only_pair"]),
            "wrong_selected_pair": case["selected_source"] == "pair",
            "trap_shape": diag["active_side"]["trap_shape"],
            "weak_favors_oracle": diag["logged_weak_gap"] < 0,
            "score_gap": case["score_gap"],
            "variants": variant_rows,
        }
        rows.append(enriched)
    return rows


def recompute_hit_harm_cases(
    hit_cases: list[dict],
    args: argparse.Namespace,
    active_scales: np.ndarray,
) -> list[dict]:
    if not hit_cases:
        return []
    rows_for_repro = {case["sample_index"]: {"truth": case["truth"]} for case in hit_cases}
    sample_ids = [case["sample_index"] for case in hit_cases]
    config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    targets = reproduce_eval_targets(sample_ids, args.eval_seed, config, rows_for_repro)
    constants = parse_constant_values(args.constant_values)
    results = []
    for case in hit_cases:
        sample_idx = case["sample_index"]
        base_seed = args.eval_seed + 17
        base_seed += (sample_idx // args.batch_size) * 100003
        base_seed += (sample_idx % args.batch_size) * 1009
        target_y = targets[sample_idx]["target_y"]
        oracle = score_combo(
            "oracle_hit",
            case["oracle"]["drift"],
            case["oracle"]["diffusion"],
            target_y,
            config,
            constants,
            base_seed,
        )
        challenger = score_combo(
            "top2_challenger",
            case["challenger"]["drift"],
            case["challenger"]["diffusion"],
            target_y,
            config,
            constants,
            base_seed + 1,
        )
        if not oracle.get("success") or not challenger.get("success"):
            continue
        variants = score_variants(challenger, oracle, active_scales)
        variant_rows = [
            {
                "name": "current_logged_rolewise",
                "challenger": float(case["challenger"]["distance"]),
                "oracle": float(case["oracle"]["distance"]),
                "hit_harmed": float(case["challenger"]["distance"])
                < float(case["oracle"]["distance"]),
                "margin": float(case["oracle"]["distance"]) - float(case["challenger"]["distance"]),
                "note": "formal log role-wise score for debug top-2 candidates",
            }
        ]
        for name, (challenger_score, oracle_score, note) in variants.items():
            harmed = challenger_score < oracle_score
            variant_rows.append(
                {
                    "name": name,
                    "challenger": challenger_score,
                    "oracle": oracle_score,
                    "hit_harmed": harmed,
                    "margin": oracle_score - challenger_score,
                    "note": note,
                }
            )
        results.append(
            {
                "sample_index": sample_idx,
                "oracle_source": case["oracle"]["source"],
                "challenger_source": case["challenger"]["source"],
                "oracle_is_pair": case["oracle"]["source"] == "pair",
                "variants": variant_rows,
            }
        )
    return results


def summarize_variant(
    name: str,
    miss_rows: list[dict],
    hit_rows: list[dict],
    pair_oracle_indices: set[int],
) -> dict:
    miss_flips = [
        row["sample_index"]
        for row in miss_rows
        if variant_by_name(row, name)["oracle_wins"]
    ]
    oracle_only_flips = [
        row["sample_index"]
        for row in miss_rows
        if row["oracle_only_pair"] and variant_by_name(row, name)["oracle_wins"]
    ]
    wrong_pair_flips = [
        row["sample_index"]
        for row in miss_rows
        if row["wrong_selected_pair"] and variant_by_name(row, name)["oracle_wins"]
    ]
    hit_harms = [
        row["sample_index"]
        for row in hit_rows
        if variant_by_name(row, name)["hit_harmed"]
    ]
    pair_hit_harms = [
        row["sample_index"]
        for row in hit_rows
        if row["oracle_is_pair"] and variant_by_name(row, name)["hit_harmed"]
    ]
    pair_oracle_flips = [idx for idx in miss_flips if idx in pair_oracle_indices]
    return {
        "name": name,
        "oracle_wins_12": len(miss_flips),
        "oracle_wins_oracle_only_pair_7": len(oracle_only_flips),
        "oracle_wins_wrong_selected_pair_8": len(wrong_pair_flips),
        "flipped_samples": miss_flips,
        "oracle_only_pair_flipped_samples": oracle_only_flips,
        "wrong_pair_flipped_samples": wrong_pair_flips,
        "known_hit_harms_top2": len(hit_harms),
        "known_hit_harmed_samples_top2": hit_harms,
        "pair_oracle_hit_harms_top2": len(pair_hit_harms),
        "pair_oracle_hit_harmed_samples_top2": pair_hit_harms,
        "pair_oracle_miss_rescues": len(pair_oracle_flips),
        "pair_oracle_miss_rescued_samples": pair_oracle_flips,
    }


def variant_by_name(row: dict, name: str) -> dict:
    for item in row["variants"]:
        if item["name"] == name:
            return item
    raise KeyError(name)


def decision(variant_summary: list[dict], hit_rows: list[dict]) -> tuple[str, str]:
    clipped_names = {
        "active_clipped_l2_p95_plus_weak",
        "active_clipped_l2_p90_plus_weak",
        "active_huber_p90_plus_weak",
        "active_top1_removed_plus_weak",
        "active_top2_removed_plus_weak",
    }
    robust_names = clipped_names | {"per_active_dim_norm_plus_weak"}
    best = max(
        (row for row in variant_summary if row["name"] in robust_names),
        key=lambda row: row["oracle_wins_12"],
    )
    best_clipped = max(
        (row for row in variant_summary if row["name"] in clipped_names),
        key=lambda row: row["oracle_wins_12"],
    )
    harms = best["known_hit_harms_top2"] if hit_rows else 0
    clipped_harms = best_clipped["known_hit_harms_top2"] if hit_rows else 0
    if best["name"] == "per_active_dim_norm_plus_weak" and best["oracle_wins_12"] >= 2:
        return (
            "C",
            "Per-dimension normalization rescues multiple misses, but it is calibrated only from miss residuals; recommend richer residual logging in a future 8- or 16-sample smoke before any scorer change.",
        )
    if best_clipped["oracle_wins_12"] >= 2 and clipped_harms == 0:
        return (
            "A",
            "Offline robust/clipped active score rescues multiple misses without harming the limited top-2 hit check; recommend one future 8- or 16-sample smoke with logging, not a full formal eval.",
        )
    if best["oracle_wins_12"] <= 1 or harms > 0:
        return (
            "B",
            "Offline robust/clipped active score rescues only one case or harms known hits; do not add a rerank mode.",
        )
    return (
        "D",
        "Weak-favoring cases are promising but mixed; keep analysis offline.",
    )


def write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    variant_summary = payload["variant_summary"]
    miss_rows = payload["miss_rows"]
    hit_rows = payload["hit_harm_rows"]
    lines = [
        "# Expanded Pairing Active Residual Ablation",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: offline scorer-side ablation using existing expanded-pairing residual vectors. "
        "No `sde_validation_probe.py`, model decoding, candidate regeneration, formal eval, 64-sample eval, grid, retraining, or rerank-mode change was run.",
        "",
        "## Executive Summary",
        "",
        f"- Selected-miss cases analyzed: `{summary['miss_cases']}`.",
        f"- Oracle-only-pair misses: `{summary['oracle_only_pair_misses']}`.",
        f"- Wrong selected pair cases: `{summary['wrong_selected_pair_cases']}`.",
        f"- Known selected hits checked against debug top-2 challenger: `{summary['known_hit_checks']}`.",
        f"- Pair-oracle hit checks: `{summary['pair_oracle_hit_checks']}`.",
        f"- Best offline variant: `{summary['best_robust_variant']}` with "
        f"`{summary['best_robust_oracle_wins']}/12` miss rescues and "
        f"`{summary['best_robust_oracle_only_wins']}/7` oracle-only-pair rescues.",
        f"- Decision: **{summary['decision_code']}**. {summary['decision_text']}",
        "",
        "Default stance remains: no formal eval and no new rerank mode from this offline evidence.",
        "",
        "## Variant Summary",
        "",
        "| Variant | Oracle wins /12 | Oracle-only pair wins /7 | Wrong pair wins /8 | Flipped samples | Top-2 hit harms | Pair-oracle hit harms |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in variant_summary:
        lines.append(
            f"| `{row['name']}` | {row['oracle_wins_12']} | "
            f"{row['oracle_wins_oracle_only_pair_7']} | "
            f"{row['oracle_wins_wrong_selected_pair_8']} | "
            f"`{','.join(map(str, row['flipped_samples'])) or 'none'}` | "
            f"{row['known_hit_harms_top2']} | {row['pair_oracle_hit_harms_top2']} |"
        )
    lines.extend(
        [
            "",
            "## Miss Case Flips",
            "",
            "| Sample | Oracle-only pair | Wrong selected pair | Trap shape | Weak favors oracle | Flipping variants |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in miss_rows:
        flipping = [item["name"] for item in row["variants"] if item["oracle_wins"]]
        lines.append(
            f"| {row['sample_index']} | {'yes' if row['oracle_only_pair'] else 'no'} | "
            f"{'yes' if row['wrong_selected_pair'] else 'no'} | {row['trap_shape']} | "
            f"{'yes' if row['weak_favors_oracle'] else 'no'} | "
            f"`{', '.join(flipping) if flipping else 'none'}` |"
        )
    lines.extend(
        [
            "",
            "## Known Hit Harm Check",
            "",
            "This is a limited check against the debug top-2 challenger for selected-hit samples. It is not a full candidate-pool safety proof.",
            "",
            "| Sample | Oracle source | Challenger source | Harmed variants |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for row in hit_rows:
        harmed = [item["name"] for item in row["variants"] if item["hit_harmed"]]
        lines.append(
            f"| {row['sample_index']} | {row['oracle_source']} | {row['challenger_source']} | "
            f"`{', '.join(harmed) if harmed else 'none'}` |"
        )
    if not hit_rows:
        lines.append("| n/a | n/a | n/a | `not recoverable from residual JSON/log` |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Robust/clipped active scoring is not yet compelling as a real rerank mode. The clipped/Huber/top-k variants rescue several misses but harm one debug top-2 selected-hit check.",
            "- Per-active-dimension normalization is the strongest offline signal, but it is calibrated only across the miss residual vectors; it is diagnostic evidence, not a stable calibration estimate.",
            "- Weak-only behavior confirms that weak sometimes favors oracles, but the signal is mixed and should not trigger another broad active/weak grid.",
            "- Pair-oracle coverage itself is not harmed by an offline score, but selected pair-oracle hits are only checked against top-2 challengers here.",
            "",
            "## Recommendation",
            "",
            "Do not add a rerank mode and do not run a formal 32-sample eval. The next useful direction is drift span diversity or deeper active-fingerprint analysis. If scorer work continues, first add richer residual logging in a small smoke so hit-safety and full candidate-pool effects can be checked offline.",
            "",
            "## Provenance",
            "",
            f"- Residual JSON: `{payload['metadata']['residual_json']}`.",
            f"- Miss JSON: `{payload['metadata']['miss_json']}`.",
            f"- Source log: `{payload['metadata']['source_log']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    residual_payload = json.loads(args.residual_json.read_text())
    miss_payload = json.loads(args.miss_json.read_text())
    pruning_payload = json.loads(args.pruning_json.read_text())
    coverage_payload = json.loads(args.coverage_json.read_text())
    cases = sorted(residual_payload["cases"], key=lambda item: item["sample_index"])
    active_scales = active_scales_from_cases(cases)
    miss_rows = miss_variant_rows(cases, active_scales)
    hit_cases = parse_debug_hit_cases(args.source_log)
    hit_harm_rows = recompute_hit_harm_cases(hit_cases, args, active_scales)
    variant_names = [item["name"] for item in miss_rows[0]["variants"]]
    pair_oracle_indices = {
        row["sample_index"] for row in pruning_payload.get("pair_oracle_samples", [])
    }
    variant_summary = [
        summarize_variant(name, miss_rows, hit_harm_rows, pair_oracle_indices)
        for name in variant_names
    ]
    robust_names = {
        "active_clipped_l2_p95_plus_weak",
        "active_clipped_l2_p90_plus_weak",
        "active_huber_p90_plus_weak",
        "active_top1_removed_plus_weak",
        "active_top2_removed_plus_weak",
        "per_active_dim_norm_plus_weak",
    }
    best_robust = max(
        (row for row in variant_summary if row["name"] in robust_names),
        key=lambda row: (row["oracle_wins_12"], row["oracle_wins_oracle_only_pair_7"]),
    )
    decision_code, decision_text = decision(variant_summary, hit_harm_rows)
    summary = {
        "miss_cases": len(miss_rows),
        "oracle_only_pair_misses": sum(row["oracle_only_pair"] for row in miss_rows),
        "wrong_selected_pair_cases": sum(row["wrong_selected_pair"] for row in miss_rows),
        "known_hit_checks": len(hit_harm_rows),
        "pair_oracle_hit_checks": sum(row["oracle_is_pair"] for row in hit_harm_rows),
        "pair_oracle_samples": len(pair_oracle_indices),
        "expanded_full_oracle_present": sum(
            1 for sample in coverage_payload.get("samples", []) if sample.get("has_full")
        ),
        "variants_tested": variant_names,
        "best_robust_variant": best_robust["name"],
        "best_robust_oracle_wins": best_robust["oracle_wins_12"],
        "best_robust_oracle_only_wins": best_robust["oracle_wins_oracle_only_pair_7"],
        "best_robust_hit_harms": best_robust["known_hit_harms_top2"],
        "decision_code": decision_code,
        "decision_text": decision_text,
        "weak_favors_oracle_count": sum(row["weak_favors_oracle"] for row in miss_rows),
    }
    payload = {
        "metadata": {
            "residual_json": str(args.residual_json),
            "miss_json": str(args.miss_json),
            "pruning_json": str(args.pruning_json),
            "coverage_json": str(args.coverage_json),
            "source_log": str(args.source_log),
            "eval_seed": args.eval_seed,
            "n_paths": args.n_paths,
            "active_paths": args.active_paths,
            "n_steps": args.n_steps,
        },
        "summary": summary,
        "variant_summary": variant_summary,
        "miss_rows": miss_rows,
        "hit_harm_rows": hit_harm_rows,
    }
    write_report(args.report, payload)
    args.json_output.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"miss_cases={summary['miss_cases']}")
    print(f"oracle_only_pair_misses={summary['oracle_only_pair_misses']}")
    print(f"wrong_selected_pair_cases={summary['wrong_selected_pair_cases']}")
    print(f"known_hit_checks={summary['known_hit_checks']}")
    print(f"best_robust_variant={summary['best_robust_variant']}")
    print(f"best_robust_oracle_wins={summary['best_robust_oracle_wins']}")
    print(f"best_robust_oracle_only_wins={summary['best_robust_oracle_only_wins']}")
    print(f"decision={summary['decision_code']}")


if __name__ == "__main__":
    main()
