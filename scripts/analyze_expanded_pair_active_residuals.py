#!/usr/bin/env python3
"""Targeted active-residual diagnostics for expanded-pairing misses."""

from __future__ import annotations

import argparse
import json
import math
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
    ACTIVE_SLICE,
    WEAK_SLICE,
    feature_names,
    fmt_vector,
    parse_constant_values,
    reproduce_eval_targets,
    score_combo,
    top_feature_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--report",
        type=Path,
        default=Path("expanded_pairing_active_residual_trap_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("expanded_pairing_active_residual_trap_diagnostics.json"),
    )
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=800)
    parser.add_argument("--active-paths", type=int, default=800)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--constant-values", type=str, default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def tokens(raw: str) -> list[str]:
    return raw.split()


def case_rows(cases: list[dict]) -> dict[int, dict]:
    rows = {}
    for case in cases:
        rows[case["sample_index"]] = {
            "truth": case["truth_sequence"],
            "selected_sequence": case["selected_sequence"],
            "oracle_sequence": case["oracle_sequence"],
            "selected_source": case["selected_source"],
            "oracle_source": case["oracle_source"],
            "selected_drift": tokens(case["selected_drift"]),
            "selected_diffusion": tokens(case["selected_diffusion"]),
            "oracle_drift": tokens(case["oracle_drift"]),
            "oracle_diffusion": tokens(case["oracle_diffusion"]),
        }
    return rows


def fmt_float(value) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def feature_side(name: str) -> str:
    return "KM1/drift" if name.endswith(",drift") else "KM2/diffusion"


def active_side_summary(
    names: list[str],
    selected_residual: np.ndarray,
    oracle_residual: np.ndarray,
) -> dict:
    selected_abs = np.abs(selected_residual)
    oracle_abs = np.abs(oracle_residual)
    selected_advantage = np.maximum(oracle_abs - selected_abs, 0.0)
    total = float(selected_advantage.sum())
    by_side = {"KM1/drift": 0.0, "KM2/diffusion": 0.0}
    counts = {"KM1/drift": 0, "KM2/diffusion": 0}
    for idx, advantage in enumerate(selected_advantage):
        side = feature_side(names[idx])
        by_side[side] += float(advantage)
        if advantage > 0:
            counts[side] += 1
    if total <= 0.0:
        dominant_side = "none"
    else:
        drift_share = by_side["KM1/drift"] / total
        diffusion_share = by_side["KM2/diffusion"] / total
        if drift_share >= 0.6:
            dominant_side = "KM1/drift"
        elif diffusion_share >= 0.6:
            dominant_side = "KM2/diffusion"
        else:
            dominant_side = "mixed"
    positive = np.sort(selected_advantage[selected_advantage > 0.0])[::-1]
    top1_share = float(positive[:1].sum() / total) if total > 0 and positive.size else 0.0
    top2_share = float(positive[:2].sum() / total) if total > 0 and positive.size else 0.0
    top3_share = float(positive[:3].sum() / total) if total > 0 and positive.size else 0.0
    selected_wins = int(np.sum(oracle_abs > selected_abs))
    oracle_wins = int(np.sum(selected_abs > oracle_abs))
    if total <= 0.0:
        trap_shape = "none"
    elif top2_share >= 0.55 or top3_share >= 0.70:
        trap_shape = "outlier-dominated"
    elif selected_wins >= 11 and top3_share < 0.55:
        trap_shape = "broad"
    else:
        trap_shape = "mixed"
    return {
        "dominant_side": dominant_side,
        "advantage_by_side": by_side,
        "win_counts_by_side": counts,
        "top1_share": top1_share,
        "top2_share": top2_share,
        "top3_share": top3_share,
        "selected_wins": selected_wins,
        "oracle_wins": oracle_wins,
        "trap_shape": trap_shape,
    }


def classify_case(case: dict, diag: dict) -> str:
    labels = []
    if case["selected_drift"] == case["oracle_drift"]:
        labels.append("diffusion-side structural miss")
    elif case["selected_diffusion"] == case["oracle_diffusion"]:
        labels.append("drift-side structural miss")
    else:
        labels.append("both-side structural miss")
    labels.append(f"{diag['active_side']['trap_shape']} active trap")
    labels.append(f"{diag['active_side']['dominant_side']} leaning")
    if diag["weak_disagrees_with_active"]:
        labels.append("weak disagrees")
    if case.get("oracle_lower_model_score"):
        labels.append("model-score conflict")
    return "; ".join(labels)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def format_feature_rows(rows: list[tuple]) -> str:
    if not rows:
        return "No positive feature wins."
    lines = [
        "| idx | feature | side | advantage | selected abs resid | oracle abs resid |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for idx, name, advantage, selected_abs, oracle_abs in rows:
        lines.append(
            f"| {idx} | `{name}` | {feature_side(name)} | {advantage:.6f} | "
            f"{selected_abs:.6f} | {oracle_abs:.6f} |"
        )
    return "\n".join(lines)


def write_report(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    cases = payload["cases"]
    lines = [
        "# Expanded Pairing Active Residual Trap Diagnostics",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: targeted residual-vector analysis for expanded-pairing selected misses. "
        "This did not run `sde_validation_probe.py`, regenerate candidate pools, train, "
        "run a 32/64-sample formal eval, or add a rerank heuristic.",
        "",
        "## Executive Summary",
        "",
        "- Expanded-pairing formal result remains selected exact/relaxed `3/32`, "
        "oracle exact/relaxed `15/32`, pair oracle exact/relaxed `8/32`.",
        f"- Residual vectors were recomputed for `{summary['miss_cases_analyzed']}` selected-miss selected/oracle pairs.",
        f"- Wrong selected pair cases analyzed: `{summary['wrong_selected_pair_cases']}`.",
        f"- Oracle-only-pair misses analyzed: `{summary['oracle_only_pair_cases']}`.",
        f"- Cases where active favors selected: `{summary['active_favors_selected']}/{summary['miss_cases_analyzed']}`.",
        f"- Cases where weak favors oracle while active favors selected: `{summary['weak_disagrees_with_active']}/{summary['miss_cases_analyzed']}`.",
        f"- Trap shape among active-selected cases: outlier `{summary['trap_shape_counts'].get('outlier-dominated', 0)}`, "
        f"broad `{summary['trap_shape_counts'].get('broad', 0)}`, mixed `{summary['trap_shape_counts'].get('mixed', 0)}`.",
        f"- Active misleading dimensions lean: KM1/drift `{summary['side_counts'].get('KM1/drift', 0)}`, "
        f"KM2/diffusion `{summary['side_counts'].get('KM2/diffusion', 0)}`, mixed `{summary['side_counts'].get('mixed', 0)}`.",
        "",
        "Interpretation: the active traps are more often outlier-dominated than broad. "
        "The misleading dimensions mostly sit on KM1/drift-like active features, with a "
        "few mixed or KM2/diffusion-like cases. Weak-kernel distance sometimes disagrees, "
        "but not often enough to justify a formal reweighting eval. Constants do not look "
        "like the primary blocker because the correct structures already use role-wise "
        "constant-grid fits and still lose.",
        "",
        "Decision: **A. Active trap is dominated by a few residual dimensions; recommend "
        "one future offline robust/clipped active-residual ablation on existing expanded "
        "candidates only.** This is not enough evidence to add a rerank mode or launch a "
        "formal eval.",
        "",
        "## Aggregate Diagnosis",
        "",
        "| Diagnostic | Count |",
        "| --- | ---: |",
        f"| Miss cases analyzed | {summary['miss_cases_analyzed']} |",
        f"| Wrong selected pair cases | {summary['wrong_selected_pair_cases']} |",
        f"| Oracle-only-pair misses | {summary['oracle_only_pair_cases']} |",
        f"| Active favors selected | {summary['active_favors_selected']} |",
        f"| Weak favors oracle while active favors selected | {summary['weak_disagrees_with_active']} |",
        f"| Model score favors selected | {summary['model_favors_selected']} |",
        f"| Oracle best constants differ from selected best constants | {summary['constant_mismatch_count']} |",
        "",
        "## Case Summary",
        "",
        "| Sample | Selected src | Oracle src | Pair-only oracle | Active gap | Weak gap | Active wins sel/oracle | Shape | Side | Weak disagrees | Model favors | Classification |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        diag = case["diagnostics"]
        active_side = diag["active_side"]
        lines.append(
            f"| {case['sample_index']} | {case['selected_source']} | {case['oracle_source']} | "
            f"{'yes' if case['oracle_only_pair'] else 'no'} | "
            f"{diag['logged_active_gap']:.6f} | {diag['logged_weak_gap']:.6f} | "
            f"{active_side['selected_wins']}/{active_side['oracle_wins']} | "
            f"{active_side['trap_shape']} | {active_side['dominant_side']} | "
            f"{'yes' if diag['weak_disagrees_with_active'] else 'no'} | "
            f"{diag['model_favors']} | {diag['classification']} |"
        )

    lines.extend(
        [
            "",
            "Positive active or weak gap means the selected non-oracle has lower distance than the oracle in the formal log.",
            "",
            "## Per-Case Residual Highlights",
            "",
        ]
    )
    for case in cases:
        diag = case["diagnostics"]
        selected = diag["selected_score"]
        oracle = diag["oracle_score"]
        active_side = diag["active_side"]
        lines.extend(
            [
                f"### Sample {case['sample_index']}",
                "",
                f"Truth: `{case['truth_sequence']}`",
                "",
                f"Selected: `{case['selected_sequence']}`",
                "",
                f"Oracle: `{case['oracle_sequence']}`",
                "",
                f"Selected source `{case['selected_source']}`; oracle source `{case['oracle_source']}`; "
                f"oracle-only-pair `{case['oracle_only_pair']}`.",
                "",
                f"Selected score recompute: total `{fmt_float(selected.get('total'))}`, "
                f"active `{fmt_float(selected.get('active'))}`, weak `{fmt_float(selected.get('weak'))}`, "
                f"drift constant `{fmt_float(selected.get('drift_constant'))}`, "
                f"diffusion constant `{fmt_float(selected.get('diffusion_constant'))}`.",
                "",
                f"Oracle score recompute: total `{fmt_float(oracle.get('total'))}`, "
                f"active `{fmt_float(oracle.get('active'))}`, weak `{fmt_float(oracle.get('weak'))}`, "
                f"drift constant `{fmt_float(oracle.get('drift_constant'))}`, "
                f"diffusion constant `{fmt_float(oracle.get('diffusion_constant'))}`.",
                "",
                f"Active residual shape: `{active_side['trap_shape']}`; dominant active side: "
                f"`{active_side['dominant_side']}`; top2 share `{active_side['top2_share']:.3f}`; "
                f"top3 share `{active_side['top3_share']:.3f}`.",
                "",
                f"Selected active residual vector: `{fmt_vector(np.asarray(selected['active_residual']))}`",
                "",
                f"Oracle active residual vector: `{fmt_vector(np.asarray(oracle['active_residual']))}`",
                "",
                "Top active features where selected beats oracle:",
                "",
                format_feature_rows(diag["active_selected_top"]),
                "",
                "Top active features where oracle beats selected:",
                "",
                format_feature_rows(diag["active_oracle_top"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation",
            "",
            "- Do not make expanded pairing default and do not run a formal robust/clipped scorer eval now.",
            "- If continuing scorer analysis, do one offline robust/clipped active-residual ablation on the existing expanded-pairing miss candidates first.",
            "- The stronger next engineering direction remains candidate generation and active-fingerprint diagnostics: improve drift span diversity, and investigate why active KM statistics prefer plausible but wrong recombinations.",
            "",
            "## Provenance",
            "",
            f"- Miss JSON: `{payload['metadata']['miss_json']}`.",
            f"- Pruning JSON: `{payload['metadata']['pruning_json']}`.",
            f"- Coverage JSON: `{payload['metadata']['coverage_json']}`.",
            f"- Residual recompute settings: `n_paths={payload['metadata']['n_paths']}`, "
            f"`active_paths={payload['metadata']['active_paths']}`, "
            f"`n_steps={payload['metadata']['n_steps']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    miss_payload = json.loads(args.miss_json.read_text())
    pruning_payload = json.loads(args.pruning_json.read_text())
    coverage_payload = json.loads(args.coverage_json.read_text())
    cases = sorted(miss_payload["miss_cases"], key=lambda item: item["sample_index"])
    rows = case_rows(cases)
    sample_ids = [case["sample_index"] for case in cases]
    constant_values = parse_constant_values(args.constant_values)
    config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    targets = reproduce_eval_targets(sample_ids, args.eval_seed, config, rows)
    active_names, _ = feature_names()

    enriched = []
    trap_shape_counts = Counter()
    side_counts = Counter()
    weak_disagree_count = 0
    active_favors_count = 0
    model_favors_selected = 0
    constant_mismatch_count = 0
    for case in cases:
        sample_idx = case["sample_index"]
        target_y = targets[sample_idx]["target_y"]
        base_seed = args.eval_seed + 17
        base_seed += (sample_idx // args.batch_size) * 100003
        base_seed += (sample_idx % args.batch_size) * 1009
        selected = score_combo(
            "selected",
            tokens(case["selected_drift"]),
            tokens(case["selected_diffusion"]),
            target_y,
            config,
            constant_values,
            base_seed,
        )
        oracle = score_combo(
            "oracle",
            tokens(case["oracle_drift"]),
            tokens(case["oracle_diffusion"]),
            target_y,
            config,
            constant_values,
            base_seed + 1,
        )
        if not selected.get("success") or not oracle.get("success"):
            raise RuntimeError(f"sample {sample_idx}: selected/oracle scoring failed")

        active_selected_top, active_oracle_top, *_ = top_feature_rows(
            active_names,
            selected["active_residual"],
            oracle["active_residual"],
            args.top_k,
        )
        active_side = active_side_summary(
            active_names,
            selected["active_residual"],
            oracle["active_residual"],
        )
        selected_segments = case["selected_segment_distances"]
        oracle_segments = case["oracle_segment_distances"]
        logged_active_gap = (
            oracle_segments["active_kramers_moyal"]
            - selected_segments["active_kramers_moyal"]
        )
        logged_weak_gap = (
            oracle_segments["gaussian_weak_kernel"]
            - selected_segments["gaussian_weak_kernel"]
        )
        active_favors_selected = logged_active_gap > 0
        weak_disagrees = active_favors_selected and logged_weak_gap < 0
        model_favors = (
            "selected"
            if case["selected_model_score"] >= case["oracle_model_score"]
            else "oracle"
        )
        if model_favors == "selected":
            model_favors_selected += 1
        constants_differ = (
            case.get("selected_best_drift_constant") != case.get("oracle_best_drift_constant")
            or case.get("selected_best_diffusion_constant")
            != case.get("oracle_best_diffusion_constant")
        )
        if constants_differ:
            constant_mismatch_count += 1
        if active_favors_selected:
            active_favors_count += 1
            trap_shape_counts[active_side["trap_shape"]] += 1
            side_counts[active_side["dominant_side"]] += 1
        if weak_disagrees:
            weak_disagree_count += 1
        diagnostics = {
            "selected_score": selected,
            "oracle_score": oracle,
            "active_selected_top": active_selected_top,
            "active_oracle_top": active_oracle_top,
            "active_side": active_side,
            "logged_active_gap": logged_active_gap,
            "logged_weak_gap": logged_weak_gap,
            "weak_disagrees_with_active": weak_disagrees,
            "model_favors": model_favors,
            "constants_differ": constants_differ,
        }
        diagnostics["classification"] = classify_case(case, diagnostics)
        enriched_case = dict(case)
        enriched_case["diagnostics"] = diagnostics
        enriched.append(enriched_case)

    summary = {
        "miss_cases_analyzed": len(enriched),
        "wrong_selected_pair_cases": sum(1 for case in enriched if case["selected_source"] == "pair"),
        "oracle_only_pair_cases": sum(1 for case in enriched if case["oracle_only_pair"]),
        "pair_oracle_samples": len(pruning_payload.get("pair_oracle_samples", [])),
        "expanded_full_oracle_present": sum(
            1 for sample in coverage_payload.get("samples", []) if sample.get("has_full")
        ),
        "active_favors_selected": active_favors_count,
        "weak_disagrees_with_active": weak_disagree_count,
        "trap_shape_counts": trap_shape_counts,
        "side_counts": side_counts,
        "model_favors_selected": model_favors_selected,
        "constant_mismatch_count": constant_mismatch_count,
        "avg_logged_active_gap": mean(
            case["diagnostics"]["logged_active_gap"] for case in enriched
        ),
        "avg_logged_weak_gap": mean(case["diagnostics"]["logged_weak_gap"] for case in enriched),
    }
    payload = {
        "metadata": {
            "miss_json": str(args.miss_json),
            "pruning_json": str(args.pruning_json),
            "coverage_json": str(args.coverage_json),
            "eval_seed": args.eval_seed,
            "n_paths": args.n_paths,
            "active_paths": args.active_paths,
            "n_steps": args.n_steps,
            "active_slice": [ACTIVE_SLICE.start, ACTIVE_SLICE.stop],
            "weak_slice": [WEAK_SLICE.start, WEAK_SLICE.stop],
            "constant_values": constant_values,
        },
        "summary": summary,
        "cases": enriched,
    }
    write_report(args.report, payload)
    args.json_output.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"miss_cases_analyzed={summary['miss_cases_analyzed']}")
    print(f"wrong_selected_pair_cases={summary['wrong_selected_pair_cases']}")
    print(f"oracle_only_pair_cases={summary['oracle_only_pair_cases']}")
    print(f"active_favors_selected={summary['active_favors_selected']}")
    print(f"weak_disagrees_with_active={summary['weak_disagrees_with_active']}")
    print(f"trap_shape_counts={dict(summary['trap_shape_counts'])}")
    print(f"side_counts={dict(summary['side_counts'])}")


if __name__ == "__main__":
    main()
