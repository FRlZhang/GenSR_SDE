#!/usr/bin/env python3
"""Offline score ablations for targeted role-wise residual vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-log", type=Path, default=None)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("rolewise_residual_vector_diagnostics.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("rolewise_residual_score_ablation.md"))
    return parser.parse_args()


def l2(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def l1(values: np.ndarray) -> float:
    return float(np.sum(np.abs(values)))


def clipped_l2(values: np.ndarray, cap: float) -> float:
    clipped = np.minimum(values * values, cap * cap)
    return float(np.sqrt(np.sum(clipped)))


def huber(values: np.ndarray, delta: float) -> float:
    abs_values = np.abs(values)
    loss = np.where(
        abs_values <= delta,
        0.5 * values * values,
        delta * (abs_values - 0.5 * delta),
    )
    return float(np.sum(loss))


def topk_removed_l2(values: np.ndarray, k: int) -> float:
    squared = values * values
    if k <= 0:
        return float(np.sqrt(np.sum(squared)))
    if k >= squared.size:
        return 0.0
    trimmed = np.sort(squared)[: squared.size - k]
    return float(np.sqrt(np.sum(trimmed)))


def combined(combo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(combo["active_residual"], dtype=np.float64)
    weak = np.asarray(combo["weak_residual"], dtype=np.float64)
    return active, weak, np.concatenate([active, weak])


def score_variants(selected: dict, oracle: dict) -> dict[str, tuple[float, float, str]]:
    s_active, s_weak, s_all = combined(selected)
    o_active, o_weak, o_all = combined(oracle)
    both_abs = np.abs(np.concatenate([s_all, o_all]))
    p90 = float(np.percentile(both_abs, 90))
    p95 = float(np.percentile(both_abs, 95))
    active_n = max(1, s_active.size)
    weak_n = max(1, s_weak.size)

    variants: dict[str, tuple[float, float, str]] = {
        "current_l2_sum": (
            selected["total"],
            oracle["total"],
            "current active L2 + weak L2",
        ),
        "active_only_l2": (l2(s_active), l2(o_active), "active segment only"),
        "weak_only_l2": (l2(s_weak), l2(o_weak), "weak segment only"),
        "l1_sum": (l1(s_all), l1(o_all), "sum absolute residuals"),
        "l2_sum": (l2(s_active) + l2(s_weak), l2(o_active) + l2(o_weak), "same geometry as current"),
        "clipped_l2_p95": (
            clipped_l2(s_active, p95) + clipped_l2(s_weak, p95),
            clipped_l2(o_active, p95) + clipped_l2(o_weak, p95),
            f"per-feature squared contribution capped at p95={p95:.4g}",
        ),
        "clipped_l2_p90": (
            clipped_l2(s_active, p90) + clipped_l2(s_weak, p90),
            clipped_l2(o_active, p90) + clipped_l2(o_weak, p90),
            f"per-feature squared contribution capped at p90={p90:.4g}",
        ),
        "huber_p90": (
            huber(s_active, p90) + huber(s_weak, p90),
            huber(o_active, p90) + huber(o_weak, p90),
            f"Huber loss with delta=p90={p90:.4g}",
        ),
        "per_segment_normalized_l2": (
            l2(s_active) / math.sqrt(active_n) + l2(s_weak) / math.sqrt(weak_n),
            l2(o_active) / math.sqrt(active_n) + l2(o_weak) / math.sqrt(weak_n),
            "active and weak L2 normalized by feature count",
        ),
        "top1_removed_l2": (
            topk_removed_l2(s_active, 1) + topk_removed_l2(s_weak, 1),
            topk_removed_l2(o_active, 1) + topk_removed_l2(o_weak, 1),
            "remove largest contribution in each segment",
        ),
        "top3_removed_l2": (
            topk_removed_l2(s_active, 3) + topk_removed_l2(s_weak, 3),
            topk_removed_l2(o_active, 3) + topk_removed_l2(o_weak, 3),
            "remove largest 3 contributions in each segment",
        ),
        "top5_removed_l2": (
            topk_removed_l2(s_active, 5) + topk_removed_l2(s_weak, 5),
            topk_removed_l2(o_active, 5) + topk_removed_l2(o_weak, 5),
            "remove largest 5 contributions in each segment",
        ),
    }
    return variants


def feature_advantage_summary(diag: dict) -> str:
    active = diag.get("active_selected_top", [])[:3]
    weak = diag.get("weak_selected_top", [])[:3]

    def fmt(rows: list) -> str:
        if not rows:
            return "none"
        parts = []
        for row in rows:
            idx, name, advantage, *_ = row
            parts.append(f"{idx}:{name} ({advantage:.4f})")
        return "; ".join(parts)

    return f"active: {fmt(active)}; weak: {fmt(weak)}"


def fmt(value: float) -> str:
    return f"{value:.6f}"


def analyze(payload: dict) -> dict[int, dict]:
    results = {}
    diagnostics = payload["diagnostics"]
    for raw_idx, diag in diagnostics.items():
        sample_idx = int(raw_idx)
        selected = diag["combos"]["selected+selected"]
        oracle = diag["combos"]["oracle+oracle"]
        variants = score_variants(selected, oracle)
        variant_rows = []
        for name, (selected_score, oracle_score, note) in variants.items():
            oracle_wins = oracle_score < selected_score
            variant_rows.append(
                {
                    "name": name,
                    "selected": selected_score,
                    "oracle": oracle_score,
                    "oracle_wins": oracle_wins,
                    "note": note,
                }
            )
        mild = {
            "l1_sum",
            "clipped_l2_p95",
            "huber_p90",
            "per_segment_normalized_l2",
        }
        extreme = {"clipped_l2_p90", "top1_removed_l2", "top3_removed_l2", "top5_removed_l2"}
        results[sample_idx] = {
            "classification": diag["classification"],
            "dominant_features": feature_advantage_summary(diag),
            "variants": variant_rows,
            "rescued_by_mild": any(row["oracle_wins"] and row["name"] in mild for row in variant_rows),
            "rescued_by_extreme": any(row["oracle_wins"] and row["name"] in extreme for row in variant_rows),
            "rescued_by_any": any(row["oracle_wins"] for row in variant_rows),
        }
    return results


def write_report(path: Path, payload: dict, results: dict[int, dict], input_log: Path | None) -> None:
    mild_count = sum(item["rescued_by_mild"] for item in results.values())
    extreme_count = sum(item["rescued_by_extreme"] and not item["rescued_by_mild"] for item in results.values())
    any_count = sum(item["rescued_by_any"] for item in results.values())
    sample24 = results.get(24, {})
    sample26 = results.get(26, {})

    lines = [
        "# Role-wise Residual Score Ablation",
        "",
        "Date: 2026-06-18",
        "",
        "Scope: offline score ablation over residual vectors for samples `13`, `19`, `24`, and `26`. "
        "This is not a formal eval and does not add a rerank mode.",
        "",
        "## Executive Summary",
        "",
        f"- Offline variants that rank oracle above selected in at least one case: `{any_count}/4`.",
        f"- Mild robust/normalization variants rescue: `{mild_count}/4`.",
        f"- Extreme clipping-only rescues: `{extreme_count}/4`.",
    ]
    if any_count == 0:
        lines.append(
            "- No tested offline score variant ranks any remaining oracle above the selected non-oracle; this points toward fingerprint ambiguity rather than an easy robust-scoring fix."
        )
    else:
        lines.append(
            "- At least one offline score variant flips a miss; inspect whether the flip requires aggressive clipping before considering a formal eval."
        )
    lines.extend(
        [
            "",
            "## Per-sample Results",
            "",
        ]
    )

    for sample_idx in sorted(results):
        item = results[sample_idx]
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
                f"Classification: `{item['classification']}`",
                "",
                f"Dominant selected-advantage features: {item['dominant_features']}",
                "",
                "| Variant | Selected | Oracle | Oracle wins | Note |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in item["variants"]:
            lines.append(
                f"| `{row['name']}` | {fmt(row['selected'])} | {fmt(row['oracle'])} | "
                f"{'yes' if row['oracle_wins'] else 'no'} | {row['note']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Aggregate Conclusion",
            "",
            f"- Mild robust scoring rescues `{mild_count}/4` remaining misses.",
            f"- Extreme clipping-only rescues `{extreme_count}/4` remaining misses.",
            f"- Any offline score rescue count is `{any_count}/4`.",
            f"- Sample 24 drift-side ambiguity rescued by any variant: `{sample24.get('rescued_by_any', False)}`.",
            f"- Sample 26 diffusion-side ambiguity rescued by any variant: `{sample26.get('rescued_by_any', False)}`.",
        ]
    )
    if mild_count == 0 and any_count == 0:
        lines.append(
            "- A future formal 32-sample eval for robust scoring is not justified yet; the offline residuals do not show a promising scorer-only flip."
        )
    elif mild_count > 0:
        lines.append(
            "- A future formal 32-sample eval may be justified only after turning the mild offline variant into a clearly specified candidate score and first smoke-testing it."
        )
    else:
        lines.append(
            "- Formal eval is not justified from extreme clipping alone; it looks too brittle."
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- JSON sidecar: `{payload.get('metadata', {}).get('json_source', 'rolewise_residual_vector_diagnostics.json')}`",
            f"- Residual helper log: `{input_log}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not args.input_json.is_file():
        raise FileNotFoundError(
            f"{args.input_json} not found; rerun scripts/analyze_rolewise_residuals.py "
            "to emit the JSON sidecar"
        )
    payload = json.loads(args.input_json.read_text())
    payload.setdefault("metadata", {})["json_source"] = str(args.input_json)
    results = analyze(payload)
    write_report(args.report, payload, results, args.input_log)
    print(f"wrote_report={args.report}")
    for sample_idx in sorted(results):
        item = results[sample_idx]
        winners = [row["name"] for row in item["variants"] if row["oracle_wins"]]
        print(
            f"sample={sample_idx} oracle_winning_variants="
            f"{','.join(winners) if winners else 'none'}"
        )


if __name__ == "__main__":
    main()
