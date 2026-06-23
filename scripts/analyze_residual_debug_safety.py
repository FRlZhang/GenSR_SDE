#!/usr/bin/env python3
"""Offline safety checks for rerank residual debug JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/private/tmp/gensr_sde_residual_debug_smoke8.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("residual_debug_safety_smoke8.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("residual_debug_safety_smoke8.json"),
    )
    return parser.parse_args()


def candidate_is_oracle(candidate: dict) -> bool:
    return bool(candidate.get("is_oracle_exact") or candidate.get("is_oracle_relaxed"))


def candidate_has_residuals(candidate: dict) -> bool:
    return (
        candidate.get("active_residual_vector") is not None
        and candidate.get("weak_residual_vector") is not None
    )


def active_scales(samples: list[dict]) -> np.ndarray | None:
    rows = []
    for sample in samples:
        for candidate in sample.get("candidates", []):
            if candidate_has_residuals(candidate):
                rows.append(np.asarray(candidate["active_residual_vector"], dtype=np.float64))
    if not rows:
        return None
    matrix = np.vstack(rows)
    scales = np.std(matrix, axis=0)
    fallback = float(np.median(scales[scales > 0])) if np.any(scales > 0) else 1.0
    return np.where(scales > 1e-8, scales, fallback)


def offline_score(candidate: dict, scales: np.ndarray) -> float:
    if not candidate_has_residuals(candidate):
        return float("inf")
    active = np.asarray(candidate["active_residual_vector"], dtype=np.float64)
    weak = np.asarray(candidate["weak_residual_vector"], dtype=np.float64)
    return float(np.linalg.norm(active / scales) + np.linalg.norm(weak))


def selected_candidate(sample: dict) -> dict | None:
    for candidate in sample.get("candidates", []):
        if candidate.get("is_selected"):
            return candidate
    return None


def analyze(payload: dict) -> dict:
    samples = payload.get("samples", [])
    scales = active_scales(samples)
    sample_rows = []
    if scales is None:
        return {
            "summary": {
                "samples": len(samples),
                "candidates": sum(len(sample.get("candidates", [])) for sample in samples),
                "residual_vectors_present": False,
                "blocked": True,
                "blocker": "no candidate residual vectors found",
            },
            "sample_rows": [],
        }

    for sample in samples:
        candidates = sample.get("candidates", [])
        for candidate in candidates:
            candidate["offline_per_active_dim_norm_plus_weak"] = offline_score(candidate, scales)
        ranked = sorted(
            candidates,
            key=lambda item: (
                not math.isfinite(item["offline_per_active_dim_norm_plus_weak"]),
                item["offline_per_active_dim_norm_plus_weak"],
                -(item.get("normalized_model_score") or 0.0),
            ),
        )
        original = selected_candidate(sample)
        offline = ranked[0] if ranked else None
        oracle_candidates = [candidate for candidate in candidates if candidate_is_oracle(candidate)]
        pair_oracles = [
            candidate
            for candidate in oracle_candidates
            if candidate.get("source") == "pair" or candidate.get("is_pair_oracle")
        ]
        original_hit = bool(original is not None and candidate_is_oracle(original))
        offline_hit = bool(offline is not None and candidate_is_oracle(offline))
        row = {
            "sample_index": sample.get("sample_index"),
            "candidate_count": len(candidates),
            "has_oracle": bool(oracle_candidates),
            "has_pair_oracle": bool(pair_oracles),
            "original_selected_rank": original.get("candidate_rank") if original else None,
            "original_selected_source": original.get("source") if original else None,
            "original_selected_is_oracle": original_hit,
            "offline_selected_rank": offline.get("candidate_rank") if offline else None,
            "offline_selected_source": offline.get("source") if offline else None,
            "offline_selected_is_oracle": offline_hit,
            "offline_selected_is_pair": bool(offline and offline.get("source") == "pair"),
            "rescued_miss": bool((not original_hit) and offline_hit),
            "harmed_hit": bool(original_hit and not offline_hit),
            "pair_oracle_selected_offline": bool(
                offline is not None
                and candidate_is_oracle(offline)
                and (offline.get("source") == "pair" or offline.get("is_pair_oracle"))
            ),
            "oracle_rank_offline": None,
            "pair_oracle_rank_offline": None,
        }
        for rank, candidate in enumerate(ranked, start=1):
            if row["oracle_rank_offline"] is None and candidate_is_oracle(candidate):
                row["oracle_rank_offline"] = rank
            if (
                row["pair_oracle_rank_offline"] is None
                and candidate_is_oracle(candidate)
                and (candidate.get("source") == "pair" or candidate.get("is_pair_oracle"))
            ):
                row["pair_oracle_rank_offline"] = rank
        sample_rows.append(row)

    summary = {
        "samples": len(samples),
        "candidates": sum(len(sample.get("candidates", [])) for sample in samples),
        "residual_vectors_present": True,
        "blocked": False,
        "samples_with_oracle": sum(row["has_oracle"] for row in sample_rows),
        "samples_with_pair_oracle": sum(row["has_pair_oracle"] for row in sample_rows),
        "original_selected_hits": sum(row["original_selected_is_oracle"] for row in sample_rows),
        "offline_selected_hits": sum(row["offline_selected_is_oracle"] for row in sample_rows),
        "selected_hits_preserved": sum(
            row["original_selected_is_oracle"] and row["offline_selected_is_oracle"]
            for row in sample_rows
        ),
        "selected_hits_harmed": sum(row["harmed_hit"] for row in sample_rows),
        "selected_misses_rescued": sum(row["rescued_miss"] for row in sample_rows),
        "offline_pair_oracle_selected": sum(row["pair_oracle_selected_offline"] for row in sample_rows),
        "offline_selected_pair_candidates": sum(row["offline_selected_is_pair"] for row in sample_rows),
    }
    return {"summary": summary, "sample_rows": sample_rows}


def write_report(path: Path, input_path: Path, analysis: dict) -> None:
    summary = analysis["summary"]
    rows = analysis["sample_rows"]
    sample_count = summary.get("samples", 0)
    lines = [
        f"# Residual Debug Safety Smoke {sample_count}",
        "",
        "Scope: offline parse of residual debug JSON only. No model decoding, candidate generation, simulation, formal eval, retraining, or scorer change was run.",
        "",
        "## Summary",
        "",
    ]
    if summary.get("blocked"):
        lines.extend(
            [
                f"- Blocked: `{summary['blocker']}`.",
                f"- Input JSON: `{input_path}`.",
                "",
            ]
        )
        path.write_text("\n".join(lines))
        return

    lines.extend(
        [
            f"- Input JSON: `{input_path}`.",
            f"- Samples: `{summary['samples']}`.",
            f"- Candidates: `{summary['candidates']}`.",
            f"- Samples with oracle: `{summary['samples_with_oracle']}`.",
            f"- Samples with pair oracle: `{summary['samples_with_pair_oracle']}`.",
            f"- Original selected hits: `{summary['original_selected_hits']}`.",
            f"- Offline selected hits: `{summary['offline_selected_hits']}`.",
            f"- Selected hits preserved: `{summary['selected_hits_preserved']}`.",
            f"- Selected hits harmed: `{summary['selected_hits_harmed']}`.",
            f"- Selected misses rescued: `{summary['selected_misses_rescued']}`.",
            f"- Offline selected pair candidates: `{summary['offline_selected_pair_candidates']}`.",
            "",
            f"The smoke is only {sample_count} samples, so absence of harm is not enough for a formal eval. It is useful for checking whether the residual logging is complete and whether a future targeted smoke is worth running.",
            "",
            "## Per Sample",
            "",
            "| Sample | Candidates | Oracle | Pair oracle | Original hit | Offline hit | Rescued | Harmed | Offline source | Offline oracle rank | Pair oracle rank |",
            "| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['sample_index']} | {row['candidate_count']} | "
            f"{'yes' if row['has_oracle'] else 'no'} | "
            f"{'yes' if row['has_pair_oracle'] else 'no'} | "
            f"{'yes' if row['original_selected_is_oracle'] else 'no'} | "
            f"{'yes' if row['offline_selected_is_oracle'] else 'no'} | "
            f"{'yes' if row['rescued_miss'] else 'no'} | "
            f"{'yes' if row['harmed_hit'] else 'no'} | "
            f"{row['offline_selected_source']} | "
            f"{row['oracle_rank_offline'] or 'n/a'} | "
            f"{row['pair_oracle_rank_offline'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use this logging path for a future targeted smoke before considering any scorer implementation. Do not run a formal 32-sample eval from this safety check alone.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text())
    analysis = analyze(payload)
    write_report(args.report, args.input_json, analysis)
    args.json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    summary = analysis["summary"]
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    for key in (
        "samples",
        "candidates",
        "samples_with_oracle",
        "original_selected_hits",
        "offline_selected_hits",
        "selected_hits_harmed",
        "selected_misses_rescued",
    ):
        if key in summary:
            print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
