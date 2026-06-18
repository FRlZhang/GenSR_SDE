#!/usr/bin/env python3
"""Targeted residual-vector diagnostics for role-wise rerank misses."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parsers import get_parser
from sde_dataset_generator import (
    clean_tokens,
    constants_to_token,
    normalize_expression_for_gensr,
)
from sde_fingerprint import FingerprintConfig
from simulator_sde import sample_sde_system, solve_fingerprint
from symbolicregression.envs.generators import string_to_node

from sde_validation_probe import (
    candidate_tokens_to_system,
    fingerprint_distance_details,
    split_sde_tokens,
)


ACTIVE_SLICE = slice(72, 90)
WEAK_SLICE = slice(90, 186)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--samples", type=str, default="13,19,24,26")
    parser.add_argument("--report", type=Path, default=Path("rolewise_residual_vector_diagnostics.md"))
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=800)
    parser.add_argument("--active-paths", type=int, default=800)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--constant-values", type=str, default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def parse_sample_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_constant_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("constant grid cannot be empty")
    return values


def parse_miss_log(path: Path, sample_ids: set[int]) -> dict[int, dict]:
    text = path.read_text()
    if "oracle_miss_cases:" not in text:
        raise ValueError(f"{path} does not contain oracle_miss_cases")
    section = text.split("oracle_miss_cases:", 1)[1].split("oracle_miss_rescue_cases:", 1)[0]
    starts = list(re.finditer(r"^sample_index=(\d+)$", section, flags=re.MULTILINE))
    rows: dict[int, dict] = {}
    for pos, match in enumerate(starts):
        sample_idx = int(match.group(1))
        if sample_idx not in sample_ids:
            continue
        end = starts[pos + 1].start() if pos + 1 < len(starts) else len(section)
        block = section[match.start() : end]
        rows[sample_idx] = parse_miss_block(sample_idx, block)
    missing = sorted(sample_ids - rows.keys())
    if missing:
        raise ValueError(f"missing sample blocks in log: {missing}")
    return rows


def parse_miss_block(sample_idx: int, block: str) -> dict:
    def line_after(prefix: str) -> str | None:
        found = re.search(rf"^{re.escape(prefix)}(.*)$", block, flags=re.MULTILINE)
        return found.group(1).strip() if found else None

    def require(prefix: str) -> str:
        value = line_after(prefix)
        if value is None:
            raise ValueError(f"sample {sample_idx}: missing {prefix}")
        return value

    sources = require("sources ")
    source_match = re.search(r"selected=(\S+) oracle=(\S+)", sources)
    score_match = re.search(
        r"selected=([0-9.eE+-]+) oracle=([0-9.eE+-]+) gap=([0-9.eE+-]+)",
        require("scores "),
    )
    model_match = re.search(
        r"selected=([0-9.eE+-]+) oracle=([0-9.eE+-]+) oracle_lower=(\d+)",
        require("model_scores "),
    )
    if source_match is None or score_match is None or model_match is None:
        raise ValueError(f"sample {sample_idx}: malformed source/score lines")

    return {
        "sample_index": sample_idx,
        "truth": require("truth: "),
        "selected_sequence": require("selected_sequence: "),
        "oracle_sequence": require("oracle_sequence: "),
        "selected_source": source_match.group(1),
        "oracle_source": source_match.group(2),
        "logged_selected_score": float(score_match.group(1)),
        "logged_oracle_score": float(score_match.group(2)),
        "logged_score_gap": float(score_match.group(3)),
        "selected_model_score": float(model_match.group(1)),
        "oracle_model_score": float(model_match.group(2)),
        "oracle_lower_model_score": bool(int(model_match.group(3))),
        "selected_drift": require("selected_drift: ").split(),
        "oracle_drift": require("oracle_drift: ").split(),
        "selected_diffusion": require("selected_diffusion: ").split(),
        "oracle_diffusion": require("oracle_diffusion: ").split(),
    }


def expression_tokens(expr: str, params) -> list[str]:
    normalized = normalize_expression_for_gensr(expr)
    return constants_to_token(clean_tokens(string_to_node(normalized, params).prefix().split(",")))


def system_tokens(system, params) -> list[str]:
    return (
        ["<DRIFT>"]
        + expression_tokens(system.drift_str, params)
        + ["<DIFFUSION>"]
        + expression_tokens(system.diffusion_str, params)
    )


def reproduce_eval_targets(
    sample_ids: list[int],
    eval_seed: int,
    config: FingerprintConfig,
    expected_rows: dict[int, dict],
) -> dict[int, dict]:
    params = get_parser().parse_args([])
    py_rng = random.Random(eval_seed)
    np_rng = np.random.default_rng(eval_seed)
    targets: dict[int, dict] = {}
    accepted_idx = 0
    attempts = 0
    max_needed = max(sample_ids)
    while accepted_idx <= max_needed and attempts < (max_needed + 1) * 100:
        attempts += 1
        system = sample_sde_system(py_rng)
        seed = int(np_rng.integers(0, 2**31 - 1))
        _, y_data, meta = solve_fingerprint(system, config, seed)
        if y_data is None or not np.all(np.isfinite(y_data)):
            continue
        if accepted_idx in sample_ids:
            tokens = system_tokens(system, params)
            expected = expected_rows[accepted_idx]["truth"].split()
            if tokens != expected:
                raise RuntimeError(
                    "reproduced sample does not match log truth for "
                    f"sample {accepted_idx}:\nexpected={expected}\nactual={tokens}"
                )
            targets[accepted_idx] = {
                "system": system,
                "seed": seed,
                "target_y": y_data,
                "meta": meta,
            }
        accepted_idx += 1
    missing = sorted(set(sample_ids) - targets.keys())
    if missing:
        raise RuntimeError(f"failed to reproduce targets for samples {missing}")
    return targets


def vector_distance(target: np.ndarray, candidate: np.ndarray, segment: slice) -> tuple[float, np.ndarray]:
    target_vec = np.asarray(target, dtype=np.float64).reshape(-1)[segment]
    candidate_vec = np.asarray(candidate, dtype=np.float64).reshape(-1)[segment]
    residual = target_vec - candidate_vec
    denom = np.linalg.norm(target_vec) + 1e-8
    return float(np.linalg.norm(residual) / denom), residual / denom


def feature_names() -> tuple[list[str], list[str]]:
    active_names = []
    for t in (0.0, 0.5, 1.0):
        for x0 in (0.5, 1.0, 1.5):
            active_names.append(f"t={t:g},x0={x0:g},drift")
            active_names.append(f"t={t:g},x0={x0:g},diffusion")
    centers = np.linspace(-2.0, 2.0, 24)
    weak_names = []
    for group in ("occupancy", "local_drift", "local_diffusion", "generator_action"):
        for center in centers:
            weak_names.append(f"{group}@{center:.3g}")
    return active_names, weak_names


def score_combo(
    label: str,
    drift_tokens: list[str],
    diffusion_tokens: list[str],
    target_y: np.ndarray,
    config: FingerprintConfig,
    constant_values: list[float],
    seed: int,
) -> dict:
    words = ["<DRIFT>"] + drift_tokens + ["<DIFFUSION>"] + diffusion_tokens
    drift_constants = constant_values if "CONSTANT" in drift_tokens else [1.0]
    diffusion_constants = constant_values if "CONSTANT" in diffusion_tokens else [1.0]
    best = None
    failures: list[str] = []
    for drift_constant in drift_constants:
        for diffusion_constant in diffusion_constants:
            system = candidate_tokens_to_system(
                words,
                drift_constant_value=drift_constant,
                diffusion_constant_value=diffusion_constant,
            )
            if system is None:
                return {
                    "label": label,
                    "words": words,
                    "success": False,
                    "failure": "parse_failed",
                }
            _, candidate_y, meta = solve_fingerprint(system, config, seed)
            if candidate_y is None or not np.all(np.isfinite(candidate_y)):
                failures.append(meta.get("error", "fingerprint_failed"))
                continue
            details = fingerprint_distance_details(
                target_y,
                candidate_y,
                "rolewise_no_multi_u0",
                (1.0, 2.0, 1.0),
            )
            active_distance, active_residual = vector_distance(target_y, candidate_y, ACTIVE_SLICE)
            weak_distance, weak_residual = vector_distance(target_y, candidate_y, WEAK_SLICE)
            record = {
                "label": label,
                "words": words,
                "success": True,
                "failure": None,
                "total": details["distance"],
                "active": active_distance,
                "weak": weak_distance,
                "drift_constant": drift_constant,
                "diffusion_constant": diffusion_constant,
                "active_residual": active_residual,
                "weak_residual": weak_residual,
            }
            if best is None or record["total"] < best["total"]:
                best = record
    if best is not None:
        return best
    return {
        "label": label,
        "words": words,
        "success": False,
        "failure": failures[0] if failures else "fingerprint_failed",
    }


def top_feature_rows(
    names: list[str],
    selected_residual: np.ndarray,
    oracle_residual: np.ndarray,
    top_k: int,
) -> tuple[list[tuple], list[tuple], int, int, str]:
    selected_abs = np.abs(selected_residual)
    oracle_abs = np.abs(oracle_residual)
    selected_advantage = oracle_abs - selected_abs
    oracle_advantage = selected_abs - oracle_abs
    selected_wins = int(np.sum(selected_advantage > 0))
    oracle_wins = int(np.sum(oracle_advantage > 0))

    def collect(values: np.ndarray) -> list[tuple]:
        order = np.argsort(-values)
        rows = []
        for idx in order:
            if values[idx] <= 0 or len(rows) >= top_k:
                break
            rows.append(
                (
                    int(idx),
                    names[int(idx)],
                    float(values[idx]),
                    float(selected_abs[idx]),
                    float(oracle_abs[idx]),
                )
            )
        return rows

    delta_sq = np.maximum(selected_advantage, 0.0) ** 2
    if float(delta_sq.sum()) <= 0:
        dominance = "none"
    else:
        top3_share = float(np.sort(delta_sq)[-3:].sum() / delta_sq.sum())
        dominance = "few_dominant_features" if top3_share >= 0.5 else "broad"
    return (
        collect(selected_advantage),
        collect(oracle_advantage),
        selected_wins,
        oracle_wins,
        dominance,
    )


def fmt_float(value) -> str:
    if value is None:
        return "nan"
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def fmt_vector(values: np.ndarray, max_items: int = 24) -> str:
    rounded = [f"{float(value):.6g}" for value in values[:max_items]]
    suffix = ", ..." if values.size > max_items else ""
    return "[" + ", ".join(rounded) + suffix + "]"


def format_feature_table(rows: list[tuple]) -> str:
    if not rows:
        return "No positive feature wins.\n"
    lines = ["| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |",
             "| ---: | --- | ---: | ---: | ---: |"]
    for idx, name, advantage, selected_abs, oracle_abs in rows:
        lines.append(
            f"| {idx} | `{name}` | {advantage:.6f} | {selected_abs:.6f} | {oracle_abs:.6f} |"
        )
    return "\n".join(lines) + "\n"


def classify(row: dict, selected: dict, oracle: dict, active_dominance: str, weak_dominance: str) -> str:
    shares_drift = row["selected_drift"] == row["oracle_drift"]
    shares_diffusion = row["selected_diffusion"] == row["oracle_diffusion"]
    labels = []
    if shares_diffusion and not shares_drift:
        labels.append("drift-side ambiguity")
    elif shares_drift and not shares_diffusion:
        labels.append("diffusion-side ambiguity")
    else:
        labels.append("both-side ambiguity")
    if active_dominance == "few_dominant_features" or weak_dominance == "few_dominant_features":
        labels.append("feature-scaling artifact")
    else:
        labels.append("fingerprint ambiguity")
    if row["oracle_lower_model_score"]:
        labels.append("model-score conflict")
    if selected["success"] and oracle["success"] and selected["active"] < oracle["active"] and selected["weak"] < oracle["weak"]:
        labels.append("selected wins both aggregate segments")
    return "; ".join(labels)


def write_report(
    path: Path,
    rows: dict[int, dict],
    diagnostics: dict[int, dict],
    command_context: argparse.Namespace,
) -> None:
    lines = [
        "# Role-wise Residual Vector Diagnostics",
        "",
        "Date: 2026-06-18",
        "",
        "Scope: targeted diagnostic for samples `13`, `19`, `24`, and `26`. "
        "This is not a formal 32-sample eval, not a 64-sample eval, and not a "
        "new rerank heuristic.",
        "",
        "## Executive Summary",
        "",
        "- Current role-wise baseline remains selected exact/relaxed `5/32` with oracle ceiling `9/32`.",
        "- The diagnostic recomputed target fingerprints for the four known miss samples from the eval seed and scored only selected/oracle role swaps.",
        "- Distances are diagnostic recomputations with deterministic per-combo seeds, so they should be interpreted as residual structure rather than a replacement for the existing formal eval metrics.",
        "",
        "## Aggregate Findings",
        "",
    ]
    classifications = [diagnostics[idx]["classification"] for idx in sorted(diagnostics)]
    lines.extend(
        [
            f"- Samples analyzed: `{', '.join(str(idx) for idx in sorted(diagnostics))}`.",
            f"- Drift-side ambiguity cases: `{sum('drift-side ambiguity' in c for c in classifications)}`.",
            f"- Diffusion-side ambiguity cases: `{sum('diffusion-side ambiguity' in c for c in classifications)}`.",
            f"- Both-side ambiguity cases: `{sum('both-side ambiguity' in c for c in classifications)}`.",
            f"- Feature-scaling artifact flags: `{sum('feature-scaling artifact' in c for c in classifications)}`.",
            f"- Model-score conflicts: `{sum('model-score conflict' in c for c in classifications)}`.",
            "",
        ]
    )

    for sample_idx in sorted(diagnostics):
        row = rows[sample_idx]
        diag = diagnostics[sample_idx]
        selected = diag["combos"]["selected+selected"]
        oracle = diag["combos"]["oracle+oracle"]
        lines.extend(
            [
                f"## Sample {sample_idx}",
                "",
                f"Truth: `{row['truth']}`",
                "",
                f"Selected: `{row['selected_sequence']}`",
                "",
                f"Oracle: `{row['oracle_sequence']}`",
                "",
                f"Selected source: `{row['selected_source']}`; oracle source: `{row['oracle_source']}`.",
                "",
                f"Selected drift: `{' '.join(row['selected_drift'])}`",
                "",
                f"Selected diffusion: `{' '.join(row['selected_diffusion'])}`",
                "",
                f"Oracle drift: `{' '.join(row['oracle_drift'])}`",
                "",
                f"Oracle diffusion: `{' '.join(row['oracle_diffusion'])}`",
                "",
                "### Role-swap Scores",
                "",
                "| Combination | Success | Total | Active | Weak | Drift constant | Diffusion constant | Failure |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for label in (
            "selected+selected",
            "oracle+oracle",
            "selected+oracle",
            "oracle+selected",
        ):
            combo = diag["combos"][label]
            lines.append(
                f"| `{label}` | {combo.get('success', False)} | "
                f"{fmt_float(combo.get('total'))} | {fmt_float(combo.get('active'))} | "
                f"{fmt_float(combo.get('weak'))} | {fmt_float(combo.get('drift_constant'))} | "
                f"{fmt_float(combo.get('diffusion_constant'))} | `{combo.get('failure') or 'none'}` |"
            )
        lines.extend(
            [
                "",
                "### Residual Vectors",
                "",
                f"Selected active residual vector: `{fmt_vector(selected['active_residual'])}`",
                "",
                f"Oracle active residual vector: `{fmt_vector(oracle['active_residual'])}`",
                "",
                f"Selected weak residual vector: `{fmt_vector(selected['weak_residual'])}`",
                "",
                f"Oracle weak residual vector: `{fmt_vector(oracle['weak_residual'])}`",
                "",
                "### Active Feature Comparison",
                "",
                f"Selected wins: `{diag['active_selected_wins']}`; oracle wins: `{diag['active_oracle_wins']}`; dominance: `{diag['active_dominance']}`.",
                "",
                "Top active features where selected beats oracle:",
                "",
                format_feature_table(diag["active_selected_top"]),
                "Top active features where oracle beats selected:",
                "",
                format_feature_table(diag["active_oracle_top"]),
                "### Weak Feature Comparison",
                "",
                f"Selected wins: `{diag['weak_selected_wins']}`; oracle wins: `{diag['weak_oracle_wins']}`; dominance: `{diag['weak_dominance']}`.",
                "",
                "Top weak features where selected beats oracle:",
                "",
                format_feature_table(diag["weak_selected_top"]),
                "Top weak features where oracle beats selected:",
                "",
                format_feature_table(diag["weak_oracle_top"]),
                "### Classification",
                "",
                diag["classification"],
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            f"Source log: `{command_context.source_log}`",
            "",
            f"Fingerprint settings: `n_paths={command_context.n_paths}`, "
            f"`active_paths={command_context.active_paths}`, `n_steps={command_context.n_steps}`.",
            "",
            "No checkpoint, dataset pickle, target format, fingerprint schema, training path, "
            "or rerank default was changed.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    sample_ids = parse_sample_ids(args.samples)
    constant_values = parse_constant_values(args.constant_values)
    rows = parse_miss_log(args.source_log, set(sample_ids))
    config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    targets = reproduce_eval_targets(sample_ids, args.eval_seed, config, rows)
    active_names, weak_names = feature_names()

    diagnostics = {}
    for sample_idx in sample_ids:
        row = rows[sample_idx]
        target_y = targets[sample_idx]["target_y"]
        base_seed = args.eval_seed + 17
        base_seed += (sample_idx // args.batch_size) * 100003
        base_seed += (sample_idx % args.batch_size) * 1009
        combos = {
            "selected+selected": (row["selected_drift"], row["selected_diffusion"]),
            "oracle+oracle": (row["oracle_drift"], row["oracle_diffusion"]),
            "selected+oracle": (row["selected_drift"], row["oracle_diffusion"]),
            "oracle+selected": (row["oracle_drift"], row["selected_diffusion"]),
        }
        scored = {}
        score_cache = {}
        for offset, (label, (drift_tokens, diffusion_tokens)) in enumerate(combos.items()):
            cache_key = (tuple(drift_tokens), tuple(diffusion_tokens))
            if cache_key not in score_cache:
                score_cache[cache_key] = score_combo(
                    label,
                    drift_tokens,
                    diffusion_tokens,
                    target_y,
                    config,
                    constant_values,
                    base_seed + len(score_cache),
                )
            scored[label] = dict(score_cache[cache_key])
            scored[label]["label"] = label
        selected = scored["selected+selected"]
        oracle = scored["oracle+oracle"]
        if not selected.get("success") or not oracle.get("success"):
            raise RuntimeError(f"sample {sample_idx}: selected/oracle diagnostic failed")
        (
            active_selected_top,
            active_oracle_top,
            active_selected_wins,
            active_oracle_wins,
            active_dominance,
        ) = top_feature_rows(
            active_names,
            selected["active_residual"],
            oracle["active_residual"],
            args.top_k,
        )
        (
            weak_selected_top,
            weak_oracle_top,
            weak_selected_wins,
            weak_oracle_wins,
            weak_dominance,
        ) = top_feature_rows(
            weak_names,
            selected["weak_residual"],
            oracle["weak_residual"],
            args.top_k,
        )
        diagnostics[sample_idx] = {
            "combos": scored,
            "active_selected_top": active_selected_top,
            "active_oracle_top": active_oracle_top,
            "active_selected_wins": active_selected_wins,
            "active_oracle_wins": active_oracle_wins,
            "active_dominance": active_dominance,
            "weak_selected_top": weak_selected_top,
            "weak_oracle_top": weak_oracle_top,
            "weak_selected_wins": weak_selected_wins,
            "weak_oracle_wins": weak_oracle_wins,
            "weak_dominance": weak_dominance,
        }
        diagnostics[sample_idx]["classification"] = classify(
            row,
            selected,
            oracle,
            active_dominance,
            weak_dominance,
        )

    write_report(args.report, rows, diagnostics, args)
    print(f"wrote_report={args.report}")
    for sample_idx in sample_ids:
        diag = diagnostics[sample_idx]
        selected = diag["combos"]["selected+selected"]
        oracle = diag["combos"]["oracle+oracle"]
        print(
            f"sample={sample_idx} selected_total={fmt_float(selected.get('total'))} "
            f"oracle_total={fmt_float(oracle.get('total'))} "
            f"classification={diag['classification']}"
        )


if __name__ == "__main__":
    main()
