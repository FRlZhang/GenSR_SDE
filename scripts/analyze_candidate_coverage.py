#!/usr/bin/env python3
"""Candidate coverage diagnostics for the 32-sample SDE decoding setup."""

from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sde_fingerprint import FingerprintConfig
from sde_validation_probe import (
    build_env,
    build_modules,
    configure_gensr,
    decode_generation,
    generate_eval_samples,
    load_checkpoint_from_path,
    normalize_sample,
    parse_temperature_list,
    prior_scores_for_joint_batch,
    same_template,
    source_bucket,
    split_sde_tokens,
    template_tokens,
)
from symbolicregression.trainer_vae import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("candidate_coverage_diagnostics.md"))
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=Path("/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth"),
    )
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=800)
    parser.add_argument("--active-paths", type=int, default=800)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--max-generated-len", type=int, default=40)
    parser.add_argument("--min-generated-len", type=int, default=8)
    parser.add_argument("--constrained-beam-size", type=int, default=0)
    parser.add_argument("--rerank-candidates", type=int, default=8)
    parser.add_argument("--rerank-topk-from-beam", type=int, default=8)
    parser.add_argument("--sample-candidates", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-temperatures", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--sample-top-k", type=int, default=8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--pair-drift-diffusion-candidates", type=int, default=8)
    parser.add_argument("--pair-drift-topk", type=int, default=4)
    parser.add_argument("--pair-diffusion-topk", type=int, default=6)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-data", type=Path, default=Path("data/sde_train_dataset.pkl"))
    parser.add_argument("--print-freq", type=int, default=25)
    parser.add_argument("--n-enc-layers", type=int, default=None)
    parser.add_argument("--n-dec-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--split-sde-loss", action="store_true")
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true", default=True)
    parser.add_argument("--checkpoint-include-optimizer", action="store_true")
    return parser.parse_args()


def parse_float_metric(text: str, key: str) -> float | None:
    match = re.search(rf"^{re.escape(key)}=([0-9.]+)$", text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_source_log(path: Path, total_samples: int) -> dict:
    text = path.read_text() if path.is_file() else ""
    selected = parse_float_metric(text, "reranked_sequence_exact")
    oracle = parse_float_metric(text, "rerank_oracle_sequence_exact")
    pair_oracle = parse_float_metric(text, "rerank_pair_oracle_sequence_exact")
    return {
        "selected_count": None if selected is None else round(selected * total_samples),
        "oracle_count": None if oracle is None else round(oracle * total_samples),
        "pair_oracle_count": None if pair_oracle is None else round(pair_oracle * total_samples),
    }


def make_probe_args(args: argparse.Namespace) -> argparse.Namespace:
    probe_args = copy.copy(args)
    probe_args.rerank_score = "constant_grid_rolewise_no_multi_u0"
    probe_args.rerank_component_weights = "1.0,2.0,1.0"
    probe_args.rerank_constant_values = "0.25,0.5,1.0,2.0,4.0"
    probe_args.rerank_tie_epsilon = 0.005
    probe_args.rerank_tie_break = "none"
    probe_args.rerank_debug_topk = 0
    return probe_args


def setup_trainer(args: argparse.Namespace) -> Trainer:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    params = configure_gensr(make_probe_args(args))
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    load_checkpoint_from_path(trainer, args.load_checkpoint, requires_grad=False)
    return trainer


def build_candidate_rows(trainer: Trainer, eval_samples: list[dict], args: argparse.Namespace):
    sample_temperatures = parse_temperature_list(
        args.sample_temperatures,
        args.sample_temperature,
    )
    all_rows = []
    n_batches = int(np.ceil(len(eval_samples) / args.batch_size))
    for batch_idx in range(n_batches):
        chunk = eval_samples[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        normalized = [normalize_sample(sample, trainer.env) for sample in chunk]
        batch = {
            "x_to_fit": [sample["x_to_fit"] for sample in normalized],
            "y_to_fit": [sample["y_to_fit"] for sample in normalized],
            "tree_encoded": [sample["tree_encoded"] for sample in normalized],
            "drift_tree_encoded": [sample["drift_tree_encoded"] for sample in normalized],
            "diffusion_tree_encoded": [sample["diffusion_tree_encoded"] for sample in normalized],
            "tree": [sample["tree"] for sample in chunk],
        }
        *_, candidate_rows = prior_scores_for_joint_batch(
            trainer,
            batch,
            args.min_generated_len,
            args.constrained_beam_size,
            args.rerank_candidates,
            args.rerank_topk_from_beam,
            args.sample_candidates,
            sample_temperatures,
            args.sample_top_k,
            args.sample_top_p,
            args.pair_drift_diffusion_candidates,
            args.pair_drift_topk,
            args.pair_diffusion_topk,
        )
        all_rows.extend(candidate_rows or [[] for _ in chunk])
    return all_rows


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        cur = [i]
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def nearest_templates(candidates: set[tuple[str, ...]], truth: list[str], limit: int = 3) -> list[tuple[int, str]]:
    truth_template = template_tokens(truth)
    rows = []
    for candidate in candidates:
        candidate_tokens = list(candidate)
        distance = edit_distance(template_tokens(candidate_tokens), truth_template)
        rows.append((distance, " ".join(candidate_tokens)))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows[:limit]


def source_set_string(sources: set[str]) -> str:
    if not sources:
        return "-"
    return ",".join(sorted(sources))


def analyze_sample(env, sample_idx: int, truth: list[str], rows: list[dict]) -> dict:
    truth_split = split_sde_tokens(truth)
    if truth_split is None:
        raise ValueError(f"sample {sample_idx} truth is not an SDE sequence")
    truth_drift, truth_diffusion = truth_split
    truth_drift_key = tuple(template_tokens(truth_drift))
    truth_diffusion_key = tuple(template_tokens(truth_diffusion))
    truth_full_key = tuple(template_tokens(truth))

    full_sources = set()
    drift_sources = set()
    diffusion_sources = set()
    source_counts = Counter()
    drift_by_source = defaultdict(set)
    diffusion_by_source = defaultdict(set)
    full_by_source = defaultdict(set)
    generated_drifts = set()
    generated_diffusions = set()
    decoded_candidates = []

    for candidate in rows:
        words = decode_generation(env, candidate["tensor"])
        split = split_sde_tokens(words)
        if split is None:
            continue
        source = source_bucket(candidate.get("source", "beam"))
        source_counts[source] += 1
        drift_tokens, diffusion_tokens = split
        drift_key = tuple(template_tokens(drift_tokens))
        diffusion_key = tuple(template_tokens(diffusion_tokens))
        full_key = tuple(template_tokens(words))
        generated_drifts.add(tuple(drift_tokens))
        generated_diffusions.add(tuple(diffusion_tokens))
        drift_by_source[source].add(drift_key)
        diffusion_by_source[source].add(diffusion_key)
        full_by_source[source].add(full_key)
        decoded_candidates.append((source, words, drift_tokens, diffusion_tokens))
        if full_key == truth_full_key:
            full_sources.add(source)
        if drift_key == truth_drift_key:
            drift_sources.add(source)
        if diffusion_key == truth_diffusion_key:
            diffusion_sources.add(source)

    has_full = bool(full_sources)
    has_drift = bool(drift_sources)
    has_diffusion = bool(diffusion_sources)
    if has_full:
        coverage = "full_oracle_present"
    elif has_drift and has_diffusion:
        coverage = "drift_and_diffusion_separate_not_paired"
    elif has_drift:
        coverage = "drift_only_present"
    elif has_diffusion:
        coverage = "diffusion_only_present"
    else:
        coverage = "neither_side_present"

    drift_nearest = nearest_templates(generated_drifts, truth_drift)
    diffusion_nearest = nearest_templates(generated_diffusions, truth_diffusion)
    missing_side = (
        "none"
        if has_full
        else "pairing"
        if has_drift and has_diffusion
        else "diffusion"
        if has_drift
        else "drift"
        if has_diffusion
        else "both"
    )
    recommendation = {
        "none": "full oracle already covered",
        "pairing": "improve pairing/ranking of exact drift and diffusion spans",
        "diffusion": "improve diffusion span generation/diversity",
        "drift": "improve drift span generation/diversity",
        "both": "increase joint candidate diversity for both roles",
    }[missing_side]
    near_miss = (
        not has_full
        and (
            (not has_drift and drift_nearest and drift_nearest[0][0] <= 2)
            or (not has_diffusion and diffusion_nearest and diffusion_nearest[0][0] <= 2)
        )
    )
    return {
        "sample_index": sample_idx,
        "truth": truth,
        "truth_drift": truth_drift,
        "truth_diffusion": truth_diffusion,
        "coverage": coverage,
        "has_full": has_full,
        "has_drift": has_drift,
        "has_diffusion": has_diffusion,
        "near_miss": near_miss,
        "full_sources": full_sources,
        "drift_sources": drift_sources,
        "diffusion_sources": diffusion_sources,
        "source_counts": source_counts,
        "drift_by_source": drift_by_source,
        "diffusion_by_source": diffusion_by_source,
        "full_by_source": full_by_source,
        "unique_drifts": len({tuple(template_tokens(list(item))) for item in generated_drifts}),
        "unique_diffusions": len({tuple(template_tokens(list(item))) for item in generated_diffusions}),
        "drift_nearest": drift_nearest,
        "diffusion_nearest": diffusion_nearest,
        "missing_side": missing_side,
        "recommendation": recommendation,
    }


def write_report(path: Path, analyses: list[dict], log_metrics: dict, args: argparse.Namespace) -> None:
    total = len(analyses)
    full_count = sum(item["has_full"] for item in analyses)
    selected_count = log_metrics.get("selected_count")
    oracle_absent = [item for item in analyses if not item["has_full"]]
    coverage_counts = Counter(item["coverage"] for item in analyses)
    near_miss_count = sum(item["near_miss"] for item in oracle_absent)

    source_exact_full = Counter()
    source_exact_drift = Counter()
    source_exact_diffusion = Counter()
    source_candidate_counts = Counter()
    for item in analyses:
        source_candidate_counts.update(item["source_counts"])
        for source in item["full_sources"]:
            source_exact_full[source] += 1
        for source in item["drift_sources"]:
            source_exact_drift[source] += 1
        for source in item["diffusion_sources"]:
            source_exact_diffusion[source] += 1

    missing_counts = Counter(item["missing_side"] for item in oracle_absent)
    main_missing = missing_counts.most_common(1)[0][0] if missing_counts else "none"

    lines = [
        "# Candidate Coverage Diagnostics",
        "",
        "Date: 2026-06-18",
        "",
        "Scope: candidate identity coverage for the existing 32-sample setup. "
        "This is not a formal rerank eval and does not compute fingerprint scores.",
        "",
        "## Executive Summary",
        "",
        "- Current selected baseline: `5/32`.",
        "- Current oracle ceiling: `9/32`.",
        f"- Reconstructed full-oracle candidate coverage: `{full_count}/{total}`.",
        f"- Oracle-absent samples: `{len(oracle_absent)}/{total}`.",
        f"- Main missing piece among oracle-absent samples: `{main_missing}`.",
        f"- Near-miss side coverage among oracle-absent samples: `{near_miss_count}/{len(oracle_absent)}`.",
        "",
        "## Aggregate Coverage",
        "",
        "| Category | Count |",
        "| --- | ---: |",
        f"| Full oracle present | {full_count} |",
        f"| Scorer selected oracle | {selected_count if selected_count is not None else 'unknown'} |",
        f"| Oracle present but selected missed | {max(0, full_count - selected_count) if selected_count is not None else 'unknown'} |",
        f"| Drift-only present | {coverage_counts['drift_only_present']} |",
        f"| Diffusion-only present | {coverage_counts['diffusion_only_present']} |",
        f"| Drift+diffusion both present separately but not paired | {coverage_counts['drift_and_diffusion_separate_not_paired']} |",
        f"| Neither side present | {coverage_counts['neither_side_present']} |",
        "",
        "## Source Breakdown",
        "",
        "| Source | Candidate rows | Exact full oracle samples | Exact drift samples | Exact diffusion samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in ("beam", "sampling", "pair", "unknown"):
        lines.append(
            f"| {source} | {source_candidate_counts[source]} | "
            f"{source_exact_full[source]} | {source_exact_drift[source]} | "
            f"{source_exact_diffusion[source]} |"
        )
    lines.extend(
        [
            "",
            "## Oracle-absent Samples",
            "",
            "| Sample | Truth drift | Truth diffusion | Nearest drift candidates | Nearest diffusion candidates | Missing side | Recommendation |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in oracle_absent:
        drift_nearest = "<br>".join(
            f"d={dist}: `{tokens}`" for dist, tokens in item["drift_nearest"]
        ) or "-"
        diffusion_nearest = "<br>".join(
            f"d={dist}: `{tokens}`" for dist, tokens in item["diffusion_nearest"]
        ) or "-"
        lines.append(
            f"| {item['sample_index']} | `{' '.join(item['truth_drift'])}` | "
            f"`{' '.join(item['truth_diffusion'])}` | {drift_nearest} | "
            f"{diffusion_nearest} | {item['missing_side']} | {item['recommendation']} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if main_missing == "both":
        lines.append(
            "The dominant issue is candidate diversity for both roles, not pairing alone. "
            "Improve drift and diffusion span diversity before adding another scorer."
        )
    elif main_missing == "drift":
        lines.append(
            "The dominant issue is missing drift spans. Prioritize drift candidate generation."
        )
    elif main_missing == "diffusion":
        lines.append(
            "The dominant issue is missing diffusion spans. Prioritize diffusion candidate generation."
        )
    elif main_missing == "pairing":
        lines.append(
            "Both sides often exist separately; prioritize pairing coverage/ranking."
        )
    else:
        lines.append("Full oracle coverage is not the current blocker in this diagnostic.")
    lines.extend(
        [
            "",
            "## Command Context",
            "",
            f"- Source log: `{args.source_log}`",
            f"- Checkpoint loaded read-only: `{args.load_checkpoint}`",
            f"- Candidate setup: beam `{args.rerank_candidates}`, sampling `{args.sample_candidates}`, pair `{args.pair_drift_diffusion_candidates}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    log_metrics = parse_source_log(args.source_log, args.eval_samples)
    trainer = setup_trainer(args)
    fingerprint_config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    eval_samples = generate_eval_samples(make_probe_args(args), trainer.env, trainer.params)
    candidate_rows = build_candidate_rows(trainer, eval_samples, args)
    analyses = []
    for sample_idx, (sample, rows) in enumerate(zip(eval_samples, candidate_rows)):
        normalized = normalize_sample(sample, trainer.env)
        analyses.append(
            analyze_sample(
                trainer.env,
                sample_idx,
                normalized["tree_encoded"],
                rows,
            )
        )
    write_report(args.report, analyses, log_metrics, args)
    full_count = sum(item["has_full"] for item in analyses)
    missing_counts = Counter(item["missing_side"] for item in analyses if not item["has_full"])
    print(f"wrote_report={args.report}")
    print(f"full_oracle_present={full_count}/{len(analyses)}")
    print(
        "oracle_absent_missing_sides="
        + ",".join(f"{key}:{value}" for key, value in sorted(missing_counts.items()))
    )


if __name__ == "__main__":
    main()
