#!/usr/bin/env python3
"""Candidate coverage diagnostics for the 32-sample SDE decoding setup."""

from __future__ import annotations

import argparse
import copy
import json
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

from sde_fingerprint import FingerprintConfig, fingerprint_length
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
from normalized_drift_admission import (
    build_p2_normalized_admission_pool,
    canonicalize_constant_chains,
    collision_summary,
    current_expanded_drift_rows,
    drift_only_candidate_rows,
    safety_checks,
)


NORMALIZED_DRIFT_ADMISSION_POLICIES = ("none", "p2_one_per_family")
DEFAULT_NORMALIZED_DRIFT_ADMISSION = "none"
DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE = "beam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("candidate_coverage_diagnostics.md"))
    parser.add_argument("--json-output", type=Path, default=None)
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
    parser.add_argument(
        "--normalized-drift-admission",
        choices=NORMALIZED_DRIFT_ADMISSION_POLICIES,
        default=DEFAULT_NORMALIZED_DRIFT_ADMISSION,
        help="Coverage-only normalized drift admission diagnostic. Default keeps existing output unchanged.",
    )
    parser.add_argument(
        "--drift-only-json",
        type=Path,
        default=None,
        help="Required when --normalized-drift-admission is enabled.",
    )
    parser.add_argument(
        "--normalized-drift-admission-source",
        choices=["beam", "all"],
        default=DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE,
        help="Source filter for normalized drift admission; beam matches validated P2 diagnostics.",
    )
    parser.add_argument(
        "--scorer-ready-json",
        type=Path,
        default=None,
        help="Optional sidecar JSON with target fingerprints and scorer-ready P2 candidate rows.",
    )
    parser.add_argument(
        "--scorer-ready-target-samples",
        type=str,
        default="",
        help="Comma-separated sample indices for scorer-ready logging; defaults to P2 newly recovered samples.",
    )
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


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def array_payload(value) -> dict:
    array = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "flat": array.reshape(-1).tolist(),
    }


def decoded_candidate_records(env, rows: list[dict]) -> list[dict]:
    records = []
    source_ranks = Counter()
    for candidate_rank, candidate in enumerate(rows, start=1):
        words = decode_generation(env, candidate["tensor"])
        split = split_sde_tokens(words)
        if split is None:
            continue
        source = source_bucket(candidate.get("source", "beam"))
        source_ranks[source] += 1
        drift_tokens, diffusion_tokens = split
        records.append(
            {
                "candidate_rank": candidate_rank,
                "source": source,
                "source_rank": source_ranks[source],
                "normalized_model_score": float(candidate.get("normalized_model_score", float("-inf"))),
                "full_sequence_tokens": words,
                "drift_tokens": drift_tokens,
                "diffusion_tokens": diffusion_tokens,
            }
        )
    return records


def first_diffusion_records(records: list[dict]) -> list[dict]:
    by_key = {}
    for record in records:
        key = tuple(template_tokens(record["diffusion_tokens"]))
        current = by_key.get(key)
        if current is None or record["candidate_rank"] < current["candidate_rank"]:
            by_key[key] = record
    return sorted(by_key.values(), key=lambda item: item["candidate_rank"])


def parse_ready(tokens: list[str]) -> bool:
    split = split_sde_tokens(tokens)
    if split is None:
        return False
    drift_tokens, diffusion_tokens = split
    return bool(drift_tokens and diffusion_tokens)


def candidate_sidecar_row(
    candidate_id: str,
    candidate_source: str,
    drift_tokens_raw: list[str],
    drift_tokens_canonical: list[str] | None,
    diffusion_tokens_raw: list[str],
    truth_drift_canonical: list[str] | None,
    truth_diffusion: list[str],
    drift_source: str | None,
    diffusion_source: str | None,
    drift_rank: int | None,
    diffusion_rank: int | None,
    pair_rank: int | None,
    is_current_expanded_candidate: bool,
) -> dict:
    full_sequence_tokens = ["<DRIFT>"] + (drift_tokens_canonical or drift_tokens_raw) + [
        "<DIFFUSION>"
    ] + diffusion_tokens_raw
    truth_drift_key = tuple(truth_drift_canonical or [])
    diffusion_key = tuple(template_tokens(diffusion_tokens_raw))
    truth_diffusion_key = tuple(template_tokens(truth_diffusion))
    canonical_key = tuple(drift_tokens_canonical or [])
    is_truth_drift = bool(truth_drift_key and canonical_key == truth_drift_key)
    is_truth_diffusion = diffusion_key == truth_diffusion_key
    return {
        "candidate_id": candidate_id,
        "candidate_source": candidate_source,
        "drift_tokens_raw": drift_tokens_raw,
        "drift_tokens_canonical": drift_tokens_canonical,
        "diffusion_tokens_raw": diffusion_tokens_raw,
        "diffusion_tokens_canonical": diffusion_tokens_raw,
        "full_sequence_tokens": full_sequence_tokens,
        "full_sequence_string": " ".join(full_sequence_tokens),
        "drift_source": drift_source,
        "diffusion_source": diffusion_source,
        "drift_rank": drift_rank,
        "diffusion_rank": diffusion_rank,
        "pair_rank": pair_rank,
        "is_truth_drift_canonical": is_truth_drift,
        "is_truth_diffusion_exact": is_truth_diffusion,
        "is_p2_canonical_oracle": candidate_source == "p2_admitted" and is_truth_drift and is_truth_diffusion,
        "is_current_expanded_candidate": is_current_expanded_candidate,
        "parse_ready_bool": parse_ready(full_sequence_tokens),
    }


def build_scorer_ready_sidecar(
    env,
    eval_samples: list[dict],
    candidate_rows: list[list[dict]],
    analyses: list[dict],
    args: argparse.Namespace,
    fingerprint_config: FingerprintConfig,
    normalized_admission_block: dict | None,
) -> dict:
    if normalized_admission_block is None:
        raise ValueError("--scorer-ready-json requires --normalized-drift-admission p2_one_per_family")

    analyses_by_index = {item["sample_index"]: item for item in analyses}
    p2_samples_by_index = {
        item["sample_index"]: item for item in normalized_admission_block.get("samples", [])
    }
    admitted_by_sample = normalized_admission_block.get("admitted_candidates_by_sample", {})
    requested = parse_int_list(args.scorer_ready_target_samples)
    target_indices = requested or list(normalized_admission_block.get("newly_recovered", []))
    samples = []
    fingerprint_lengths = []

    for sample_idx in target_indices:
        sample = eval_samples[sample_idx]
        normalized = normalize_sample(sample, env)
        analysis = analyses_by_index[sample_idx]
        p2_sample = p2_samples_by_index.get(sample_idx, {})
        records = decoded_candidate_records(env, candidate_rows[sample_idx])
        diffusion_records = first_diffusion_records(records)
        truth_sequence = normalized["tree_encoded"]
        truth_drift, truth_diffusion = split_sde_tokens(truth_sequence)
        truth_drift_canonical = canonicalize_constant_chains(truth_drift)
        y_payload = array_payload(normalized["y_to_fit"])
        x_payload = array_payload(normalized["x_to_fit"])
        fingerprint_lengths.append(len(y_payload["flat"]))

        candidate_sidecar_rows = []
        for record in records:
            drift_canonical = canonicalize_constant_chains(record["drift_tokens"])
            candidate_sidecar_rows.append(
                candidate_sidecar_row(
                    candidate_id=f"sample{sample_idx}_current_{record['candidate_rank']:04d}",
                    candidate_source="current_expanded",
                    drift_tokens_raw=record["drift_tokens"],
                    drift_tokens_canonical=drift_canonical,
                    diffusion_tokens_raw=record["diffusion_tokens"],
                    truth_drift_canonical=truth_drift_canonical,
                    truth_diffusion=truth_diffusion,
                    drift_source=record["source"],
                    diffusion_source=record["source"],
                    drift_rank=record["source_rank"],
                    diffusion_rank=record["source_rank"],
                    pair_rank=record["candidate_rank"] if record["source"] == "pair" else None,
                    is_current_expanded_candidate=True,
                )
            )

        p2_pair_rank = 0
        admitted_rows = admitted_by_sample.get(sample_idx, admitted_by_sample.get(str(sample_idx), []))
        for admitted in admitted_rows:
            for diffusion_record in diffusion_records:
                p2_pair_rank += 1
                candidate_sidecar_rows.append(
                    candidate_sidecar_row(
                        candidate_id=f"sample{sample_idx}_p2_{p2_pair_rank:04d}",
                        candidate_source="p2_admitted",
                        drift_tokens_raw=admitted.get("raw_drift_tokens") or admitted.get("canonical_drift_tokens") or [],
                        drift_tokens_canonical=admitted.get("canonical_drift_tokens"),
                        diffusion_tokens_raw=diffusion_record["diffusion_tokens"],
                        truth_drift_canonical=truth_drift_canonical,
                        truth_diffusion=truth_diffusion,
                        drift_source=admitted.get("source"),
                        diffusion_source=diffusion_record["source"],
                        drift_rank=admitted.get("rank"),
                        diffusion_rank=diffusion_record["source_rank"],
                        pair_rank=p2_pair_rank,
                        is_current_expanded_candidate=False,
                    )
                )

        samples.append(
            {
                "sample_index": sample_idx,
                "truth_sequence": truth_sequence,
                "truth_drift_tokens": truth_drift,
                "truth_diffusion_tokens": truth_diffusion,
                "target_y_to_fit": y_payload,
                "target_fingerprint": y_payload,
                "target_x_to_fit": x_payload,
                "target_fingerprint_length": len(y_payload["flat"]),
                "p2_newly_recovered_bool": sample_idx in set(normalized_admission_block.get("newly_recovered", [])),
                "current_expanded_oracle_present_bool": bool(analysis.get("has_full")),
                "canonicalized_expanded_oracle_present_bool": bool(
                    p2_sample.get("current_canonical_full_oracle")
                ),
                "p2_oracle_present_bool": bool(p2_sample.get("p2_normalized_full_oracle")),
                "exact_diffusion_present_bool": bool(p2_sample.get("exact_diffusion_present")),
                "candidates": candidate_sidecar_rows,
            }
        )

    readiness_status = (
        "complete"
        if samples
        and len(set(fingerprint_lengths)) == 1
        and all(sample.get("target_y_to_fit", {}).get("flat") for sample in samples)
        else "incomplete"
    )
    schema_length = fingerprint_length(fingerprint_config)
    return {
        "metadata": {
            "source_coverage_json": str(args.json_output or args.report.with_suffix(".json")),
            "source_log": str(args.source_log),
            "checkpoint": str(args.load_checkpoint),
            "normalized_drift_admission_policy": args.normalized_drift_admission,
            "normalized_drift_admission_source": args.normalized_drift_admission_source,
            "target_samples_included": target_indices,
            "fingerprint_schema_or_length": {
                "fingerprint_length_from_config": schema_length,
                "observed_lengths": sorted(set(fingerprint_lengths)),
            },
            "scorer_readiness_status": readiness_status,
            "semantic_scoring_run": False,
        },
        "samples": samples,
    }


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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def p2_source_breakdown(samples: list[dict]) -> dict[str, int]:
    counts = Counter()
    for sample in samples:
        if not sample.get("p2_normalized_full_oracle"):
            continue
        truth_key = tuple(sample.get("canonical_truth_drift_tokens") or [])
        sources = {
            row.get("source")
            for row in sample.get("admitted_candidates", [])
            if tuple(row.get("canonical_drift_tokens", [])) == truth_key and row.get("source")
        }
        if sources:
            counts["+".join(sorted(sources))] += 1
    return dict(counts)


def p2_sampling_recovery_count(samples: list[dict]) -> int:
    count = 0
    for sample in samples:
        truth_key = tuple(sample.get("canonical_truth_drift_tokens") or [])
        if any(
            row.get("source") == "sampling"
            and tuple(row.get("canonical_drift_tokens", [])) == truth_key
            for row in sample.get("admitted_candidates", [])
        ):
            count += 1
    return count


def build_normalized_drift_admission_block(
    coverage_payload: dict,
    drift_only_payload: dict,
    policy: str = "p2_one_per_family",
    source: str = DEFAULT_NORMALIZED_DRIFT_ADMISSION_SOURCE,
) -> dict:
    if policy != "p2_one_per_family":
        raise ValueError(f"unsupported normalized drift admission policy: {policy}")

    include_sampling = source == "all"
    pool = build_p2_normalized_admission_pool(
        coverage_payload,
        drift_only_payload,
        admission_source=source,
        include_sampling=include_sampling,
    )
    target_samples = pool["samples"]
    current_full = sum(1 for sample in coverage_payload["samples"] if sample.get("has_full"))
    current_canonical_hits = [sample for sample in target_samples if sample["current_canonical_full_oracle"]]
    p2_hits = [sample for sample in target_samples if sample["p2_normalized_full_oracle"]]
    newly = [
        sample["sample_index"]
        for sample in p2_hits
        if not sample["current_canonical_full_oracle"]
    ]
    remaining = [sample["sample_index"] for sample in target_samples if not sample["p2_normalized_full_oracle"]]
    family_counts = Counter(sample["drift_family"] for sample in p2_hits)

    expanded_by_index = {sample["sample_index"]: sample for sample in coverage_payload["samples"]}
    all_rows = []
    polynomial_truths = []
    for drift_sample in drift_only_payload["samples"]:
        sample_idx = drift_sample["sample_index"]
        expanded_sample = expanded_by_index[sample_idx]
        if drift_sample.get("drift_family") == "polynomial-like drift":
            polynomial_truths.append(expanded_sample["truth_drift"])
        all_rows.extend(current_expanded_drift_rows(expanded_sample))
        all_rows.extend(drift_only_candidate_rows(drift_sample, include_sampling=True))
    safety = collision_summary(all_rows, polynomial_truths)
    safety["self_checks"] = safety_checks()

    admitted_by_sample = {
        sample["sample_index"]: [
            {
                "raw_drift_tokens": row.get("raw_drift_tokens"),
                "canonical_drift_tokens": row.get("canonical_drift_tokens"),
                "source": row.get("source"),
                "rank": row.get("source_rank"),
                "family": row.get("family"),
                "filter_reason": row.get("filter_reason"),
            }
            for row in sample.get("admitted_candidates", [])
        ]
        for sample in target_samples
    }
    return {
        "policy": policy,
        "source": source,
        "sampling_excluded": not include_sampling,
        "target_oracle_absent_samples_analyzed": len(target_samples),
        "current_expanded_full_oracle": current_full,
        "canonicalized_expanded_pool_coverage": current_full + len(current_canonical_hits),
        "p2_normalized_admission_coverage": current_full + len(p2_hits),
        "p2_pair_count": sum(sample["pair_count"] for sample in target_samples),
        "newly_recovered": newly,
        "remaining_missing": remaining,
        "sampling_recovered_canonical_drift_count": p2_sampling_recovery_count(target_samples),
        "unsafe_collision_flag": bool(safety["unsafe_collision_flag"]),
        "source_breakdown": p2_source_breakdown(target_samples),
        "family_breakdown": {
            family: family_counts.get(family, 0)
            for family in [
                "linear drift",
                "sin drift",
                "nested-mul linear drift",
                "polynomial-like drift",
                "other",
            ]
        },
        "safety": safety,
        "admitted_candidates_by_sample": admitted_by_sample,
        "samples": [
            {
                "sample_index": sample["sample_index"],
                "truth_drift_tokens": sample["truth_drift_tokens"],
                "canonical_truth_drift_tokens": sample["canonical_truth_drift_tokens"],
                "truth_diffusion_tokens": sample["truth_diffusion_tokens"],
                "drift_family": sample["drift_family"],
                "exact_diffusion_present": sample["exact_diffusion_present"],
                "current_expanded_full_oracle": sample["current_expanded_full_oracle"],
                "current_canonical_full_oracle": sample["current_canonical_full_oracle"],
                "p2_normalized_full_oracle": sample["p2_normalized_full_oracle"],
                "pair_count": sample["pair_count"],
            }
            for sample in target_samples
        ],
    }


def drift_family(tokens: list[str]) -> str:
    templ = template_tokens(tokens)
    templ_tuple = tuple(templ)
    if templ_tuple == ("mul", "x_0"):
        return "linear drift"
    if templ_tuple == ("mul",):
        return "constant drift"
    if "sub" in templ and "x_0" in templ:
        return "mean-reverting / affine drift"
    if "sin" in templ:
        return "sin drift"
    if "pow2" in templ or "pow3" in templ or "pow" in templ:
        return "polynomial-like drift"
    if templ and templ[0] == "mul" and templ.count("mul") >= 2:
        return "nested mul structural variant"
    return "other"


def analyze_sample(
    env,
    sample_idx: int,
    truth: list[str],
    rows: list[dict],
    pair_drift_topk: int,
    pair_diffusion_topk: int,
    pair_candidate_cap: int,
) -> dict:
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
    source_ranks = Counter()
    exact_drift_occurrences = []
    exact_diffusion_occurrences = []
    exact_full_occurrences = []
    drift_span_scores = {}
    diffusion_span_scores = {}

    for candidate_rank, candidate in enumerate(rows, start=1):
        words = decode_generation(env, candidate["tensor"])
        split = split_sde_tokens(words)
        if split is None:
            continue
        source = source_bucket(candidate.get("source", "beam"))
        source_ranks[source] += 1
        source_rank = source_ranks[source]
        model_score = float(candidate.get("normalized_model_score", float("-inf")))
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
        decoded_candidates.append(
            (source, source_rank, candidate_rank, words, drift_tokens, diffusion_tokens)
        )
        if source != "pair":
            drift_record = drift_span_scores.get(drift_key)
            if drift_record is None or model_score > drift_record["score"]:
                drift_span_scores[drift_key] = {
                    "tokens": drift_tokens,
                    "score": model_score,
                    "source": source,
                    "source_rank": source_rank,
                    "candidate_rank": candidate_rank,
                }
            diffusion_record = diffusion_span_scores.get(diffusion_key)
            if diffusion_record is None or model_score > diffusion_record["score"]:
                diffusion_span_scores[diffusion_key] = {
                    "tokens": diffusion_tokens,
                    "score": model_score,
                    "source": source,
                    "source_rank": source_rank,
                    "candidate_rank": candidate_rank,
                }
        if full_key == truth_full_key:
            full_sources.add(source)
            exact_full_occurrences.append(
                {
                    "source": source,
                    "source_rank": source_rank,
                    "candidate_rank": candidate_rank,
                    "score": model_score,
                }
            )
        if drift_key == truth_drift_key:
            drift_sources.add(source)
            exact_drift_occurrences.append(
                {
                    "source": source,
                    "source_rank": source_rank,
                    "candidate_rank": candidate_rank,
                    "score": model_score,
                }
            )
        if diffusion_key == truth_diffusion_key:
            diffusion_sources.add(source)
            exact_diffusion_occurrences.append(
                {
                    "source": source,
                    "source_rank": source_rank,
                    "candidate_rank": candidate_rank,
                    "score": model_score,
                }
            )

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
    drift_nearest_detailed = nearest_templates_detailed(
        generated_drifts,
        truth_drift,
        decoded_candidates,
    )
    diffusion_nearest_detailed = nearest_templates_detailed(
        generated_diffusions,
        truth_diffusion,
        decoded_candidates,
        role="diffusion",
    )
    drift_span_rank = rank_span(drift_span_scores, truth_drift_key)
    diffusion_span_rank = rank_span(diffusion_span_scores, truth_diffusion_key)
    pair_cross_rank = None
    if drift_span_rank is not None and diffusion_span_rank is not None:
        pair_cross_rank = rank_pair_cross_product(
            drift_span_scores,
            diffusion_span_scores,
            truth_drift_key,
            truth_diffusion_key,
            drift_topk=0,
            diffusion_topk=0,
        )
    limited_pair_cross_rank = None
    if drift_span_rank is not None and diffusion_span_rank is not None:
        limited_pair_cross_rank = rank_pair_cross_product(
            drift_span_scores,
            diffusion_span_scores,
            truth_drift_key,
            truth_diffusion_key,
            drift_topk=pair_drift_topk,
            diffusion_topk=pair_diffusion_topk,
        )
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
        "truth_drift_family": drift_family(truth_drift),
        "coverage": coverage,
        "has_full": has_full,
        "has_drift": has_drift,
        "has_diffusion": has_diffusion,
        "near_miss": near_miss,
        "full_sources": full_sources,
        "drift_sources": drift_sources,
        "diffusion_sources": diffusion_sources,
        "exact_full_occurrences": exact_full_occurrences,
        "exact_drift_occurrences": exact_drift_occurrences,
        "exact_diffusion_occurrences": exact_diffusion_occurrences,
        "drift_span_rank": drift_span_rank,
        "diffusion_span_rank": diffusion_span_rank,
        "pair_cross_rank": pair_cross_rank,
        "limited_pair_cross_rank": limited_pair_cross_rank,
        "exact_drift_outside_pair_topk": (
            drift_span_rank is None or drift_span_rank > pair_drift_topk
        ),
        "exact_diffusion_outside_pair_topk": (
            diffusion_span_rank is None or diffusion_span_rank > pair_diffusion_topk
        ),
        "pair_cap_prevents": (
            drift_span_rank is not None
            and diffusion_span_rank is not None
            and drift_span_rank <= pair_drift_topk
            and diffusion_span_rank <= pair_diffusion_topk
            and (
                limited_pair_cross_rank is None
                or limited_pair_cross_rank > pair_candidate_cap
            )
        ),
        "source_counts": source_counts,
        "drift_by_source": drift_by_source,
        "diffusion_by_source": diffusion_by_source,
        "full_by_source": full_by_source,
        "unique_drifts": len({tuple(template_tokens(list(item))) for item in generated_drifts}),
        "unique_diffusions": len({tuple(template_tokens(list(item))) for item in generated_diffusions}),
        "drift_nearest": drift_nearest,
        "diffusion_nearest": diffusion_nearest,
        "drift_nearest_detailed": drift_nearest_detailed,
        "diffusion_nearest_detailed": diffusion_nearest_detailed,
        "missing_side": missing_side,
        "recommendation": recommendation,
    }


def nearest_templates_detailed(
    candidates: set[tuple[str, ...]],
    truth: list[str],
    decoded_candidates: list[tuple],
    role: str = "drift",
    limit: int = 5,
) -> list[dict]:
    truth_template = template_tokens(truth)
    rows = []
    for candidate in candidates:
        candidate_tokens = list(candidate)
        distance = edit_distance(template_tokens(candidate_tokens), truth_template)
        occurrences = []
        for source, source_rank, candidate_rank, _words, drift_tokens, diffusion_tokens in decoded_candidates:
            role_tokens = drift_tokens if role == "drift" else diffusion_tokens
            if tuple(role_tokens) == tuple(candidate_tokens):
                occurrences.append(
                    {
                        "source": source,
                        "source_rank": source_rank,
                        "candidate_rank": candidate_rank,
                    }
                )
        rows.append(
            {
                "distance": distance,
                "tokens": candidate_tokens,
                "occurrences": sorted(
                    occurrences,
                    key=lambda item: (item["source"], item["source_rank"]),
                ),
                "sources": sorted({item["source"] for item in occurrences}),
                "family": drift_family(candidate_tokens) if role == "drift" else None,
                "right_family": (
                    drift_family(candidate_tokens) == drift_family(truth)
                    if role == "drift"
                    else None
                ),
            }
        )
    rows.sort(key=lambda item: (item["distance"], " ".join(item["tokens"])))
    return rows[:limit]


def rank_span(span_scores: dict, target_key: tuple[str, ...]) -> int | None:
    ranked = sorted(
        span_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    for rank, (key, _record) in enumerate(ranked, start=1):
        if key == target_key:
            return rank
    return None


def rank_pair_cross_product(
    drift_span_scores: dict,
    diffusion_span_scores: dict,
    truth_drift_key: tuple[str, ...],
    truth_diffusion_key: tuple[str, ...],
    drift_topk: int,
    diffusion_topk: int,
) -> int | None:
    ranked_drifts = sorted(
        drift_span_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    ranked_diffusions = sorted(
        diffusion_span_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    if drift_topk > 0:
        ranked_drifts = ranked_drifts[:drift_topk]
    if diffusion_topk > 0:
        ranked_diffusions = ranked_diffusions[:diffusion_topk]
    pairs = []
    for drift_key, drift_record in ranked_drifts:
        for diffusion_key, diffusion_record in ranked_diffusions:
            pairs.append(
                (
                    drift_record["score"] + diffusion_record["score"],
                    drift_key,
                    diffusion_key,
                )
            )
    pairs.sort(key=lambda item: item[0], reverse=True)
    for rank, (_score, drift_key, diffusion_key) in enumerate(pairs, start=1):
        if drift_key == truth_drift_key and diffusion_key == truth_diffusion_key:
            return rank
    return None


def write_report(
    path: Path,
    analyses: list[dict],
    log_metrics: dict,
    args: argparse.Namespace,
    normalized_admission_block: dict | None = None,
) -> None:
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
    if normalized_admission_block is not None:
        lines.extend(
            [
                "",
                "## P2 Normalized Drift Admission",
                "",
                "This optional section is coverage-only and uses the reusable normalized drift admission hook. "
                "It does not run reranking, fingerprint scoring, or candidate generation beyond the current coverage run.",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Policy | `{normalized_admission_block['policy']}` |",
                f"| Source | `{normalized_admission_block['source']}` |",
                f"| Current expanded full oracle | `{normalized_admission_block['current_expanded_full_oracle']}/{total}` |",
                f"| Canonicalized expanded-pool coverage | `{normalized_admission_block['canonicalized_expanded_pool_coverage']}/{total}` |",
                f"| P2 normalized admission coverage | `{normalized_admission_block['p2_normalized_admission_coverage']}/{total}` |",
                f"| P2 pair count | `{normalized_admission_block['p2_pair_count']}` |",
                f"| Newly recovered | `{','.join(str(i) for i in normalized_admission_block['newly_recovered'])}` |",
                f"| Remaining missing | `{','.join(str(i) for i in normalized_admission_block['remaining_missing'])}` |",
                f"| Sampling excluded | `{normalized_admission_block['sampling_excluded']}` |",
                f"| Sampling recovered canonical drift count | `{normalized_admission_block['sampling_recovered_canonical_drift_count']}` |",
                f"| Unsafe collision flag | `{normalized_admission_block['unsafe_collision_flag']}` |",
                "",
                f"- Source breakdown: `{json.dumps(normalized_admission_block['source_breakdown'], sort_keys=True)}`.",
                f"- Family breakdown: `{json.dumps(normalized_admission_block['family_breakdown'], sort_keys=True)}`.",
            ]
        )
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


def json_ready(value):
    if isinstance(value, Counter):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, defaultdict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json_sidecar(
    path: Path,
    analyses: list[dict],
    log_metrics: dict,
    args: argparse.Namespace,
    normalized_admission_block: dict | None = None,
) -> None:
    payload = {
        "metadata": {
            "source_log": str(args.source_log),
            "checkpoint": str(args.load_checkpoint),
            "eval_samples": args.eval_samples,
            "pair_drift_topk": args.pair_drift_topk,
            "pair_diffusion_topk": args.pair_diffusion_topk,
            "pair_drift_diffusion_candidates": args.pair_drift_diffusion_candidates,
        },
        "log_metrics": log_metrics,
        "samples": analyses,
    }
    if normalized_admission_block is not None:
        payload["p2_normalized_admission"] = normalized_admission_block
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.normalized_drift_admission != "none" and args.drift_only_json is None:
        raise SystemExit("--drift-only-json is required when --normalized-drift-admission is enabled")
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
                args.pair_drift_topk,
                args.pair_diffusion_topk,
                args.pair_drift_diffusion_candidates,
            )
        )
    normalized_admission_block = None
    if args.normalized_drift_admission == "p2_one_per_family":
        coverage_payload = {"samples": json_ready(analyses)}
        drift_only_payload = load_json(args.drift_only_json)
        normalized_admission_block = build_normalized_drift_admission_block(
            coverage_payload,
            drift_only_payload,
            policy=args.normalized_drift_admission,
            source=args.normalized_drift_admission_source,
        )
    write_report(args.report, analyses, log_metrics, args, normalized_admission_block)
    json_output = args.json_output or args.report.with_suffix(".json")
    write_json_sidecar(json_output, analyses, log_metrics, args, normalized_admission_block)
    if args.scorer_ready_json is not None:
        scorer_ready_payload = build_scorer_ready_sidecar(
            trainer.env,
            eval_samples,
            candidate_rows,
            analyses,
            args,
            fingerprint_config,
            normalized_admission_block,
        )
        args.scorer_ready_json.write_text(json.dumps(json_ready(scorer_ready_payload), indent=2, sort_keys=True))
    full_count = sum(item["has_full"] for item in analyses)
    missing_counts = Counter(item["missing_side"] for item in analyses if not item["has_full"])
    print(f"wrote_report={args.report}")
    print(f"wrote_json={json_output}")
    if args.scorer_ready_json is not None:
        print(f"wrote_scorer_ready_json={args.scorer_ready_json}")
    print(f"full_oracle_present={full_count}/{len(analyses)}")
    print(
        "oracle_absent_missing_sides="
        + ",".join(f"{key}:{value}" for key, value in sorted(missing_counts.items()))
    )
    if normalized_admission_block is not None:
        print(
            "p2_normalized_admission_coverage="
            f"{normalized_admission_block['p2_normalized_admission_coverage']}/{len(analyses)}"
        )
        print(f"p2_pair_count={normalized_admission_block['p2_pair_count']}")


if __name__ == "__main__":
    main()
