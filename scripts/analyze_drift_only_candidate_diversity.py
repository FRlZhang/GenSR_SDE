#!/usr/bin/env python3
"""Drift-only candidate diversity smoke for expanded-pairing oracle misses.

This is a diagnostic-only helper. It reuses the validation probe setup and
checkpoint loading, but generates only drift-role candidates for the known
expanded-pairing oracle-absent samples. It does not run reranking, fingerprint
scoring, formal eval, retraining, or production candidate generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sde_validation_probe import (
    _sample_allowed_token,
    _sde_token_sets,
    build_env,
    build_modules,
    configure_gensr,
    encode_numeric,
    generate_eval_samples,
    load_checkpoint_from_path,
    normalize_sample,
    parse_temperature_list,
    template_tokens,
)
from symbolicregression.trainer_vae import Trainer


FAMILY_ORDER = [
    "linear drift",
    "sin drift",
    "nested-mul linear drift",
    "polynomial-like drift",
    "constant drift",
    "other / unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drift-diversity-json",
        type=Path,
        default=Path("expanded_pairing_oracle_absent_drift_diversity.json"),
    )
    parser.add_argument(
        "--expanded-coverage-json",
        type=Path,
        default=Path("candidate_coverage_pair_expanded.json"),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=Path("/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth"),
    )
    parser.add_argument(
        "--checkpoint-backup",
        type=Path,
        default=Path("checkpoints/gensr_sde_role_token_2000/role_token_2000.pth"),
    )
    parser.add_argument("--report", type=Path, default=Path("drift_only_candidate_diversity_smoke.md"))
    parser.add_argument("--json-output", type=Path, default=Path("drift_only_candidate_diversity_smoke.json"))
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=800)
    parser.add_argument("--active-paths", type=int, default=800)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--max-generated-len", type=int, default=40)
    parser.add_argument("--drift-max-len", type=int, default=16)
    parser.add_argument("--min-generated-len", type=int, default=8)
    parser.add_argument("--drift-beam-size", type=int, default=16)
    parser.add_argument("--drift-beam-candidates", type=int, default=32)
    parser.add_argument("--drift-sample-candidates", type=int, default=48)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-temperatures", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--sample-top-k", type=int, default=8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
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


def make_probe_args(args: argparse.Namespace) -> argparse.Namespace:
    probe_args = copy.copy(args)
    probe_args.constrained_beam_size = 0
    probe_args.rerank_candidates = 0
    probe_args.rerank_topk_from_beam = 0
    probe_args.rerank_score = "constant_grid_rolewise_no_multi_u0"
    probe_args.rerank_component_weights = "1.0,2.0,1.0"
    probe_args.rerank_constant_values = "0.25,0.5,1.0,2.0,4.0"
    probe_args.rerank_tie_epsilon = 0.0
    probe_args.rerank_tie_break = "none"
    probe_args.rerank_debug_topk = 0
    probe_args.rerank_residual_debug_json = None
    probe_args.sample_candidates = 0
    probe_args.pair_drift_diffusion_candidates = 0
    probe_args.pair_drift_topk = 0
    probe_args.pair_diffusion_topk = 0
    return probe_args


def ensure_checkpoint(args: argparse.Namespace) -> None:
    checkpoint = args.load_checkpoint.expanduser()
    if checkpoint.is_file():
        return
    backup = args.checkpoint_backup.expanduser()
    if not backup.is_absolute():
        backup = REPO_ROOT / backup
    if not backup.is_file():
        raise FileNotFoundError(f"missing checkpoint {checkpoint} and backup {backup}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint.symlink_to(backup)
    except FileExistsError:
        if not checkpoint.is_file():
            raise
    print(f"restored_checkpoint_symlink={checkpoint} -> {backup}")


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def token_text(tokens: list[str] | None) -> str:
    return " ".join(tokens or [])


def drift_family(tokens: list[str]) -> str:
    templ = template_tokens(tokens)
    if "pow2" in templ or "pow3" in templ or "pow" in templ:
        return "polynomial-like drift"
    if "sin" in templ:
        return "sin drift"
    if templ == ["mul", "x_0"]:
        return "linear drift"
    if templ == ["mul"]:
        return "constant drift"
    if len(templ) >= 3 and templ[:2] == ["mul", "mul"] and "x_0" in templ:
        return "nested-mul linear drift"
    return "other / unknown"


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        cur = [i]
        for j, right_token in enumerate(right, start=1):
            cost = 0 if left_token == right_token else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def collapse_labels(truth: list[str], candidate: list[str]) -> list[str]:
    truth_template = template_tokens(truth)
    candidate_template = template_tokens(candidate)
    labels = []
    if truth_template == ["mul", "x_0"] and candidate_template == ["mul"]:
        labels.append("constant-heavy template")
    elif "x_0" in truth_template and "x_0" not in candidate_template:
        labels.append("missing state dependence")
    if candidate_template.count("mul") > truth_template.count("mul"):
        labels.append("over-nested template")
    if (
        "sin" in truth_template
        and "sin" in candidate_template
        and candidate_template.count("mul") > truth_template.count("mul")
    ):
        labels.append("sin over-nesting")
    if not labels:
        labels.append("other")
    return labels


def _drift_next_allowed(phase: str, holes: int, token_sets: dict[str, set[int] | int], eos_id: int) -> set[int]:
    if phase == "need_drift_role":
        return {int(token_sets["drift"])}
    if phase == "drift_expr":
        return set(token_sets["unary"]) | set(token_sets["binary"]) | set(token_sets["terminal"])
    if phase == "complete":
        return {eos_id}
    return set()


def _advance_drift_state(
    token_id: int,
    phase: str,
    holes: int,
    token_sets: dict[str, set[int] | int],
    eos_id: int,
) -> tuple[str, int] | None:
    if phase == "need_drift_role":
        if token_id != int(token_sets["drift"]):
            return None
        return "drift_expr", 1
    if phase == "drift_expr":
        if holes <= 0:
            return None
        if token_id in token_sets["terminal"]:
            holes -= 1
        elif token_id in token_sets["unary"]:
            pass
        elif token_id in token_sets["binary"]:
            holes += 1
        else:
            return None
        if holes == 0:
            return "complete", 0
        return "drift_expr", holes
    if phase == "complete":
        if token_id != eos_id:
            return None
        return "ended", 0
    return None


def words_from_ids(env, token_ids: list[int]) -> list[str]:
    eos_id = env.equation_word2id["<EOS>"]
    pad_id = env.equation_word2id["<PAD>"]
    words = []
    for token_id in token_ids:
        if token_id == pad_id:
            break
        if token_id == eos_id:
            continue
        words.append(env.equation_id2word[int(token_id)])
    return words


def drift_tokens_from_words(words: list[str]) -> list[str] | None:
    if "<DRIFT>" not in words:
        return None
    idx = words.index("<DRIFT>")
    drift = words[idx + 1 :]
    return drift or None


def make_candidate(
    env,
    token_ids: list[int],
    score: float,
    source: str,
    source_rank: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    words = words_from_ids(env, token_ids)
    drift_tokens = drift_tokens_from_words(words)
    valid = drift_tokens is not None
    return {
        "source": source,
        "source_rank": source_rank,
        "score": score,
        "normalized_score": score / (max(1, len(token_ids)) ** 0.7),
        "words": words,
        "drift_tokens": drift_tokens or [],
        "valid": valid,
        **(metadata or {}),
    }


def generate_drift_only_beam_candidates(
    decoder,
    src_enc,
    env,
    max_len: int,
    beam_size: int,
    n_candidates: int,
    length_penalty: float = 0.7,
) -> list[list[dict[str, Any]]]:
    token_sets = _sde_token_sets(env)
    eos_id = decoder.eos_index
    device = src_enc.device
    all_rows = []
    for sample_idx in range(src_enc.size(0)):
        one_src = src_enc[sample_idx : sample_idx + 1]
        beams = [
            {
                "tokens": [eos_id],
                "score": 0.0,
                "phase": "need_drift_role",
                "holes": 0,
                "ended": False,
            }
        ]
        complete = []
        for _ in range(1, max_len):
            expanded = []
            for beam in beams:
                if beam["ended"]:
                    complete.append(beam)
                    continue
                prefix = torch.tensor(beam["tokens"], dtype=torch.long, device=device).view(-1, 1)
                lengths = torch.tensor([len(beam["tokens"])], dtype=torch.long, device=device)
                tensor = decoder(
                    "fwd",
                    x=prefix,
                    lengths=lengths,
                    causal=True,
                    src_enc=one_src,
                    src_len=None,
                )
                logits = decoder.lm_head(tensor.data[-1, :, :].to(decoder.dtype))
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)[0]
                allowed = _drift_next_allowed(beam["phase"], beam["holes"], token_sets, eos_id)
                if not allowed:
                    continue
                candidate_ids = torch.tensor(sorted(allowed), dtype=torch.long, device=device)
                candidate_scores = log_probs.index_select(0, candidate_ids)
                k = min(beam_size, candidate_scores.numel())
                values, indices = torch.topk(candidate_scores, k)
                for value, rel_idx in zip(values.tolist(), indices.tolist()):
                    token_id = int(candidate_ids[int(rel_idx)].item())
                    next_state = _advance_drift_state(
                        token_id,
                        beam["phase"],
                        beam["holes"],
                        token_sets,
                        eos_id,
                    )
                    if next_state is None:
                        continue
                    next_phase, next_holes = next_state
                    expanded.append(
                        {
                            "tokens": beam["tokens"] + [token_id],
                            "score": beam["score"] + float(value),
                            "phase": next_phase,
                            "holes": next_holes,
                            "ended": next_phase == "ended",
                        }
                    )
            if not expanded:
                break
            complete.extend([beam for beam in expanded if beam["ended"]])
            live = [beam for beam in expanded if not beam["ended"]]
            if not live:
                break
            beams = sorted(
                live,
                key=lambda item: item["score"] / (len(item["tokens"]) ** length_penalty),
                reverse=True,
            )[:beam_size]
        candidates = complete or beams
        ranked = sorted(
            candidates,
            key=lambda item: item["score"] / (len(item["tokens"]) ** length_penalty),
            reverse=True,
        )[: max(1, n_candidates)]
        rows = []
        seen = set()
        for candidate in ranked:
            token_ids = list(candidate["tokens"])
            if not token_ids or token_ids[-1] != eos_id:
                token_ids.append(eos_id)
            key = tuple(token_ids)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                make_candidate(
                    env,
                    token_ids,
                    float(candidate["score"]),
                    "beam",
                    len(rows) + 1,
                    {"ended": bool(candidate.get("ended"))},
                )
            )
        all_rows.append(rows)
    return all_rows


def generate_drift_only_sample_candidates(
    decoder,
    src_enc,
    env,
    max_len: int,
    n_candidates: int,
    temperatures: list[float],
    top_k: int,
    top_p: float,
) -> list[list[dict[str, Any]]]:
    token_sets = _sde_token_sets(env)
    eos_id = decoder.eos_index
    device = src_enc.device
    all_rows = []
    for sample_idx in range(src_enc.size(0)):
        one_src = src_enc[sample_idx : sample_idx + 1]
        rows = []
        for candidate_idx in range(n_candidates):
            temperature = temperatures[candidate_idx % len(temperatures)]
            token_ids = [eos_id]
            score = 0.0
            phase = "need_drift_role"
            holes = 0
            ended = False
            for _ in range(1, max_len):
                prefix = torch.tensor(token_ids, dtype=torch.long, device=device).view(-1, 1)
                lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
                tensor = decoder(
                    "fwd",
                    x=prefix,
                    lengths=lengths,
                    causal=True,
                    src_enc=one_src,
                    src_len=None,
                )
                logits = decoder.lm_head(tensor.data[-1, :, :].to(decoder.dtype))
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)[0]
                allowed = _drift_next_allowed(phase, holes, token_sets, eos_id)
                if not allowed:
                    break
                token_id, token_score = _sample_allowed_token(
                    log_probs,
                    allowed,
                    temperature,
                    top_k,
                    top_p,
                )
                next_state = _advance_drift_state(token_id, phase, holes, token_sets, eos_id)
                if next_state is None:
                    break
                phase, holes = next_state
                token_ids.append(token_id)
                score += token_score
                if phase == "ended":
                    ended = True
                    break
            rows.append(
                make_candidate(
                    env,
                    token_ids,
                    score,
                    "sampling",
                    len(rows) + 1,
                    {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "ended": ended,
                    },
                )
            )
        all_rows.append(rows)
    return all_rows


def drift_prior_source_encoding(trainer: Trainer, samples: dict[str, Any]) -> torch.Tensor:
    env = trainer.env
    vae_model = trainer.modules["cvae"]
    embedder_f = trainer.modules["data_encoder"]
    embedder_e = trainer.modules["token_embed"]
    feature_fusion = trainer.modules["feature_fusion"]
    for module in [vae_model, embedder_f, embedder_e, feature_fusion]:
        module.eval()
    with torch.no_grad():
        x1, len1 = embedder_f(encode_numeric(samples))
        drift_x, drift_len = env.batch_equations(
            env.word_to_idx(samples["drift_tree_encoded"], float_input=False)
        )
        drift_x_e = embedder_e(drift_x.transpose(0, 1)).transpose(0, 1)
        prior_mu, prior_logvar, *_ = vae_model(
            x1,
            drift_x_e,
            len1,
            drift_len,
            mode="train",
        )
        return feature_fusion(prior_mu, prior_logvar)


def generate_rows_for_targets(
    trainer: Trainer,
    eval_samples: list[dict],
    target_indices: list[int],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    temperatures = parse_temperature_list(args.sample_temperatures, args.sample_temperature)
    decoder = trainer.modules.get("drift_decoder", trainer.modules["seq_decoder"])
    decoder.eval()
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for start in range(0, len(target_indices), args.batch_size):
        batch_indices = target_indices[start : start + args.batch_size]
        chunk = [eval_samples[idx] for idx in batch_indices]
        normalized = [normalize_sample(sample, trainer.env) for sample in chunk]
        batch = {
            "x_to_fit": [sample["x_to_fit"] for sample in normalized],
            "y_to_fit": [sample["y_to_fit"] for sample in normalized],
            "tree_encoded": [sample["tree_encoded"] for sample in normalized],
            "drift_tree_encoded": [sample["drift_tree_encoded"] for sample in normalized],
            "diffusion_tree_encoded": [sample["diffusion_tree_encoded"] for sample in normalized],
            "tree": [sample["tree"] for sample in chunk],
        }
        with torch.no_grad():
            src_enc = drift_prior_source_encoding(trainer, batch)
            beam_rows = generate_drift_only_beam_candidates(
                decoder,
                src_enc,
                trainer.env,
                args.drift_max_len,
                args.drift_beam_size,
                args.drift_beam_candidates,
            )
            sample_rows = generate_drift_only_sample_candidates(
                decoder,
                src_enc,
                trainer.env,
                args.drift_max_len,
                args.drift_sample_candidates,
                temperatures,
                args.sample_top_k,
                args.sample_top_p,
            )
        for rel_idx, sample_idx in enumerate(batch_indices):
            rows_by_index[sample_idx] = beam_rows[rel_idx] + sample_rows[rel_idx]
    return rows_by_index


def nearest_candidate(rows: list[dict[str, Any]], truth_drift: list[str]) -> dict[str, Any] | None:
    valid_rows = [row for row in rows if row.get("valid")]
    if not valid_rows:
        return None
    truth_template = template_tokens(truth_drift)
    ranked = []
    for row in valid_rows:
        distance = edit_distance(template_tokens(row["drift_tokens"]), truth_template)
        ranked.append(
            (
                distance,
                -float(row.get("normalized_score", float("-inf"))),
                row["source"],
                row["source_rank"],
                row,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    distance, _neg_score, _source, _rank, row = ranked[0]
    result = dict(row)
    result["distance"] = distance
    result["right_family"] = drift_family(row["drift_tokens"]) == drift_family(truth_drift)
    result["collapse_labels"] = collapse_labels(truth_drift, row["drift_tokens"])
    return result


def source_exact_status(rows: list[dict[str, Any]], truth_drift: list[str]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    truth_key = tuple(template_tokens(truth_drift))
    exact_rows = [
        row
        for row in rows
        if row.get("valid") and tuple(template_tokens(row["drift_tokens"])) == truth_key
    ]
    return exact_rows, {
        "beam": any(row["source"] == "beam" for row in exact_rows),
        "sampling": any(row["source"] == "sampling" for row in exact_rows),
    }


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source",
        "source_rank",
        "score",
        "normalized_score",
        "drift_tokens",
        "temperature",
        "top_k",
        "top_p",
        "ended",
        "valid",
    ]
    return {key: row[key] for key in keys if key in row}


def analyze_sample(
    target: dict[str, Any],
    coverage: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_drift = target["truth_drift_tokens"]
    exact_rows, exact_sources = source_exact_status(rows, truth_drift)
    nearest = nearest_candidate(rows, truth_drift)
    valid_rows = [row for row in rows if row.get("valid")]
    beam_rows = [row for row in valid_rows if row["source"] == "beam"]
    sampling_rows = [row for row in valid_rows if row["source"] == "sampling"]
    unique_all = {tuple(template_tokens(row["drift_tokens"])) for row in valid_rows}
    unique_beam = {tuple(template_tokens(row["drift_tokens"])) for row in beam_rows}
    unique_sampling = {tuple(template_tokens(row["drift_tokens"])) for row in sampling_rows}
    duplicate_counts = {
        "beam": len(beam_rows) - len(unique_beam),
        "sampling": len(sampling_rows) - len(unique_sampling),
    }
    exact_rank = None
    if exact_rows:
        exact_rank = min(row["source_rank"] for row in exact_rows)
    return {
        "sample_index": target["sample_index"],
        "truth_full_expression": target["truth_full_expression"],
        "truth_drift_tokens": truth_drift,
        "truth_diffusion_tokens": target["truth_diffusion_tokens"],
        "drift_family": target["drift_family"],
        "exact_diffusion_present_previous_coverage": bool(coverage.get("has_diffusion")),
        "current_nearest_logged_drift": target.get("best_available_drift_candidate"),
        "exact_truth_drift_appears": bool(exact_rows),
        "exact_truth_drift_rank": exact_rank,
        "exact_truth_drift_sources": exact_sources,
        "exact_truth_drift_occurrences": [compact_candidate(row) for row in exact_rows[:10]],
        "nearest_drift_only_candidate": None if nearest is None else compact_candidate(nearest)
        | {
            "distance": nearest["distance"],
            "right_family": nearest["right_family"],
            "collapse_labels": nearest["collapse_labels"],
        },
        "candidate_counts": {
            "total_valid": len(valid_rows),
            "unique_drift_candidates": len(unique_all),
            "beam_valid": len(beam_rows),
            "unique_beam_drift_candidates": len(unique_beam),
            "sampling_valid": len(sampling_rows),
            "unique_sampling_drift_candidates": len(unique_sampling),
            "duplicates_per_source": duplicate_counts,
            "invalid_constrained_candidates": len(rows) - len(valid_rows),
            "grammar_rejection_count": None,
        },
        "beam_candidates": [compact_candidate(row) for row in beam_rows[:20]],
        "sampling_candidates": [compact_candidate(row) for row in sampling_rows[:20]],
    }


def source_bucket(sample: dict[str, Any]) -> str:
    sources = sample["exact_truth_drift_sources"]
    if sources["beam"] and sources["sampling"]:
        return "both"
    if sources["beam"]:
        return "beam"
    if sources["sampling"]:
        return "sampling"
    return "neither"


def choose_decision(summary: dict[str, Any]) -> dict[str, str]:
    if summary["exact_truth_drift_found"] == 0:
        return {
            "choice": "C",
            "label": "Exact drifts remain absent from drift-only beam and sampling",
            "recommended_next_action": (
                "Run a model/grammar-side drift generation diagnostic or a small "
                "template-family seeding probe; do not run formal eval."
            ),
        }
    by_source = summary["exact_truth_drift_found_by_source"]
    if by_source["sampling"] > by_source["beam"]:
        return {
            "choice": "B",
            "label": "Exact drifts appear mainly in sampling",
            "recommended_next_action": (
                "Run a future drift-sampling diversity smoke; do not run formal eval."
            ),
        }
    return {
        "choice": "A",
        "label": "Exact drifts appear in drift-only tails",
        "recommended_next_action": (
            "Run a future candidate-coverage-only top-k / drift-only admission "
            "diagnostic; do not run formal eval."
        ),
    }


def build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    exact_found = [sample for sample in samples if sample["exact_truth_drift_appears"]]
    exact_missing = [sample for sample in samples if not sample["exact_truth_drift_appears"]]
    by_bucket = Counter(source_bucket(sample) for sample in samples)
    by_source = {
        "beam": sum(sample["exact_truth_drift_sources"]["beam"] for sample in samples),
        "sampling": sum(sample["exact_truth_drift_sources"]["sampling"] for sample in samples),
        "both": by_bucket["both"],
        "neither": by_bucket["neither"],
    }
    recovered_by_family = Counter(sample["drift_family"] for sample in exact_found)
    missing_by_family = Counter(sample["drift_family"] for sample in exact_missing)
    nearest_right_family = sum(
        bool((sample.get("nearest_drift_only_candidate") or {}).get("right_family"))
        for sample in samples
    )
    collapse_counts = Counter()
    for sample in samples:
        nearest = sample.get("nearest_drift_only_candidate") or {}
        for label in nearest.get("collapse_labels", ["other"]):
            collapse_counts[label] += 1
    unique_beam_total = sum(
        sample["candidate_counts"]["unique_beam_drift_candidates"] for sample in samples
    )
    unique_sampling_total = sum(
        sample["candidate_counts"]["unique_sampling_drift_candidates"] for sample in samples
    )
    summary = {
        "target_samples_analyzed": len(samples),
        "exact_truth_drift_found": len(exact_found),
        "exact_truth_drift_missing": len(exact_missing),
        "exact_truth_drift_found_by_source": by_source,
        "recovered_by_family": {family: recovered_by_family.get(family, 0) for family in FAMILY_ORDER},
        "missing_by_family": {family: missing_by_family.get(family, 0) for family in FAMILY_ORDER},
        "exact_diffusion_present_previous_coverage": sum(
            sample["exact_diffusion_present_previous_coverage"] for sample in samples
        ),
        "nearest_drift_only_right_family": nearest_right_family,
        "collapse_pattern_counts": dict(collapse_counts),
        "unique_beam_drift_candidates_sum": unique_beam_total,
        "unique_sampling_drift_candidates_sum": unique_sampling_total,
        "grammar_rejection_counts_available": False,
        "token_logits_or_entropy_available": False,
    }
    summary["decision"] = choose_decision(summary)
    return summary


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    decision = summary["decision"]
    lines = [
        "# Drift-Only Candidate Diversity Smoke",
        "",
        "Date: 2026-06-23",
        "",
        "Scope: diagnostic-only drift-role beam/sampling smoke for the 17 expanded-pairing oracle-absent samples. It loads the current checkpoint and generates drift-only candidates, but does not run reranking, fingerprint scoring, formal eval, 64-sample eval, grids, retraining, scorer changes, or production candidate-generation changes.",
        "",
        "## Executive Summary",
        "",
        f"- Target samples analyzed: `{summary['target_samples_analyzed']}`.",
        f"- Exact truth drift found in drift-only candidates: `{summary['exact_truth_drift_found']}/{summary['target_samples_analyzed']}`.",
        f"- Exact truth drift missing from drift-only beam+sampling: `{summary['exact_truth_drift_missing']}/{summary['target_samples_analyzed']}`.",
        f"- Exact diffusion was already present in previous expanded coverage for `{summary['exact_diffusion_present_previous_coverage']}/{summary['target_samples_analyzed']}` samples.",
        f"- Nearest drift-only candidate has the right family in `{summary['nearest_drift_only_right_family']}/{summary['target_samples_analyzed']}` samples.",
        f"- Decision: `{decision['choice']}` - {decision['label']}.",
        "",
        "## Exact Drift Recovery By Source",
        "",
        "| Source bucket | Count |",
        "| --- | ---: |",
    ]
    by_source = summary["exact_truth_drift_found_by_source"]
    for key in ("beam", "sampling", "both", "neither"):
        lines.append(f"| {key} | {by_source[key]} |")
    lines.extend(
        [
            "",
            "## Drift-Family Recovery",
            "",
            "| Family | Recovered | Missing |",
            "| --- | ---: | ---: |",
        ]
    )
    for family in FAMILY_ORDER:
        lines.append(
            f"| {family} | {summary['recovered_by_family'][family]} | "
            f"{summary['missing_by_family'][family]} |"
        )
    lines.extend(
        [
            "",
            "## Diversity And Collapse",
            "",
            f"- Unique beam drift candidates summed over samples: `{summary['unique_beam_drift_candidates_sum']}`.",
            f"- Unique sampling drift candidates summed over samples: `{summary['unique_sampling_drift_candidates_sum']}`.",
            "- Beam contributes more new drift diversity than sampling if its summed unique count is higher; sampling remains under-diverse if it mostly duplicates the same few templates and does not recover exact drifts.",
            "- Grammar rejection counts and token entropy/logits are unavailable without more invasive instrumentation. Constrained decoding emitted parse-valid drift-role candidates, so raw grammar rejection versus low-probability absence cannot be separated here.",
            "",
            "Collapse labels on nearest drift-only candidates:",
            "",
            "| Collapse label | Count |",
            "| --- | ---: |",
        ]
    )
    for label, count in sorted(summary["collapse_pattern_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Per-Sample Diagnostics",
            "",
            "| Sample | Family | Truth drift | Exact found | Source bucket | Exact rank | Nearest drift-only candidate | Nearest right family | Collapse labels | Unique beam | Unique sampling |",
            "| ---: | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for sample in payload["samples"]:
        nearest = sample.get("nearest_drift_only_candidate") or {}
        nearest_text = token_text(nearest.get("drift_tokens")) if nearest else "-"
        collapse = ",".join(nearest.get("collapse_labels", [])) if nearest else "-"
        lines.append(
            f"| {sample['sample_index']} | {sample['drift_family']} | "
            f"`{token_text(sample['truth_drift_tokens'])}` | "
            f"{sample['exact_truth_drift_appears']} | {source_bucket(sample)} | "
            f"{sample['exact_truth_drift_rank'] if sample['exact_truth_drift_rank'] is not None else '-'} | "
            f"`{nearest_text}` | {nearest.get('right_family', '-')} | {collapse} | "
            f"{sample['candidate_counts']['unique_beam_drift_candidates']} | "
            f"{sample['candidate_counts']['unique_sampling_drift_candidates']} |"
        )
    lines.extend(
        [
            "",
            "## Required Conclusions",
            "",
            "1. The report analyzes only the 17 expanded-pairing oracle-absent samples.",
            f"2. Exact truth drift found count: `{summary['exact_truth_drift_found']}/{summary['target_samples_analyzed']}`.",
            "3. Source split is shown above for beam, sampling, both, and neither.",
            "4. Family recovery/missing split is shown above for linear, sin, nested-mul linear, polynomial-like, constant, and other/unknown.",
            "5. Whether exact drifts are below current top-k or absent entirely is decided from the drift-only tail result: recovered cases are below/admission misses; missing cases remain absent from this drift-only beam+sampling smoke.",
            "6. Beam versus sampling diversity is summarized by unique drift candidate counts and exact recovery source.",
            "7. Sampling under-diversity is inferred from emitted candidates only because logits/entropy are unavailable.",
            "8. Grammar-constrained decoding collapse is summarized by nearest-candidate collapse labels; raw rejection counts are unavailable.",
            f"9. Next intervention: {decision['recommended_next_action']}",
            "",
            "## Missing Fields",
            "",
            "- `token_logits_or_raw_sampling_entropy`: unavailable",
            "- `raw_grammar_rejection_counts`: unavailable",
            "- `invalid_unconstrained_candidates`: unavailable",
            "",
            "No formal eval is recommended from this smoke.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_checkpoint(args)
    drift_diversity = load_json(args.drift_diversity_json)
    expanded_coverage = load_json(args.expanded_coverage_json)
    target_indices = drift_diversity["summary"]["oracle_absent_sample_indices"]
    target_by_index = {sample["sample_index"]: sample for sample in drift_diversity["samples"]}
    coverage_by_index = {sample["sample_index"]: sample for sample in expanded_coverage["samples"]}

    trainer = setup_trainer(args)
    eval_samples = generate_eval_samples(make_probe_args(args), trainer.env, trainer.params)
    rows_by_index = generate_rows_for_targets(trainer, eval_samples, target_indices, args)

    sample_reports = []
    for sample_idx in target_indices:
        sample_reports.append(
            analyze_sample(
                target_by_index[sample_idx],
                coverage_by_index[sample_idx],
                rows_by_index[sample_idx],
            )
        )
    summary = build_summary(sample_reports)
    payload = {
        "metadata": {
            "checkpoint": str(args.load_checkpoint),
            "target_indices": target_indices,
            "drift_max_len": args.drift_max_len,
            "drift_beam_size": args.drift_beam_size,
            "drift_beam_candidates": args.drift_beam_candidates,
            "drift_sample_candidates": args.drift_sample_candidates,
            "sample_temperatures": parse_temperature_list(
                args.sample_temperatures,
                args.sample_temperature,
            ),
            "sample_top_k": args.sample_top_k,
            "sample_top_p": args.sample_top_p,
        },
        "summary": summary,
        "samples": sample_reports,
    }
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.report, payload)
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"target_samples={summary['target_samples_analyzed']}")
    print(f"exact_truth_drift_found={summary['exact_truth_drift_found']}")
    print(
        "exact_truth_drift_found_by_source="
        + ",".join(
            f"{key}:{value}"
            for key, value in sorted(summary["exact_truth_drift_found_by_source"].items())
        )
    )
    print(f"decision={summary['decision']['choice']}")


if __name__ == "__main__":
    main()
