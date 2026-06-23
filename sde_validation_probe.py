#!/usr/bin/env python3
"""Train/evaluate a small GenSR-SDE validation probe.

This script keeps the first validation loop deliberately narrow:

1. load an SDE training pickle,
2. train the CVAE for a small number of steps using real SDE samples,
3. generate held-out SDE samples from a different seed,
4. report prior-only teacher-forced token loss/top-k and greedy sequence hits.

It does not save model checkpoints by default.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.cuda.set_device = lambda *args, **kwargs: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda *args: None
    if not hasattr(torch._C, "_cuda_setDevice"):
        torch._C._cuda_setDevice = lambda *args, **kwargs: None

from parsers import get_parser
from sde_dataset_generator import build_sample, encode_sde
from sde_fingerprint import FingerprintConfig
from simulator_sde import SDESystem, sample_sde_system, solve_fingerprint
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules, check_model_params
from symbolicregression.trainer_vae import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("data/sde_train_dataset.pkl"))
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--active-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--print-freq", type=int, default=25)
    parser.add_argument("--max-generated-len", type=int, default=40)
    parser.add_argument("--min-generated-len", type=int, default=8)
    parser.add_argument("--n-enc-layers", type=int, default=None)
    parser.add_argument("--n-dec-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--split-sde-loss", action="store_true")
    parser.add_argument("--constrained-beam-size", type=int, default=0)
    parser.add_argument("--rerank-candidates", type=int, default=0)
    parser.add_argument("--rerank-topk-from-beam", type=int, default=0)
    parser.add_argument(
        "--rerank-score",
        choices=(
            "full_fingerprint",
            "short_moment",
            "componentwise",
            "constant_grid_full",
            "constant_grid_componentwise",
            "constant_grid_componentwise_no_multi_u0",
            "constant_grid_rolewise_no_multi_u0",
            "constant_grid_active_weak",
            "constant_grid_active_only",
            "constant_grid_moments_downweighted",
        ),
        default="full_fingerprint",
    )
    parser.add_argument("--rerank-component-weights", type=str, default="1.0,2.0,1.0")
    parser.add_argument("--rerank-constant-values", type=str, default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--rerank-tie-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--rerank-tie-break",
        choices=("none", "model_score", "active_distance", "state_dependent_drift"),
        default="none",
    )
    parser.add_argument("--rerank-debug-topk", type=int, default=0)
    parser.add_argument("--rerank-residual-debug-json", type=Path, default=None)
    parser.add_argument("--sample-candidates", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-temperatures", type=str, default="")
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-top-p", type=float, default=1.0)
    parser.add_argument("--pair-drift-diffusion-candidates", type=int, default=0)
    parser.add_argument("--pair-drift-topk", type=int, default=0)
    parser.add_argument("--pair-diffusion-topk", type=int, default=0)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-include-optimizer", action="store_true")
    return parser.parse_args()


def configure_gensr(args: argparse.Namespace):
    params = get_parser().parse_args([])
    params.min_input_dimension = 1
    params.max_input_dimension = 1
    params.min_output_dimension = 1
    params.max_output_dimension = 1
    params.batch_size = args.batch_size
    params.batch_size_eval = args.batch_size
    params.n_steps_per_epoch = args.train_steps
    params.max_epoch = 1
    params.num_workers = 0
    params.cpu = True
    params.device = torch.device("cpu")
    params.multi_gpu = False
    params.wandb_disabled = True
    params.save_periodic = 0
    params.print_freq = args.print_freq
    params.max_generated_output_len = args.max_generated_len
    params.max_target_len = max(params.max_target_len, args.max_generated_len)
    if args.n_enc_layers is not None:
        params.n_enc_layers = args.n_enc_layers
    if args.n_dec_layers is not None:
        params.n_dec_layers = args.n_dec_layers
    if args.n_heads is not None:
        params.n_enc_heads = args.n_heads
        params.n_dec_heads = args.n_heads
    if args.model_dim is not None:
        params.d_model = args.model_dim
        params.d_input = args.model_dim
        params.enc_emb_dim = args.model_dim
        params.dec_emb_dim = args.model_dim
        params.latent_dim = args.model_dim
    params.is_master = True
    params.global_rank = 0
    params.local_rank = 0
    params.master_port = -1
    params.n_gpu_per_node = 1
    params.world_size = 1
    params.multi_node = False
    params.is_slurm_job = False
    params.dump_path = "/private/tmp/gensr_sde_validation_probe"
    params.split_sde_loss = args.split_sde_loss
    check_model_params(params)
    return params


def ids_to_words_if_needed(encoded, env):
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    elif isinstance(encoded, np.ndarray):
        encoded = encoded.tolist()
    if len(encoded) == 0:
        return encoded
    first = encoded[0]
    if isinstance(first, torch.Tensor):
        first = int(first.item())
    if isinstance(first, (int, np.integer)):
        return [env.equation_id2word[int(token_id)] for token_id in encoded]
    return encoded


def normalize_sample(sample: dict, env) -> dict:
    sample = copy.deepcopy(sample)
    reencoded = None
    if "tree" in sample and " | " in sample["tree"]:
        drift, diffusion = sample["tree"].split(" | ", 1)
        reencoded = encode_sde(SDESystem(drift, diffusion), env, env.params)
    if reencoded is not None:
        sample.update(reencoded)
    else:
        sample["tree_encoded"] = ids_to_words_if_needed(sample["tree_encoded"], env)
        sample["skeleton_tree_encoded"] = ids_to_words_if_needed(
            sample["skeleton_tree_encoded"], env
        )
    if "drift_tree_encoded" in sample and "diffusion_tree_encoded" in sample:
        sample["drift_tree_encoded"] = ids_to_words_if_needed(
            sample["drift_tree_encoded"], env
        )
        sample["diffusion_tree_encoded"] = ids_to_words_if_needed(
            sample["diffusion_tree_encoded"], env
        )
    elif "SPECIAL" in sample["tree_encoded"]:
        sep = sample["tree_encoded"].index("SPECIAL")
        sample["drift_tree_encoded"] = ["<DRIFT>"] + sample["tree_encoded"][:sep]
        sample["diffusion_tree_encoded"] = ["<DIFFUSION>"] + sample["tree_encoded"][sep + 1 :]
        sample["tree_encoded"] = (
            sample["drift_tree_encoded"] + sample["diffusion_tree_encoded"]
        )
        sample["skeleton_tree_encoded"] = sample["tree_encoded"]

    if "drift_skeleton_tree_encoded" in sample and "diffusion_skeleton_tree_encoded" in sample:
        sample["drift_skeleton_tree_encoded"] = ids_to_words_if_needed(
            sample["drift_skeleton_tree_encoded"], env
        )
        sample["diffusion_skeleton_tree_encoded"] = ids_to_words_if_needed(
            sample["diffusion_skeleton_tree_encoded"], env
        )
    else:
        sample["drift_skeleton_tree_encoded"] = sample["drift_tree_encoded"]
        sample["diffusion_skeleton_tree_encoded"] = sample["diffusion_tree_encoded"]
    infos = sample.get("infos", {})
    sample["infos"] = {
        "n_input_points": int(infos.get("n_input_points", sample["x_to_fit"].shape[0])),
        "input_sequence_length": int(
            infos.get("input_sequence_length", sample["x_to_fit"].shape[0])
        ),
        "d_in": int(infos.get("d_in", 1)),
        "d_out": int(infos.get("d_out", 1)),
        "fingerprint_length": int(
            infos.get("fingerprint_length", sample["x_to_fit"].shape[0])
        ),
    }
    return sample


def install_sde_pool(env, train_pool: list[dict]) -> None:
    from symbolicregression.envs.environment import EnvDataset

    def sde_generate_sample(self):
        return normalize_sample(random.choice(train_pool), self.env)

    EnvDataset.generate_sample = sde_generate_sample
    env.sde_data_pool = train_pool


def sample_batch(samples: list[dict], env, batch_size: int, rng: random.Random) -> dict:
    batch = [normalize_sample(rng.choice(samples), env) for _ in range(batch_size)]
    return {
        "x_to_fit": [s["x_to_fit"] for s in batch],
        "y_to_fit": [s["y_to_fit"] for s in batch],
        "tree_encoded": [s["tree_encoded"] for s in batch],
        "drift_tree_encoded": [s["drift_tree_encoded"] for s in batch],
        "diffusion_tree_encoded": [s["diffusion_tree_encoded"] for s in batch],
        "tree": [s["tree"] for s in batch],
    }


def encode_numeric(samples: dict):
    x1 = []
    for seq_id in range(len(samples["x_to_fit"])):
        seq = []
        for seq_l in range(len(samples["x_to_fit"][seq_id])):
            seq.append(
                [
                    samples["x_to_fit"][seq_id][seq_l],
                    samples["y_to_fit"][seq_id][seq_l],
                ]
            )
        x1.append(seq)
    return x1


def generate_from_latent_min_len(decoder, src_enc, max_len: int, min_len: int):
    bs = src_enc.size(0)
    generated = src_enc.new(max_len, bs).long()
    generated.fill_(decoder.pad_index)
    generated[0].fill_(decoder.eos_index)

    positions = (
        torch.arange(max_len, device=src_enc.device)
        .long()
        .unsqueeze(1)
        .expand(max_len, bs)
    )

    cur_len = 1
    gen_len = src_enc.new(bs).long().fill_(1)
    unfinished_sents = src_enc.new(bs).long().fill_(1)
    decoder.cache = {"slen": 0}

    while cur_len < max_len:
        tensor = decoder.forward(
            "fwd",
            x=generated[:cur_len],
            lengths=gen_len,
            positions=positions[:cur_len],
            causal=True,
            src_enc=src_enc,
            src_len=None,
            use_cache=True,
        )
        scores = decoder.lm_head(tensor.data[-1, :, :].to(decoder.dtype))
        if cur_len < min_len:
            scores[:, decoder.eos_index] = -float("inf")
            scores[:, decoder.pad_index] = -float("inf")
        next_words = torch.topk(scores, 1)[1].squeeze(1)
        generated[cur_len] = next_words * unfinished_sents + decoder.pad_index * (
            1 - unfinished_sents
        )
        gen_len.add_(unfinished_sents)
        unfinished_sents.mul_(next_words.ne(decoder.eos_index).long())
        cur_len += 1
        if unfinished_sents.max() == 0:
            break

    if cur_len == max_len:
        generated[-1].masked_fill_(unfinished_sents.bool(), decoder.eos_index)
    return generated, gen_len


def _sde_token_sets(env):
    unary = {"abs", "sqrt", "sin"}
    binary = {"add", "sub", "mul"}
    terminal = {"x_0", "CONSTANT"}
    role = {"<DRIFT>", "<DIFFUSION>"}
    allowed_words = unary | binary | terminal | role
    word_to_id = env.equation_word2id
    return {
        "unary": {word_to_id[w] for w in unary if w in word_to_id},
        "binary": {word_to_id[w] for w in binary if w in word_to_id},
        "terminal": {word_to_id[w] for w in terminal if w in word_to_id},
        "drift": word_to_id["<DRIFT>"],
        "diffusion": word_to_id["<DIFFUSION>"],
        "allowed": {word_to_id[w] for w in allowed_words if w in word_to_id},
    }


def _prefix_next_allowed(phase: str, holes: int, token_sets: dict, eos_id: int) -> set[int]:
    if phase == "need_drift_role":
        return {token_sets["drift"]}
    if phase in {"drift_expr", "diffusion_expr"}:
        return token_sets["unary"] | token_sets["binary"] | token_sets["terminal"]
    if phase == "need_diffusion_role":
        return {token_sets["diffusion"]}
    if phase == "complete":
        return {eos_id}
    return set()


def _advance_sde_state(token_id: int, phase: str, holes: int, token_sets: dict, eos_id: int):
    if phase == "need_drift_role":
        if token_id != token_sets["drift"]:
            return None
        return "drift_expr", 1
    if phase in {"drift_expr", "diffusion_expr"}:
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
        if holes == 0 and phase == "drift_expr":
            return "need_diffusion_role", 0
        if holes == 0 and phase == "diffusion_expr":
            return "complete", 0
        return phase, holes
    if phase == "need_diffusion_role":
        if token_id != token_sets["diffusion"]:
            return None
        return "diffusion_expr", 1
    if phase == "complete":
        if token_id != eos_id:
            return None
        return "ended", 0
    return None


def generate_sde_constrained_beam(
    decoder,
    src_enc,
    env,
    max_len: int,
    beam_size: int,
    length_penalty: float = 0.7,
) -> torch.Tensor:
    candidate_rows = generate_sde_constrained_beam_candidates(
        decoder,
        src_enc,
        env,
        max_len,
        beam_size,
        1,
        length_penalty,
    )
    return torch.stack([rows[0]["tensor"] for rows in candidate_rows], dim=0)


def generate_sde_constrained_beam_candidates(
    decoder,
    src_enc,
    env,
    max_len: int,
    beam_size: int,
    n_candidates: int,
    length_penalty: float = 0.7,
) -> list[list[dict]]:
    token_sets = _sde_token_sets(env)
    eos_id = decoder.eos_index
    pad_id = decoder.pad_index
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
                prefix = torch.tensor(
                    beam["tokens"], dtype=torch.long, device=device
                ).view(-1, 1)
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
                allowed = _prefix_next_allowed(
                    beam["phase"], beam["holes"], token_sets, eos_id
                )
                if not allowed:
                    continue
                candidate_ids = torch.tensor(
                    sorted(allowed), dtype=torch.long, device=device
                )
                candidate_scores = log_probs.index_select(0, candidate_ids)
                k = min(beam_size, candidate_scores.numel())
                values, indices = torch.topk(candidate_scores, k)
                for value, rel_idx in zip(values.tolist(), indices.tolist()):
                    token_id = int(candidate_ids[rel_idx].item())
                    next_state = _advance_sde_state(
                        token_id, beam["phase"], beam["holes"], token_sets, eos_id
                    )
                    if next_state is None:
                        continue
                    next_phase, next_holes = next_state
                    next_tokens = beam["tokens"] + [token_id]
                    expanded.append(
                        {
                            "tokens": next_tokens,
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
            beams = heapq.nlargest(
                beam_size,
                live,
                key=lambda item: item["score"] / (len(item["tokens"]) ** length_penalty),
            )

        candidates = complete or beams
        ranked = heapq.nlargest(
            max(1, n_candidates),
            candidates,
            key=lambda item: item["score"] / (len(item["tokens"]) ** length_penalty),
        )
        rows = []
        seen = set()
        for candidate in ranked:
            tokens = candidate["tokens"]
            if tokens[-1] != eos_id:
                tokens = tokens + [eos_id]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                tokens[-1] = eos_id
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            row = torch.full((max_len,), pad_id, dtype=torch.long, device=device)
            row[: len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            rows.append(
                {
                    "tensor": row,
                    "tokens": tokens,
                    "model_score": candidate["score"],
                    "normalized_model_score": candidate["score"]
                    / (len(tokens) ** length_penalty),
                }
            )
        if not rows:
            row = torch.full((max_len,), pad_id, dtype=torch.long, device=device)
            rows.append(
                {
                    "tensor": row,
                    "tokens": [],
                    "model_score": -float("inf"),
                    "normalized_model_score": -float("inf"),
                }
            )
        all_rows.append(rows)

    return all_rows


def parse_temperature_list(sample_temperatures: str, default_temperature: float) -> list[float]:
    if not sample_temperatures.strip():
        return [default_temperature]
    temperatures = []
    for raw in sample_temperatures.split(","):
        raw = raw.strip()
        if not raw:
            continue
        temperatures.append(float(raw))
    return temperatures or [default_temperature]


def _candidate_row_from_tokens(
    tokens: list[int],
    score: float,
    max_len: int,
    pad_id: int,
    eos_id: int,
    device,
    source: str,
) -> dict:
    if not tokens or tokens[-1] != eos_id:
        tokens = tokens + [eos_id]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = eos_id
    row = torch.full((max_len,), pad_id, dtype=torch.long, device=device)
    row[: len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    return {
        "tensor": row,
        "tokens": tokens,
        "model_score": score,
        "normalized_model_score": score / (len(tokens) ** 0.7),
        "source": source,
    }


def _sample_allowed_token(
    log_probs: torch.Tensor,
    allowed: set[int],
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[int, float]:
    candidate_ids = torch.tensor(sorted(allowed), dtype=torch.long, device=log_probs.device)
    candidate_log_probs = log_probs.index_select(0, candidate_ids)
    temp = max(1e-6, float(temperature))
    sampling_logits = candidate_log_probs / temp

    if top_k > 0 and top_k < sampling_logits.numel():
        values, indices = torch.topk(sampling_logits, top_k)
        candidate_ids = candidate_ids.index_select(0, indices)
        candidate_log_probs = candidate_log_probs.index_select(0, indices)
        sampling_logits = values

    if 0.0 < top_p < 1.0 and sampling_logits.numel() > 1:
        sorted_logits, sorted_indices = torch.sort(sampling_logits, descending=True)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=0)
        keep = torch.cumsum(sorted_probs, dim=0) <= top_p
        keep[0] = True
        kept_indices = sorted_indices[keep]
        candidate_ids = candidate_ids.index_select(0, kept_indices)
        candidate_log_probs = candidate_log_probs.index_select(0, kept_indices)
        sampling_logits = sampling_logits.index_select(0, kept_indices)

    probs = torch.nn.functional.softmax(sampling_logits, dim=0)
    rel_idx = int(torch.multinomial(probs, 1).item())
    token_id = int(candidate_ids[rel_idx].item())
    return token_id, float(candidate_log_probs[rel_idx].item())


def generate_sde_constrained_sample_candidates(
    decoder,
    src_enc,
    env,
    max_len: int,
    n_candidates: int,
    temperatures: list[float],
    top_k: int,
    top_p: float,
) -> list[list[dict]]:
    token_sets = _sde_token_sets(env)
    eos_id = decoder.eos_index
    pad_id = decoder.pad_index
    device = src_enc.device
    all_rows = []

    for sample_idx in range(src_enc.size(0)):
        one_src = src_enc[sample_idx : sample_idx + 1]
        rows = []
        for candidate_idx in range(n_candidates):
            temperature = temperatures[candidate_idx % len(temperatures)]
            tokens = [eos_id]
            score = 0.0
            phase = "need_drift_role"
            holes = 0
            ended = False

            for _ in range(1, max_len):
                prefix = torch.tensor(tokens, dtype=torch.long, device=device).view(-1, 1)
                lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
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
                allowed = _prefix_next_allowed(phase, holes, token_sets, eos_id)
                if not allowed:
                    break
                token_id, token_score = _sample_allowed_token(
                    log_probs,
                    allowed,
                    temperature,
                    top_k,
                    top_p,
                )
                next_state = _advance_sde_state(token_id, phase, holes, token_sets, eos_id)
                if next_state is None:
                    break
                phase, holes = next_state
                tokens.append(token_id)
                score += token_score
                if phase == "ended":
                    ended = True
                    break

            rows.append(
                _candidate_row_from_tokens(
                    tokens,
                    score,
                    max_len,
                    pad_id,
                    eos_id,
                    device,
                    f"sample_t={temperature:g}",
                )
            )
        all_rows.append(rows)

    return all_rows


def merge_candidate_rows(candidate_groups: list[list[list[dict]] | None]) -> list[list[dict]] | None:
    present_groups = [group for group in candidate_groups if group is not None]
    if not present_groups:
        return None
    merged = []
    batch_size = len(present_groups[0])
    for sample_idx in range(batch_size):
        seen = set()
        rows = []
        for group in present_groups:
            for candidate in group[sample_idx]:
                key = tuple(candidate["tokens"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(candidate)
        merged.append(rows)
    return merged


def generate_paired_candidate_rows(
    env,
    candidate_rows: list[list[dict]] | None,
    max_len: int,
    n_pairs: int,
    drift_topk: int,
    diffusion_topk: int,
) -> list[list[dict]] | None:
    if candidate_rows is None or n_pairs <= 0:
        return None

    eos_id = env.equation_word2id["<EOS>"]
    pad_id = env.equation_word2id["<PAD>"]
    drift_id = env.equation_word2id["<DRIFT>"]
    diffusion_id = env.equation_word2id["<DIFFUSION>"]
    all_rows = []

    for rows in candidate_rows:
        if rows:
            device = rows[0]["tensor"].device
        else:
            device = torch.device("cpu")
        drift_spans = {}
        diffusion_spans = {}
        for candidate in rows:
            words = decode_generation(env, candidate["tensor"])
            split = split_sde_tokens(words)
            if split is None:
                continue
            drift_tokens, diffusion_tokens = split
            drift_key = tuple(drift_tokens)
            diffusion_key = tuple(diffusion_tokens)
            score = float(candidate["normalized_model_score"])
            if drift_key not in drift_spans or score > drift_spans[drift_key]:
                drift_spans[drift_key] = score
            if diffusion_key not in diffusion_spans or score > diffusion_spans[diffusion_key]:
                diffusion_spans[diffusion_key] = score

        ranked_drifts = sorted(
            drift_spans.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_diffusions = sorted(
            diffusion_spans.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if drift_topk > 0:
            ranked_drifts = ranked_drifts[:drift_topk]
        if diffusion_topk > 0:
            ranked_diffusions = ranked_diffusions[:diffusion_topk]

        pair_rows = []
        pair_ranked = []
        for drift_tokens, drift_score in ranked_drifts:
            for diffusion_tokens, diffusion_score in ranked_diffusions:
                pair_ranked.append((drift_score + diffusion_score, drift_tokens, diffusion_tokens))
        pair_ranked.sort(key=lambda item: item[0], reverse=True)
        for score, drift_tokens, diffusion_tokens in pair_ranked[:n_pairs]:
            try:
                drift_token_ids = [env.equation_word2id[token] for token in drift_tokens]
                diffusion_token_ids = [
                    env.equation_word2id[token] for token in diffusion_tokens
                ]
            except KeyError:
                continue
            tokens = [eos_id, drift_id] + drift_token_ids + [diffusion_id] + diffusion_token_ids
            pair_rows.append(
                _candidate_row_from_tokens(
                    tokens,
                    score,
                    max_len,
                    pad_id,
                    eos_id,
                    device,
                    "pair",
                )
            )
        all_rows.append(pair_rows)

    return all_rows


def prior_scores_for_batch(trainer: Trainer, samples: dict, min_generated_len: int):
    env = trainer.env
    params = trainer.params
    vae_model = trainer.modules["cvae"]
    drift_decoder = trainer.modules.get("drift_decoder", trainer.modules["seq_decoder"])
    diffusion_decoder = trainer.modules.get("diffusion_decoder", trainer.modules["seq_decoder"])
    embedder_f = trainer.modules["data_encoder"]
    embedder_e = trainer.modules["token_embed"]
    feature_fusion = trainer.modules["feature_fusion"]

    for module in set([
        vae_model,
        drift_decoder,
        diffusion_decoder,
        embedder_f,
        embedder_e,
        feature_fusion,
    ]):
        module.eval()

    with torch.no_grad():
        x1, len1 = embedder_f(encode_numeric(samples))
        drift_x, drift_len = env.batch_equations(
            env.word_to_idx(samples["drift_tree_encoded"], float_input=False)
        )
        diffusion_x, diffusion_len = env.batch_equations(
            env.word_to_idx(samples["diffusion_tree_encoded"], float_input=False)
        )
        x2_e = embedder_e(drift_x.transpose(0, 1)).transpose(0, 1)
        prior_mu, prior_logvar, _, _, _, _, _ = vae_model(
            x1, x2_e, len1, drift_len, mode="train"
        )
        mapped_src_enc_prior = feature_fusion(prior_mu, prior_logvar)

        def scores_for(decoder, x, lengths):
            alen = torch.arange(params.max_src_len, dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            decoded = decoder(
                "fwd",
                x=x,
                lengths=lengths,
                causal=True,
                src_enc=mapped_src_enc_prior,
                src_len=len1,
            )
            scores, loss = decoder(
                "predict",
                tensor=decoded,
                pred_mask=pred_mask,
                y=y,
                get_scores=True,
            )
            generations, gen_len = decoder.generate_from_latent(
                mapped_src_enc_prior,
                sample_temperature=None,
                max_len=trainer.params.max_generated_output_len,
            )
            constrained_generations, constrained_len = generate_from_latent_min_len(
                decoder,
                mapped_src_enc_prior,
                trainer.params.max_generated_output_len,
                min_generated_len,
            )
            return scores, y, loss, generations, gen_len, constrained_generations, constrained_len

        drift = scores_for(drift_decoder, drift_x, drift_len)
        diffusion = scores_for(diffusion_decoder, diffusion_x, diffusion_len)
    return (
        drift,
        diffusion,
    )


def prior_scores_for_joint_batch(
    trainer: Trainer,
    samples: dict,
    min_generated_len: int,
    constrained_beam_size: int = 0,
    rerank_candidates: int = 0,
    rerank_topk_from_beam: int = 0,
    sample_candidates: int = 0,
    sample_temperatures: list[float] | None = None,
    sample_top_k: int = 0,
    sample_top_p: float = 1.0,
    pair_drift_diffusion_candidates: int = 0,
    pair_drift_topk: int = 0,
    pair_diffusion_topk: int = 0,
):
    env = trainer.env
    params = trainer.params
    vae_model = trainer.modules["cvae"]
    decoder = trainer.modules["seq_decoder"]
    embedder_f = trainer.modules["data_encoder"]
    embedder_e = trainer.modules["token_embed"]
    feature_fusion = trainer.modules["feature_fusion"]

    for module in [vae_model, decoder, embedder_f, embedder_e, feature_fusion]:
        module.eval()

    with torch.no_grad():
        x1, len1 = embedder_f(encode_numeric(samples))
        x2, len2 = env.batch_equations(
            env.word_to_idx(samples["tree_encoded"], float_input=False)
        )
        x2_e = embedder_e(x2.transpose(0, 1)).transpose(0, 1)
        prior_mu, prior_logvar, _, _, _, _, _ = vae_model(
            x1, x2_e, len1, len2, mode="train"
        )
        mapped_src_enc_prior = feature_fusion(prior_mu, prior_logvar)
        alen = torch.arange(params.max_src_len, dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1
        y = x2[1:].masked_select(pred_mask[:-1])
        decoded = decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc=mapped_src_enc_prior,
            src_len=len1,
        )
        scores, loss = decoder(
            "predict",
            tensor=decoded,
            pred_mask=pred_mask,
            y=y,
            get_scores=True,
        )
        generations, gen_len = decoder.generate_from_latent(
            mapped_src_enc_prior,
            sample_temperature=None,
            max_len=trainer.params.max_generated_output_len,
        )
        constrained_generations, constrained_len = generate_from_latent_min_len(
            decoder,
            mapped_src_enc_prior,
            trainer.params.max_generated_output_len,
            min_generated_len,
        )
        constrained_beam = None
        if constrained_beam_size > 0:
            constrained_beam = generate_sde_constrained_beam(
                decoder,
                mapped_src_enc_prior,
                env,
                trainer.params.max_generated_output_len,
                constrained_beam_size,
            )
        rerank_beam_candidates = None
        rerank_sample_candidates = None
        if rerank_candidates > 0 or pair_drift_diffusion_candidates > 0:
            rerank_beam_size = max(
                1,
                rerank_topk_from_beam,
                constrained_beam_size,
                rerank_candidates,
                pair_drift_topk,
                pair_diffusion_topk,
            )
            rerank_beam_candidates = generate_sde_constrained_beam_candidates(
                decoder,
                mapped_src_enc_prior,
                env,
                trainer.params.max_generated_output_len,
                rerank_beam_size,
                max(1, rerank_candidates, pair_drift_topk, pair_diffusion_topk),
            )
        if sample_candidates > 0:
            rerank_sample_candidates = generate_sde_constrained_sample_candidates(
                decoder,
                mapped_src_enc_prior,
                env,
                trainer.params.max_generated_output_len,
                sample_candidates,
                sample_temperatures or [1.0],
                sample_top_k,
                sample_top_p,
            )
        rerank_candidates_all = merge_candidate_rows(
            [rerank_beam_candidates, rerank_sample_candidates]
        )
        pair_candidates = generate_paired_candidate_rows(
            env,
            rerank_candidates_all,
            trainer.params.max_generated_output_len,
            pair_drift_diffusion_candidates,
            pair_drift_topk,
            pair_diffusion_topk,
        )
        rerank_candidates_all = merge_candidate_rows(
            [rerank_candidates_all, pair_candidates]
        )
    return (
        scores,
        y,
        loss,
        generations.transpose(0, 1),
        gen_len,
        constrained_generations.transpose(0, 1),
        constrained_len,
        constrained_beam,
        rerank_candidates_all,
    )


def topk_metrics(scores: torch.Tensor, y: torch.Tensor, ks=(1, 3, 5)) -> dict:
    out = {}
    for k in ks:
        top = scores.topk(k, dim=1).indices
        out[f"token_top{k}"] = float((top == y[:, None]).any(dim=1).float().mean().item())
    return out


def decode_generation(env, ids: torch.Tensor) -> list[str]:
    pad = env.equation_word2id["<PAD>"]
    eos = env.equation_word2id["<EOS>"]
    out = []
    for token_id in ids.tolist():
        if token_id == pad:
            break
        if token_id == eos:
            continue
        out.append(env.equation_id2word[int(token_id)])
    return out


def template_tokens(tokens: list[str]) -> list[str]:
    return [w for w in tokens if w != "CONSTANT"]


def structural_token_fraction(tokens: list[str]) -> float:
    structural = {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "pow2",
        "pow3",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "neg",
        "inv",
        "x_0",
        "CONSTANT",
        "SPECIAL",
    }
    if not tokens:
        return 0.0
    return sum(token in structural for token in tokens) / len(tokens)


def sequence_metrics(
    env,
    truth_sequences: list[list[str]],
    generations: torch.Tensor,
    prefix: str,
) -> dict:
    exact = 0
    relaxed = 0
    nonempty = 0
    avg_len = 0.0
    struct_frac = 0.0
    decoded_examples = []
    for truth, gen_ids in zip(truth_sequences, generations):
        pred = decode_generation(env, gen_ids)
        exact += int(pred == truth)
        relaxed += int(template_tokens(pred) == template_tokens(truth))
        nonempty += int(len(pred) > 0)
        avg_len += len(pred)
        struct_frac += structural_token_fraction(pred)
        if len(decoded_examples) < 5:
            decoded_examples.append({"truth": " ".join(truth), "pred": " ".join(pred)})
    n = max(1, len(truth_sequences))
    return {
        f"{prefix}_sequence_exact": exact / n,
        f"{prefix}_sequence_relaxed_no_constants": relaxed / n,
        f"{prefix}_sequence_nonempty": nonempty / n,
        f"{prefix}_sequence_avg_len": avg_len / n,
        f"{prefix}_sequence_structural_token_frac": struct_frac / n,
        "examples": decoded_examples,
    }


def _parse_prefix_expr(tokens: list[str], pos: int = 0, constant_value: float = 1.0):
    if pos >= len(tokens):
        return None, pos
    token = tokens[pos]
    if token in {"x_0", "CONSTANT"}:
        return ("u" if token == "x_0" else repr(float(constant_value))), pos + 1
    if token in {"add", "sub", "mul"}:
        left, next_pos = _parse_prefix_expr(tokens, pos + 1, constant_value)
        if left is None:
            return None, pos
        right, next_pos = _parse_prefix_expr(tokens, next_pos, constant_value)
        if right is None:
            return None, pos
        op = {"add": "+", "sub": "-", "mul": "*"}[token]
        return f"({left} {op} {right})", next_pos
    if token in {"abs", "sqrt", "sin"}:
        child, next_pos = _parse_prefix_expr(tokens, pos + 1, constant_value)
        if child is None:
            return None, pos
        return f"{token}({child})", next_pos
    return None, pos


def candidate_tokens_to_system(
    tokens: list[str],
    constant_value: float = 1.0,
    drift_constant_value: float | None = None,
    diffusion_constant_value: float | None = None,
) -> SDESystem | None:
    split = split_sde_tokens(tokens)
    if split is None:
        return None
    drift_tokens, diffusion_tokens = split
    drift_value = constant_value if drift_constant_value is None else drift_constant_value
    diffusion_value = (
        constant_value if diffusion_constant_value is None else diffusion_constant_value
    )
    drift_expr, drift_pos = _parse_prefix_expr(
        drift_tokens,
        constant_value=drift_value,
    )
    diffusion_expr, diffusion_pos = _parse_prefix_expr(
        diffusion_tokens,
        constant_value=diffusion_value,
    )
    if drift_expr is None or diffusion_expr is None:
        return None
    if drift_pos != len(drift_tokens) or diffusion_pos != len(diffusion_tokens):
        return None
    return SDESystem(drift_expr, diffusion_expr)


def split_sde_tokens(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    if "<DRIFT>" not in tokens or "<DIFFUSION>" not in tokens:
        return None
    drift_idx = tokens.index("<DRIFT>")
    diffusion_idx = tokens.index("<DIFFUSION>")
    if drift_idx > diffusion_idx:
        return None
    drift_tokens = tokens[drift_idx + 1 : diffusion_idx]
    diffusion_tokens = tokens[diffusion_idx + 1 :]
    if not drift_tokens or not diffusion_tokens:
        return None
    return drift_tokens, diffusion_tokens


def parse_component_weights(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--rerank-component-weights must have three comma-separated values")
    return tuple(float(part) for part in parts)


def parse_constant_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--rerank-constant-values must include at least one value")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("--rerank-constant-values must be finite")
    return values


def rerank_base_score_kind(score_kind: str) -> str:
    if score_kind == "constant_grid_full":
        return "full_fingerprint"
    if score_kind == "constant_grid_componentwise":
        return "componentwise"
    if score_kind == "constant_grid_componentwise_no_multi_u0":
        return "componentwise_no_multi_u0"
    if score_kind == "constant_grid_rolewise_no_multi_u0":
        return "rolewise_no_multi_u0"
    if score_kind == "constant_grid_active_weak":
        return "componentwise"
    if score_kind == "constant_grid_active_only":
        return "active_only"
    if score_kind == "constant_grid_moments_downweighted":
        return "moments_downweighted"
    return score_kind


def uses_constant_grid(score_kind: str) -> bool:
    return score_kind.startswith("constant_grid_")


def empty_distance_details() -> dict:
    return {
        "distance": None,
        "full_distance": None,
        "multi_u0_moments_distance": None,
        "active_kramers_moyal_distance": None,
        "gaussian_weak_kernel_distance": None,
        "baseline_distance": None,
        "baseline_full_distance": None,
        "baseline_multi_u0_moments_distance": None,
        "baseline_active_kramers_moyal_distance": None,
        "baseline_gaussian_weak_kernel_distance": None,
        "best_constant": None,
        "best_drift_constant": None,
        "best_diffusion_constant": None,
        "shared_constant_distance": None,
        "shared_constant_full_distance": None,
        "shared_constant_multi_u0_moments_distance": None,
        "shared_constant_active_kramers_moyal_distance": None,
        "shared_constant_gaussian_weak_kernel_distance": None,
        "shared_best_constant": None,
        "active_kramers_moyal_residual": None,
        "gaussian_weak_kernel_residual": None,
    }


def _normalized_l2(target: np.ndarray, candidate: np.ndarray) -> float:
    diff = target - candidate
    if not np.all(np.isfinite(diff)):
        return float("inf")
    denom = np.linalg.norm(target) + 1e-8
    return float(np.linalg.norm(diff) / denom)


def _normalized_residual_vector(target: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    diff = target - candidate
    denom = np.linalg.norm(target) + 1e-8
    return diff / denom


def fingerprint_distance_details(
    target_y: np.ndarray,
    candidate_y: np.ndarray,
    score_kind: str,
    component_weights: tuple[float, float, float],
) -> dict:
    target = np.asarray(target_y, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_y, dtype=np.float64).reshape(-1)
    if target.shape != candidate.shape:
        inf = float("inf")
        return {
            "distance": inf,
            "full_distance": inf,
            "multi_u0_moments_distance": inf,
            "active_kramers_moyal_distance": inf,
            "gaussian_weak_kernel_distance": inf,
            "active_kramers_moyal_residual": None,
            "gaussian_weak_kernel_residual": None,
        }

    segment_slices = {
        "multi_u0_moments_distance": slice(0, 72),
        "active_kramers_moyal_distance": slice(72, 90),
        "gaussian_weak_kernel_distance": slice(90, 186),
    }
    details = {
        "full_distance": _normalized_l2(target, candidate),
    }
    for name, segment_slice in segment_slices.items():
        details[name] = _normalized_l2(target[segment_slice], candidate[segment_slice])
    details["active_kramers_moyal_residual"] = _normalized_residual_vector(
        target[segment_slices["active_kramers_moyal_distance"]],
        candidate[segment_slices["active_kramers_moyal_distance"]],
    )
    details["gaussian_weak_kernel_residual"] = _normalized_residual_vector(
        target[segment_slices["gaussian_weak_kernel_distance"]],
        candidate[segment_slices["gaussian_weak_kernel_distance"]],
    )

    if score_kind in {"short_moment", "active_only"}:
        details["distance"] = details["active_kramers_moyal_distance"]
    elif score_kind == "componentwise":
        w_multi, w_active, w_weak = component_weights
        details["distance"] = (
            w_multi * details["multi_u0_moments_distance"]
            + w_active * details["active_kramers_moyal_distance"]
            + w_weak * details["gaussian_weak_kernel_distance"]
        )
    elif score_kind == "componentwise_no_multi_u0":
        details["distance"] = (
            details["active_kramers_moyal_distance"]
            + details["gaussian_weak_kernel_distance"]
        )
    elif score_kind == "rolewise_no_multi_u0":
        _, w_active, w_weak = component_weights
        if component_weights == (1.0, 2.0, 1.0):
            w_active, w_weak = 1.0, 1.0
        details["distance"] = (
            w_active * details["active_kramers_moyal_distance"]
            + w_weak * details["gaussian_weak_kernel_distance"]
        )
    elif score_kind == "moments_downweighted":
        details["distance"] = (
            0.25 * details["multi_u0_moments_distance"]
            + 2.0 * details["active_kramers_moyal_distance"]
            + details["gaussian_weak_kernel_distance"]
        )
    else:
        details["distance"] = details["full_distance"]
    return details


def score_candidate_system(
    words: list[str],
    target_y: np.ndarray,
    config: FingerprintConfig,
    score_kind: str,
    component_weights: tuple[float, float, float],
    constant_values: list[float],
    seed: int,
) -> tuple[dict | None, str | None, int]:
    base_score_kind = rerank_base_score_kind(score_kind)
    grid_enabled = uses_constant_grid(score_kind)
    rolewise_grid_enabled = score_kind == "constant_grid_rolewise_no_multi_u0"
    split = split_sde_tokens(words)
    if split is None:
        return None, "parse_failed", 0
    drift_tokens, diffusion_tokens = split
    drift_constants = (
        constant_values
        if rolewise_grid_enabled and "CONSTANT" in drift_tokens
        else [1.0]
    )
    diffusion_constants = (
        constant_values
        if rolewise_grid_enabled and "CONSTANT" in diffusion_tokens
        else [1.0]
    )
    eval_constants = constant_values if grid_enabled and "CONSTANT" in words else [1.0]
    baseline_details = None
    best_details = None
    shared_details = None
    fingerprint_failures = 0
    failure_reasons = []

    if rolewise_grid_enabled:
        eval_constant_pairs = [
            (drift_value, diffusion_value)
            for drift_value in drift_constants
            for diffusion_value in diffusion_constants
        ]
    else:
        eval_constant_pairs = [(value, value) for value in eval_constants]

    for drift_value, diffusion_value in eval_constant_pairs:
        system = candidate_tokens_to_system(
            words,
            drift_constant_value=drift_value,
            diffusion_constant_value=diffusion_value,
        )
        if system is None:
            return None, "parse_failed", 0
        _, y_data, meta = solve_fingerprint(system, config, seed)
        if y_data is None or not np.all(np.isfinite(y_data)):
            fingerprint_failures += 1
            failure_reasons.append(meta.get("error", "fingerprint_failed"))
            continue
        details = fingerprint_distance_details(
            target_y,
            y_data,
            base_score_kind,
            component_weights,
        )
        details["best_constant"] = (
            drift_value if drift_value == diffusion_value else None
        )
        details["best_drift_constant"] = drift_value
        details["best_diffusion_constant"] = diffusion_value
        if drift_value == 1.0 and diffusion_value == 1.0:
            baseline_details = details
        if drift_value == diffusion_value and np.isfinite(details["distance"]) and (
            shared_details is None or details["distance"] < shared_details["distance"]
        ):
            shared_details = details
        if np.isfinite(details["distance"]) and (
            best_details is None or details["distance"] < best_details["distance"]
        ):
            best_details = details

    if grid_enabled and baseline_details is None:
        baseline_system = candidate_tokens_to_system(
            words,
            drift_constant_value=1.0,
            diffusion_constant_value=1.0,
        )
        if baseline_system is None:
            return None, "parse_failed", fingerprint_failures
        _, baseline_y, meta = solve_fingerprint(baseline_system, config, seed)
        if baseline_y is None or not np.all(np.isfinite(baseline_y)):
            failure_reasons.append(meta.get("error", "fingerprint_failed"))
        else:
            baseline_details = fingerprint_distance_details(
                target_y,
                baseline_y,
                base_score_kind,
                component_weights,
            )
            baseline_details["best_constant"] = 1.0
            baseline_details["best_drift_constant"] = 1.0
            baseline_details["best_diffusion_constant"] = 1.0

    if best_details is None:
        reason = failure_reasons[0] if failure_reasons else "fingerprint_failed"
        return None, reason, max(1, fingerprint_failures)

    if baseline_details is None:
        baseline_details = best_details
    if shared_details is None:
        shared_details = best_details

    return {
        "distance": best_details["distance"],
        "full_distance": best_details["full_distance"],
        "multi_u0_moments_distance": best_details["multi_u0_moments_distance"],
        "active_kramers_moyal_distance": best_details[
            "active_kramers_moyal_distance"
        ],
        "gaussian_weak_kernel_distance": best_details[
            "gaussian_weak_kernel_distance"
        ],
        "baseline_distance": baseline_details["distance"],
        "baseline_full_distance": baseline_details["full_distance"],
        "baseline_multi_u0_moments_distance": baseline_details[
            "multi_u0_moments_distance"
        ],
        "baseline_active_kramers_moyal_distance": baseline_details[
            "active_kramers_moyal_distance"
        ],
        "baseline_gaussian_weak_kernel_distance": baseline_details[
            "gaussian_weak_kernel_distance"
        ],
        "best_constant": best_details["best_constant"],
        "best_drift_constant": best_details["best_drift_constant"],
        "best_diffusion_constant": best_details["best_diffusion_constant"],
        "shared_constant_distance": shared_details["distance"],
        "shared_constant_full_distance": shared_details["full_distance"],
        "shared_constant_multi_u0_moments_distance": shared_details[
            "multi_u0_moments_distance"
        ],
        "shared_constant_active_kramers_moyal_distance": shared_details[
            "active_kramers_moyal_distance"
        ],
        "shared_constant_gaussian_weak_kernel_distance": shared_details[
            "gaussian_weak_kernel_distance"
        ],
        "shared_best_constant": shared_details["best_drift_constant"],
        "active_kramers_moyal_residual": best_details[
            "active_kramers_moyal_residual"
        ],
        "gaussian_weak_kernel_residual": best_details[
            "gaussian_weak_kernel_residual"
        ],
    }, None, fingerprint_failures


SEGMENT_DISTANCE_KEYS = (
    ("multi_u0_moments", "multi_u0_moments_distance"),
    ("active_kramers_moyal", "active_kramers_moyal_distance"),
    ("gaussian_weak_kernel", "gaussian_weak_kernel_distance"),
)


def _finite_delta(left, right):
    if (
        left is None
        or right is None
        or not np.isfinite(left)
        or not np.isfinite(right)
    ):
        return None
    return left - right


def score_segment_weights(
    score_kind: str,
    component_weights: tuple[float, float, float],
) -> dict:
    base_score_kind = rerank_base_score_kind(score_kind)
    if base_score_kind == "componentwise":
        w_multi, w_active, w_weak = component_weights
        return {
            "multi_u0_moments": w_multi,
            "active_kramers_moyal": w_active,
            "gaussian_weak_kernel": w_weak,
        }
    if base_score_kind == "componentwise_no_multi_u0":
        return {
            "multi_u0_moments": 0.0,
            "active_kramers_moyal": 1.0,
            "gaussian_weak_kernel": 1.0,
        }
    if base_score_kind == "rolewise_no_multi_u0":
        _, w_active, w_weak = component_weights
        if component_weights == (1.0, 2.0, 1.0):
            w_active, w_weak = 1.0, 1.0
        return {
            "multi_u0_moments": 0.0,
            "active_kramers_moyal": w_active,
            "gaussian_weak_kernel": w_weak,
        }
    if base_score_kind == "active_only" or base_score_kind == "short_moment":
        return {
            "multi_u0_moments": 0.0,
            "active_kramers_moyal": 1.0,
            "gaussian_weak_kernel": 0.0,
        }
    if base_score_kind == "moments_downweighted":
        return {
            "multi_u0_moments": 0.25,
            "active_kramers_moyal": 2.0,
            "gaussian_weak_kernel": 1.0,
        }
    return {
        "multi_u0_moments": 1.0,
        "active_kramers_moyal": 1.0,
        "gaussian_weak_kernel": 1.0,
    }


def has_state_dependent_drift(drift_tokens: list[str]) -> bool:
    state_tokens = {"x_0", "sin", "cos", "abs", "sqrt"}
    return any(token in state_tokens for token in drift_tokens)


def _finite_distance(value) -> bool:
    return value is not None and np.isfinite(value)


def _record_base_sort_key(record: dict, distance_key: str = "distance") -> tuple:
    value = record[distance_key]
    return (
        not _finite_distance(value),
        value if value is not None else float("inf"),
        -record["normalized_model_score"],
    )


def _tie_break_sort_key(record: dict, tie_break: str) -> tuple:
    if tie_break == "model_score":
        return (-record["normalized_model_score"], record["distance"])
    if tie_break == "active_distance":
        active = record["active_kramers_moyal_distance"]
        return (
            active if active is not None else float("inf"),
            record["distance"],
            -record["normalized_model_score"],
        )
    if tie_break == "state_dependent_drift":
        return (
            not record["state_dependent_drift"],
            record["active_kramers_moyal_distance"]
            if record["active_kramers_moyal_distance"] is not None
            else float("inf"),
            record["distance"],
            -record["normalized_model_score"],
        )
    return (record["distance"], -record["normalized_model_score"])


def tie_aware_rank_records(
    records: list[dict],
    tie_epsilon: float,
    tie_break: str,
    distance_key: str = "distance",
) -> list[dict]:
    sorted_records = sorted(records, key=lambda item: _record_base_sort_key(item, distance_key))
    if tie_epsilon <= 0.0 or tie_break == "none" or distance_key != "distance":
        return sorted_records

    ranked = []
    i = 0
    while i < len(sorted_records):
        anchor = sorted_records[i][distance_key]
        if not _finite_distance(anchor):
            ranked.extend(sorted_records[i:])
            break
        group = []
        while i < len(sorted_records):
            value = sorted_records[i][distance_key]
            if not _finite_distance(value) or value > anchor + tie_epsilon:
                break
            group.append(sorted_records[i])
            i += 1
        ranked.extend(sorted(group, key=lambda item: _tie_break_sort_key(item, tie_break)))
    return ranked


def top_tie_group(
    ranked_records: list[dict],
    tie_epsilon: float,
    debug_topk: int,
) -> list[dict]:
    if not ranked_records:
        return []
    top_distance = ranked_records[0]["distance"]
    if not _finite_distance(top_distance):
        return ranked_records[:debug_topk]
    limit = top_distance + max(0.0, tie_epsilon)
    group = [
        record
        for record in ranked_records
        if _finite_distance(record["distance"]) and record["distance"] <= limit
    ]
    return group[:debug_topk]


def oracle_gap_diagnostics(
    oracle: dict,
    candidate: dict,
    score_kind: str,
    component_weights: tuple[float, float, float],
) -> dict:
    segment_deltas = {
        name: _finite_delta(oracle[key], candidate[key])
        for name, key in SEGMENT_DISTANCE_KEYS
    }
    weights = score_segment_weights(score_kind, component_weights)
    weighted_segment_deltas = {
        name: None if value is None else value * weights[name]
        for name, value in segment_deltas.items()
    }
    finite_weighted_segments = {
        name: value
        for name, value in weighted_segment_deltas.items()
        if value is not None and weights[name] > 0.0
    }
    dominant_segment = None
    if finite_weighted_segments:
        dominant_segment = max(
            finite_weighted_segments.items(),
            key=lambda item: item[1],
        )[0]
    return {
        "distance_delta": _finite_delta(oracle["distance"], candidate["distance"]),
        "segment_deltas": segment_deltas,
        "weighted_segment_deltas": weighted_segment_deltas,
        "dominant_gap_segment": dominant_segment,
    }


def source_bucket(source: str | None) -> str:
    if source == "beam":
        return "beam"
    if source == "pair":
        return "pair"
    if source is not None and source.startswith("sample"):
        return "sampling"
    return "unknown"


def is_oracle_record(record: dict) -> bool:
    return bool(record["exact"] or record["relaxed"])


def same_template(left: list[str], right: list[str]) -> bool:
    return template_tokens(left) == template_tokens(right)


def classify_oracle_miss(selected: dict | None, oracle: dict, oracle_sources: set[str]) -> dict:
    if selected is None:
        return {
            "shares_drift": False,
            "shares_diffusion": False,
            "constant_mismatch": False,
            "miss_side": "no_selected_candidate",
            "oracle_lower_model_score": False,
            "oracle_only_pair": oracle_sources == {"pair"},
            "oracle_from_beam_or_sampling": bool(oracle_sources & {"beam", "sampling"}),
        }

    shares_drift = same_template(selected["drift_tokens"], oracle["drift_tokens"])
    shares_diffusion = same_template(
        selected["diffusion_tokens"],
        oracle["diffusion_tokens"],
    )
    exact_drift = selected["drift_tokens"] == oracle["drift_tokens"]
    exact_diffusion = selected["diffusion_tokens"] == oracle["diffusion_tokens"]
    constant_mismatch = (
        shares_drift
        and shares_diffusion
        and not (exact_drift and exact_diffusion)
    )
    if constant_mismatch:
        miss_side = "constant_mismatch"
    elif shares_diffusion and not shares_drift:
        miss_side = "drift"
    elif shares_drift and not shares_diffusion:
        miss_side = "diffusion"
    elif not shares_drift and not shares_diffusion:
        miss_side = "both"
    else:
        miss_side = "unknown"
    return {
        "shares_drift": shares_drift,
        "shares_diffusion": shares_diffusion,
        "constant_mismatch": constant_mismatch,
        "miss_side": miss_side,
        "oracle_lower_model_score": (
            oracle["normalized_model_score"] < selected["normalized_model_score"]
        ),
        "oracle_only_pair": oracle_sources == {"pair"},
        "oracle_from_beam_or_sampling": bool(oracle_sources & {"beam", "sampling"}),
    }


def empty_oracle_diagnostic_counts() -> dict:
    counts = {
        "rerank_oracle_samples_any_count": 0,
        "rerank_oracle_samples_selected_count": 0,
        "rerank_oracle_samples_missed_count": 0,
        "rerank_oracle_miss_constant_mismatch_count": 0,
        "rerank_oracle_miss_same_diffusion_wrong_drift_count": 0,
        "rerank_oracle_miss_same_drift_wrong_diffusion_count": 0,
        "rerank_oracle_miss_both_sides_wrong_count": 0,
        "rerank_oracle_miss_oracle_lower_model_score_count": 0,
        "rerank_oracle_miss_oracle_only_pair_count": 0,
        "rerank_oracle_miss_beam_or_sampling_score_miss_count": 0,
        "rerank_oracle_miss_rescue_cases_total": 0,
        "rerank_oracle_miss_rescued_by_rolewise_constant_grid": 0,
        "rerank_oracle_miss_still_missed_after_rolewise_constant_grid": 0,
        "rerank_selected_exact_shared_constant_count": 0,
        "rerank_selected_relaxed_shared_constant_count": 0,
        "rerank_selected_exact_rolewise_constant_count": 0,
        "rerank_selected_relaxed_rolewise_constant_count": 0,
        "rerank_oracle_exact_shared_constant_count": 0,
        "rerank_oracle_relaxed_shared_constant_count": 0,
        "rerank_oracle_exact_rolewise_constant_count": 0,
        "rerank_oracle_relaxed_rolewise_constant_count": 0,
    }
    for bucket in ("beam", "sampling", "pair", "unknown"):
        counts[f"rerank_oracle_best_source_{bucket}_count"] = 0
        counts[f"rerank_oracle_any_source_{bucket}_count"] = 0
        counts[f"rerank_selected_hit_source_{bucket}_count"] = 0
    return counts


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_residual_debug_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def residual_debug_candidate_record(
    record: dict,
    rank: int,
    selected_record: dict | None,
) -> dict:
    return {
        "candidate_rank": rank,
        "candidate_index": record["candidate_idx"],
        "source": record["source"],
        "source_rank": record.get("source_rank"),
        "sequence": record["words"],
        "drift_tokens": record["drift_tokens"],
        "diffusion_tokens": record["diffusion_tokens"],
        "is_selected": record is selected_record,
        "is_oracle_exact": bool(record["exact"]),
        "is_oracle_relaxed": bool(record["relaxed"]),
        "is_pair_oracle": bool(record["source"] == "pair" and is_oracle_record(record)),
        "is_pair": bool(record["source"] == "pair"),
        "rerank_score": record["distance"],
        "active_score": record["active_kramers_moyal_distance"],
        "weak_score": record["gaussian_weak_kernel_distance"],
        "full_score": record["full_distance"],
        "multi_u0_score": record["multi_u0_moments_distance"],
        "baseline_score": record["baseline_distance"],
        "shared_constant_score": record["shared_constant_distance"],
        "best_constant": record["best_constant"],
        "best_drift_constant": record["best_drift_constant"],
        "best_diffusion_constant": record["best_diffusion_constant"],
        "shared_best_constant": record["shared_best_constant"],
        "normalized_model_score": record["normalized_model_score"],
        "model_score": record["model_score"],
        "state_dependent_drift": record["state_dependent_drift"],
        "failure_reason": record["failure_reason"],
        "active_residual_vector": record["active_kramers_moyal_residual"],
        "weak_residual_vector": record["gaussian_weak_kernel_residual"],
    }


def residual_debug_sample_record(
    sample_index: int,
    truth: list[str],
    greedy_words: list[str],
    beam_words: list[str],
    ranked_records: list[dict],
    selected_record: dict | None,
    oracle_record: dict | None,
) -> dict:
    selected_rank = None
    oracle_rank = None
    for rank, record in enumerate(ranked_records, start=1):
        if record is selected_record:
            selected_rank = rank
        if record is oracle_record:
            oracle_rank = rank
    return {
        "sample_index": sample_index,
        "truth_sequence": truth,
        "greedy_sequence": greedy_words,
        "constrained_beam_sequence": beam_words,
        "selected_sequence": (
            selected_record["words"] if selected_record is not None else None
        ),
        "selected_rank": selected_rank,
        "selected_is_oracle": (
            bool(is_oracle_record(selected_record)) if selected_record is not None else False
        ),
        "oracle_rank": oracle_rank,
        "oracle_sequence": (
            oracle_record["words"] if oracle_record is not None else None
        ),
        "has_oracle": oracle_record is not None,
        "has_pair_oracle": any(
            record["source"] == "pair" and is_oracle_record(record)
            for record in ranked_records
        ),
        "candidate_count": len(ranked_records),
        "candidates": [
            residual_debug_candidate_record(record, rank, selected_record)
            for rank, record in enumerate(ranked_records, start=1)
        ],
    }


def rerank_candidate_rows(
    env,
    samples: list[dict],
    truth_sequences: list[list[str]],
    greedy_generations: torch.Tensor,
    constrained_beam: torch.Tensor | None,
    candidate_rows: list[list[dict]] | None,
    config: FingerprintConfig,
    score_kind: str,
    component_weights: tuple[float, float, float],
    constant_values: list[float],
    tie_epsilon: float,
    tie_break: str,
    base_seed: int,
    max_len: int,
    debug_topk: int,
    sample_offset: int = 0,
    residual_debug: bool = False,
) -> tuple[torch.Tensor, dict]:
    pad_id = env.equation_word2id["<PAD>"]
    rows = []
    attempted = 0
    valid = 0
    parse_failures = 0
    fingerprint_failures = 0
    oracle_exact = 0
    oracle_relaxed = 0
    pair_attempted = 0
    pair_valid = 0
    pair_oracle_exact = 0
    pair_oracle_relaxed = 0
    unique_candidate_total = 0
    unique_drift_total = 0
    unique_diffusion_total = 0
    unique_pair_candidate_total = 0
    debug_rows = []
    oracle_diagnostic_counts = empty_oracle_diagnostic_counts()
    oracle_miss_rows = []
    oracle_rescue_rows = []
    residual_debug_rows = []

    for sample_idx, sample in enumerate(samples):
        global_sample_idx = sample_offset + sample_idx
        records = []
        candidates = candidate_rows[sample_idx] if candidate_rows is not None else []
        unique_candidates = set()
        unique_drifts = set()
        unique_diffusions = set()
        unique_pair_candidates = set()
        truth = truth_sequences[sample_idx]
        greedy_words = decode_generation(env, greedy_generations[sample_idx])
        beam_words = (
            decode_generation(env, constrained_beam[sample_idx])
            if constrained_beam is not None
            else []
        )
        selection_ranked_records = []
        source_rank_counts = {}
        for candidate_idx, candidate in enumerate(candidates):
            attempted += 1
            words = decode_generation(env, candidate["tensor"])
            unique_candidates.add(tuple(words))
            source = candidate.get("source", "beam")
            source_rank_counts[source] = source_rank_counts.get(source, 0) + 1
            source_rank = source_rank_counts[source]
            is_pair = source == "pair"
            if is_pair:
                pair_attempted += 1
                unique_pair_candidates.add(tuple(words))
            split = split_sde_tokens(words)
            drift_tokens = split[0] if split is not None else []
            diffusion_tokens = split[1] if split is not None else []
            if drift_tokens:
                unique_drifts.add(tuple(drift_tokens))
            if diffusion_tokens:
                unique_diffusions.add(tuple(diffusion_tokens))
            exact_hit = words == truth
            relaxed_hit = template_tokens(words) == template_tokens(truth)
            failure_reason = None
            distance = None
            distance_details = empty_distance_details()
            system = candidate_tokens_to_system(words)
            if system is None:
                failure_reason = "parse_failed"
                parse_failures += 1
                records.append(
                    {
                        "words": words,
                        "drift_tokens": drift_tokens,
                        "diffusion_tokens": diffusion_tokens,
                        **distance_details,
                        "model_score": candidate["model_score"],
                        "normalized_model_score": candidate["normalized_model_score"],
                        "source": source,
                        "tensor": candidate["tensor"],
                        "candidate_idx": candidate_idx,
                        "source_rank": source_rank,
                        "state_dependent_drift": has_state_dependent_drift(drift_tokens),
                        "exact": exact_hit,
                        "relaxed": relaxed_hit,
                        "failure_reason": failure_reason,
                    }
                )
                continue
            seed = base_seed + sample_idx * 1009 + candidate_idx
            distance_details, failure_reason, _ = score_candidate_system(
                words,
                sample["y_to_fit"],
                config,
                score_kind,
                component_weights,
                constant_values,
                seed,
            )
            if distance_details is None:
                distance_details = empty_distance_details()
                fingerprint_failures += 1
                records.append(
                    {
                        "words": words,
                        "drift_tokens": drift_tokens,
                        "diffusion_tokens": diffusion_tokens,
                        **distance_details,
                        "model_score": candidate["model_score"],
                        "normalized_model_score": candidate["normalized_model_score"],
                        "source": source,
                        "tensor": candidate["tensor"],
                        "candidate_idx": candidate_idx,
                        "source_rank": source_rank,
                        "state_dependent_drift": has_state_dependent_drift(drift_tokens),
                        "exact": exact_hit,
                        "relaxed": relaxed_hit,
                        "failure_reason": failure_reason,
                    }
                )
                continue
            valid += 1
            if is_pair:
                pair_valid += 1
            distance = distance_details["distance"]
            if not np.isfinite(distance):
                fingerprint_failures += 1
                failure_reason = "nonfinite_distance"
                records.append(
                    {
                        "words": words,
                        "drift_tokens": drift_tokens,
                        "diffusion_tokens": diffusion_tokens,
                        **distance_details,
                        "model_score": candidate["model_score"],
                        "normalized_model_score": candidate["normalized_model_score"],
                        "source": source,
                        "tensor": candidate["tensor"],
                        "candidate_idx": candidate_idx,
                        "source_rank": source_rank,
                        "state_dependent_drift": has_state_dependent_drift(drift_tokens),
                        "exact": exact_hit,
                        "relaxed": relaxed_hit,
                        "failure_reason": failure_reason,
                    }
                )
                continue
            record = {
                "words": words,
                "drift_tokens": drift_tokens,
                "diffusion_tokens": diffusion_tokens,
                **distance_details,
                "model_score": candidate["model_score"],
                "normalized_model_score": candidate["normalized_model_score"],
                "source": source,
                "tensor": candidate["tensor"],
                "candidate_idx": candidate_idx,
                "source_rank": source_rank,
                "state_dependent_drift": has_state_dependent_drift(drift_tokens),
                "exact": exact_hit,
                "relaxed": relaxed_hit,
                "failure_reason": failure_reason,
            }
            records.append(record)
        oracle_exact += int(any(record["exact"] for record in records))
        oracle_relaxed += int(any(record["relaxed"] for record in records))
        pair_oracle_exact += int(
            any(record["exact"] and record["source"] == "pair" for record in records)
        )
        pair_oracle_relaxed += int(
            any(record["relaxed"] and record["source"] == "pair" for record in records)
        )
        unique_candidate_total += len(unique_candidates)
        unique_drift_total += len(unique_drifts)
        unique_diffusion_total += len(unique_diffusions)
        unique_pair_candidate_total += len(unique_pair_candidates)
        selection_ranked_records = tie_aware_rank_records(
            records,
            tie_epsilon,
            tie_break,
        )
        shared_ranked_records = tie_aware_rank_records(
            records,
            0.0,
            "none",
            distance_key="shared_constant_distance",
        )
        selected_record = next(
            (
                record
                for record in selection_ranked_records
                if _finite_distance(record["distance"])
            ),
            None,
        )
        shared_selected_record = next(
            (
                record
                for record in shared_ranked_records
                if _finite_distance(record["shared_constant_distance"])
            ),
            None,
        )
        if selected_record is None:
            rows.append(torch.full((max_len,), pad_id, dtype=torch.long))
        else:
            rows.append(selected_record["tensor"].detach().cpu())

        oracle_records = [record for record in selection_ranked_records if is_oracle_record(record)]
        best_oracle_record = oracle_records[0] if oracle_records else None
        shared_oracle_records = [
            record for record in shared_ranked_records if is_oracle_record(record)
        ]
        shared_best_oracle_record = (
            shared_oracle_records[0] if shared_oracle_records else None
        )
        shared_oracle_rank = None
        if shared_best_oracle_record is not None:
            shared_oracle_rank = shared_ranked_records.index(shared_best_oracle_record) + 1
        rolewise_oracle_rank = None
        if best_oracle_record is not None:
            rolewise_oracle_rank = selection_ranked_records.index(best_oracle_record) + 1
        oracle_source_buckets = {
            source_bucket(record["source"])
            for record in records
            if is_oracle_record(record)
        }
        if best_oracle_record is not None:
            oracle_diagnostic_counts["rerank_oracle_samples_any_count"] += 1
            oracle_diagnostic_counts[
                f"rerank_oracle_best_source_{source_bucket(best_oracle_record['source'])}_count"
            ] += 1
            for bucket in oracle_source_buckets:
                oracle_diagnostic_counts[
                    f"rerank_oracle_any_source_{bucket}_count"
                ] += 1
        selected_is_oracle = selected_record is not None and is_oracle_record(selected_record)
        shared_selected_is_oracle = (
            shared_selected_record is not None and is_oracle_record(shared_selected_record)
        )
        if shared_selected_record is not None:
            oracle_diagnostic_counts["rerank_selected_exact_shared_constant_count"] += int(
                shared_selected_record["exact"]
            )
            oracle_diagnostic_counts[
                "rerank_selected_relaxed_shared_constant_count"
            ] += int(shared_selected_record["relaxed"])
        if selected_record is not None:
            oracle_diagnostic_counts[
                "rerank_selected_exact_rolewise_constant_count"
            ] += int(selected_record["exact"])
            oracle_diagnostic_counts[
                "rerank_selected_relaxed_rolewise_constant_count"
            ] += int(selected_record["relaxed"])
        oracle_diagnostic_counts["rerank_oracle_exact_shared_constant_count"] += int(
            any(record["exact"] for record in records)
        )
        oracle_diagnostic_counts[
            "rerank_oracle_relaxed_shared_constant_count"
        ] += int(any(record["relaxed"] for record in records))
        oracle_diagnostic_counts["rerank_oracle_exact_rolewise_constant_count"] += int(
            any(record["exact"] for record in records)
        )
        oracle_diagnostic_counts[
            "rerank_oracle_relaxed_rolewise_constant_count"
        ] += int(any(record["relaxed"] for record in records))
        if shared_best_oracle_record is not None and not shared_selected_is_oracle:
            rescued_by_rolewise = selected_is_oracle
            oracle_diagnostic_counts["rerank_oracle_miss_rescue_cases_total"] += 1
            oracle_diagnostic_counts[
                "rerank_oracle_miss_rescued_by_rolewise_constant_grid"
            ] += int(rescued_by_rolewise)
            oracle_diagnostic_counts[
                "rerank_oracle_miss_still_missed_after_rolewise_constant_grid"
            ] += int(not rescued_by_rolewise)
            oracle_rescue_rows.append(
                {
                    "sample_index": global_sample_idx,
                    "truth": truth,
                    "shared_selected": shared_selected_record,
                    "rolewise_selected": selected_record,
                    "oracle": best_oracle_record or shared_best_oracle_record,
                    "shared_oracle_rank": shared_oracle_rank,
                    "rolewise_oracle_rank": rolewise_oracle_rank,
                    "rescued": rescued_by_rolewise,
                    "oracle_rank_improved": (
                        shared_oracle_rank is not None
                        and rolewise_oracle_rank is not None
                        and rolewise_oracle_rank < shared_oracle_rank
                    ),
                }
            )
        if selected_is_oracle:
            oracle_diagnostic_counts["rerank_oracle_samples_selected_count"] += 1
            oracle_diagnostic_counts[
                f"rerank_selected_hit_source_{source_bucket(selected_record['source'])}_count"
            ] += 1
        elif best_oracle_record is not None:
            oracle_diagnostic_counts["rerank_oracle_samples_missed_count"] += 1
            miss_info = classify_oracle_miss(
                selected_record,
                best_oracle_record,
                oracle_source_buckets,
            )
            if miss_info["constant_mismatch"]:
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_constant_mismatch_count"
                ] += 1
            if miss_info["miss_side"] == "drift":
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_same_diffusion_wrong_drift_count"
                ] += 1
            elif miss_info["miss_side"] == "diffusion":
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_same_drift_wrong_diffusion_count"
                ] += 1
            elif miss_info["miss_side"] == "both":
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_both_sides_wrong_count"
                ] += 1
            if miss_info["oracle_lower_model_score"]:
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_oracle_lower_model_score_count"
                ] += 1
            if miss_info["oracle_only_pair"]:
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_oracle_only_pair_count"
                ] += 1
            if miss_info["oracle_from_beam_or_sampling"]:
                oracle_diagnostic_counts[
                    "rerank_oracle_miss_beam_or_sampling_score_miss_count"
                ] += 1
            oracle_miss_rows.append(
                {
                    "sample_index": global_sample_idx,
                    "truth": truth,
                    "selected": selected_record,
                    "oracle": best_oracle_record,
                    "score_gap": (
                        best_oracle_record["distance"] - selected_record["distance"]
                        if selected_record is not None
                        and _finite_distance(best_oracle_record["distance"])
                        and _finite_distance(selected_record["distance"])
                        else None
                    ),
                    **miss_info,
                }
            )
        if debug_topk > 0:
            ranked_records = selection_ranked_records
            baseline_ranked_records = tie_aware_rank_records(
                records,
                0.0,
                "none",
                distance_key="baseline_distance",
            )
            oracle_rank = None
            oracle_record = None
            for rank, record in enumerate(ranked_records, start=1):
                if record["exact"] or record["relaxed"]:
                    oracle_rank = rank
                    oracle_record = record
                    break
            oracle_baseline_rank = None
            oracle_baseline_record = None
            for rank, record in enumerate(baseline_ranked_records, start=1):
                if record["exact"] or record["relaxed"]:
                    oracle_baseline_rank = rank
                    oracle_baseline_record = record
                    break
            top_distance = (
                ranked_records[0]["distance"]
                if ranked_records and ranked_records[0]["distance"] is not None
                else None
            )
            top_record = ranked_records[0] if ranked_records else None
            tie_group = top_tie_group(ranked_records, tie_epsilon, debug_topk)
            oracle_distance = oracle_record["distance"] if oracle_record is not None else None
            oracle_baseline_distance = (
                oracle_baseline_record["baseline_distance"]
                if oracle_baseline_record is not None
                else None
            )
            distance_delta = (
                oracle_distance - top_distance
                if oracle_distance is not None
                and top_distance is not None
                and np.isfinite(oracle_distance)
                and np.isfinite(top_distance)
                else None
            )
            candidates_before_oracle = []
            if oracle_record is not None and oracle_rank is not None:
                for rank, record in enumerate(ranked_records[: oracle_rank - 1], start=1):
                    if len(candidates_before_oracle) >= debug_topk:
                        break
                    gap = oracle_gap_diagnostics(
                        oracle_record,
                        record,
                        score_kind,
                        component_weights,
                    )
                    candidates_before_oracle.append(
                        {
                            "rank": rank,
                            "record": record,
                            **gap,
                        }
                    )
            if len(debug_rows) < debug_topk or oracle_record is not None:
                debug_rows.append(
                    {
                        "sample_index": global_sample_idx,
                        "truth": truth,
                        "greedy": greedy_words,
                        "constrained_beam": beam_words,
                        "candidates": ranked_records[:debug_topk],
                        "tie_group": tie_group,
                        "tie_break": tie_break,
                        "tie_epsilon": tie_epsilon,
                        "tie_selected_candidate": selected_record,
                        "top_candidate": top_record,
                        "oracle_rank": oracle_rank,
                        "oracle_baseline_rank": oracle_baseline_rank,
                        "oracle_candidate": oracle_record,
                        "top_distance": top_distance,
                        "oracle_distance": oracle_distance,
                        "oracle_baseline_distance": oracle_baseline_distance,
                        "oracle_top_distance_delta": distance_delta,
                        "candidates_before_oracle": candidates_before_oracle,
                    }
                )
        if residual_debug:
            residual_debug_rows.append(
                residual_debug_sample_record(
                    global_sample_idx,
                    truth,
                    greedy_words,
                    beam_words,
                    selection_ranked_records,
                    selected_record,
                    best_oracle_record,
                )
            )

    return torch.stack(rows, dim=0), {
        "rerank_candidates_attempted": attempted,
        "rerank_candidates_valid": valid,
        "rerank_parse_failures": parse_failures,
        "rerank_fingerprint_failures": fingerprint_failures,
        "rerank_pair_candidates_attempted": pair_attempted,
        "rerank_pair_candidates_valid": pair_valid,
        "rerank_oracle_sequence_exact_count": oracle_exact,
        "rerank_oracle_sequence_relaxed_no_constants_count": oracle_relaxed,
        "rerank_pair_oracle_sequence_exact_count": pair_oracle_exact,
        "rerank_pair_oracle_sequence_relaxed_no_constants_count": pair_oracle_relaxed,
        "rerank_oracle_total": len(samples),
        "rerank_unique_candidate_total": unique_candidate_total,
        "rerank_unique_drift_total": unique_drift_total,
        "rerank_unique_diffusion_total": unique_diffusion_total,
        "rerank_unique_paired_candidate_total": unique_pair_candidate_total,
        "rerank_debug": debug_rows,
        "rerank_oracle_misses": oracle_miss_rows,
        "rerank_oracle_rescue_cases": oracle_rescue_rows,
        "rerank_residual_debug": residual_debug_rows,
        **oracle_diagnostic_counts,
    }


def generate_eval_samples(args: argparse.Namespace, env, params) -> list[dict]:
    config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    py_rng = random.Random(args.eval_seed)
    np_rng = np.random.default_rng(args.eval_seed)
    samples = []
    attempts = 0
    while len(samples) < args.eval_samples and attempts < args.eval_samples * 100:
        attempts += 1
        system = sample_sde_system(py_rng)
        seed = int(np_rng.integers(0, 2**31 - 1))
        sample = build_sample(system, env, params, config, seed)
        if sample is not None:
            samples.append(sample)
    if len(samples) < args.eval_samples:
        raise RuntimeError(f"generated only {len(samples)} eval samples")
    return samples


def train_steps(trainer: Trainer, steps: int) -> None:
    task = trainer.params.tasks[0]
    for step in range(steps):
        _, _, _, loss = trainer.enc_dec_vae_step(task)
        trainer.iter()
        if (step + 1) % trainer.params.print_freq == 0 or step + 1 == steps:
            losses = trainer.current_vae_losses
            print(
                f"train_step={step+1} total={losses['total_loss']:.4f} "
                f"prior={losses['pred_loss_prior']:.4f} "
                f"post={losses['pred_loss_post']:.4f} "
                f"kl={losses['kl_loss']:.4f}"
            )


def save_checkpoint_to_path(
    trainer: Trainer,
    checkpoint_path: Path,
    include_optimizer: bool = False,
) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    old_dump_path = trainer.params.dump_path
    trainer.params.dump_path = str(checkpoint_path.parent)
    try:
        trainer.save_checkpoint(
            checkpoint_path.stem,
            include_optimizer=include_optimizer,
        )
    finally:
        trainer.params.dump_path = old_dump_path
    print(f"saved_checkpoint={checkpoint_path}")


def load_checkpoint_from_path(
    trainer: Trainer,
    checkpoint_path: Path,
    requires_grad: bool = True,
) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    trainer.reload_checkpoint(path=str(checkpoint_path), requires_grad=requires_grad)
    print(f"loaded_checkpoint={checkpoint_path}")


def evaluate(
    trainer: Trainer,
    eval_samples: list[dict],
    batch_size: int,
    seed: int,
    min_generated_len: int,
    constrained_beam_size: int,
    rerank_candidates: int,
    rerank_topk_from_beam: int,
    rerank_score: str,
    rerank_component_weights: tuple[float, float, float],
    rerank_constant_values: list[float],
    rerank_tie_epsilon: float,
    rerank_tie_break: str,
    fingerprint_config: FingerprintConfig,
    rerank_debug_topk: int,
    sample_candidates: int,
    sample_temperatures: list[float],
    sample_top_k: int,
    sample_top_p: float,
    pair_drift_diffusion_candidates: int,
    pair_drift_topk: int,
    pair_diffusion_topk: int,
    residual_debug: bool = False,
) -> dict:
    rows = []
    seq_rows = []
    examples = []
    rerank_stat_rows = []
    n_batches = int(np.ceil(len(eval_samples) / batch_size))
    for i in range(n_batches):
        chunk = eval_samples[i * batch_size : (i + 1) * batch_size]
        normalized = [normalize_sample(s, trainer.env) for s in chunk]
        batch = {
            "x_to_fit": [s["x_to_fit"] for s in normalized],
            "y_to_fit": [s["y_to_fit"] for s in normalized],
            "tree_encoded": [s["tree_encoded"] for s in normalized],
            "drift_tree_encoded": [s["drift_tree_encoded"] for s in normalized],
            "diffusion_tree_encoded": [s["diffusion_tree_encoded"] for s in normalized],
            "tree": [s["tree"] for s in chunk],
        }
        if not getattr(trainer.params, "split_sde_loss", False):
            (
                scores,
                y,
                loss,
                generations,
                _,
                constrained_generations,
                _,
                constrained_beam,
                rerank_beam_candidates,
            ) = (
                prior_scores_for_joint_batch(
                    trainer,
                    batch,
                    min_generated_len,
                    constrained_beam_size,
                    rerank_candidates,
                    rerank_topk_from_beam,
                    sample_candidates,
                    sample_temperatures,
                    sample_top_k,
                    sample_top_p,
                    pair_drift_diffusion_candidates,
                    pair_drift_topk,
                    pair_diffusion_topk,
                )
            )
            metrics = topk_metrics(scores, y)
            metrics["loss"] = float(loss.item())
            seq = sequence_metrics(
                trainer.env,
                batch["tree_encoded"],
                generations,
                "greedy",
            )
            constrained_seq = sequence_metrics(
                trainer.env,
                batch["tree_encoded"],
                constrained_generations,
                "minlen_greedy",
            )
            seq.update({k: v for k, v in constrained_seq.items() if k != "examples"})
            if constrained_beam is not None:
                beam_seq = sequence_metrics(
                    trainer.env,
                    batch["tree_encoded"],
                    constrained_beam,
                    "constrained_beam",
                )
                seq.update({k: v for k, v in beam_seq.items() if k != "examples"})
                examples.extend(beam_seq["examples"])
            if (
                rerank_candidates > 0
                or sample_candidates > 0
                or pair_drift_diffusion_candidates > 0
            ):
                reranked_generations, rerank_stats = rerank_candidate_rows(
                    trainer.env,
                    normalized,
                    batch["tree_encoded"],
                    generations,
                    constrained_beam,
                    rerank_beam_candidates,
                    fingerprint_config,
                    rerank_score,
                    rerank_component_weights,
                    rerank_constant_values,
                    rerank_tie_epsilon,
                    rerank_tie_break,
                    seed + i * 100003,
                    trainer.params.max_generated_output_len,
                    rerank_debug_topk,
                    i * batch_size,
                    residual_debug,
                )
                reranked_seq = sequence_metrics(
                    trainer.env,
                    batch["tree_encoded"],
                    reranked_generations,
                    "reranked",
                )
                seq.update({k: v for k, v in reranked_seq.items() if k != "examples"})
                rerank_stat_rows.append(rerank_stats)
                if constrained_beam is None:
                    examples.extend(reranked_seq["examples"])
            seq_rows.append({k: v for k, v in seq.items() if k != "examples"})
            if constrained_beam is None:
                examples.extend(constrained_seq["examples"])
            rows.append(metrics)
            continue

        drift, diffusion = prior_scores_for_batch(trainer, batch, min_generated_len)
        (
            drift_scores,
            drift_y,
            drift_loss,
            drift_generations,
            _,
            drift_constrained,
            _,
        ) = drift
        (
            diffusion_scores,
            diffusion_y,
            diffusion_loss,
            diffusion_generations,
            _,
            diffusion_constrained,
            _,
        ) = diffusion

        metrics = {}
        for name, scores, y, loss in [
            ("drift", drift_scores, drift_y, drift_loss),
            ("diffusion", diffusion_scores, diffusion_y, diffusion_loss),
        ]:
            part_metrics = topk_metrics(scores, y)
            metrics.update({f"{name}_{k}": v for k, v in part_metrics.items()})
            metrics[f"{name}_loss"] = float(loss.item())
        metrics["loss"] = metrics["drift_loss"] + metrics["diffusion_loss"]

        seq = sequence_metrics(
            trainer.env,
            batch["drift_tree_encoded"],
            drift_generations.transpose(0, 1),
            "drift_greedy",
        )
        seq.update(
            {
                k: v
                for k, v in sequence_metrics(
                    trainer.env,
                    batch["diffusion_tree_encoded"],
                    diffusion_generations.transpose(0, 1),
                    "diffusion_greedy",
                ).items()
                if k != "examples"
            }
        )
        seq.update(
            {
                k: v
                for k, v in sequence_metrics(
                    trainer.env,
                    batch["drift_tree_encoded"],
                    drift_constrained.transpose(0, 1),
                    "drift_minlen_greedy",
                ).items()
                if k != "examples"
            }
        )
        constrained_seq = sequence_metrics(
            trainer.env,
            batch["diffusion_tree_encoded"],
            diffusion_constrained.transpose(0, 1),
            "diffusion_minlen_greedy",
        )
        seq.update({k: v for k, v in constrained_seq.items() if k != "examples"})
        seq_rows.append({k: v for k, v in seq.items() if k != "examples"})
        examples.extend(constrained_seq["examples"])
        rows.append(metrics)

    result = {}
    for key in rows[0]:
        result[key] = float(np.mean([row[key] for row in rows]))
    for key in seq_rows[0]:
        result[key] = float(np.mean([row[key] for row in seq_rows]))
    if rerank_stat_rows:
        for key in [
            "rerank_candidates_attempted",
            "rerank_candidates_valid",
            "rerank_parse_failures",
            "rerank_fingerprint_failures",
            "rerank_pair_candidates_attempted",
            "rerank_pair_candidates_valid",
        ]:
            result[key] = float(np.sum([row[key] for row in rerank_stat_rows]))
        for key in [
            "rerank_oracle_samples_any_count",
            "rerank_oracle_samples_selected_count",
            "rerank_oracle_samples_missed_count",
            "rerank_oracle_miss_constant_mismatch_count",
            "rerank_oracle_miss_same_diffusion_wrong_drift_count",
            "rerank_oracle_miss_same_drift_wrong_diffusion_count",
            "rerank_oracle_miss_both_sides_wrong_count",
            "rerank_oracle_miss_oracle_lower_model_score_count",
            "rerank_oracle_miss_oracle_only_pair_count",
            "rerank_oracle_miss_beam_or_sampling_score_miss_count",
            "rerank_oracle_miss_rescue_cases_total",
            "rerank_oracle_miss_rescued_by_rolewise_constant_grid",
            "rerank_oracle_miss_still_missed_after_rolewise_constant_grid",
            "rerank_selected_exact_shared_constant_count",
            "rerank_selected_relaxed_shared_constant_count",
            "rerank_selected_exact_rolewise_constant_count",
            "rerank_selected_relaxed_rolewise_constant_count",
            "rerank_oracle_exact_shared_constant_count",
            "rerank_oracle_relaxed_shared_constant_count",
            "rerank_oracle_exact_rolewise_constant_count",
            "rerank_oracle_relaxed_rolewise_constant_count",
            "rerank_oracle_best_source_beam_count",
            "rerank_oracle_best_source_sampling_count",
            "rerank_oracle_best_source_pair_count",
            "rerank_oracle_best_source_unknown_count",
            "rerank_oracle_any_source_beam_count",
            "rerank_oracle_any_source_sampling_count",
            "rerank_oracle_any_source_pair_count",
            "rerank_oracle_any_source_unknown_count",
            "rerank_selected_hit_source_beam_count",
            "rerank_selected_hit_source_sampling_count",
            "rerank_selected_hit_source_pair_count",
            "rerank_selected_hit_source_unknown_count",
        ]:
            result[key] = float(np.sum([row[key] for row in rerank_stat_rows]))
        oracle_total_raw = float(
            np.sum([row["rerank_oracle_total"] for row in rerank_stat_rows])
        )
        result["rerank_oracle_total"] = oracle_total_raw
        oracle_total = max(1.0, oracle_total_raw)
        result["rerank_oracle_sequence_exact"] = float(
            np.sum([row["rerank_oracle_sequence_exact_count"] for row in rerank_stat_rows])
            / oracle_total
        )
        result["rerank_oracle_sequence_relaxed_no_constants"] = float(
            np.sum(
                [
                    row["rerank_oracle_sequence_relaxed_no_constants_count"]
                    for row in rerank_stat_rows
                ]
            )
            / oracle_total
        )
        result["rerank_pair_oracle_sequence_exact"] = float(
            np.sum([row["rerank_pair_oracle_sequence_exact_count"] for row in rerank_stat_rows])
            / oracle_total
        )
        result["rerank_pair_oracle_sequence_relaxed_no_constants"] = float(
            np.sum(
                [
                    row["rerank_pair_oracle_sequence_relaxed_no_constants_count"]
                    for row in rerank_stat_rows
                ]
            )
            / oracle_total
        )
        result["rerank_unique_candidate_avg"] = float(
            np.sum([row["rerank_unique_candidate_total"] for row in rerank_stat_rows])
            / oracle_total
        )
        result["rerank_unique_drift_avg"] = float(
            np.sum([row["rerank_unique_drift_total"] for row in rerank_stat_rows])
            / oracle_total
        )
        result["rerank_unique_diffusion_avg"] = float(
            np.sum([row["rerank_unique_diffusion_total"] for row in rerank_stat_rows])
            / oracle_total
        )
        result["rerank_unique_paired_candidate_avg"] = float(
            np.sum(
                [row["rerank_unique_paired_candidate_total"] for row in rerank_stat_rows]
            )
            / oracle_total
        )
        debug_rows = []
        oracle_miss_rows = []
        oracle_rescue_rows = []
        residual_debug_rows = []
        for row in rerank_stat_rows:
            debug_rows.extend(row["rerank_debug"])
            oracle_miss_rows.extend(row["rerank_oracle_misses"])
            oracle_rescue_rows.extend(row["rerank_oracle_rescue_cases"])
            residual_debug_rows.extend(row["rerank_residual_debug"])
        result["rerank_debug"] = debug_rows
        result["rerank_oracle_misses"] = oracle_miss_rows
        result["rerank_oracle_rescue_cases"] = oracle_rescue_rows
        result["rerank_residual_debug"] = residual_debug_rows
    result["examples"] = examples[:5]
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    params = configure_gensr(args)
    env = build_env(params)

    with open(args.train_data, "rb") as f:
        train_pool = pickle.load(f)
    install_sde_pool(env, train_pool)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    if args.load_checkpoint is not None:
        load_checkpoint_from_path(
            trainer,
            args.load_checkpoint,
            requires_grad=not args.eval_only,
        )

    print(f"loaded_train_samples={len(train_pool)}")
    if args.eval_only:
        print("phase=train_skipped")
    else:
        print("phase=train")
        train_steps(trainer, args.train_steps)
        if args.save_checkpoint is not None:
            save_checkpoint_to_path(
                trainer,
                args.save_checkpoint,
                include_optimizer=args.checkpoint_include_optimizer,
            )

    print("phase=generate_eval")
    eval_samples = generate_eval_samples(args, env, params)
    print(f"generated_eval_samples={len(eval_samples)}")

    print("phase=evaluate")
    fingerprint_config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        active_paths=args.active_paths,
    )
    sample_temperatures = parse_temperature_list(
        args.sample_temperatures,
        args.sample_temperature,
    )
    rerank_component_weights = parse_component_weights(args.rerank_component_weights)
    rerank_constant_values = parse_constant_values(args.rerank_constant_values)
    metrics = evaluate(
        trainer,
        eval_samples,
        args.batch_size,
        args.eval_seed + 17,
        args.min_generated_len,
        args.constrained_beam_size,
        args.rerank_candidates,
        args.rerank_topk_from_beam,
        args.rerank_score,
        rerank_component_weights,
        rerank_constant_values,
        args.rerank_tie_epsilon,
        args.rerank_tie_break,
        fingerprint_config,
        args.rerank_debug_topk,
        args.sample_candidates,
        sample_temperatures,
        args.sample_top_k,
        args.sample_top_p,
        args.pair_drift_diffusion_candidates,
        args.pair_drift_topk,
        args.pair_diffusion_topk,
        args.rerank_residual_debug_json is not None,
    )
    if args.rerank_residual_debug_json is not None:
        write_residual_debug_json(
            args.rerank_residual_debug_json,
            {
                "metadata": {
                    "eval_samples": args.eval_samples,
                    "batch_size": args.batch_size,
                    "eval_seed": args.eval_seed,
                    "n_paths": args.n_paths,
                    "active_paths": args.active_paths,
                    "n_steps": args.n_steps,
                    "rerank_score": args.rerank_score,
                    "rerank_constant_values": rerank_constant_values,
                    "rerank_tie_epsilon": args.rerank_tie_epsilon,
                    "rerank_tie_break": args.rerank_tie_break,
                    "pair_drift_diffusion_candidates": args.pair_drift_diffusion_candidates,
                    "pair_drift_topk": args.pair_drift_topk,
                    "pair_diffusion_topk": args.pair_diffusion_topk,
                },
                "samples": metrics.get("rerank_residual_debug", []),
            },
        )
        print(f"rerank_residual_debug_json={args.rerank_residual_debug_json}")
    for key, value in metrics.items():
        if key in {
            "examples",
            "rerank_debug",
            "rerank_oracle_misses",
            "rerank_oracle_rescue_cases",
            "rerank_residual_debug",
        }:
            continue
        print(f"{key}={value:.6f}")
    print("examples:")
    for ex in metrics["examples"]:
        print(f"truth: {ex['truth']}")
        print(f"pred : {ex['pred']}")

    def _fmt_distance(value):
        return "nan" if value is None or not np.isfinite(value) else f"{value:.6f}"

    def _fmt_constant(value):
        return "none" if value is None or not np.isfinite(value) else f"{value:.6g}"

    def _record_words(record):
        return "" if record is None else " ".join(record["words"])

    def _record_tokens(record, key):
        return "" if record is None else " ".join(record[key])

    def _record_value(record, key):
        return None if record is None else record[key]

    if "rerank_oracle_total" in metrics:
        print("oracle_case_summary:")
        print(f"total_samples={int(metrics['rerank_oracle_total'])}")
        print(
            "samples_with_any_oracle_candidate="
            f"{int(metrics['rerank_oracle_samples_any_count'])}"
        )
        print(
            "samples_where_selected_is_oracle="
            f"{int(metrics['rerank_oracle_samples_selected_count'])}"
        )
        print(
            "samples_where_oracle_exists_but_selected_misses="
            f"{int(metrics['rerank_oracle_samples_missed_count'])}"
        )
        print(
            "oracle_best_source_counts "
            f"beam={int(metrics['rerank_oracle_best_source_beam_count'])} "
            f"sampling={int(metrics['rerank_oracle_best_source_sampling_count'])} "
            f"pair={int(metrics['rerank_oracle_best_source_pair_count'])} "
            f"unknown={int(metrics['rerank_oracle_best_source_unknown_count'])}"
        )
        print(
            "oracle_any_source_counts "
            f"beam={int(metrics['rerank_oracle_any_source_beam_count'])} "
            f"sampling={int(metrics['rerank_oracle_any_source_sampling_count'])} "
            f"pair={int(metrics['rerank_oracle_any_source_pair_count'])} "
            f"unknown={int(metrics['rerank_oracle_any_source_unknown_count'])}"
        )
        print(
            "selected_hit_source_counts "
            f"beam={int(metrics['rerank_selected_hit_source_beam_count'])} "
            f"sampling={int(metrics['rerank_selected_hit_source_sampling_count'])} "
            f"pair={int(metrics['rerank_selected_hit_source_pair_count'])} "
            f"unknown={int(metrics['rerank_selected_hit_source_unknown_count'])}"
        )
        print("oracle_miss_type_summary:")
        print(
            "constant_mismatch="
            f"{int(metrics['rerank_oracle_miss_constant_mismatch_count'])}"
        )
        print(
            "same_diffusion_but_wrong_drift="
            f"{int(metrics['rerank_oracle_miss_same_diffusion_wrong_drift_count'])}"
        )
        print(
            "same_drift_but_wrong_diffusion="
            f"{int(metrics['rerank_oracle_miss_same_drift_wrong_diffusion_count'])}"
        )
        print(
            "both_drift_and_diffusion_wrong="
            f"{int(metrics['rerank_oracle_miss_both_sides_wrong_count'])}"
        )
        print(
            "oracle_lower_model_score_candidate="
            f"{int(metrics['rerank_oracle_miss_oracle_lower_model_score_count'])}"
        )
        print(
            "oracle_only_appears_from_pair="
            f"{int(metrics['rerank_oracle_miss_oracle_only_pair_count'])}"
        )
        print(
            "oracle_from_beam_or_sampling_but_score_misses="
            f"{int(metrics['rerank_oracle_miss_beam_or_sampling_score_miss_count'])}"
        )
        print(f"parse_failures={int(metrics['rerank_parse_failures'])}")
        print(f"fingerprint_failures={int(metrics['rerank_fingerprint_failures'])}")
        print("oracle_miss_rescue_summary:")
        print(
            "miss_cases_total="
            f"{int(metrics['rerank_oracle_miss_rescue_cases_total'])}"
        )
        print(
            "rescued_by_rolewise_constant_grid="
            f"{int(metrics['rerank_oracle_miss_rescued_by_rolewise_constant_grid'])}"
        )
        print(
            "still_missed_after_rolewise_constant_grid="
            f"{int(metrics['rerank_oracle_miss_still_missed_after_rolewise_constant_grid'])}"
        )
        print(
            "selected_exact_shared_constant="
            f"{int(metrics['rerank_selected_exact_shared_constant_count'])}"
        )
        print(
            "selected_exact_rolewise_constant="
            f"{int(metrics['rerank_selected_exact_rolewise_constant_count'])}"
        )
        print(
            "oracle_exact_shared_constant="
            f"{int(metrics['rerank_oracle_exact_shared_constant_count'])}"
        )
        print(
            "oracle_exact_rolewise_constant="
            f"{int(metrics['rerank_oracle_exact_rolewise_constant_count'])}"
        )
    if metrics.get("rerank_oracle_misses") is not None:
        print("oracle_miss_cases:")
        for miss in metrics["rerank_oracle_misses"]:
            selected = miss["selected"]
            oracle = miss["oracle"]
            print(f"sample_index={miss['sample_index']}")
            print(f"truth: {' '.join(miss['truth'])}")
            print(f"selected_sequence: {_record_words(selected)}")
            print(f"oracle_sequence: {_record_words(oracle)}")
            print(
                "sources "
                f"selected={selected['source'] if selected is not None else 'none'} "
                f"oracle={oracle['source']}"
            )
            print(
                "scores "
                f"selected={_fmt_distance(selected['distance'] if selected is not None else None)} "
                f"oracle={_fmt_distance(oracle['distance'])} "
                f"gap={_fmt_distance(miss['score_gap'])}"
            )
            print(
                "shared_constant_scores "
                f"selected="
                f"{_fmt_distance(_record_value(selected, 'shared_constant_distance'))} "
                f"oracle={_fmt_distance(oracle['shared_constant_distance'])}"
            )
            print(
                "best_constants "
                f"selected="
                f"{_fmt_constant(selected['best_constant'] if selected is not None else None)} "
                f"oracle={_fmt_constant(oracle['best_constant'])}"
            )
            print(
                "rolewise_best_constants "
                f"selected_drift="
                f"{_fmt_constant(_record_value(selected, 'best_drift_constant'))} "
                f"selected_diffusion="
                f"{_fmt_constant(_record_value(selected, 'best_diffusion_constant'))} "
                f"oracle_drift={_fmt_constant(oracle['best_drift_constant'])} "
                f"oracle_diffusion={_fmt_constant(oracle['best_diffusion_constant'])}"
            )
            print(
                "model_scores "
                f"selected="
                f"{selected['normalized_model_score'] if selected is not None else float('nan'):.6f} "
                f"oracle={oracle['normalized_model_score']:.6f} "
                f"oracle_lower={int(miss['oracle_lower_model_score'])}"
            )
            print(f"selected_drift: {_record_tokens(selected, 'drift_tokens')}")
            print(f"oracle_drift: {_record_tokens(oracle, 'drift_tokens')}")
            print(f"selected_diffusion: {_record_tokens(selected, 'diffusion_tokens')}")
            print(f"oracle_diffusion: {_record_tokens(oracle, 'diffusion_tokens')}")
            print(
                "segment_distances_selected "
                f"multi_u0_moments="
                f"{_fmt_distance(selected['multi_u0_moments_distance'] if selected is not None else None)} "
                f"active_kramers_moyal="
                f"{_fmt_distance(selected['active_kramers_moyal_distance'] if selected is not None else None)} "
                f"gaussian_weak_kernel="
                f"{_fmt_distance(selected['gaussian_weak_kernel_distance'] if selected is not None else None)}"
            )
            print(
                "segment_distances_oracle "
                f"multi_u0_moments={_fmt_distance(oracle['multi_u0_moments_distance'])} "
                f"active_kramers_moyal={_fmt_distance(oracle['active_kramers_moyal_distance'])} "
                f"gaussian_weak_kernel={_fmt_distance(oracle['gaussian_weak_kernel_distance'])}"
            )
            print(
                "shared_segment_distances_selected "
                f"active_kramers_moyal="
                f"{_fmt_distance(_record_value(selected, 'shared_constant_active_kramers_moyal_distance'))} "
                f"gaussian_weak_kernel="
                f"{_fmt_distance(_record_value(selected, 'shared_constant_gaussian_weak_kernel_distance'))}"
            )
            print(
                "shared_segment_distances_oracle "
                f"active_kramers_moyal="
                f"{_fmt_distance(oracle['shared_constant_active_kramers_moyal_distance'])} "
                f"gaussian_weak_kernel="
                f"{_fmt_distance(oracle['shared_constant_gaussian_weak_kernel_distance'])}"
            )
            print(
                "match_flags "
                f"shares_drift={int(miss['shares_drift'])} "
                f"shares_diffusion={int(miss['shares_diffusion'])} "
                f"constant_mismatch={int(miss['constant_mismatch'])} "
                f"miss_side={miss['miss_side']} "
                f"oracle_only_pair={int(miss['oracle_only_pair'])}"
            )
    if metrics.get("rerank_oracle_rescue_cases") is not None:
        print("oracle_miss_rescue_cases:")
        for rescue in metrics["rerank_oracle_rescue_cases"]:
            shared_selected = rescue["shared_selected"]
            rolewise_selected = rescue["rolewise_selected"]
            oracle = rescue["oracle"]
            print(f"sample_index={rescue['sample_index']}")
            print(f"truth: {' '.join(rescue['truth'])}")
            print(f"shared_selected_sequence: {_record_words(shared_selected)}")
            print(f"rolewise_selected_sequence: {_record_words(rolewise_selected)}")
            print(f"oracle_sequence: {_record_words(oracle)}")
            print(
                "ranks "
                f"shared_oracle_rank={rescue['shared_oracle_rank']} "
                f"rolewise_oracle_rank={rescue['rolewise_oracle_rank']} "
                f"oracle_rank_improved={int(rescue['oracle_rank_improved'])} "
                f"rescued={int(rescue['rescued'])}"
            )
            print(
                "rolewise_scores "
                f"selected={_fmt_distance(_record_value(rolewise_selected, 'distance'))} "
                f"oracle={_fmt_distance(oracle['distance'])}"
            )
            print(
                "shared_scores "
                f"selected={_fmt_distance(_record_value(shared_selected, 'shared_constant_distance'))} "
                f"oracle={_fmt_distance(oracle['shared_constant_distance'])}"
            )
            print(
                "rolewise_best_constants "
                f"selected_drift="
                f"{_fmt_constant(_record_value(rolewise_selected, 'best_drift_constant'))} "
                f"selected_diffusion="
                f"{_fmt_constant(_record_value(rolewise_selected, 'best_diffusion_constant'))} "
                f"oracle_drift={_fmt_constant(oracle['best_drift_constant'])} "
                f"oracle_diffusion={_fmt_constant(oracle['best_diffusion_constant'])}"
            )
            print(
                "rolewise_segment_distances_selected "
                f"active_kramers_moyal="
                f"{_fmt_distance(_record_value(rolewise_selected, 'active_kramers_moyal_distance'))} "
                f"gaussian_weak_kernel="
                f"{_fmt_distance(_record_value(rolewise_selected, 'gaussian_weak_kernel_distance'))}"
            )
            print(
                "rolewise_segment_distances_oracle "
                f"active_kramers_moyal="
                f"{_fmt_distance(oracle['active_kramers_moyal_distance'])} "
                f"gaussian_weak_kernel="
                f"{_fmt_distance(oracle['gaussian_weak_kernel_distance'])}"
            )
    if args.rerank_debug_topk > 0 and metrics.get("rerank_debug"):
        print("rerank_debug:")

        for debug in metrics["rerank_debug"]:
            print(f"sample_index={debug['sample_index']}")
            print(f"truth: {' '.join(debug['truth'])}")
            print(f"greedy: {' '.join(debug['greedy'])}")
            print(f"constrained_beam: {' '.join(debug['constrained_beam'])}")
            print(f"tie_break={debug['tie_break']}")
            print(f"tie_epsilon={_fmt_distance(debug['tie_epsilon'])}")
            print(f"oracle_rank={debug['oracle_rank']}")
            print(f"oracle_rank_constant_1={debug['oracle_baseline_rank']}")
            print(f"top_ranked_distance={_fmt_distance(debug['top_distance'])}")
            if debug["top_candidate"] is not None:
                print(
                    "top_ranked_best_constant="
                    f"{_fmt_constant(debug['top_candidate']['best_constant'])}"
                )
            print(f"oracle_distance={_fmt_distance(debug['oracle_distance'])}")
            print(
                "oracle_distance_constant_1="
                f"{_fmt_distance(debug['oracle_baseline_distance'])}"
            )
            print(
                "oracle_top_distance_delta="
                f"{_fmt_distance(debug['oracle_top_distance_delta'])}"
            )
            if debug["tie_selected_candidate"] is not None:
                selected = debug["tie_selected_candidate"]
                print(
                    "tie_selected_candidate "
                    f"source={selected['source']} "
                    f"is_pair={int(selected['source'] == 'pair')} "
                    f"state_dependent_drift="
                    f"{int(selected['state_dependent_drift'])} "
                    f"best_constant={_fmt_constant(selected['best_constant'])} "
                    f"distance={_fmt_distance(selected['distance'])} "
                    f"active_kramers_moyal_distance="
                    f"{_fmt_distance(selected['active_kramers_moyal_distance'])} "
                    f"gaussian_weak_kernel_distance="
                    f"{_fmt_distance(selected['gaussian_weak_kernel_distance'])} "
                    f"model_score={selected['model_score']:.6f} "
                    f"normalized_model_score="
                    f"{selected['normalized_model_score']:.6f} "
                    f"exact={int(selected['exact'])} "
                    f"relaxed_no_constants={int(selected['relaxed'])}"
                )
            if debug.get("tie_group"):
                print("tie_group_candidates:")
                for rank, candidate in enumerate(debug["tie_group"], start=1):
                    print(
                        f"tie_group_rank={rank} "
                        f"source={candidate['source']} "
                        f"is_pair={int(candidate['source'] == 'pair')} "
                        f"state_dependent_drift="
                        f"{int(candidate['state_dependent_drift'])} "
                        f"best_constant={_fmt_constant(candidate['best_constant'])} "
                        f"distance={_fmt_distance(candidate['distance'])} "
                        f"active_kramers_moyal_distance="
                        f"{_fmt_distance(candidate['active_kramers_moyal_distance'])} "
                        f"gaussian_weak_kernel_distance="
                        f"{_fmt_distance(candidate['gaussian_weak_kernel_distance'])} "
                        f"model_score={candidate['model_score']:.6f} "
                        f"normalized_model_score="
                        f"{candidate['normalized_model_score']:.6f} "
                        f"exact={int(candidate['exact'])} "
                        f"relaxed_no_constants={int(candidate['relaxed'])}"
                    )
            if debug["oracle_candidate"] is not None:
                oracle = debug["oracle_candidate"]
                print(
                    "oracle_candidate "
                    f"source={oracle['source']} "
                    f"state_dependent_drift="
                    f"{int(oracle['state_dependent_drift'])} "
                    f"best_constant={_fmt_constant(oracle['best_constant'])} "
                    f"distance={_fmt_distance(oracle['distance'])} "
                    f"baseline_distance="
                    f"{_fmt_distance(oracle['baseline_distance'])} "
                    f"full_distance={_fmt_distance(oracle['full_distance'])} "
                    f"multi_u0_moments_distance="
                    f"{_fmt_distance(oracle['multi_u0_moments_distance'])} "
                    f"active_kramers_moyal_distance="
                    f"{_fmt_distance(oracle['active_kramers_moyal_distance'])} "
                    f"gaussian_weak_kernel_distance="
                    f"{_fmt_distance(oracle['gaussian_weak_kernel_distance'])}"
                )
                print(f"oracle_drift_tokens: {' '.join(oracle['drift_tokens'])}")
                print(f"oracle_diffusion_tokens: {' '.join(oracle['diffusion_tokens'])}")
            if debug.get("candidates_before_oracle"):
                print("candidates_before_oracle:")
                for item in debug["candidates_before_oracle"]:
                    candidate = item["record"]
                    deltas = item["segment_deltas"]
                    weighted_deltas = item["weighted_segment_deltas"]
                    print(
                        f"pre_oracle_rank={item['rank']} "
                        f"source={candidate['source']} "
                        f"is_pair={int(candidate['source'] == 'pair')} "
                        f"state_dependent_drift="
                        f"{int(candidate['state_dependent_drift'])} "
                        f"best_constant={_fmt_constant(candidate['best_constant'])} "
                        f"distance={_fmt_distance(candidate['distance'])} "
                        f"distance_delta="
                        f"{_fmt_distance(item['distance_delta'])} "
                        f"multi_u0_moments_delta="
                        f"{_fmt_distance(deltas['multi_u0_moments'])} "
                        f"active_kramers_moyal_delta="
                        f"{_fmt_distance(deltas['active_kramers_moyal'])} "
                        f"gaussian_weak_kernel_delta="
                        f"{_fmt_distance(deltas['gaussian_weak_kernel'])} "
                        f"dominant_gap_segment="
                        f"{item['dominant_gap_segment'] or 'none'} "
                        f"model_score={candidate['model_score']:.6f} "
                        f"normalized_model_score="
                        f"{candidate['normalized_model_score']:.6f} "
                        f"exact={int(candidate['exact'])} "
                        f"relaxed_no_constants={int(candidate['relaxed'])} "
                        f"failure_reason={candidate['failure_reason'] or 'none'}"
                    )
                    print(
                        "pre_oracle_weighted_deltas "
                        f"multi_u0_moments="
                        f"{_fmt_distance(weighted_deltas['multi_u0_moments'])} "
                        f"active_kramers_moyal="
                        f"{_fmt_distance(weighted_deltas['active_kramers_moyal'])} "
                        f"gaussian_weak_kernel="
                        f"{_fmt_distance(weighted_deltas['gaussian_weak_kernel'])}"
                    )
                    print(
                        "pre_oracle_segment_distances "
                        f"multi_u0_moments="
                        f"{_fmt_distance(candidate['multi_u0_moments_distance'])} "
                        f"active_kramers_moyal="
                        f"{_fmt_distance(candidate['active_kramers_moyal_distance'])} "
                        f"gaussian_weak_kernel="
                        f"{_fmt_distance(candidate['gaussian_weak_kernel_distance'])}"
                    )
                    print(f"pre_oracle_drift_tokens: {' '.join(candidate['drift_tokens'])}")
                    print(
                        "pre_oracle_diffusion_tokens: "
                        f"{' '.join(candidate['diffusion_tokens'])}"
                    )
            for rank, candidate in enumerate(debug["candidates"], start=1):
                print(
                    f"candidate_rank={rank} "
                    f"source={candidate['source']} "
                    f"is_pair={int(candidate['source'] == 'pair')} "
                    f"state_dependent_drift="
                    f"{int(candidate['state_dependent_drift'])} "
                    f"best_constant={_fmt_constant(candidate['best_constant'])} "
                    f"distance={_fmt_distance(candidate['distance'])} "
                    f"baseline_distance="
                    f"{_fmt_distance(candidate['baseline_distance'])} "
                    f"full_distance={_fmt_distance(candidate['full_distance'])} "
                    f"multi_u0_moments_distance="
                    f"{_fmt_distance(candidate['multi_u0_moments_distance'])} "
                    f"active_kramers_moyal_distance="
                    f"{_fmt_distance(candidate['active_kramers_moyal_distance'])} "
                    f"gaussian_weak_kernel_distance="
                    f"{_fmt_distance(candidate['gaussian_weak_kernel_distance'])} "
                    f"model_score={candidate['model_score']:.6f} "
                    f"normalized_model_score={candidate['normalized_model_score']:.6f} "
                    f"exact={int(candidate['exact'])} "
                    f"relaxed_no_constants={int(candidate['relaxed'])} "
                    f"failure_reason={candidate['failure_reason'] or 'none'}"
                )
                print(f"drift_tokens: {' '.join(candidate['drift_tokens'])}")
                print(f"diffusion_tokens: {' '.join(candidate['diffusion_tokens'])}")


if __name__ == "__main__":
    main()
