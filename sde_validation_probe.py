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
from simulator_sde import SDESystem, sample_sde_system
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
    token_sets = _sde_token_sets(env)
    eos_id = decoder.eos_index
    pad_id = decoder.pad_index
    device = src_enc.device
    rows = []

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
        best = max(
            candidates,
            key=lambda item: item["score"] / (len(item["tokens"]) ** length_penalty),
        )
        tokens = best["tokens"]
        if tokens[-1] != eos_id:
            tokens = tokens + [eos_id]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = eos_id
        row = torch.full((max_len,), pad_id, dtype=torch.long, device=device)
        row[: len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        rows.append(row)

    return torch.stack(rows, dim=0)


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
    return (
        scores,
        y,
        loss,
        generations.transpose(0, 1),
        gen_len,
        constrained_generations.transpose(0, 1),
        constrained_len,
        constrained_beam,
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
) -> dict:
    rows = []
    seq_rows = []
    examples = []
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
            ) = (
                prior_scores_for_joint_batch(
                    trainer,
                    batch,
                    min_generated_len,
                    constrained_beam_size,
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
    metrics = evaluate(
        trainer,
        eval_samples,
        args.batch_size,
        args.eval_seed + 17,
        args.min_generated_len,
        args.constrained_beam_size,
    )
    for key, value in metrics.items():
        if key == "examples":
            continue
        print(f"{key}={value:.6f}")
    print("examples:")
    for ex in metrics["examples"]:
        print(f"truth: {ex['truth']}")
        print(f"pred : {ex['pred']}")


if __name__ == "__main__":
    main()
