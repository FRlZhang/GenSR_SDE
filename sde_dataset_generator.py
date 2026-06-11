#!/usr/bin/env python3
"""Generate GenSR-compatible SDE samples with unified fingerprints."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np

from parsers import get_parser
from sde_fingerprint import FingerprintConfig
from simulator_sde import SDESystem, sample_sde_system, solve_fingerprint
from symbolicregression.envs.environment import FunctionEnvironment
from symbolicregression.envs.generators import string_to_node


SEPARATOR_TOKEN = "SPECIAL"


def clean_tokens(token_list: list[str]) -> list[str]:
    return [w.split(".")[0] if w.endswith(".0") else w for w in token_list]


def skeletonize(tokens: list[str]) -> list[str]:
    result = []
    for token in tokens:
        try:
            float(token)
            result.append("CONSTANT")
        except ValueError:
            result.append(token)
    return result


def constants_to_token(tokens: list[str]) -> list[str]:
    converted = []
    for token in tokens:
        try:
            float(token)
            converted.append("CONSTANT")
        except ValueError:
            converted.append(token)
    return converted


def normalize_expression_for_gensr(expr: str) -> str:
    return re.sub(r"\bu\b", "x_0", expr)


def encode_sde(system: SDESystem, env: FunctionEnvironment, params):
    drift_expr = normalize_expression_for_gensr(system.drift_str)
    diffusion_expr = normalize_expression_for_gensr(system.diffusion_str)
    drift_tokens = constants_to_token(
        clean_tokens(string_to_node(drift_expr, params).prefix().split(","))
    )
    diffusion_tokens = constants_to_token(
        clean_tokens(string_to_node(diffusion_expr, params).prefix().split(","))
    )
    words = drift_tokens + [SEPARATOR_TOKEN] + diffusion_tokens
    skeleton_words = skeletonize(words)
    tree_encoded = env.word_to_idx([words], float_input=False)[0]
    skeleton_tree_encoded = env.word_to_idx([skeleton_words], float_input=False)[0]
    return words, skeleton_words, tree_encoded, skeleton_tree_encoded


def build_sample(
    system: SDESystem,
    env: FunctionEnvironment,
    params,
    config: FingerprintConfig,
    seed: int,
) -> dict | None:
    x_data, y_data, meta = solve_fingerprint(system, config, seed)
    if x_data is None or y_data is None:
        return None
    if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
        return None

    try:
        words, skeleton_words, tree_encoded, skeleton_tree_encoded = encode_sde(
            system, env, params
        )
    except Exception:
        return None

    fingerprint_config = meta["fingerprint_config"]
    sample = {
        "x_to_fit": x_data.astype(np.float32),
        "y_to_fit": y_data.astype(np.float32),
        "tree_encoded": tree_encoded,
        "skeleton_tree_encoded": skeleton_tree_encoded,
        "tree": f"{system.drift_str} | {system.diffusion_str}",
        "skeleton_tree": " ".join(skeleton_words),
        "infos": {
            "problem_type": "sde",
            "fingerprint_type": fingerprint_config["kind"],
            "fingerprint_length": fingerprint_config["fingerprint_length"],
            "fingerprint_schema": meta["part_slices"],
            "fingerprint_normalization": meta["normalization"],
            "fingerprint_config": fingerprint_config,
            "equation": system.equation,
            "drift": system.drift_str,
            "diffusion": system.diffusion_str,
            "n_input_points": x_data.shape[0],
            "input_sequence_length": x_data.shape[0],
            "d_in": 1,
            "d_out": 1,
        },
    }
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/sde_train_dataset.pkl"))
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--active-paths", type=int, default=2000)
    parser.add_argument("--n-kernels", type=int, default=24)
    parser.add_argument("--kernel-bandwidth", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    config = FingerprintConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        t_final=args.t_final,
        active_paths=args.active_paths,
        n_kernels=args.n_kernels,
        kernel_bandwidth=args.kernel_bandwidth,
    )
    if config.to_dict()["fingerprint_length"] > 200:
        raise ValueError("Fingerprint exceeds GenSR default max_len=200.")

    print("Initializing GenSR environment...")
    params = get_parser().parse_args([])
    params.min_input_dimension = 1
    params.max_input_dimension = 1
    params.min_output_dimension = 1
    params.max_output_dimension = 1
    params.max_len = max(params.max_len, config.to_dict()["fingerprint_length"])
    env = FunctionEnvironment(params)

    samples = []
    attempts = 0
    max_attempts = args.num_samples * 100
    print(
        f"Generating {args.num_samples} SDE fingerprint samples "
        f"(length={config.to_dict()['fingerprint_length']})..."
    )

    while len(samples) < args.num_samples and attempts < max_attempts:
        attempts += 1
        system = sample_sde_system(py_rng)
        seed = int(np_rng.integers(0, 2**31 - 1))
        sample = build_sample(system, env, params, config, seed)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) == 1 or len(samples) % 20 == 0:
            print(
                f"[{len(samples):04d}/{args.num_samples}] "
                f"{sample['infos']['equation']}"
            )

    if len(samples) < args.num_samples:
        raise RuntimeError(
            f"Only generated {len(samples)} samples after {attempts} attempts."
        )

    os.makedirs(args.output.parent, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(samples, f)

    print("=" * 60)
    print(f"Saved {len(samples)} SDE fingerprint samples to {args.output}")
    print(f"Attempts: {attempts}")
    print(f"Fingerprint schema: {samples[0]['infos']['fingerprint_schema']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
