#!/usr/bin/env python3
"""Small SDE simulation helpers used by the GenSR-SDE data pipeline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import sympy as sp

from sde_fingerprint import (
    FingerprintConfig,
    compute_unified_fingerprint,
    fingerprint_to_xy,
    simulate_paths,
)


@dataclass(frozen=True)
class SDESystem:
    drift_str: str
    diffusion_str: str

    @property
    def equation(self) -> str:
        return f"du = ({self.drift_str})dt + ({self.diffusion_str})dW"


def lambdify_sde(system: SDESystem):
    u, t = sp.symbols("u t")
    drift_func = sp.lambdify((u, t), sp.sympify(system.drift_str), "numpy")
    diffusion_func = sp.lambdify((u, t), sp.sympify(system.diffusion_str), "numpy")

    def drift(x, time):
        return np.asarray(drift_func(x, time), dtype=np.float64) + np.zeros_like(x)

    def diffusion(x, time):
        return np.asarray(diffusion_func(x, time), dtype=np.float64) + np.zeros_like(x)

    return drift, diffusion


def sample_sde_system(rng: random.Random | None = None) -> SDESystem:
    rng = rng or random
    u_terms = [
        "1.0",
        "u",
        "u**2",
        "u*(1-u)",
        "sin(u)",
    ]
    time_terms = ["0.0"]
    diffusion_terms = [
        "1.0",
        "u",
        "abs(u)",
        "sqrt(abs(u))",
        "1.0 + abs(u)",
    ]

    a = rng.uniform(-2.0, 2.0)
    while abs(a) < 0.15:
        a = rng.uniform(-2.0, 2.0)
    b = rng.uniform(-1.0, 1.0)
    c = rng.uniform(0.15, 1.2)
    d = rng.uniform(0.05, 0.8)

    u_term = rng.choice(u_terms)
    t_term = rng.choice(time_terms)
    drift = f"{a:.6g}*{u_term}"
    if t_term != "0.0":
        drift = f"{drift} + {b:.6g}*{t_term}"

    diffusion_term = rng.choice(diffusion_terms)
    if diffusion_term == "1.0 + abs(u)":
        diffusion = f"{c:.6g} + {d:.6g}*abs(u)"
    else:
        diffusion = f"{c:.6g}*{diffusion_term}"
    return SDESystem(drift, diffusion)


def solve_moments(
    system: SDESystem,
    u0: float = 1.0,
    t_final: float = 1.0,
    n_steps: int = 100,
    n_paths: int = 2000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    rng = np.random.default_rng(seed)
    drift, diffusion = lambdify_sde(system)
    paths = simulate_paths(drift, diffusion, u0, rng, n_paths, n_steps, t_final)
    if paths is None:
        return None, None
    t_grid = np.linspace(0.0, t_final, n_steps + 1, dtype=np.float32).reshape(-1, 1)
    y = np.column_stack([paths.mean(axis=0), paths.var(axis=0)]).astype(np.float32)
    return t_grid, y


def solve_fingerprint(
    system: SDESystem,
    config: FingerprintConfig | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, dict]:
    config = config or FingerprintConfig()
    rng = np.random.default_rng(seed)
    drift, diffusion = lambdify_sde(system)
    fingerprint, meta = compute_unified_fingerprint(drift, diffusion, rng, config)
    if fingerprint is None:
        return None, None, meta
    x_to_fit, y_to_fit = fingerprint_to_xy(fingerprint)
    return x_to_fit, y_to_fit, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py_rng = random.Random(args.seed)
    system = sample_sde_system(py_rng)
    config = FingerprintConfig(n_paths=args.n_paths, n_steps=args.n_steps)
    x_to_fit, y_to_fit, meta = solve_fingerprint(system, config, args.seed)
    print(system.equation)
    if x_to_fit is None:
        print("fingerprint failed:", meta)
        return
    print("x_to_fit:", x_to_fit.shape)
    print("y_to_fit:", y_to_fit.shape)
    print("fingerprint length:", meta["fingerprint_config"]["fingerprint_length"])
    print("part slices:", meta["part_slices"])
    print("first values:", y_to_fit[:8, 0])


if __name__ == "__main__":
    main()
