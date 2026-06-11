#!/usr/bin/env python3
"""
Test whether moment fingerprints can distinguish 1D SDEs.

The script generates random SDEs, simulates many trajectories, builds several
deterministic fingerprints from the simulated distribution, and checks nearest
neighbour collisions between different symbolic structures.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Template:
    name: str
    fn: object
    expr: str


DRIFT_TEMPLATES = [
    Template("const", lambda x, t, a, b: a + 0.0 * x, "{a:.3g}"),
    Template("linear_x", lambda x, t, a, b: a * x, "{a:.3g}*x"),
    Template("affine_x", lambda x, t, a, b: a * x + b, "{a:.3g}*x + {b:.3g}"),
    Template("logistic", lambda x, t, a, b: a * x * (1.0 - x), "{a:.3g}*x*(1-x)"),
    Template("quadratic", lambda x, t, a, b: a * x * x, "{a:.3g}*x^2"),
    Template("sin_x", lambda x, t, a, b: a * np.sin(x), "{a:.3g}*sin(x)"),
    Template("time_linear", lambda x, t, a, b: a * x + b * t, "{a:.3g}*x + {b:.3g}*t"),
]

DIFFUSION_TEMPLATES = [
    Template("const", lambda x, t, c, d: np.abs(c) + 0.0 * x, "{c:.3g}"),
    Template("linear_x", lambda x, t, c, d: np.abs(c) * np.abs(x), "{c:.3g}*|x|"),
    Template("sqrt_abs_x", lambda x, t, c, d: np.abs(c) * np.sqrt(np.abs(x) + 1e-6), "{c:.3g}*sqrt(|x|)"),
    Template("affine_abs", lambda x, t, c, d: np.abs(c) + np.abs(d) * np.abs(x), "{c:.3g}+{d:.3g}*|x|"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sdes", type=int, default=500)
    parser.add_argument("--n-paths", type=int, default=600)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--local-dt", type=float, default=1e-2)
    parser.add_argument("--n-local-times", type=int, default=3)
    parser.add_argument("--probe-x0s", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--bin-range", type=str, default="-2.0,2.0")
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def sample_coeff(rng: np.random.Generator, low: float, high: float, min_abs: float) -> float:
    for _ in range(100):
        value = rng.uniform(low, high)
        if abs(value) >= min_abs:
            return float(value)
    return float(min_abs)


def simulate_sde(
    drift_template: Template,
    diffusion_template: Template,
    coeffs: tuple[float, float, float, float],
    u0: float,
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    t_final: float,
) -> np.ndarray | None:
    a, b, c, d = coeffs
    dt = t_final / n_steps
    sqrt_dt = math.sqrt(dt)
    x = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    x[:, 0] = u0

    for i in range(n_steps):
        t = i * dt
        xi = x[:, i]
        drift = drift_template.fn(xi, t, a, b)
        diffusion = diffusion_template.fn(xi, t, c, d)
        d_w = rng.normal(0.0, sqrt_dt, size=n_paths)
        nxt = xi + drift * dt + diffusion * d_w
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e6:
            return None
        x[:, i + 1] = nxt
    return x


def moment_features(paths: np.ndarray, include_high_moments: bool) -> np.ndarray:
    mean = paths.mean(axis=0)
    centered = paths - mean[None, :]
    var = (centered * centered).mean(axis=0)
    parts = [mean, var]
    if include_high_moments:
        std = np.sqrt(var + 1e-12)
        skew = (centered**3).mean(axis=0) / (std**3 + 1e-12)
        kurt = (centered**4).mean(axis=0) / (std**4 + 1e-12)
        parts.extend([skew, kurt])
    return np.concatenate(parts)


def local_kramers_moyal_features(
    drift_template: Template,
    diffusion_template: Template,
    coeffs: tuple[float, float, float, float],
    probe_x0s: tuple[float, ...],
    probe_times: np.ndarray,
    rng: np.random.Generator,
    n_paths: int,
    dt: float,
) -> np.ndarray | None:
    """Estimate short-time conditional increment moments at probe states.

    For Brownian SDEs, E[dX | x,t] / dt estimates drift and
    Var[dX | x,t] / dt estimates diffusion^2.
    """
    a, b, c, d = coeffs
    sqrt_dt = math.sqrt(dt)
    features = []
    for t in probe_times:
        for x0 in probe_x0s:
            x = np.full(n_paths, x0, dtype=np.float64)
            drift = drift_template.fn(x, float(t), a, b)
            diffusion = diffusion_template.fn(x, float(t), c, d)
            d_w = rng.normal(0.0, sqrt_dt, size=n_paths)
            delta = drift * dt + diffusion * d_w
            if not np.all(np.isfinite(delta)):
                return None
            features.append(delta.mean() / dt)
            features.append(delta.var() / dt)
    return np.asarray(features, dtype=np.float64)


def oracle_generator_features(
    drift_template: Template,
    diffusion_template: Template,
    coeffs: tuple[float, float, float, float],
    probe_x0s: tuple[float, ...],
    probe_times: np.ndarray,
) -> np.ndarray:
    """Use exact drift and diffusion^2 at probe states as an identifiability upper bound."""
    a, b, c, d = coeffs
    features = []
    for t in probe_times:
        for x0 in probe_x0s:
            x = np.asarray([x0], dtype=np.float64)
            drift = drift_template.fn(x, float(t), a, b)[0]
            diffusion = diffusion_template.fn(x, float(t), c, d)[0]
            features.append(float(drift))
            features.append(float(diffusion * diffusion))
    return np.asarray(features, dtype=np.float64)


def binned_transition_features(
    paths: np.ndarray,
    t_final: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Estimate local increment moments from naturally visited states."""
    n_steps = paths.shape[1] - 1
    dt = t_final / n_steps
    x_t = paths[:, :-1].reshape(-1)
    dx = (paths[:, 1:] - paths[:, :-1]).reshape(-1)
    scaled_mean = dx / dt
    scaled_var = (dx * dx) / dt

    bin_idx = np.digitize(x_t, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    total = max(1, x_t.size)
    features = []
    global_mean = float(np.mean(scaled_mean))
    global_second = float(np.mean(scaled_var))

    for idx in range(n_bins):
        mask = bin_idx == idx
        coverage = float(mask.mean())
        if np.any(mask):
            mean_est = float(np.mean(scaled_mean[mask]))
            second_est = float(np.mean(scaled_var[mask]))
        else:
            mean_est = global_mean
            second_est = global_second
        features.extend([coverage, mean_est, second_est])

    outside = float(((x_t < bin_edges[0]) | (x_t > bin_edges[-1])).sum() / total)
    features.append(outside)
    return np.asarray(features, dtype=np.float64)


def binned_local_linear_features(
    paths: np.ndarray,
    t_final: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Fit local linear conditional moment regressions inside state bins."""
    n_paths, n_observations = paths.shape
    n_steps = n_observations - 1
    dt = t_final / n_steps
    x_t = paths[:, :-1].reshape(-1)
    dx = (paths[:, 1:] - paths[:, :-1]).reshape(-1)
    t_grid = np.linspace(0.0, t_final - dt, n_steps)
    t_t = np.tile(t_grid, n_paths)
    y_drift = dx / dt
    y_second = (dx * dx) / dt

    bin_idx = np.digitize(x_t, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    total = max(1, x_t.size)
    features = []

    def fit_coefficients(mask: np.ndarray, center: float, y: np.ndarray) -> np.ndarray:
        if mask.sum() < 8:
            return np.zeros(3, dtype=np.float64)
        design = np.column_stack(
            [
                np.ones(mask.sum()),
                x_t[mask] - center,
                t_t[mask] - np.mean(t_t[mask]),
            ]
        )
        coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        return coef.astype(np.float64)

    for idx in range(n_bins):
        left, right = bin_edges[idx], bin_edges[idx + 1]
        center = 0.5 * (left + right)
        mask = bin_idx == idx
        coverage = float(mask.mean())
        drift_coef = fit_coefficients(mask, center, y_drift)
        second_coef = fit_coefficients(mask, center, y_second)
        features.extend([coverage, *drift_coef, *second_coef])

    outside = float(((x_t < bin_edges[0]) | (x_t > bin_edges[-1])).sum() / total)
    features.append(outside)
    return np.asarray(features, dtype=np.float64)


def standardize(features: np.ndarray) -> np.ndarray:
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (features - mu) / sigma


def nearest_neighbour_stats(
    features: np.ndarray,
    structure_labels: list[tuple[str, str]],
    formula_labels: list[str],
    top_k: int,
) -> dict:
    z = standardize(features)
    sq_norm = np.sum(z * z, axis=1, keepdims=True)
    d2 = sq_norm + sq_norm.T - 2.0 * (z @ z.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)
    dist = np.sqrt(d2 / z.shape[1])

    nn = np.argmin(dist, axis=1)
    nn_dist = dist[np.arange(len(nn)), nn]
    same_structure = np.array(
        [structure_labels[i] == structure_labels[j] for i, j in enumerate(nn)]
    )

    diff_pairs = []
    triu_i, triu_j = np.triu_indices(len(features), k=1)
    pair_dist = dist[triu_i, triu_j]
    order = np.argsort(pair_dist)
    for idx in order:
        i = int(triu_i[idx])
        j = int(triu_j[idx])
        if structure_labels[i] != structure_labels[j]:
            diff_pairs.append((float(pair_dist[idx]), i, j))
            if len(diff_pairs) >= top_k:
                break

    return {
        "nn_same_structure_rate": float(same_structure.mean()),
        "nn_distance_median": float(np.median(nn_dist)),
        "nn_distance_p05": float(np.quantile(nn_dist, 0.05)),
        "nn_distance_p95": float(np.quantile(nn_dist, 0.95)),
        "closest_diff_pairs": diff_pairs,
        "formula_labels": formula_labels,
        "structure_labels": structure_labels,
    }


def print_stats(name: str, stats: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"nearest-neighbour same-structure rate: {stats['nn_same_structure_rate']:.3f}")
    print(
        "nearest-neighbour normalized distance: "
        f"p05={stats['nn_distance_p05']:.4f}, "
        f"median={stats['nn_distance_median']:.4f}, "
        f"p95={stats['nn_distance_p95']:.4f}"
    )
    print("closest different-structure pairs:")
    for rank, (dist, i, j) in enumerate(stats["closest_diff_pairs"], start=1):
        labels = stats["formula_labels"]
        structs = stats["structure_labels"]
        print(f"  {rank:02d}. dist={dist:.5f}")
        print(f"      [{i}] {structs[i]} :: {labels[i]}")
        print(f"      [{j}] {structs[j]} :: {labels[j]}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fixed_ev = []
    fixed_high = []
    multi_ev = []
    local_km = []
    combined_multi_local = []
    oracle_km = []
    binned_km = []
    binned_local_linear = []
    structure_labels = []
    formula_labels = []
    probe_x0s = tuple(float(x) for x in args.probe_x0s.split(","))
    probe_times = np.linspace(0.0, args.t_final, args.n_local_times)
    bin_min, bin_max = (float(x) for x in args.bin_range.split(","))
    bin_edges = np.linspace(bin_min, bin_max, args.n_bins + 1)

    attempts = 0
    while len(formula_labels) < args.n_sdes:
        attempts += 1
        if attempts > args.n_sdes * 100:
            raise RuntimeError("Too many rejected simulations; reduce the search space.")

        drift = rng.choice(DRIFT_TEMPLATES)
        diffusion = rng.choice(DIFFUSION_TEMPLATES)
        a = sample_coeff(rng, -2.0, 2.0, 0.15)
        b = sample_coeff(rng, -1.0, 1.0, 0.10)
        c = sample_coeff(rng, 0.15, 1.2, 0.15)
        d = sample_coeff(rng, 0.05, 0.8, 0.05)
        coeffs = (a, b, c, d)

        paths = simulate_sde(
            drift, diffusion, coeffs, 1.0, rng,
            args.n_paths, args.n_steps, args.t_final
        )
        if paths is None:
            continue

        multi_parts = []
        valid = True
        for u0 in probe_x0s:
            p = simulate_sde(
                drift, diffusion, coeffs, u0, rng,
                max(200, args.n_paths // 2), args.n_steps, args.t_final
            )
            if p is None:
                valid = False
                break
            multi_parts.append(moment_features(p, include_high_moments=False))
        if not valid:
            continue

        km_features = local_kramers_moyal_features(
            drift,
            diffusion,
            coeffs,
            probe_x0s,
            probe_times,
            rng,
            args.n_paths,
            args.local_dt,
        )
        if km_features is None:
            continue

        fixed_ev.append(moment_features(paths, include_high_moments=False))
        fixed_high.append(moment_features(paths, include_high_moments=True))
        binned_km.append(binned_transition_features(paths, args.t_final, bin_edges))
        binned_local_linear.append(
            binned_local_linear_features(paths, args.t_final, bin_edges)
        )
        multi_features = np.concatenate(multi_parts)
        multi_ev.append(multi_features)
        local_km.append(km_features)
        combined_multi_local.append(np.concatenate([multi_features, km_features]))
        oracle_km.append(
            oracle_generator_features(drift, diffusion, coeffs, probe_x0s, probe_times)
        )

        structure_labels.append((drift.name, diffusion.name))
        drift_expr = drift.expr.format(a=a, b=b)
        diff_expr = diffusion.expr.format(c=c, d=d)
        formula_labels.append(f"dX=({drift_expr})dt + ({diff_expr})dW")

    print("Generated SDEs:", len(formula_labels))
    print("Simulation attempts:", attempts)
    print("Unique structural templates:", len(set(structure_labels)))

    experiments = [
        ("fixed u0, E+Var", np.vstack(fixed_ev)),
        ("fixed u0, E+Var+Skew+Kurt", np.vstack(fixed_high)),
        ("single natural u0, binned KM", np.vstack(binned_km)),
        ("single natural u0, binned local linear KM", np.vstack(binned_local_linear)),
        ("multi u0, E+Var", np.vstack(multi_ev)),
        ("local KM, multi x0/t, E[dX]/dt+Var[dX]/dt", np.vstack(local_km)),
        ("combined multi u0 E+Var + local KM", np.vstack(combined_multi_local)),
        ("oracle generator, multi x0/t, f+g^2", np.vstack(oracle_km)),
    ]
    for name, features in experiments:
        stats = nearest_neighbour_stats(
            features, structure_labels, formula_labels, args.top_k
        )
        print_stats(name, stats)


if __name__ == "__main__":
    main()
