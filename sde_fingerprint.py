#!/usr/bin/env python3
"""Unified SDE fingerprints for the GenSR numerical branch.

The first training version keeps the fingerprint shorter than GenSR's default
``max_len=200`` by encoding it as a scalar sequence:

    x_to_fit[i] = normalized feature index
    y_to_fit[i] = standardized fingerprint value
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class FingerprintConfig:
    kind: str = "multi_active_weak_v1"
    t_final: float = 1.0
    n_steps: int = 80
    n_paths: int = 2000
    natural_u0: float = 1.0
    multi_u0s: tuple[float, ...] = (0.5, 1.0, 1.5)
    moment_time_samples: int = 12
    active_probe_x0s: tuple[float, ...] = (0.5, 1.0, 1.5)
    active_probe_times: tuple[float, ...] = (0.0, 0.5, 1.0)
    active_dt: float = 1e-2
    active_paths: int = 2000
    kernel_min: float = -2.0
    kernel_max: float = 2.0
    n_kernels: int = 24
    kernel_bandwidth: float = 0.35
    clip_value: float = 1e6
    standardize: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fingerprint_length"] = fingerprint_length(self)
        return data


def fingerprint_length(config: FingerprintConfig) -> int:
    multi_len = len(config.multi_u0s) * config.moment_time_samples * 2
    active_len = (
        len(config.active_probe_times)
        * len(config.active_probe_x0s)
        * 2
    )
    weak_len = config.n_kernels * 4
    return multi_len + active_len + weak_len


def simulate_paths(
    drift_fn,
    diffusion_fn,
    u0: float,
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    t_final: float,
    clip_value: float = 1e6,
) -> np.ndarray | None:
    dt = t_final / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, t_final, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = u0

    for step in range(n_steps):
        x = paths[:, step]
        t = float(t_grid[step])
        drift = drift_fn(x, t)
        diffusion = diffusion_fn(x, t)
        nxt = x + drift * dt + diffusion * rng.normal(0.0, sqrt_dt, size=n_paths)
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > clip_value:
            return None
        paths[:, step + 1] = nxt
    return paths


def moment_features(paths: np.ndarray, n_time_samples: int) -> np.ndarray:
    indices = np.linspace(0, paths.shape[1] - 1, n_time_samples, dtype=int)
    sliced = paths[:, indices]
    mean = sliced.mean(axis=0)
    var = sliced.var(axis=0)
    return np.column_stack([mean, var]).reshape(-1)


def active_kramers_moyal_features(
    drift_fn,
    diffusion_fn,
    rng: np.random.Generator,
    probe_x0s: tuple[float, ...],
    probe_times: tuple[float, ...],
    n_paths: int,
    dt: float,
) -> np.ndarray | None:
    sqrt_dt = np.sqrt(dt)
    features: list[float] = []
    for t in probe_times:
        for x0 in probe_x0s:
            x = np.full(n_paths, x0, dtype=np.float64)
            delta = drift_fn(x, float(t)) * dt
            delta = delta + diffusion_fn(x, float(t)) * rng.normal(
                0.0, sqrt_dt, size=n_paths
            )
            if not np.all(np.isfinite(delta)):
                return None
            features.append(float(delta.mean() / dt))
            features.append(float(delta.var() / dt))
    return np.asarray(features, dtype=np.float64)


def gaussian_weak_kernel_features(
    paths: np.ndarray,
    t_final: float,
    centers: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    n_steps = paths.shape[1] - 1
    dt = t_final / n_steps
    x_t = paths[:, :-1].reshape(-1)
    x_next = paths[:, 1:].reshape(-1)
    dx = x_next - x_t
    n_samples = max(1, x_t.size)
    n_kernels = centers.size

    mass = np.zeros(n_kernels, dtype=np.float64)
    drift_sum = np.zeros(n_kernels, dtype=np.float64)
    second_sum = np.zeros(n_kernels, dtype=np.float64)
    generator_sum = np.zeros(n_kernels, dtype=np.float64)

    chunk = 50000
    for start in range(0, x_t.size, chunk):
        end = min(start + chunk, x_t.size)
        x0 = x_t[start:end]
        x1 = x_next[start:end]
        delta = dx[start:end]
        k0 = np.exp(-0.5 * ((x0[:, None] - centers[None, :]) / bandwidth) ** 2)
        k1 = np.exp(-0.5 * ((x1[:, None] - centers[None, :]) / bandwidth) ** 2)
        mass += k0.sum(axis=0)
        drift_sum += k0.T @ delta
        second_sum += k0.T @ (delta * delta)
        generator_sum += (k1 - k0).sum(axis=0)

    denom = mass + 1e-12
    occupancy = mass / n_samples
    local_drift = drift_sum / (denom * dt)
    local_diffusion = second_sum / (denom * dt)
    generator_action = generator_sum / (n_samples * dt)
    return np.concatenate([occupancy, local_drift, local_diffusion, generator_action])


def standardize_feature_vector(features: np.ndarray) -> tuple[np.ndarray, dict]:
    center = float(np.median(features))
    scale = float(np.quantile(np.abs(features - center), 0.75))
    if scale < 1e-8:
        scale = float(np.std(features))
    if scale < 1e-8:
        scale = 1.0
    normalized = np.clip((features - center) / scale, -20.0, 20.0)
    return normalized.astype(np.float32), {
        "center": center,
        "scale": scale,
        "clip": 20.0,
    }


def compute_unified_fingerprint(
    drift_fn,
    diffusion_fn,
    rng: np.random.Generator,
    config: FingerprintConfig,
) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    natural_paths = simulate_paths(
        drift_fn,
        diffusion_fn,
        config.natural_u0,
        rng,
        config.n_paths,
        config.n_steps,
        config.t_final,
        config.clip_value,
    )
    if natural_paths is None:
        return None, {"error": "natural_simulation_failed"}

    parts = []
    part_slices = {}
    cursor = 0

    multi_parts = []
    for u0 in config.multi_u0s:
        paths = simulate_paths(
            drift_fn,
            diffusion_fn,
            u0,
            rng,
            max(200, config.n_paths // 2),
            config.n_steps,
            config.t_final,
            config.clip_value,
        )
        if paths is None:
            return None, {"error": f"multi_simulation_failed_u0={u0}"}
        multi_parts.append(moment_features(paths, config.moment_time_samples))
    multi_features = np.concatenate(multi_parts)
    parts.append(multi_features)
    part_slices["multi_u0_moments"] = [cursor, cursor + multi_features.size]
    cursor += multi_features.size

    active_features = active_kramers_moyal_features(
        drift_fn,
        diffusion_fn,
        rng,
        config.active_probe_x0s,
        config.active_probe_times,
        config.active_paths,
        config.active_dt,
    )
    if active_features is None:
        return None, {"error": "active_km_failed"}
    parts.append(active_features)
    part_slices["active_kramers_moyal"] = [cursor, cursor + active_features.size]
    cursor += active_features.size

    kernel_centers = np.linspace(
        config.kernel_min, config.kernel_max, config.n_kernels
    )
    weak_features = gaussian_weak_kernel_features(
        natural_paths,
        config.t_final,
        kernel_centers,
        config.kernel_bandwidth,
    )
    parts.append(weak_features)
    part_slices["gaussian_weak_kernel"] = [cursor, cursor + weak_features.size]
    cursor += weak_features.size

    raw = np.concatenate(parts).astype(np.float64)
    if not np.all(np.isfinite(raw)):
        return None, {"error": "nonfinite_fingerprint"}

    if config.standardize:
        fingerprint, normalization = standardize_feature_vector(raw)
    else:
        fingerprint = raw.astype(np.float32)
        normalization = {"center": 0.0, "scale": 1.0, "clip": None}

    meta = {
        "fingerprint_config": config.to_dict(),
        "part_slices": part_slices,
        "normalization": normalization,
        "kernel_centers": kernel_centers.astype(np.float32).tolist(),
    }
    return fingerprint, meta


def fingerprint_to_xy(fingerprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = fingerprint.shape[0]
    if length <= 1:
        x = np.zeros((length, 1), dtype=np.float32)
    else:
        x = np.linspace(-1.0, 1.0, length, dtype=np.float32).reshape(-1, 1)
    y = fingerprint.astype(np.float32).reshape(-1, 1)
    return x, y
