# Expanded Pairing Active Residual Ablation

Date: 2026-06-22

Scope: offline scorer-side ablation using existing expanded-pairing residual vectors. No `sde_validation_probe.py`, model decoding, candidate regeneration, formal eval, 64-sample eval, grid, retraining, or rerank-mode change was run.

## Executive Summary

- Selected-miss cases analyzed: `12`.
- Oracle-only-pair misses: `7`.
- Wrong selected pair cases: `8`.
- Known selected hits checked against debug top-2 challenger: `3`.
- Pair-oracle hit checks: `1`.
- Best offline variant: `per_active_dim_norm_plus_weak` with `6/12` miss rescues and `4/7` oracle-only-pair rescues.
- Decision: **C**. Per-dimension normalization rescues multiple misses, but it is calibrated only from miss residuals; recommend richer residual logging in a future 8- or 16-sample smoke before any scorer change.

Default stance remains: no formal eval and no new rerank mode from this offline evidence.

## Variant Summary

| Variant | Oracle wins /12 | Oracle-only pair wins /7 | Wrong pair wins /8 | Flipped samples | Top-2 hit harms | Pair-oracle hit harms |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `current_logged_rolewise` | 0 | 0 | 0 | `none` | 0 | 0 |
| `current_recomputed` | 5 | 2 | 5 | `12,13,17,21,30` | 2 | 0 |
| `active_clipped_l2_p95_plus_weak` | 5 | 3 | 4 | `12,13,21,30,31` | 1 | 0 |
| `active_clipped_l2_p90_plus_weak` | 5 | 3 | 4 | `5,12,13,21,30` | 1 | 0 |
| `active_huber_p90_plus_weak` | 4 | 2 | 4 | `12,13,21,30` | 1 | 0 |
| `active_top1_removed_plus_weak` | 5 | 3 | 4 | `12,17,21,30,31` | 1 | 0 |
| `active_top2_removed_plus_weak` | 3 | 2 | 3 | `12,21,30` | 1 | 0 |
| `per_active_dim_norm_plus_weak` | 6 | 4 | 5 | `8,12,17,21,24,30` | 0 | 0 |
| `weak_only_diagnostic` | 6 | 4 | 4 | `5,8,12,13,21,30` | 2 | 0 |

## Miss Case Flips

| Sample | Oracle-only pair | Wrong selected pair | Trap shape | Weak favors oracle | Flipping variants |
| ---: | --- | --- | --- | --- | --- |
| 5 | yes | no | outlier-dominated | no | `active_clipped_l2_p90_plus_weak, weak_only_diagnostic` |
| 8 | yes | no | mixed | yes | `per_active_dim_norm_plus_weak, weak_only_diagnostic` |
| 11 | yes | yes | outlier-dominated | no | `none` |
| 12 | yes | yes | outlier-dominated | yes | `current_recomputed, active_clipped_l2_p95_plus_weak, active_clipped_l2_p90_plus_weak, active_huber_p90_plus_weak, active_top1_removed_plus_weak, active_top2_removed_plus_weak, per_active_dim_norm_plus_weak, weak_only_diagnostic` |
| 13 | no | yes | broad | no | `current_recomputed, active_clipped_l2_p95_plus_weak, active_clipped_l2_p90_plus_weak, active_huber_p90_plus_weak, weak_only_diagnostic` |
| 17 | no | yes | mixed | yes | `current_recomputed, active_top1_removed_plus_weak, per_active_dim_norm_plus_weak` |
| 19 | no | no | outlier-dominated | no | `none` |
| 21 | no | yes | outlier-dominated | no | `current_recomputed, active_clipped_l2_p95_plus_weak, active_clipped_l2_p90_plus_weak, active_huber_p90_plus_weak, active_top1_removed_plus_weak, active_top2_removed_plus_weak, per_active_dim_norm_plus_weak, weak_only_diagnostic` |
| 24 | yes | yes | outlier-dominated | no | `per_active_dim_norm_plus_weak` |
| 26 | no | yes | broad | no | `none` |
| 30 | yes | yes | outlier-dominated | yes | `current_recomputed, active_clipped_l2_p95_plus_weak, active_clipped_l2_p90_plus_weak, active_huber_p90_plus_weak, active_top1_removed_plus_weak, active_top2_removed_plus_weak, per_active_dim_norm_plus_weak, weak_only_diagnostic` |
| 31 | yes | no | mixed | no | `active_clipped_l2_p95_plus_weak, active_top1_removed_plus_weak` |

## Known Hit Harm Check

This is a limited check against the debug top-2 challenger for selected-hit samples. It is not a full candidate-pool safety proof.

| Sample | Oracle source | Challenger source | Harmed variants |
| ---: | --- | --- | --- |
| 18 | beam | pair | `current_recomputed, weak_only_diagnostic` |
| 23 | pair | pair | `none` |
| 27 | beam | pair | `current_recomputed, active_clipped_l2_p95_plus_weak, active_clipped_l2_p90_plus_weak, active_huber_p90_plus_weak, active_top1_removed_plus_weak, active_top2_removed_plus_weak, weak_only_diagnostic` |

## Interpretation

- Robust/clipped active scoring is not yet compelling as a real rerank mode. The clipped/Huber/top-k variants rescue several misses but harm one debug top-2 selected-hit check.
- Per-active-dimension normalization is the strongest offline signal, but it is calibrated only across the miss residual vectors; it is diagnostic evidence, not a stable calibration estimate.
- Weak-only behavior confirms that weak sometimes favors oracles, but the signal is mixed and should not trigger another broad active/weak grid.
- Pair-oracle coverage itself is not harmed by an offline score, but selected pair-oracle hits are only checked against top-2 challengers here.

## Recommendation

Do not add a rerank mode and do not run a formal 32-sample eval. The next useful direction is drift span diversity or deeper active-fingerprint analysis. If scorer work continues, first add richer residual logging in a small smoke so hit-safety and full candidate-pool effects can be checked offline.

## Provenance

- Residual JSON: `expanded_pairing_active_residual_trap_diagnostics.json`.
- Miss JSON: `expanded_pairing_oracle_miss_ranking_diagnostics.json`.
- Source log: `/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log`.
