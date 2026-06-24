# P2 Residual Calibration Counterfactuals

## Scope

Offline fixed counterfactuals on the 7 P2 oracle score misses. Lower score is better. No formal eval, rerank-mode implementation, scorer grid, or production default change was run.

## Variants

- `V0_current_recomputed`
- `V1_weak_only_diagnostic`
- `V2_active_only_diagnostic`
- `V3_active_without_dim_76`
- `V4_active_without_dims_76_72_82_88`
- `V5_weak_without_dims_136_137_133`
- `V6_active_and_weak_without_recurring_dims`
- `V7_top1_active_advantage_removed_per_sample`
- `V8_top2_active_advantages_removed_per_sample`
- `V9_selected_constants_applied_to_oracle`
- `V10_oracle_constants_applied_to_selected`
- `V11_shared_constants_only_best_of_grid`

## Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 7 |
| Baseline oracle wins | 0 |
| Samples flipped by any variant | 5 |
| Samples flipped by global recurring dims | 2 |
| Samples flipped by per-sample top dims | 2 |
| Samples flipped by constant counterfactuals | 5 |
| Near-tie samples flipped | 2 |
| Active-trap samples flipped | 2 |
| Weak-trap samples flipped | 0 |

Variant oracle wins: `{"V0_current_recomputed": 0, "V10_oracle_constants_applied_to_selected": 2, "V11_shared_constants_only_best_of_grid": 3, "V1_weak_only_diagnostic": 2, "V2_active_only_diagnostic": 0, "V3_active_without_dim_76": 2, "V4_active_without_dims_76_72_82_88": 2, "V5_weak_without_dims_136_137_133": 0, "V6_active_and_weak_without_recurring_dims": 2, "V7_top1_active_advantage_removed_per_sample": 2, "V8_top2_active_advantages_removed_per_sample": 2, "V9_selected_constants_applied_to_oracle": 0}`

## Per-Sample Best Counterfactual

| Sample | Classification | Baseline gap | Best variant | Best gap | Any flip |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | weak_trap | 0.163991 | V2_active_only_diagnostic | 0.023534 | False |
| 2 | current_candidate_trap | 0.123196 | V10_oracle_constants_applied_to_selected | -3.374593 | True |
| 6 | near_tie | 0.060832 | V11_shared_constants_only_best_of_grid | -0.456385 | True |
| 7 | near_tie | 0.073651 | V8_top2_active_advantages_removed_per_sample | -0.206970 | True |
| 15 | active_trap | 0.291663 | V11_shared_constants_only_best_of_grid | -0.005123 | True |
| 25 | active_trap | 1.430532 | V1_weak_only_diagnostic | 0.660977 | False |
| 29 | active_trap | 0.568912 | V11_shared_constants_only_best_of_grid | -0.174420 | True |

## Decision

Decision `A`: A stable fixed counterfactual flips several P2 misses; one future small smoke may be justified, but no formal eval.

No formal eval or rerank-mode implementation is recommended unless a later smoke improves selected behavior safely.
