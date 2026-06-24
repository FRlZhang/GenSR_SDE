# P2 Calibration All32 Smoke

## Scope

Offline all-32 shared-constant / fixed residual calibration smoke on the scorer-ready sidecar. No formal eval, rerank-mode implementation, or production scoring change was run.

## Inputs

- Sidecar: `p2_scorer_ready_candidates_all32.json`
- P2 counterfactuals: `p2_residual_calibration_counterfactuals.json`
- Baseline coverage: `candidate_coverage_diagnostics.json`
- P2 coverage: `candidate_coverage_pair_expanded_p2_flag.json`
- Target samples: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31`

## Baseline Coverage Context

| Item | Value |
| --- | ---: |
| Current expanded full oracle | 15 / 32 |
| P2 normalized admission coverage | 30 / 32 |
| Baseline full oracle | 9 / 32 |

## Variants

- `V0_current_rolewise_no_multi_u0`
- `V11_shared_constants_only_best_of_grid`
- `V3_active_without_dim_76`
- `V4_active_without_dims_76_72_82_88`
- `V6_active_and_weak_without_recurring_dims`

## Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 32 |
| Candidate rows analyzed | 970 |
| Valid candidates | 970 |
| Parse failures | 0 |
| Fingerprint failures | 0 |
| V0 selected exact | 6 |
| V0 selected relaxed | 6 |
| V11 selected exact | 4 |
| V11 selected relaxed | 4 |
| Best variant by selected exact | V4_active_without_dims_76_72_82_88 |
| Best variant by selected relaxed | V4_active_without_dims_76_72_82_88 |
| V0 selected oracle samples | [9, 10, 14, 17, 18, 21] |
| V0 selected P2 oracle samples | [14] |

## Variant Comparison

| Variant | Selected exact | Selected relaxed | Selected source current | Selected source P2 | Selected oracle | Selected P2 oracle | Rescues vs V0 | Harms vs V0 | Net exact delta | Net relaxed delta | V0-hit preserved | V0-hit harmed | Near-tie rescues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0_current_rolewise_no_multi_u0 | 6 | 6 | 25 | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V11_shared_constants_only_best_of_grid | 4 | 4 | 23 | 9 | 4 | 1 | 1 | 3 | -2 | -2 | 3 | 3 | 0 |
| V3_active_without_dim_76 | 7 | 7 | 22 | 10 | 7 | 3 | 2 | 1 | 1 | 1 | 5 | 1 | 1 |
| V4_active_without_dims_76_72_82_88 | 9 | 9 | 23 | 9 | 9 | 2 | 5 | 2 | 3 | 3 | 4 | 2 | 1 |
| V6_active_and_weak_without_recurring_dims | 7 | 7 | 23 | 9 | 7 | 2 | 4 | 3 | 1 | 1 | 3 | 3 | 1 |

## P2 Subset

Samples: `0,2,6,7,15,25,29`

| Variant | Oracle on P2 subset | P2 oracle on P2 subset |
| --- | ---: | ---: |
| V0_current_rolewise_no_multi_u0 | 0 | 0 |
| V11_shared_constants_only_best_of_grid | 0 | 0 |
| V3_active_without_dim_76 | 1 | 1 |
| V4_active_without_dims_76_72_82_88 | 1 | 1 |
| V6_active_and_weak_without_recurring_dims | 1 | 1 |

## Decision

Decision `B`: The variant rescues some P2 cases but either harms V0 hits or does not improve net selected recovery; use this as scorer diagnostics only.

No formal eval or rerank-mode implementation unless this smoke gives net selected improvement with no V0 hit harms.
