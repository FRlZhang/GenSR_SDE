# P2 Calibration Gate Feasibility

## Scope

Offline helper-only gate feasibility diagnostic over existing JSON artifacts. No model decoding, candidate generation, formal eval, scorer mode, or production default change was run.

## Input All32 Smoke Summary

| Item | Value |
| --- | ---: |
| Target samples analyzed | 32 |
| V0 selected exact/relaxed | 6 |
| V0 selected P2 oracle | 1 |
| V0 selected hit samples | [9, 10, 14, 17, 18, 21] |

## Variants Considered

- `V3_active_without_dim_76`
- `V4_active_without_dims_76_72_82_88`
- `V6_active_and_weak_without_recurring_dims`

## Observable Gates

| Gate | Status | Reason |
| --- | --- | --- |
| `G1_variant_selected_source_is_p2_admitted` | available | - |
| `G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded` | available | - |
| `G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable` | available | - |
| `G4_v0_selected_active_top_dim_is_76` | unavailable | all-32 selected residual top-dimension fields are missing; only 7/32 P2 score-miss samples have residual diagnostics |
| `G5_v0_selected_active_top_dim_in_76_72_82_88` | unavailable | all-32 selected residual top-dimension fields are missing; only 7/32 P2 score-miss samples have residual diagnostics |
| `G6_v0_selected_active_recurring_dims_share_ge_0_50` | unavailable | all-32 selected residual top-dimension fields are missing; only 7/32 P2 score-miss samples have residual diagnostics |
| `G7_v0_selected_active_recurring_dims_share_ge_0_35` | unavailable | all-32 selected residual top-dimension fields are missing; only 7/32 P2 score-miss samples have residual diagnostics |
| `G8_v0_selected_weak_top_dim_in_136_137_133` | unavailable | all-32 selected residual top-dimension fields are missing; only 7/32 P2 score-miss samples have residual diagnostics |
| `G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted` | available | - |

## Best Observable Gates

| Variant | Gate | Exact | Relaxed | P2 oracle | Rescues | Harms | Net | Zero harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V4_active_without_dims_76_72_82_88 | `G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable` | 9 | 9 | 2 | 5 | 2 | 3 | False |
| V3_active_without_dim_76 | `G1_variant_selected_source_is_p2_admitted` | 7 | 7 | 3 | 2 | 1 | 1 | False |
| V3_active_without_dim_76 | `G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable` | 7 | 7 | 3 | 2 | 1 | 1 | False |
| V6_active_and_weak_without_recurring_dims | `G3_variant_selected_score_improves_v0_score_within_same_variant_report_if_comparable` | 7 | 7 | 2 | 4 | 3 | 1 | False |
| V3_active_without_dim_76 | `G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded` | 6 | 6 | 2 | 1 | 1 | 0 | False |
| V3_active_without_dim_76 | `G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted` | 6 | 6 | 2 | 1 | 1 | 0 | False |
| V4_active_without_dims_76_72_82_88 | `G1_variant_selected_source_is_p2_admitted` | 6 | 6 | 2 | 1 | 1 | 0 | False |
| V6_active_and_weak_without_recurring_dims | `G1_variant_selected_source_is_p2_admitted` | 6 | 6 | 2 | 1 | 1 | 0 | False |
| V4_active_without_dims_76_72_82_88 | `G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded` | 5 | 5 | 1 | 0 | 1 | -1 | False |
| V4_active_without_dims_76_72_82_88 | `G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted` | 5 | 5 | 1 | 0 | 1 | -1 | False |
| V6_active_and_weak_without_recurring_dims | `G2_variant_selected_source_is_p2_admitted_and_v0_selected_source_is_current_expanded` | 5 | 5 | 1 | 0 | 1 | -1 | False |
| V6_active_and_weak_without_recurring_dims | `G9_v0_selected_source_is_current_expanded_and_variant_selected_source_is_p2_admitted` | 5 | 5 | 1 | 0 | 1 | -1 | False |

## Oracle Upper Bounds

These gates use oracle labels or known oracle-recovered samples and are not implementable.

| Variant | Gate | Exact | Relaxed | P2 oracle | Rescues | Harms | Net | Zero harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3_active_without_dim_76 | `U1_apply_variant_only_when_V0_is_not_oracle` | 8 | 8 | 3 | 2 | 0 | 2 | True |
| V3_active_without_dim_76 | `U2_apply_variant_only_when_variant_is_oracle` | 8 | 8 | 3 | 2 | 0 | 2 | True |
| V3_active_without_dim_76 | `U3_apply_variant_only_on_known_P2_newly_recovered_samples` | 7 | 7 | 2 | 1 | 0 | 1 | True |
| V4_active_without_dims_76_72_82_88 | `U1_apply_variant_only_when_V0_is_not_oracle` | 11 | 11 | 2 | 5 | 0 | 5 | True |
| V4_active_without_dims_76_72_82_88 | `U2_apply_variant_only_when_variant_is_oracle` | 11 | 11 | 2 | 5 | 0 | 5 | True |
| V4_active_without_dims_76_72_82_88 | `U3_apply_variant_only_on_known_P2_newly_recovered_samples` | 7 | 7 | 2 | 1 | 0 | 1 | True |
| V6_active_and_weak_without_recurring_dims | `U1_apply_variant_only_when_V0_is_not_oracle` | 10 | 10 | 2 | 4 | 0 | 4 | True |
| V6_active_and_weak_without_recurring_dims | `U2_apply_variant_only_when_variant_is_oracle` | 10 | 10 | 2 | 4 | 0 | 4 | True |
| V6_active_and_weak_without_recurring_dims | `U3_apply_variant_only_on_known_P2_newly_recovered_samples` | 7 | 7 | 2 | 1 | 0 | 1 | True |

## Decision

Decision `B`: Observable gates rescue some cases or improve net, but positive gates harm V0 hits or fail the zero-harm criterion. Do not implement; close fixed residual calibration implementation path for now.

No formal eval, no rerank mode, and no P2 eval integration unless an observable zero-harm gate exists.
