# P2 Fingerprint Variance Anatomy Diagnostic

## Why This Follows Path-Budget

The path-budget diagnostic showed B1 active-paths-high improves some
V4 rescue/harm oracle-pair orderings, but most pairs remain
noise-dominated. This helper asks whether the remaining variance is
localized to a small set of active/weak dimensions or spread across
segments. No model decoding, candidate generation, `sde_validation_probe.py`,
formal eval, rerank mode, P2 eval integration, or scorer change was run.

## Data Source

| Item | Value |
| --- | ---: |
| Existing path-budget data sufficient | no |
| Recomputed B0/B1 residuals | yes |
| Samples analyzed | 7 |
| Candidate pairs analyzed | 8 |
| Budget modes analyzed | B0_current_budget, B1_active_paths_high |

Note: the anatomy recomputation uses 3 repeats for both B0 and B1 so
per-dimension residual vectors are directly comparable. The previous
path-budget report used 5 repeats for B0 and 3 repeats for B1, so B0
stable/noise counts can differ slightly here.

## B0 vs B1 Pair Stability

| Budget | Stable | Noise dominated | Active variance share | Weak variance share |
| --- | ---: | ---: | ---: | ---: |
| B0 | 2 | 6 | 0.476542 | 0.523458 |
| B1 | 4 | 6 | 0.477825 | 0.522175 |

## Active Dimension Variance

| Budget | Top variance dims | Top improved dims | Top not-improved dims |
| --- | --- | --- | --- |
| B0 | 88, 76, 80, 86, 82, 74, 72, 84 | NA | NA |
| B1 | 82, 88, 80, 86, 76, 74, 72, 77 | 76, 88, 80, 78, 86, 74, 84, 87 | 82, 77, 89, 75, 81, 85, 73 |

## Weak Dimension Variance

| Budget | Top variance dims | Top improved dims | Top not-improved dims |
| --- | --- | --- | --- |
| B0 | 114, 115, 116, 137, 117, 118, 136, 107 | NA | NA |
| B1 | 114, 115, 116, 161, 117, 160, 137, 159 | 119, 137, 120, 118, 136, 132, 133, 131 | 161, 160, 159, 158, 114, 157, 156, 155 |

## Recurring Dimension Check

| Dimension set | Reappears in top variance dims |
| --- | --- |
| Active 76/72/82/88 | yes |
| Weak 136/137/133 | yes |

## Pair-Level Diagnosis

| Sample | Case | Pair | B1 stabilizes | B1 noise dominated | Reduces gap std | Reduces active std | Reduces weak std | Diagnosis |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | V4_rescue | V0_selected_vs_oracle | no | no | no | no | yes | active_budget_counterproductive |
| 1 | V4_rescue | V4_selected_vs_oracle | no | yes | no | yes | yes | active_budget_counterproductive |
| 7 | V4_rescue | V0_selected_vs_oracle | no | yes | yes | yes | yes | ordering_intrinsically_small_gap |
| 9 | V4_harm | V4_selected_vs_oracle | no | yes | yes | yes | yes | mixed_noise_persists |
| 21 | V4_harm | V4_selected_vs_oracle | no | yes | yes | yes | yes | mixed_noise_persists |
| 22 | V4_rescue | V0_selected_vs_oracle | yes | yes | no | yes | yes | active_local_noise_reduced |
| 23 | V4_rescue | V0_selected_vs_oracle | yes | no | yes | yes | yes | active_local_noise_reduced |
| 31 | V4_rescue | V0_selected_vs_oracle | no | yes | yes | no | no | mixed_noise_persists |

## Decision

Decision `B`: B1 helps some pairs, but variance remains mixed or spread across dimensions. Keep scorer calibration closed; return to drift-span/candidate-generation unless broader fingerprint infrastructure is explicitly desired.

No formal eval, no rerank mode, no P2 eval integration, and no scorer
change are recommended from this diagnostic alone.
