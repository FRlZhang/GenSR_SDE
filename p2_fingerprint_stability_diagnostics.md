# P2 Fingerprint Stability Diagnostics

## Why This Diagnostic Was Run

The previous V0/V4 fingerprint ambiguity report found stable ordering in only
2/7 small resimulation comparisons. This helper narrows that result to
fingerprint simulation stability for the 5 V4 rescues and 2 V4 harms. It
does not run model decoding, candidate generation, `sde_validation_probe.py`,
formal eval, scorer grids, rerank-mode implementation, or production default
changes.

## Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 7 |
| Candidate oracle pairs analyzed | 8 |
| Repeats | 5 |
| M1 ordering stable | 1 |
| M1 ordering unstable | 7 |
| Noise-dominated pairs | 8 |
| Stable clear pairs | 0 |
| Active noise driver | 2 |
| Weak noise driver | 1 |
| Mixed noise driver | 5 |

## Stability Modes

| Mode | Status |
| --- | --- |
| M0_stored_target_stored_candidate | available |
| M1_stored_target_resim_candidate | available |
| M2_resim_target_stored_candidate_if_possible | unavailable: sidecar does not store candidate fingerprint vectors |
| M3_paired_resim_target_and_candidate | unavailable: raw numeric target SDE constants are not stored, so target fingerprint Monte Carlo resimulation is not cleanly reproducible |

## Per-Sample Ordering Stability

| Sample | Case | Pair | Stable | Oracle wins | Selected wins | Noise dominated | Driver | Mean gap | Std gap | Z |
| ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | V4_rescue | V0_selected_vs_oracle | yes | 0 | 5 | yes | mixed | -0.176665 | 0.235516 | 0.750119 |
| 1 | V4_rescue | V4_selected_vs_oracle | no | 2 | 3 | yes | mixed | -0.216129 | 0.386451 | 0.559267 |
| 7 | V4_rescue | V0_selected_vs_oracle | no | 3 | 2 | yes | mixed | 0.331597 | 0.568202 | 0.583591 |
| 22 | V4_rescue | V0_selected_vs_oracle | no | 2 | 3 | yes | active | -0.014819 | 0.0496083 | 0.298721 |
| 23 | V4_rescue | V0_selected_vs_oracle | no | 3 | 2 | yes | mixed | -0.0171489 | 0.180671 | 0.0949179 |
| 31 | V4_rescue | V0_selected_vs_oracle | no | 1 | 4 | yes | active | -0.0312498 | 0.0550983 | 0.567164 |
| 9 | V4_harm | V4_selected_vs_oracle | no | 3 | 2 | yes | mixed | 0.0903003 | 0.256111 | 0.352583 |
| 21 | V4_harm | V4_selected_vs_oracle | no | 1 | 4 | yes | weak | -0.145884 | 0.284866 | 0.512113 |

## Gap Mean/Std/Z Score

| Sample | Pair | Baseline gap | Active gap mean/std | Weak gap mean/std | Sign flip fraction |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | V0_selected_vs_oracle | -0.154757 | -0.168259/0.0959753 | -0.00840592/0.15486 | 0 |
| 1 | V4_selected_vs_oracle | 0.00140961 | -0.0829653/0.224798 | -0.133164/0.230617 | 0.4 |
| 7 | V0_selected_vs_oracle | 0.00615239 | 0.244277/0.38293 | 0.0873201/0.217725 | 0.4 |
| 22 | V0_selected_vs_oracle | 0.00317607 | -0.0136841/0.0398195 | -0.00113498/0.0144554 | 0.4 |
| 23 | V0_selected_vs_oracle | -0.0536874 | -0.0500717/0.114515 | 0.0329227/0.108044 | 0.4 |
| 31 | V0_selected_vs_oracle | 0.138803 | -0.0293772/0.0533374 | -0.00187262/0.0261214 | 0.2 |
| 9 | V4_selected_vs_oracle | -0.267068 | 0.0288882/0.189344 | 0.0614121/0.11808 | 0.4 |
| 21 | V4_selected_vs_oracle | -0.36288 | 0.026724/0.0424489 | -0.172608/0.255516 | 0.2 |

## Removed-Dimension Stability

| Sample | Case | Contribution mean | Contribution std | Sign flip fraction | Noise dominated | Stable explanation |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | V4_rescue | -0.0785121 | 0.143731 | 0.4 | yes | no |
| 7 | V4_rescue | 0.198748 | 0.252638 | 0.2 | yes | no |
| 22 | V4_rescue | 0.088249 | 0.146435 | 0.2 | yes | no |
| 23 | V4_rescue | -0.110824 | 0.167246 | 0.4 | yes | no |
| 31 | V4_rescue | 0.0385129 | 0.0948629 | 0.4 | yes | no |
| 9 | V4_harm | 0.0210748 | 0.145345 | 0.2 | yes | no |
| 21 | V4_harm | -0.113063 | 0.146658 | 0.2 | yes | no |

## Interpretation

M1 is available and isolates candidate fingerprint resimulation with the
stored target fingerprint fixed. M2 is unavailable because candidate
fingerprint vectors are not stored. M3 is unavailable because the sidecar
stores target fingerprint vectors but not the raw numeric target SDE
constants needed to resimulate the target truth process exactly.

## Decision

Decision `B`: Ordering instability is visible with stored target and candidate resimulation. Diagnose candidate fingerprint variance or path budget before scorer changes.

No formal eval, no rerank mode, no P2 eval integration, and no scorer
change are recommended from this diagnostic alone.
