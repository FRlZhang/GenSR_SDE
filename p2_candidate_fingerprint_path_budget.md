# P2 Candidate Fingerprint Path-Budget Diagnostic

## Why This Diagnostic Was Run

The previous fingerprint stability diagnostic found that candidate
resimulation alone made V4 rescue/harm oracle-pair ordering unstable. This
helper checks whether a small fixed increase in candidate fingerprint path
budget stabilizes those same pairs. No model decoding, candidate generation,
`sde_validation_probe.py`, formal eval, scorer mode, or production default
change was run.

## Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 7 |
| Candidate pairs analyzed | 8 |
| Budget modes available | B0_current_budget, B1_active_paths_high, B2_weak_paths_high, B3_both_paths_high |
| Budget modes unavailable | none |

## Per-Budget Ordering Stability

| Budget | Stable | Unstable | Noise dominated | Stable clear | Active driver | Weak driver | Mixed driver | Mean gap z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_current_budget | 0 | 8 | 8 | 0 | 1 | 0 | 7 | 0.685335 |
| B1_active_paths_high | 4 | 4 | 6 | 2 | 1 | 1 | 6 | 1.43536 |
| B2_weak_paths_high | 3 | 5 | 8 | 0 | 2 | 2 | 4 | 0.937109 |
| B3_both_paths_high | 1 | 7 | 7 | 1 | 5 | 2 | 1 | 0.78611 |

## Per-Pair Path-Budget Metrics

| Sample | Case | Pair | Budget | Gap mean | Gap std | Z | Stable | Noise dominated | Driver |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | V4_rescue | V0_selected_vs_oracle | B0_current_budget | -0.112968 | 0.124 | 0.911037 | no | yes | mixed |
| 1 | V4_rescue | V0_selected_vs_oracle | B1_active_paths_high | -0.235189 | 0.0693264 | 3.39249 | yes | no | mixed |
| 1 | V4_rescue | V0_selected_vs_oracle | B2_weak_paths_high | 0.266734 | 0.25588 | 1.04242 | no | yes | weak |
| 1 | V4_rescue | V0_selected_vs_oracle | B3_both_paths_high | 0.106716 | 0.273379 | 0.390357 | no | yes | weak |
| 1 | V4_rescue | V4_selected_vs_oracle | B0_current_budget | -0.226997 | 0.356131 | 0.637397 | no | yes | mixed |
| 1 | V4_rescue | V4_selected_vs_oracle | B1_active_paths_high | -0.483698 | 0.256665 | 1.88455 | yes | yes | weak |
| 1 | V4_rescue | V4_selected_vs_oracle | B2_weak_paths_high | 0.262733 | 0.391371 | 0.671314 | no | yes | weak |
| 1 | V4_rescue | V4_selected_vs_oracle | B3_both_paths_high | 0.269566 | 0.370905 | 0.72678 | no | yes | weak |
| 7 | V4_rescue | V0_selected_vs_oracle | B0_current_budget | -0.331491 | 0.411872 | 0.80484 | no | yes | active |
| 7 | V4_rescue | V0_selected_vs_oracle | B1_active_paths_high | -0.244223 | 0.31277 | 0.780837 | no | yes | active |
| 7 | V4_rescue | V0_selected_vs_oracle | B2_weak_paths_high | -0.218662 | 0.118483 | 1.84551 | yes | yes | mixed |
| 7 | V4_rescue | V0_selected_vs_oracle | B3_both_paths_high | -0.329694 | 0.128485 | 2.56601 | yes | no | active |
| 9 | V4_harm | V4_selected_vs_oracle | B0_current_budget | 0.179934 | 0.281755 | 0.638618 | no | yes | mixed |
| 9 | V4_harm | V4_selected_vs_oracle | B1_active_paths_high | 0.15227 | 0.149507 | 1.01849 | no | yes | mixed |
| 9 | V4_harm | V4_selected_vs_oracle | B2_weak_paths_high | 0.0328476 | 0.0718456 | 0.457198 | no | yes | mixed |
| 9 | V4_harm | V4_selected_vs_oracle | B3_both_paths_high | 0.115348 | 0.195755 | 0.589248 | no | yes | active |
| 21 | V4_harm | V4_selected_vs_oracle | B0_current_budget | -0.118804 | 0.116593 | 1.01896 | no | yes | mixed |
| 21 | V4_harm | V4_selected_vs_oracle | B1_active_paths_high | 0.0124459 | 0.054497 | 0.228378 | no | yes | mixed |
| 21 | V4_harm | V4_selected_vs_oracle | B2_weak_paths_high | 0.0548616 | 0.0358536 | 1.53016 | yes | yes | mixed |
| 21 | V4_harm | V4_selected_vs_oracle | B3_both_paths_high | 0.0302417 | 0.0906593 | 0.333575 | no | yes | mixed |
| 22 | V4_rescue | V0_selected_vs_oracle | B0_current_budget | -0.0308884 | 0.0434976 | 0.710117 | no | yes | mixed |
| 22 | V4_rescue | V0_selected_vs_oracle | B1_active_paths_high | -0.0626466 | 0.060436 | 1.03658 | yes | yes | mixed |
| 22 | V4_rescue | V0_selected_vs_oracle | B2_weak_paths_high | -0.0202155 | 0.0815647 | 0.247847 | no | yes | active |
| 22 | V4_rescue | V0_selected_vs_oracle | B3_both_paths_high | -0.0445551 | 0.0865657 | 0.514697 | no | yes | active |
| 23 | V4_rescue | V0_selected_vs_oracle | B0_current_budget | -0.0215408 | 0.189446 | 0.113704 | no | yes | mixed |
| 23 | V4_rescue | V0_selected_vs_oracle | B1_active_paths_high | -0.207877 | 0.0822198 | 2.52831 | yes | no | mixed |
| 23 | V4_rescue | V0_selected_vs_oracle | B2_weak_paths_high | 0.0241666 | 0.0930074 | 0.259835 | no | yes | mixed |
| 23 | V4_rescue | V0_selected_vs_oracle | B3_both_paths_high | 0.108253 | 0.25035 | 0.432408 | no | yes | active |
| 31 | V4_rescue | V0_selected_vs_oracle | B0_current_budget | 0.0434496 | 0.0670507 | 0.648012 | no | yes | mixed |
| 31 | V4_rescue | V0_selected_vs_oracle | B1_active_paths_high | -0.0415197 | 0.0677043 | 0.61325 | no | yes | mixed |
| 31 | V4_rescue | V0_selected_vs_oracle | B2_weak_paths_high | 0.0772593 | 0.0535558 | 1.44259 | yes | yes | active |
| 31 | V4_rescue | V0_selected_vs_oracle | B3_both_paths_high | -0.0450446 | 0.0612176 | 0.735812 | no | yes | active |

## Path-Budget Improvement

| Comparison | Stable gain | Noise-dominated reduction |
| --- | ---: | ---: |
| B1 vs B0 | 4 | 2 |
| B2 vs B0 | 3 | 0 |
| B3 vs B0 | 1 | 1 |

Best budget mode: `B1_active_paths_high`.

## Decision

Decision `B`: Higher path budget improves stability somewhat but not enough. Keep scorer calibration closed and consider targeted active/weak fingerprint variance reduction diagnostics.

No formal eval, no rerank mode, no P2 eval integration, and no scorer
change are recommended from this diagnostic alone.
