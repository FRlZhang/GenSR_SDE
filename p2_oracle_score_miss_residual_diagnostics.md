# P2 Oracle Score Miss Residual Diagnostics

## Scorer

`constant_grid_rolewise_no_multi_u0`, active:weak `1:1`, constants `0.25,0.5,1.0,2.0,4.0`; active slice `[72,90)`, weak slice `[90,186)`.

No formal eval or P2 eval integration was run.

## Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 7 |
| Constant mismatch count | 4 |

Selected source counts: `{"current_expanded": 4, "p2_admitted": 3}`
Dominant gap segment counts: `{"active": 3, "mixed": 1, "near_tie": 2, "weak": 1}`
Classification counts: `{"active_trap": 3, "current_candidate_trap": 1, "near_tie": 2, "weak_trap": 1}`
Near-gap counts: `{"0.005": 0, "0.01": 0, "0.05": 0, "0.1": 2}`
Score gap summary: `{"max": 1.4305316842684028, "median": 0.16399053942537534, "min": 0.06083171418627509}`
Active gap summary: `{"max": 0.7695548795077873, "median": 0.18516699840619022, "min": 0.0235337161186768}`
Weak gap summary: `{"max": 0.6609768047606153, "median": 0.05786285456016643, "min": -0.11151582255864523}`

## Per-Sample Comparison

| Sample | Selected source | P2 rank | Score gap | Active gap | Weak gap | Dominant | Classification |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | p2_admitted | 4 | 0.163991 | 0.023534 | 0.140457 | weak | weak_trap |
| 2 | current_expanded | 3 | 0.123196 | 0.065333 | 0.057863 | mixed | current_candidate_trap |
| 6 | current_expanded | 2 | 0.060832 | 0.114242 | -0.053410 | near_tie | near_tie |
| 7 | p2_admitted | 2 | 0.073651 | 0.185167 | -0.111516 | near_tie | near_tie |
| 15 | current_expanded | 10 | 0.291663 | 0.255418 | 0.036244 | active | active_trap |
| 25 | current_expanded | 32 | 1.430532 | 0.769555 | 0.660977 | active | active_trap |
| 29 | p2_admitted | 14 | 0.568912 | 0.374191 | 0.194721 | active | active_trap |

## Top Residual Dimensions

- Active selected-advantage frequency: `{"72": 2, "76": 5, "78": 1, "80": 1, "82": 2, "86": 1, "88": 2}`
- Weak selected-advantage frequency: `{"115": 1, "116": 1, "132": 1, "133": 2, "136": 3, "137": 3, "160": 1, "161": 1, "182": 1}`

## Decision

Decision `B`: Mostly active/weak residual traps; do not integrate P2 yet, diagnose residual behavior on these cases only.

No formal eval or P2 eval integration is recommended unless residual diagnostics later support it.
