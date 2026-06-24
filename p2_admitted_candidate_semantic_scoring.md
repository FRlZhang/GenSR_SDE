# P2 Admitted Candidate Semantic Scoring

## Scorer

`constant_grid_rolewise_no_multi_u0`, active:weak `1:1`, constants `0.25,0.5,1.0,2.0,4.0`, with separate drift/diffusion constants and `multi_u0` excluded.

Lower score is better. No formal eval was run.

## Summary

| Item | Value |
| --- | ---: |
| Target samples analyzed | 7 |
| Candidate rows analyzed | 245 |
| Valid candidates | 245 |
| Parse failures | 0 |
| Fingerprint failures | 0 |
| P2 oracle present | 7 |
| P2 oracle selected | 0 |
| P2 oracle score misses | 7 |
| Combined selected from P2 | 3 |
| Combined selected from current | 4 |
| Median P2 oracle rank | 4 |
| Max P2 oracle rank | 32 |

Near-gap counts:

- `<= 0.005`: 0
- `<= 0.01`: 0
- `<= 0.05`: 0
- `<= 0.1`: 2

Score gap summary: `{"max": 1.4305316842684028, "median": 0.16399053942537534, "min": 0.06083171418627509}`
Active gap summary: `{"max": 0.7695548795077873, "median": 0.18516699840619022, "min": 0.0235337161186768}`
Weak gap summary: `{"max": 0.6609768047606153, "median": 0.05786285456016643, "min": -0.11151582255864523}`

## Per-Sample P2 Oracle

| Sample | Selected source | P2 oracle rank | P2 oracle score | Selected score | Gap | Near <=0.10 | Miss reason |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | p2_admitted | 4 | 0.49762605172014723 | 0.3336355122947719 | 0.16399053942537534 | False | p2_oracle_present_but_score_miss |
| 2 | current_expanded | 3 | 0.7128264188136054 | 0.5896305932145873 | 0.1231958255990181 | False | p2_oracle_present_but_score_miss |
| 6 | current_expanded | 2 | 0.5915156996754629 | 0.5306839854891878 | 0.06083171418627509 | True | p2_oracle_present_but_score_miss |
| 7 | p2_admitted | 2 | 1.359635215981526 | 1.2859840401339808 | 0.0736511758475451 | True | p2_oracle_present_but_score_miss |
| 15 | current_expanded | 10 | 0.6908803725905245 | 0.39921778700494825 | 0.29166258558557623 | False | p2_oracle_present_but_score_miss |
| 25 | current_expanded | 32 | 2.295592326606074 | 0.865060642337671 | 1.4305316842684028 | False | p2_oracle_present_but_score_miss |
| 29 | p2_admitted | 14 | 1.5295500008551777 | 0.9606380659615013 | 0.5689119348936764 | False | p2_oracle_present_but_score_miss |

## Decision

Decision `B`: P2 oracle candidates are present but mostly lose under current scorer; diagnose residual/segment behavior before eval integration.

No formal eval is recommended unless a later small eval-integration smoke improves selected behavior.
