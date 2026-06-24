# P2 Scorer-Ready Logging Validation

## Summary

| Item | Value |
| --- | ---: |
| Sidecar JSON produced | True |
| Target samples included | 0,2,6,7,15,25,29 |
| Sample count | 7 |
| Target vector present | 7/7 |
| Fingerprint length(s) | 186 |
| P2 oracle rows | 7 |
| P2 oracle present | 7/7 |
| Exact diffusion present | 7/7 |
| P2 oracle source beam | 7/7 |
| P2 oracle rank 5 | 7/7 |
| Full sequence fields present | 7/7 |
| Parse-ready rows present | 7/7 |
| Semantic scoring run | False |

Candidate source rows:

- `current_expanded`: 140
- `p2_admitted`: 105

## Decision

Decision `A`: Next step can be offline semantic scoring of P2 admitted candidates.

No semantic scoring or formal eval was run.
