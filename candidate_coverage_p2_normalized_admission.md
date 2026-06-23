# P2 Normalized Drift Admission Coverage Hook

Date: 2026-06-23

Scope: coverage-only runner over existing JSON artifacts. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.

## Summary

- Total samples analyzed: `32`.
- Target oracle-absent samples analyzed: `17`.
- Current expanded full oracle coverage: `15/32`.
- Canonicalized expanded-pool coverage: `23/32`.
- P2 normalized admission coverage: `30/32`.
- Full normalized admission coverage: `30/32`.
- Newly recovered samples: `0,2,6,7,15,25,29`.
- Remaining missing samples: `16,28`.
- Sampling excluded: `True`; sampling recovered canonical drifts: `0`.

## Pair Pressure

- Pre-admission target pair count: `370`.
- P2 pair count: `340`.
- Full normalized admission pair count: `1105`.
- Pair-pressure reduction versus full: `765` (`69.2%`).
- Truth-aware upper bound pair count: `320`.

## Breakdowns

- Source breakdown: `{"beam": 15}`.
- Family breakdown: `{"linear drift": 6, "nested-mul linear drift": 3, "other": 0, "polynomial-like drift": 0, "sin drift": 6}`.

## Admitted Candidates

| Sample | Family | Admitted representatives |
| ---: | --- | --- |
| 0 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 1 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 2 | nested-mul linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 3 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 4 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 6 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 7 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 9 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 10 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 14 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 15 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 16 | polynomial-like drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 20 | sin drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 22 | linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 25 | nested-mul linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 28 | polynomial-like drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |
| 29 | nested-mul linear drift | `mul CONSTANT mul x_0 sub CONSTANT x_0` (other, beam:1)<br>`mul CONSTANT x_0` (linear drift, beam:5)<br>`mul CONSTANT sin x_0` (sin drift, beam:15) |

## Safety And Self-Checks

- Unsafe collision flag: `False`.
- Nonconstant structural merge detected: `False`.
- Polynomial-like truth collides with linear-like: `False`.
- Canonicalizer self-checks: `{"const_pow2_not_const_x0": true, "pow2_x0_not_x0": true, "sin_x0_not_x0": true}`.
- Collision groups: `9`.
- Hook self-check failures: `[]`.

This is token-structure-only validation, not semantic fingerprint/rerank validation.

No formal eval is recommended.
