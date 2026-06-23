# Capped Normalized Drift Admission Diagnostics

Date: 2026-06-23

Scope: offline JSON-only comparison of fixed capped normalized drift-only admission policies. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.

## Executive Summary

- Total samples analyzed: `32`.
- Target oracle-absent samples analyzed: `17`.
- Baseline current expanded full oracle coverage: `15/32`.
- Baseline canonicalized expanded-pool coverage: `23/32`.
- Baseline full normalized drift-only admission coverage: `30/32`.
- Baseline full normalized admission target pair count: `1105`; pre-admission target pair count: `370`.
- Best practical policy: `P2_one_per_family` with coverage `30/32` and pair count `340`.
- Truth-aware upper bound: coverage `30/32` and pair count `320`.
- Sampling can be dropped in this diagnostic: `True`.
- Collision unsafe flag: `False`.
- Decision: `A` - A capped practical rule preserves nearly all coverage with much lower pair pressure.

## Policy Comparison

| Policy | Coverage | Recovered targets | Newly beyond current canonical | Remaining missing | Admitted raw/canonical | Pair count | Pair reduction vs full | Coverage retained vs full | Sampling drop |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `P0_full_admission` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 366/221 | 1105 | 0 (0.0%) | 100.0% | True |
| `P1_one_per_canonical_form` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 221/221 | 1105 | 0 (0.0%) | 100.0% | True |
| `P2_one_per_family` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 51/51 | 340 | 765 (69.2%) | 100.0% | True |
| `P3_top_1_canonical_per_sample` | 23/32 | `1,3,4,9,10,14,20,22` | `` | `0,2,6,7,15,16,25,28,29` | 17/17 | 285 | 820 (74.2%) | 76.7% | True |
| `P3_top_2_canonical_per_sample` | 23/32 | `1,3,4,9,10,14,20,22` | `` | `0,2,6,7,15,16,25,28,29` | 34/34 | 370 | 735 (66.5%) | 76.7% | True |
| `P3_top_3_canonical_per_sample` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 51/51 | 425 | 680 (61.5%) | 100.0% | True |
| `P3_top_5_canonical_per_sample` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 85/85 | 595 | 510 (46.2%) | 100.0% | True |
| `P4_recovery_targeted_oracle_upper_bound` | 30/32 | `0,1,2,3,4,6,7,9,10,14,15,20,22,25,29` | `0,2,6,7,15,25,29` | `16,28` | 15/15 | 320 | 785 (71.0%) | 100.0% | True |
| `P5_rank_bucket_representative` | 23/32 | `1,3,4,9,10,14,20,22` | `` | `0,2,6,7,15,16,25,28,29` | 51/51 | 370 | 735 (66.5%) | 76.7% | True |

## Best Practical Policy

`P2_one_per_family` is the best practical policy under the priority order: it preserves `30/32` coverage, admits `51` canonical drift representatives, and reduces pair pressure by `765` pairs versus full normalized admission.

Truth-aware upper bound comparison:

- Upper-bound policy: `P4_recovery_targeted_oracle_upper_bound`.
- Upper-bound coverage: `30/32`.
- Upper-bound pair count: `320`.
- Best practical extra pair pressure over upper bound: `20`.

## Family And Source Breakdown

| Policy | Source breakdown | Linear | Sin | Nested-mul linear | Polynomial-like | Other |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `P0_full_admission` | beam=15 | 6 | 6 | 3 | 0 | 0 |
| `P1_one_per_canonical_form` | beam=15 | 6 | 6 | 3 | 0 | 0 |
| `P2_one_per_family` | beam=15 | 6 | 6 | 3 | 0 | 0 |
| `P3_top_1_canonical_per_sample` | current_expanded=8 | 2 | 6 | 0 | 0 | 0 |
| `P3_top_2_canonical_per_sample` | current_expanded=8 | 2 | 6 | 0 | 0 | 0 |
| `P3_top_3_canonical_per_sample` | beam=9, current_expanded=6 | 6 | 6 | 3 | 0 | 0 |
| `P3_top_5_canonical_per_sample` | beam=9, current_expanded=6 | 6 | 6 | 3 | 0 | 0 |
| `P4_recovery_targeted_oracle_upper_bound` | beam=15 | 6 | 6 | 3 | 0 | 0 |
| `P5_rank_bucket_representative` | current_expanded=8 | 2 | 6 | 0 | 0 | 0 |

## Per-Sample Table

| Sample | Truth drift | Canonical truth | Exact diffusion | Current canonical full | Full normalized full | P1 | P2 | P3 k=1/2/3/5 | P4 upper | P5 | Pair counts P1/P2/P3-5/P4/P5 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/15/20 |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 3 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/20/25 |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/20/25 |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/15/20 |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/15/20 |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/20/25 |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | True | False | False | N | N | N/N/N/N | N | N | 65/20/35/20/25 |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/15/20 |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | True | True | Y | Y | Y/Y/Y/Y | Y | Y | 65/20/35/20/25 |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | True | False | False | N | N | N/N/N/N | N | N | 65/20/35/20/25 |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | Y | Y | N/N/Y/Y | Y | N | 65/20/35/20/20 |

## Collision / Overmerge Safety

- Unsafe collision flag: `False`.
- Nonconstant structural merge detected: `False`.
- Polynomial-like truth collides with linear-like candidates: `False`.
- Required safety checks: `{"const_pow2_not_const_x0": true, "pow2_x0_not_x0": true, "sin_x0_not_x0": true}`.
- Collision groups: `9`.

| Canonical drift | Raw count | Examples |
| --- | ---: | --- |
| `CONSTANT` | 3 | `mul CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT mul CONSTANT CONSTANT` |
| `mul CONSTANT x_0` | 2 | `mul mul CONSTANT CONSTANT mul CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` |

This is token-structure-only validation, not semantic fingerprint or selected-rerank validation.

## Decision

`A. A capped practical rule preserves nearly all coverage with much lower pair pressure`: Use `P2_one_per_family` for a future small coverage-only implementation hook. No formal eval.

No formal eval is recommended.
