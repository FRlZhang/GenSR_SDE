# Candidate Normalization Constant-Folding Diagnostics

Date: 2026-06-23

Scope: offline JSON-only diagnostic for applying narrow multiplicative constant-chain folding at candidate-normalization / pairing time. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.

## Executive Summary

- Total samples analyzed: `32`.
- Current expanded full oracle coverage: `15/32`.
- Canonicalized expanded-pool coverage: `23/32` (`+8`).
- Exact-tail admission coverage: `18/32` (`+3`).
- Canonicalized drift-only admission coverage: `30/32` (`+15`).
- Current expanded-pool source breakdown: beam+pair=7, both=1.
- Drift-only tail canonicalized source breakdown: beam=15.
- Collision groups: `9`; unsafe collision flag: `False`.
- Serialized raw/canonical drift count: `440` -> `278` (`36.8%` reduction).
- Decision: `B` - Drift-only tail admission plus canonicalization is needed.

## Coverage By Source

| Coverage mode | Coverage | Gain | Samples | Source breakdown |
| --- | ---: | ---: | --- | --- |
| Current expanded pool | 15/32 | - | - | - |
| Current expanded pool + canonicalized drift candidates | 23/32 | +8 | `1, 3, 4, 9, 10, 14, 20, 22` | beam+pair=7, both=1 |
| Drift-only exact tail admission | 18/32 | +3 | `2, 25, 29` | beam=3 |
| Drift-only tail + canonicalized admission | 30/32 | +15 | `0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 20, 22, 25, 29` | beam=15 |

Current expanded-pool canonicalization and drift-only tail admission are intentionally separated. The current expanded JSON supplies some serialized raw drift candidates through `drift_nearest_detailed`; drift-only tail admission uses the separate drift-only smoke JSON.

## Per-Family Breakdown

| Family | Current expanded canonical gain | Exact tail gain | Drift-only canonical gain |
| --- | ---: | ---: | ---: |
| linear drift | 2 | 0 | 6 |
| sin drift | 6 | 0 | 6 |
| nested-mul linear drift | 0 | 3 | 3 |
| polynomial-like drift | 0 | 0 | 0 |
| constant drift | 0 | 0 | 0 |
| other / unknown | 0 | 0 | 0 |

## Per-Sample Table

| Sample | Truth drift | Canonical truth | Exact drift found | Current expanded canonical found | Drift-only canonical found | Exact diffusion present | Projected full oracle after normalization |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | False (neither) | True (beam) | True | True |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (beam+pair) | True (beam) | True | True |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False (neither) | True (beam) | True | True |
| 3 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True (beam+pair) | True (beam) | True | True |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (beam+pair) | True (beam) | True | True |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | False (neither) | True (beam) | True | True |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | False (neither) | True (beam) | True | True |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (both) | True (beam) | True | True |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (beam+pair) | True (beam) | True | True |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (beam+pair) | True (beam) | True | True |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | False (neither) | True (beam) | True | True |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | False | False (neither) | False (neither) | True | False |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True (beam+pair) | True (beam) | True | True |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True (beam+pair) | True (beam) | True | True |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False (neither) | True (beam) | True | True |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | False | False (neither) | False (neither) | True | False |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False (neither) | True (beam) | True | True |

## Collision And Overmerge Risk

- Raw serialized drift expressions: `23`.
- Canonical drift expressions: `13`.
- Collision groups: `9`.
- Any collision changes nonconstant structure: `False`.
- Unsafe checks: `{"const_pow2_equals_const_x0": false, "pow2_x0_equals_x0": false, "sin_x0_equals_x0": false}`.
- Unsafe overall: `False`.

Largest collision groups:

| Canonical drift | Raw count | Raw examples |
| --- | ---: | --- |
| `CONSTANT` | 3 | `mul CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT mul CONSTANT CONSTANT` |
| `mul CONSTANT x_0` | 2 | `mul mul CONSTANT CONSTANT mul CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` |

The required unsafe cases are all false: `pow2 x_0` does not canonicalize to `x_0`, `mul CONSTANT pow2 x_0` does not canonicalize to `mul CONSTANT x_0`, and `sin x_0` does not canonicalize to `x_0`.

## Candidate Count / Dedup Impact

- Serialized raw unique drift count sum: `440`.
- Serialized canonical unique drift count sum: `278`.
- Serialized reduction: `36.8%`.
- Reported drift-only raw unique count sum from smoke metadata: `544`.
- Reported current expanded raw unique drift count sum: `143`.

Per-sample serialized drift-only tail count changes:

| Sample | Raw unique | Canonical unique | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | 22 | 13 | 40.9% |
| 1 | 21 | 13 | 38.1% |
| 2 | 22 | 13 | 40.9% |
| 3 | 22 | 13 | 40.9% |
| 4 | 21 | 13 | 38.1% |
| 6 | 22 | 13 | 40.9% |
| 7 | 22 | 13 | 40.9% |
| 9 | 21 | 13 | 38.1% |
| 10 | 21 | 13 | 38.1% |
| 14 | 21 | 13 | 38.1% |
| 15 | 22 | 13 | 40.9% |
| 16 | 21 | 13 | 38.1% |
| 20 | 21 | 13 | 38.1% |
| 22 | 22 | 13 | 40.9% |
| 25 | 22 | 13 | 40.9% |
| 28 | 21 | 13 | 38.1% |
| 29 | 22 | 13 | 40.9% |

Canonicalization reduces duplicated over-nested constant templates in the observed serialized candidates. It should reduce pair combinations rather than increase them, but this remains token-structure-only evidence, not semantic fingerprint validation.

## Decision

`B. Drift-only tail admission plus canonicalization is needed`: Run one small coverage-only smoke/admission diagnostic with normalized drift-only candidates, not formal eval.

No formal eval is recommended.

## Limitations

- Current expanded-pool canonicalization uses raw drift tokens serialized in `candidate_coverage_pair_expanded.json`, primarily `drift_nearest_detailed`; the coverage JSON does not contain every raw candidate sequence.
- Drift-only tail canonicalization uses serialized drift-only smoke candidates and nearest/exact rows.
- Collision safety is token-structure-only; semantic equivalence and selected-rerank behavior still require later diagnostics.
