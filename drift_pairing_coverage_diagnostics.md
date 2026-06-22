# Drift and Pairing Coverage Diagnostics

Date: 2026-06-22

Scope: offline taxonomy over candidate coverage identities. No formal eval, no fingerprint scoring, no scorer changes.

## Executive Summary

- Current oracle ceiling: `9/32`.
- Oracle-absent samples: `23/32`.
- Drift-missing with exact diffusion present: `17/23`.
- Pairing-missing with exact drift and diffusion separate: `6/23`.
- Attack drift diversity first, then pairing/recombination coverage.

## Drift Miss Taxonomy

Diffusion-only sample indices: `0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 16, 20, 22, 25, 28, 29`.

Truth drift family distribution:

| Family | Count |
| --- | ---: |
| linear drift | 6 |
| sin drift | 6 |
| nested mul structural variant | 3 |
| polynomial-like drift | 2 |

Most common missing drift templates:

| Drift template | Count |
| --- | ---: |
| `mul x_0` | 6 |
| `mul sin x_0` | 6 |
| `mul mul x_0` | 3 |
| `mul mul pow2 x_0` | 2 |

Nearest generated drift has the right operator family in `9/17` drift-missing cases.

| Sample | Truth drift | Truth diffusion | Exact diffusion source/rank | Nearest generated drift candidates | Truth drift family | Recommended target |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | beam#1 (candidate 1); beam#6 (candidate 6); beam#7 (candidate 7) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2))<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#1 (candidate 1)) | linear drift | improve drift span generation/diversity |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | beam#1 (candidate 1); beam#6 (candidate 6); beam#7 (candidate 7) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#1 (candidate 1))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2)) | sin drift | improve drift span generation/diversity |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | beam#3 (candidate 3); beam#8 (candidate 8); sampling#2 (candidate 10) | d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, same-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#1 (candidate 1))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6)) | nested mul structural variant | improve drift span generation/diversity |
| 3 | `mul CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` | beam#2 (candidate 2); pair#1 (candidate 10); pair#3 (candidate 12) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#7 (candidate 7))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4)) | linear drift | improve drift span generation/diversity |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT CONSTANT` | beam#6 (candidate 6) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#7 (candidate 7)) | sin drift | improve drift span generation/diversity |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | beam#4 (candidate 4); pair#3 (candidate 13) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2))<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#1 (candidate 1)) | linear drift | improve drift span generation/diversity |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | beam#5 (candidate 5); sampling#2 (candidate 10) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2))<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#1 (candidate 1)) | linear drift | improve drift span generation/diversity |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | beam#2 (candidate 2); beam#6 (candidate 6); beam#7 (candidate 7) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#2 (candidate 2))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#1 (candidate 1)) | sin drift | improve drift span generation/diversity |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | beam#3 (candidate 3); beam#8 (candidate 8); sampling#1 (candidate 9) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#2 (candidate 2))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#1 (candidate 1)) | sin drift | improve drift span generation/diversity |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | beam#3 (candidate 3); sampling#1 (candidate 9) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#7 (candidate 7)) | sin drift | improve drift span generation/diversity |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | beam#4 (candidate 4); pair#4 (candidate 12) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#1 (candidate 1))<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#2 (candidate 2)) | linear drift | improve drift span generation/diversity |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT x_0` | beam#5 (candidate 5) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#1 (candidate 1))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4))<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` (mean-reverting / affine drift, other-family, beam#8 (candidate 8)) | polynomial-like drift | improve drift span generation/diversity |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT x_0` | beam#4 (candidate 4); pair#3 (candidate 12) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, same-family, beam#1 (candidate 1))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6))<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2)) | sin drift | improve drift span generation/diversity |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | beam#5 (candidate 5) | d=1: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#7 (candidate 7))<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, other-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4)) | linear drift | improve drift span generation/diversity |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | beam#3 (candidate 3); beam#8 (candidate 8); sampling#1 (candidate 9) | d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, same-family, beam#1 (candidate 1))<br>d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#2 (candidate 2))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6)) | nested mul structural variant | improve drift span generation/diversity |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT sqrt abs x_0` | beam#2 (candidate 2); beam#4 (candidate 4); beam#7 (candidate 7) | d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#2 (candidate 2))<br>d=1: `mul mul CONSTANT CONSTANT x_0` (nested mul structural variant, other-family, beam#4 (candidate 4))<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` (mean-reverting / affine drift, other-family, beam#8 (candidate 8)) | polynomial-like drift | improve drift span generation/diversity |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | beam#5 (candidate 5); sampling#1 (candidate 9) | d=1: `mul mul CONSTANT CONSTANT CONSTANT` (nested mul structural variant, same-family, beam#1 (candidate 1))<br>d=1: `mul mul CONSTANT CONSTANT sin x_0` (sin drift, other-family, beam#2 (candidate 2))<br>d=2: `mul CONSTANT CONSTANT` (constant drift, other-family, beam#6 (candidate 6)) | nested mul structural variant | improve drift span generation/diversity |

## Pairing Opportunity

Pairing-missing sample indices: `5, 8, 11, 23, 30, 31`.

| Sample | Truth drift | Truth diffusion | Exact drift rank/source | Exact diffusion rank/source | Outside drift top-k | Outside diffusion top-k | Pair cap prevents | Recommendation |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | rank 4; beam#7 (candidate 7) | rank 4; beam#5 (candidate 5) | False | False | True | increase pair_drift_diffusion_candidates to at least 19 |
| 8 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | rank 4; beam#7 (candidate 7) | rank 4; beam#5 (candidate 5) | False | False | True | increase pair_drift_diffusion_candidates to at least 19 |
| 11 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | rank 3; beam#4 (candidate 4); pair#3 (candidate 13) | rank 4; beam#5 (candidate 5) | False | False | True | increase pair_drift_diffusion_candidates to at least 17 |
| 23 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | rank 5; beam#8 (candidate 8) | rank 4; beam#5 (candidate 5) | True | False | False | increase pair_drift_topk to at least 5; combined pair rank would be 24 after top-k expansion |
| 30 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT CONSTANT` | rank 4; beam#7 (candidate 7); beam#8 (candidate 8) | rank 5; beam#5 (candidate 5); sampling#1 (candidate 9) | False | False | True | increase pair_drift_diffusion_candidates to at least 20 |
| 31 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | rank 4; beam#7 (candidate 7); beam#8 (candidate 8) | rank 4; beam#4 (candidate 4); pair#3 (candidate 13) | False | False | True | increase pair_drift_diffusion_candidates to at least 19 |

Pairing bottleneck summary:

| Bottleneck | Count |
| --- | ---: |
| increase_pair_candidate_cap | 5 |
| increase_pair_drift_topk | 1 |

## Recommendation

Minimal candidate-generation patch to try next: add a drift-focused span expansion path before reranking. The most direct low-risk version is to increase drift span extraction diversity from beam/sampling outputs and allow more drift spans into pairing, while keeping diffusion settings mostly unchanged.

Pairing should be the secondary patch: for the 6 pairing-missing samples, 5 already have exact drift and diffusion spans inside the current pair top-k but are blocked by the pair candidate cap, while 1 needs `pair_drift_topk` raised to include the exact drift span.

A future formal 32-sample eval is justified only after a candidate-generation patch increases offline full-oracle coverage above `9/32` in this diagnostic.
