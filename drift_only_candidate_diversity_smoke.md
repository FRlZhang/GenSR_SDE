# Drift-Only Candidate Diversity Smoke

Date: 2026-06-23

Scope: diagnostic-only drift-role beam/sampling smoke for the 17 expanded-pairing oracle-absent samples. It loads the current checkpoint and generates drift-only candidates, but does not run reranking, fingerprint scoring, formal eval, 64-sample eval, grids, retraining, scorer changes, or production candidate-generation changes.

## Executive Summary

- Target samples analyzed: `17`.
- Exact truth drift found in drift-only candidates: `3/17`.
- Exact truth drift missing from drift-only beam+sampling: `14/17`.
- Exact diffusion was already present in previous expanded coverage for `17/17` samples.
- Nearest drift-only candidate has the right family in `9/17` samples.
- Decision: `A` - Exact drifts appear in drift-only tails.

## Exact Drift Recovery By Source

| Source bucket | Count |
| --- | ---: |
| beam | 3 |
| sampling | 0 |
| both | 0 |
| neither | 14 |

## Drift-Family Recovery

| Family | Recovered | Missing |
| --- | ---: | ---: |
| linear drift | 0 | 6 |
| sin drift | 0 | 6 |
| nested-mul linear drift | 3 | 0 |
| polynomial-like drift | 0 | 2 |
| constant drift | 0 | 0 |
| other / unknown | 0 | 0 |

## Diversity And Collapse

- Unique beam drift candidates summed over samples: `544`.
- Unique sampling drift candidates summed over samples: `34`.
- Beam contributes more new drift diversity than sampling if its summed unique count is higher; sampling remains under-diverse if it mostly duplicates the same few templates and does not recover exact drifts.
- Grammar rejection counts and token entropy/logits are unavailable without more invasive instrumentation. Constrained decoding emitted parse-valid drift-role candidates, so raw grammar rejection versus low-probability absence cannot be separated here.

Collapse labels on nearest drift-only candidates:

| Collapse label | Count |
| --- | ---: |
| other | 3 |
| over-nested template | 14 |
| sin over-nesting | 6 |

## Per-Sample Diagnostics

| Sample | Family | Truth drift | Exact found | Source bucket | Exact rank | Nearest drift-only candidate | Nearest right family | Collapse labels | Unique beam | Unique sampling |
| ---: | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 0 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 1 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 2 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | True | beam | 30 | `mul mul CONSTANT CONSTANT x_0` | True | other | 32 | 2 |
| 3 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 4 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 6 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 7 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 9 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 10 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 14 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 15 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 16 | polynomial-like drift | `mul mul CONSTANT CONSTANT pow2 x_0` | False | neither | - | `mul mul CONSTANT CONSTANT mul CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 20 | sin drift | `mul CONSTANT sin x_0` | False | neither | - | `mul mul CONSTANT CONSTANT sin x_0` | True | over-nested template,sin over-nesting | 32 | 2 |
| 22 | linear drift | `mul CONSTANT x_0` | False | neither | - | `mul mul CONSTANT CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 25 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | True | beam | 30 | `mul mul CONSTANT CONSTANT x_0` | True | other | 32 | 2 |
| 28 | polynomial-like drift | `mul mul CONSTANT CONSTANT pow2 x_0` | False | neither | - | `mul mul CONSTANT CONSTANT mul CONSTANT x_0` | False | over-nested template | 32 | 2 |
| 29 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | True | beam | 30 | `mul mul CONSTANT CONSTANT x_0` | True | other | 32 | 2 |

## Required Conclusions

1. The report analyzes only the 17 expanded-pairing oracle-absent samples.
2. Exact truth drift found count: `3/17`.
3. Source split is shown above for beam, sampling, both, and neither.
4. Family recovery/missing split is shown above for linear, sin, nested-mul linear, polynomial-like, constant, and other/unknown.
5. Whether exact drifts are below current top-k or absent entirely is decided from the drift-only tail result: recovered cases are below/admission misses; missing cases remain absent from this drift-only beam+sampling smoke.
6. Beam versus sampling diversity is summarized by unique drift candidate counts and exact recovery source.
7. Sampling under-diversity is inferred from emitted candidates only because logits/entropy are unavailable.
8. Grammar-constrained decoding collapse is summarized by nearest-candidate collapse labels; raw rejection counts are unavailable.
9. Next intervention: Run a future candidate-coverage-only top-k / drift-only admission diagnostic; do not run formal eval.

## Missing Fields

- `token_logits_or_raw_sampling_entropy`: unavailable
- `raw_grammar_rejection_counts`: unavailable
- `invalid_unconstrained_candidates`: unavailable

No formal eval is recommended from this smoke.
