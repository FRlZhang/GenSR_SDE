# Drift-Only Admission and Constant-Folding Diagnostics

Date: 2026-06-23

Scope: offline JSON-only diagnostic over the completed drift-only candidate diversity smoke. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.

## Executive Summary

- Target samples analyzed: `17`.
- Exact truth drift found: `3/17`.
- Exact-tail projected full-oracle coverage: `15/32 + 3 = 18/32`.
- Canonicalized drift matches: `15/17`.
- Exact-missing cases that become canonicalized matches: `12`.
- Constant-folded projected full-oracle coverage: `15/32 + 15 = 30/32`.
- Linear/sin canonical matches: `12/12`.
- Polynomial-like missing after constant folding: `2/2`.
- Decision: `B` - Constant-folding explains most misses.

## Exact-Tail Admission

Exact truth drift appears for samples: `2, 25, 29`.

| Sample | Family | Source | Rank | Exact diffusion present |
| ---: | --- | --- | ---: | --- |
| 2 | nested-mul linear drift | beam | 30 | True |
| 25 | nested-mul linear drift | beam | 30 | True |
| 29 | nested-mul linear drift | beam | 30 | True |

All exact-tail hits are beam-only rank-30 nested-mul linear drift cases. This supports an admission ceiling of up to `+3` full-oracle candidates because exact diffusion is already present for those samples. A naive global expansion to beam top-30/top-32 may be expensive and should not become a default from this diagnostic alone.

## Constant-Folded Admission

Canonicalized drift matches appear for samples: `0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 20, 22, 25, 29`.

Among exact-missing cases, canonicalization adds `12` matches: `0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 20, 22`.

The diagnostic canonicalizer folds only multiplicative chains of `CONSTANT` factors and preserves nonconstant structure such as `sin`, `pow2`, and `x_0`. It is diagnostic-only and does not change training targets, candidate generation, reranking, or production normalization.

## Per-Family Breakdown

| Family | Exact found | Exact missing | Canonical match | Canonical missing |
| --- | ---: | ---: | ---: | ---: |
| linear drift | 0 | 6 | 6 | 0 |
| sin drift | 0 | 6 | 6 | 0 |
| nested-mul linear drift | 3 | 0 | 3 | 0 |
| polynomial-like drift | 0 | 2 | 0 | 2 |
| constant drift | 0 | 0 | 0 | 0 |
| other / unknown | 0 | 0 | 0 | 0 |

## Per-Sample Table

| Sample | Family | Truth drift | Nearest drift-only | Canonical truth | Canonical nearest | Exact found | Canonical match | Exact diffusion present |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 1 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 2 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | True | True |
| 3 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 4 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 6 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 7 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 9 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 10 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 14 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 15 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 16 | polynomial-like drift | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul mul CONSTANT CONSTANT mul CONSTANT x_0` | `mul CONSTANT pow2 x_0` | `mul CONSTANT x_0` | False | False | True |
| 20 | sin drift | `mul CONSTANT sin x_0` | `mul mul CONSTANT CONSTANT sin x_0` | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | False | True | True |
| 22 | linear drift | `mul CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | False | True | True |
| 25 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | True | True |
| 28 | polynomial-like drift | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul mul CONSTANT CONSTANT mul CONSTANT x_0` | `mul CONSTANT pow2 x_0` | `mul CONSTANT x_0` | False | False | True |
| 29 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | True | True |

## Interpretation

- Exact tail admission alone can rescue only the three nested-mul linear rank-30 beam cases.
- Constant-chain folding explains all linear and sin exact misses in the available JSON: `12/12` linear/sin cases become canonicalized matches.
- The two polynomial-like cases remain missing after constant folding because `pow2 x_0` is preserved and does not match the linear-like candidate `mul CONSTANT x_0`.
- Since exact diffusion is already present for every matched case, canonicalized drift matching projects up to `+15` full-oracle candidates over the current expanded `15/32` ceiling, for a diagnostic-only canonicalized ceiling of `30/32`.
- This is not selected recovery evidence and does not justify a formal eval by itself.

## Decision

`B. Constant-folding explains most misses`: Run a future helper-only or small smoke candidate-normalization diagnostic for constant-chain folding before widening beam/top-k. Do not run formal eval.

No formal eval is recommended.

## Notes

- drift_only_candidate_diversity_smoke.json stores nearest candidates, exact occurrences, and serialized top beam/sampling candidates; canonical matching is therefore JSON-only over available serialized candidates, not a new model run.
- The canonicalizer is intentionally narrow: it folds multiplicative `CONSTANT` chains but does not erase `pow2`, state dependence, or unrelated operator structure.
