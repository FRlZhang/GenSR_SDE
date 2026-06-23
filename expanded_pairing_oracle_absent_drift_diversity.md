# Expanded-Pairing Oracle-Absent Drift Diversity

Date: 2026-06-23

Scope: offline parsing of existing expanded-pairing coverage JSON and reports. No model decoding, candidate regeneration, fingerprint simulation, formal eval, 64-sample eval, grid, retraining, scorer change, or candidate-generation change was run.

## Executive Summary

- Expanded-pairing full oracle coverage: `15/32`.
- Expanded-pairing oracle-absent samples analyzed: `17/32`.
- Exact-drift-missing among oracle-absent samples: `17/17`.
- Pairing-missing among oracle-absent samples: `0/17`.
- Diffusion-missing among oracle-absent samples: `0/17`.
- Exact diffusion is already present for `17/17` oracle-absent samples.
- Exact drift is present for `0/17` oracle-absent samples.
- Nearest logged drift has the right family in `9/17` cases.
- Decision: `A` - Drift spans are missing but nearby drift families appear.

## Baseline vs Expanded Check

| Metric | Baseline | Expanded |
| --- | ---: | ---: |
| Full oracle present | 9/32 | 15/32 |
| Oracle absent | 23/32 | 17/32 |
| Diffusion-only present | 17 | 17 |
| Drift+diffusion separate, not paired | 6 | 0 |

The known six baseline pairing-missing cases are no longer in the oracle-absent set. The remaining failures are all exact-drift misses with exact diffusion already present.

## Drift-Family Breakdown

| Drift family | Count |
| --- | ---: |
| linear drift | 6 |
| sin drift | 6 |
| nested-mul linear drift | 3 |
| polynomial-like drift | 2 |
| constant drift | 0 |
| other / unknown | 0 |

Nearest logged drift templates are concentrated in a few nearby families: over-nested sin/linear variants, constant-heavy `mul CONSTANT CONSTANT`, and nested-mul constants. This supports a drift span diversity issue rather than a diffusion or pairing issue.

## Source Diversity

| Source | Samples with drift spans | Unique drift templates | Avg drift templates/sample | Samples with exact/partial diffusion spans | Unique diffusion templates |
| --- | ---: | ---: | ---: | ---: | ---: |
| beam | 17 | 5 | 4.35 | 17 | 5 |
| pair | 17 | 5 | 4.35 | 17 | 5 |
| sampling | 16 | 2 | 1.41 | 16 | 2 |

Beam contributes the broadest useful drift span set in these logs. Sampling appears in most samples but mostly repeats the same small drift-template families and does not add any exact drift spans. Pairing recombines existing spans, so it improves full-expression coverage only when the exact drift span already exists; it cannot rescue these 17 cases.

## Rank And Blocker Interpretation

- Truth drift rank is `null` for all 17 samples because the exact drift span is absent from the logged candidate pool.
- Exact diffusion rank is present for all 17 samples, with ranks between the logged top source spans.
- Current logs do not contain unlogged beam/sampling tail ranks, raw token probabilities, or grammar rejection counts. Therefore they prove current-pool absence, but cannot distinguish a below-unlogged-top-k miss from a model/grammar miss.
- The repeated nearest templates indicate collapse toward over-nested or constant-heavy drift spans, especially for simple linear and sin truth drifts.

## Per-Sample Diagnostics

| Sample | Truth drift | Truth diffusion | Family | Exact drift | Exact diffusion/rank | Best logged drift | Drift source(s) | Collapse pattern | Blocker |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | linear drift | False | True / 1 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | sin drift | False | True / 1 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | nested-mul linear drift | False | True / 3 | d=1: `mul mul CONSTANT CONSTANT CONSTANT` | beam,pair,sampling | state-collapsed | exact drift absent from logged candidate sources |
| 3 | `mul CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` | linear drift | False | True / 2 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT CONSTANT` | sin drift | False | True / 5 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | linear drift | False | True / 4 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | linear drift | False | True / 5 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | sin drift | False | True / 2 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair,sampling | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | sin drift | False | True / 3 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | sin drift | False | True / 3 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | linear drift | False | True / 4 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT x_0` | polynomial-like drift | False | True / 4 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair,sampling | drops-polynomial-power | exact drift absent from logged candidate sources |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT x_0` | sin drift | False | True / 4 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair | over-nested, over-nested-sin | exact drift absent from logged candidate sources |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | linear drift | False | True / 4 | d=1: `mul CONSTANT CONSTANT` | beam,pair | state-collapsed, constant-heavy | exact drift absent from logged candidate sources |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | nested-mul linear drift | False | True / 3 | d=1: `mul mul CONSTANT CONSTANT CONSTANT` | beam,pair,sampling | state-collapsed | exact drift absent from logged candidate sources |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT sqrt abs x_0` | polynomial-like drift | False | True / 2 | d=1: `mul mul CONSTANT CONSTANT sin x_0` | beam,pair,sampling | drops-polynomial-power | exact drift absent from logged candidate sources |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | nested-mul linear drift | False | True / 5 | d=1: `mul mul CONSTANT CONSTANT CONSTANT` | beam,pair | state-collapsed | exact drift absent from logged candidate sources |

## Required Conclusions

1. Oracle-absent after expanded pairing: `17/32`.
2. Split: exact-drift-missing `17`, pairing-missing `0`, diffusion-missing `0`.
3. Drift-family breakdown: linear `6`, sin `6`, nested-mul linear `3`, polynomial-like `2`, constant `0`, other/unknown `0`.
4. Exact diffusion is already present in all `17/17` remaining cases.
5. Missing exact drift candidates are absent from all logged sources, not merely blocked by the expanded pair top-k/cap. Logs are insufficient to prove whether they would appear in unlogged beam/sampling tails.
6. Beam contributes more usable drift diversity than sampling in the emitted pool; sampling mostly duplicates nearby templates and adds no exact drift spans.
7. Sampling temperature/top-k/top-p appears under-diverse for the missing linear, sin, nested-linear, and polynomial-like drift families, but this is an emitted-candidate inference because entropy/logit fields are absent.
8. Grammar-constrained beam appears to collapse toward repeated over-nested or constant-heavy templates, especially `mul CONSTANT CONSTANT`, `mul mul CONSTANT CONSTANT CONSTANT`, and `mul mul CONSTANT CONSTANT sin x_0`.
9. Most plausible intervention: one diagnostic-only drift-only candidate diversity smoke with explicit drift-span ranks/sources. Do not run formal eval from this report alone.

## Decision

`A. Drift spans are missing but nearby drift families appear`: Run one future drift-only candidate diversity smoke that logs drift-only candidate spans/ranks before any rerank or formal eval change.

No rerank mode, scorer change, formal 32-sample eval, 64-sample eval, grid, retraining, checkpoint change, target-format change, fingerprint change, or candidate-generation logic change is recommended from this report.

## Missing Fields

- `truth_full_expression`: available
- `truth_drift_tokens`: available
- `truth_diffusion_tokens`: available
- `exact_drift_present`: available
- `exact_diffusion_present`: available
- `exact_diffusion_rank`: available
- `nearest_drift_candidate`: available
- `nearest_drift_rank_within_logged_candidates`: available
- `source_breakdown_for_logged_candidates`: available
- `unlogged_beam_or_sampling_tail_ranks`: missing
- `token_logits_or_raw_sampling_entropy`: missing
- `grammar_rejection_counts`: missing

Limitations:
- Existing coverage JSON proves exact drift is absent from the logged candidate pool, but it does not contain ranks for unlogged beam/sampling tails.
- Sampling entropy, token logits, and grammar rejection counts are not present, so sampling under-diversity and grammar collapse are inferred from emitted spans only.
