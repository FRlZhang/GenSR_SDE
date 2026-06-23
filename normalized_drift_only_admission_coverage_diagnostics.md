# Normalized Drift-Only Admission Coverage Diagnostics

Date: 2026-06-23

Scope: offline JSON-only coverage diagnostic for a candidate pool equal to the current expanded-pairing pool plus normalized drift-only tail candidates for the 17 oracle-absent samples. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production candidate-generation default changes were run.

## Executive Summary

- Total samples analyzed: `32`.
- Target oracle-absent samples analyzed: `17`.
- Current expanded full oracle coverage: `15/32`.
- Canonicalized expanded-pool coverage: `23/32` (`+8`).
- Exact-tail admission coverage: `18/32` (`+3`).
- Normalized drift-only admission coverage: `30/32` (`+15`).
- Newly recovered beyond current expanded canonical pool: `0, 2, 6, 7, 15, 25, 29`.
- Remaining missing samples: `16, 28`.
- Drift-only normalized source breakdown: beam=15.
- Pair count estimate, raw current before vs canonical after admission: `370` -> `1105` (`+735`, `198.6%`).
- Collision unsafe flag: `False`.
- Decision: `B` - Normalized admission is coverage-positive but pair pressure is high.

## Coverage

| Mode | Coverage | Gain | Samples | Source breakdown |
| --- | ---: | ---: | --- | --- |
| Current expanded pool | 15/32 | - | - | - |
| Current expanded pool + canonicalized drift | 23/32 | +8 | - | beam+pair=7, both=1 |
| Exact drift-only tail admission | 18/32 | +3 | - | beam=3 |
| Normalized drift-only admission | 30/32 | +15 | `0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 20, 22, 25, 29` | beam=15 |

All recovered normalized-admission samples already have exact diffusion present in the expanded coverage JSON.

## Family Breakdown

| Family | Normalized recovered | Newly beyond current canonical | Remaining missing |
| --- | ---: | ---: | ---: |
| linear drift | 6 | 4 | 0 |
| sin drift | 6 | 0 | 0 |
| nested-mul linear drift | 3 | 3 | 0 |
| polynomial-like drift | 0 | 0 | 2 |
| constant drift | 0 | 0 | 0 |
| other / unknown | 0 | 0 | 0 |

## Per-Sample Table

| Sample | Truth drift | Canonical truth | Exact diffusion | Current full | Current canonical full | Normalized admitted full | Drift-only source | Drift counts before/after | Pair counts before/after |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 4/3 -> 22/13 | 20 -> 65 |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 3 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | True | beam | 5/4 -> 23/13 | 25 -> 65 |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 5/4 -> 23/13 | 25 -> 65 |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 4/3 -> 22/13 | 20 -> 65 |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 4/3 -> 22/13 | 20 -> 65 |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 5/4 -> 23/13 | 25 -> 65 |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | True | False | False | False | neither | 5/4 -> 23/13 | 25 -> 65 |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT sin x_0` | True | False | True | True | beam | 4/3 -> 22/13 | 20 -> 65 |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | True | False | True | True | beam | 5/4 -> 23/13 | 25 -> 65 |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT pow2 x_0` | True | False | False | False | neither | 5/4 -> 23/13 | 25 -> 65 |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | True | False | False | True | beam | 4/3 -> 23/13 | 20 -> 65 |

## Candidate / Pair Pressure

- Raw expanded drift count, target sum: `74`.
- Canonical expanded drift count from serialized current rows, target sum: `57`.
- Raw drift-only admitted count from serialized tail rows: `366`.
- Canonical drift-only admitted count: `221`.
- Combined raw drift count estimate: `387`.
- Combined canonical drift count: `221`.
- Pair count before, raw expanded drift x raw diffusion: `370`.
- Pair count after, raw estimate: `1935`.
- Pair count after, canonical drift x raw diffusion: `1105`.
- Drift-only tail canonical dedup reduction: `39.6%`.

Top samples by canonical pair-count increase:

| Sample | Before raw pairs | After canonical pairs | Increase |
| ---: | ---: | ---: | ---: |
| 0 | 20 | 65 | 45 |
| 1 | 20 | 65 | 45 |
| 2 | 20 | 65 | 45 |
| 6 | 20 | 65 | 45 |
| 7 | 20 | 65 | 45 |
| 9 | 20 | 65 | 45 |
| 10 | 20 | 65 | 45 |
| 15 | 20 | 65 | 45 |

Canonicalization deduplicates many over-nested templates, but normalized admission still increases pair pressure substantially relative to the current expanded pool.

## Collision / Overmerge Safety

- Unsafe collision flag: `False`.
- Collision groups: `9`.
- Any nonconstant structural merge detected: `False`.
- Polynomial-like truth collides with linear-like candidates: `False`.
- Required safety checks: `{"const_pow2_not_const_x0": true, "pow2_x0_not_x0": true, "sin_x0_not_x0": true}`.

Largest collision groups:

| Canonical drift | Raw count | Examples |
| --- | ---: | --- |
| `CONSTANT` | 3 | `mul CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT CONSTANT`<br>`mul mul CONSTANT CONSTANT mul CONSTANT CONSTANT` |
| `mul CONSTANT x_0` | 2 | `mul mul CONSTANT CONSTANT mul CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt sqrt abs x_0` |
| `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` | 2 | `mul CONSTANT mul x_0 sub CONSTANT sqrt abs x_0`<br>`mul mul CONSTANT CONSTANT mul x_0 sub CONSTANT sqrt abs x_0` |

Unresolved risk: this is token-structure-only validation, not semantic fingerprint or selected-rerank validation.

## Decision

`B. Normalized admission is coverage-positive but pair pressure is high`: Use a smaller coverage-only admission rule, such as one canonical representative per drift family/source/rank bucket or a per-sample cap. No formal eval.

No formal eval is recommended.

## Limitations

- Current expanded-pool canonical counts use serialized raw drift rows available in `candidate_coverage_pair_expanded.json`, primarily `drift_nearest_detailed`; the full raw expanded candidate list is not serialized.
- Diffusion spans are not canonicalized; pair estimates use the raw diffusion count from the expanded coverage JSON.
- The diagnostic simulates oracle coverage/admission only and does not evaluate ranking or selected recovery.
