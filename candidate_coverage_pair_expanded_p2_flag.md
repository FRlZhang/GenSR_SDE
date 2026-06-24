# Candidate Coverage Diagnostics

Date: 2026-06-18

Scope: candidate identity coverage for the existing 32-sample setup. This is not a formal rerank eval and does not compute fingerprint scores.

## Executive Summary

- Current selected baseline: `5/32`.
- Current oracle ceiling: `9/32`.
- Reconstructed full-oracle candidate coverage: `15/32`.
- Oracle-absent samples: `17/32`.
- Main missing piece among oracle-absent samples: `drift`.
- Near-miss side coverage among oracle-absent samples: `17/17`.

## Aggregate Coverage

| Category | Count |
| --- | ---: |
| Full oracle present | 15 |
| Scorer selected oracle | 5 |
| Oracle present but selected missed | 10 |
| Drift-only present | 0 |
| Diffusion-only present | 17 |
| Drift+diffusion both present separately but not paired | 0 |
| Neither side present | 0 |

## Source Breakdown

| Source | Candidate rows | Exact full oracle samples | Exact drift samples | Exact diffusion samples |
| --- | ---: | ---: | ---: | ---: |
| beam | 256 | 6 | 15 | 32 |
| sampling | 46 | 1 | 3 | 8 |
| pair | 413 | 8 | 15 | 32 |
| unknown | 0 | 0 | 0 | 0 |

## Oracle-absent Samples

| Sample | Truth drift | Truth diffusion | Nearest drift candidates | Nearest diffusion candidates | Missing side | Recommendation |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 3 | `mul CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | d=0: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |

## Recommendation

The dominant issue is missing drift spans. Prioritize drift candidate generation.

## P2 Normalized Drift Admission

This optional section is coverage-only and uses the reusable normalized drift admission hook. It does not run reranking, fingerprint scoring, or candidate generation beyond the current coverage run.

| Metric | Value |
| --- | ---: |
| Policy | `p2_one_per_family` |
| Source | `beam` |
| Current expanded full oracle | `15/32` |
| Canonicalized expanded-pool coverage | `23/32` |
| P2 normalized admission coverage | `30/32` |
| P2 pair count | `340` |
| Newly recovered | `0,2,6,7,15,25,29` |
| Remaining missing | `16,28` |
| Sampling excluded | `True` |
| Sampling recovered canonical drift count | `0` |
| Unsafe collision flag | `False` |

- Source breakdown: `{"beam": 15}`.
- Family breakdown: `{"linear drift": 6, "nested-mul linear drift": 3, "other": 0, "polynomial-like drift": 0, "sin drift": 6}`.

## Command Context

- Source log: `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`
- Checkpoint loaded read-only: `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`
- Candidate setup: beam `8`, sampling `8`, pair `32`.
