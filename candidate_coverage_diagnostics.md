# Candidate Coverage Diagnostics

Date: 2026-06-18

Scope: candidate identity coverage for the existing 32-sample setup. This is not a formal rerank eval and does not compute fingerprint scores.

## Executive Summary

- Current selected baseline: `5/32`.
- Current oracle ceiling: `9/32`.
- Reconstructed full-oracle candidate coverage: `9/32`.
- Oracle-absent samples: `23/32`.
- Main missing piece among oracle-absent samples: `drift`.
- Near-miss side coverage among oracle-absent samples: `17/23`.

## Aggregate Coverage

| Category | Count |
| --- | ---: |
| Full oracle present | 9 |
| Scorer selected oracle | unknown |
| Oracle present but selected missed | unknown |
| Drift-only present | 0 |
| Diffusion-only present | 17 |
| Drift+diffusion both present separately but not paired | 6 |
| Neither side present | 0 |

## Source Breakdown

| Source | Candidate rows | Exact full oracle samples | Exact drift samples | Exact diffusion samples |
| --- | ---: | ---: | ---: | ---: |
| beam | 256 | 6 | 15 | 32 |
| sampling | 46 | 1 | 3 | 8 |
| pair | 103 | 2 | 7 | 17 |
| unknown | 0 | 0 | 0 | 0 |

## Oracle-absent Samples

| Sample | Truth drift | Truth diffusion | Nearest drift candidates | Nearest diffusion candidates | Missing side | Recommendation |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 3 | `mul CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | d=0: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 5 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 8 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 11 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | d=0: `mul mul CONSTANT CONSTANT x_0`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 23 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | d=0: `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul mul CONSTANT CONSTANT x_0` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT abs x_0`<br>d=1: `add CONSTANT mul CONSTANT abs x_0`<br>d=1: `mul CONSTANT sqrt abs x_0` | drift | improve drift span generation/diversity |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | d=0: `mul CONSTANT sqrt abs x_0`<br>d=1: `mul CONSTANT abs x_0`<br>d=2: `add CONSTANT mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | drift | improve drift span generation/diversity |
| 30 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=0: `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0`<br>d=3: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT x_0`<br>d=2: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |
| 31 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | d=0: `mul CONSTANT mul x_0 sub CONSTANT x_0`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0`<br>d=3: `mul mul CONSTANT CONSTANT CONSTANT` | d=0: `mul CONSTANT x_0`<br>d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul CONSTANT abs x_0` | pairing | improve pairing/ranking of exact drift and diffusion spans |

## Recommendation

The dominant issue is missing drift spans. Prioritize drift candidate generation.

## Command Context

- Source log: `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`
- Checkpoint loaded read-only: `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`
- Candidate setup: beam `8`, sampling `8`, pair `8`.
