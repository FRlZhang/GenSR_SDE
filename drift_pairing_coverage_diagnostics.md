# Drift and Pairing Coverage Diagnostics

Date: 2026-06-22

Scope: partial analysis from existing candidate coverage markdown. Exact candidate span ranks require a JSON sidecar that could not be generated because the checkpoint was missing.

## Executive Summary

- Current oracle ceiling: `9/32`.
- Drift-missing with exact diffusion present: `17`.
- Pairing-missing with exact drift and diffusion separate: `6`.
- The available evidence still points to drift diversity first, pairing second.

## Drift Miss Taxonomy

Diffusion-only sample indices: `0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 16, 20, 22, 25, 28, 29`.

| Family | Count |
| --- | ---: |
| linear drift | 6 |
| sin drift | 6 |
| nested mul linear drift | 3 |
| polynomial-like drift | 2 |

| Sample | Truth drift | Truth diffusion | Nearest generated drift candidates | Recommended target |
| ---: | --- | --- | --- | --- |
| 0 | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | improve drift span generation/diversity |
| 1 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | improve drift span generation/diversity |
| 2 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | improve drift span generation/diversity |
| 3 | `mul CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | improve drift span generation/diversity |
| 4 | `mul CONSTANT sin x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | improve drift span generation/diversity |
| 6 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | improve drift span generation/diversity |
| 7 | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | improve drift span generation/diversity |
| 9 | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | improve drift span generation/diversity |
| 10 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | improve drift span generation/diversity |
| 14 | `mul CONSTANT sin x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT CONSTANT` | improve drift span generation/diversity |
| 15 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT sin x_0` | improve drift span generation/diversity |
| 16 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | improve drift span generation/diversity |
| 20 | `mul CONSTANT sin x_0` | `mul CONSTANT x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT`<br>d=2: `mul mul CONSTANT CONSTANT CONSTANT` | improve drift span generation/diversity |
| 22 | `mul CONSTANT x_0` | `mul CONSTANT x_0` | d=1: `mul CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT x_0` | improve drift span generation/diversity |
| 25 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | improve drift span generation/diversity |
| 28 | `mul mul CONSTANT CONSTANT pow2 x_0` | `mul CONSTANT sqrt abs x_0` | d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=1: `mul mul CONSTANT CONSTANT x_0`<br>d=2: `mul CONSTANT mul x_0 sub CONSTANT x_0` | improve drift span generation/diversity |
| 29 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | d=1: `mul mul CONSTANT CONSTANT CONSTANT`<br>d=1: `mul mul CONSTANT CONSTANT sin x_0`<br>d=2: `mul CONSTANT CONSTANT` | improve drift span generation/diversity |

## Pairing Opportunity

Pairing-missing sample indices: `5, 8, 11, 23, 30, 31`.

Exact drift/diffusion ranks, top-k membership, pair-cap checks, and dedup/filtering diagnosis are blocked without the machine-readable JSON sidecar.

| Sample | Truth drift | Truth diffusion | Available inference |
| ---: | --- | --- | --- |
| 5 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |
| 8 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |
| 11 | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT x_0` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |
| 23 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |
| 30 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT CONSTANT` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |
| 31 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |

## Blocker

The existing coverage markdown does not contain candidate span ranks. A rerun of candidate generation was attempted to emit `candidate_coverage_diagnostics.json`, but the checkpoint `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth` is missing.

Smallest command after regenerating or restoring the checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_diagnostics.md \
  --json-output candidate_coverage_diagnostics.json \
  > /private/tmp/gensr_sde_candidate_coverage_json.log 2>&1

/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_pairing_coverage.py \
  --coverage-json candidate_coverage_diagnostics.json \
  --report drift_pairing_coverage_diagnostics.md \
  > /private/tmp/gensr_sde_drift_pairing_coverage.log 2>&1
```
