# Drift-Only Candidate Diversity Smoke

Date: 2026-06-23

Status: blocked by runtime budget inside Codex.

Scope: diagnostic-only drift-role beam/sampling smoke for the 17 expanded-pairing
oracle-absent samples. The helper was added and compiled, but the model-backed
run exceeded the 10-minute Codex budget and was interrupted before sample-level
metrics were produced. No formal eval, 64-sample eval, grid, retraining,
scorer change, rerank-mode change, checkpoint change, data change, target-format
change, fingerprint change, or production candidate-generation default change
was run.

## What Was Added

- `scripts/analyze_drift_only_candidate_diversity.py`
- `scripts/run_drift_only_candidate_diversity_smoke.sh`

The helper reuses the validation probe setup and current checkpoint, generates
drift-only constrained beam and stochastic sampling candidates for the target
sample indices, and writes this report plus
`drift_only_candidate_diversity_smoke.json` when it completes.

## Target Set

Target sample indices from
`expanded_pairing_oracle_absent_drift_diversity.json`:

```text
0, 1, 2, 3, 4, 6, 7, 9, 10, 14, 15, 16, 20, 22, 25, 28, 29
```

Known pre-smoke facts:

```text
target_samples=17
exact_drift_missing_in_expanded_pool=17/17
exact_diffusion_present_in_expanded_pool=17/17
pairing_missing=0/17
diffusion_missing=0/17
nearest_drift_right_family=9/17
```

Drift-family breakdown:

| Family | Count |
| --- | ---: |
| linear drift | 6 |
| sin drift | 6 |
| nested-mul linear drift | 3 |
| polynomial-like drift | 2 |
| constant drift | 0 |
| other / unknown | 0 |

## Runtime Blocker

The initial smoke command was:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_only_candidate_diversity.py \
  > /private/tmp/gensr_sde_drift_only_candidate_diversity_smoke.log 2>&1
```

It loaded the checkpoint successfully and began drift-only beam generation, but
was still running after the 10-minute budget. The run was interrupted. The log
is at:

```text
/private/tmp/gensr_sde_drift_only_candidate_diversity_smoke.log
```

The helper defaults and standalone script were then reduced to a lighter
diagnostic setting:

```text
batch_size=2
drift_max_len=16
drift_beam_size=16
drift_beam_candidates=32
drift_sample_candidates=48
sample_temperatures=0.8,1.0,1.2
sample_top_k=8
sample_top_p=0.95
```

## Fields Not Yet Available

Because the smoke did not complete, these requested fields remain unavailable:

- exact truth drift found count;
- exact truth drift rank;
- exact truth drift found by beam/sampling/both/neither;
- recovered/missing split by drift family;
- drift-only nearest candidate per sample;
- beam versus sampling unique drift diversity;
- sampling under-diversity evidence from emitted drift-only candidates;
- grammar-collapse counts from nearest drift-only candidates.

The following fields would still be unavailable even after the helper completes,
unless more invasive instrumentation is added:

- token logits or entropy;
- raw unconstrained grammar rejection counts.

The helper uses grammar-constrained drift-only decoding, so invalid emitted rows
can be counted, but raw rejection/unreachable-token statistics are not exposed.

## Decision

`D. Logs are still insufficient`: the helper exists, but the model-backed smoke
did not finish within the Codex runtime budget. Run the standalone script before
choosing between wider drift admission, drift sampling, role-conditioned drift
decoding, or template-family seeding.

Suggested command:

```bash
scripts/run_drift_only_candidate_diversity_smoke.sh
```

Do not run formal eval from this blocked report.
