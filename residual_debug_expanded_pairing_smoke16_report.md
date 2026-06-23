# Expanded Pairing Residual Debug Smoke 16 Report

Date: 2026-06-23

## Status

Completed by user locally with:

```bash
bash scripts/run_expanded_pairing_residual_debug_smoke16.sh
```

This was a 16-sample smoke plus offline safety parse, not a formal 32-sample
eval, 64-sample eval, active/weak grid, retraining run, scorer change,
candidate-generation change, or rerank heuristic.

## Prepared Local Script

The local command is:

```bash
bash scripts/run_expanded_pairing_residual_debug_smoke16.sh
```

The script runs exactly:

1. one 16-sample expanded-pairing residual-debug smoke;
2. one offline safety parse with `scripts/analyze_residual_debug_safety.py`.

It writes logs and JSON to:

```text
/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.log
/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.json
/private/tmp/gensr_sde_residual_debug_safety_expanded_pairing_smoke16.log
residual_debug_safety_expanded_pairing_smoke16.md
residual_debug_safety_expanded_pairing_smoke16.json
```

## Requested Metrics

| Item | Value |
| --- | ---: |
| Smoke completed | yes |
| Samples logged | 16 |
| Candidates logged | 350 |
| Parse failures | 0 |
| Fingerprint failures | 0 |
| Samples with oracle candidates | 5 |
| Selected hits | 0 |
| Oracle-present selected misses | 5 |
| Pair-oracle samples | 4 |
| Active residual vector length | 18 |
| Weak residual vector length | 96 |
| Offline selected hits | 0 |
| Selected hits harmed | 0 |
| Oracle-present selected misses rescued | 0 |
| Offline selected pair candidates | 8 |

The `selected-hit harms=0` count is not strong safety evidence because the
expanded-pairing smoke had 0 selected hits to harm.

Oracle ranks under the offline `per_active_dim_norm_plus_weak` score:

```text
sample 5: oracle rank 8
sample 8: oracle rank 6
sample 11: oracle rank 3
sample 12: oracle rank 3
sample 13: oracle rank 11
```

## Baseline Comparison

Previous baseline 16-sample residual-debug smoke:

```text
samples logged=16
candidates logged=201
samples with oracle=2
selected hits=1
pair oracle samples=1
selected-hit harms=0
oracle-present selected-miss rescues=0
```

## Recommendation

Do not add `per_active_dim_norm_plus_weak` as a scorer mode and do not run a
formal 32-sample eval for this scorer idea. Expanded pairing improved oracle
availability from 2/16 to 5/16 and pair-oracle availability from 1/16 to 4/16,
but selected recovery dropped from 1/16 to 0/16, and offline normalization
rescued 0/5 oracle-present selected misses. Stop scorer-side normalization for
now and return to drift span diversity or deeper fingerprint ambiguity
diagnostics.
