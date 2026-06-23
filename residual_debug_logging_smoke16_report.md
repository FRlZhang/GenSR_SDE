# Residual Debug Logging Smoke 16 Report

Date: 2026-06-23

## Scope

This was a single 16-sample eval-only smoke using the existing
`--rerank-residual-debug-json` logging path and the restored 2000-step
checkpoint. It was not a formal 32-sample eval, 64-sample eval, active/weak
grid, retraining run, scorer change, candidate-generation change, or rerank
heuristic.

## Smoke Output

```text
/private/tmp/gensr_sde_residual_debug_smoke16.log
/private/tmp/gensr_sde_residual_debug_smoke16.json
```

| Item | Value |
| --- | ---: |
| Samples logged | 16 |
| Candidates logged | 201 |
| Valid candidates | 201 |
| Parse failures | 0 |
| Fingerprint failures | 0 |
| Active residual vector length | 18 |
| Weak residual vector length | 96 |
| Samples with oracle candidate | 2 |
| Selected oracle hits | 1 |
| Oracle-present selected misses | 1 |
| Pair-oracle samples | 1 |
| Selected exact/relaxed | 1/16 |
| Oracle exact/relaxed | 2/16 |
| Pair oracle exact/relaxed | 1/16 |

## Offline Safety Parse

```text
residual_debug_safety_smoke16.md
residual_debug_safety_smoke16.json
/private/tmp/gensr_sde_residual_debug_safety_smoke16.log
```

The safety helper parsed the residual-debug JSON and reranked candidates with
the offline `per_active_dim_norm_plus_weak` diagnostic score.

| Item | Value |
| --- | ---: |
| Original selected hits | 1 |
| Offline selected hits | 1 |
| Selected hits preserved | 1 |
| Selected hits harmed | 0 |
| Selected misses rescued | 0 |
| Offline selected pair candidates | 2 |

Per-sample safety signal:

- Sample 12 had an oracle and pair oracle; the original selected hit was
  preserved by the offline diagnostic score.
- Sample 13 had an oracle but was a selected miss; the offline diagnostic score
  also missed it, with the oracle ranked 9.

## Interpretation

The 16-sample smoke is more informative than the earlier 8-sample smoke because
it contains oracle and selected-hit cases. It provides a small positive safety
check for residual logging and offline normalization: the one selected hit was
preserved, including the pair-oracle case.

It does not justify implementing `per_active_dim_norm_plus_weak` as a rerank
mode or launching a formal 32-sample eval. The diagnostic rescued 0/1
oracle-present selected misses, so the earlier offline miss-only ablation signal
is still not enough evidence for a scorer change.

## Recommendation

Do not add a new scorer or run a formal eval from this smoke alone. If scorer
safety remains the priority, the smallest next check is one future 16-sample
residual-debug smoke with expanded pairing, then rerun the same offline safety
helper. Otherwise, return to drift span diversity or deeper fingerprint
ambiguity diagnostics.
