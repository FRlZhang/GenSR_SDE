# Residual Debug Logging Smoke Report

Date: 2026-06-23

Scope: code/logging plus one small 8-sample eval-only smoke. This was not a
formal 32-sample eval, not a 64-sample eval, not retraining, not candidate
regeneration beyond the smoke run, and not a scorer/default change.

## Logging Added

Added optional CLI flag:

```text
--rerank-residual-debug-json PATH
```

When absent, rerank behavior is unchanged. When present, the probe writes a JSON
sidecar containing per-sample, per-candidate residual/debug records from the
existing rerank scoring path.

Candidate records include:

- sample index and truth sequence;
- selected sequence and selected/oracle labels;
- candidate sequence, drift tokens, diffusion tokens;
- candidate source, candidate rank, source rank, pair flag;
- exact/relaxed oracle flags and pair-oracle flag;
- rerank, active, weak, full, multi-u0, baseline, and shared-constant scores;
- best shared/role-wise constants;
- normalized model score;
- active residual vector and weak residual vector.

## Smoke Run

Command completed using the baseline 8-sample setup and checkpoint:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

Artifacts:

```text
/private/tmp/gensr_sde_residual_debug_smoke8.log
/private/tmp/gensr_sde_residual_debug_smoke8.json
residual_debug_safety_smoke8.md
residual_debug_safety_smoke8.json
```

Smoke metrics:

| Item | Value |
| --- | ---: |
| Samples logged | 8 |
| Candidates logged | 101 |
| Valid candidates | 101 |
| Parse failures | 0 |
| Fingerprint failures | 0 |
| Samples with oracle candidate | 0 |
| Selected hits | 0 |
| Pair-oracle samples | 0 |

The JSON contains active residual vectors of length 18 and weak residual vectors
of length 96 for logged valid candidates. Candidate-level source, rank,
selected, oracle, pair, constants, and model-score labels are present.

## Offline Safety Check

Added:

```text
scripts/analyze_residual_debug_safety.py
```

The helper parses the debug JSON only and computes the offline diagnostic score:

```text
per_active_dim_norm_plus_weak
```

Safety result on this 8-sample smoke:

| Check | Result |
| --- | ---: |
| Original selected hits | 0 |
| Offline selected hits | 0 |
| Selected hits preserved | 0 |
| Selected hits harmed | 0 |
| Selected misses rescued | 0 |
| Offline selected pair candidates | 1 |

The safety check ran, but the smoke is inconclusive because this 8-sample
baseline candidate pool contained no oracle candidates and no selected hits.

## Recommendation

The logging is sufficient for candidate-level offline safety analysis. The
8-sample smoke is too small and unlucky to judge scorer safety. Do not add a
rerank mode and do not run a formal 32-sample eval from this evidence.

Recommended next step: one future 16-sample smoke with the same residual debug
JSON flag, then rerun the offline safety helper. If that still has too few
oracle/hit cases, return to drift span diversity rather than scorer-side
normalization.
