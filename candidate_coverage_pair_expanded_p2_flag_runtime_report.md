# Candidate Coverage P2 Flag Runtime Smoke

Date: 2026-06-24

Scope: small local coverage-only candidate coverage smoke for the explicit
`--normalized-drift-admission p2_one_per_family` flag.

## Summary

- Status: `blocked`.
- Decision: `D` - blocked by missing source log.
- Missing required input: `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`.
- Model-backed candidate coverage run: `not_run`.
- Runtime output produced: `candidate_coverage_pair_expanded_p2_flag.md/json` not produced.
- Default behavior remains intended unchanged when the flag is omitted: `true`.
- No formal eval is recommended.

## Expected Command

Run locally after restoring or recreating the source log:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_pair_expanded_p2_flag.md \
  --json-output candidate_coverage_pair_expanded_p2_flag.json \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --pair-drift-diffusion-candidates 32 \
  --normalized-drift-admission p2_one_per_family \
  --normalized-drift-admission-source beam \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  > /private/tmp/gensr_sde_candidate_coverage_pair_expanded_p2_flag.log 2>&1
```

## Metrics

Runtime metrics are unavailable because the smoke was not run.

Expected reference metrics from prior JSON-only validation:

| Metric | Expected |
| --- | ---: |
| Expanded-pairing coverage | `15/32` |
| Canonicalized expanded-pool coverage | `23/32` |
| P2 normalized admission coverage | `30/32` |
| P2 pair count | `340` |
| Newly recovered samples | `0,2,6,7,15,25,29` |
| Remaining missing samples | `16,28` |
| Sampling excluded | `true` |
| Sampling recovered canonical drift count | `0` |
| Unsafe collision flag | `false` |

## Recommendation

Do not regenerate the missing source log inside Codex. Restore or recreate
`/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`, then run the exact
coverage-only command above locally. No formal eval is recommended.
