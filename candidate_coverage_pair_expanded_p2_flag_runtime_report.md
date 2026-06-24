# Candidate Coverage P2 Flag Runtime Smoke

Date: 2026-06-24

Scope: small local coverage-only candidate coverage smoke for the explicit
`--normalized-drift-admission p2_one_per_family` flag.

## Summary

- Status: `completed`.
- Decision: `A` - runtime P2 flag smoke matches prior validation.
- Runtime command log: `/private/tmp/gensr_sde_candidate_coverage_pair_expanded_p2_flag.log`.
- Runtime output:
  - `candidate_coverage_pair_expanded_p2_flag.md`
  - `candidate_coverage_pair_expanded_p2_flag.json`
- Model-backed candidate coverage run: `yes`, coverage-only candidate generation.
- Formal rerank eval / 64-sample eval / retraining: `not_run`.
- Default behavior remains intended unchanged when the flag is omitted: `true`.
- No formal eval is recommended.

## Runtime Metrics

| Metric | Runtime value | Prior validation |
| --- | ---: | ---: |
| Total samples analyzed | `32` | `32` |
| Expanded-pairing full oracle coverage | `15/32` | `15/32` |
| Oracle-absent missing side | `drift:17` | `drift:17` |
| Canonicalized expanded-pool coverage | `23/32` | `23/32` |
| P2 normalized admission coverage | `30/32` | `30/32` |
| P2 pair count | `340` | `340` |
| Newly recovered samples | `0,2,6,7,15,25,29` | `0,2,6,7,15,25,29` |
| Remaining missing samples | `16,28` | `16,28` |
| Sampling excluded | `true` | `true` |
| Sampling recovered canonical drift count | `0` | `0` |
| Unsafe collision flag | `false` | `false` |

## Interpretation

The runtime coverage-only path now exercises the integrated candidate coverage
flag and reproduces the JSON-only P2 hook validation exactly. This confirms the
default-off `p2_one_per_family` diagnostic path is runnable in the actual
candidate coverage pipeline.

This is still oracle-coverage evidence only. It does not test selected rerank
behavior, semantic fingerprint validation after admission, or production default
behavior.

## Recommendation

Keep the P2 flag as an explicit coverage-only diagnostic path. Do not make it a
production default and do not run formal eval from this result alone.
