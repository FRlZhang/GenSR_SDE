# P2 Admitted Candidate Scoring Diagnostics

## Summary

| Item | Value |
| --- | ---: |
| Target samples analyzed | 7 |
| P2 oracle present in coverage JSON | 7 |
| P2 oracle selected by current scorer | unavailable |
| P2 oracle score misses | unavailable |
| Parse failures | unavailable |
| Fingerprint failures | unavailable |
| Median oracle rank | 5 |
| Max oracle rank | 5 |

## Schema Readiness

| Field group | Status |
| --- | --- |
| Target fingerprint / `y_to_fit` | missing |
| Current expanded candidates | available_as_template_tokens |
| P2 admitted candidates | available |
| Paired diffusion candidates | available_as_exact_diffusion_presence_and_tokens |
| Candidate source labels | available |
| Canonicalized candidate tokens | available_for_p2_admitted_candidates |
| Raw candidate tokens | available_for_p2_admitted_drift_candidates |
| Existing semantic scores | missing |

Semantic scoring is blocked because the coverage JSONs do not contain the target
`y_to_fit` / fingerprint vector, nor enough raw eval-sample information with
numeric constants to reconstruct the exact target fingerprint used by
`constant_grid_rolewise_no_multi_u0`. The available `score` fields are
candidate/model scores from coverage generation, not fingerprint distances.

Missing fields needed for this diagnostic:

- per-sample target `y_to_fit` or target fingerprint vector used by `score_candidate_system`
- or raw normalized eval sample plus numeric constants sufficient to reconstruct the exact target fingerprint
- semantic fingerprint distances for current expanded candidates, if scores are to be reused instead of recomputed
- full P2 paired candidate rows with scorer-ready full tokens, source, source_rank, and candidate_rank

## Per-Sample Readiness

| Sample | Family | Truth drift | Truth diffusion | Exact diffusion | P2 oracle present | P2 source | P2 rank | Current candidates | Failure reason |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 0 | linear drift | `mul CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 2 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 6 | linear drift | `mul CONSTANT x_0` | `mul CONSTANT x_0` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 7 | linear drift | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 15 | linear drift | `mul CONSTANT x_0` | `mul CONSTANT x_0` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 25 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT abs x_0` | yes | yes | beam | 5 | 20 | insufficient_schema |
| 29 | nested-mul linear drift | `mul mul CONSTANT CONSTANT x_0` | `mul CONSTANT CONSTANT` | yes | yes | beam | 5 | 20 | insufficient_schema |

## Interpretation

The P2 admission artifacts are coverage-ready for the seven newly recovered
samples, and all seven have exact diffusion present plus a P2 canonical
oracle drift from beam. They are not scoring-ready from JSON alone.
Running semantic rerank-readiness would require logging target fingerprints
or exact normalized eval samples alongside candidate rows. No formal eval is
recommended.

## Decision

**D. Logs are still insufficient.** Add minimal target-fingerprint /
candidate semantic-score logging before any scoring or eval integration
decision. Do not run a model-backed smoke unless explicitly requested.
