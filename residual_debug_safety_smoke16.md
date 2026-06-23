# Residual Debug Safety Smoke 16

Scope: offline parse of residual debug JSON only. No model decoding, candidate generation, simulation, formal eval, retraining, or scorer change was run.

## Summary

- Input JSON: `/private/tmp/gensr_sde_residual_debug_smoke16.json`.
- Samples: `16`.
- Candidates: `201`.
- Samples with oracle: `2`.
- Samples with pair oracle: `1`.
- Original selected hits: `1`.
- Offline selected hits: `1`.
- Selected hits preserved: `1`.
- Selected hits harmed: `0`.
- Selected misses rescued: `0`.
- Offline selected pair candidates: `2`.

The smoke is only 16 samples, so absence of harm is not enough for a formal eval. It is useful for checking whether the residual logging is complete and whether a future targeted smoke is worth running.

## Per Sample

| Sample | Candidates | Oracle | Pair oracle | Original hit | Offline hit | Rescued | Harmed | Offline source | Offline oracle rank | Pair oracle rank |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 1 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 2 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 3 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 4 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 5 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 6 | 13 | no | no | no | no | no | no | sample_t=1 | n/a | n/a |
| 7 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 8 | 12 | no | no | no | no | no | no | sample_t=1.2 | n/a | n/a |
| 9 | 13 | no | no | no | no | no | no | pair | n/a | n/a |
| 10 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 11 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 12 | 13 | yes | yes | yes | yes | no | no | pair | 1 | 1 |
| 13 | 13 | yes | no | no | no | no | no | beam | 9 | n/a |
| 14 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 15 | 12 | no | no | no | no | no | no | beam | n/a | n/a |

## Recommendation

Use this logging path for a future targeted smoke before considering any scorer implementation. Do not run a formal 32-sample eval from this safety check alone.
