# Residual Debug Safety Smoke 8

Scope: offline parse of residual debug JSON only. No model decoding, candidate generation, simulation, formal eval, retraining, or scorer change was run.

## Summary

- Input JSON: `/private/tmp/gensr_sde_residual_debug_smoke8.json`.
- Samples: `8`.
- Candidates: `101`.
- Samples with oracle: `0`.
- Samples with pair oracle: `0`.
- Original selected hits: `0`.
- Offline selected hits: `0`.
- Selected hits preserved: `0`.
- Selected hits harmed: `0`.
- Selected misses rescued: `0`.
- Offline selected pair candidates: `1`.

The smoke is only 8 samples, so absence of harm is not enough for a formal eval. It is useful for checking whether the residual logging is complete and whether a future 16-sample smoke is worth running.

## Per Sample

| Sample | Candidates | Oracle | Pair oracle | Original hit | Offline hit | Rescued | Harmed | Offline source | Offline oracle rank | Pair oracle rank |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 1 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 2 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 3 | 12 | no | no | no | no | no | no | beam | n/a | n/a |
| 4 | 12 | no | no | no | no | no | no | pair | n/a | n/a |
| 5 | 13 | no | no | no | no | no | no | beam | n/a | n/a |
| 6 | 13 | no | no | no | no | no | no | sample_t=1 | n/a | n/a |
| 7 | 13 | no | no | no | no | no | no | beam | n/a | n/a |

## Recommendation

Use this logging path for one future 16-sample smoke before considering any scorer implementation. Do not run a formal 32-sample eval from this 8-sample safety check alone.
