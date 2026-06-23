# Residual Debug Safety Smoke 16

Scope: offline parse of residual debug JSON only. No model decoding, candidate generation, simulation, formal eval, retraining, or scorer change was run.

## Summary

- Input JSON: `/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.json`.
- Samples: `16`.
- Candidates: `350`.
- Samples with oracle: `5`.
- Samples with pair oracle: `4`.
- Original selected hits: `0`.
- Offline selected hits: `0`.
- Selected hits preserved: `0`.
- Selected hits harmed: `0`.
- Selected misses rescued: `0`.
- Offline selected pair candidates: `8`.

The smoke is only 16 samples, so absence of harm is not enough for a formal eval. It is useful for checking whether the residual logging is complete and whether a future targeted smoke is worth running.

## Per Sample

| Sample | Candidates | Oracle | Pair oracle | Original hit | Offline hit | Rescued | Harmed | Offline source | Offline oracle rank | Pair oracle rank |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0 | 20 | no | no | no | no | no | no | beam | n/a | n/a |
| 1 | 20 | no | no | no | no | no | no | beam | n/a | n/a |
| 2 | 20 | no | no | no | no | no | no | beam | n/a | n/a |
| 3 | 25 | no | no | no | no | no | no | pair | n/a | n/a |
| 4 | 25 | no | no | no | no | no | no | pair | n/a | n/a |
| 5 | 25 | yes | yes | no | no | no | no | beam | 8 | 8 |
| 6 | 20 | no | no | no | no | no | no | pair | n/a | n/a |
| 7 | 20 | no | no | no | no | no | no | beam | n/a | n/a |
| 8 | 25 | yes | yes | no | no | no | no | sample_t=1.2 | 6 | 6 |
| 9 | 20 | no | no | no | no | no | no | pair | n/a | n/a |
| 10 | 20 | no | no | no | no | no | no | beam | n/a | n/a |
| 11 | 25 | yes | yes | no | no | no | no | pair | 3 | 3 |
| 12 | 20 | yes | yes | no | no | no | no | pair | 3 | 3 |
| 13 | 20 | yes | no | no | no | no | no | beam | 11 | n/a |
| 14 | 25 | no | no | no | no | no | no | pair | n/a | n/a |
| 15 | 20 | no | no | no | no | no | no | pair | n/a | n/a |

## Recommendation

Use this logging path for a future targeted smoke before considering any scorer implementation. Do not run a formal 32-sample eval from this safety check alone.
