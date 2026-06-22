# Pairing Expansion Coverage Report

Date: 2026-06-22

Scope: candidate-generation coverage only. No formal rerank eval, no 64-sample
eval, no retraining, no active/weak grid, and no scorer/default changes.

## Executive Summary

The expanded pairing configuration raised full-oracle candidate coverage from
`9/32` to `15/32`. All six baseline pairing-missing samples were rescued as
full oracle candidates from the `pair` source.

The expanded configuration was:

```text
pair_drift_topk=5
pair_diffusion_topk=6
pair_drift_diffusion_candidates=32
```

## Baseline vs Expanded

| Metric | Baseline | Expanded |
| --- | ---: | ---: |
| Full oracle present | 9/32 | 15/32 |
| Oracle absent | 23/32 | 17/32 |
| Drift-only present | 0 | 0 |
| Diffusion-only present | 17 | 17 |
| Drift+diffusion separate, not paired | 6 | 0 |
| Neither side present | 0 | 0 |

Source contribution changed as expected: paired candidates increased from 103
rows to 413 rows, and pair-source full oracle samples increased from 2 to 8.
The remaining 17 oracle-absent samples are still drift-missing with exact
diffusion present.

## Pairing-Missing Samples

| Sample | Baseline status | Expanded status | Full oracle source | Interpretation |
| ---: | --- | --- | --- | --- |
| 5 | separate drift/diffusion, pair rank 22, cap-blocked | full oracle present | pair | cap 32 was enough |
| 8 | separate drift/diffusion, pair rank 22, cap-blocked | full oracle present | pair | cap 32 was enough |
| 11 | separate drift/diffusion, pair rank 20, cap-blocked | full oracle present | pair | cap 32 was enough |
| 23 | exact drift rank 5 was outside drift top-k 4 | full oracle present | pair | drift top-k 5 was enough |
| 30 | separate drift/diffusion, pair rank 20, cap-blocked | full oracle present | pair | cap 32 was enough |
| 31 | separate drift/diffusion, pair rank 19, cap-blocked | full oracle present | pair | cap 32 was enough |

## Parameter Interpretation

`pair_drift_diffusion_candidates=32` rescued all five cap-blocked cases. The
needed pair ranks were 19, 20, or 22, so a smaller cap around 24 might be enough
for this exact 32-sample set, but 32 is the cleaner validation setting.

`pair_drift_topk=5` rescued sample 23 by admitting the exact drift span. Its
combined pair rank was 24 after the top-k expansion, so top-k alone would not
have been enough without also increasing the pair candidate cap.

The expanded setting appears saturated for the known pairing-missing cases:
there are no remaining separate-but-unpaired samples. Further oracle ceiling
growth must come from drift span diversity, because all 17 remaining
oracle-absent samples still contain exact diffusion but miss exact drift.

## Recommendation

Expanded pairing raises the offline oracle ceiling from `9/32` to `15/32`, so
one future formal 32-sample eval is justified with the current strongest scorer
and this pairing configuration. If selected recovery does not improve there,
return to score selection among the larger oracle pool; if it does improve,
keep the expanded pairing as the next candidate-generation baseline.

After that eval, the next ceiling-focused work should return to drift span
diversity for the 17 remaining oracle-absent samples.
