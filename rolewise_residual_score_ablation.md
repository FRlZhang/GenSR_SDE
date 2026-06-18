# Role-wise Residual Score Ablation

Date: 2026-06-18

Scope: offline score ablation over residual vectors for samples `13`, `19`, `24`, and `26`. This is not a formal eval and does not add a rerank mode.

## Executive Summary

- Offline variants that rank oracle above selected in at least one case: `1/4`.
- Mild robust/normalization variants rescue: `0/4`.
- Extreme clipping-only rescues: `1/4`.
- At least one offline score variant flips a miss; inspect whether the flip requires aggressive clipping before considering a formal eval.

## Per-sample Results

### Sample 13

Classification: `both-side ambiguity; feature-scaling artifact; model-score conflict; selected wins both aggregate segments`

Dominant selected-advantage features: active: 16:t=1,x0=1.5,drift (0.2205); 6:t=0.5,x0=0.5,drift (0.1302); 13:t=1,x0=0.5,diffusion (0.1174); weak: 30:local_drift@-0.957 (0.1566); 29:local_drift@-1.13 (0.1539); 31:local_drift@-0.783 (0.1485)

| Variant | Selected | Oracle | Oracle wins | Note |
| --- | ---: | ---: | --- | --- |
| `current_l2_sum` | 1.485172 | 1.749075 | no | current active L2 + weak L2 |
| `active_only_l2` | 0.688209 | 0.705235 | no | active segment only |
| `weak_only_l2` | 0.796964 | 1.043840 | no | weak segment only |
| `l1_sum` | 7.679179 | 9.866102 | no | sum absolute residuals |
| `l2_sum` | 1.485172 | 1.749075 | no | same geometry as current |
| `clipped_l2_p95` | 1.290021 | 1.592566 | no | per-feature squared contribution capped at p95=0.2573 |
| `clipped_l2_p90` | 1.033817 | 1.269889 | no | per-feature squared contribution capped at p90=0.1471 |
| `huber_p90` | 0.444036 | 0.662305 | no | Huber loss with delta=p90=0.1471 |
| `per_segment_normalized_l2` | 0.243552 | 0.272762 | no | active and weak L2 normalized by feature count |
| `top1_removed_l2` | 1.197940 | 1.605857 | no | remove largest contribution in each segment |
| `top3_removed_l2` | 0.926858 | 1.369095 | no | remove largest 3 contributions in each segment |
| `top5_removed_l2` | 0.778157 | 1.152537 | no | remove largest 5 contributions in each segment |

### Sample 19

Classification: `both-side ambiguity; feature-scaling artifact; selected wins both aggregate segments`

Dominant selected-advantage features: active: 10:t=0.5,x0=1.5,drift (0.4386); 14:t=1,x0=1,drift (0.2296); 8:t=0.5,x0=1,drift (0.1656); weak: 45:local_drift@1.65 (0.0460); 24:local_drift@-2 (0.0425); 46:local_drift@1.83 (0.0382)

| Variant | Selected | Oracle | Oracle wins | Note |
| --- | ---: | ---: | --- | --- |
| `current_l2_sum` | 1.613375 | 1.903220 | no | current active L2 + weak L2 |
| `active_only_l2` | 0.722964 | 0.941829 | no | active segment only |
| `weak_only_l2` | 0.890411 | 0.961391 | no | weak segment only |
| `l1_sum` | 6.719814 | 7.210325 | no | sum absolute residuals |
| `l2_sum` | 1.613375 | 1.903220 | no | same geometry as current |
| `clipped_l2_p95` | 1.480054 | 1.623212 | no | per-feature squared contribution capped at p95=0.3293 |
| `clipped_l2_p90` | 1.001981 | 0.997366 | yes | per-feature squared contribution capped at p90=0.1594 |
| `huber_p90` | 0.501542 | 0.621402 | no | Huber loss with delta=p90=0.1594 |
| `per_segment_normalized_l2` | 0.261281 | 0.320113 | no | active and weak L2 normalized by feature count |
| `top1_removed_l2` | 1.393756 | 1.614575 | no | remove largest contribution in each segment |
| `top3_removed_l2` | 1.027232 | 1.189597 | no | remove largest 3 contributions in each segment |
| `top5_removed_l2` | 0.723328 | 0.739754 | no | remove largest 5 contributions in each segment |

### Sample 24

Classification: `drift-side ambiguity; feature-scaling artifact; selected wins both aggregate segments`

Dominant selected-advantage features: active: 8:t=0.5,x0=1,drift (0.3037); 4:t=0,x0=1.5,drift (0.2444); 2:t=0,x0=1,drift (0.1991); weak: 47:local_drift@2 (0.1432); 46:local_drift@1.83 (0.1158); 45:local_drift@1.65 (0.0926)

| Variant | Selected | Oracle | Oracle wins | Note |
| --- | ---: | ---: | --- | --- |
| `current_l2_sum` | 1.478760 | 1.724972 | no | current active L2 + weak L2 |
| `active_only_l2` | 0.645621 | 0.805957 | no | active segment only |
| `weak_only_l2` | 0.833140 | 0.919014 | no | weak segment only |
| `l1_sum` | 8.268821 | 9.720367 | no | sum absolute residuals |
| `l2_sum` | 1.478760 | 1.724972 | no | same geometry as current |
| `clipped_l2_p95` | 1.380499 | 1.514151 | no | per-feature squared contribution capped at p95=0.2063 |
| `clipped_l2_p90` | 1.264901 | 1.431791 | no | per-feature squared contribution capped at p90=0.1628 |
| `huber_p90` | 0.528257 | 0.679171 | no | Huber loss with delta=p90=0.1628 |
| `per_segment_normalized_l2` | 0.237206 | 0.283762 | no | active and weak L2 normalized by feature count |
| `top1_removed_l2` | 1.366817 | 1.573062 | no | remove largest contribution in each segment |
| `top3_removed_l2` | 1.179044 | 1.384001 | no | remove largest 3 contributions in each segment |
| `top5_removed_l2` | 1.007879 | 1.209957 | no | remove largest 5 contributions in each segment |

### Sample 26

Classification: `diffusion-side ambiguity; feature-scaling artifact; selected wins both aggregate segments`

Dominant selected-advantage features: active: 0:t=0,x0=0.5,drift (0.2054); 11:t=0.5,x0=1.5,diffusion (0.1618); 17:t=1,x0=1.5,diffusion (0.1545); weak: 71:local_diffusion@2 (0.0947); 53:local_diffusion@-1.13 (0.0936); 54:local_diffusion@-0.957 (0.0934)

| Variant | Selected | Oracle | Oracle wins | Note |
| --- | ---: | ---: | --- | --- |
| `current_l2_sum` | 1.236200 | 1.445924 | no | current active L2 + weak L2 |
| `active_only_l2` | 0.545276 | 0.674083 | no | active segment only |
| `weak_only_l2` | 0.690924 | 0.771841 | no | weak segment only |
| `l1_sum` | 7.175223 | 9.403579 | no | sum absolute residuals |
| `l2_sum` | 1.236200 | 1.445924 | no | same geometry as current |
| `clipped_l2_p95` | 1.096453 | 1.349548 | no | per-feature squared contribution capped at p95=0.1739 |
| `clipped_l2_p90` | 0.986196 | 1.223097 | no | per-feature squared contribution capped at p90=0.1296 |
| `huber_p90` | 0.352030 | 0.494634 | no | Huber loss with delta=p90=0.1296 |
| `per_segment_normalized_l2` | 0.199040 | 0.237659 | no | active and weak L2 normalized by feature count |
| `top1_removed_l2` | 1.116644 | 1.381100 | no | remove largest contribution in each segment |
| `top3_removed_l2` | 0.901199 | 1.259008 | no | remove largest 3 contributions in each segment |
| `top5_removed_l2` | 0.762295 | 1.151719 | no | remove largest 5 contributions in each segment |

## Aggregate Conclusion

- Mild robust scoring rescues `0/4` remaining misses.
- Extreme clipping-only rescues `1/4` remaining misses.
- Any offline score rescue count is `1/4`.
- Sample 24 drift-side ambiguity rescued by any variant: `False`.
- Sample 26 diffusion-side ambiguity rescued by any variant: `False`.
- Formal eval is not justified from extreme clipping alone; it looks too brittle.

## Inputs

- JSON sidecar: `rolewise_residual_vector_diagnostics.json`
- Residual helper log: `/private/tmp/gensr_sde_rolewise_residual_vectors.log`
