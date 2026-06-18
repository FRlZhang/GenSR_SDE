# Role-wise Residual Vector Diagnostics

Date: 2026-06-18

Scope: targeted diagnostic for samples `13`, `19`, `24`, and `26`. This is not a formal 32-sample eval, not a 64-sample eval, and not a new rerank heuristic.

## Executive Summary

- Current role-wise baseline remains selected exact/relaxed `5/32` with oracle ceiling `9/32`.
- The diagnostic recomputed target fingerprints for the four known miss samples from the eval seed and scored only selected/oracle role swaps.
- Distances are diagnostic recomputations with deterministic per-combo seeds, so they should be interpreted as residual structure rather than a replacement for the existing formal eval metrics.

## Aggregate Findings

- Samples analyzed: `13, 19, 24, 26`.
- Drift-side ambiguity cases: `1`.
- Diffusion-side ambiguity cases: `1`.
- Both-side ambiguity cases: `2`.
- Feature-scaling artifact flags: `4`.
- Model-score conflicts: `1`.

## Sample 13

Truth: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected source: `beam`; oracle source: `sample_t=1.2`.

Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`

Selected diffusion: `mul CONSTANT sqrt abs x_0`

Oracle drift: `mul mul CONSTANT CONSTANT sin x_0`

Oracle diffusion: `mul CONSTANT CONSTANT`

### Role-swap Scores

| Combination | Success | Total | Active | Weak | Drift constant | Diffusion constant | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `selected+selected` | True | 1.485172 | 0.688209 | 0.796964 | 0.500000 | 0.500000 | `none` |
| `oracle+oracle` | True | 1.749075 | 0.705235 | 1.043840 | 0.250000 | 1.000000 | `none` |
| `selected+oracle` | True | 1.392000 | 0.517972 | 0.874028 | 0.250000 | 0.500000 | `none` |
| `oracle+selected` | True | 1.832572 | 1.012815 | 0.819757 | 0.250000 | 1.000000 | `none` |

### Residual Vectors

Selected active residual vector: `[-0.250299, 0.0428275, -0.166918, -0.00812588, 0.24132, -0.0956156, -0.0189369, 0.0484575, -0.0979181, -0.00382041, 0.105358, -0.0764541, -0.137128, 0.0282192, -0.497481, -0.0339943, -0.111071, -0.0633017]`

Oracle active residual vector: `[-0.200617, -0.137589, -0.272086, -0.121157, -0.210231, -0.11911, -0.149097, -0.132716, -0.000767183, -0.107475, 0.00280418, -0.140104, -0.0474732, -0.145613, -0.226568, -0.146216, -0.331607, -0.117987]`

Selected weak residual vector: `[-0.0396852, -0.0396814, -0.0396645, -0.0396014, -0.0394058, -0.0388934, -0.0377464, -0.0355388, -0.0319053, -0.0269562, -0.0219382, -0.0197933, -0.0248058, -0.0405768, -0.0668174, -0.0972685, -0.121372, -0.12982, -0.120344, -0.0989473, -0.0753873, -0.0570482, -0.0461608, -0.0411742, ...]`

Oracle weak residual vector: `[0.0261659, 0.0258036, 0.0252898, 0.0246342, 0.0238985, 0.0232194, 0.0228359, 0.0231081, 0.0245116, 0.0276086, 0.0329723, 0.0410026, 0.0516203, 0.0638936, 0.0756336, 0.0832532, 0.0827019, 0.0718473, 0.0527296, 0.0311632, 0.0134617, 0.00312719, 7.25118e-05, 0.00207069, ...]`

### Active Feature Comparison

Selected wins: `12`; oracle wins: `6`; dominance: `few_dominant_features`.

Top active features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 16 | `t=1,x0=1.5,drift` | 0.220535 | 0.111071 | 0.331607 |
| 6 | `t=0.5,x0=0.5,drift` | 0.130160 | 0.018937 | 0.149097 |
| 13 | `t=1,x0=0.5,diffusion` | 0.117394 | 0.028219 | 0.145613 |
| 3 | `t=0,x0=1,diffusion` | 0.113031 | 0.008126 | 0.121157 |
| 15 | `t=1,x0=1,diffusion` | 0.112221 | 0.033994 | 0.146216 |
| 2 | `t=0,x0=1,drift` | 0.105168 | 0.166918 | 0.272086 |
| 9 | `t=0.5,x0=1,diffusion` | 0.103655 | 0.003820 | 0.107475 |
| 1 | `t=0,x0=0.5,diffusion` | 0.094762 | 0.042828 | 0.137589 |

Top active features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 14 | `t=1,x0=1,drift` | 0.270913 | 0.497481 | 0.226568 |
| 10 | `t=0.5,x0=1.5,drift` | 0.102554 | 0.105358 | 0.002804 |
| 8 | `t=0.5,x0=1,drift` | 0.097151 | 0.097918 | 0.000767 |
| 12 | `t=1,x0=0.5,drift` | 0.089655 | 0.137128 | 0.047473 |
| 0 | `t=0,x0=0.5,drift` | 0.049682 | 0.250299 | 0.200617 |
| 4 | `t=0,x0=1.5,drift` | 0.031089 | 0.241320 | 0.210231 |

### Weak Feature Comparison

Selected wins: `53`; oracle wins: `43`; dominance: `broad`.

Top weak features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 30 | `local_drift@-0.957` | 0.156636 | 0.077842 | 0.234478 |
| 29 | `local_drift@-1.13` | 0.153935 | 0.122225 | 0.276160 |
| 31 | `local_drift@-0.783` | 0.148476 | 0.038138 | 0.186615 |
| 28 | `local_drift@-1.3` | 0.141744 | 0.169306 | 0.311051 |
| 32 | `local_drift@-0.609` | 0.130487 | 0.004115 | 0.134603 |
| 27 | `local_drift@-1.48` | 0.120734 | 0.216443 | 0.337177 |
| 26 | `local_drift@-1.65` | 0.088852 | 0.261053 | 0.349905 |
| 47 | `local_drift@2` | 0.077365 | 0.008505 | 0.085871 |

Top weak features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 19 | `occupancy@1.3` | 0.067784 | 0.098947 | 0.031163 |
| 18 | `occupancy@1.13` | 0.067614 | 0.120344 | 0.052730 |
| 35 | `local_drift@-0.087` | 0.061977 | 0.063657 | 0.001680 |
| 20 | `occupancy@1.48` | 0.061926 | 0.075387 | 0.013462 |
| 17 | `occupancy@0.957` | 0.057972 | 0.129820 | 0.071847 |
| 21 | `occupancy@1.65` | 0.053921 | 0.057048 | 0.003127 |
| 36 | `local_drift@0.087` | 0.046483 | 0.076486 | 0.030003 |
| 22 | `occupancy@1.83` | 0.046088 | 0.046161 | 0.000073 |

### Classification

both-side ambiguity; feature-scaling artifact; model-score conflict; selected wins both aggregate segments

## Sample 19

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected source: `beam`; oracle source: `beam`.

Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`

Selected diffusion: `mul CONSTANT sqrt abs x_0`

Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`

Oracle diffusion: `add CONSTANT mul CONSTANT abs x_0`

### Role-swap Scores

| Combination | Success | Total | Active | Weak | Drift constant | Diffusion constant | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `selected+selected` | True | 1.613375 | 0.722964 | 0.890411 | 0.250000 | 0.500000 | `none` |
| `oracle+oracle` | True | 1.903220 | 0.941829 | 0.961391 | 0.250000 | 0.250000 | `none` |
| `selected+oracle` | True | 1.497770 | 0.740972 | 0.756798 | 0.250000 | 0.250000 | `none` |
| `oracle+selected` | True | 2.112431 | 1.268458 | 0.843973 | 0.250000 | 0.500000 | `none` |

### Residual Vectors

Selected active residual vector: `[0.151104, -0.0134008, 0.183313, -0.0672496, 0.334094, -0.149386, -0.178915, -0.00744395, -0.208792, -0.0884625, -0.122445, -0.150021, -0.407049, -0.00514986, 0.0809771, -0.089105, 0.0619069, -0.150248]`

Oracle active residual vector: `[0.00103382, 0.0295649, 0.130452, -0.0548247, 0.320388, -0.101758, -0.320527, 0.0108013, -0.37437, -0.0466781, -0.561017, -0.107925, -0.192458, 0.0245837, 0.310586, -0.0886733, -0.142966, -0.138406]`

Selected weak residual vector: `[-0.0194243, -0.0194216, -0.0194152, -0.0194011, -0.0193698, -0.0192971, -0.0191238, -0.0187278, -0.0179291, -0.0166179, -0.015061, -0.0142286, -0.0157358, -0.0210686, -0.0303094, -0.0412654, -0.0500414, -0.0531453, -0.0496973, -0.0418748, -0.0332279, -0.0264391, -0.0223175, -0.0203082, ...]`

Oracle weak residual vector: `[-0.0050326, -0.00502993, -0.00502353, -0.00500941, -0.00497829, -0.0049062, -0.00473314, -0.00432748, -0.00346359, -0.00189662, 0.000368414, 0.00264866, 0.00328521, -6.15807e-05, -0.0092139, -0.0237393, -0.0399954, -0.0523924, -0.0565849, -0.0521181, -0.0422366, -0.031288, -0.0222101, -0.0158455, ...]`

### Active Feature Comparison

Selected wins: `8`; oracle wins: `10`; dominance: `few_dominant_features`.

Top active features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | 0.438572 | 0.122445 | 0.561017 |
| 14 | `t=1,x0=1,drift` | 0.229609 | 0.080977 | 0.310586 |
| 8 | `t=0.5,x0=1,drift` | 0.165579 | 0.208792 | 0.374370 |
| 6 | `t=0.5,x0=0.5,drift` | 0.141612 | 0.178915 | 0.320527 |
| 16 | `t=1,x0=1.5,drift` | 0.081059 | 0.061907 | 0.142966 |
| 13 | `t=1,x0=0.5,diffusion` | 0.019434 | 0.005150 | 0.024584 |
| 1 | `t=0,x0=0.5,diffusion` | 0.016164 | 0.013401 | 0.029565 |
| 7 | `t=0.5,x0=0.5,diffusion` | 0.003357 | 0.007444 | 0.010801 |

Top active features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 12 | `t=1,x0=0.5,drift` | 0.214591 | 0.407049 | 0.192458 |
| 0 | `t=0,x0=0.5,drift` | 0.150070 | 0.151104 | 0.001034 |
| 2 | `t=0,x0=1,drift` | 0.052861 | 0.183313 | 0.130452 |
| 5 | `t=0,x0=1.5,diffusion` | 0.047627 | 0.149386 | 0.101758 |
| 11 | `t=0.5,x0=1.5,diffusion` | 0.042096 | 0.150021 | 0.107925 |
| 9 | `t=0.5,x0=1,diffusion` | 0.041784 | 0.088463 | 0.046678 |
| 4 | `t=0,x0=1.5,drift` | 0.013706 | 0.334094 | 0.320388 |
| 3 | `t=0,x0=1,diffusion` | 0.012425 | 0.067250 | 0.054825 |

### Weak Feature Comparison

Selected wins: `47`; oracle wins: `49`; dominance: `broad`.

Top weak features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 45 | `local_drift@1.65` | 0.045984 | 0.000026 | 0.046010 |
| 24 | `local_drift@-2` | 0.042474 | 0.378720 | 0.421195 |
| 46 | `local_drift@1.83` | 0.038166 | 0.011626 | 0.049793 |
| 44 | `local_drift@1.48` | 0.036204 | 0.010031 | 0.046234 |
| 25 | `local_drift@-1.83` | 0.035103 | 0.398476 | 0.433579 |
| 47 | `local_drift@2` | 0.034704 | 0.024742 | 0.059446 |
| 43 | `local_drift@1.3` | 0.029611 | 0.018549 | 0.048160 |
| 26 | `local_drift@-1.65` | 0.028495 | 0.395589 | 0.424084 |

Top weak features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 87 | `generator_action@0.609` | 0.033363 | 0.035232 | 0.001869 |
| 88 | `generator_action@0.783` | 0.031949 | 0.034459 | 0.002510 |
| 89 | `generator_action@0.957` | 0.028206 | 0.029915 | 0.001709 |
| 86 | `generator_action@0.435` | 0.025325 | 0.029650 | 0.004325 |
| 90 | `generator_action@1.13` | 0.021139 | 0.025642 | 0.004504 |
| 14 | `occupancy@0.435` | 0.021095 | 0.030309 | 0.009214 |
| 13 | `occupancy@0.261` | 0.021007 | 0.021069 | 0.000062 |
| 15 | `occupancy@0.609` | 0.017526 | 0.041265 | 0.023739 |

### Classification

both-side ambiguity; feature-scaling artifact; selected wins both aggregate segments

## Sample 24

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected source: `beam`; oracle source: `pair`.

Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`

Selected diffusion: `mul CONSTANT sqrt abs x_0`

Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`

Oracle diffusion: `mul CONSTANT sqrt abs x_0`

### Role-swap Scores

| Combination | Success | Total | Active | Weak | Drift constant | Diffusion constant | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `selected+selected` | True | 1.478760 | 0.645621 | 0.833140 | 0.250000 | 0.500000 | `none` |
| `oracle+oracle` | True | 1.724972 | 0.805957 | 0.919014 | 0.250000 | 0.500000 | `none` |
| `selected+oracle` | True | 1.478760 | 0.645621 | 0.833140 | 0.250000 | 0.500000 | `none` |
| `oracle+selected` | True | 1.724972 | 0.805957 | 0.919014 | 0.250000 | 0.500000 | `none` |

### Residual Vectors

Selected active residual vector: `[-0.257621, -0.00849839, -0.0569318, -0.0676338, -0.039695, -0.112063, -0.216532, -0.0090414, -0.14117, -0.0447934, -0.213103, -0.114875, -0.258511, -0.0132299, -0.313196, -0.0677324, 0.13828, -0.110089]`

Oracle active residual vector: `[-0.246551, -0.00713363, -0.256018, -0.0642949, -0.284079, -0.144073, -0.321386, -0.00662317, -0.444886, -0.0655195, -0.159408, -0.109431, -0.167446, -0.0127944, -0.155165, -0.0647918, -0.0644891, -0.118413]`

Selected weak residual vector: `[0.0219866, 0.0219999, 0.0220613, 0.0222958, 0.0230381, 0.0249767, 0.0291393, 0.0364313, 0.0466695, 0.05762, 0.0648883, 0.0630123, 0.0472913, 0.015755, -0.0289203, -0.0781403, -0.118589, -0.137427, -0.129441, -0.100451, -0.0633281, -0.0300006, -0.0062366, 0.00804747, ...]`

Oracle weak residual vector: `[0.0279317, 0.027945, 0.0280064, 0.0282409, 0.0289835, 0.0309241, 0.0350983, 0.0424433, 0.0528711, 0.0643587, 0.07282, 0.0729787, 0.0597198, 0.0295889, -0.0171145, -0.0736894, -0.126239, -0.15823, -0.159678, -0.133657, -0.0935503, -0.0537723, -0.0226248, -0.00164751, ...]`

### Active Feature Comparison

Selected wins: `7`; oracle wins: `11`; dominance: `few_dominant_features`.

Top active features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 8 | `t=0.5,x0=1,drift` | 0.303716 | 0.141170 | 0.444886 |
| 4 | `t=0,x0=1.5,drift` | 0.244384 | 0.039695 | 0.284079 |
| 2 | `t=0,x0=1,drift` | 0.199086 | 0.056932 | 0.256018 |
| 6 | `t=0.5,x0=0.5,drift` | 0.104854 | 0.216532 | 0.321386 |
| 5 | `t=0,x0=1.5,diffusion` | 0.032011 | 0.112063 | 0.144073 |
| 9 | `t=0.5,x0=1,diffusion` | 0.020726 | 0.044793 | 0.065519 |
| 17 | `t=1,x0=1.5,diffusion` | 0.008324 | 0.110089 | 0.118413 |

Top active features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 14 | `t=1,x0=1,drift` | 0.158032 | 0.313196 | 0.155165 |
| 12 | `t=1,x0=0.5,drift` | 0.091064 | 0.258511 | 0.167446 |
| 16 | `t=1,x0=1.5,drift` | 0.073791 | 0.138280 | 0.064489 |
| 10 | `t=0.5,x0=1.5,drift` | 0.053695 | 0.213103 | 0.159408 |
| 0 | `t=0,x0=0.5,drift` | 0.011070 | 0.257621 | 0.246551 |
| 11 | `t=0.5,x0=1.5,diffusion` | 0.005444 | 0.114875 | 0.109431 |
| 3 | `t=0,x0=1,diffusion` | 0.003339 | 0.067634 | 0.064295 |
| 15 | `t=1,x0=1,diffusion` | 0.002941 | 0.067732 | 0.064792 |

### Weak Feature Comparison

Selected wins: `77`; oracle wins: `19`; dominance: `few_dominant_features`.

Top weak features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 47 | `local_drift@2` | 0.143206 | 0.037776 | 0.180982 |
| 46 | `local_drift@1.83` | 0.115751 | 0.058657 | 0.174408 |
| 45 | `local_drift@1.65` | 0.092639 | 0.075324 | 0.167963 |
| 44 | `local_drift@1.48` | 0.074317 | 0.088441 | 0.162758 |
| 43 | `local_drift@1.3` | 0.061023 | 0.098645 | 0.159668 |
| 42 | `local_drift@1.13` | 0.051991 | 0.106665 | 0.158656 |
| 41 | `local_drift@0.957` | 0.045824 | 0.113214 | 0.159037 |
| 40 | `local_drift@0.783` | 0.041064 | 0.118882 | 0.159946 |

Top weak features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 24 | `local_drift@-2` | 0.050584 | 0.224753 | 0.174169 |
| 25 | `local_drift@-1.83` | 0.045705 | 0.210817 | 0.165112 |
| 26 | `local_drift@-1.65` | 0.040835 | 0.198006 | 0.157171 |
| 87 | `generator_action@0.609` | 0.039085 | 0.070767 | 0.031681 |
| 86 | `generator_action@0.435` | 0.036703 | 0.079375 | 0.042673 |
| 27 | `local_drift@-1.48` | 0.035911 | 0.186872 | 0.150960 |
| 88 | `generator_action@0.783` | 0.035031 | 0.036206 | 0.001174 |
| 85 | `generator_action@0.261` | 0.031168 | 0.053928 | 0.022760 |

### Classification

drift-side ambiguity; feature-scaling artifact; selected wins both aggregate segments

## Sample 26

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT CONSTANT`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected source: `beam`; oracle source: `beam`.

Selected drift: `mul mul CONSTANT CONSTANT CONSTANT`

Selected diffusion: `mul CONSTANT CONSTANT`

Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`

Oracle diffusion: `add CONSTANT mul CONSTANT abs x_0`

### Role-swap Scores

| Combination | Success | Total | Active | Weak | Drift constant | Diffusion constant | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `selected+selected` | True | 1.236200 | 0.545276 | 0.690924 | 0.250000 | 1.000000 | `none` |
| `oracle+oracle` | True | 1.445924 | 0.674083 | 0.771841 | 0.250000 | 0.500000 | `none` |
| `selected+oracle` | True | 1.445924 | 0.674083 | 0.771841 | 0.250000 | 0.500000 | `none` |
| `oracle+selected` | True | 1.236200 | 0.545276 | 0.690924 | 0.250000 | 1.000000 | `none` |

### Residual Vectors

Selected active residual vector: `[0.0289152, -0.0600511, -0.159452, 0.0244306, -0.248893, 0.0684449, -0.15096, -0.0741777, -0.287048, 0.00379068, -0.10907, 0.0378225, -0.258328, -0.0426046, -0.0777019, -0.0132555, 0.0315218, 0.0211087]`

Oracle active residual vector: `[-0.234336, 0.0907744, -0.163273, 0.016946, -0.229497, -0.0927127, -0.213089, 0.10368, -0.254888, 0.00659459, 0.142742, -0.199597, -0.194508, 0.113614, -0.189923, 0.0350396, 0.052687, -0.175575]`

Selected weak residual vector: `[0.0858186, 0.0864747, 0.087121, 0.0876956, 0.0881177, 0.0882867, 0.0880872, 0.0873634, 0.0858505, 0.0831064, 0.0785131, 0.0713792, 0.0611389, 0.0476268, 0.03141, 0.0141399, -0.00134948, -0.0116951, -0.0144386, -0.00927353, 0.00196023, 0.0164022, 0.0313651, 0.044938, ...]`

Oracle weak residual vector: `[0.0751644, 0.0760749, 0.0771618, 0.0784216, 0.0798125, 0.0812152, 0.0823773, 0.0828353, 0.0818465, 0.0784432, 0.0717024, 0.0611294, 0.046886, 0.0298036, 0.0114602, -0.00560336, -0.0180694, -0.0229735, -0.0192959, -0.00867057, 0.00552992, 0.019964, 0.0325542, 0.0425978, ...]`

### Active Feature Comparison

Selected wins: `14`; oracle wins: `4`; dominance: `few_dominant_features`.

Top active features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 0 | `t=0,x0=0.5,drift` | 0.205421 | 0.028915 | 0.234336 |
| 11 | `t=0.5,x0=1.5,diffusion` | 0.161775 | 0.037823 | 0.199597 |
| 17 | `t=1,x0=1.5,diffusion` | 0.154466 | 0.021109 | 0.175575 |
| 14 | `t=1,x0=1,drift` | 0.112221 | 0.077702 | 0.189923 |
| 13 | `t=1,x0=0.5,diffusion` | 0.071009 | 0.042605 | 0.113614 |
| 6 | `t=0.5,x0=0.5,drift` | 0.062130 | 0.150960 | 0.213089 |
| 10 | `t=0.5,x0=1.5,drift` | 0.033672 | 0.109070 | 0.142742 |
| 1 | `t=0,x0=0.5,diffusion` | 0.030723 | 0.060051 | 0.090774 |

Top active features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 12 | `t=1,x0=0.5,drift` | 0.063820 | 0.258328 | 0.194508 |
| 8 | `t=0.5,x0=1,drift` | 0.032160 | 0.287048 | 0.254888 |
| 4 | `t=0,x0=1.5,drift` | 0.019396 | 0.248893 | 0.229497 |
| 3 | `t=0,x0=1,diffusion` | 0.007485 | 0.024431 | 0.016946 |

### Weak Feature Comparison

Selected wins: `56`; oracle wins: `40`; dominance: `broad`.

Top weak features where selected beats oracle:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 71 | `local_diffusion@2` | 0.094746 | 0.055561 | 0.150306 |
| 53 | `local_diffusion@-1.13` | 0.093557 | 0.001558 | 0.095115 |
| 54 | `local_diffusion@-0.957` | 0.093432 | 0.007250 | 0.100683 |
| 30 | `local_drift@-0.957` | 0.092930 | 0.006693 | 0.099623 |
| 55 | `local_diffusion@-0.783` | 0.090448 | 0.014453 | 0.104901 |
| 29 | `local_drift@-1.13` | 0.090338 | 0.022786 | 0.113124 |
| 56 | `local_diffusion@-0.609` | 0.084883 | 0.021092 | 0.105974 |
| 52 | `local_diffusion@-1.3` | 0.076707 | 0.012955 | 0.089662 |

Top weak features where oracle beats selected:

| idx | feature | advantage | selected_abs_resid | oracle_abs_resid |
| ---: | --- | ---: | ---: | ---: |
| 24 | `local_drift@-2` | 0.145372 | 0.225646 | 0.080274 |
| 25 | `local_drift@-1.83` | 0.075353 | 0.168011 | 0.092659 |
| 85 | `generator_action@0.261` | 0.032963 | 0.054846 | 0.021882 |
| 86 | `generator_action@0.435` | 0.032755 | 0.070731 | 0.037976 |
| 87 | `generator_action@0.609` | 0.030819 | 0.105055 | 0.074236 |
| 84 | `generator_action@0.087` | 0.029994 | 0.054253 | 0.024259 |
| 88 | `generator_action@0.783` | 0.027769 | 0.146362 | 0.118593 |
| 83 | `generator_action@-0.087` | 0.023400 | 0.060632 | 0.037231 |

### Classification

diffusion-side ambiguity; feature-scaling artifact; selected wins both aggregate segments

## Notes

Source log: `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`

Fingerprint settings: `n_paths=800`, `active_paths=800`, `n_steps=60`.

No checkpoint, dataset pickle, target format, fingerprint schema, training path, or rerank default was changed.
