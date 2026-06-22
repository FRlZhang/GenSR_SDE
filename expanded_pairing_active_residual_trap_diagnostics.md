# Expanded Pairing Active Residual Trap Diagnostics

Date: 2026-06-22

Scope: targeted residual-vector analysis for expanded-pairing selected misses. This did not run `sde_validation_probe.py`, regenerate candidate pools, train, run a 32/64-sample formal eval, or add a rerank heuristic.

## Executive Summary

- Expanded-pairing formal result remains selected exact/relaxed `3/32`, oracle exact/relaxed `15/32`, pair oracle exact/relaxed `8/32`.
- Residual vectors were recomputed for `12` selected-miss selected/oracle pairs.
- Wrong selected pair cases analyzed: `8`.
- Oracle-only-pair misses analyzed: `7`.
- Cases where active favors selected: `12/12`.
- Cases where weak favors oracle while active favors selected: `4/12`.
- Trap shape among active-selected cases: outlier `7`, broad `2`, mixed `3`.
- Active misleading dimensions lean: KM1/drift `10`, KM2/diffusion `1`, mixed `1`.

Interpretation: the active traps are more often outlier-dominated than broad. The misleading dimensions mostly sit on KM1/drift-like active features, with a few mixed or KM2/diffusion-like cases. Weak-kernel distance sometimes disagrees, but not often enough to justify a formal reweighting eval. Constants do not look like the primary blocker because the correct structures already use role-wise constant-grid fits and still lose.

Decision: **A. Active trap is dominated by a few residual dimensions; recommend one future offline robust/clipped active-residual ablation on existing expanded candidates only.** This is not enough evidence to add a rerank mode or launch a formal eval.

## Aggregate Diagnosis

| Diagnostic | Count |
| --- | ---: |
| Miss cases analyzed | 12 |
| Wrong selected pair cases | 8 |
| Oracle-only-pair misses | 7 |
| Active favors selected | 12 |
| Weak favors oracle while active favors selected | 4 |
| Model score favors selected | 5 |
| Oracle best constants differ from selected best constants | 6 |

## Case Summary

| Sample | Selected src | Oracle src | Pair-only oracle | Active gap | Weak gap | Active wins sel/oracle | Shape | Side | Weak disagrees | Model favors | Classification |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 5 | beam | pair | yes | 0.202635 | 0.006944 | 9/9 | outlier-dominated | KM1/drift | no | oracle | drift-side structural miss; outlier-dominated active trap; KM1/drift leaning |
| 8 | beam | pair | yes | 0.092465 | -0.003524 | 9/9 | mixed | KM1/drift | yes | oracle | drift-side structural miss; mixed active trap; KM1/drift leaning; weak disagrees |
| 11 | pair | pair | yes | 0.130216 | 0.012444 | 6/12 | outlier-dominated | KM1/drift | no | oracle | drift-side structural miss; outlier-dominated active trap; KM1/drift leaning |
| 12 | pair | pair | yes | 0.151355 | -0.011016 | 10/8 | outlier-dominated | KM1/drift | yes | oracle | drift-side structural miss; outlier-dominated active trap; KM1/drift leaning; weak disagrees |
| 13 | pair | sample_t=1.2 | no | 0.192891 | 0.297125 | 13/5 | broad | KM2/diffusion | no | selected | both-side structural miss; broad active trap; KM2/diffusion leaning; model-score conflict |
| 17 | pair | beam | no | 0.161225 | -0.099731 | 6/12 | mixed | KM1/drift | yes | selected | diffusion-side structural miss; mixed active trap; KM1/drift leaning; weak disagrees; model-score conflict |
| 19 | beam | beam | no | 0.220527 | 0.104047 | 8/10 | outlier-dominated | KM1/drift | no | oracle | both-side structural miss; outlier-dominated active trap; KM1/drift leaning |
| 21 | pair | beam | no | 0.044176 | 0.243536 | 1/17 | outlier-dominated | KM1/drift | no | selected | diffusion-side structural miss; outlier-dominated active trap; KM1/drift leaning; model-score conflict |
| 24 | pair | pair | yes | 0.474953 | 0.078794 | 5/13 | outlier-dominated | KM1/drift | no | oracle | both-side structural miss; outlier-dominated active trap; KM1/drift leaning |
| 26 | pair | beam | no | 0.064286 | 0.251065 | 14/4 | broad | mixed | no | selected | both-side structural miss; broad active trap; mixed leaning; model-score conflict |
| 30 | pair | pair | yes | 0.217149 | -0.030966 | 2/16 | outlier-dominated | KM1/drift | yes | selected | diffusion-side structural miss; outlier-dominated active trap; KM1/drift leaning; weak disagrees; model-score conflict |
| 31 | beam | pair | yes | 0.324522 | 0.036457 | 10/8 | mixed | KM1/drift | no | oracle | diffusion-side structural miss; mixed active trap; KM1/drift leaning |

Positive active or weak gap means the selected non-oracle has lower distance than the oracle in the formal log.

## Per-Case Residual Highlights

### Sample 5

Truth: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Selected: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Oracle: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Selected source `beam`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `1.054519`, active `0.651771`, weak `0.402749`, drift constant `1.000000`, diffusion constant `0.500000`.

Oracle score recompute: total `1.206525`, active `0.863015`, weak `0.343510`, drift constant `1.000000`, diffusion constant `0.500000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `0.556`; top3 share `0.747`.

Selected active residual vector: `[-0.0213887, -0.0710917, -0.257343, 0.0082381, 0.0485454, 0.0993112, 0.0658124, -0.0661657, -0.410589, -0.0103915, 0.0569842, 0.130226, -0.023648, -0.0674138, -0.298833, -0.010599, 0.192339, 0.107995]`

Oracle active residual vector: `[-0.0928975, -0.0510567, -0.449514, 0.0277009, -0.304062, 0.0835434, 0.0618507, -0.0451472, -0.564374, 0.0110627, 0.126656, 0.135762, 0.0608552, -0.0514561, -0.152862, -0.00896815, 0.176316, 0.10692]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.255516 | 0.048545 | 0.304062 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.192171 | 0.257343 | 0.449514 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.153785 | 0.410589 | 0.564374 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.071509 | 0.021389 | 0.092898 |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.069671 | 0.056984 | 0.126656 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.145970 | 0.298833 | 0.152862 |
| 7 | `t=0.5,x0=0.5,diffusion` | KM2/diffusion | 0.021019 | 0.066166 | 0.045147 |
| 1 | `t=0,x0=0.5,diffusion` | KM2/diffusion | 0.020035 | 0.071092 | 0.051057 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.016024 | 0.192339 | 0.176316 |
| 13 | `t=1,x0=0.5,diffusion` | KM2/diffusion | 0.015958 | 0.067414 | 0.051456 |

### Sample 8

Truth: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Selected: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Oracle: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Selected source `beam`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `0.445216`, active `0.173276`, weak `0.271939`, drift constant `2.000000`, diffusion constant `1.000000`.

Oracle score recompute: total `0.484355`, active `0.221548`, weak `0.262807`, drift constant `2.000000`, diffusion constant `0.500000`.

Active residual shape: `mixed`; dominant active side: `KM1/drift`; top2 share `0.482`; top3 share `0.654`.

Selected active residual vector: `[-0.00543335, 0.0352586, 0.0131117, 0.0319202, 0.00484558, 0.0188531, -0.00441384, 0.0356543, -0.0816229, 0.034145, -0.0240773, 0.0166806, 0.0456425, 0.035874, 0.0497078, 0.0294905, -0.101824, 0.0112516]`

Oracle active residual vector: `[-0.0305267, -0.0321778, 0.0120891, -0.00943343, -0.0494288, 0.0281697, -0.0527278, -0.0331649, -0.0845421, -0.00977228, -0.100712, 0.025179, 0.00855366, -0.031526, 0.00521995, -0.0143849, -0.142215, 0.0147748]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.076635 | 0.024077 | 0.100712 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.048314 | 0.004414 | 0.052728 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.044583 | 0.004846 | 0.049429 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.040392 | 0.101824 | 0.142215 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.025093 | 0.005433 | 0.030527 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.044488 | 0.049708 | 0.005220 |
| 12 | `t=1,x0=0.5,drift` | KM1/drift | 0.037089 | 0.045642 | 0.008554 |
| 9 | `t=0.5,x0=1,diffusion` | KM2/diffusion | 0.024373 | 0.034145 | 0.009772 |
| 3 | `t=0,x0=1,diffusion` | KM2/diffusion | 0.022487 | 0.031920 | 0.009433 |
| 15 | `t=1,x0=1,diffusion` | KM2/diffusion | 0.015106 | 0.029490 | 0.014385 |

### Sample 11

Truth: `<DRIFT> mul mul CONSTANT CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Selected: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Selected source `pair`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `0.731198`, active `0.327414`, weak `0.403784`, drift constant `0.250000`, diffusion constant `1.000000`.

Oracle score recompute: total `1.075591`, active `0.652224`, weak `0.423367`, drift constant `0.250000`, diffusion constant `1.000000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `0.849`; top3 share `0.972`.

Selected active residual vector: `[-0.0532548, 0.00870532, -0.0240842, -0.032593, -0.102508, -0.0193651, 0.0723812, -0.0105511, -0.0825743, -0.0586566, 0.1926, -0.0373733, -0.0188676, 0.00386779, -0.0247983, 0.024852, -0.172004, -0.0804545]`

Oracle active residual vector: `[-0.00729363, 0.00598929, 0.0375638, -0.0345494, -0.186822, -0.00354477, -0.00916453, 0.00602122, 0.152447, -0.0284179, 0.59016, 0.0257937, 0.0118649, 0.00454827, -0.0152512, 0.0126989, -0.11538, -0.0282408]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.397560 | 0.192600 | 0.590160 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.084314 | 0.102508 | 0.186822 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.069873 | 0.082574 | 0.152447 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.013480 | 0.024084 | 0.037564 |
| 3 | `t=0,x0=1,diffusion` | KM2/diffusion | 0.001956 | 0.032593 | 0.034549 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.063217 | 0.072381 | 0.009165 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.056624 | 0.172004 | 0.115380 |
| 17 | `t=1,x0=1.5,diffusion` | KM2/diffusion | 0.052214 | 0.080454 | 0.028241 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.045961 | 0.053255 | 0.007294 |
| 9 | `t=0.5,x0=1,diffusion` | KM2/diffusion | 0.030239 | 0.058657 | 0.028418 |

### Sample 12

Truth: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected source `pair`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `1.198925`, active `0.661572`, weak `0.537353`, drift constant `0.250000`, diffusion constant `1.000000`.

Oracle score recompute: total `1.095315`, active `0.590364`, weak `0.504951`, drift constant `0.250000`, diffusion constant `1.000000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `0.556`; top3 share `0.654`.

Selected active residual vector: `[-0.0661321, -0.00858366, -0.200993, -0.0461216, -0.290515, -0.0280244, -0.163793, -0.0144319, -0.412093, -0.0166246, -0.196316, -0.046977, -0.0842647, -0.00513137, -0.197503, -0.0346217, -0.137862, -0.0360679]`

Oracle active residual vector: `[-0.0982233, -0.0181617, -0.123942, -0.0293506, -0.12372, -0.0334168, -0.124357, -0.0129192, -0.394974, -0.0288077, -0.175047, -0.064977, -0.0919391, -0.00221823, -0.207418, -0.0475843, -0.207574, -0.0417716]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.069712 | 0.137862 | 0.207574 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.032091 | 0.066132 | 0.098223 |
| 11 | `t=0.5,x0=1.5,diffusion` | KM2/diffusion | 0.018000 | 0.046977 | 0.064977 |
| 15 | `t=1,x0=1,diffusion` | KM2/diffusion | 0.012963 | 0.034622 | 0.047584 |
| 9 | `t=0.5,x0=1,diffusion` | KM2/diffusion | 0.012183 | 0.016625 | 0.028808 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.166796 | 0.290515 | 0.123720 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.077052 | 0.200993 | 0.123942 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.039436 | 0.163793 | 0.124357 |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.021269 | 0.196316 | 0.175047 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.017120 | 0.412093 | 0.394974 |

### Sample 13

Truth: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected source `pair`; oracle source `sample_t=1.2`; oracle-only-pair `False`.

Selected score recompute: total `1.888200`, active `0.695842`, weak `1.192358`, drift constant `0.250000`, diffusion constant `0.250000`.

Oracle score recompute: total `1.749075`, active `0.705235`, weak `1.043840`, drift constant `0.250000`, diffusion constant `1.000000`.

Active residual shape: `broad`; dominant active side: `KM2/diffusion`; top2 share `0.219`; top3 share `0.319`.

Selected active residual vector: `[-0.225763, 0.0445544, -0.191106, 0.00199545, 0.100698, -0.0933771, 0.0110684, 0.0496884, -0.122048, 0.00631794, -0.0364315, -0.0738832, -0.107033, 0.0297965, -0.520605, -0.0238117, -0.252851, -0.0605785]`

Oracle active residual vector: `[-0.200617, -0.137589, -0.272086, -0.121157, -0.210231, -0.11911, -0.149097, -0.132716, -0.000767183, -0.107475, 0.00280418, -0.140104, -0.0474732, -0.145613, -0.226568, -0.146216, -0.331607, -0.117987]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.138029 | 0.011068 | 0.149097 |
| 15 | `t=1,x0=1,diffusion` | KM2/diffusion | 0.122404 | 0.023812 | 0.146216 |
| 3 | `t=0,x0=1,diffusion` | KM2/diffusion | 0.119161 | 0.001995 | 0.121157 |
| 13 | `t=1,x0=0.5,diffusion` | KM2/diffusion | 0.115817 | 0.029797 | 0.145613 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.109533 | 0.100698 | 0.210231 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.294037 | 0.520605 | 0.226568 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.121281 | 0.122048 | 0.000767 |
| 12 | `t=1,x0=0.5,drift` | KM1/drift | 0.059560 | 0.107033 | 0.047473 |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.033627 | 0.036432 | 0.002804 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.025146 | 0.225763 | 0.200617 |

### Sample 17

Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected source `pair`; oracle source `beam`; oracle-only-pair `False`.

Selected score recompute: total `0.965375`, active `0.639965`, weak `0.325410`, drift constant `1.000000`, diffusion constant `0.500000`.

Oracle score recompute: total `0.963785`, active `0.564534`, weak `0.399251`, drift constant `0.250000`, diffusion constant `0.500000`.

Active residual shape: `mixed`; dominant active side: `KM1/drift`; top2 share `0.495`; top3 share `0.668`.

Selected active residual vector: `[0.034082, 0.0456099, 0.30834, 0.0636735, 0.140513, -0.0271176, 0.0297419, 0.0363858, -0.212215, 0.0301958, 0.198688, -0.00698365, -0.191308, 0.0408899, -0.361814, 0.0755922, -0.155232, -0.00589114]`

Oracle active residual vector: `[0.0858086, -0.000518766, 0.257608, 0.0322397, 0.20373, 0.0344555, 0.115955, -0.00914089, -0.189573, 0.00938313, -0.122179, 0.0591344, -0.00724591, -0.00720901, -0.349233, 0.0305549, -0.0896623, 0.0471131]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.086213 | 0.029742 | 0.115955 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.063217 | 0.140513 | 0.203730 |
| 11 | `t=0.5,x0=1.5,diffusion` | KM2/diffusion | 0.052151 | 0.006984 | 0.059134 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.051727 | 0.034082 | 0.085809 |
| 17 | `t=1,x0=1.5,diffusion` | KM2/diffusion | 0.041222 | 0.005891 | 0.047113 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 12 | `t=1,x0=0.5,drift` | KM1/drift | 0.184062 | 0.191308 | 0.007246 |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.076509 | 0.198688 | 0.122179 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.065569 | 0.155232 | 0.089662 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.050731 | 0.308340 | 0.257608 |
| 1 | `t=0,x0=0.5,diffusion` | KM2/diffusion | 0.045091 | 0.045610 | 0.000519 |

### Sample 19

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected source `beam`; oracle source `beam`; oracle-only-pair `False`.

Selected score recompute: total `1.613375`, active `0.722964`, weak `0.890411`, drift constant `0.250000`, diffusion constant `0.500000`.

Oracle score recompute: total `1.903220`, active `0.941829`, weak `0.961391`, drift constant `0.250000`, diffusion constant `0.250000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `0.610`; top3 share `0.761`.

Selected active residual vector: `[0.151104, -0.0134008, 0.183313, -0.0672496, 0.334094, -0.149386, -0.178915, -0.00744395, -0.208792, -0.0884625, -0.122445, -0.150021, -0.407049, -0.00514986, 0.0809771, -0.089105, 0.0619069, -0.150248]`

Oracle active residual vector: `[0.00103382, 0.0295649, 0.130452, -0.0548247, 0.320388, -0.101758, -0.320527, 0.0108013, -0.37437, -0.0466781, -0.561017, -0.107925, -0.192458, 0.0245837, 0.310586, -0.0886733, -0.142966, -0.138406]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.438572 | 0.122445 | 0.561017 |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.229609 | 0.080977 | 0.310586 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.165579 | 0.208792 | 0.374370 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.141612 | 0.178915 | 0.320527 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.081059 | 0.061907 | 0.142966 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 12 | `t=1,x0=0.5,drift` | KM1/drift | 0.214591 | 0.407049 | 0.192458 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.150070 | 0.151104 | 0.001034 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.052861 | 0.183313 | 0.130452 |
| 5 | `t=0,x0=1.5,diffusion` | KM2/diffusion | 0.047627 | 0.149386 | 0.101758 |
| 11 | `t=0.5,x0=1.5,diffusion` | KM2/diffusion | 0.042096 | 0.150021 | 0.107925 |

### Sample 21

Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0`

Oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected source `pair`; oracle source `beam`; oracle-only-pair `False`.

Selected score recompute: total `0.688689`, active `0.333773`, weak `0.354916`, drift constant `1.000000`, diffusion constant `0.250000`.

Oracle score recompute: total `0.495259`, active `0.225171`, weak `0.270089`, drift constant `1.000000`, diffusion constant `0.250000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `1.000`; top3 share `1.000`.

Selected active residual vector: `[-0.0440676, 0.00602951, 0.0291805, -0.0128734, 0.0432835, -0.0494625, 0.0420269, 0.00526541, 0.0643654, -0.00999011, -0.224949, -0.0386171, 0.0983966, 0.00283926, 0.165001, -0.0107897, -0.0865094, -0.0379851]`

Oracle active residual vector: `[0.0316386, 0.00280238, -0.0270442, -0.00408391, -0.0967046, -0.0123438, 0.00744586, 0.00394803, -0.00898649, -0.00113128, -0.133162, -0.0118633, 0.0949267, 8.18191e-05, 0.0813596, -0.0056486, -0.0754905, -0.00731882]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.053421 | 0.043284 | 0.096705 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.091787 | 0.224949 | 0.133162 |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.083642 | 0.165001 | 0.081360 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.055379 | 0.064365 | 0.008986 |
| 5 | `t=0,x0=1.5,diffusion` | KM2/diffusion | 0.037119 | 0.049462 | 0.012344 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.034581 | 0.042027 | 0.007446 |

### Sample 24

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`

Selected source `pair`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `1.433299`, active `0.722476`, weak `0.710824`, drift constant `0.500000`, diffusion constant `1.000000`.

Oracle score recompute: total `1.724972`, active `0.805957`, weak `0.919014`, drift constant `0.250000`, diffusion constant `0.500000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `0.590`; top3 share `0.819`.

Selected active residual vector: `[-0.232056, -0.155585, -0.0362147, -0.1692, -0.0374607, -0.150015, -0.162916, -0.159042, -0.126494, -0.122937, -0.190203, -0.152328, -0.239804, -0.175888, -0.309249, -0.171335, 0.11461, -0.145488]`

Oracle active residual vector: `[-0.246551, -0.00713363, -0.256018, -0.0642949, -0.284079, -0.144073, -0.321386, -0.00662317, -0.444886, -0.0655195, -0.159408, -0.109431, -0.167446, -0.0127944, -0.155165, -0.0647918, -0.0644891, -0.118413]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.318393 | 0.126494 | 0.444886 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.246618 | 0.037461 | 0.284079 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.219803 | 0.036215 | 0.256018 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.158471 | 0.162916 | 0.321386 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.014495 | 0.232056 | 0.246551 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 13 | `t=1,x0=0.5,diffusion` | KM2/diffusion | 0.163094 | 0.175888 | 0.012794 |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.154084 | 0.309249 | 0.155165 |
| 7 | `t=0.5,x0=0.5,diffusion` | KM2/diffusion | 0.152419 | 0.159042 | 0.006623 |
| 1 | `t=0,x0=0.5,diffusion` | KM2/diffusion | 0.148451 | 0.155585 | 0.007134 |
| 15 | `t=1,x0=1,diffusion` | KM2/diffusion | 0.106544 | 0.171335 | 0.064792 |

### Sample 26

Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`

Selected source `pair`; oracle source `beam`; oracle-only-pair `False`.

Selected score recompute: total `0.931769`, active `0.511687`, weak `0.420082`, drift constant `0.250000`, diffusion constant `1.000000`.

Oracle score recompute: total `1.445924`, active `0.674083`, weak `0.771841`, drift constant `0.250000`, diffusion constant `0.500000`.

Active residual shape: `broad`; dominant active side: `mixed`; top2 share `0.333`; top3 share `0.473`.

Selected active residual vector: `[-0.0337575, -0.0585255, -0.143244, 0.0231569, -0.150407, 0.0668637, -0.182055, -0.0710974, -0.259787, 0.00204071, 0.0126713, 0.0363794, -0.286244, -0.0432393, -0.0633909, -0.0129376, 0.115847, 0.0228564]`

Oracle active residual vector: `[-0.234336, 0.0907744, -0.163273, 0.016946, -0.229497, -0.0927127, -0.213089, 0.10368, -0.254888, 0.00659459, 0.142742, -0.199597, -0.194508, 0.113614, -0.189923, 0.0350396, 0.052687, -0.175575]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.200579 | 0.033758 | 0.234336 |
| 11 | `t=0.5,x0=1.5,diffusion` | KM2/diffusion | 0.163218 | 0.036379 | 0.199597 |
| 17 | `t=1,x0=1.5,diffusion` | KM2/diffusion | 0.152719 | 0.022856 | 0.175575 |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.130070 | 0.012671 | 0.142742 |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.126532 | 0.063391 | 0.189923 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 12 | `t=1,x0=0.5,drift` | KM1/drift | 0.091736 | 0.286244 | 0.194508 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.063160 | 0.115847 | 0.052687 |
| 3 | `t=0,x0=1,diffusion` | KM2/diffusion | 0.006211 | 0.023157 | 0.016946 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.004900 | 0.259787 | 0.254888 |

### Sample 30

Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`

Selected source `pair`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `0.847039`, active `0.412840`, weak `0.434199`, drift constant `1.000000`, diffusion constant `0.250000`.

Oracle score recompute: total `0.486025`, active `0.249975`, weak `0.236050`, drift constant `1.000000`, diffusion constant `0.500000`.

Active residual shape: `outlier-dominated`; dominant active side: `KM1/drift`; top2 share `1.000`; top3 share `1.000`.

Selected active residual vector: `[-0.0127649, 0.0158197, 0.102134, -0.02062, 0.124465, -0.0800113, 0.00470799, 0.0145106, 0.132361, -0.01905, -0.0683471, -0.0793767, 0.0559535, 0.0163859, -0.095753, -0.0186995, 0.297406, -0.0825746]`

Oracle active residual vector: `[-0.0191556, -0.00190029, -0.08039, -0.00322844, 0.0716187, -0.00235309, 0.000324625, -0.00629071, 0.117407, -0.00427771, -0.171029, -0.00409819, 0.0512828, -0.00528843, -0.0536482, -0.00406024, 0.0422961, -0.00757785]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.102682 | 0.068347 | 0.171029 |
| 0 | `t=0,x0=0.5,drift` | KM1/drift | 0.006391 | 0.012765 | 0.019156 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.255110 | 0.297406 | 0.042296 |
| 5 | `t=0,x0=1.5,diffusion` | KM2/diffusion | 0.077658 | 0.080011 | 0.002353 |
| 11 | `t=0.5,x0=1.5,diffusion` | KM2/diffusion | 0.075279 | 0.079377 | 0.004098 |
| 17 | `t=1,x0=1.5,diffusion` | KM2/diffusion | 0.074997 | 0.082575 | 0.007578 |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.052846 | 0.124465 | 0.071619 |

### Sample 31

Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0`

Oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`

Selected source `beam`; oracle source `pair`; oracle-only-pair `True`.

Selected score recompute: total `0.580129`, active `0.335521`, weak `0.244608`, drift constant `1.000000`, diffusion constant `0.500000`.

Oracle score recompute: total `0.603624`, active `0.352763`, weak `0.250862`, drift constant `1.000000`, diffusion constant `0.500000`.

Active residual shape: `mixed`; dominant active side: `KM1/drift`; top2 share `0.464`; top3 share `0.666`.

Selected active residual vector: `[0.00513677, 0.00379222, -0.11218, 0.0268997, 0.0165968, 0.113978, 0.00429095, 0.0057759, -0.047588, 0.0405562, 0.166408, 0.100237, -0.00220147, 0.00627092, -0.0821666, 0.0454542, -0.175855, 0.0668727]`

Oracle active residual vector: `[0.024735, 0.00509308, -0.0323036, 0.0259954, 0.106415, 0.084117, 0.0749704, 0.00633027, -0.102239, 0.0241043, 0.0453563, 0.0966624, 0.00571141, 0.00341224, 0.0535741, 0.0475499, -0.245539, 0.100621]`

Top active features where selected beats oracle:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 4 | `t=0,x0=1.5,drift` | KM1/drift | 0.089818 | 0.016597 | 0.106415 |
| 6 | `t=0.5,x0=0.5,drift` | KM1/drift | 0.070679 | 0.004291 | 0.074970 |
| 16 | `t=1,x0=1.5,drift` | KM1/drift | 0.069684 | 0.175855 | 0.245539 |
| 8 | `t=0.5,x0=1,drift` | KM1/drift | 0.054651 | 0.047588 | 0.102239 |
| 17 | `t=1,x0=1.5,diffusion` | KM2/diffusion | 0.033749 | 0.066873 | 0.100621 |

Top active features where oracle beats selected:

| idx | feature | side | advantage | selected abs resid | oracle abs resid |
| ---: | --- | --- | ---: | ---: | ---: |
| 10 | `t=0.5,x0=1.5,drift` | KM1/drift | 0.121052 | 0.166408 | 0.045356 |
| 2 | `t=0,x0=1,drift` | KM1/drift | 0.079877 | 0.112180 | 0.032304 |
| 5 | `t=0,x0=1.5,diffusion` | KM2/diffusion | 0.029861 | 0.113978 | 0.084117 |
| 14 | `t=1,x0=1,drift` | KM1/drift | 0.028592 | 0.082167 | 0.053574 |
| 9 | `t=0.5,x0=1,diffusion` | KM2/diffusion | 0.016452 | 0.040556 | 0.024104 |

## Recommendation

- Do not make expanded pairing default and do not run a formal robust/clipped scorer eval now.
- If continuing scorer analysis, do one offline robust/clipped active-residual ablation on the existing expanded-pairing miss candidates first.
- The stronger next engineering direction remains candidate generation and active-fingerprint diagnostics: improve drift span diversity, and investigate why active KM statistics prefer plausible but wrong recombinations.

## Provenance

- Miss JSON: `expanded_pairing_oracle_miss_ranking_diagnostics.json`.
- Pruning JSON: `expanded_pairing_wrong_pair_pruning_diagnostics.json`.
- Coverage JSON: `candidate_coverage_pair_expanded.json`.
- Residual recompute settings: `n_paths=800`, `active_paths=800`, `n_steps=60`.
