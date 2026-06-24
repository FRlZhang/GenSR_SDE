# P2 Fingerprint Ambiguity Diagnostics

## Why This Diagnostic Was Run

The previous gate-feasibility diagnostic closed the fixed residual calibration
implementation path because the best observable gate (`V4 + G3`) improved net
selected recovery but still harmed V0 hits. This helper inspects the V0/V4
changed cases offline to see whether rescues and harms have separable
fingerprint ambiguity patterns. No model decoding, candidate generation,
`sde_validation_probe.py`, formal eval, rerank-mode implementation, or
production default change was run.

## Aggregate Summary

| Item | Value |
| --- | ---: |
| Samples analyzed | 19 |
| V4 rescues vs V0 | 5 |
| V4 harms vs V0 | 2 |
| Changed nonoracle to nonoracle | 8 |
| V0 preserved hits | 4 |
| Removed dims explain flips | 2 |
| Removed dims explain harms | 0 |
| Active dominated ambiguity | 1 |
| Weak dominated ambiguity | 2 |
| Mixed ambiguity | 4 |

## V4 Rescue/Harm Samples

| Sample | Type | Truth drift | Truth diffusion | V0 selected | V4 selected | V0 oracle gap | V4 oracle gap | Removed dims explain flip | Ambiguity | Selected error |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | V4_rescue | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | -0.154757 | -0.0241631 | no | mixed | both_sides_wrong |
| 7 | V4_rescue | `mul CONSTANT x_0` | `mul CONSTANT CONSTANT` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT` | `<DRIFT> mul CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT` | 0.00615239 | 0 | yes | mixed | same_diffusion_wrong_drift |
| 22 | V4_rescue | `mul CONSTANT x_0` | `mul CONSTANT x_0` | `<DRIFT> mul CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0` | 0.00317607 | 0 | yes | mixed | same_drift_wrong_diffusion |
| 23 | V4_rescue | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0` | -0.0536874 | 0 | no | mixed | same_drift_wrong_diffusion |
| 31 | V4_rescue | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0` | 0.138803 | 0 | no | weak | same_drift_wrong_diffusion |
| 9 | V4_harm | `mul CONSTANT sin x_0` | `mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT x_0` | 0 | -0.267068 | no | active | same_drift_wrong_diffusion |
| 21 | V4_harm | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT` | 0 | -0.36288 | no | weak | same_drift_wrong_diffusion |

## Per-Sample Fingerprint Gap Explanation

| Sample | Pair gap V0 | Pair gap V4 | Active gap V0 | Active gap after removal | Weak gap | Removed contribution | Removed fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.154757 | 0.00140961 | -0.0373143 | 0.118852 | -0.117443 | 0.156167 | 4.18517 |
| 7 | 0.00615239 | -0.0467055 | -0.11009 | -0.162947 | 0.116242 | -0.0528578 | 0.480135 |
| 22 | 0.00317607 | -0.00166381 | -0.00413187 | -0.00897174 | 0.00730794 | -0.00483988 | 1.17135 |
| 23 | -0.0536874 | -0.0582375 | 0.139239 | 0.134689 | -0.192926 | -0.00455014 | 0.0326787 |
| 31 | 0.138803 | 0.0372267 | 0.0914811 | -0.0100947 | 0.0473214 | -0.101576 | 1.11035 |
| 9 | -0.348088 | -0.267068 | -0.294651 | -0.213631 | -0.0534369 | 0.08102 | 0.274969 |
| 21 | -0.262884 | -0.36288 | 0.004842 | -0.0951545 | -0.267726 | -0.0999965 | 20.6519 |

## V0 Preserved Hits

| Sample | Truth | V0 selected | V4 selected | V4 changes candidate | Outcome |
| ---: | --- | --- | --- | --- | --- |
| 10 | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | no | preserves |
| 14 | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | `<DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT abs x_0` | no | preserves |
| 17 | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | no | preserves |
| 18 | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0` | no | preserves |

## Resimulation Stability

Available: yes.

| Sample | Ordering stable | Oracle beats nonoracle in any repeat | Oracle beats nonoracle in majority | Oracle V4 mean/std | Nonoracle V4 mean/std |
| ---: | --- | --- | --- | --- | --- |
| 1 | no | yes | no | 0.784079/0.0936409 | 0.705645/0.0347094 |
| 7 | no | yes | yes | 0.945671/0.197593 | 1.42536/0.244203 |
| 22 | no | yes | no | 0.516249/0.0356252 | 0.505139/0.0309485 |
| 23 | no | yes | no | 0.674897/0.0914916 | 0.65176/0.0764086 |
| 31 | no | yes | yes | 0.514543/0.0375516 | 0.54543/0.0614151 |
| 9 | yes | no | no | 1.03303/0.0777603 | 0.929025/0.0896892 |
| 21 | yes | no | no | 0.467827/0.119573 | 0.279826/0.0816555 |

## Decision

Decision `C`: Some small resimulation orderings changed. Diagnose fingerprint simulation stability before scorer changes.

No formal eval, no rerank mode, and no P2 eval integration are recommended
from this diagnostic alone.
