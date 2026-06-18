# Role-wise Remaining Miss Analysis

## Executive Summary

Current strongest no-retraining setting:

```text
rerank_score=constant_grid_rolewise_no_multi_u0
active:weak=1:1
selected exact/relaxed=5/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
```

Role-wise constants rescued one shared-constant miss, sample 17. The remaining
oracle-present selected misses are samples `13`, `19`, `24`, and `26`.

These 4 misses are not candidate-absence failures: each has an oracle candidate
in the pool. They mostly look like score artifacts / fingerprint ambiguity in
the active+weak score: the selected non-oracle has lower active distance and
lower weak distance than the oracle in all 4 cases. Model score conflict is only
clear in sample 13, where the oracle has lower model score than the selected
candidate. Sample 24 is the only remaining miss whose best oracle is pair-only.

Candidate generation is still the larger global ceiling, because only `9/32`
samples contain any oracle candidate.

## Per-sample Details

| Sample | Relation | Source | Total score selected/oracle | Active selected/oracle | Weak selected/oracle | Constants selected drift/diff / oracle drift/diff | Model score selected/oracle | Shared rank -> role-wise rank | Variant effect | Explanation |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 13 | both drift and diffusion wrong | beam / `sample_t=1.2` | `1.303022 / 1.674129` | `0.537241 / 0.730690` | `0.765781 / 0.943439` | `0.5,0.5 / 0.25,1` | `-0.628455 / -0.714439` | `13 -> 4` | 2:1 and 1:0.5 improve rank to 3 but do not rescue | Role-wise constants substantially lower oracle score from shared `2.641107`, but the selected non-oracle remains better on both active and weak segments. This is the clearest model-score conflict too. |
| 19 | both drift and diffusion wrong | beam / beam | `1.578646 / 1.903220` | `0.721302 / 0.941829` | `0.857344 / 0.961391` | `0.25,0.5 / 0.25,0.25` | `-0.627859 / -0.536391` | `4 -> 7` | 2:1 and 1:0.5 still miss; no help | Oracle has better model score, but worse active and weak distances. Role-wise constants slightly improve the selected non-oracle and worsen oracle rank. |
| 24 | same diffusion, wrong drift | beam / pair | `1.479376 / 2.019098` | `0.648062 / 1.071399` | `0.831314 / 0.947699` | `0.5,0.5 / 0.25,0.5` | `-0.627931 / -0.168960` | `9 -> 11` | 2:1 and 1:0.5 still miss; no help | Oracle score improves from shared `2.270083`, but still loses badly on active distance. This is pair-only for the oracle; pairing provides coverage but the score rejects the paired drift. |
| 26 | same drift, wrong diffusion | beam / beam | `1.222154 / 1.445924` | `0.580150 / 0.674083` | `0.642003 / 0.771841` | `0.25,1 / 0.25,0.5` | `-0.617298 / -0.536036` | `3 -> 3` | 2:1 and 1:0.5 hurt rank to 4; no rescue | Role-wise constants create a stronger non-oracle with the correct drift but constant-only diffusion. The oracle has better model score but loses on both active and weak distances. |

## Sequences

### Sample 13

Truth / oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Selected drift:

```text
mul CONSTANT mul x_0 sub CONSTANT x_0
```

Oracle drift:

```text
mul mul CONSTANT CONSTANT sin x_0
```

Selected diffusion:

```text
mul CONSTANT sqrt abs x_0
```

Oracle diffusion:

```text
mul CONSTANT CONSTANT
```

### Sample 19

Truth / oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Selected drift:

```text
mul CONSTANT mul x_0 sub CONSTANT x_0
```

Oracle drift:

```text
mul mul CONSTANT CONSTANT CONSTANT
```

Selected diffusion:

```text
mul CONSTANT sqrt abs x_0
```

Oracle diffusion:

```text
add CONSTANT mul CONSTANT abs x_0
```

### Sample 24

Truth / oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Selected drift:

```text
mul CONSTANT mul x_0 sub CONSTANT x_0
```

Oracle drift:

```text
mul mul CONSTANT CONSTANT CONSTANT
```

Selected diffusion and oracle diffusion match:

```text
mul CONSTANT sqrt abs x_0
```

### Sample 26

Truth / oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0
```

Selected:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT CONSTANT
```

Selected drift and oracle drift match:

```text
mul mul CONSTANT CONSTANT CONSTANT
```

Selected diffusion:

```text
mul CONSTANT CONSTANT
```

Oracle diffusion:

```text
add CONSTANT mul CONSTANT abs x_0
```

## Aggregate Diagnosis

- Remaining oracle-present misses: `4`.
- Wrong drift with same diffusion: `1` (`sample 24`).
- Wrong diffusion with same drift: `1` (`sample 26`).
- Both drift and diffusion wrong: `2` (`samples 13, 19`).
- Improved by role-wise constants but not rescued: `2` clearly (`samples 13, 24`); sample 26 changes selected behavior but still misses.
- Worsened by role-wise constants: `2` by oracle rank (`samples 19, 24`); sample 26 stays rank 3 but selects a new non-oracle.
- Active-heavy `2:1` helped no remaining miss. It improved sample 13's oracle rank from 4 to 3 but still missed, and it lost the sample 17 rescue.
- Weak-downweighted `1:0.5` helped no remaining miss. Its behavior matched the active-heavy run qualitatively and also lost the sample 17 rescue.

The common pattern is that the selected non-oracle wins both active and weak
segment distances. That makes these cases unlikely to be fixed by a simple
global active/weak scalar. The score is treating some wrong templates as
fingerprint-near under the current active+weak view.

## Next Recommendation

Keep `constant_grid_rolewise_no_multi_u0` active:weak `1:1` as the current
strongest setting, but stop broad active/weak weight search.

The next low-risk step is targeted diagnostics, not a new heuristic:

- inspect per-feature active Kramers-Moyal and weak-kernel residuals for samples
  `13`, `19`, `24`, and `26`;
- compare whether the selected non-oracle is genuinely fingerprint-close under
  simulated trajectories or exploiting a normalization/constant artifact;
- preserve role-wise constants in future decoding experiments.

If a scoring change is still desired after that, it should be motivated by
which active or weak subfeatures are ambiguous in these 4 cases. Otherwise,
move attention back to candidate generation, because 23/32 samples still have
no oracle candidate.

No new formal eval is needed for this analysis. A future eval would only be
needed after adding a targeted diagnostic or scoring change.
