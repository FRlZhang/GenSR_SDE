# Role-wise Residual Diagnostics

Date: 2026-06-18

Scope: existing-log analysis only. No new formal 32-sample eval, no 64-sample
eval, no retraining, no active/weak grid, and no new rerank heuristic.

Inputs:

- `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`
- `rolewise_remaining_miss_analysis.md`
- project status files

## Executive Summary

Current role-wise baseline:

- selected exact/relaxed: `5/32`
- oracle ceiling: `9/32`
- pair oracle exact/relaxed: `2/32`
- remaining oracle-present selected misses: `4` (`13`, `19`, `24`, `26`)

From the existing role-wise log, all four remaining misses are scored as better
non-oracles under both included score segments:

- `active_kramers_moyal`
- `gaussian_weak_kernel`

This means the remaining selected/oracle gap is not a near-tie issue and was not
helped by the already-tested active-heavy `2:1` or weak-downweighted `1:0.5`
variants. The misses look mostly like fingerprint ambiguity or feature
normalization/scorer calibration artifacts under the current active+weak
statistics. Candidate generation is still the larger ceiling overall because
only `9/32` samples contain any oracle candidate, but these four specific cases
already contain an oracle and need residual-level diagnostics before more search
or heuristics.

Important limitation: the existing logs contain aggregate segment distances, not
feature-by-feature residual vectors. Exact "top features where selected beats
oracle" and exact role-swap scores for all combinations cannot be reconstructed
from the logs alone. This report records what is available and lists the
smallest missing diagnostic at the end.

## Per-sample Summary

| Sample | Miss side | Selected source | Oracle source | Selected total | Oracle total | Gap | Selected active | Oracle active | Selected weak | Oracle weak | Drift constants | Diffusion constants | Shared roles | Model score relation | Rank shared -> role-wise | Variant effect |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 13 | both | beam | sample_t=1.2 | 1.303022 | 1.674129 | 0.371107 | 0.537241 | 0.730690 | 0.765781 | 0.943439 | selected 0.5, oracle 0.25 | selected 0.5, oracle 1 | none | oracle has lower model score | 13 -> 4 | 2:1 and 1:0.5 improve rank only, no rescue |
| 19 | both | beam | beam | 1.578646 | 1.903220 | 0.324574 | 0.721302 | 0.941829 | 0.857344 | 0.961391 | selected 0.25, oracle 0.25 | selected 0.5, oracle 0.25 | none | oracle model score is better | 4 -> 7 | no help |
| 24 | drift | beam | pair | 1.479376 | 2.019098 | 0.539722 | 0.648062 | 1.071399 | 0.831314 | 0.947699 | selected 0.5, oracle 0.25 | selected 0.5, oracle 0.5 | diffusion shared | oracle model score is better but pair-only | 9 -> 11 | no help |
| 26 | diffusion | beam | beam | 1.222154 | 1.445924 | 0.223770 | 0.580150 | 0.674083 | 0.642003 | 0.771841 | selected 0.25, oracle 0.25 | selected 1, oracle 0.5 | drift shared | oracle model score is better | 3 -> 3 | 2:1 and 1:0.5 hurt rank to 4, no rescue |

Segment-level pattern:

- Selected wins active and weak on `4/4` remaining misses.
- Active contributes the larger selected advantage on samples `13`, `19`, and
  `24`.
- Weak contributes the larger selected advantage on sample `26`.
- There is no evidence in the existing aggregate logs that a simple active/weak
  scalar weight would rescue any case; the two tested weight variants rescued
  none.

## Sample 13

Truth and oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Available aggregate residual comparison:

| Segment | Selected | Oracle | Selected advantage |
| --- | ---: | ---: | ---: |
| active_kramers_moyal | 0.537241 | 0.730690 | 0.193449 |
| gaussian_weak_kernel | 0.765781 | 0.943439 | 0.177658 |
| total active+weak | 1.303022 | 1.674129 | 0.371107 |

Role-wise constants lowered the oracle score substantially versus the shared
constant score (`2.641107 -> 1.674129`) and improved the oracle rank
(`13 -> 4`), but the selected non-oracle still wins both aggregate segments.
The oracle also has a lower model score (`-0.714439` vs `-0.628455`), so this is
the only remaining miss with an explicit model-score conflict in addition to the
active+weak score conflict.

Counterfactual role-swap status:

| Combination | Status |
| --- | --- |
| selected drift + selected diffusion | parseable/fingerprintable; total `1.303022`, active `0.537241`, weak `0.765781`, constants `0.5/0.5` |
| oracle drift + oracle diffusion | parseable/fingerprintable; total `1.674129`, active `0.730690`, weak `0.943439`, constants `0.25/1` |
| selected drift + oracle diffusion | not present in existing log |
| oracle drift + selected diffusion | not present in existing log |

Classification: both-side ambiguity with model-score conflict. Exact
feature-level residuals are needed to tell whether the selected advantage is
broad or dominated by a few active/weak features.

## Sample 19

Truth and oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Available aggregate residual comparison:

| Segment | Selected | Oracle | Selected advantage |
| --- | ---: | ---: | ---: |
| active_kramers_moyal | 0.721302 | 0.941829 | 0.220527 |
| gaussian_weak_kernel | 0.857344 | 0.961391 | 0.104047 |
| total active+weak | 1.578646 | 1.903220 | 0.324574 |

The oracle model score is better than the selected model score
(`-0.536391` vs `-0.627859`), so this is not a model-score conflict. Role-wise
constants slightly improve the selected candidate and leave the oracle rank
worse (`4 -> 7`), which suggests the role-wise constant degrees of freedom make
nearby non-oracle templates even more competitive.

One useful partial role-swap is available from candidates before the oracle:

```text
oracle drift + selected diffusion
```

This pair candidate scores `1.683207` (`active=0.756883`,
`weak=0.926324`), between selected and oracle. It improves over the oracle on
both active and weak while keeping the oracle drift, which points to the oracle
diffusion as one part of the score penalty. The selected drift + oracle diffusion
combination is not present in the existing log.

Counterfactual role-swap status:

| Combination | Status |
| --- | --- |
| selected drift + selected diffusion | parseable/fingerprintable; total `1.578646`, active `0.721302`, weak `0.857344`, constants `0.25/0.5` |
| oracle drift + oracle diffusion | parseable/fingerprintable; total `1.903220`, active `0.941829`, weak `0.961391`, constants `0.25/0.25` |
| selected drift + oracle diffusion | not present in existing log |
| oracle drift + selected diffusion | present as pair; total `1.683207`, active `0.756883`, weak `0.926324`, constants not separately logged for this rank |

Classification: both-side ambiguity, with evidence that the diffusion side is a
major contributor to the oracle penalty. Exact feature residuals are still
needed before changing the score.

## Sample 24

Truth and oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Selected:

```text
<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Available aggregate residual comparison:

| Segment | Selected | Oracle | Selected advantage |
| --- | ---: | ---: | ---: |
| active_kramers_moyal | 0.648062 | 1.071399 | 0.423337 |
| gaussian_weak_kernel | 0.831314 | 0.947699 | 0.116386 |
| total active+weak | 1.479376 | 2.019098 | 0.539722 |

This is the cleanest drift-side miss: selected and oracle share diffusion
exactly, and only the drift differs. The oracle is pair-only, so pairing is
necessary for candidate availability, but the current active+weak score strongly
prefers the non-oracle drift. Role-wise constants lower the oracle score
(`2.270083 -> 2.019098`) but worsen its rank (`9 -> 11`).

Counterfactual role-swap status:

Because selected and oracle share diffusion, the four role-swap combinations
collapse to two unique expressions:

| Combination | Status |
| --- | --- |
| selected drift + selected diffusion | parseable/fingerprintable; total `1.479376`, active `0.648062`, weak `0.831314`, constants `0.5/0.5` |
| oracle drift + oracle diffusion | parseable/fingerprintable; total `2.019098`, active `1.071399`, weak `0.947699`, constants `0.25/0.5` |
| selected drift + oracle diffusion | same as selected |
| oracle drift + selected diffusion | same as oracle |

Classification: drift-side ambiguity plus pair-only oracle evidence. This case
is the strongest warning that current active+weak statistics can prefer the
wrong drift even when diffusion is held fixed.

## Sample 26

Truth and oracle:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0
```

Selected:

```text
<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT CONSTANT
```

Available aggregate residual comparison:

| Segment | Selected | Oracle | Selected advantage |
| --- | ---: | ---: | ---: |
| active_kramers_moyal | 0.580150 | 0.674083 | 0.093933 |
| gaussian_weak_kernel | 0.642003 | 0.771841 | 0.129838 |
| total active+weak | 1.222154 | 1.445924 | 0.223770 |

This is the cleanest diffusion-side miss: selected and oracle share drift
exactly, and only the diffusion differs. Under shared constants, the oracle beat
the role-wise selected expression, but after role-wise constants the selected
constant diffusion becomes much stronger (`shared selected score 2.215338`,
`role-wise selected score 1.222154`). This suggests role-wise constants can
amplify a simple non-oracle diffusion when the active+weak fingerprint does not
separate the diffusion forms strongly enough.

Counterfactual role-swap status:

Because selected and oracle share drift, the four role-swap combinations
collapse to two unique expressions:

| Combination | Status |
| --- | --- |
| selected drift + selected diffusion | parseable/fingerprintable; total `1.222154`, active `0.580150`, weak `0.642003`, constants `0.25/1` |
| oracle drift + oracle diffusion | parseable/fingerprintable; total `1.445924`, active `0.674083`, weak `0.771841`, constants `0.25/0.5` |
| selected drift + oracle diffusion | same as oracle |
| oracle drift + selected diffusion | same as selected |

Classification: diffusion-side ambiguity with role-wise constant amplification.
This is a targeted candidate for feature normalization diagnostics, especially
inside the weak-kernel residuals where the selected advantage is largest.

## Aggregate Diagnosis

Counts over the four remaining role-wise misses:

- wrong drift only: `1` (`24`)
- wrong diffusion only: `1` (`26`)
- both drift and diffusion wrong: `2` (`13`, `19`)
- selected wins both active and weak aggregate distances: `4`
- model-score conflict: `1` (`13`)
- oracle only from pair: `1` (`24`)
- role-wise constants improved oracle score but did not rescue: `2` clear cases
  (`13`, `24`)
- role-wise constants worsened oracle rank: `2` (`19`, `24`)
- active-heavy `2:1` or weak-downweighted `1:0.5` helped any miss: `0`

Interpretation:

- The remaining misses do not look like tie-break errors.
- They do not look fixable by broad active/weak scalar reweighting.
- Samples `24` and `26` are especially useful because one role is held fixed,
  exposing a direct drift-side and diffusion-side ambiguity respectively.
- Samples `13` and `19` need true role-swap residuals before a targeted score
  change can be justified.

## Missing Residual Fields

The existing logs are insufficient for exact per-feature residual breakdown.
Missing fields:

- active feature residual vector for selected and oracle
- weak-kernel feature residual vector for selected and oracle
- feature names or stable feature indices for both segments
- per-feature normalization/scaling factors
- role-swap scores for `selected drift + oracle diffusion` and
  `oracle drift + selected diffusion` where those combinations are not already
  present in the candidate log

Because those fields are absent, this report cannot truthfully list top
feature-level wins/losses. It can only say that selected wins both aggregate
segments on all four remaining misses.

## Smallest Suggested Next Diagnostic

Do not run a formal eval. The smallest useful follow-up is a targeted residual
dump for the four known samples, reusing the same checkpoint/eval setup but
printing only:

- selected/oracle active residual vectors
- selected/oracle weak residual vectors
- the four role-swap candidates per sample
- per-feature deltas sorted by selected advantage and oracle advantage

Suggested command shape, for a future workflow only:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_rolewise_residuals.py \
  --samples 13,19,24,26 \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
  --rerank-score constant_grid_rolewise_no_multi_u0 \
  --rerank-constant-values 0.25,0.5,1.0,2.0,4.0 \
  > /private/tmp/gensr_sde_rolewise_residual_vectors.log 2>&1
```

That helper should be narrow and diagnostic-only. It should not change target
format, fingerprint schema, training path, or rerank selection behavior.

## Recommendation

Do not add another rerank heuristic from the current aggregate logs alone.
Before touching the scorer, run one targeted residual-vector diagnostic for
samples `13`, `19`, `24`, and `26`.

If feature-level deltas show only a few dominant normalized features driving the
wrong selection, continue with feature normalization diagnostics. If selected
wins broadly across many active and weak features, stop squeezing the scorer for
these four cases and move back to candidate generation and fingerprint
identifiability checks.
