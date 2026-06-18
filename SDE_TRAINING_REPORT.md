# GenSR-SDE Training Report

Date: 2026-06-11

Update, 2026-06-18 candidate coverage diagnostic: added
`scripts/analyze_candidate_coverage.py` and wrote
`candidate_coverage_diagnostics.md`. This was a candidate-generation diagnostic
only: no fingerprint rerank scoring, no formal 32-sample eval, no 64-sample
eval, and no retraining. The helper loaded the 2000-step checkpoint read-only
and regenerated the 32 held-out candidate pools with beam, sampling, and
drift/diffusion pairing.

Coverage summary:

| Category | Count |
| --- | ---: |
| Full oracle present | 9 |
| Scorer selected oracle | 5 |
| Oracle present but selected missed | 4 |
| Drift-only present | 0 |
| Diffusion-only present | 17 |
| Drift+diffusion separately present but not paired | 6 |
| Neither side present | 0 |

Source breakdown:

| Source | Candidate rows | Full oracle samples | Exact drift samples | Exact diffusion samples |
| --- | ---: | ---: | ---: | ---: |
| beam | 256 | 6 | 15 | 32 |
| sampling | 46 | 1 | 3 | 8 |
| pair | 103 | 2 | 7 | 17 |

Interpretation: the `9/32` oracle ceiling is mainly a drift-span coverage
problem. Among the 23 oracle-absent samples, 17 already contain the truth
diffusion but miss the truth drift; the other 6 contain exact drift and exact
diffusion separately but do not pair them into a full oracle. No oracle-absent
sample is missing both sides. Next candidate-generation work should prioritize
drift diversity first, then pairing coverage/ranking.

Update, 2026-06-18 offline residual score ablation: added
`scripts/analyze_residual_score_ablation.py`. Because the previous residual
helper's log did not contain machine-readable full residual vectors, the
residual helper now also emits
`rolewise_residual_vector_diagnostics.json`. The ablation report is
`rolewise_residual_score_ablation.md`.

This was offline analysis only: no formal 32-sample eval, no 64-sample eval, no
retraining, no active/weak grid, and no new rerank mode.

Results over samples `13`, `19`, `24`, and `26`:

| Offline score variant family | Oracle wins |
| --- | ---: |
| Current active+weak L2 sum | 0/4 |
| L1 residual sum | 0/4 |
| L2 residual sum | 0/4 |
| Mild clipped/Huber/per-segment normalization | 0/4 |
| Aggressive clipping / top-k removal | 1/4 |

Only sample 19 flipped, and only under aggressive `clipped_l2_p90`; sample 24
drift-side ambiguity and sample 26 diffusion-side ambiguity did not flip under
any tested offline variant. Interpretation: a formal 32-sample robust-scoring
eval is not justified yet. The remaining misses look more like fingerprint
ambiguity / candidate evidence limitations than an easy feature-normalization
scorer fix.

Update, 2026-06-18 targeted residual-vector diagnostics: added
`scripts/analyze_rolewise_residuals.py` and wrote
`rolewise_residual_vector_diagnostics.md`. This was a diagnostic-only run, not
a formal 32-sample eval: it reproduced the known remaining role-wise selected
misses (`13`, `19`, `24`, `26`) from the eval seed and scored only the selected
/ oracle role-swap combinations.

The helper completed in about 9 seconds with:

```text
/private/tmp/gensr_sde_rolewise_residual_vectors.log
```

Diagnostic classifications:

| Sample | Classification |
| ---: | --- |
| 13 | both-side ambiguity; feature-scaling artifact; model-score conflict |
| 19 | both-side ambiguity; feature-scaling artifact |
| 24 | drift-side ambiguity; feature-scaling artifact |
| 26 | diffusion-side ambiguity; feature-scaling artifact |

The recomputed residual vectors still show selected non-oracles beating the
oracle on aggregate active+weak distance for all four cases. The role-swap
diagnostic is most decisive for samples 24 and 26 because one role is shared:
sample 24 isolates a drift-side ambiguity, while sample 26 isolates a
diffusion-side ambiguity. This supports a targeted feature-normalization /
residual-scaling diagnostic before adding another rerank heuristic. It does not
change the current headline metric: role-wise selected exact/relaxed remains
`5/32` against an oracle ceiling of `9/32`.

Update, 2026-06-18 role-wise constant handling: added
`constant_grid_rolewise_no_multi_u0`, which keeps the no-`multi_u0` active+weak
score but searches separate role-wise constants:

```text
drift_CONSTANT_value in 0.25,0.5,1.0,2.0,4.0
diffusion_CONSTANT_value in 0.25,0.5,1.0,2.0,4.0
```

The 32-sample eval-only checkpoint setup was unchanged otherwise. Results:

| Setting | Selected exact | Selected relaxed | Oracle exact | Pair oracle exact | Shared-miss rescues |
| --- | ---: | ---: | ---: | ---: | ---: |
| `constant_grid_componentwise_no_multi_u0` shared constant | 0.125000 | 0.125000 | 0.281250 | 0.062500 | - |
| `constant_grid_rolewise_no_multi_u0` active:weak 1:1 | 0.156250 | 0.156250 | 0.281250 | 0.062500 | 1/5 |
| `constant_grid_rolewise_no_multi_u0` active:weak 2:1 | 0.093750 | 0.093750 | 0.281250 | 0.062500 | 0/6 |
| `constant_grid_rolewise_no_multi_u0` active:weak 1:0.5 | 0.093750 | 0.093750 | 0.281250 | 0.062500 | 0/6 |

The default 1:1 role-wise setting rescued sample 17, a same-drift /
wrong-diffusion miss under shared constants:

```text
truth/oracle diffusion: mul CONSTANT sqrt abs x_0
shared selected diffusion: mul CONSTANT abs x_0
rolewise selected/oracle score: 0.818141
shared selected/oracle scores: 0.872129 / 0.903798
rolewise best constants: drift=1, diffusion=0.5
```

Role-wise constants also lowered several oracle scores without selecting them.
For example sample 13 improved oracle score from `2.641107` to `1.674129`, and
sample 24 improved from `2.270083` to `2.019098`, but the selected non-oracles
still had lower active+weak scores. Active-heavy and weak-downweighted variants
both regressed to `3/32`, so broad active/weak weight search is not the next
best move. Interpretation: role-wise constant handling is useful and should be
kept as the current strongest decoding setting, but the remaining selected
misses require better score discrimination or candidate-level evidence, not
just another simple active/weak weighting.

Update, 2026-06-18 oracle-gap diagnostics: `sde_validation_probe.py` now prints
an oracle case summary and per-case miss diagnostics for reranked candidates.
The 32-sample no-tie eval-only setting was rerun from the same checkpoint with
the same candidate/scoring configuration:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

The run reproduced the previous selected/oracle metrics:

```text
reranked_sequence_exact=0.125000
reranked_sequence_relaxed_no_constants=0.125000
rerank_oracle_sequence_exact=0.281250
rerank_oracle_sequence_relaxed_no_constants=0.281250
rerank_pair_oracle_sequence_exact=0.062500
rerank_pair_oracle_sequence_relaxed_no_constants=0.062500
rerank_candidates_attempted=405
rerank_candidates_valid=405
rerank_parse_failures=0
rerank_fingerprint_failures=0
```

Oracle case summary:

| Diagnostic | Count |
| --- | ---: |
| Total samples | 32 |
| Samples with any oracle candidate | 9 |
| Samples where selected is oracle | 4 |
| Samples where oracle exists but selected misses | 5 |
| Best oracle source: beam | 6 |
| Best oracle source: sampling | 1 |
| Best oracle source: pair | 2 |
| Selected hit source: beam | 3 |
| Selected hit source: pair | 1 |

Selected-miss types:

| Miss type | Count |
| --- | ---: |
| Constant-only mismatch | 0 |
| Same diffusion but wrong drift | 2 |
| Same drift but wrong diffusion | 1 |
| Both drift and diffusion wrong | 2 |
| Oracle is lower model-score candidate | 2 |
| Oracle only appears from pair | 1 |
| Oracle appears from beam/sampling but score misses | 4 |

Interpretation: the immediate selected-versus-oracle gap is a rerank
score/constant-handling problem. In the 5 miss cases the oracle template is
present, but the selected non-oracle usually has a lower active/weak rerank
distance. The larger ceiling remains candidate generation: 23/32 samples still
have no oracle candidate at all.

Update, 2026-06-18: the near-tie rerank tie-breaks were validated on a larger
32-sample eval-only probe using the 2000-step role-token checkpoint:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

All three settings used `constant_grid_componentwise_no_multi_u0`,
`--rerank-tie-epsilon 0.005`, beam+sampling candidates, and drift/diffusion
pairing. Only `--rerank-tie-break` changed:

| Tie-break | Selected exact | Selected relaxed | Oracle exact | Oracle relaxed | Pair oracle exact | Pair oracle relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0.125000 | 0.125000 | 0.281250 | 0.281250 | 0.062500 | 0.062500 |
| `active_distance` | 0.125000 | 0.125000 | 0.281250 | 0.281250 | 0.062500 | 0.062500 |
| `state_dependent_drift` | 0.125000 | 0.125000 | 0.281250 | 0.281250 | 0.062500 | 0.062500 |

Shared candidate diagnostics:

```text
rerank_candidates_attempted=405
rerank_candidates_valid=405
rerank_fingerprint_failures=0
rerank_unique_candidate_avg=12.656250
rerank_unique_drift_avg=4.468750
rerank_unique_diffusion_avg=5.000000
rerank_unique_paired_candidate_avg=3.218750
```

Interpretation: the 16-sample near-tie hit was real as a diagnostic, but it did
not scale into a stable improvement over the no-tie baseline on 32 samples.
Selected reranking now recovers 4/32 while the candidate oracle contains 9/32,
so the next work should focus on candidate generation, pairing coverage, and
constant handling rather than adding another tie-break heuristic. Because
neither tie-break exceeded baseline and each 32-sample run was CPU-expensive,
the 64-sample extension was not run.

Update, 2026-06-12: the original SDE data injection patched
`env.generate_sample`, but the GenSR train dataloader actually calls
`EnvDataset.generate_sample`. The 100-step and 500-step loss probes below are
therefore useful as training-loop smoke tests, but they should not be treated as
valid SDE training evidence. The injection path has been corrected to patch
`EnvDataset.generate_sample`, and old pickle files with token-id tensors are
converted back to token words at load time.

First corrected SDE smoke test:

```text
step 1
VAE-TOTAL      18.1638
VAE-PRED-PRIOR 9.0763
VAE-PRED-POST  9.0876
```

This confirms that real `multi_active_weak_v1` SDE samples now enter the CVAE
training loop.

Corrected 100-step held-out validation probe:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 100 --eval-samples 32 --batch-size 8 \
  --print-freq 25 --n-paths 600 --active-paths 600 \
  --n-steps 60 --max-generated-len 40
```

Training loss:

| Step | Prior pred | Post pred | KL | KL weight |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 9.0098 | 8.9975 | 3.3874 | 0.0960 |
| 50 | 8.1580 | 8.1466 | 6.0940 | 0.1960 |
| 75 | 7.7758 | 7.7675 | 5.3590 | 0.2000 |
| 100 | 7.4673 | 7.4612 | 4.5623 | 0.2000 |

Held-out SDE metrics on 32 newly generated samples:

```text
loss=7.674738
token_top1=0.098266
token_top3=0.190858
token_top5=0.190858
sequence_exact=0.000000
sequence_relaxed_no_constants=0.000000
```

Interpretation: real SDE training now shows a clear teacher-forced signal, with
training prior loss dropping from about 9.01 to 7.47 and held-out loss reaching
7.67 after only 100 steps. Full sequence generation is still not working in this
short run because greedy decoding often terminates immediately. The next
validation step should combine longer true-SDE training with beam or
minimum-length-constrained decoding.

Corrected 500-step held-out validation probe:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 500 --eval-samples 64 --batch-size 8 \
  --print-freq 100 --n-paths 800 --active-paths 800 \
  --n-steps 60 --max-generated-len 40 --min-generated-len 8
```

Training loss:

| Step | Prior pred | Post pred | KL |
| ---: | ---: | ---: | ---: |
| 100 | 7.3236 | 7.3181 | 1.7207 |
| 200 | 4.8055 | 4.8067 | 1.3003 |
| 300 | 3.3877 | 3.3882 | 0.4917 |
| 400 | 2.0467 | 2.0468 | 0.1700 |
| 500 | 1.2253 | 1.2253 | 0.0904 |

Held-out SDE metrics on 64 newly generated samples:

```text
loss=1.377840
token_top1=0.652865
token_top3=0.871775
token_top5=0.917750
greedy_sequence_exact=0.000000
greedy_sequence_relaxed_no_constants=0.000000
greedy_sequence_nonempty=1.000000
greedy_sequence_avg_len=7.000000
greedy_sequence_structural_token_frac=1.000000
minlen_greedy_sequence_exact=0.000000
minlen_greedy_sequence_relaxed_no_constants=0.000000
minlen_greedy_sequence_nonempty=1.000000
minlen_greedy_sequence_avg_len=7.000000
minlen_greedy_sequence_structural_token_frac=1.000000
```

Typical generated sequence:

```text
truth: mul CONSTANT x_0 SPECIAL mul CONSTANT sqrt abs x_0
pred : mul CONSTANT CONSTANT CONSTANT CONSTANT CONSTANT x_0
```

Interpretation: the unified SDE fingerprint is strong enough for held-out
teacher-forced token prediction. A 500-step true-SDE run reaches 65.3% top-1
token accuracy and 91.8% top-5 token accuracy on unseen SDE samples. The current
failure is sequence-level autoregressive decoding: greedy decoding collapses to
short structural patterns dominated by `CONSTANT`, so exact expression recovery
is still zero. This supports continuing the SDE-fingerprint route, but the next
work item should be beam/reranking or template-constrained decoding rather than
changing the fingerprint immediately.

## Split Drift/Diffusion Decoder Probe

Following the sequence-collapse result above, an explicit SDE-role version was
implemented:

- add `<DRIFT>` and `<DIFFUSION>` role tokens to the equation vocabulary
- store `drift_tree_encoded` and `diffusion_tree_encoded` in generated SDE data
- keep the numerical fingerprint encoder and CVAE shared
- compare split-loss decoding against a single role-token sequence
- optionally train with `CE_drift + CE_diffusion + KL`

The old pickle format is still supported by reconstructing token sequences from
the stored `drift | diffusion` expression string.

Because two full 16-layer decoders are slow on CPU, the first diagnostic run used
a smaller model:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 500 --eval-samples 64 --batch-size 8 \
  --print-freq 100 --n-paths 600 --active-paths 600 \
  --n-steps 60 --max-generated-len 32 --min-generated-len 6 \
  --n-enc-layers 4 --n-dec-layers 4 --n-heads 8 --model-dim 256
```

Training loss:

| Step | Prior pred | Post pred | KL |
| ---: | ---: | ---: | ---: |
| 100 | 18.0082 | 18.0127 | 0.7913 |
| 200 | 16.8219 | 16.8268 | 1.0814 |
| 300 | 16.9757 | 16.9798 | 0.8175 |
| 400 | 15.8931 | 15.8964 | 0.4914 |
| 500 | 14.3644 | 14.3661 | 0.2988 |

Held-out SDE metrics on 64 newly generated samples:

```text
drift_loss=7.069069
drift_token_top1=0.163330
drift_token_top3=0.225208
drift_token_top5=0.225208
diffusion_loss=7.580018
diffusion_token_top1=0.027432
diffusion_token_top3=0.100124
diffusion_token_top5=0.285800
```

Interpretation: explicit drift/diffusion role tokens are a good idea, but two
fully independent decoders are not sample-efficient in this first implementation.
Each decoder only sees half of the symbolic supervision while the number of
decoder parameters roughly doubles, so short training underperforms the earlier
single-decoder role-agnostic probe. The next better variant is not "two full
decoders", but a shared decoder trunk with role prompts or small role-specific
heads, followed by beam/reranking.

## Role-Token Single-Sequence Probe

The next diagnostic kept one full GenSR decoder but changed the symbolic target
from

```text
drift SPECIAL diffusion
```

to

```text
<DRIFT> drift <DIFFUSION> diffusion
```

This preserves all token supervision in one autoregressive sequence while giving
the decoder explicit SDE role markers.

Small-model smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 500 --eval-samples 64 --batch-size 8 \
  --print-freq 100 --n-paths 600 --active-paths 600 \
  --n-steps 60 --max-generated-len 40 --min-generated-len 8 \
  --n-enc-layers 4 --n-dec-layers 4 --n-heads 8 --model-dim 256
```

Held-out metrics were weak with the reduced model:

```text
loss=7.951664
token_top1=0.106068
token_top3=0.117006
token_top5=0.132820
greedy_sequence_exact=0.000000
```

The same role-token target was then tested with the full GenSR decoder:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 500 --eval-samples 64 --batch-size 8 \
  --print-freq 100 --n-paths 800 --active-paths 800 \
  --n-steps 60 --max-generated-len 40 --min-generated-len 8
```

Training loss:

| Step | Prior pred | Post pred | KL |
| ---: | ---: | ---: | ---: |
| 100 | 6.9348 | 6.9467 | 1.3534 |
| 200 | 4.4803 | 4.4844 | 0.9812 |
| 300 | 3.1044 | 3.1049 | 0.3420 |
| 400 | 1.8576 | 1.8577 | 0.1284 |
| 500 | 1.1279 | 1.1281 | 0.0723 |

Held-out SDE metrics on 64 newly generated samples:

```text
loss=1.266209
token_top1=0.686656
token_top3=0.881275
token_top5=0.927509
greedy_sequence_exact=0.000000
greedy_sequence_relaxed_no_constants=0.046875
greedy_sequence_nonempty=1.000000
greedy_sequence_avg_len=11.000000
greedy_sequence_structural_token_frac=0.818182
```

Typical generated sequence:

```text
truth: <DRIFT> mul CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
pred : <DRIFT> mul CONSTANT CONSTANT CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT x_0
```

Interpretation: explicit role tokens do not hurt the main teacher-forced signal.
With the full decoder, the role-token target slightly improves the corrected
500-step held-out token metrics over the old `SPECIAL` separator baseline:
top-1 rises from 65.3% to 68.7%, and top-5 rises from 91.8% to 92.8%.
The remaining failure is still autoregressive sequence recovery, not numerical
fingerprint separability. The recommended near-term path is therefore:

1. keep `<DRIFT>` / `<DIFFUSION>` as the default symbolic target format;
2. keep split drift/diffusion loss as an optional diagnostic, not the default;
3. add grammar or template-constrained decoding plus fingerprint-based reranking;
4. run a longer checkpointed role-token training job after the decoding path is
   measurable.

### 2000-Step Checkpointed Role-Token Probe

The validation probe now supports checkpoint reuse:

```text
--save-checkpoint PATH
--load-checkpoint PATH
--eval-only
--constrained-beam-size N
```

Checkpoint save/load was smoke-tested with a tiny 2-step model. A longer
role-token run was then launched with:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 2000 --eval-samples 64 --batch-size 8 \
  --print-freq 250 --n-paths 800 --active-paths 800 \
  --n-steps 60 --max-generated-len 40 --min-generated-len 8 \
  --save-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

Saved checkpoint:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

Training loss:

| Step | Prior pred | Post pred | KL |
| ---: | ---: | ---: | ---: |
| 250 | 3.9715 | 3.9739 | 0.1653 |
| 500 | 1.1305 | 1.1307 | 0.0319 |
| 750 | 0.6654 | 0.6654 | 0.0228 |
| 1000 | 0.4722 | 0.4721 | 0.0132 |
| 1250 | 0.5423 | 0.5421 | 0.0086 |
| 1500 | 0.5529 | 0.5529 | 0.0066 |
| 1750 | 0.4077 | 0.4079 | 0.0051 |
| 2000 | 0.3772 | 0.3771 | 0.0039 |

Held-out SDE metrics on 64 newly generated samples:

```text
loss=0.354153
token_top1=0.820002
token_top3=0.960833
token_top5=1.000000
greedy_sequence_exact=0.015625
greedy_sequence_relaxed_no_constants=0.015625
greedy_sequence_nonempty=1.000000
greedy_sequence_avg_len=10.000000
greedy_sequence_structural_token_frac=0.800000
```

The same checkpoint was then evaluated without retraining using an SDE
grammar-constrained beam:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --eval-only \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
  --eval-samples 32 --batch-size 8 \
  --n-paths 800 --active-paths 800 --n-steps 60 \
  --max-generated-len 40 --min-generated-len 8 \
  --constrained-beam-size 8
```

Constrained-beam held-out metrics on 32 newly generated samples:

```text
loss=0.360775
token_top1=0.802700
token_top3=0.959856
token_top5=1.000000
greedy_sequence_exact=0.000000
greedy_sequence_relaxed_no_constants=0.000000
constrained_beam_sequence_exact=0.031250
constrained_beam_sequence_relaxed_no_constants=0.031250
constrained_beam_sequence_nonempty=1.000000
constrained_beam_sequence_avg_len=13.000000
constrained_beam_sequence_structural_token_frac=0.846154
```

Typical constrained-beam sequence:

```text
truth: <DRIFT> mul CONSTANT sin x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
pred : <DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0
```

Interpretation: longer role-token training strongly improves teacher-forced
recognition. Top-1 token accuracy rises from 68.7% at 500 steps to 82.0% at
2000 steps, and top-5 reaches 100.0%. This is enough evidence that the current
SDE fingerprint is usable inside the GenSR numerical encoder.

However, autoregressive recovery remains the bottleneck. Greedy exact recovery is
still only 1/64, and grammar-constrained beam improves a 32-sample eval from 0
to only 1 exact/relaxed hit. The decoder's probability mass is concentrated on a
small number of plausible but wrong templates. The next experiment should
therefore generate diverse candidate templates and rerank them by SDE fingerprint
distance, instead of relying on greedy or a single constrained beam objective.

## Context

The current GenSR-SDE prototype extends GenSR's numerical branch from algebraic
function values to SDE fingerprints. The active training dataset is generated
from 1D autonomous SDEs and uses the unified `multi_active_weak_v1`
fingerprint.

The fingerprint length is 186, kept below GenSR's default sequence limit of 200.

Fingerprint schema:

| Segment | Slice | Description |
| --- | ---: | --- |
| `multi_u0_moments` | `[0, 72]` | Mean and variance over multiple initial conditions |
| `active_kramers_moyal` | `[72, 90]` | Short-time local drift and diffusion probes |
| `gaussian_weak_kernel` | `[90, 186]` | State-kernel weak statistics from simulated paths |

Current dataset:

- Path: `data/sde_train_dataset.pkl`
- Size: 1000 samples
- `x_to_fit` shape: `(186, 1)`
- `y_to_fit` shape: `(186, 1)`
- Fingerprint type: `multi_active_weak_v1`

## Training Command

The 100-step loss probe was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 train.py \
  --min_input_dimension 1 --max_input_dimension 1 \
  --min_output_dimension 1 --max_output_dimension 1 \
  --num_workers 0 --cpu True \
  --dump_path /private/tmp/gensr_sde_long_probe \
  --print_freq 5 --n_steps_per_epoch 100 --max_epoch 1 \
  --batch_size 8 --batch_size_eval 8 \
  --beam_eval False --save_periodic 0 --wandb_disabled
```

Run directory:

- `/private/tmp/gensr_sde_long_probe/debug/8qsrpsc4qq`

The run completed 100 steps in about 10 minutes and 39 seconds. It did not hit
shape errors, token errors, NaNs, or training crashes.

## Loss Summary

| Window | VAE total | Prior pred | Post pred | KL | KL weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 5-25 | 20.395 | 8.947 | 8.946 | 2.501 | 0.056 |
| Step 40-60 | 23.285 | 8.152 | 8.154 | 6.979 | 0.186 |
| Step 80-100 | 21.193 | 7.709 | 7.717 | 5.767 | 0.200 |
| KL fixed, step 55-100 | 22.154 | 7.919 | 7.925 | 6.311 | 0.200 |

Final printed step:

```text
step 100
VAE-TOTAL      20.8497
VAE-PRED-PRIOR 7.7285
VAE-PRED-POST  7.7377
VAE-KL         5.3835
VAE-KL-WEIGHT  0.2000
```

Best printed prior prediction loss:

```text
step 95
VAE-PRED-PRIOR 7.1782
VAE-PRED-POST  7.1842
```

Worst printed prior prediction loss:

```text
step 25
VAE-PRED-PRIOR 9.1951
VAE-PRED-POST  9.1947
```

## Interpretation

The result is positive as a training-chain sanity check. The total loss rises
early mainly because the KL weight anneals up to 0.2. The more important prior
and posterior prediction losses decrease from roughly 8.95 in the early window
to roughly 7.71 in the final window.

This does not yet prove successful symbolic identification. It shows that the
unified SDE fingerprint is numerically stable inside the GenSR CVAE training
loop and contains learnable signal. The next evaluation should use held-out
SDE systems and report token top-k, same-template relaxed accuracy, and
generator-semantic accuracy.

## Save Logic Fix

Before this report, `train.py` saved periodic checkpoints during the first five
epochs even when `--save_periodic 0` was passed. Each checkpoint was about
1.8 GB in this Mac CPU run.

The save condition was changed so early periodic checkpointing only happens
when `save_periodic > 0`. This makes `--save_periodic 0` mean "disable periodic
checkpoint saving", which is safer for longer CPU probes.

## Suggested Commit Message

```text
Add unified SDE fingerprints and training data pipeline

- add multi-initial-condition, active Kramers-Moyal, and Gaussian weak-kernel
  fingerprint construction for 1D SDEs
- add SDE simulator helpers and GenSR-compatible dataset generation
- inject generated SDE samples into the GenSR CVAE training loop
- support configurable SDE dataset paths through GENSR_SDE_DATASET
- add Mac CPU compatibility handling for local training/evaluation probes
- avoid forced early periodic checkpoints when --save_periodic 0 is used
- document the first 100-step SDE fingerprint training probe
```

## 500-Step Follow-Up Probe

After fixing checkpoint saving, a longer no-checkpoint loss probe was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 train.py \
  --min_input_dimension 1 --max_input_dimension 1 \
  --min_output_dimension 1 --max_output_dimension 1 \
  --num_workers 0 --cpu True \
  --dump_path /private/tmp/gensr_sde_500step_probe \
  --print_freq 25 --n_steps_per_epoch 500 --max_epoch 1 \
  --batch_size 8 --batch_size_eval 8 \
  --beam_eval False --save_periodic 0 --wandb_disabled
```

Run directory:

- `/private/tmp/gensr_sde_500step_probe/debug/cg5i6k34cq`

The run completed 500 steps in about 47 minutes and 34 seconds. No periodic
checkpoint was written; the run directory was only about 20 KB.

| Window | VAE total | Prior pred | Post pred | KL | KL weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 25-100 | 17.852 | 8.247 | 8.247 | 1.358 | 0.049 |
| Step 125-250 | 13.904 | 6.187 | 6.191 | 1.527 | 0.149 |
| Step 275-400 | 7.218 | 3.347 | 3.347 | 0.524 | 0.200 |
| Step 425-500 | 5.408 | 2.585 | 2.585 | 0.238 | 0.200 |
| KL fixed, step 250-500 | 6.966 | 3.247 | 3.248 | 0.471 | 0.200 |

Final printed step:

```text
step 500
VAE-TOTAL      4.8910
VAE-PRED-PRIOR 2.3237
VAE-PRED-POST  2.3239
VAE-KL         0.2434
VAE-KL-WEIGHT  0.2000
```

Best printed prior prediction loss:

```text
step 500
VAE-PRED-PRIOR 2.3237
VAE-PRED-POST  2.3239
```

Worst printed prior prediction loss:

```text
step 25
VAE-PRED-PRIOR 9.1941
VAE-PRED-POST  9.1949
```

### Interpretation

The longer probe confirms that the SDE fingerprint pipeline is learnable by the
current GenSR CVAE training loop. Prediction loss decreased from about 8.25 in
the first printed window to about 2.59 in the final window, and the prior and
posterior prediction losses stayed nearly identical.

This is strong evidence for training-set fit, not yet evidence for symbolic
generalization. The next necessary experiment is a held-out SDE validation set
with token top-k accuracy, same-template relaxed accuracy, and
generator-semantic accuracy.
