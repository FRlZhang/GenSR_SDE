# GenSR-SDE Training Report

Date: 2026-06-11

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
