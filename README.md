<div align="center">
  <h2><b> (ICLR'26) GenSR: Symbolic Regression Based on Equation Generative Space </b></h2>
</div>

The official implementation of our ICLR-2026 paper "**GenSR: Symbolic Regression Based on Equation Generative Space**" [[OpenReview](https://openreview.net/forum?id=8emIjwUQZg)] [[WandB](https://wandb.ai/yuxiao-hu-the-hong-kong-polytechnic-university/ICLR26-GenSR)].

```
@inproceedings{
li2026gensr,
title={Gen{SR}: Symbolic regression based on equation generative space},
author={Qian Li and Yuxiao Hu and Juncheng Liu and Yuntian Chen},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=8emIjwUQZg}
}
```

## Introduction

<p align="center">
<img src="./framework.png" height = "360" alt="" align=center />
</p>

GenSR constructs an **equation generative space** via a Conditional Variational Autoencoder (CVAE), where each point in the latent space maps to a symbolic equation. At inference time, a **degraded CMA-ES** searches this space to find equations that best fit the observed data, with constants refined via BFGS. Specifically, GenSR first pretrains a dual-branch Conditional Variational Autoencoder (CVAE) to reparameterize symbolic equations into a generative latent space with symbolic continuity and local numerical smoothness. At inference, the CVAE coarsely localizes the input data to promising regions in the latent space. Then, a modified CMA-ES refines the candi- date region, leveraging smooth latent gradients.

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Pretrained Weights

Download the checkpoint from [Google Drive](https://drive.google.com/file/d/1TbcRSzO3rGQBuJIPN5P6fQpYYEv4__K4/view?usp=sharing) and place it in `weights/`:

```
weights/checkpoint.pth
```

### Training

```bash
bash scripts/train.sh
```

### Evaluation

```bash
bash scripts/eval.sh
```

Change `DATA_TYPE` in the script to evaluate on different SRBench datasets: `feynman`, `strogatz`, or `blackbox`.

## Current Fork Status

This repository is currently being extended from algebraic symbolic regression
to autonomous 1D SDE symbolic discovery. The original paper README above is
still valid for the upstream GenSR project. The sections below describe the
current local SDE workflow and should be treated as the main recovery entry for
new Codex threads or other agents.

## Overview

The active research direction is:

- keep the GenSR CVAE / latent-search framework;
- replace algebraic function-value fingerprints with SDE fingerprints;
- predict symbolic sequences in the form
  `<DRIFT> drift_tokens <DIFFUSION> diffusion_tokens`;
- improve sequence-level recovery with constrained candidate generation and SDE
  fingerprint-distance reranking.

Current best result so far:

- a 2000-step role-token probe reached `token_top1=0.820002`,
  `token_top3=0.960833`, `token_top5=1.000000`;
- greedy exact recovery is still low (`1/64`), so decoding remains the main
  bottleneck.
- first-pass fingerprint reranking is implemented, but a small checkpoint eval
  did not improve exact / relaxed recovery yet.
- drift/diffusion pairing produced nonzero oracle hits and, with constant-grid
  scoring, selected reranked recovery now reaches `4/32` on a 32-sample
  eval-only checkpoint probe.
- componentwise reranking has been added for diagnostics; it also missed the
  oracle candidate in the first 16-sample probe.
- global constant-grid reranking improved the inspected oracle rank in the
  original 16-sample diagnostic but still did not select it there, so constant
  sensitivity is only part of the bottleneck.
- dropping `multi_u0_moments` from the constant-grid score nearly selected the
  inspected oracle candidate in the 16-sample diagnostic, but no-tie selected
  exact / relaxed recovery was still 0 there.
- epsilon tie-breaking on active distance or state-dependent drift produced the
  first selected reranked exact / relaxed hit (`1/16`) in a checkpoint probe.
- on a larger 32-sample eval-only validation, baseline no tie,
  `active_distance`, and `state_dependent_drift` all matched at selected
  exact/relaxed `4/32` while oracle exact/relaxed remained `9/32`, so the
  near-tie heuristics are not yet stable improvements over baseline.
- oracle-miss diagnostics show the 5 selected misses are score-selection
  failures, not parse/fingerprint failures: 2 keep the oracle diffusion but
  choose the wrong drift, 1 keeps the oracle drift but chooses the wrong
  diffusion, and 2 miss both sides. The immediate next target is rerank score
  calibration / constant handling; candidate generation remains the larger
  ceiling because 23/32 samples still have no oracle candidate.
- role-wise constant-grid scoring, with separate drift/diffusion constants,
  rescues one selected miss and improves selected exact / relaxed recovery to
  `5/32`; active-heavy and weak-downweighted role-wise variants regressed to
  `3/32`.
- expanded pairing coverage with `pair_drift_topk=5` and
  `pair_drift_diffusion_candidates=32` raises offline full-oracle candidate
  coverage from `9/32` to `15/32`;
- the matching formal 32-sample eval reproduced oracle exact / relaxed
  `15/32` and pair oracle `8/32`, but selected exact / relaxed dropped to
  `3/32`, so expanded pairing is diagnostic-only for now and should not become
  the default before paired-oracle ranking is understood.
- expanded-pairing miss diagnostics show only `2/12` selected misses are
  near-tie losses and only `1/7` oracle-only-pair misses is near-tie; wrong
  pair candidates are selected often enough that pruning/ranking diagnostics are
  safer than a blanket pair bonus.
- wrong-pair pruning diagnostics did not find an oracle-preserving pruning rule:
  pair-oracle source ranks extend to 15 and harmful pair structures overlap
  with true pair-oracle families, so the next step is active-kramers-moyal
  residual analysis rather than pruning or a pair bonus.
- targeted active-residual diagnostics show active distance favors selected
  non-oracles in all 12 expanded-pairing misses; 7/12 are outlier-dominated and
  10/12 lean KM1/drift-like. This supports only an offline robust/clipped
  active-residual ablation on existing candidates, not a formal eval or new
  default rerank mode.

See [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md)
and [PROJECT_STATUS.md](/Users/lzhang/Documents/GenSR_SDE/PROJECT_STATUS.md) for
the current experiment state.

## Folder Structure

- `symbolicregression/`: upstream GenSR model, environment, trainer, and search code.
- `train.py`: local training entry; injects SDE samples into `EnvDataset.generate_sample`.
- `sde_fingerprint.py`: unified SDE fingerprint construction.
- `simulator_sde.py`: 1D autonomous SDE sampling and trajectory simulation helpers.
- `sde_dataset_generator.py`: builds GenSR-compatible SDE dataset pickles.
- `sde_validation_probe.py`: main train/eval probe for SDE teacher-forced and decoding diagnostics.
- `fingerprint_identifiability_test.py`: standalone fingerprint separability tests.
- `data/sde_train_dataset.pkl`: current in-repo SDE training dataset.
- `dump/debug/*`: historical debug / experiment outputs.
- `SDE_TRAINING_REPORT.md`: experiment log with validated SDE results.

## How To Run

### Environment

Upstream provides `requirements.txt` and `environment.yml`. The local verified
setup for recent SDE experiments used the existing conda environment:

```bash
conda activate gensr
```

and, when needed explicitly:

```bash
/opt/miniconda3/envs/gensr/bin/python3
```

Note: `environment.yml` is Linux-oriented from the upstream repo. On macOS, the
existing `gensr` environment has been used successfully for the SDE work.

### Generate / refresh the SDE dataset

```bash
/opt/miniconda3/envs/gensr/bin/python3 sde_dataset_generator.py \
  --num-samples 1000 \
  --output data/sde_train_dataset.pkl \
  --seed 20260610 \
  --n-paths 2000 \
  --active-paths 2000 \
  --n-steps 80
```

### Run the current SDE validation probe

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 500 \
  --eval-samples 64 \
  --batch-size 8 \
  --print-freq 100 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8
```

### Train a longer checkpointed probe

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 2000 \
  --eval-samples 64 \
  --batch-size 8 \
  --print-freq 250 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8 \
  --save-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

### Reuse a checkpoint for eval-only decoding tests

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --eval-only \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
  --eval-samples 32 \
  --batch-size 8 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8 \
  --constrained-beam-size 8 \
  --rerank-candidates 8 \
  --rerank-topk-from-beam 8 \
  --rerank-score full_fingerprint \
  --rerank-debug-topk 2 \
  --sample-candidates 8 \
  --sample-temperatures 0.8,1.0,1.2 \
  --sample-top-k 8 \
  --sample-top-p 0.95 \
  --pair-drift-diffusion-candidates 8 \
  --pair-drift-topk 4 \
  --pair-diffusion-topk 6
```

### Full training entry

`train.py` now reads SDE data from:

- `GENSR_SDE_DATASET` if that environment variable is set;
- otherwise `data/sde_train_dataset.pkl`.

It converts legacy `SPECIAL`-separated samples into the newer
`<DRIFT> ... <DIFFUSION> ...` token format on load.

## How To Test / Verify

- `PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache /opt/miniconda3/envs/gensr/bin/python3 -m py_compile sde_validation_probe.py`
- `git diff --check`
- run a small checkpoint smoke test with:
  `sde_validation_probe.py --train-steps 2 --eval-samples 2 --save-checkpoint ...`
- for meaningful model checks, rely on `sde_validation_probe.py` metrics rather
  than training loss alone.

Recommended verification order after code changes that affect SDE training:

1. run `py_compile`;
2. run a 2-step or 5-step probe smoke test;
3. run a held-out validation probe;
4. if decoding changed, compare `greedy_*` and `constrained_beam_*` metrics.

## Important Files

- [README.md](/Users/lzhang/Documents/GenSR_SDE/README.md): stable project description and commands.
- [AGENTS.md](/Users/lzhang/Documents/GenSR_SDE/AGENTS.md): stable agent workflow and safety rules.
- [PROJECT_STATUS.md](/Users/lzhang/Documents/GenSR_SDE/PROJECT_STATUS.md): current goal, decisions, blockers, and recovery steps.
- [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md): validated experiment history.
- [sde_validation_probe.py](/Users/lzhang/Documents/GenSR_SDE/sde_validation_probe.py): current main evaluation entrypoint.
- [sde_dataset_generator.py](/Users/lzhang/Documents/GenSR_SDE/sde_dataset_generator.py): dataset generation and token encoding.
- [sde_fingerprint.py](/Users/lzhang/Documents/GenSR_SDE/sde_fingerprint.py): fingerprint definition.
- [train.py](/Users/lzhang/Documents/GenSR_SDE/train.py): full training path with SDE sample injection.

## Deprecated / Disfavored Directions

- Treating the old 100-step / 500-step loss-only probes from before the
  `EnvDataset.generate_sample` fix as valid SDE evidence.
- Using two fully independent drift / diffusion decoders as the default path.
- Focusing first on changing the fingerprint again before improving candidate
  diversity and reranking scores.

## Acknowledge

We appreciate the following repos for their valuable code:

[[Multimodal-Math-Pretraining](https://github.com/deep-symbolic-mathematics/Multimodal-Math-Pretraining)] [[End-to-end Symbolic Regression](https://github.com/facebookresearch/symbolicregression)]


## License

This repository is licensed under the MIT License.
