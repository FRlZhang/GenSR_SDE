# Common Commands

## Python / Conda

Recent local work used:

```bash
/opt/miniconda3/envs/gensr/bin/python3
```

Useful macOS temporary environment variables:

```bash
PYTHONDONTWRITEBYTECODE=1
MPLCONFIGDIR=/private/tmp/mpl
XDG_CACHE_HOME=/private/tmp/cache
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache
```

## Compile Check

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  sde_validation_probe.py \
  sde_dataset_generator.py \
  sde_fingerprint.py \
  simulator_sde.py \
  train.py
```

## Whitespace Check

```bash
git diff --check
```

## Small Smoke Test

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 2 \
  --eval-samples 2 \
  --batch-size 1 \
  --print-freq 1 \
  --n-paths 50 \
  --active-paths 50 \
  --n-steps 10 \
  --max-generated-len 24 \
  --min-generated-len 4 \
  --n-enc-layers 1 \
  --n-dec-layers 1 \
  --n-heads 4 \
  --model-dim 64 \
  --save-checkpoint /private/tmp/gensr_sde_git_clean_smoke/smoke.pth
```

## Eval-Only Checkpoint Decoding

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
  --rerank-score componentwise \
  --rerank-component-weights 1.0,2.0,1.0 \
  --rerank-debug-topk 2 \
  --sample-candidates 8 \
  --sample-temperatures 0.8,1.0,1.2 \
  --sample-top-k 8 \
  --sample-top-p 0.95 \
  --pair-drift-diffusion-candidates 8 \
  --pair-drift-topk 4 \
  --pair-diffusion-topk 6
```

## Regenerate 2000-Step Checkpoint

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
