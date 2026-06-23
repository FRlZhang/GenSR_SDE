#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/opt/miniconda3/envs/gensr/bin/python3
CHECKPOINT=/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
SMOKE_JSON=/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.json
SMOKE_LOG=/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.log
SAFETY_LOG=/private/tmp/gensr_sde_residual_debug_safety_expanded_pairing_smoke16.log

mkdir -p /private/tmp/mpl /private/tmp/cache

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
"${PYTHON_BIN}" sde_validation_probe.py \
  --eval-only \
  --load-checkpoint "${CHECKPOINT}" \
  --eval-samples 16 \
  --batch-size 8 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8 \
  --constrained-beam-size 8 \
  --rerank-candidates 8 \
  --rerank-topk-from-beam 8 \
  --rerank-score constant_grid_rolewise_no_multi_u0 \
  --rerank-constant-values 0.25,0.5,1.0,2.0,4.0 \
  --rerank-tie-epsilon 0.005 \
  --rerank-tie-break none \
  --sample-candidates 8 \
  --sample-temperatures 0.8,1.0,1.2 \
  --sample-top-k 8 \
  --sample-top-p 0.95 \
  --pair-drift-diffusion-candidates 32 \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --rerank-debug-topk 2 \
  --rerank-residual-debug-json "${SMOKE_JSON}" \
  > "${SMOKE_LOG}" 2>&1

PYTHONDONTWRITEBYTECODE=1 \
"${PYTHON_BIN}" scripts/analyze_residual_debug_safety.py \
  --input-json "${SMOKE_JSON}" \
  --report residual_debug_safety_expanded_pairing_smoke16.md \
  --json-output residual_debug_safety_expanded_pairing_smoke16.json \
  > "${SAFETY_LOG}" 2>&1

echo "smoke_log=${SMOKE_LOG}"
echo "smoke_json=${SMOKE_JSON}"
echo "safety_log=${SAFETY_LOG}"
echo "safety_report=residual_debug_safety_expanded_pairing_smoke16.md"
echo "safety_json=residual_debug_safety_expanded_pairing_smoke16.json"
