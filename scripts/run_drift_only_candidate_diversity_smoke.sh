#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/mpl \
XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_only_candidate_diversity.py \
  --drift-diversity-json expanded_pairing_oracle_absent_drift_diversity.json \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
  --checkpoint-backup checkpoints/gensr_sde_role_token_2000/role_token_2000.pth \
  --report drift_only_candidate_diversity_smoke.md \
  --json-output drift_only_candidate_diversity_smoke.json \
  --batch-size 2 \
  --drift-max-len 16 \
  --drift-beam-size 16 \
  --drift-beam-candidates 32 \
  --drift-sample-candidates 48 \
  --sample-temperatures 0.8,1.0,1.2 \
  --sample-top-k 8 \
  --sample-top-p 0.95 \
  > /private/tmp/gensr_sde_drift_only_candidate_diversity_smoke.log 2>&1
