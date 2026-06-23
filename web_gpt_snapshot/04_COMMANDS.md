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

## Targeted Residual-vector Diagnostic

This is not a formal 32-sample eval. It reproduces only known role-wise miss
samples and scores selected/oracle role swaps.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_rolewise_residuals.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --samples 13,19,24,26 \
  --report rolewise_residual_vector_diagnostics.md \
  --json-output rolewise_residual_vector_diagnostics.json \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  > /private/tmp/gensr_sde_rolewise_residual_vectors.log 2>&1
```

Compile check for the helper:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_rolewise_residuals.py
```

## Offline Residual Score Ablation

This is not a formal eval and does not add a rerank mode.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_residual_score_ablation.py \
  --input-log /private/tmp/gensr_sde_rolewise_residual_vectors.log \
  --report rolewise_residual_score_ablation.md \
  > /private/tmp/gensr_sde_residual_score_ablation.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_rolewise_residuals.py \
  scripts/analyze_residual_score_ablation.py
```

## Candidate Coverage Diagnostic

This reruns candidate generation only for the 32 held-out samples. It is not a
formal rerank eval and does not compute fingerprint scores.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_diagnostics.md \
  --json-output candidate_coverage_diagnostics.json \
  > /private/tmp/gensr_sde_candidate_coverage_json.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_candidate_coverage.py
```

## Drift / Pairing Coverage Diagnostic

This uses `candidate_coverage_diagnostics.json` when available. If the JSON
sidecar is missing, it can still write a partial report from
`candidate_coverage_diagnostics.md`, but exact span-rank analysis will be
blocked.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_pairing_coverage.py \
  --source-log /private/tmp/gensr_sde_candidate_coverage_json.log \
  --coverage-json candidate_coverage_diagnostics.json \
  --report drift_pairing_coverage_diagnostics.md \
  > /private/tmp/gensr_sde_drift_pairing_coverage_full.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_candidate_coverage.py \
  scripts/analyze_drift_pairing_coverage.py
```

## Expanded Pairing Coverage Diagnostic

This is candidate-generation coverage only. It does not run formal rerank
scoring.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_pair_expanded.md \
  --json-output candidate_coverage_pair_expanded.json \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --pair-drift-diffusion-candidates 32 \
  > /private/tmp/gensr_sde_candidate_coverage_pair_expanded.log 2>&1
```

Latest expanded-pairing coverage:

```text
full_oracle_present=15/32
oracle_absent=17/32
rescued pairing samples=5,8,11,23,30,31
```

## Expanded Pairing Oracle-Absent Drift Diversity Diagnostic

This parses existing coverage JSON only. It does not run model decoding,
candidate generation/regeneration, fingerprint simulation, formal eval,
64-sample eval, grids, retraining, scorer changes, or candidate-generation
changes.

```bash
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py \
  --coverage-json candidate_coverage_pair_expanded.json \
  --baseline-json candidate_coverage_diagnostics.json \
  --report expanded_pairing_oracle_absent_drift_diversity.md \
  --json-output expanded_pairing_oracle_absent_drift_diversity.json
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py
```

Latest result:

```text
oracle_absent=17/32
exact_drift_missing=17/17
pairing_missing=0/17
diffusion_missing=0/17
exact_diffusion_present=17/17
decision=A
```

## Drift-Only Candidate Diversity Smoke

This is diagnostic-only and targets the 17 expanded-pairing oracle-absent
samples. It loads the current checkpoint and generates drift-only constrained
beam/sampling candidates. It does not run reranking, fingerprint scoring,
formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode
changes, checkpoint/data changes, target-format changes, fingerprint changes,
or production candidate-generation default changes.

Standalone command:

```bash
scripts/run_drift_only_candidate_diversity_smoke.sh
```

Equivalent explicit command:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
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
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  sde_validation_probe.py \
  scripts/analyze_drift_only_candidate_diversity.py
```

Latest result:

```text
target_samples=17
exact_truth_drift_found=3/17
found_by_source: beam=3, sampling=0, both=0, neither=14
exact_hit_samples=2,25,29
exact_hit_rank=30 for all three
recovered_family: nested-mul linear=3/3
missing_family: linear=6/6, sin=6/6, polynomial-like=2/2
exact_diffusion_present=17/17
decision=A
```

## Drift-Only Admission + Constant-Folding Diagnostic

This parses existing JSON only. It does not run model decoding, candidate
generation, fingerprint simulation, reranking, formal eval, 64-sample eval,
grids, retraining, scorer changes, rerank-mode changes, checkpoint/data
changes, target-format changes, fingerprint changes, or production
candidate-generation default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_only_admission_and_constant_folding.py \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --previous-diversity-json expanded_pairing_oracle_absent_drift_diversity.json \
  --report drift_only_admission_constant_folding_diagnostics.md \
  --json-output drift_only_admission_constant_folding_diagnostics.json \
  > /private/tmp/gensr_sde_drift_only_admission_constant_folding.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_drift_only_admission_and_constant_folding.py
```

Latest result:

```text
current_expanded_full_oracle=15/32
exact_tail_projected_gain=+3
exact_tail_projected_coverage=18/32
canonicalized_drift_matches=15/17
exact_missing_cases_become_canonical_matches=12
canonicalized_projected_gain=+15
canonicalized_projected_coverage=30/32
linear_sin_canonical_matches=12/12
polynomial_like_missing_after_canonical=2/2
decision=B
```

Do not run formal eval from this diagnostic. Next step should be a helper-only
or small-smoke candidate-normalization diagnostic for constant-chain folding
before widening beam/top-k.

## Candidate-Normalization Constant-Folding Diagnostic

This parses existing JSON only and applies narrow multiplicative
constant-chain folding at candidate-normalization / pairing diagnostic level.
It does not run model decoding, candidate generation, fingerprint simulation,
reranking, formal eval, 64-sample eval, grids, retraining, scorer changes,
rerank-mode changes, checkpoint/data changes, target-format changes,
fingerprint changes, or production candidate-generation default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_normalization_constant_folding.py \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --constant-folding-json drift_only_admission_constant_folding_diagnostics.json \
  --report candidate_normalization_constant_folding_diagnostics.md \
  --json-output candidate_normalization_constant_folding_diagnostics.json \
  > /private/tmp/gensr_sde_candidate_normalization_constant_folding.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_candidate_normalization_constant_folding.py
```

Latest result:

```text
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
exact_tail_admission_coverage=18/32
canonicalized_drift_only_admission_coverage=30/32
current_expanded_pool_new_samples=1,3,4,9,10,14,20,22
exact_tail_recovered_samples=2,25,29
canonicalized_drift_only_recovered_samples=0,1,2,3,4,6,7,9,10,14,15,20,22,25,29
polynomial_like_missing_after_normalization=16,28
collision_groups=9
unsafe_collision_flag=False
serialized_raw_to_canonical=440 -> 278
serialized_reduction=36.8%
decision=B
```

Do not run formal eval from this diagnostic. Next step should be one small
coverage-only normalized drift-only admission diagnostic before widening
beam/top-k.

## Normalized Drift-Only Admission Coverage Diagnostic

This parses existing JSON only and simulates the diagnostic candidate pool:

```text
current expanded-pairing candidate pool
+ normalized drift-only tail candidates for the 17 target samples
```

It does not run model decoding, candidate generation, fingerprint simulation,
reranking, formal eval, 64-sample eval, grids, retraining, scorer changes,
rerank-mode changes, checkpoint/data changes, target-format changes,
fingerprint changes, or production candidate-generation default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_normalized_drift_only_admission_coverage.py \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --normalization-json candidate_normalization_constant_folding_diagnostics.json \
  --report normalized_drift_only_admission_coverage_diagnostics.md \
  --json-output normalized_drift_only_admission_coverage_diagnostics.json \
  > /private/tmp/gensr_sde_normalized_drift_only_admission_coverage.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_normalized_drift_only_admission_coverage.py
```

Latest result:

```text
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
exact_tail_admission_coverage=18/32
normalized_drift_only_admission_coverage=30/32
newly_recovered_beyond_current_expanded_canonical_pool=0,2,6,7,15,25,29
remaining_missing_samples=16,28
drift_only_tail_normalized_sources: beam=15
pair_count_before_raw_sum=370
pair_count_after_canonical_sum=1105
pair_count_canonical_increase=+735 (+198.6%)
unsafe_collision_flag=False
decision=B
```

Do not run formal eval from this diagnostic. Next step should be a capped
normalized drift-only admission coverage diagnostic.

## Expanded Pairing Formal Eval Result

The formal eval log is:

```text
/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log
```

Result:

```text
selected exact/relaxed=3/32
oracle exact/relaxed=15/32
pair oracle exact/relaxed=8/32
rerank_candidates_valid=715
```

Do not make this expanded pairing setting the default yet, and do not run
64-sample expansion from it. Next use the existing log for oracle-miss ranking
diagnostics.

## Expanded Pairing Oracle-Miss Ranking Diagnostic

This parses existing logs only; it does not run the model or regenerate
candidates.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_expanded_pairing_misses.py \
  --source-log /private/tmp/gensr_sde_32_expanded_pairing_rolewise.log \
  --report expanded_pairing_oracle_miss_ranking_diagnostics.md \
  --json-output expanded_pairing_oracle_miss_ranking_diagnostics.json \
  > /private/tmp/gensr_sde_expanded_pairing_miss_ranking.log 2>&1
```

Latest result:

```text
selected misses analyzed=12
oracle-only-pair misses=7
near score gaps <=0.10=2/12
oracle-only-pair near gaps <=0.10=1/7
selected source is pair=8/12
```

## Expanded Pairing Wrong-Pair Pruning Diagnostic

This parses existing JSON/log-derived reports only.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_expanded_pair_pruning.py \
  --miss-json expanded_pairing_oracle_miss_ranking_diagnostics.json \
  --coverage-json candidate_coverage_pair_expanded.json \
  --report expanded_pairing_wrong_pair_pruning_diagnostics.md \
  --json-output expanded_pairing_wrong_pair_pruning_diagnostics.json \
  > /private/tmp/gensr_sde_expanded_pair_pruning.log 2>&1
```

Latest result:

```text
wrong selected pair cases=8
pair-oracle source ranks=2..15
approx wrong-pair source ranks=1..14
safe_pruning_rule_found=0
```

## Expanded Pairing Active-Residual Trap Diagnostic

This recomputes residual vectors for existing expanded-pairing miss cases only.
It does not run model decoding, candidate regeneration, formal eval, or a grid.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_expanded_pair_active_residuals.py \
  --miss-json expanded_pairing_oracle_miss_ranking_diagnostics.json \
  --pruning-json expanded_pairing_wrong_pair_pruning_diagnostics.json \
  --coverage-json candidate_coverage_pair_expanded.json \
  --report expanded_pairing_active_residual_trap_diagnostics.md \
  --json-output expanded_pairing_active_residual_trap_diagnostics.json \
  > /private/tmp/gensr_sde_expanded_pair_active_residuals.log 2>&1
```

Latest result:

```text
selected misses analyzed=12
wrong selected pair cases=8
oracle-only-pair misses=7
active favors selected=12/12
weak favors oracle while active favors selected=4/12
outlier-dominated active traps=7/12
KM1/drift-leaning active advantages=10/12
```

## Expanded Pairing Active-Residual Ablation

This is offline analysis over existing residual/candidate logs. It does not run
formal eval, model decoding, candidate regeneration, or a grid.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_expanded_pair_active_ablation.py \
  --residual-json expanded_pairing_active_residual_trap_diagnostics.json \
  --miss-json expanded_pairing_oracle_miss_ranking_diagnostics.json \
  --pruning-json expanded_pairing_wrong_pair_pruning_diagnostics.json \
  --coverage-json candidate_coverage_pair_expanded.json \
  --source-log /private/tmp/gensr_sde_32_expanded_pairing_rolewise.log \
  --report expanded_pairing_active_residual_ablation.md \
  --json-output expanded_pairing_active_residual_ablation.json \
  > /private/tmp/gensr_sde_expanded_pair_active_ablation.log 2>&1
```

Latest result:

```text
best offline variant=per_active_dim_norm_plus_weak
oracle wins=6/12
oracle-only-pair wins=4/7
wrong selected pair wins=5/8
debug top2 hit harms=0/3 for best variant
decision=C
```

## Residual Debug Logging Smoke

This is an 8- or 16-sample eval-only smoke for candidate-level residual JSON
logging. It is not a formal eval and does not add a scorer mode.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --eval-only \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
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
  --pair-drift-diffusion-candidates 8 \
  --pair-drift-topk 4 \
  --pair-diffusion-topk 6 \
  --rerank-debug-topk 2 \
  --rerank-residual-debug-json /private/tmp/gensr_sde_residual_debug_smoke16.json \
  > /private/tmp/gensr_sde_residual_debug_smoke16.log 2>&1
```

Offline safety parser:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_residual_debug_safety.py \
  --input-json /private/tmp/gensr_sde_residual_debug_smoke16.json \
  --report residual_debug_safety_smoke16.md \
  --json-output residual_debug_safety_smoke16.json \
  > /private/tmp/gensr_sde_residual_debug_safety_smoke16.log 2>&1
```

Latest 16-sample result:

```text
samples logged=16
candidates logged=201
samples with oracle=2
selected hits=1
pair oracle samples=1
selected-hit harms=0
oracle-present selected-miss rescues=0
result=limited no-harm evidence, no formal eval justified
```

## Expanded Pairing Residual Debug Smoke

Local script:

```bash
bash scripts/run_expanded_pairing_residual_debug_smoke16.sh
```

The script uses:

```text
pair_drift_diffusion_candidates=32
pair_drift_topk=5
pair_diffusion_topk=6
rerank_residual_debug_json=/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.json
safety_report=residual_debug_safety_expanded_pairing_smoke16.md
```

User-run result:

```text
candidates logged=350
samples with oracle=5/16
pair-oracle samples=4/16
selected hits=0/16
oracle-present selected misses=5/16
parse failures=0
fingerprint failures=0
active residual vector length=18
weak residual vector length=96
offline per_active_dim_norm_plus_weak selected hits=0
offline selected-hit harms=0
offline selected-miss rescues=0/5
offline selected pair candidates=8
```

Do not add `per_active_dim_norm_plus_weak` as a rerank mode, and do not run a
formal 32-sample eval for this scorer idea. Next return to drift span diversity
or deeper fingerprint ambiguity diagnostics.

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
  --rerank-constant-values 0.25,0.5,1.0,2.0,4.0 \
  --rerank-debug-topk 2 \
  --sample-candidates 8 \
  --sample-temperatures 0.8,1.0,1.2 \
  --sample-top-k 8 \
  --sample-top-p 0.95 \
  --pair-drift-diffusion-candidates 8 \
  --pair-drift-topk 4 \
  --pair-diffusion-topk 6
```

For constant-sensitivity diagnostics, change the rerank mode to:

```bash
--rerank-score constant_grid_componentwise \
--rerank-component-weights 1.0,2.0,1.0 \
--rerank-constant-values 0.25,0.5,1.0,2.0,4.0
```

Other lightweight score modes now available:

```bash
--rerank-score constant_grid_componentwise_no_multi_u0
--rerank-score constant_grid_rolewise_no_multi_u0
--rerank-score constant_grid_active_weak
--rerank-score constant_grid_active_only
--rerank-score constant_grid_moments_downweighted
```

Current strongest 32-sample validation setting:

```bash
--rerank-score constant_grid_rolewise_no_multi_u0 \
--rerank-constant-values 0.25,0.5,1.0,2.0,4.0 \
--rerank-tie-epsilon 0.005 \
--rerank-tie-break none
```

Shared-constant baseline for comparison:

```bash
--rerank-score constant_grid_componentwise_no_multi_u0
```

Tie-break variants to compare, not defaults:

```bash
--rerank-tie-break active_distance
--rerank-tie-break state_dependent_drift
```

Latest 32-sample eval-only result: shared-constant no-tie baseline selected
exact/relaxed `4/32`; role-wise constants with active:weak `1:1` improved this
to `5/32`, with oracle exact/relaxed still `9/32`. Active-heavy `2:1` and
weak-downweighted `1:0.5` role-wise variants regressed to `3/32`. Do not run
64-sample expansion unless a setting first improves selected metrics over the
32-sample role-wise baseline.

## Regenerate 2000-Step Checkpoint

Preferred checkpoint paths:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
checkpoints/gensr_sde_role_token_2000/role_token_2000.pth
```

`checkpoints/` is ignored by git. Do not commit model weights. If the
persistent checkpoint exists but `/private/tmp` is missing, recreate the tmp
path with a symlink:

```bash
mkdir -p /private/tmp/gensr_sde_role_token_2000
ln -sf /Users/lzhang/Documents/GenSR_SDE/checkpoints/gensr_sde_role_token_2000/role_token_2000.pth \
  /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

If both checkpoint paths are missing, regenerate only when explicitly requested:

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
