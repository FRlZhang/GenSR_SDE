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

## Capped Normalized Drift Admission Diagnostic

This parses existing JSON only and compares fixed capped normalized drift-only
admission policies. It does not run model decoding, candidate generation,
fingerprint simulation, reranking, formal eval, 64-sample eval, grids,
retraining, scorer changes, rerank-mode changes, checkpoint/data changes,
target-format changes, fingerprint changes, or production candidate-generation
default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_capped_normalized_drift_admission.py \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --normalized-admission-json normalized_drift_only_admission_coverage_diagnostics.json \
  --report capped_normalized_drift_admission_diagnostics.md \
  --json-output capped_normalized_drift_admission_diagnostics.json \
  > /private/tmp/gensr_sde_capped_normalized_drift_admission.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_capped_normalized_drift_admission.py
```

Latest result:

```text
best_practical_policy=P2_one_per_family
baseline_current_expanded=15/32
baseline_current_expanded_canonical=23/32
baseline_full_normalized_admission=30/32
pre_admission_target_pair_count=370
full_normalized_admission_target_pair_count=1105
P2_one_per_family_coverage=30/32
P2_one_per_family_pair_count=340
P2_pair_reduction_vs_full=765 (69.2%)
truth_aware_upper_bound_coverage=30/32
truth_aware_upper_bound_pair_count=320
newly_recovered_beyond_current=0,2,6,7,15,25,29
remaining_missing=16,28
sampling_can_be_dropped=True
unsafe_collision_flag=False
decision=A
```

Do not run formal eval from this diagnostic. Next step should be a small
coverage-only implementation hook for `P2_one_per_family`.

## P2 Normalized Drift Admission Coverage Hook

This parses existing JSON only and applies the reusable P2 admission hook. It
does not run model decoding, candidate generation, fingerprint simulation,
reranking, formal eval, 64-sample eval, grids, retraining, scorer changes,
rerank-mode changes, checkpoint/data changes, target-format changes,
fingerprint changes, or production default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/run_p2_normalized_admission_coverage.py \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --report candidate_coverage_p2_normalized_admission.md \
  --json-output candidate_coverage_p2_normalized_admission.json \
  > /private/tmp/gensr_sde_p2_normalized_admission_coverage.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/normalized_drift_admission.py \
  scripts/run_p2_normalized_admission_coverage.py
```

Latest result:

```text
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
P2_normalized_admission_coverage=30/32
P2_pair_count=340
newly_recovered=0,2,6,7,15,25,29
remaining_missing=16,28
sampling_excluded=True
sampling_recovered_canonical_drift_count=0
self_check_failures=0
```

No formal eval is recommended from this hook alone.

## P2 Hook Cross-Validation

This parses existing JSON only and validates that the reusable P2 hook agrees
with previous helper diagnostics. It does not run model decoding, candidate
generation, fingerprint simulation, reranking, formal eval, 64-sample eval,
grids, retraining, scorer changes, rerank-mode changes, checkpoint/data
changes, target-format changes, fingerprint changes, or production defaults.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/validate_p2_normalized_admission_hook.py \
  --p2-json candidate_coverage_p2_normalized_admission.json \
  --capped-json capped_normalized_drift_admission_diagnostics.json \
  --normalized-admission-json normalized_drift_only_admission_coverage_diagnostics.json \
  --normalization-json candidate_normalization_constant_folding_diagnostics.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --report p2_normalized_admission_hook_validation.md \
  --json-output p2_normalized_admission_hook_validation.json \
  > /private/tmp/gensr_sde_p2_hook_validation.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/normalized_drift_admission.py \
  scripts/run_p2_normalized_admission_coverage.py \
  scripts/validate_p2_normalized_admission_hook.py
```

Latest result:

```text
json_reports_validated=6
metric_cross_check_status=ok
schema_status=ok
canonicalizer_safety_status=ok
mismatch_count=0
decision=A
```

No formal eval is recommended from this validation.

## Candidate Coverage P2 Flag

The candidate coverage pipeline now exposes a default-off coverage-only P2
diagnostic flag. Default behavior is unchanged when the flag is omitted.

Future small local coverage-only command:

Preflight requirement:

```text
/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log
```

Status: regenerated on 2026-06-24 and backed up under
`experiment_logs/2026-06-24_tmp_gensr_sde_logs/`.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_pair_expanded_p2_flag.md \
  --json-output candidate_coverage_pair_expanded_p2_flag.json \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --pair-drift-diffusion-candidates 32 \
  --normalized-drift-admission p2_one_per_family \
  --normalized-drift-admission-source beam \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  > /private/tmp/gensr_sde_candidate_coverage_pair_expanded_p2_flag.log 2>&1
```

JSON-only validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/validate_candidate_coverage_p2_flag.py \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --p2-json candidate_coverage_p2_normalized_admission.json \
  --hook-validation-json p2_normalized_admission_hook_validation.json \
  --report candidate_coverage_p2_flag_validation.md \
  --json-output candidate_coverage_p2_flag_validation.json \
  > /private/tmp/gensr_sde_candidate_coverage_p2_flag_validation.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_candidate_coverage.py \
  scripts/normalized_drift_admission.py \
  scripts/run_p2_normalized_admission_coverage.py \
  scripts/validate_p2_normalized_admission_hook.py \
  scripts/validate_candidate_coverage_p2_flag.py
```

Latest JSON-only validation result:

```text
metric_cross_check_status=ok
schema_status=ok
canonicalizer_safety_status=ok
mismatch_count=0
decision=A
```

Latest runtime-smoke result:

```text
status=completed
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
P2_normalized_admission_coverage=30/32
P2_pair_count=340
newly_recovered=0,2,6,7,15,25,29
remaining_missing=16,28
sampling_excluded=True
sampling_recovered_canonical_drift_count=0
unsafe_collision_flag=False
decision=A
```

No formal eval is recommended from this validation.

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

## P2 Admitted Candidate Scoring Readiness

This parses existing JSON only. It checks whether the P2 admitted candidates
can be scored offline with the current semantic scorer. The current result is a
clean readiness block because target `y_to_fit` / fingerprint vectors are not
stored in the coverage JSON.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  scripts/normalized_drift_admission.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py \
  sde_validation_probe.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_admitted_candidate_scoring.py \
  --p2-coverage-json candidate_coverage_pair_expanded_p2_flag.json \
  --expanded-coverage-json candidate_coverage_pair_expanded.json \
  --baseline-coverage-json candidate_coverage_diagnostics.json \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --target-samples 0,2,6,7,15,25,29 \
  --report p2_admitted_candidate_scoring_diagnostics.md \
  --json-output p2_admitted_candidate_scoring_diagnostics.json \
  > /private/tmp/gensr_sde_p2_admitted_candidate_scoring.log 2>&1
```

Latest result:

```text
target_samples_analyzed=7
p2_oracle_present_count=7
p2_oracle_source=beam for all 7
p2_oracle_source_rank=5 for all 7
exact_diffusion_present=7/7
p2_oracle_selected_count=unavailable
target_fingerprint_status=missing
semantic_scores_status=missing
decision=D
```

Do not run formal eval from this readiness report. Add minimal target
fingerprint / `y_to_fit` logging before P2 semantic scoring or eval integration.

## P2 Scorer-Ready Sidecar Logging

The candidate coverage pipeline now has default-off scorer-ready sidecar
logging. This is coverage-only logging; it does not run semantic scoring or
formal eval.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_pair_expanded_p2_flag.md \
  --json-output candidate_coverage_pair_expanded_p2_flag.json \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --pair-drift-diffusion-candidates 32 \
  --normalized-drift-admission p2_one_per_family \
  --normalized-drift-admission-source beam \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --scorer-ready-target-samples 0,2,6,7,15,25,29 \
  --scorer-ready-json p2_scorer_ready_candidates.json \
  > /private/tmp/gensr_sde_p2_scorer_ready_logging.log 2>&1
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/miniconda3/envs/gensr/bin/python3 scripts/validate_p2_scorer_ready_logging.py \
  --scorer-ready-json p2_scorer_ready_candidates.json \
  --readiness-json p2_admitted_candidate_scoring_diagnostics.json \
  --report p2_scorer_ready_logging_validation.md \
  --json-output p2_scorer_ready_logging_validation.json \
  > /private/tmp/gensr_sde_p2_scorer_ready_logging_validation.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_candidate_coverage.py \
  scripts/normalized_drift_admission.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  scripts/validate_p2_scorer_ready_logging.py
```

Latest result:

```text
target_samples=0,2,6,7,15,25,29
target_y_to_fit_present=7/7
target_fingerprint_shape=[186,1]
target_fingerprint_length=186
p2_oracle_rows=7
p2_oracle_present=7/7
exact_diffusion_present=7/7
p2_oracle_source=beam for all 7
p2_oracle_rank=5 for all 7
candidate_rows: current_expanded=140, p2_admitted=105
semantic_scoring_run=False
decision=A
```

Next step can be offline semantic scoring of P2 admitted candidates from
`p2_scorer_ready_candidates.json`; do not run formal eval from readiness alone.

## P2 Admitted Candidate Offline Semantic Scoring

This scores existing sidecar candidates only. It does not run model decoding,
candidate generation/regeneration, formal eval, 64-sample eval, retraining,
scorer grids, active/weak grids, checkpoint/data changes, target-format
changes, fingerprint schema changes, production default changes, or a new
rerank mode.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_admitted_candidate_scoring.py \
  --scorer-ready-json p2_scorer_ready_candidates.json \
  --target-samples 0,2,6,7,15,25,29 \
  --report p2_admitted_candidate_semantic_scoring.md \
  --json-output p2_admitted_candidate_semantic_scoring.json \
  > /private/tmp/gensr_sde_p2_admitted_candidate_semantic_scoring.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  scripts/normalized_drift_admission.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py \
  sde_validation_probe.py
```

Latest result:

```text
scorer=constant_grid_rolewise_no_multi_u0
active:weak=1:1
constant_values=0.25,0.5,1.0,2.0,4.0
candidate_rows_analyzed=245
valid_candidate_count=245
parse_failures=0
fingerprint_failures=0
p2_oracle_present=7/7
p2_oracle_selected=0/7
p2_oracle_score_misses=7/7
combined_selected_from_p2=3
combined_selected_from_current=4
median_p2_oracle_rank=4
max_p2_oracle_rank=32
near_gap_counts:
  <=0.005: 0
  <=0.01: 0
  <=0.05: 0
  <=0.10: 2
decision=B
```

Next step should be a targeted residual/segment diagnostic for the 7 P2 oracle
score misses. Do not run formal eval or integrate P2 admission into eval yet.

## P2 Oracle Score-Miss Residual Diagnostic

This recomputes selected-vs-P2-oracle active/weak residual vectors from
existing sidecar/scoring JSON only. It does not run model decoding, candidate
generation/regeneration, formal eval, 64-sample eval, retraining, scorer grids,
active/weak grids, checkpoint/data changes, target-format changes, fingerprint
schema changes, production default changes, or a new rerank mode.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_oracle_score_miss_residuals.py \
  --semantic-scoring-json p2_admitted_candidate_semantic_scoring.json \
  --scorer-ready-json p2_scorer_ready_candidates.json \
  --target-samples 0,2,6,7,15,25,29 \
  --report p2_oracle_score_miss_residual_diagnostics.md \
  --json-output p2_oracle_score_miss_residual_diagnostics.json \
  > /private/tmp/gensr_sde_p2_oracle_score_miss_residuals.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_oracle_score_miss_residuals.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
selected_source_counts: current_expanded=4, p2_admitted=3
dominant_gap_segment_counts: active=3, weak=1, near_tie=2, mixed=1
classification_counts: active_trap=3, weak_trap=1, near_tie=2, current_candidate_trap=1
near_gap_counts: <=0.005=0, <=0.01=0, <=0.05=0, <=0.10=2
top_active_dimensions_frequency: 76=5, 72=2, 82=2, 88=2
top_weak_dimensions_frequency: 136=3, 137=3, 133=2
constant_mismatch_count=4
decision=B
```

Do not integrate P2 admission into eval yet; residual behavior is still the
blocker.

## P2 Residual Calibration Counterfactual Diagnostic

This tests fixed offline counterfactuals on the 7 P2 oracle score misses. It
uses existing JSON only and does not run model decoding, candidate generation,
formal eval, 64-sample eval, retraining, scorer grids, active/weak grids,
checkpoint/data changes, target-format changes, fingerprint schema changes,
production default changes, or a new rerank mode.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_residual_calibration_counterfactuals.py \
  --residual-json p2_oracle_score_miss_residual_diagnostics.json \
  --semantic-scoring-json p2_admitted_candidate_semantic_scoring.json \
  --scorer-ready-json p2_scorer_ready_candidates.json \
  --target-samples 0,2,6,7,15,25,29 \
  --report p2_residual_calibration_counterfactuals.md \
  --json-output p2_residual_calibration_counterfactuals.json \
  > /private/tmp/gensr_sde_p2_residual_calibration_counterfactuals.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_residual_calibration_counterfactuals.py \
  scripts/analyze_p2_oracle_score_miss_residuals.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
samples_analyzed=7
baseline_oracle_wins=0
samples_flipped_by_any_variant=5
V11_shared_constants_only_best_of_grid=3/7
V1_weak_only_diagnostic=2/7
V3_active_without_dim_76=2/7
V4_active_without_dims_76_72_82_88=2/7
V6_active_and_weak_without_recurring_dims=2/7
V7_top1_active_advantage_removed_per_sample=2/7
V8_top2_active_advantages_removed_per_sample=2/7
V10_oracle_constants_applied_to_selected=2/7
samples_flipped_by_global_recurring_dims=2
samples_flipped_by_per_sample_top_dims=2
samples_flipped_by_constant_counterfactual=5
near_tie_samples_flipped=2
active_trap_samples_flipped=2
weak_trap_samples_flipped=0
decision=A
```

Use this only to design at most one future small smoke around shared-constant /
fixed residual calibration. Do not run formal eval or add a rerank mode from
this offline diagnostic alone.

## P2 Calibration All32 Smoke

This produces an all-32 scorer-ready sidecar with the existing coverage-only
candidate coverage helper, then scores existing sidecar candidates offline. It
does not run `sde_validation_probe.py`, formal eval, 64-sample eval, retraining,
scorer grids, active/weak grids, checkpoint/data changes, target-format
changes, fingerprint schema changes, production default changes, or a new
rerank mode.

Sidecar reconstruction:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \
  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \
  --report candidate_coverage_pair_expanded_p2_flag.md \
  --json-output candidate_coverage_pair_expanded_p2_flag.json \
  --pair-drift-topk 5 \
  --pair-diffusion-topk 6 \
  --pair-drift-diffusion-candidates 32 \
  --normalized-drift-admission p2_one_per_family \
  --normalized-drift-admission-source beam \
  --drift-only-json drift_only_candidate_diversity_smoke.json \
  --scorer-ready-target-samples 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  > /private/tmp/gensr_sde_p2_scorer_ready_all32.log 2>&1
```

Offline smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_calibration_all32_smoke.py \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  --p2-counterfactual-json p2_residual_calibration_counterfactuals.json \
  --baseline-coverage-json candidate_coverage_diagnostics.json \
  --p2-coverage-json candidate_coverage_pair_expanded_p2_flag.json \
  --target-samples 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --report p2_calibration_all32_smoke.md \
  --json-output p2_calibration_all32_smoke.json \
  > /private/tmp/gensr_sde_p2_calibration_all32_smoke.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_calibration_all32_smoke.py \
  scripts/analyze_p2_residual_calibration_counterfactuals.py \
  scripts/analyze_p2_oracle_score_miss_residuals.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  scripts/analyze_candidate_coverage.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
samples_analyzed=32
candidate_rows_analyzed=970
valid_candidate_count=970
parse_failures=0
fingerprint_failures=0
V0_selected_exact_relaxed=6/32
V11_selected_exact_relaxed=4/32
best_variant=V4_active_without_dims_76_72_82_88
best_variant_selected_exact_relaxed=9/32
best_variant_harms_vs_V0=2
selected_p2_oracle:
  V0=1
  V11=1
  V3=3
  V4=2
  V6=2
decision=B
```

Use this as scorer diagnostics only. Do not implement a rerank mode, integrate
P2 into eval, or run formal eval from this evidence.

## P2 Calibration Gate Feasibility

This parses existing JSON only and checks observable gates plus oracle upper
bounds for fixed residual calibration. It does not run model decoding,
candidate generation, fingerprint simulation, reranking, formal eval,
64-sample eval, grids, retraining, scorer changes, rerank-mode changes,
checkpoint/data changes, target-format changes, fingerprint changes, or
production default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_calibration_gate_feasibility.py \
  --all32-smoke-json p2_calibration_all32_smoke.json \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  --counterfactual-json p2_residual_calibration_counterfactuals.json \
  --target-samples 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --report p2_calibration_gate_feasibility.md \
  --json-output p2_calibration_gate_feasibility.json \
  > /private/tmp/gensr_sde_p2_calibration_gate_feasibility.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_calibration_gate_feasibility.py \
  scripts/analyze_p2_calibration_all32_smoke.py \
  scripts/analyze_p2_residual_calibration_counterfactuals.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
target_samples_analyzed=32
observable_gate_count=4
unavailable_gate_count=5
available_observable_gates=G1,G2,G3,G9
unavailable_gates=G4,G5,G6,G7,G8
best_observable=V4 + G3
best_observable_selected_exact_relaxed=9/32
best_observable_p2_oracle_selected=2
best_observable_rescues=5
best_observable_harms=2
best_observable_net=+3
best_zero_harm_observable=None
best_oracle_upper_bound=V4 + U1, 11/32, harms=0, not implementable
decision=B
```

Use this to close fixed residual calibration implementation path for now and
return to drift span / fingerprint ambiguity diagnostics. Do not run formal
eval, integrate P2 into eval, or add a rerank mode from this evidence alone.

## P2 Fingerprint Ambiguity Diagnostic

This parses existing JSON and does a small 3-repeat fingerprint stability check
for V4 rescue/harm candidate pairs only. It does not run model decoding,
candidate generation, `sde_validation_probe.py`, formal eval, 64-sample eval,
grids, retraining, scorer changes, rerank-mode changes, checkpoint/data
changes, target-format changes, fingerprint schema changes, or production
default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_fingerprint_ambiguity.py \
  --gate-json p2_calibration_gate_feasibility.json \
  --all32-smoke-json p2_calibration_all32_smoke.json \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  --p2-coverage-json candidate_coverage_pair_expanded_p2_flag.json \
  --baseline-coverage-json candidate_coverage_diagnostics.json \
  --report p2_fingerprint_ambiguity_diagnostics.md \
  --json-output p2_fingerprint_ambiguity_diagnostics.json \
  > /private/tmp/gensr_sde_p2_fingerprint_ambiguity.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_fingerprint_ambiguity.py \
  scripts/analyze_p2_calibration_gate_feasibility.py \
  scripts/analyze_p2_calibration_all32_smoke.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
samples_analyzed=19
V4_rescue_count=5
V4_harm_count=2
changed_nonoracle_to_nonoracle_count=8
V0_preserved_hit_count=4
removed_dims_explain_flip_count=2
removed_dims_explain_harm_count=0
active_dominated_ambiguity_count=1
weak_dominated_ambiguity_count=2
mixed_ambiguity_count=4
same_diffusion_wrong_drift_count=1
same_drift_wrong_diffusion_count=5
both_sides_wrong_count=1
constant_only_or_near_constant_count=0
resimulation_available_bool=True
resimulation_ordering_stable_count=2
resimulation_oracle_recovers_count=2
decision=C
```

Use this to prioritize fingerprint simulation stability before any further
scorer calibration changes. Do not run formal eval, integrate P2 into eval, or
add a rerank mode from this evidence alone.

## P2 Fingerprint Simulation Stability Diagnostic

This uses existing JSON plus small controlled candidate fingerprint
resimulation for the 7 V4 rescue/harm cases only. It does not run model
decoding, candidate generation, `sde_validation_probe.py`, formal eval,
64-sample eval, grids, retraining, scorer changes, rerank-mode changes,
checkpoint/data changes, target-format changes, fingerprint schema changes, or
production default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_fingerprint_stability.py \
  --ambiguity-json p2_fingerprint_ambiguity_diagnostics.json \
  --all32-smoke-json p2_calibration_all32_smoke.json \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  --target-case-types V4_rescue,V4_harm \
  --repeats 5 \
  --report p2_fingerprint_stability_diagnostics.md \
  --json-output p2_fingerprint_stability_diagnostics.json \
  > /private/tmp/gensr_sde_p2_fingerprint_stability.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_fingerprint_stability.py \
  scripts/analyze_p2_fingerprint_ambiguity.py \
  scripts/analyze_p2_calibration_all32_smoke.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
samples_analyzed=7
candidate_pairs_analyzed=8
repeats=5
M0_stored_target_stored_candidate=available
M1_stored_target_resim_candidate=available
M2_resim_target_stored_candidate=unavailable
M3_paired_resim_target_and_candidate=unavailable
M1_ordering_stable_count=1
M1_ordering_unstable_count=7
M3_ordering_stable_count=NA
M3_ordering_unstable_count=NA
noise_dominated_pair_count=8
stable_clear_pair_count=0
active_noise_driver_count=2
weak_noise_driver_count=1
mixed_noise_driver_count=5
rescue_cases_noise_dominated_count=6
harm_cases_noise_dominated_count=2
rescue_cases_oracle_recovers_majority_count=2
harm_cases_oracle_recovers_majority_count=1
removed_dims_noise_dominated_count=7
removed_dims_stable_explanation_count=0
decision=B
```

Use this to diagnose candidate fingerprint variance / path budget before any
scorer calibration changes. M2/M3 target-resimulation attribution is blocked by
missing stored candidate fingerprint vectors and raw numeric target constants.
Do not run formal eval, integrate P2 into eval, or add a rerank mode from this
evidence alone.

## P2 Candidate Fingerprint Path-Budget Diagnostic

This follows the stability diagnostic and tests fixed path-budget modes for the
same 7 V4 rescue/harm samples and 8 oracle-pair comparisons. It keeps the
stored target fixed and resimulates candidate fingerprints only. It does not
run model decoding, candidate generation, `sde_validation_probe.py`, formal
eval, 64-sample eval, retraining, scorer changes, rerank-mode changes,
checkpoint/data changes, target-format changes, fingerprint schema changes, or
production default changes.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_p2_candidate_fingerprint_path_budget.py \
  --stability-json p2_fingerprint_stability_diagnostics.json \
  --ambiguity-json p2_fingerprint_ambiguity_diagnostics.json \
  --all32-smoke-json p2_calibration_all32_smoke.json \
  --scorer-ready-json p2_scorer_ready_candidates_all32.json \
  --target-case-types V4_rescue,V4_harm \
  --report p2_candidate_fingerprint_path_budget.md \
  --json-output p2_candidate_fingerprint_path_budget.json \
  > /private/tmp/gensr_sde_p2_candidate_fingerprint_path_budget.log 2>&1
```

Compile check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache \
/opt/miniconda3/envs/gensr/bin/python3 -m py_compile \
  scripts/analyze_p2_candidate_fingerprint_path_budget.py \
  scripts/analyze_p2_fingerprint_stability.py \
  scripts/analyze_p2_fingerprint_ambiguity.py \
  scripts/analyze_p2_calibration_all32_smoke.py \
  scripts/analyze_p2_admitted_candidate_scoring.py \
  sde_fingerprint.py \
  simulator_sde.py \
  sde_dataset_generator.py
```

Latest result:

```text
samples_analyzed=7
candidate_pairs_analyzed=8
B0_current_budget: stable=0/8, noise_dominated=8/8, stable_clear=0
B1_active_paths_high: stable=4/8, noise_dominated=6/8, stable_clear=2
B2_weak_paths_high: stable=3/8, noise_dominated=8/8, stable_clear=0
B3_both_paths_high: stable=1/8, noise_dominated=7/8, stable_clear=1
best_budget_mode=B1_active_paths_high
decision=B
```

Use this as variance evidence only. Active-path budget helps somewhat, but the
best tested budget still leaves most pair orderings noise-dominated. Do not run
formal eval, integrate P2 into eval, or add a rerank mode from this evidence
alone.
