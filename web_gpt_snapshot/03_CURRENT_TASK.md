# Current Task

## Next Engineering Task

Decide the next scorer-side residual/calibration diagnostic after P2
score-miss residuals showed mostly active/weak traps. Do not widen drift
beam/top-k or change production candidate-generation defaults from this
coverage-only result alone.
The drift-only smoke is complete: exact drift appears in only `3/17` targets,
all beam rank-30 nested-mul linear cases. Candidate-normalization folding is
complete in helper-only mode: current expanded-pool folding projects `23/32`,
exact-tail admission projects `18/32`, full normalized drift-only admission
projects `30/32` with pair count `1105`, capped `P2_one_per_family` preserves
`30/32` with pair count `340`, and the reusable hook reproduces those numbers
with strict self-checks. The candidate coverage pipeline now has a default-off
P2 flag, validated from existing JSON with metric_cross_check_status `ok`,
schema_status `ok`, canonicalizer_safety_status `ok`, zero mismatches, decision
`A`. The attempted runtime smoke was blocked before launch because the required
source log is missing; `candidate_coverage_pair_expanded_p2_flag.md/json` was
not produced. That source log has since been regenerated and backed up, so the
runtime smoke was retried and passed with decision `A`. Residual-debug safety
checks do not support continuing scorer-side normalization or adding a new
rerank mode.
The JSON-only P2 admitted-candidate scoring readiness diagnostic found that the
7 newly recovered P2 samples (`0,2,6,7,15,25,29`) are coverage-ready but not
semantic-scoring-ready from existing JSON: exact diffusion is present and the
P2 canonical oracle drift is beam-source rank 5 for each sample, but target
`y_to_fit` / fingerprint vectors are missing.
The minimal logging gap has now been fixed:
`scripts/analyze_candidate_coverage.py` has default-off `--scorer-ready-json`
and `--scorer-ready-target-samples`, and `p2_scorer_ready_candidates.json`
validates with decision `A`. Offline semantic scoring on that sidecar is now
complete and selects 0/7 P2 oracle candidates.
The targeted residual/segment diagnostic is now complete: classifications are
active-trap 3, weak-trap 1, near-tie 2, current-candidate trap 1. This does not
support P2 eval integration yet.

Current strongest no-retraining decoding setting:

```text
rerank_score=constant_grid_rolewise_no_multi_u0
active:weak=1:1
selected exact/relaxed=5/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
```

Expanded-pairing formal eval:

```text
pair_drift_topk=5
pair_diffusion_topk=6
pair_drift_diffusion_candidates=32
selected exact/relaxed=3/32
oracle exact/relaxed=15/32
pair oracle exact/relaxed=8/32
oracle-present samples=15
selected oracle hits=3
selected oracle misses=12
oracle-only-pair misses=7
beam/sampling score misses=5
```

Expanded-pairing miss ranking:

```text
selected misses analyzed=12
oracle-only-pair misses=7
near score gaps <=0.10=2/12
oracle-only-pair near gaps <=0.10=1/7
selected source is pair=8/12
oracle-only-pair misses dominated by active_kramers_moyal=7/7
```

Wrong-pair pruning diagnostic:

```text
wrong selected pair cases=8
pair-oracle samples to preserve=8
pair-oracle source ranks=2..15
approx wrong-pair source ranks=1..14
active-distance traps among wrong selected pairs=5/8
safe_pruning_rule_found=0
```

Expanded-pairing active residual trap diagnostic:

```text
selected misses analyzed=12
wrong selected pair cases=8
oracle-only-pair misses=7
active favors selected=12/12
weak favors oracle while active favors selected=4/12
outlier-dominated active traps=7/12
broad active traps=2/12
mixed active traps=3/12
KM1/drift-leaning active advantages=10/12
```

Expanded-pairing active residual ablation:

```text
variants=current_logged_rolewise,current_recomputed,active_clipped_l2_p95_plus_weak,active_clipped_l2_p90_plus_weak,active_huber_p90_plus_weak,active_top1_removed_plus_weak,active_top2_removed_plus_weak,per_active_dim_norm_plus_weak,weak_only_diagnostic
best offline variant=per_active_dim_norm_plus_weak
best offline variant rescues=6/12
oracle-only-pair rescues=4/7
wrong selected pair rescues=5/8
debug top2 selected-hit harms=0/3 for best variant
clipped/Huber/top-k variants harm one debug top2 selected hit
decision=C
```

Residual debug logging smoke:

```text
new flag=--rerank-residual-debug-json
latest smoke samples=16
candidates logged=201
active residual vector length=18
weak residual vector length=96
samples with oracle=2
selected hits=1
pair oracle samples=1
selected-hit harms under offline safety score=0
oracle-present selected-miss rescues=0
safety helper result=limited no-harm evidence, no rescue
```

Prepared expanded-pairing residual-debug smoke:

```text
script=scripts/run_expanded_pairing_residual_debug_smoke16.sh
status=user ran locally
candidates logged=350
samples with oracle=5/16
pair-oracle samples=4/16
selected hits=0/16
oracle-present selected misses=5/16
offline per_active_dim_norm_plus_weak rescues=0/5
offline selected pair candidates=8
oracle ranks under offline score=5:8,8:6,11:3,12:3,13:11
```

Candidate-coverage-only result:

```text
expanded pairing: pair_drift_topk=5, pair_diffusion_topk=6, pair_drift_diffusion_candidates=32
full_oracle_present=15/32
oracle_absent=17/32
rescued pairing-missing samples=5,8,11,23,30,31
```

Expanded-pairing oracle-absent drift diversity report:

```text
script=scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py
report=expanded_pairing_oracle_absent_drift_diversity.md
json=expanded_pairing_oracle_absent_drift_diversity.json
oracle_absent=17/32
exact_drift_missing=17/17
pairing_missing=0/17
diffusion_missing=0/17
exact_diffusion_present=17/17
nearest_drift_right_family=9/17
decision=A
```

Drift-family breakdown among the 17 exact-drift-missing cases:

```text
linear drift=6
sin drift=6
nested-mul linear drift=3
polynomial-like drift=2
constant drift=0
other/unknown=0
```

Interpretation: exact drift is absent from all logged sources for the remaining
17 samples, while exact diffusion is already present. Beam contributes more
usable drift diversity than sampling in the emitted pool; sampling mostly
duplicates a small set of nearby templates. Nearest logged drifts often collapse
toward over-nested or constant-heavy forms such as `mul CONSTANT CONSTANT`,
`mul mul CONSTANT CONSTANT CONSTANT`, and
`mul mul CONSTANT CONSTANT sin x_0`.

Existing logs are sufficient to prove current-pool exact-drift absence, but
insufficient to distinguish unlogged beam/sampling tail misses from model or
grammar distribution misses because they lack unlogged tail ranks, token
logits/entropy, and grammar rejection counts.

Drift-only candidate diversity helper:

```text
script=scripts/analyze_drift_only_candidate_diversity.py
standalone=scripts/run_drift_only_candidate_diversity_smoke.sh
blocked_report=drift_only_candidate_diversity_smoke.md
blocked_json=drift_only_candidate_diversity_smoke.json
status=user completed standalone smoke
decision=A
```

Standalone smoke settings:

```text
batch_size=2
drift_max_len=16
drift_beam_size=16
drift_beam_candidates=32
drift_sample_candidates=48
sample_temperatures=0.8,1.0,1.2
sample_top_k=8
sample_top_p=0.95
```

Completed drift-only smoke result:

```text
target_samples=17
exact_truth_drift_found=3/17
found_by_source: beam=3, sampling=0, both=0, neither=14
exact_hit_samples=2,25,29
exact_hit_rank=30 for all three
recovered_family: nested-mul linear=3/3
missing_family: linear=6/6, sin=6/6, polynomial-like=2/2
exact_diffusion_present=17/17
```

Admission + constant-folding diagnostic:

```text
script=scripts/analyze_drift_only_admission_and_constant_folding.py
report=drift_only_admission_constant_folding_diagnostics.md
json=drift_only_admission_constant_folding_diagnostics.json
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

Candidate-normalization constant-folding diagnostic:

```text
script=scripts/analyze_candidate_normalization_constant_folding.py
report=candidate_normalization_constant_folding_diagnostics.md
json=candidate_normalization_constant_folding_diagnostics.json
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
current_expanded_pool_gain=+8
current_expanded_pool_new_samples=1,3,4,9,10,14,20,22
exact_tail_admission_coverage=18/32
exact_tail_recovered_samples=2,25,29
canonicalized_drift_only_admission_coverage=30/32
canonicalized_drift_only_recovered_samples=0,1,2,3,4,6,7,9,10,14,15,20,22,25,29
polynomial_like_missing_after_normalization=16,28
source_breakdown_current_pool: beam+pair=7, both=1
source_breakdown_drift_only_tail: beam=15
collision_groups=9
unsafe_collision_flag=False
serialized_raw_to_canonical=440 -> 278
serialized_reduction=36.8%
decision=B
```

Normalized drift-only admission coverage diagnostic:

```text
script=scripts/analyze_normalized_drift_only_admission_coverage.py
report=normalized_drift_only_admission_coverage_diagnostics.md
json=normalized_drift_only_admission_coverage_diagnostics.json
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
exact_tail_admission_coverage=18/32
normalized_drift_only_admission_coverage=30/32
newly_recovered_beyond_current_expanded_canonical_pool=0,2,6,7,15,25,29
remaining_missing_samples=16,28
drift_only_tail_normalized_sources: beam=15
sampling_contributes_recovered_canonical_drift=False
raw_expanded_drift_count_sum_targets=74
canonical_drift_only_admitted_count_sum=221
pair_count_before_raw_sum=370
pair_count_after_canonical_sum=1105
pair_count_canonical_increase=+735 (+198.6%)
tail_canonical_dedup_reduction=39.6%
unsafe_collision_flag=False
decision=B
```

Capped normalized drift admission diagnostic:

```text
script=scripts/analyze_capped_normalized_drift_admission.py
report=capped_normalized_drift_admission_diagnostics.md
json=capped_normalized_drift_admission_diagnostics.json
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

Reusable P2 normalized admission hook:

```text
module=scripts/normalized_drift_admission.py
runner=scripts/run_p2_normalized_admission_coverage.py
report=candidate_coverage_p2_normalized_admission.md
json=candidate_coverage_p2_normalized_admission.json
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
P2_normalized_admission_coverage=30/32
P2_pair_count=340
full_normalized_admission_pair_count=1105
truth_aware_upper_bound_pair_count=320
newly_recovered=0,2,6,7,15,25,29
remaining_missing=16,28
source_breakdown: beam=15
sampling_excluded=True
sampling_recovered_canonical_drift_count=0
self_check_failures=0
```

Reusable P2 hook cross-validation:

```text
script=scripts/validate_p2_normalized_admission_hook.py
report=p2_normalized_admission_hook_validation.md
json=p2_normalized_admission_hook_validation.json
json_reports_validated=6
metric_cross_check_status=ok
schema_status=ok
canonicalizer_safety_status=ok
mismatch_count=0
decision=A
```

Candidate coverage P2 flag integration:

```text
script=scripts/validate_candidate_coverage_p2_flag.py
report=candidate_coverage_p2_flag_validation.md
json=candidate_coverage_p2_flag_validation.json
new_flag=--normalized-drift-admission none|p2_one_per_family
default=none
source_flag=--normalized-drift-admission-source beam
metric_cross_check_status=ok
schema_status=ok
canonicalizer_safety_status=ok
mismatch_count=0
model_backed_candidate_generation_avoided=True
decision=A
```

P2 flag runtime smoke attempt:

```text
report=candidate_coverage_pair_expanded_p2_flag_runtime_report.md
json=candidate_coverage_pair_expanded_p2_flag_runtime_report.json
status=blocked
missing_input=/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log
model_backed_candidate_coverage=not_run
candidate_coverage_pair_expanded_p2_flag.md/json produced=False
decision=D
```

P2 flag runtime smoke completed:

```text
report=candidate_coverage_pair_expanded_p2_flag_runtime_report.md
json=candidate_coverage_pair_expanded_p2_flag_runtime_report.json
runtime_output=candidate_coverage_pair_expanded_p2_flag.md/json
current_expanded_full_oracle=15/32
canonicalized_expanded_pool_coverage=23/32
P2_normalized_admission_coverage=30/32
P2_pair_count=340
newly_recovered=0,2,6,7,15,25,29
remaining_missing=16,28
sampling_recovered_canonical_drift_count=0
unsafe_collision_flag=False
decision=A
```

Restored source log and backup:

```text
regenerated=/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log
selected=5/32
oracle=9/32
pair_oracle=2/32
backup_dir=experiment_logs/2026-06-24_tmp_gensr_sde_logs/
backup_status=all surviving /private/tmp/gensr_sde*.log/json copied
```

Interpretation: current expanded-pool canonicalization helps but is not enough,
full normalized drift-only admission is too high-pressure to adopt wholesale,
and `P2_one_per_family` preserves the coverage ceiling with much lower pair
pressure. The hook is now cross-validated and integrated behind a default-off
coverage-only candidate coverage flag, and the runtime path now matches the
prior validation. A follow-up scorer-ready sidecar logging run now provides
target fingerprint / `y_to_fit` fields and full candidate rows for the 7 P2
newly recovered samples. Offline semantic scoring now shows all 7 P2 oracle
candidates lose under the current scorer, so the next useful step is a targeted
residual/calibration diagnostic on the observed active/weak traps, not naive
global beam/top-k expansion or formal eval.

Do not launch formal eval from these diagnostics. Exact-tail admission alone is
weak and expensive-looking because all exact hits are rank-30 beam cases.
Constant-chain folding is the stronger offline signal.

Role-wise constants rescued one shared-constant miss, sample 17, by allowing
drift constant `1.0` and diffusion constant `0.5`. The active-heavy `2:1` and
weak-downweighted `1:0.5` variants both regressed to `3/32`, so do not continue
broad active/weak grid search.

Targeted residual-vector diagnostics now exist:

```text
scripts/analyze_rolewise_residuals.py
rolewise_residual_vector_diagnostics.md
rolewise_residual_vector_diagnostics.json
/private/tmp/gensr_sde_rolewise_residual_vectors.log
```

Residual-vector classifications:

```text
sample 13: both-side ambiguity; feature-scaling artifact; model-score conflict
sample 19: both-side ambiguity; feature-scaling artifact
sample 24: drift-side ambiguity; feature-scaling artifact
sample 26: diffusion-side ambiguity; feature-scaling artifact
```

Offline score ablation also exists:

```text
scripts/analyze_residual_score_ablation.py
rolewise_residual_score_ablation.md
/private/tmp/gensr_sde_residual_score_ablation.log
```

Key result: mild robust/normalization variants rescued `0/4`; only aggressive
`clipped_l2_p90` flipped sample 19. Samples 24 and 26 did not flip under any
tested offline score. Do not add a robust-scoring rerank mode from this evidence
alone.

Candidate coverage diagnostics now exist:

```text
scripts/analyze_candidate_coverage.py
candidate_coverage_diagnostics.md
/private/tmp/gensr_sde_candidate_coverage.log
```

Key result:

```text
full_oracle_present=9/32
oracle_absent=23/32
diffusion_only_present=17
drift_and_diffusion_separately_present_but_not_paired=6
drift_only_present=0
neither_side_present=0
```

Interpretation: prioritize drift span diversity first, then pairing
coverage/ranking. Diffusion coverage is already broad in this candidate pool.

Drift/pairing taxonomy exists:

```text
scripts/analyze_drift_pairing_coverage.py
drift_pairing_coverage_diagnostics.md
/private/tmp/gensr_sde_drift_pairing_coverage_full.log
```

Available taxonomy and rank diagnosis:

```text
linear drift missing=6
sin drift missing=6
nested-mul linear drift missing=3
polynomial-like drift missing=2
pairing-missing samples=5,8,11,23,30,31
pairing actions after expansion=all six rescued as pair-source full oracles
```

Pairing details: samples `5`, `8`, `11`, `30`, and `31` were rescued by the
larger pair cap. Sample `23` was rescued by `pair_drift_topk=5` plus the larger
cap; its combined pair rank was 24.

## Primary File To Modify

No primary source file needs to change for a pure offline follow-up. The
candidate-normalization question should be tested in helper-only mode first. Do
not modify `sde_validation_probe.py` unless a concrete narrow logging or
diagnostic hook is proven necessary.

Likely next actions:

- design a helper-only candidate-normalization diagnostic for constant-chain
  folding before pairing/admission;
- avoid naive global beam top-k expansion to 30/32 as a default from this
  evidence alone;
- do not add `per_active_dim_norm_plus_weak` or any robust/clipped normalized
  active scorer mode from current evidence;
- do not run a formal 32-sample eval from the drift-only or constant-folding
  diagnostics alone;
- keep role-wise constant diagnostics in every rerank experiment;
- consider production candidate-generation changes only after the
  constant-folding idea is validated as a candidate-normalization diagnostic.

## Supporting Files If Needed

- `sde_fingerprint.py`: read score component semantics only; do not redesign the
  schema first.
- `simulator_sde.py`: expression lambdification and fingerprint evaluation.
- `sde_dataset_generator.py`: role-token encoding and expression normalization.

## Do Not Touch First

- Do not make `active_distance` or `state_dependent_drift` the default.
- Do not run more broad active/weak weight grids.
- Do not redesign `multi_active_weak_v1`.
- Do not change `<DRIFT> ... <DIFFUSION> ...`.
- Do not retrain.
- Do not overwrite data pickles or checkpoints.
- Do not run 64-sample expansion until a 32-sample setting improves selected
  recovery again.
- Do not make expanded pairing the default: formal selected exact/relaxed
  regressed to `3/32` even though oracle rose to `15/32`.
- Do not add a simple pair-aware tie-break from current evidence: only 1/7
  oracle-only-pair misses is near-tie.
- Do not add pair pruning from current evidence: preserving all pair oracles
  requires keeping ranks up to 15 and harmful structures overlap with true
  oracle families.

## Verification Order

1. `py_compile` edited Python files.
2. `git diff --check`.
3. For helper-only diagnostics, run the helper and redirect output to
   `/private/tmp/<descriptive>.log`.
4. 8- or 16-sample eval-only smoke if probe debug/output changes.
5. 32-sample eval-only probe from:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Success Standard

The next useful scorer-side step is one 16-sample residual-debug smoke so full
candidate-pool hit safety can be checked offline. The other main direction
remains drift span diversity for the 17 expanded-pairing oracle-absent samples.
