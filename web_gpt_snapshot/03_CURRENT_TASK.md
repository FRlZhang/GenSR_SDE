# Current Task

## Next Engineering Task

Run the standalone diagnostic-only drift-only candidate diversity smoke for the
expanded-pairing oracle-absent samples. The helper exists, but the first
Codex-run attempt exceeded the 10-minute budget and was interrupted before
sample metrics were produced. The drift diversity report is complete: the
remaining 17 oracle-absent samples are all exact-drift misses with exact
diffusion present. Residual-debug safety checks do not support continuing
scorer-side normalization or adding a new rerank mode.

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
status=blocked inside Codex after exceeding 10-minute budget
decision=D
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

Run this script next, then inspect `drift_only_candidate_diversity_smoke.md/json`.
Do not launch formal eval from the blocked report.

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

No primary source file needs to change for the next step. The drift-only helper
and standalone script already exist. Do not modify `sde_validation_probe.py`
unless the standalone run reveals a concrete narrow logging bug.

Likely next actions:

- run `scripts/run_drift_only_candidate_diversity_smoke.sh`;
- inspect `drift_only_candidate_diversity_smoke.md/json` after it completes;
- do not add `per_active_dim_norm_plus_weak` or any robust/clipped normalized
  active scorer mode from current evidence;
- do not run a formal 32-sample eval for this scorer idea or from the drift
  diversity report alone;
- keep role-wise constant diagnostics in every rerank experiment;
- consider candidate-generation changes only after a drift-only diagnostic shows
  whether exact drift is below unlogged top-k/sampling tails or absent from the
  current model/grammar distribution.

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
