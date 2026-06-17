# GenSR_SDE Web GPT Index

## One-Line Goal

Extend GenSR from algebraic symbolic regression to autonomous 1D SDE symbolic discovery: recover drift and diffusion expressions from SDE fingerprints.

## Fork Relationship

This repo is a local research fork of upstream GenSR. Upstream GenSR's CVAE / symbolic regression code remains under `symbolicregression/`, while this fork adds SDE simulation, unified SDE fingerprints, SDE dataset generation, role-token symbolic targets, and a fast validation probe.

The current SDE target format is:

```text
<DRIFT> drift_tokens <DIFFUSION> diffusion_tokens
```

## Current Mainline

The unified `multi_active_weak_v1` fingerprint is learnable. A 2000-step role-token probe reached strong teacher-forced token metrics:

```text
token_top1=0.820002
token_top3=0.960833
token_top5=1.000000
```

But sequence-level recovery is still weak: greedy exact recovery is about `1/64`, and constrained beam only slightly improves exact hits. First-pass fingerprint reranking, grammar-constrained stochastic sampling, and drift/diffusion pairing are implemented. Pairing produced nonzero oracle hits in a 16-sample checkpoint eval, but full and componentwise fingerprint-distance scores did not select them. The bottleneck is now score diagnostics around paired candidates and constant sensitivity, not fingerprint redesign.

Current best checkpoint:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Read These First

Web GPT should read these lightweight files first, not the full repo:

1. `00_WEB_GPT_INDEX.md`
2. `01_PROJECT_STATUS.md`
3. `02_RECENT_CHANGES.md`
4. `03_CURRENT_TASK.md`
5. `04_COMMANDS.md`

Only ask for source files when writing a concrete patch.

## Do Not Prioritize

- Do not first redesign the SDE fingerprint.
- Do not default to two fully independent drift/diffusion decoders.
- Do not use old pre-fix loss-only probes as SDE recovery evidence.
- Do not retrain for every decoding experiment; prefer checkpoint `--load-checkpoint` and `--eval-only`.
- Do not request `data/*.pkl`, checkpoints, `weights/`, `dump/`, `.git/`, or full `symbolicregression/` unless a specific bug requires it.

## Source Files To Upload Only For Patch Work

- `sde_validation_probe.py`: primary file for candidate diversity and reranking improvements.
- `sde_fingerprint.py`: fingerprint scoring/recomputation.
- `sde_dataset_generator.py`: target token format and expression encoding.
- `simulator_sde.py`: `SDESystem` and `solve_fingerprint`.
- `train.py`: only if changing the full training path or SDE data injection.
