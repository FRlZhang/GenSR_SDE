# AGENTS.md

## Project Goal

Extend GenSR from algebraic symbolic regression to autonomous 1D SDE symbolic
discovery. The current target is not just teacher-forced token recognition, but
sequence-level recovery of drift and diffusion expressions from SDE
fingerprints.

## Required Reading

Before making meaningful changes, read these files in order:

1. [README.md](/Users/lzhang/Documents/GenSR_SDE/README.md)
2. [PROJECT_STATUS.md](/Users/lzhang/Documents/GenSR_SDE/PROJECT_STATUS.md)
3. [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md)
4. `git status --short`
5. the relevant entrypoint for the task:
   `sde_validation_probe.py`, `sde_dataset_generator.py`, `sde_fingerprint.py`,
   `train.py`, or files under `symbolicregression/`

Do not rely on prior chat context if the project files disagree.

## Workflow Rules

- Preserve the upstream GenSR code path unless the task clearly requires
  changing it.
- Prefer incremental changes over broad refactors.
- Keep the current SDE symbolic target format as
  `<DRIFT> drift_tokens <DIFFUSION> diffusion_tokens` unless there is strong
  evidence to change it.
- Treat `sde_validation_probe.py` as the main fast-turnaround experiment entry.
- Reuse checkpoints with `--save-checkpoint`, `--load-checkpoint`, and
  `--eval-only` instead of retraining for every decoding experiment.
- When comparing experiments, prefer the same seeds and similar `n-paths`,
  `active-paths`, `n-steps`, and batch size unless the point of the experiment
  is to change those settings.
- Record important decisions in `PROJECT_STATUS.md` or `README.md`, not only in
  chat.

## Verification Rules

- Always run `py_compile` on edited Python entrypoints.
- Always run `git diff --check` after manual edits.
- For training / decoding changes, run at least one smoke test with
  `sde_validation_probe.py`.
- Do not claim SDE evidence from pure training loss alone; prefer held-out
  metrics such as `token_top1`, `token_top3`, `token_top5`,
  `greedy_sequence_exact`, `greedy_sequence_relaxed_no_constants`, and
  `constrained_beam_*`.
- If a checkpoint path outside the repo is important, record it in
  `PROJECT_STATUS.md`.

## Status Update Rule

- Update [PROJECT_STATUS.md](/Users/lzhang/Documents/GenSR_SDE/PROJECT_STATUS.md)
  whenever one of these happens:
  - a major experiment finishes;
  - a decision changes the recommended direction;
  - a new checkpoint becomes the main resume point;
  - a blocker or known issue changes;
  - work is being handed to a new thread, provider, or agent.
- Update [README.md](/Users/lzhang/Documents/GenSR_SDE/README.md) when entry
  commands, data paths, file structure, or the main workflow change.
- Do not turn `AGENTS.md` into a lab notebook. Keep it stable.

## Web GPT Snapshot Rule

At the end of any Codex workflow that changes project state, update the
lightweight Web GPT snapshot in `web_gpt_snapshot/`.

Always update:

- `02_RECENT_CHANGES.md`
- `03_CURRENT_TASK.md`

Also update `01_PROJECT_STATUS.md` when the current goal, best checkpoint, key
metrics, blocker, or next-step priority changes.

Update `04_COMMANDS.md` only when commands, paths, environment variables, or
verification steps change.

Update `00_WEB_GPT_INDEX.md` only when the overall project direction or
high-level handoff policy changes.

Do not generate many timestamped handoff files. Keep the fixed small snapshot
files stable so they can be manually replaced in ChatGPT Web Project Sources.

Do not copy datasets, checkpoints, `dump/`, `weights/`, `.git/`, caches, or the
full `symbolicregression/` directory into the snapshot.

If a future Web GPT task needs code-level patching, update
`NEEDED_SOURCE_FILES.md` to list the minimal source files the user should
temporarily upload.

## Safety Rules

- Do not overwrite or delete datasets, debug outputs, or reports unless the
  change is deliberate and documented.
- Do not trust old pre-fix SDE results unless they are explicitly marked
  corrected in the report.
- Assume checkpoint paths under `/private/tmp` are ephemeral; if a checkpoint is
  essential, record the command needed to regenerate it.
- Do not use destructive git commands such as `reset --hard` or checkout-based
  reverts unless explicitly requested.
