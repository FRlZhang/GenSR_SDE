# Needed Source Files For Patch Work

Do not upload source files by default. Upload these only when web GPT is asked to write or review a concrete patch.

- `sde_validation_probe.py`: primary target for diverse candidate generation, reranking, and new metrics.
- `sde_fingerprint.py`: needed to design or reuse fingerprint-distance scoring.
- `simulator_sde.py`: needed to turn candidate drift/diffusion strings into SDE fingerprints via `SDESystem` / `solve_fingerprint`.
- `sde_dataset_generator.py`: needed for role-token encoding, expression normalization, and GenSR prefix-token conventions.
- `train.py`: only needed if modifying full training or SDE data injection, not for most decoding experiments.

Do not upload `symbolicregression/` as a whole unless a specific stack trace points there.
