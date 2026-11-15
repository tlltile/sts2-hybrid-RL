> **Public repository scope:** This document describes the complete
> locally validated reproducibility package. The current public GitHub
> repository contains the verified statistical reproduction materials
> and selected research files, but does not yet provide a standalone
> executable hybrid agent. Commands referencing `run_agent.py`,
> `sts2_env`, or `card_reference_cache.py` require the complete package
> and the corresponding authorized reference inputs.

# Reproducibility

## Install and verify

Run from the candidate root. The frozen Phase 7H environment was Windows x64, Python 3.12.10, NumPy 2.5.4, Gymnasium 1.4.0, Torch 2.14.0+cu130 and Stable-Baselines3/sb3-contrib 2.9.2. Validation used Python 3.12.13 with the matching core dependency versions and CPU inference. See `models.json`, `requirements-frozen.txt`, and the snapshot's `uv.lock`. The lockfile is retained historical metadata. Use the explicit dependency pins for this candidate.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -I -m pip --isolated install --disable-pip-version-check --no-cache-dir --retries 0 --timeout 12 --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-frozen.txt
.\.venv\Scripts\python.exe -I -m pip --isolated install --no-index --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -B run_agent.py --verify-models
```

For an offline installation, replace the index option with `--no-index --find-links D:\authorized\wheelhouse` using a wheelhouse for the listed Windows versions and build dependencies. Both commands passed using fresh Python 3.12.13 with no borrowed packages. The pinned Torch wheel downloaded about 2 GB. Inference runs on CPU.

## Run the trained hybrid

```powershell
python -B run_agent.py --root 1 --card-reference 'D:\authorized\decompiled\MegaCrit.Sts2.Core.Models.Cards'
python -B run_agent.py --root 2 --seed 12104718 --development --card-reference 'D:\authorized\decompiled\MegaCrit.Sts2.Core.Models.Cards' --max-card-decisions 1
```

The second command launches a new run on an already-open G1 development origin. Remove the card-decision limit for a full episode. This launcher admits seed 42 and the opened G1 development panel only. It rejects a non-default origin outside that panel, and the frozen environment also screens external and derived native seeds against its deny ranges. The demo uses an independent Torch action generator and the original masked sampling and step metadata. It does not implement the research RNG-role reservation or exact final assessment workflow. `--without-card-reference` is an explicit diagnostic mode with unverified fidelity.

### Reference provenance and validation

- `docs/CARDS_REFERENCE.md` is mandatory for the native card factory and included in the local candidate. The retained checkout copy has SHA-256 `fa437fef041e737c1ad5d212821a315a8054d0b9a1926c2be1fc3c3e3af84e60`. The launcher checks it before gameplay. It is game-derived and redistribution is pending.
- `frozen_source.json` → `external_card_reference` is the complete 579-file top-level C# inventory for the retained v0.107.1 card directory. Supply an authorized copy of the byte-matching `MegaCrit.Sts2.Core.Models.Cards` folder from the retained checkout or equivalent authorized reference snapshot. The launcher checks membership and per-file SHA-256 before binding the frozen parser. It rejects missing, extra or modified C# files.
- These are the reference inputs used by the tested hybrid path. The frozen parser enumerates the directory as a registry. No smaller C# subset has been validated for faithful gameplay. Historical file-level provenance for these supplemental reference inputs is the retained checkout inventory, and the Phase 7H source snapshot does not independently seal this entire reference corpus.

## Verify report results offline

```powershell
python -B reproduce_results.py
python -I -B release/phase9c_v1/reproduce.py
```

The first command requires NumPy for the exact Phase 7H resampling procedure. The second uses the standard library and verifies the Phase 9C subset. Neither command starts the simulator.

## Historical training and evaluation

These are source-supported commands and configuration records. They were not executed in this build. Historical full-run reproduction requires the original version-specific source, reference data, manifests, role ledgers, initialization/model files and sealed paths. This candidate pins the Phase 7H inference snapshot. Original Combat pretraining and HP training have separate historical source/runtime provenance. The sanitized inference copies in `checkpoints/` are for the submission launcher. Historical trainers and loaders still expect their original archives and locations.

```powershell
python -B -m scripts.train_combat --seed 0 --encounter-pool hallway --total-timesteps 100000 --n-envs 4 --n-steps 512 --batch-size 256 --n-epochs 10 --lr 0.0003 --gamma 0.99 --ent-coef 0.01 --output-dir local_runs/combat_seed0
python -B -m scripts.train_combat_hp --source local_runs/combat_seed0 --arm curriculum --seed 1 --total-timesteps 100000 --n-envs 4 --n-steps 512 --batch-size 256 --n-epochs 10 --eval-freq 25000 --eval-workers 4 --eval-inference scalar --checkpoint-freq 25000 --validation-seeds 20 --output-dir local_runs/combat_hp_r1
python -B -m scripts.train_card_ppo --describe
```

Use fresh output paths. These two Combat commands describe a new run with the historical configuration. Exact equivalence to the original training runs is unverified with the Phase 7H snapshot. For Card PPO, the historical execution syntax was `python -B -m scripts.train_card_ppo --execute --plan <sealed-manifest> --expected-sha256 <manifest-digest> --output <canonical-output>`. G1 used three scratch roots, 128 origins × 16 passes per learner, 64 completed episodes per update, 32 updates, Adam 0.0003, four epochs, minibatch 64, clip 0.2, entropy 0.01, value coefficient 0.5, gamma/lambda 1, KL threshold 0.03 and gradient cap 0.5. The original G1 manifest and output are already consumed. Independent Phase 7H re-execution is outside this candidate.

The validated local cache workflow and its provenance limits are in [Card reference portability](REFERENCE_PORTABILITY.md).


