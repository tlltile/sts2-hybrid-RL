# Slay the Spire 2 — Hybrid Reinforcement Learning Agent

**Columbia University — ELEN E6885 Reinforcement Learning Final Project**

This repository accompanies a reinforcement learning project investigating combat control and long-term card-selection decisions in *Slay the Spire 2*.

The project focuses on **Ironclad, Ascension 0, Act 1**, with the objective of clearing the Act 1 boss.

The research system combines:

- **Combat PPO:** A learned policy for tactical combat decisions.
- **Card-Selection PPO:** A candidate-aware policy for post-combat card rewards.
- **Hybrid Strategic Controller:** Rule-based management of routing, shops, events, and rest sites.
- **Frozen Python Simulator:** A version-specific research environment used for training and evaluation.

## 1. Research Results

The project reports the following experimental findings:

| Experiment | Result |
|---|---|
| Historical hybrid baseline | 3/40 Act 1 boss clears |
| Compact combat validation | 192/192 wins per root and arm on the evaluated panel |
| Card PPO Root 1 | 10/100 clears |
| Card PPO Root 2 | 2/100 clears |
| Card PPO Root 3 | 8/100 clears |
| Fixed card-selection baseline | 4/100 clears |
| Uniform card-selection baseline | 4/100 clears |
| Phase 8 origin-diversity comparison | 33/384 vs. 27/384 clears |

The final Card PPO comparison did not establish a statistically significant improvement over the matched baselines. No winning root was selected based on this final assessment.

See [RESULTS.md](RESULTS.md) for additional details and limitations.

## 2. Reproduce the Reported Statistics

The Phase 9C results can be reproduced directly from the included evaluation data.

**Requirements:**

- Python 3.12
- No GPU required
- No game installation required
- No external simulator required for this statistical reproduction

From the repository root, run:

```powershell
py -3.12 -I -B .\release\phase9c_v1\reproduce.py
```

The script checks the retained evidence files and recomputes the supported experimental statistics.

**Verified:** This command was executed successfully in an isolated Python 3.12 environment, with 17 evidence files verified.

This workflow reproduces statistical summaries from stored evidence. It does **not** rerun the original gameplay episodes or PPO training.

## 3. Reinforcement Learning Implementation

The repository includes selected RL implementation files under:

```text
agent/
  card_ppo/
  combat_ppo/
```

The Card PPO implementation includes policy-model and training components.

The Combat PPO implementation includes training and HP curriculum components.

These files preserve selected research implementations, but they are **not a standalone executable simulator package**. Their imports and runtime dependencies refer to the original `sts2_env` package structure.

The complete experimental environment and training configuration are described in the accompanying project documentation.

## 4. Trained Model Checkpoints

The local release candidate contains four trained model checkpoints:

```text
checkpoints/
  combat_v1.zip
  card_root_1.pt
  card_root_2.pt
  card_root_3.pt
```

These checkpoints are intended for inference with the matching frozen simulator, model definitions, and dependency versions.

The presence of checkpoint files alone does not make the public repository independently runnable.

Checkpoint availability in the public release is subject to final distribution review.

## 5. Running the Hybrid RL Agent

The complete local reproducibility candidate has been validated separately on Windows with Python 3.12.

Its supported workflows include:

```powershell
python -B run_agent.py --verify-models
python -B run_agent.py
python -B reproduce_results.py
```

These commands apply to the **complete reproducibility package**, not to the current public source subset.

Running the Hybrid Agent requires:

1. The matching frozen `sts2_env` simulator.
2. The trained Combat and Card PPO checkpoints.
3. The correct Python dependencies and model definitions.
4. Authorized matching card-reference data.

The full simulator depends on version-specific card metadata. The verified native reference workflow uses a matching card-reference snapshot, rather than automatically downloading data from the current game version.

The frozen research environment should not be interpreted as an exact reproduction of the current Steam release.

Access to the complete execution environment must follow the applicable upstream and game-data permissions.

## 6. Reproducibility Scope and Limitations

Three levels of reproduction are distinguished:

**A. Statistical reproduction — Verified**

The included Phase 9C evidence can be used to recompute the reported statistical results.

**B. Trained-agent inference — Verified in the complete local candidate**

The trained Combat and Card PPO models have been loaded and exercised with the frozen Python simulator. This has not yet been independently validated from the public source subset.

**C. Historical training and final evaluation — Not independently rerun**

The original training and evaluation protocols require additional historical artifacts, environment state, reference inputs, and sealed experiment configurations. The present release should not be interpreted as an independently verified exact rerun of those historical experiments.

## 7. Attribution

The simulator development originated from the upstream project:

[zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent)

The research environment was based on a frozen upstream revision, with subsequent modifications and original RL implementation work.

See [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md) for source provenance and the pinned revision.

*Slay the Spire 2* is developed by Mega Crit. This project is an independent academic research project and is not affiliated with or endorsed by Mega Crit.

## 8. Project Scope

The implemented and evaluated target is **Ironclad A0, Act 1**.

The system architecture was designed with potential extension to later acts, additional cards, and broader strategic decisions in mind. Such extensions are outside the validated results presented here.

The accompanying final report provides the problem formulation, methodology, experiments, statistical analysis, and research limitations.