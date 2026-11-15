"""Opt-in HP initial conditions and evaluation for native combat v1.

No game effects, actor features, action semantics or task rewards are replaced.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import gymnasium as gym

from sts2_env.gym_env.combat_generalization import summarize
from sts2_env.gym_env.episode_trace import MethodEvents
from sts2_env.gym_env.run_native_combat import HALLWAY_SCENARIOS, RunNativeCombatEnv, TASK_VERSION

HP_VERSION = "starter-hallway-hp-blocks-v1"
HP_LEVELS = (80, 60, 40)
ASSESSMENT_HP_LEVELS = (80, 60, 40, 20)
VALIDATION_SEEDS = tuple(range(300000, 300020))
ASSESSMENT_SEEDS = tuple(range(400000, 400100))


def worker_seed(continuation_seed, worker_index):
    """Separate worker streams across root seeds, rather than overlapping seed+i."""
    return int(np.random.default_rng(np.random.SeedSequence(
        [continuation_seed, worker_index, 6888])).integers(0, 2**31))


class ContinuationWorker(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.worker_seed = seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self.worker_seed if seed is not None else None, options=options)

    def action_masks(self):
        return self.env.action_masks()


def sampled_hp(arm, seed, episode_index):
    if arm not in ("control", "curriculum") or seed < 0 or episode_index < 0:
        raise ValueError("Invalid HP arm, seed or episode index")
    if arm == "control":
        return 80
    block, index = divmod(episode_index, len(HP_LEVELS))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6887, block]))
    return int(rng.permutation(HP_LEVELS)[index])


def rounded_budget(requested, n_steps, n_envs):
    if min(requested, n_steps, n_envs) < 1:
        raise ValueError("Budget and rollout sizes must be positive")
    rollout = n_steps * n_envs
    return ((requested + rollout - 1) // rollout) * rollout


class HPHallwayEnv(RunNativeCombatEnv):
    """Versioned initial HP, sampled independently of the base reset RNG.

Reseeding resets the worker's block sequence. Explicit starting_hp is reserved
for fixed-panel evaluation. Equal worker seeds share episode/encounter streams;
episode boundaries in timestep space can diverge as policies and HP differ.
"""
    def __init__(self, arm="control", forbidden_seeds=(), **kwargs):
        sampled_hp(arm, 0, 0)
        self.arm = arm
        self.hp_seed = 0
        self.hp_episode_index = 0
        self.forbidden_seeds = frozenset(forbidden_seeds)
        self._base_scenarios = tuple(kwargs.pop("scenarios", HALLWAY_SCENARIOS))
        super().__init__(scenarios=self._base_scenarios,
                         curriculum_version=f"{HP_VERSION}:{arm}", **kwargs)

    def reset(self, *, seed=None, options=None):
        if seed is not None and int(seed) in self.forbidden_seeds:
            raise RuntimeError("Training reset collided with a reserved evaluation seed")
        if seed is not None:
            self.hp_seed = int(seed)
            self.hp_episode_index = 0
        options = options or {}
        self.sample_index = self.hp_episode_index
        hp = options.get("starting_hp", sampled_hp(self.arm, self.hp_seed, self.sample_index))
        if hp not in ASSESSMENT_HP_LEVELS:
            raise ValueError("Explicit evaluation HP must be 80/60/40/20")
        self.starting_hp = int(hp)

        def prepare(mgr, original=None):
            if original is not None:
                original(mgr)
            if mgr.run_state.player.max_hp != 80:
                raise ValueError("HP curriculum v1 requires starter max HP 80")
            # Initial condition before native room entry, not damage or healing.
            mgr.run_state.player.current_hp = self.starting_hp

        from functools import partial
        self.scenarios = tuple(replace(s, prepare_run=partial(prepare, original=s.prepare_run))
                               for s in self._base_scenarios)
        obs, info = super().reset(seed=seed, options=options)
        if self.forbidden_seeds.intersection((info["episode_seed"], info["run_seed"])):
            raise RuntimeError("Training reset collided with a reserved evaluation seed")
        self.hp_episode_index += 1
        self.hp_lost = self.combat_healing = self.run_healing = 0
        if self.combat.player.current_hp != self.starting_hp:
            raise RuntimeError("Unexpected native start-combat HP change in starter fixture")
        return obs, self._metrics(info)

    def _metrics(self, info):
        return dict(info, starting_hp=self.starting_hp, hp_version=HP_VERSION,
                    hp_arm=self.arm, hp_sample_index=self.sample_index,
                    hp_lost=self.hp_lost, combat_healing=self.combat_healing,
                    run_healing=self.run_healing)

    def step(self, action):
        # Scoped per step: multiple DummyVecEnv instances cannot leave nested
        # global observers installed. Each original method is called once.
        with MethodEvents() as observer:
            obs, reward, terminated, truncated, info = super().step(action)
        for event in observer.events:
            if event["type"] == "lose_hp" and event["scope"] == "Creature":
                self.hp_lost += max(0, event["hp_before"] - event["hp_after"])
            if event["type"] == "heal":
                amount = max(0, event["hp_after"] - event["hp_before"])
                if event["scope"] == "Creature":
                    self.combat_healing += amount
                elif event["scope"] == "PlayerState":
                    self.run_healing += amount
        return obs, reward, terminated, truncated, self._metrics(info)


def policy_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def prepare_output(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source artifact tree")
    output.mkdir(parents=True, exist_ok=False)
    return output


def verify_pair(control, curriculum, protocol="strict"):
    """Fail before interpreting runs that differ in more than HP distribution."""
    paths = [Path(control), Path(curriculum)]
    configs = [json.loads((p / "config.json").read_text(encoding="utf-8")) for p in paths]
    if [c["arm"] for c in configs] != ["control", "curriculum"]:
        raise ValueError("Pass control then curriculum")
    if configs[0]["training_hp"] != [80] or configs[1]["training_hp"] != list(HP_LEVELS):
        raise ValueError("Incorrect arm HP profiles")
    keys = ("seed", "source_checkpoint_sha256", "initial_policy_sha256", "source_timesteps",
            "hp_version", "n_envs", "n_steps", "batch_size", "n_epochs", "total_timesteps",
            "expected_additional_timesteps", "worker_seeds", "validation_seeds", "reserved_assessment_seeds",
            "observation_size", "action_size", "max_decisions", "max_combat_turns", "eval_freq",
            "checkpoint_freq", "inherited_hyperparameters", "source_code_sha256")
    if protocol not in ("strict", "final", "common-grid"):
        raise ValueError("Unknown comparison protocol")
    differences = [key for key in keys if configs[0][key] != configs[1][key]]
    allowed = set() if protocol == "strict" else {"eval_freq", "source_code_sha256"}
    # Explicit schedule-migration comparisons still require identical engine code.
    if protocol != "strict":
        for filename in ("run_native_combat.py", "run_env.py"):
            hashes = [[value for key, value in config["source_code_sha256"].items()
                       if key.replace("\\", "/").endswith("/" + filename)] for config in configs]
            if len(hashes[0]) != 1 or hashes[0] != hashes[1]:
                raise ValueError(f"Environment implementation differs: {filename}")
    if set(differences) - allowed:
        raise ValueError(f"Unmatched continuation configs: {differences}")
    completed = [(p / "completion.json").exists() for p in paths]
    if any(completed) and not all(completed):
        raise ValueError("Only one arm completed")
    if all(completed):
        for p, config in zip(paths, configs):
            completion = json.loads((p / "completion.json").read_text(encoding="utf-8"))
            if completion["actual_additional_timesteps"] != config["expected_additional_timesteps"]:
                raise ValueError("Completed arm has the wrong training budget")
            if not json.loads((p / "source_preservation.json").read_text())["unchanged"]:
                raise ValueError("Source artifact protection failed")
    return dict(matched=True, completed=all(completed), continuation_seed=configs[0]["seed"],
                comparison_protocol=protocol, reviewed_schedule_code_differences=differences,
                initial_policy_sha256=configs[0]["initial_policy_sha256"],
                additional_timesteps=configs[0]["expected_additional_timesteps"])


def load_continuation(checkpoint, env, seed, **overrides):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(checkpoint, env=env, seed=seed, device="cpu", **overrides)
    if (getattr(model, "combat_task_version", None) != TASK_VERSION
            or getattr(model, "combat_observation_schema", None) != "combat-prefix-217-v1"
            or getattr(model, "combat_action_schema", None) != "run-prefix-115-v1"):
        raise ValueError("Continuation requires a compatible run-native combat v1 checkpoint")
    if model.observation_space.shape != (217,) or model.action_space.n != 115:
        raise ValueError("Combat checkpoint interface mismatch")
    return model


def row_from_info(info, policy="MaskablePPO"):
    return dict(policy=policy, encounter_type="hallway", group=info["encounter_group"],
        scenario=info["scenario_id"], seed=info["episode_seed"], run_seed=info["run_seed"],
        starting_hp=info["starting_hp"], win=info["is_success"], reason=info["termination_reason"],
        timeout=info["timeout"], hp_lost=info["hp_lost"], combat_healing=info["combat_healing"],
        run_healing=info["run_healing"], final_hp=info["combat_hp"], final_run_hp=info["run_hp"],
        turns=info["turns"], decisions=info["step"])


def evaluate_hp_panel(model, seeds, *, directory, tag, scenarios=HALLWAY_SCENARIOS,
                      max_decisions=2000, max_combat_turns=200, hp_levels=HP_LEVELS):
    """Fixed balanced hallway x HP panel. Never implicitly uses assessment seeds."""
    from stable_baselines3.common.monitor import Monitor
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / f"{tag}{suffix}").exists() for suffix in (".json", ".monitor.csv")):
        raise FileExistsError(f"Evaluation artifacts already exist for {tag}")
    env = Monitor(HPHallwayEnv(max_decisions=max_decisions, max_combat_turns=max_combat_turns,
                              scenarios=scenarios), filename=str(directory / f"{tag}.monitor.csv"))
    rows = []
    try:
        for hp in hp_levels:
            for scenario in scenarios:
                for seed in seeds:
                    obs, _ = env.reset(seed=seed, options={"starting_hp": hp, "scenario_id": scenario.scenario_id})
                    while True:
                        mask = env.unwrapped.action_masks()
                        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        if not mask[int(action)]:
                            raise AssertionError("Evaluation action violates its mask")
                        obs, _, terminated, truncated, info = env.step(int(action))
                        if terminated or truncated:
                            rows.append(row_from_info(info))
                            break
    finally:
        env.close()
    by_type = summarize(rows, ("encounter_type", "starting_hp"))
    report = dict(seeds=list(seeds), hp_levels=hp_levels, episodes=rows,
        by_type_hp=by_type,
        by_group_hp=summarize(rows, ("group", "starting_hp")),
        by_encounter_hp=summarize(rows, ("encounter_type", "scenario", "starting_hp")),
        selection_metric="macro win rate over balanced hallway encounter x HP cells",
        macro_win_rate=float(np.mean([r["win_rate"] for r in by_type])))
    with (directory / f"{tag}.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report
