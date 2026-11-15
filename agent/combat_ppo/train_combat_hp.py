"""Matched native-combat HP continuations; assessment is a separate explicit mode."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from sts2_env.core.rng import INT_MAX_EXCLUSIVE
from sts2_env.gym_env.hp_curriculum import (
    ASSESSMENT_SEEDS, ASSESSMENT_HP_LEVELS, VALIDATION_SEEDS, HP_VERSION, HP_LEVELS,
    HPHallwayEnv, ContinuationWorker, worker_seed,
    evaluate_hp_panel, load_continuation, policy_digest, prepare_output,
    rounded_budget, source_hashes, verify_pair,
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def assert_panel_separation(source):
    seen = set(range(10000, 10100)) | set(range(200000, 200020))
    for path in source.rglob("*.monitor.csv"):
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(line for line in stream if not line.startswith("#")):
                for name in ("episode_seed", "run_seed"):
                    if row.get(name): seen.add(int(row[name]))
    panels = []
    for seeds in (VALIDATION_SEEDS, ASSESSMENT_SEEDS):
        expanded = set(seeds) | {int(np.random.default_rng(s).integers(0, INT_MAX_EXCLUSIVE)) for s in seeds}
        if expanded & seen:
            raise ValueError("Reserved panel overlaps recorded source or earlier evaluation seeds")
        panels.append(expanded)
    if panels[0] & panels[1]:
        raise ValueError("Validation and assessment panels overlap")


def make_hp_env(arm, monitor_path, max_decisions, max_turns, seed):
    def factory():
        from stable_baselines3.common.monitor import Monitor
        reserved = set(VALIDATION_SEEDS + ASSESSMENT_SEEDS)
        reserved.update(int(np.random.default_rng(s).integers(0, INT_MAX_EXCLUSIVE)) for s in tuple(reserved))
        return Monitor(ContinuationWorker(HPHallwayEnv(arm=arm, max_decisions=max_decisions,
            max_combat_turns=max_turns, forbidden_seeds=reserved), seed),
            filename=str(monitor_path), info_keywords=("episode_seed", "run_seed", "scenario_id",
                "starting_hp", "hp_sample_index", "is_success", "termination_reason", "timeout",
                "hp_lost", "combat_healing", "run_healing", "combat_hp", "run_hp", "turns"))
    return factory


def make_callback(output, start_steps, eval_freq, checkpoint_freq, validation_seeds, limits,
                  eval_workers=4, eval_inference="scalar"):
    from stable_baselines3.common.callbacks import BaseCallback

    class HPCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = eval_freq
            self.next_checkpoint = checkpoint_freq
            self.best_score = -1.0
            self.best_path = None
            self.best_steps = None
            self.evaluator = None
            self.validation_seconds = self.checkpoint_seconds = 0.0

        def close(self):
            if self.evaluator is not None:
                self.evaluator.close()
                self.evaluator = None

        def evaluate(self, steps, suffix=""):
            start = perf_counter()
            if eval_workers:
                if self.evaluator is None:
                    from sts2_env.gym_env.hp_parallel_evaluation import ParallelHPEvaluator
                    self.evaluator = ParallelHPEvaluator(eval_workers, eval_inference, **limits)
                report = self.evaluator.evaluate(self.model, validation_seeds,
                    directory=output / "validation", tag=f"validation_{steps}{suffix}")
            else:
                report = evaluate_hp_panel(self.model, validation_seeds,
                    directory=output / "validation", tag=f"validation_{steps}{suffix}", **limits)
            elapsed = perf_counter() - start
            self.validation_seconds += elapsed
            self.logger.record("runtime/validation_seconds", elapsed)
            score = report["macro_win_rate"]
            for row in report["by_type_hp"]:
                for key in ("win_rate", "mean_hp_lost", "mean_final_hp", "mean_turns", "mean_decisions", "timeout_rate"):
                    self.logger.record(f"validation/hp_{row['starting_hp']}/{key}", row[key])
            self.logger.record("validation/macro_win_rate", score)
            self.logger.dump(self.num_timesteps)
            if score > self.best_score:
                self.best_score = score
                self.best_steps = steps
                self.best_path = output / "selected" / f"best_at_{steps}{suffix}.zip"
                self.model.save(self.best_path)
            print(f"Validation at +{steps}{suffix}: macro win rate={score:.4f}", flush=True)
            return report

        def _on_step(self):
            steps = self.num_timesteps - start_steps
            if steps >= self.next_checkpoint:
                before = perf_counter()
                self.model.save(output / "checkpoints" / f"continuation_{steps}.zip")
                self.checkpoint_seconds += perf_counter() - before
                while self.next_checkpoint <= steps:
                    self.next_checkpoint += checkpoint_freq
            if steps >= self.next_eval:
                self.evaluate(steps)
                while self.next_eval <= steps:
                    self.next_eval += eval_freq
            return True

    return HPCallback()


def run(args):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    torch.set_num_threads(1)
    source = Path(args.source).resolve()
    checkpoint = source / "final_model.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    assert_panel_separation(source)
    before = source_hashes(source)
    output = prepare_output(args.output_dir, source)
    for name in ("train_logs", "checkpoints", "selected"):
        (output / name).mkdir()
    env = callback = None
    try:
        limits = dict(max_decisions=args.max_decisions, max_combat_turns=args.max_combat_turns)
        if args.evaluate_checkpoint:
            # No training; opening the reserved assessment panel is deliberate.
            model = load_continuation(args.evaluate_checkpoint, None, args.seed)
            seeds = ASSESSMENT_SEEDS if args.panel == "assessment" else VALIDATION_SEEDS
            if args.eval_workers:
                from sts2_env.gym_env.hp_parallel_evaluation import ParallelHPEvaluator
                evaluator = ParallelHPEvaluator(args.eval_workers, args.eval_inference, **limits)
                try:
                    evaluator.evaluate(model, seeds, directory=output / args.panel, tag=args.panel,
                        hp_levels=ASSESSMENT_HP_LEVELS if args.panel == "assessment" else HP_LEVELS)
                finally: evaluator.close()
            else:
                evaluate_hp_panel(model, seeds, directory=output / args.panel, tag=args.panel,
                    hp_levels=ASSESSMENT_HP_LEVELS if args.panel == "assessment" else HP_LEVELS, **limits)
            write_json(output / "evaluation_manifest.json", dict(checkpoint=args.evaluate_checkpoint,
                checkpoint_sha256=hashlib.sha256(Path(args.evaluate_checkpoint).read_bytes()).hexdigest(),
                panel=args.panel, seeds=seeds, trained=False))
            return
        # Always isolate training simulators, even with one worker: manual
        # validation resets process-global card instance counters in the parent.
        env = SubprocVecEnv([make_hp_env(args.arm, output / "train_logs" / f"worker_{i}",
                    args.max_decisions, args.max_combat_turns, worker_seed(args.seed, i)) for i in range(args.n_envs)])
        model = load_continuation(checkpoint, env, args.seed, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=args.n_epochs,
            tensorboard_log=str(output / "tb_logs"), verbose=1)
        start_steps = model.num_timesteps
        initial_digest = policy_digest(model)
        # A second seed-independent load verifies the initial trainable tensors.
        reference = load_continuation(checkpoint, None, args.seed)
        if initial_digest != policy_digest(reference):
            raise AssertionError("Continuation initial weights differ from the source")
        del reference
        # Reference loading reseeds globals; restore the exact continuation seed.
        model.set_random_seed(args.seed)
        # Source rolling episode statistics describe the old distribution.
        # Clear logging buffers equally; keep all optimizer tensors/counters.
        model.ep_info_buffer = model.ep_success_buffer = None
        model.combat_curriculum_version = f"{HP_VERSION}:{args.arm}"
        validation_seeds = VALIDATION_SEEDS[:args.validation_seeds]
        expected = rounded_budget(args.total_timesteps, args.n_steps, args.n_envs)
        config = dict(vars(args), hp_version=HP_VERSION, source_checkpoint=str(checkpoint),
            source_checkpoint_sha256=before["final_model.zip"], initial_policy_sha256=initial_digest,
            source_timesteps=start_steps, expected_additional_timesteps=expected,
            worker_seeds=[worker_seed(args.seed, i) for i in range(args.n_envs)],
            expected_final_timesteps=start_steps + expected,
            training_hp=[80] if args.arm == "control" else HP_LEVELS,
            validation_seeds=validation_seeds, reserved_assessment_seeds=ASSESSMENT_SEEDS,
            assessment_used=False, observation_size=217, action_size=115,
            optimizer_state="restored from source in both arms; fresh matched RNG and environment",
            reset_num_timesteps=False, model_selection="strict macro validation win rate; earliest tie",
            inherited_hyperparameters=dict(learning_rate=model.learning_rate, gamma=model.gamma,
                ent_coef=model.ent_coef, gae_lambda=model.gae_lambda, max_grad_norm=model.max_grad_norm),
            source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path(__file__), Path("sts2_env/gym_env/hp_curriculum.py"),
                Path("sts2_env/gym_env/hp_parallel_evaluation.py"),
                Path("sts2_env/gym_env/run_native_combat.py"), Path("sts2_env/gym_env/run_env.py"))})
        write_json(output / "config.json", config)
        callback = make_callback(output, start_steps, args.eval_freq, args.checkpoint_freq,
                                 validation_seeds, limits, args.eval_workers, args.eval_inference)
        before_learn = perf_counter()
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False,
                    callback=callback, tb_log_name="hp_continuation")
        learn_seconds = perf_counter() - before_learn
        validation_during_learn = callback.validation_seconds
        actual = model.num_timesteps - start_steps
        if actual != expected:
            raise AssertionError(f"Unequal budget: expected {expected}, got {actual}")
        model.save(output / "final_model.zip")
        callback.evaluate(actual, "_final")  # Post-update final policy, not pre-update callback state.
        with callback.best_path.open("rb") as src, (output / "best_model.zip").open("xb") as dst:
            dst.write(src.read())
        write_json(output / "completion.json", dict(actual_additional_timesteps=actual,
            final_timesteps=model.num_timesteps, best_steps=callback.best_steps,
            best_validation_macro_win_rate=callback.best_score, selected_checkpoint=str(callback.best_path),
            assessment_used=False, learn_seconds=learn_seconds,
            validation_during_learn_seconds=validation_during_learn,
            training_logging_other_seconds=learn_seconds-validation_during_learn-callback.checkpoint_seconds,
            intermediate_checkpoint_seconds=callback.checkpoint_seconds,
            all_validation_seconds=callback.validation_seconds))
    finally:
        if callback is not None: callback.close()
        if env is not None: env.close()
        unchanged = before == source_hashes(source)
        write_json(output / "source_preservation.json", dict(unchanged=unchanged, source_sha256=before))
        if not unchanged:
            raise AssertionError("Source training artifacts changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="output/combat_hallway_100k_seed0_29259_11153")
    parser.add_argument("--arm", choices=("control", "curriculum"), default="control")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=25000,
                        help="Validation interval in additional steps; independent of checkpoints")
    parser.add_argument("--eval-workers", type=int, default=4, help="Persistent evaluation workers; 0 = legacy serial")
    parser.add_argument("--eval-inference", choices=("scalar", "batch"), default="scalar")
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--validation-seeds", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--max-combat-turns", type=int, default=200)
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-pair", nargs=2, metavar=("CONTROL_DIR", "CURRICULUM_DIR"),
                        help="Read-only matched-run and completed-budget check; no training")
    parser.add_argument("--comparison-protocol", choices=("strict", "final", "common-grid"), default="strict",
                        help="For pair checks: explicit schedule migration; canonical revalidation is required")
    parser.add_argument("--evaluate-checkpoint", help="Frozen evaluation only; no learning")
    parser.add_argument("--panel", choices=("validation", "assessment"), default="validation")
    args = parser.parse_args()
    if args.verify_pair:
        print(json.dumps(verify_pair(*args.verify_pair, protocol=args.comparison_protocol), indent=2))
        return
    if not args.output_dir:
        parser.error("--output-dir is required for training/evaluation")
    positive = ("total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs", "eval_freq",
                "checkpoint_freq", "validation_seeds", "max_decisions", "max_combat_turns")
    if args.seed < 0 or any(getattr(args, key) < 1 for key in positive):
        parser.error("Seed must be nonnegative; counts must be positive")
    if args.validation_seeds > len(VALIDATION_SEEDS):
        parser.error("Validation panel has exactly 20 reserved seeds")
    if args.eval_workers < 0:
        parser.error("--eval-workers must be nonnegative")
    if args.panel == "assessment" and not args.evaluate_checkpoint:
        parser.error("Assessment requires explicit --evaluate-checkpoint; never used during training")
    if args.batch_size > args.n_steps * args.n_envs:
        parser.error("Batch size cannot exceed rollout size")
    run(args)


if __name__ == "__main__":
    main()
