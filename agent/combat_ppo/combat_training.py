"""Seeded masked combat evaluation and per-encounter diagnostic reports."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from sts2_env.gym_env.run_native_combat import RunNativeCombatEnv,HALLWAY_SCENARIOS,TASK_VERSION


def evaluation_panel(scenarios=HALLWAY_SCENARIOS, seed_start=10000, repeats=2):
    if repeats<1:raise ValueError("repeats must be positive")
    return [(scenario.scenario_id,seed_start+repeat) for repeat in range(repeats) for scenario in scenarios]


class CombatEvaluationSchedule(gym.Wrapper):
    def __init__(self,env,panel):
        super().__init__(env)
        self.panel=tuple(panel)
        if not self.panel:raise ValueError("Empty evaluation panel")
        self.rewind_combat_evaluation()

    def rewind_combat_evaluation(self):
        self.index=0

    def reset(self,**kwargs):
        scenario,seed=self.panel[self.index%len(self.panel)];self.index+=1
        kwargs.update(seed=seed,options={"scenario_id":scenario})
        return self.env.reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()


def episode_row(info):
    return {key:info[key] for key in (
        "task_version","curriculum_version","episode_seed","run_seed","scenario_id",
        "encounter_group","step","turns","combat_hp","run_hp","combat_over","combat_won","is_success","outcome",
        "termination_reason","timeout","base_truncated")}


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["scenario_id"]].append(row)
    def stats(items):
        return dict(episodes=len(items),win_rate=float(np.mean([r["is_success"] for r in items])),
                    timeouts=sum(r["timeout"] for r in items),
                    mean_decisions=float(np.mean([r["step"] for r in items])),
                    mean_turns=float(np.mean([r["turns"] for r in items])),
                    mean_combat_hp=float(np.mean([r["combat_hp"] for r in items])))
    per={name:stats(items) for name,items in sorted(groups.items())}
    return dict(task_version=TASK_VERSION,overall=stats(rows),per_encounter=per,
                macro_win_rate=float(np.mean([v["win_rate"] for v in per.values()])),episodes=rows)


def evaluate_combat(model=None,*,scenarios=HALLWAY_SCENARIOS,seed_start=10000,repeats=2,
                    max_decisions=2000,max_combat_turns=200,report_path=None):
    panel=evaluation_panel(scenarios,seed_start,repeats)
    env=RunNativeCombatEnv(scenarios,max_decisions,max_combat_turns)
    if model is not None:
        if model.observation_space!=env.observation_space or model.action_space!=env.action_space:
            env.close()
            raise ValueError("Combat checkpoint interface mismatch")
        if getattr(model,"combat_task_version",None)!=TASK_VERSION:
            env.close()
            raise ValueError("Checkpoint missing/mismatched run-native combat task version")
    rng=np.random.RandomState(42)
    rows=[]
    try:
        for scenario,seed in panel:
            obs,info=env.reset(seed=seed,options={"scenario_id":scenario})
            while True:
                mask=env.action_masks()
                if model is None:action=int(rng.choice(np.flatnonzero(mask)))
                else:
                    action,_=model.predict(obs,action_masks=mask,deterministic=True)
                    action=int(action)
                obs,_,terminated,truncated,info=env.step(action)
                if terminated or truncated:break
            rows.append(episode_row(info))
    finally:env.close()
    report=summarize(rows)
    report.update(policy="Random Legal-Action" if model is None else "MaskablePPO",
                  seed_start=seed_start,repeats=repeats,max_decisions=max_decisions,
                  max_combat_turns=max_combat_turns)
    if report_path:
        with Path(report_path).open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2);stream.write("\n")
    return report
