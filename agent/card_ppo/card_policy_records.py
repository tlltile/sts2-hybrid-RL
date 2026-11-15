"""Append-only card-policy decisions and terminal credit, without fabricated data."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from sts2_env.gym_env.card_selection_observation import canonical


@dataclass(frozen=True)
class PolicyMetadata:
    version: str
    log_probability: float
    value_estimate: float | None
    behavior_kind: str = "learned"

    def validate(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Explicit policy version required")
        if not math.isfinite(self.log_probability) or self.log_probability > 0:
            raise ValueError("Finite selected-action log probability <=0 required")
        if self.behavior_kind not in ("learned", "fixed"):
            raise ValueError("Unknown behavior kind")
        if self.value_estimate is None:
            if self.behavior_kind != "fixed":
                raise ValueError("Learned-policy decisions require an actual value estimate")
        elif not math.isfinite(self.value_estimate):
            raise ValueError("Nonfinite value estimate")


class CardPolicyJournal:
    """path=None is an in-memory unit-test journal; production uses a fresh file."""

    def __init__(self, path: Path | None = None):
        self.events: list[dict] = []
        self.stream = Path(path).open("x", encoding="utf-8") if path is not None else None
        self._runs: set[str] = set()
        self._decisions: set[str] = set()

    def append(self, kind: str, **record) -> None:
        if kind == "run_start" and record["run_id"] in self._runs:
            raise ValueError("Duplicate run ID; use unique worker/run prefixes")
        if kind == "decision" and record["decision_id"] in self._decisions:
            raise ValueError("Duplicate card decision ID")
        event = json.loads(canonical(dict(schema="card-policy-record-1", kind=kind, **record)))
        if self.stream is not None:
            self.stream.write(canonical(event)+"\n")
            self.stream.flush()
        self.events.append(event)
        if kind == "run_start":
            self._runs.add(record["run_id"])
        if kind == "decision":
            self._decisions.add(record["decision_id"])

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


def training_records(events: list[dict]) -> list[dict]:
    """Join exact pre-action records to transitions and terminal episode credit.

    Missing terminal receipts remain incomplete. Errors have no invented return.
    Returns are undiscounted terminal game rewards, not per-card causal effects.
    """
    for kind, key in (("run_start", "run_id"), ("decision", "decision_id"),
                      ("transition", "decision_id"), ("run_end", "run_id")):
        rows = [e for e in events if e["kind"] == kind]
        if len({e[key] for e in rows}) != len(rows):
            raise ValueError(f"Duplicate {kind} identity in credit ledger")
    transitions = {e["decision_id"]: e for e in events if e["kind"] == "transition"}
    endpoints = {e["run_id"]: e for e in events if e["kind"] == "run_end"}
    result = []
    for e in events:
        if e["kind"] != "decision":
            continue
        end = endpoints.get(e["run_id"])
        transition = transitions.get(e["decision_id"])
        outcome = end["outcome"] if end else "incomplete"
        valid = outcome in ("boss_clear", "death", "timeout", "run_failure") and transition is not None
        result.append(dict(e, transition=transition, terminal_outcome=outcome,
                           undiscounted_return=end["terminal_reward"] if valid else None,
                           training_eligible=valid and e["behavior_kind"] == "learned"))
    return result
