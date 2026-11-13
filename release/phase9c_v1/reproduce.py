"""Recompute selected results from the compact release inputs (standard library only)."""
import csv
import hashlib
import json
from collections import Counter
from math import comb, isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / "data" / name).read_text(encoding="utf-8"))


def table(name):
    with (HERE / "data" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(rows, positive, negative, adjusted):
    scores = []
    for row in rows:
        a, b = int(row[positive]), int(row[negative])
        n = a + b
        p = min(1, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2**n)
        require(isclose(p, float(row["exact_mcnemar_p"])), "Exact paired p mismatch")
        scores.append(p)
    running = 0
    for i, j in enumerate(sorted(range(len(rows)), key=scores.__getitem__)):
        running = max(running, min(1, scores[j] * (len(rows) - i)))
        require(isclose(running, float(rows[j][adjusted])), "Holm adjustment mismatch")
    return scores


def bootstrap(histogram):
    n = sum(histogram.values())
    distribution = {0: 1}
    for _ in range(n):
        updated = Counter()
        for subtotal, count in distribution.items():
            for value, frequency in histogram.items():
                updated[subtotal + value] += count * frequency
        distribution = updated
    cumulative, limits = 0, []
    for subtotal, count in sorted(distribution.items()):
        cumulative += count
        while len(limits) < 2 and 40 * cumulative >= (1, 39)[len(limits)] * n**n:
            limits.append(subtotal / (3 * n))
    return limits


def reproduce():
    phase5 = read("phase5.json")
    phase6 = table("phase6r_panels.csv")
    final = [r for r in phase6 if r["checkpoint"] == "200704"]
    random = next(r for r in phase6 if r["arm"] == "random")
    require(len(final) == 6 and all((r["wins"], r["assigned"]) == ("192", "192") for r in final),
            "Controlled final panel totals")
    phase7 = table("phase7h_arms.csv")
    contrasts7 = table("phase7h_paired.csv")
    arms = {r["arm"]: int(r["boss_clears"]) for r in phase7}
    require(len(phase7) == 5 and all(int(r["n"]) == 100 for r in phase7)
            and len(contrasts7) == 6, "Independent assessment coverage")
    for r in contrasts7:
        a, b, both, neither = (int(r[k]) for k in ("gained", "lost", "both_clear", "both_fail"))
        require(a + b + both + neither == 100 and a + both == arms[r["root"]]
                and b + both == arms[r["comparator"]], "Independent paired marginal mismatch")
    p7 = paired(contrasts7, "gained", "lost", "holm6_adjusted_p")
    phase8 = table("phase8_root_metrics.csv")
    conditions8 = table("phase8_condition_metrics.csv")
    for r in conditions8:
        panel = [x for x in phase8 if x["condition"] == r["condition"] and x["update"] == r["update"]]
        require(len(panel) == 3 and all(sum(int(x[k]) for x in panel) == int(r[k])
                for k in ("cases", "boss_clears", "deaths", "timeouts", "card_decisions", "skips")),
                "Condition and root panel totals disagree")
    contrasts8 = table("phase8_update32_paired.csv")
    end = [r for r in conditions8 if r["update"] == "32"]
    clears = {r["condition"]: int(r["boss_clears"]) for r in end}
    for r in contrasts8:
        a, b, both, neither = (int(r[k]) for k in ("sparse128_only", "sparse64_only", "both_clear", "both_fail"))
        require(a + b + both + neither == 128 and a + both == int(r["sparse128_clears"])
                and b + both == int(r["sparse64_clears"]), "Development paired marginal mismatch")
    p8 = paired(contrasts8, "sparse128_only", "sparse64_only", "holm_p")
    histogram = {int(k): v for k, v in read("phase8_joint_histogram.json").items()}
    require(sum(histogram.values()) == 128 and len(contrasts8) == 3 and len(phase8) == 30
            and len(conditions8) == 10, "Development panel coverage")
    delta = (clears["Sparse128"] - clears["Sparse64"]) / 384
    require(sum(k * v for k, v in histogram.items()) / 384 == delta, "Joint origin mean mismatch")
    interval = bootstrap(histogram)
    joint = read("phase8_update32_joint.json")
    require([delta, *interval] == [joint["difference"], joint["conditional_ci_low"],
                                 joint["conditional_ci_high"]], "Joint interval mismatch")
    return dict(phase5=f"{phase5['act1_clears']}/{phase5['denominator']}",
                phase6r=dict(final="192/192 per root and arm", uniform=f"{random['wins']}/{random['assigned']}"),
                phase7h=dict(clears=arms, n_per_arm=100, exact_p=p7),
                phase8=dict(clears=clears, n_per_condition=384, difference=delta,
                            conditional_interval=interval, exact_p=p8))


def verify_manifest():
    manifest = json.loads((HERE / "artifact_manifest.json").read_text(encoding="utf-8"))
    root = HERE.parents[1]
    for name, expected in manifest["bundled_files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Unsafe manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
                and path.stat().st_size == expected["bytes"], f"Artifact mismatch: {name}")
    return len(manifest["bundled_files"])


if __name__ == "__main__":
    print(json.dumps(dict(verified_files=verify_manifest(), **reproduce()), indent=2, sort_keys=True))
