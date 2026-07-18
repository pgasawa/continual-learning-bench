#!/usr/bin/env python3
"""Aggregate final_results runs into normalized leaderboard data.

The script is intentionally file-format tolerant: run-all artifacts have enough
history in them that old task traces can coexist with newer summaries.  We
prefer per-run instance outcomes for reward/gain/error bars, and fall back to
summary headline values only when needed.

This module is used by scripts/generate_leaderboard.py, which is the supported
workflow for updating the website leaderboard.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.registry import get_task_class, list_tasks


TASK_ALIASES = {}

# Tasks omitted from leaderboard aggregates and task_summary / run_points (artifacts may still exist on disk).
TASKS_EXCLUDED_FROM_ANALYSIS = []

DEFAULT_RUN_NAMES = {
    "icl-gpt-5.4",
    "icl-gemini-3-flash",
    "ace-gpt-5.4",
    "icl-gemini-3.1-pro-preview",
    "mem0-gpt-5.4",
    "icl-notepad-gemini-3.1-pro-preview",
    "codex-gpt-5.4",
    "icl-claude-sonnet-4.6",
    "icl-notepad-gpt-5.4",
    "claude-code-sonnet-4.6",
    "icl-notepad-claude-sonnet-4-6",
    "icl-claude-opus-4.7",
}

POKER_VALID_COMPLETE_RUN_LIMITS: dict[str, int] = {}
POKER_REFERENCE_DIR = Path("src/tasks/exploitable_poker/references")


@dataclass
class RunPoint:
    run_name: str
    task: str
    run_index: int
    reward: float
    baseline_reward: float
    reference_reward: float
    gain: float
    normalized_reward: float | None
    normalized_gain: float | None
    cost_usd: float | None
    latency_seconds: float | None
    instance_count: int
    reward_curve: list[float] = field(default_factory=list)
    baseline_reward_curve: list[float] = field(default_factory=list)
    gain_curve: list[float] = field(default_factory=list)
    cost_curve: list[float] = field(default_factory=list)


@dataclass
class TaskSummary:
    run_name: str
    task: str
    n_runs: int
    n_instances: int
    reward_mean: float
    reward_se: float
    baseline_reward: float
    reference_reward: float
    gain_mean: float
    gain_se: float
    normalized_reward_mean: float | None
    normalized_reward_se: float | None
    normalized_gain_mean: float | None
    normalized_gain_se: float | None
    cost_mean: float | None
    cost_se: float | None
    latency_mean: float | None
    latency_se: float | None
    warnings: list[str] = field(default_factory=list)


def mean(xs: Iterable[float]) -> float:
    values = list(xs)
    return sum(values) / len(values) if values else float("nan")


def se(xs: Iterable[float]) -> float:
    values = list(xs)
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def clean_task_name(name: str) -> str:
    return TASK_ALIASES.get(name, name)


def load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def artifact_task_name(path: Path, artifact: dict[str, Any]) -> str:
    summary_name = ((artifact.get("summary") or {}).get("task") or {}).get("name")
    if summary_name:
        return clean_task_name(str(summary_name))
    name = path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return clean_task_name(name)


def discover_task_artifacts(
    run_dir: Path, results_dir: Path
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}

    combined = run_dir / "task_artifacts.json.gz"
    if combined.exists():
        data = load_json(combined)
        for key, artifact in data.items():
            if not isinstance(artifact, dict):
                continue
            artifacts[clean_task_name(str(key))] = artifact

    task_dir = run_dir / "tasks"
    if task_dir.is_dir():
        for path in sorted(
            list(task_dir.glob("*.json")) + list(task_dir.glob("*.json.gz"))
        ):
            artifact = load_json(path)
            artifacts[artifact_task_name(path, artifact)] = artifact

    # Some current artifacts intentionally keep only a subset expanded.  For the
    # fixed final run names, fall back to old expanded artifacts when the current
    # combined artifact has a null task entry.
    old_task_dir = results_dir / "old" / run_dir.name / "tasks"
    if old_task_dir.is_dir():
        for path in sorted(
            list(old_task_dir.glob("*.json")) + list(old_task_dir.glob("*.json.gz"))
        ):
            artifact = load_json(path)
            task = artifact_task_name(path, artifact)
            artifacts.setdefault(task, artifact)

    return artifacts


def usage_cost(execution: dict[str, Any] | None) -> float | None:
    if not execution:
        return None
    usage = execution.get("usage") or {}
    for key in ("interaction", "respond"):
        bucket = usage.get(key)
        if isinstance(bucket, dict) and isinstance(
            bucket.get("cost_usd"), (int, float)
        ):
            return float(bucket["cost_usd"])
    if isinstance(usage.get("cost_usd"), (int, float)):
        return float(usage["cost_usd"])
    return None


def execution_latency(execution: dict[str, Any] | None) -> float | None:
    if not execution:
        return None
    value = execution.get("avg_response_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def outcome_reward_sum(result: dict[str, Any]) -> tuple[float, int]:
    outcomes = result.get("instance_outcomes") or []
    total = 0.0
    for outcome in outcomes:
        reward = outcome.get("reward")
        if isinstance(reward, (int, float)):
            total += float(reward)
    return total, len(outcomes)


def database_max_queries(summary: dict[str, Any]) -> float:
    params = (summary.get("task") or {}).get("params") or {}
    value = params.get("max_queries_per_question")
    return float(value) if isinstance(value, (int, float)) and value > 0 else 15.0


def canonical_outcome_reward(
    task: str,
    outcome: dict[str, Any],
    *,
    database_budget: float,
) -> float | None:
    reward = outcome.get("reward")

    if task == "cohort_studies":
        # Cohort studies moved from a clamped relative skill score (reward = max(0, 1 -
        # mean_kl_divergence / mean_reference_kl)) to the signed information gain (reward =
        # mean_reference_kl - mean_kl_divergence, negatives legitimate). Recompute from the stored
        # cohort-mean KLs so old (clamped) and new (signed) artifacts compare on the same signed
        # scale -- otherwise runs scored under the two metrics are summed and normalized together
        # (and the fixed gpt-5.4 reference baseline is itself clamped), which distorts the ranking.
        # Mirrors the database_exploration recompute below.
        metadata = outcome.get("metadata") or {}
        ref_kl = metadata.get("mean_reference_kl")
        kl = metadata.get("mean_kl_divergence")
        if isinstance(ref_kl, (int, float)) and isinstance(kl, (int, float)):
            return float(ref_kl) - float(kl)
        return float(reward) if isinstance(reward, (int, float)) else None

    if task != "database_exploration":
        return float(reward) if isinstance(reward, (int, float)) else None

    # Database exploration moved from reward=-regret to a bounded post-processed
    # reward, max(0, 1 - regret / budget).  Recompute from raw regret so old and
    # new artifacts compare on the same 0..1 scale.
    regret = outcome.get("raw_metric_value")
    if not isinstance(regret, (int, float)):
        regret = (outcome.get("metadata") or {}).get("regret")
    if isinstance(regret, (int, float)):
        return max(0.0, 1.0 - float(regret) / database_budget)
    return float(reward) if isinstance(reward, (int, float)) else None


def canonical_reward_sum(
    task: str,
    result: dict[str, Any],
    *,
    database_budget: float,
) -> tuple[float, int]:
    outcomes = result.get("instance_outcomes") or []
    total = 0.0
    for outcome in outcomes:
        reward = canonical_outcome_reward(
            task,
            outcome,
            database_budget=database_budget,
        )
        if reward is not None:
            total += reward
    return total, len(outcomes)


def index_curves(
    task: str,
    result: dict[str, Any],
    baseline_result: dict[str, Any],
    *,
    database_budget: float,
) -> tuple[list[float], list[float], list[float]]:
    outcomes = result.get("instance_outcomes") or []
    baseline_outcomes = baseline_result.get("instance_outcomes") or []
    reward_curve: list[float] = []
    baseline_reward_curve: list[float] = []
    gain_curve: list[float] = []
    cost_curve: list[float] = []
    for i, outcome in enumerate(outcomes):
        reward = canonical_outcome_reward(
            task, outcome, database_budget=database_budget
        )
        reward_value = reward if reward is not None else 0.0
        baseline_value = 0.0
        if i < len(baseline_outcomes):
            baseline_reward = canonical_outcome_reward(
                task,
                baseline_outcomes[i],
                database_budget=database_budget,
            )
            baseline_value = baseline_reward if baseline_reward is not None else 0.0
        cost = outcome.get("cost_usd")
        reward_curve.append(reward_value)
        baseline_reward_curve.append(baseline_value)
        gain_curve.append(reward_value - baseline_value)
        cost_curve.append(float(cost) if isinstance(cost, (int, float)) else 0.0)
    return reward_curve, baseline_reward_curve, gain_curve, cost_curve


def expected_instances(summary: dict[str, Any]) -> int | None:
    params = (summary.get("task") or {}).get("params") or {}
    for key in ("expected_num_instances", "num_instances", "num_questions"):
        value = params.get(key)
        if isinstance(value, int):
            return value
    return None


def load_task_r_max_values() -> dict[str, float]:
    """Load task-owned per-instance r_max values from registered task classes."""

    references: dict[str, float] = {}
    for task_name in list_tasks():
        cls = get_task_class(task_name)
        value = getattr(cls, "r_max", None)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Task '{task_name}' must define numeric r_max.")
        references[clean_task_name(task_name)] = float(value)
    return references


def reference_reward(
    task: str,
    instance_count: int,
    result: dict[str, Any],
    r_max_values: dict[str, float],
) -> float:
    per_instance = r_max_values.get(task)
    if per_instance is not None:
        return per_instance * instance_count

    eval_metrics = result.get("eval_metrics") or {}
    value = eval_metrics.get("optimal_performance")
    if isinstance(value, (int, float)):
        return float(value)
    return float(instance_count)


def normalize(reward: float, baseline: float, reference: float) -> float | None:
    denom = reference - baseline
    if abs(denom) < 1e-12:
        return None
    return (reward - baseline) / denom


def collect_task_points(
    run_name: str,
    artifact: dict[str, Any],
    task_hint: str,
    reward_normalization_baseline: float | None,
    r_max_values: dict[str, float],
) -> tuple[list[RunPoint], list[str]]:
    summary = artifact.get("summary") or {}
    raw_task = ((summary.get("task") or {}).get("name")) or task_hint
    task = clean_task_name(str(raw_task))
    warnings: list[str] = []
    db_budget = database_max_queries(summary)

    baseline_result = (artifact.get("baseline_trace") or {}).get("result") or {}
    baseline_reward, baseline_instances = canonical_reward_sum(
        task,
        baseline_result,
        database_budget=db_budget,
    )
    expected = expected_instances(summary)
    if expected is not None and baseline_instances and baseline_instances != expected:
        warnings.append(
            f"baseline has {baseline_instances} instances but expected {expected}"
        )

    run_traces = artifact.get("run_traces") or []
    poker_complete_limit = (
        POKER_VALID_COMPLETE_RUN_LIMITS.get(run_name)
        if task == "exploitable_poker"
        else None
    )
    if poker_complete_limit is not None:
        warnings.append(
            f"using first {poker_complete_limit} complete poker rollouts for incomplete mini run"
        )

    points: list[RunPoint] = []
    for item in run_traces:
        run_index = int(item.get("run_index", len(points)))
        result = ((item.get("trace") or {}).get("result")) or {}
        reward, n_instances = canonical_reward_sum(
            task,
            result,
            database_budget=db_budget,
        )
        if expected is not None and n_instances != expected:
            warnings.append(
                f"excluded run {run_index}: {n_instances} instances but expected {expected}"
            )
            continue
        if n_instances == 0:
            warnings.append(f"excluded run {run_index}: no instance outcomes")
            continue

        ref = reference_reward(task, n_instances, result, r_max_values)
        fixed_reward_baseline = (
            reward_normalization_baseline
            if reward_normalization_baseline is not None
            else baseline_reward
        )
        gain = reward - baseline_reward
        normalized_reward = normalize(reward, fixed_reward_baseline, ref)
        normalized_gain = normalize(gain, 0.0, ref - baseline_reward)
        cost = usage_cost(
            (item.get("trace") or {}).get("execution") or result.get("execution")
        )
        latency = execution_latency(
            (item.get("trace") or {}).get("execution") or result.get("execution")
        )
        reward_curve, baseline_reward_curve, gain_curve, cost_curve = index_curves(
            task,
            result,
            baseline_result,
            database_budget=db_budget,
        )
        points.append(
            RunPoint(
                run_name=run_name,
                task=task,
                run_index=run_index,
                reward=reward,
                baseline_reward=baseline_reward,
                reference_reward=ref,
                gain=gain,
                normalized_reward=normalized_reward,
                normalized_gain=normalized_gain,
                cost_usd=cost,
                latency_seconds=latency,
                instance_count=n_instances,
                reward_curve=reward_curve,
                baseline_reward_curve=baseline_reward_curve,
                gain_curve=gain_curve,
                cost_curve=cost_curve,
            )
        )
        if poker_complete_limit is not None and len(points) >= poker_complete_limit:
            break

    return points, warnings


def artifact_baseline_reward(artifact: dict[str, Any], task_hint: str) -> float | None:
    summary = artifact.get("summary") or {}
    raw_task = ((summary.get("task") or {}).get("name")) or task_hint
    task = clean_task_name(str(raw_task))
    baseline_result = (artifact.get("baseline_trace") or {}).get("result") or {}
    reward, n_instances = canonical_reward_sum(
        task,
        baseline_result,
        database_budget=database_max_queries(summary),
    )
    return reward if n_instances else None


def summarize_task(points: list[RunPoint], warnings: list[str]) -> TaskSummary:
    first = points[0]
    reward_norms = [
        p.normalized_reward for p in points if p.normalized_reward is not None
    ]
    gain_norms = [p.normalized_gain for p in points if p.normalized_gain is not None]
    costs = [p.cost_usd for p in points if p.cost_usd is not None]
    lats = [p.latency_seconds for p in points if p.latency_seconds is not None]
    return TaskSummary(
        run_name=first.run_name,
        task=first.task,
        n_runs=len(points),
        n_instances=first.instance_count,
        reward_mean=mean(p.reward for p in points),
        reward_se=se(p.reward for p in points),
        baseline_reward=first.baseline_reward,
        reference_reward=first.reference_reward,
        gain_mean=mean(p.gain for p in points),
        gain_se=se(p.gain for p in points),
        normalized_reward_mean=mean(reward_norms) if reward_norms else None,
        normalized_reward_se=se(reward_norms) if reward_norms else None,
        normalized_gain_mean=mean(gain_norms) if gain_norms else None,
        normalized_gain_se=se(gain_norms) if gain_norms else None,
        cost_mean=mean(costs) if costs else None,
        cost_se=se(costs) if costs else None,
        latency_mean=mean(lats) if lats else None,
        latency_se=se(lats) if lats else None,
        warnings=warnings,
    )


def aggregate_leaderboard(task_summaries: list[TaskSummary]) -> list[dict[str, Any]]:
    by_run: dict[str, list[TaskSummary]] = {}
    for summary in task_summaries:
        by_run.setdefault(summary.run_name, []).append(summary)

    rows = []
    for run_name, summaries in sorted(by_run.items()):
        norm_values = [
            s.normalized_reward_mean
            for s in summaries
            if s.normalized_reward_mean is not None
        ]
        gain_values = [
            s.normalized_gain_mean
            for s in summaries
            if s.normalized_gain_mean is not None
        ]
        cost_values = [s.cost_mean for s in summaries if s.cost_mean is not None]
        rows.append(
            {
                "run_name": run_name,
                "tasks_included": len(norm_values),
                "normalized_reward_mean": mean(norm_values) if norm_values else None,
                "normalized_gain_mean": mean(gain_values) if gain_values else None,
                "total_cost_usd_mean": sum(cost_values) if cost_values else None,
                "mean_task_cost_usd": mean(cost_values) if cost_values else None,
            }
        )
    rows.sort(
        key=lambda r: (
            r["normalized_reward_mean"] is None,
            -(r["normalized_reward_mean"] or -1e9),
        )
    )
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def as_row(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        row = {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    else:
        row = dict(obj)
    if isinstance(row.get("warnings"), list):
        row["warnings"] = "; ".join(row["warnings"])
    return row


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_reference_overrides(reference_dir: Path) -> dict[str, float]:
    """Deprecated compatibility hook; task classes now own reference maxima."""

    _ = reference_dir
    return {}


def viewer_payload(
    task_summaries: list[TaskSummary],
    run_points: list[RunPoint],
    leaderboard: list[dict[str, Any]],
    warnings: dict[str, list[str]],
    reward_normalization_baselines: dict[str, float],
    r_max_values: dict[str, float],
) -> dict[str, Any]:
    return {
        "leaderboard": leaderboard,
        "task_summary": [as_row(s) for s in task_summaries],
        "run_points": [as_row(p) for p in run_points],
        "warnings": warnings,
        "reference_per_instance": r_max_values,
        "reward_normalization_baselines": reward_normalization_baselines,
        "poker_valid_complete_run_limits": POKER_VALID_COMPLETE_RUN_LIMITS,
        "normalization": {
            "reward": "(reward - stateless baseline of gpt-5.4) / (reference_max - stateless baseline of gpt-5.4)",
            "gain": "gain / (reference_max - stateless baseline of that system)",
            "reference_max": "Task-defined mean per-instance maximum reward from each task class's r_max.",
        },
    }


def build_analysis_payload(
    *,
    results_dir: Path = Path("final_results/runs"),
    runs: Iterable[str] | None = None,
    include_all_runs: bool = False,
    reference_dir: Path = POKER_REFERENCE_DIR,
) -> dict[str, Any]:
    task_summaries: list[TaskSummary] = []
    run_points: list[RunPoint] = []
    warnings: dict[str, list[str]] = {}
    skipped = frozenset(clean_task_name(t) for t in TASKS_EXCLUDED_FROM_ANALYSIS)
    selected_runs = set(runs or DEFAULT_RUN_NAMES)
    artifacts_by_run: dict[str, dict[str, dict[str, Any]]] = {}

    r_max_values = load_task_r_max_values()
    r_max_values.update(load_reference_overrides(reference_dir))

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "old":
            continue
        if not include_all_runs and run_dir.name not in selected_runs:
            continue
        artifacts = discover_task_artifacts(run_dir, results_dir)
        if not artifacts:
            continue
        artifacts_by_run[run_dir.name] = artifacts

    reward_normalization_baselines = {
        task: baseline
        for task, artifact in artifacts_by_run.get("icl-gpt-5.4", {}).items()
        if (baseline := artifact_baseline_reward(artifact, task)) is not None
    }

    for run_name, artifacts in sorted(artifacts_by_run.items()):
        for task_name, artifact in sorted(artifacts.items()):
            if clean_task_name(task_name) in skipped:
                continue
            points, task_warnings = collect_task_points(
                run_name,
                artifact,
                task_name,
                reward_normalization_baselines.get(task_name),
                r_max_values,
            )
            if task_warnings:
                warnings[f"{run_name}/{task_name}"] = task_warnings
            if not points:
                continue
            run_points.extend(points)
            task_summaries.append(summarize_task(points, task_warnings))

    leaderboard = aggregate_leaderboard(task_summaries)
    return viewer_payload(
        task_summaries,
        run_points,
        leaderboard,
        warnings,
        reward_normalization_baselines,
        r_max_values,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="final_results/runs", type=Path)
    parser.add_argument("--reference-dir", default=POKER_REFERENCE_DIR, type=Path)
    parser.add_argument(
        "--out", type=Path, help="Optional raw analysis JSON output path."
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Run directory name to include. Defaults to the finished public runs.",
    )
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Include every run directory with task artifacts instead of the public-run default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_analysis_payload(
        results_dir=args.results_dir,
        runs=args.runs,
        include_all_runs=args.include_all_runs,
        reference_dir=args.reference_dir,
    )
    if args.out:
        write_json(args.out, payload)
        print(f"Wrote leaderboard analysis payload to {args.out}")
    else:
        print("Built leaderboard analysis payload")
    for row in payload["leaderboard"]:
        score = row["normalized_reward_mean"]
        print(
            f"{row['rank']}. {row['run_name']}: normalized_reward={score:.4f} over {row['tasks_included']} tasks"
        )
    if payload["warnings"]:
        print("\nWarnings:")
        for key, values in payload["warnings"].items():
            for value in values:
                print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
