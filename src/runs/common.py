"""Shared helpers for the ``runs`` package.

Holds enums, light utilities, and the per-future result handlers used by
the baseline and rollout drain loops.  These handlers exist so the
sequential, parallel, and mixed (baseline+rollout in one pool) paths can
all funnel through the same accounting code.
"""

from __future__ import annotations

import inspect
import logging
import statistics
from concurrent.futures import as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    EvalMetrics,
    InstanceOutcome,
    TaskResult,
)
from ..tasks.environments import is_environment_backend_param
from ..trace_storage import _atomic_write_json

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    """Controls which sources of variance differ across repeated benchmark runs."""

    REPLICATE = "replicate"
    RESAMPLE = "resample"
    PERMUTE = "permute"


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def filter_init_params(
    cls: type[ContinualLearningTask] | type[ContinualLearningSystem],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return params accepted by ``cls.__init__``."""
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    valid_keys = set(sig.parameters.keys()) - {"self"}
    unsupported_environment_keys = [
        key
        for key in params
        if is_environment_backend_param(key) and key not in valid_keys
    ]
    if unsupported_environment_keys:
        raise ValueError(
            f"Task does not support {unsupported_environment_keys[0]}. Daytona is only available for shell/workspace tasks in this release."
        )
    return {k: v for k, v in params.items() if k in valid_keys}


def task_accepts_parallel_execution(
    task_class: type[ContinualLearningTask],
    system_class: type[ContinualLearningSystem],
) -> bool:
    """Return whether task and system may run in parallel worker processes."""
    return getattr(system_class, "parallel_safe", True) and getattr(
        task_class, "parallel_safe", True
    )


def get_task_num_instances(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
) -> int:
    """Instantiate a probe task and return its declared ``num_instances``."""
    probe_task = task_class(**filter_init_params(task_class, task_params))
    num_instances = getattr(probe_task, "num_instances", None)
    if not isinstance(num_instances, int) or num_instances <= 0:
        raise ValueError(
            f"{task_class.__name__} must expose a positive num_instances during __init__."
        )
    return num_instances


# ---------------------------------------------------------------------------
# Blocked-result builders
# ---------------------------------------------------------------------------


def build_blocked_task_result(
    refusal: ProviderRefusalError,
    completed_outcomes: list[InstanceOutcome],
) -> TaskResult:
    """Build a TaskResult representing a refusal-aborted run.

    The score is the mean reward over completed instances (or 0.0 when
    nothing completed); ``metrics`` carries a ``blocked`` flag so downstream
    consumers can recognise the run is partial without inspecting the trace.
    """
    summary = (
        f"Run blocked after {len(completed_outcomes)} instance(s): "
        f"{refusal.kind.replace('_', ' ')}"
    )
    return TaskResult(
        metrics={"blocked": True, "blocked_reason": refusal.kind},
        summary=summary,
        eval_metrics=EvalMetrics(
            loss_curve=[],
            optimal_performance=0.0,
            actual_performance=0.0,
        ),
        instance_outcomes=list(completed_outcomes),
    )


def serialize_blocked_baseline_instance(
    refusal: ProviderRefusalError, *, fallback_instance_index: int | None = None
) -> dict[str, Any]:
    """Serialize one blocked baseline instance for traces and summaries."""
    payload = refusal.to_record()
    payload["status"] = "blocked"
    if payload.get("instance_index") is None and fallback_instance_index is not None:
        payload["instance_index"] = fallback_instance_index
    return payload


def build_blocked_run_stub(
    refusal: ProviderRefusalError,
    *,
    run_index: int,
    run_group_id: str,
) -> tuple[TaskResult, dict[str, Any], dict[str, Any]]:
    """Build a (result, trace, execution_summary) triple for a no-progress refusal.

    Used when a ``ProviderRefusalError`` escapes ``run_single``'s inner trace
    recorder — i.e. the refusal happened during system construction or task
    setup, before any interaction was recorded.  We still want the run to
    appear in the summary instead of vanishing.
    """
    blocked_payload = refusal.to_record()
    stub_result = build_blocked_task_result(refusal, [])
    stub_execution: dict[str, Any] = {
        "status": "blocked",
        "phase": "run",
        "task_brief": None,
        "instance_outcomes": [],
        "blocked_reason": blocked_payload,
        "run_group_id": run_group_id,
        "run_index": run_index,
    }
    stub_trace: dict[str, Any] = {
        "status": "blocked",
        "phase": "run",
        "blocked_reason": blocked_payload,
        "execution": stub_execution,
        "interactions": [],
        "instance_outcomes": [],
    }
    return stub_result, stub_trace, stub_execution


# ---------------------------------------------------------------------------
# Per-result handlers (shared between sequential, parallel, and mixed paths)
# ---------------------------------------------------------------------------

BaselineSuccess = tuple[int, TaskResult, dict[str, Any]]
RunSuccess = tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]


def record_baseline_outcome(
    i: int,
    success: BaselineSuccess | None,
    refusal: ProviderRefusalError | None,
    *,
    num_instances: int,
    instance_results: list[Optional[BaselineSuccess]],
    blocked_instances: list[dict[str, Any]],
    completed_so_far: list[BaselineSuccess],
    on_progress: Callable[[list[BaselineSuccess]], None],
) -> None:
    """Apply one baseline outcome (success or refusal) to the shared state."""
    if success is not None:
        idx, result, trace_data = success
        instance_results[idx] = (idx, result, trace_data)
        completed_so_far.append((idx, result, trace_data))
        n_done = len(completed_so_far)
        running_mean = statistics.mean(r.score for _, r, _ in completed_so_far)
        print(
            f"  Baseline {n_done}/{num_instances} complete "
            f"(instance #{idx + 1}) - mean so far: {running_mean:.4f}"
        )
    else:
        assert refusal is not None
        blocked_instances.append(
            serialize_blocked_baseline_instance(refusal, fallback_instance_index=i)
        )
        n_done = len(completed_so_far)
        print(
            f"  Baseline {n_done + 1}/{num_instances} blocked "
            f"(instance #{i + 1}) - {refusal.kind.replace('_', ' ')}"
        )
    on_progress(completed_so_far)


def record_run_outcome(
    i: int,
    success: RunSuccess | None,
    refusal: ProviderRefusalError | None,
    *,
    runs: int,
    run_results: list[Optional[RunSuccess]],
    run_group_id: str,
    live_trace_path: Optional[Path] = None,
) -> None:
    """Apply one rollout outcome (success, blocked tuple, or bare refusal)."""
    if success is not None:
        run_index, result, trace_data, execution_summary = success
        assert run_index is not None
        run_results[run_index] = (run_index, result, trace_data, execution_summary)
        status = execution_summary.get("status", "completed")
        if status == "blocked":
            reason = (execution_summary.get("blocked_reason") or {}).get(
                "kind", "blocked"
            )
            print(
                f"  Run {run_index + 1}/{runs} blocked - "
                f"{reason.replace('_', ' ')} "
                f"(partial: {len(result.instance_outcomes)} instance(s))"
            )
        else:
            print(f"  Run {run_index + 1}/{runs} complete - score: {result.score:.4f}")
        return

    # Refusal that escaped ``run_single`` (e.g. system construction failed).
    assert refusal is not None
    logger.error(
        "Run %d/%d refused before trace setup: %s",
        i + 1,
        runs,
        refusal,
    )
    stub_result, stub_trace, stub_execution = build_blocked_run_stub(
        refusal, run_index=i, run_group_id=run_group_id
    )
    run_results[i] = (i, stub_result, stub_trace, stub_execution)
    if live_trace_path is not None:
        _atomic_write_json(live_trace_path, stub_trace)
    print(
        f"  Run {i + 1}/{runs} blocked - {refusal.kind.replace('_', ' ')} (no progress)"
    )


# ---------------------------------------------------------------------------
# Drain loops
# ---------------------------------------------------------------------------


def drain_baseline_futures(
    futures: dict[Any, int],
    *,
    num_instances: int,
    instance_results: list[Optional[BaselineSuccess]],
    blocked_instances: list[dict[str, Any]],
    on_progress: Callable[[list[BaselineSuccess]], None],
) -> None:
    """Iterate baseline futures, recording successes and refusals.

    Non-refusal exceptions are re-raised as ``RuntimeError`` so the caller
    can decide whether to abort.
    """
    completed_so_far: list[BaselineSuccess] = []
    for future in as_completed(futures):
        i = futures[future]
        try:
            success: BaselineSuccess | None = future.result()
            refusal: ProviderRefusalError | None = None
        except ProviderRefusalError as exc:
            success, refusal = None, exc
        except Exception as exc:
            logger.error(
                "Baseline instance %d/%d failed: %s",
                i + 1,
                num_instances,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Baseline instance {i + 1}/{num_instances} failed: {exc}"
            ) from exc
        record_baseline_outcome(
            i,
            success,
            refusal,
            num_instances=num_instances,
            instance_results=instance_results,
            blocked_instances=blocked_instances,
            completed_so_far=completed_so_far,
            on_progress=on_progress,
        )


def drain_run_futures(
    futures: dict[Any, int],
    *,
    runs: int,
    run_results: list[Optional[RunSuccess]],
    run_group_id: str,
    live_trace_paths: Optional[dict[int, Path]] = None,
) -> None:
    """Iterate rollout futures, recording successes, blocked tuples, and refusals."""
    for future in as_completed(futures):
        i = futures[future]
        try:
            success: RunSuccess | None = future.result()
            refusal: ProviderRefusalError | None = None
        except ProviderRefusalError as exc:
            success, refusal = None, exc
        except Exception as exc:
            logger.error("Run %d/%d failed: %s", i + 1, runs, exc, exc_info=True)
            raise RuntimeError(f"Run {i + 1}/{runs} failed: {exc}") from exc
        record_run_outcome(
            i,
            success,
            refusal,
            runs=runs,
            run_results=run_results,
            run_group_id=run_group_id,
            live_trace_path=(
                None if live_trace_paths is None else live_trace_paths.get(i)
            ),
        )
