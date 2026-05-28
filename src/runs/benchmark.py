"""Multi-run benchmark orchestration.

``run_benchmark_runs`` runs N rollouts only.  ``run_benchmark`` is the
unified entry point that runs the baseline and the rollouts together —
when both task and system are ``parallel_safe`` it submits everything to
a single ``ProcessPoolExecutor`` so baseline instances and rollout runs
overlap in wall-clock time.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    EvalMetrics,
    InstanceOutcome,
    TaskResult,
    TaskResults,
)
from ..logging_utils import bind_logging_context
from ..run_ids import new_timestamp_run_id
from ..trace_storage import _atomic_write_json
from .common import (
    BaselineSuccess,
    RunMode,
    RunSuccess,
    drain_run_futures,
    get_task_num_instances,
    record_baseline_outcome,
    record_run_outcome,
    task_accepts_parallel_execution,
)
from .baseline import (
    _merge_baseline_instance_results,
    _run_baseline_instance,
    run_baseline,
)
from .single import run_single

logger = logging.getLogger(__name__)


def _live_run_path(task_name: str, run_group_id: str, run_index: int) -> Path:
    return Path(f"results/{task_name}/live/{run_group_id}/run_{run_index + 1}.json")


def _live_baseline_path(task_name: str, run_group_id: str) -> Path:
    return Path(f"results/{task_name}/live/{run_group_id}/baseline.json")


def _try_load_completed_baseline(
    task_name: str, run_group_id: str
) -> Optional[tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]]:
    # Return the saved baseline trace tuple, or None if it's missing or partial.
    path = _live_baseline_path(task_name, run_group_id)
    if not path.is_file():
        return None
    try:
        trace_data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not parse %s for resume: %s", path, exc)
        return None
    if trace_data.get("status") != "completed":
        return None
    # status="completed" can be set even when instances are blocked, so verify
    # the count too before treating the baseline as reusable.
    execution = trace_data.get("execution") or {}
    completed = execution.get("instances_completed")
    total = execution.get("instances_total")
    if (
        not isinstance(completed, int)
        or not isinstance(total, int)
        or completed < total
    ):
        logger.info(
            "resume.baseline_incomplete",
            extra={
                "path": str(path),
                "completed": completed,
                "total": total,
            },
        )
        return None
    result_payload = trace_data.get("result")
    if not isinstance(result_payload, dict):
        return None

    em = result_payload.get("eval_metrics") or {}
    eval_metrics = EvalMetrics(
        loss_curve=em.get("loss_curve", []),
        optimal_performance=em.get("optimal_performance", 0.0),
        actual_performance=em.get("actual_performance", 0.0),
        extra=em.get("extra"),
    )
    instance_outcome_fields = set(InstanceOutcome.__dataclass_fields__)
    instance_outcomes = [
        InstanceOutcome(**{k: v for k, v in io.items() if k in instance_outcome_fields})
        for io in (result_payload.get("instance_outcomes") or [])
        if isinstance(io, dict)
    ]
    task_result = TaskResult(
        metrics=result_payload.get("metrics") or {},
        summary=result_payload.get("summary", ""),
        eval_metrics=eval_metrics,
        instance_outcomes=instance_outcomes,
    )
    execution_summary = {
        **execution,
        "status": trace_data.get("status", "completed"),
        "phase": trace_data.get("phase", "baseline"),
        "task_brief": trace_data.get("task_brief"),
        "instance_outcomes": trace_data.get("instance_outcomes", []),
    }
    logger.info(
        "resume.skip_completed_baseline",
        extra={
            "path": str(path),
            "score": task_result.score,
            "instances": completed,
        },
    )
    return (None, task_result, trace_data, execution_summary)


def _try_load_completed_run(
    task_name: str, run_group_id: str, run_index: int
) -> Optional[RunSuccess]:
    # Return a RunSuccess from the saved live trace, or None if not reusable.
    # Only completed runs are eligible; the model state of partial runs is gone.
    path = _live_run_path(task_name, run_group_id, run_index)
    if not path.is_file():
        return None
    try:
        trace_data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not parse %s for resume: %s", path, exc)
        return None
    if trace_data.get("status") != "completed":
        return None
    result_payload = trace_data.get("result")
    if not isinstance(result_payload, dict):
        return None

    em = result_payload.get("eval_metrics") or {}
    eval_metrics = EvalMetrics(
        loss_curve=em.get("loss_curve", []),
        optimal_performance=em.get("optimal_performance", 0.0),
        actual_performance=em.get("actual_performance", 0.0),
        extra=em.get("extra"),
    )
    instance_outcome_fields = set(InstanceOutcome.__dataclass_fields__)
    instance_outcomes = [
        InstanceOutcome(**{k: v for k, v in io.items() if k in instance_outcome_fields})
        for io in (result_payload.get("instance_outcomes") or [])
        if isinstance(io, dict)
    ]
    task_result = TaskResult(
        metrics=result_payload.get("metrics") or {},
        summary=result_payload.get("summary", ""),
        eval_metrics=eval_metrics,
        instance_outcomes=instance_outcomes,
    )

    # Mirrors the execution_summary shape that run_single builds in memory.
    execution_summary = {
        **(trace_data.get("execution") or {}),
        "status": trace_data.get("status", "completed"),
        "phase": trace_data.get("phase", "run"),
        "task_brief": trace_data.get("task_brief"),
        "instance_outcomes": trace_data.get("instance_outcomes", []),
    }
    logger.info(
        "resume.skip_completed_run",
        extra={
            "run_index": run_index,
            "path": str(path),
            "score": task_result.score,
        },
    )
    return (run_index, task_result, trace_data, execution_summary)


def derive_run_task_params(
    task_params: dict[str, Any],
    run_mode: RunMode,
    run_index: int,
) -> dict[str, Any]:
    """Return per-run task params per *run_mode*."""
    run_task_params = {**task_params}
    if run_mode is RunMode.REPLICATE:
        return run_task_params
    if run_mode in (RunMode.RESAMPLE, RunMode.PERMUTE):
        run_task_params["run_index"] = run_index
        run_task_params["rollout_index"] = run_index
    return run_task_params


def run_benchmark_runs(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    runs: int,
    max_workers: int,
    system_name: str,
    task_name: str,
    run_group_id: Optional[str] = None,
    output_path: Optional[Path] = None,
    run_mode: RunMode = RunMode.PERMUTE,
    verbose_runs: bool = False,
    live_trace_paths: Optional[dict[int, Path]] = None,
) -> tuple[TaskResults, list[dict[str, Any]]]:
    """Orchestrate one or more independent benchmark runs, optionally in parallel."""
    if run_group_id is None:
        run_group_id = new_timestamp_run_id()

    can_parallelize = task_accepts_parallel_execution(task_class, system_class)
    run_results: list[Optional[RunSuccess]] = [None] * runs
    effective_workers = min(max_workers, runs) if can_parallelize else 1

    print(
        f"Starting {runs} run(s) with max workers {effective_workers} "
        f"[group {run_group_id}, mode={run_mode.value}]"
    )
    logger.info(
        "runs.started",
        extra={
            "run_group_id": run_group_id,
            "task": task_name,
            "system": system_name,
            "runs": runs,
            "max_workers": max_workers,
            "effective_workers": effective_workers,
            "parallel_safe": can_parallelize,
            "run_mode": run_mode.value,
        },
    )

    if runs == 1 or effective_workers == 1:
        for i in range(runs):
            try:
                with bind_logging_context(run_index=i, phase="run"):
                    logger.info("run.submitted")
                    outcome: RunSuccess | None = run_single(
                        task_class=task_class,
                        task_params=derive_run_task_params(task_params, run_mode, i),
                        system_class=system_class,
                        system_params=system_params,
                        run_index=i,
                        run_group_id=run_group_id,
                        system_name=system_name,
                        task_name=task_name,
                        output_path=output_path,
                        show_progress=(runs == 1),
                        verbose_runs=verbose_runs,
                        live_trace_path=None
                        if live_trace_paths is None
                        else live_trace_paths.get(i),
                    )
                refusal: ProviderRefusalError | None = None
            except ProviderRefusalError as exc:
                outcome, refusal = None, exc
            except Exception as exc:
                logger.error(
                    "Run %d/%d failed: %s",
                    i + 1,
                    runs,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"Run {i + 1}/{runs} failed: {exc}") from exc
            record_run_outcome(
                i,
                outcome,
                refusal,
                runs=runs,
                run_results=run_results,
                run_group_id=run_group_id,
                live_trace_path=(
                    None if live_trace_paths is None else live_trace_paths.get(i)
                ),
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            logger.info(
                "run_pool.started",
                extra={"effective_workers": effective_workers},
            )
            futures = {
                pool.submit(
                    run_single,
                    task_class=task_class,
                    task_params=derive_run_task_params(task_params, run_mode, i),
                    system_class=system_class,
                    system_params=system_params,
                    run_index=i,
                    run_group_id=run_group_id,
                    system_name=system_name,
                    task_name=task_name,
                    verbose_runs=verbose_runs,
                    live_trace_path=(
                        None if live_trace_paths is None else live_trace_paths.get(i)
                    ),
                ): i
                for i in range(runs)
            }

            if verbose_runs:
                for i in range(runs):
                    print(f"  Run {i + 1}/{runs} starting")

            drain_run_futures(
                futures,
                runs=runs,
                run_results=run_results,
                run_group_id=run_group_id,
                live_trace_paths=live_trace_paths,
            )

    task_results_list: list[TaskResult] = []
    trace_data_list: list[dict[str, Any]] = []
    execution_summaries: list[dict[str, Any]] = []
    for entry in run_results:
        assert entry is not None
        _, result, trace_data, execution_summary = entry
        task_results_list.append(result)
        trace_data_list.append(trace_data)
        execution_summaries.append(execution_summary)

    task_results = TaskResults(
        run_group_id=run_group_id,
        results=task_results_list,
        execution_summaries=execution_summaries,
    )
    logger.info(
        "runs.finished",
        extra={"runs": len(task_results_list), "run_group_id": run_group_id},
    )

    return task_results, trace_data_list


def run_benchmark(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    runs: int,
    max_workers: int,
    system_name: str,
    task_name: str,
    run_group_id: Optional[str] = None,
    run_mode: "RunMode" = None,  # type: ignore[assignment]
    verbose_runs: bool = False,
    include_baseline: bool = True,
    baseline_task_params: Optional[dict[str, Any]] = None,
    baseline_live_path: Optional[Path] = None,
    live_trace_paths: Optional[dict[int, Path]] = None,
    output_path: Optional[Path] = None,
    resume: bool = False,
) -> tuple[
    Optional[tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]],
    TaskResults,
    list[dict[str, Any]],
]:
    """Unified benchmark entry point: baseline (optional) + rollout runs.

    When both classes are ``parallel_safe`` the baseline instances and all
    rollout runs are submitted to a single ``ProcessPoolExecutor`` so they
    overlap in wall-clock time.  Otherwise the baseline runs first (sequential
    or parallel-instances depending on task support) and the rollout runs
    follow.
    """
    if run_mode is None:
        run_mode = RunMode.PERMUTE
    if run_group_id is None:
        run_group_id = new_timestamp_run_id()

    can_parallelize = task_accepts_parallel_execution(task_class, system_class)
    logger.info(
        "started",
        extra={
            "run_group_id": run_group_id,
            "task": task_name,
            "system": system_name,
            "runs": runs,
            "max_workers": max_workers,
            "include_baseline": include_baseline,
            "parallel_safe": can_parallelize,
            "run_mode": run_mode.value,
        },
    )

    # ── Sequential baseline then parallel runs (safe fallback) ──────────────
    if not include_baseline or not can_parallelize:
        baseline_info = None
        if include_baseline:
            logger.info("phase.baseline.started")
            baseline_info = run_baseline(
                task_class=task_class,
                task_params=task_params,
                system_class=system_class,
                system_params=system_params,
                max_workers=max_workers,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                baseline_task_params=baseline_task_params,
                live_trace_path=baseline_live_path,
                verbose=verbose_runs,
            )

        logger.info("phase.rollout.started")
        task_results, run_trace_data = run_benchmark_runs(
            task_class=task_class,
            task_params=task_params,
            system_class=system_class,
            system_params=system_params,
            runs=runs,
            max_workers=max_workers,
            system_name=system_name,
            task_name=task_name,
            run_group_id=run_group_id,
            run_mode=run_mode,
            verbose_runs=verbose_runs,
            live_trace_paths=live_trace_paths,
            output_path=output_path,
        )
        logger.info("finished", extra={"path": "sequential"})
        return baseline_info, task_results, run_trace_data

    # ── Fully parallel: baseline instances + rollout runs in one pool ───────
    merged_baseline_params = {**task_params, **(baseline_task_params or {})}
    num_baseline_instances = get_task_num_instances(task_class, merged_baseline_params)
    total_jobs = num_baseline_instances + runs
    effective_workers = min(max_workers, total_jobs)
    print(
        f"Starting {runs} run(s) + {num_baseline_instances}-instance baseline "
        f"with {effective_workers} parallel workers "
        f"[group {run_group_id}, mode={run_mode.value}]"
    )
    logger.info(
        "pool.started",
        extra={
            "runs": runs,
            "baseline_instances": num_baseline_instances,
            "effective_workers": effective_workers,
        },
    )

    baseline_instance_results: list[Optional[BaselineSuccess]] = [
        None
    ] * num_baseline_instances
    blocked_baseline_instances: list[dict[str, Any]] = []
    run_results: list[Optional[RunSuccess]] = [None] * runs

    # When resume=True, reuse completed rollout traces and a completed baseline
    # trace from the live dir instead of redoing them.
    resumed_indices: list[int] = []
    preloaded_baseline_info: Optional[
        tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]
    ] = None
    if resume:
        for i in range(runs):
            preloaded = _try_load_completed_run(task_name, run_group_id, i)
            if preloaded is not None:
                run_results[i] = preloaded
                resumed_indices.append(i)
        if resumed_indices:
            print(
                f"  Resuming: {len(resumed_indices)} of {runs} rollout(s) already "
                f"completed (indices {resumed_indices}); skipping their submission."
            )
        preloaded_baseline_info = _try_load_completed_baseline(task_name, run_group_id)
        if preloaded_baseline_info is not None:
            print(
                f"  Resuming: baseline already complete "
                f"(score {preloaded_baseline_info[1].score:.4f}); "
                f"skipping baseline submission."
            )

    baseline_start_time = datetime.now().isoformat()

    def _write_partial_baseline_snapshot(
        completed_results: list[BaselineSuccess],
    ) -> None:
        if baseline_live_path is None:
            return
        _, partial_trace, _ = _merge_baseline_instance_results(
            completed_results,
            run_group_id=run_group_id,
            system_name=system_name,
            task_name=task_name,
            task_params=merged_baseline_params,
            system_params=system_params,
            start_time=baseline_start_time,
            end_time=datetime.now().isoformat(),
            blocked_instances=blocked_baseline_instances,
            expected_num_instances=num_baseline_instances,
        )
        _atomic_write_json(baseline_live_path, partial_trace)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
        # Submit rollouts first — they're the sequential bottleneck, so they
        # should claim workers before the parallel baseline instances do.
        run_futures = {
            pool.submit(
                run_single,
                task_class=task_class,
                task_params=derive_run_task_params(task_params, run_mode, i),
                system_class=system_class,
                system_params=system_params,
                run_index=i,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                verbose_runs=verbose_runs,
                live_trace_path=(
                    None if live_trace_paths is None else live_trace_paths.get(i)
                ),
            ): i
            for i in range(runs)
            if i not in resumed_indices
        }
        # Baseline instances fill the remaining worker slots. Skipped entirely
        # when --resume found a completed baseline trace.
        baseline_futures = (
            {
                pool.submit(
                    _run_baseline_instance,
                    task_class=task_class,
                    task_params=merged_baseline_params,
                    system_class=system_class,
                    system_params=system_params,
                    instance_index=i,
                    run_group_id=run_group_id,
                    system_name=system_name,
                    task_name=task_name,
                    verbose=verbose_runs,
                ): i
                for i in range(num_baseline_instances)
            }
            if preloaded_baseline_info is None
            else {}
        )

        all_futures: dict[Any, tuple[str, int]] = {
            **{f: ("baseline", i) for f, i in baseline_futures.items()},
            **{f: ("run", i) for f, i in run_futures.items()},
        }

        completed_baseline_so_far: list[BaselineSuccess] = []
        for future in as_completed(all_futures):
            kind, idx = all_futures[future]
            try:
                payload = future.result()
                refusal: ProviderRefusalError | None = None
            except ProviderRefusalError as exc:
                payload, refusal = None, exc
            except Exception as exc:
                label = (
                    f"Baseline instance {idx + 1}/{num_baseline_instances}"
                    if kind == "baseline"
                    else f"Run {idx + 1}/{runs}"
                )
                logger.error("%s failed: %s", label, exc, exc_info=True)
                raise RuntimeError(f"{label} failed: {exc}") from exc

            if kind == "baseline":
                record_baseline_outcome(
                    idx,
                    payload,  # type: ignore[arg-type]
                    refusal,
                    num_instances=num_baseline_instances,
                    instance_results=baseline_instance_results,
                    blocked_instances=blocked_baseline_instances,
                    completed_so_far=completed_baseline_so_far,
                    on_progress=_write_partial_baseline_snapshot,
                )
            else:
                record_run_outcome(
                    idx,
                    payload,  # type: ignore[arg-type]
                    refusal,
                    runs=runs,
                    run_results=run_results,
                    run_group_id=run_group_id,
                    live_trace_path=(
                        None if live_trace_paths is None else live_trace_paths.get(idx)
                    ),
                )

    baseline_end_time = datetime.now().isoformat()

    if preloaded_baseline_info is not None:
        # Resume: prior baseline trace is reusable as-is.
        baseline_info = preloaded_baseline_info
    else:
        completed_baseline = [r for r in baseline_instance_results if r is not None]
        merged_result, baseline_trace_data, baseline_execution = (
            _merge_baseline_instance_results(
                completed_baseline,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                task_params=merged_baseline_params,
                system_params=system_params,
                start_time=baseline_start_time,
                end_time=baseline_end_time,
                blocked_instances=blocked_baseline_instances,
                expected_num_instances=num_baseline_instances,
            )
        )
        baseline_info = (None, merged_result, baseline_trace_data, baseline_execution)

        # Write the completed baseline trace so the live viewer can display it.
        if baseline_live_path is not None:
            _atomic_write_json(baseline_live_path, baseline_trace_data)
            logger.debug(
                "baseline.live_trace.written",
                extra={"path": str(baseline_live_path)},
            )

    # Aggregate rollout runs.
    task_results_list: list[TaskResult] = []
    run_trace_data_list: list[dict[str, Any]] = []
    execution_summaries: list[dict[str, Any]] = []
    for entry in run_results:
        assert entry is not None
        _, result, trace_data, execution_summary = entry
        task_results_list.append(result)
        run_trace_data_list.append(trace_data)
        execution_summaries.append(execution_summary)

    task_results = TaskResults(
        run_group_id=run_group_id,
        results=task_results_list,
        execution_summaries=execution_summaries,
    )
    logger.info(
        "finished",
        extra={"path": "parallel", "runs": len(task_results_list)},
    )

    return baseline_info, task_results, run_trace_data_list
