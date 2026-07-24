"""
Task runner orchestration for continual learning benchmark loops.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
import logging
import time
from typing import Any, Optional

from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    InstanceOutcome,
    instance_outcome_identity,
    observation_marks_instance_complete,
    Query,
    Response,
    require_query_instance_identity,
    TaskResult,
    TaskStepResult,
)
from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType
from ..logging_utils import bind_logging_context
from ..usage import serialize_usage_events, summarize_usage_events

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


_VERBOSE_METADATA_SKIP = {
    "done",
    "instance_complete",
    "step",
    "option_labels",
    "support_labels",
}

_VERBOSE_METADATA_VALUE_MAX_CHARS = 48


def _format_verbose_metadata(metadata: Optional[dict[str, Any]]) -> list[str]:
    """Return compact key=value items for non-internal query metadata."""
    if not metadata:
        return []

    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            rendered = value
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = repr(value)
        if len(rendered) <= _VERBOSE_METADATA_VALUE_MAX_CHARS:
            return rendered
        return f"{rendered[: _VERBOSE_METADATA_VALUE_MAX_CHARS - 3]}..."

    parts: list[str] = []
    for key in sorted(metadata):
        if key in _VERBOSE_METADATA_SKIP:
            continue
        parts.append(f"{key}={_format_value(metadata[key])}")
    return parts


def _outcome_key(outcome: InstanceOutcome) -> tuple[str, int]:
    """Return the stable identity tuple for an instance outcome."""
    return instance_outcome_identity(outcome)


def _upsert_instance_outcomes(
    recorded: list[InstanceOutcome],
    observed: list[InstanceOutcome],
) -> list[InstanceOutcome]:
    """Merge observed outcomes into ``recorded`` by instance identity."""
    index_by_key = {_outcome_key(outcome): idx for idx, outcome in enumerate(recorded)}
    changed: list[InstanceOutcome] = []
    for outcome in observed:
        key = _outcome_key(outcome)
        existing_idx = index_by_key.get(key)
        if existing_idx is None:
            recorded.append(outcome)
            index_by_key[key] = len(recorded) - 1
            changed.append(outcome)
            continue
        if recorded[existing_idx] != outcome:
            recorded[existing_idx] = outcome
            changed.append(outcome)
    return changed


def _collect_step_outcomes(
    task: ContinualLearningTask,
    step_outcome: Optional[InstanceOutcome],
) -> list[InstanceOutcome]:
    """Return all outcomes visible after one task step."""
    observed = list(task.get_instance_outcomes())
    if step_outcome is None:
        return observed
    observed_keys = {_outcome_key(outcome) for outcome in observed}
    if _outcome_key(step_outcome) not in observed_keys:
        observed.append(step_outcome)
    return observed


def _copy_latest_outcome(
    outcome: Optional[InstanceOutcome],
) -> Optional[dict[str, Any]]:
    """Return a tiny display payload for progress logging."""
    if outcome is None:
        return None
    return {
        "instance_id": outcome.instance_id,
        "instance_index": outcome.instance_index,
        "reward": outcome.reward,
    }


def _is_recoverable_system_response_error(exc: Exception) -> bool:
    """Return whether a system response failure may be scored as zero credit."""
    return isinstance(exc, RuntimeError) and "LLM call failed:" in str(exc)


def _system_error_payload(exc: Exception) -> dict[str, Any]:
    """Build trace-friendly metadata for a recoverable system response failure."""
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "zero_credit": True,
    }


def _query_is_terminal(query: Query) -> bool:
    """Return whether ``query`` indicates the task is already complete."""
    metadata = query.metadata or {}
    return bool(metadata.get("done"))


def _finalize_task_result(
    task: ContinualLearningTask,
    system: ContinualLearningSystem,
    trace_recorder: Optional[Any],
    runtime_instance_outcomes: list[InstanceOutcome],
) -> TaskResult:
    """Evaluate the task and merge any streamed instance outcomes."""
    task.cleanup()
    task_result = task.evaluate()
    final_outcomes = list(task_result.instance_outcomes) or task.get_instance_outcomes()
    _upsert_instance_outcomes(runtime_instance_outcomes, final_outcomes)
    task_result.instance_outcomes = list(runtime_instance_outcomes)

    if trace_recorder is not None:
        trace_recorder.sync_instance_outcomes(task_result.instance_outcomes)

    if trace_recorder is not None:
        try:
            trace_recorder.record_system_artifacts(system.get_run_artifacts())
        except Exception:
            pass

    return task_result


def _make_fallback_action(query: Query) -> BaseModel:
    """Return a zero-value action when the LLM call fails so the run can continue."""
    schema = query.response_schema
    defaults: dict[str, Any] = {}
    for field_name, field_info in schema.model_fields.items():
        if not isinstance(field_info.default, PydanticUndefinedType):
            defaults[field_name] = field_info.default
        else:
            ann = field_info.annotation
            if ann is str:
                defaults[field_name] = ""
            elif ann is int:
                defaults[field_name] = 0
            elif ann is bool:
                defaults[field_name] = False
            elif ann is list:
                defaults[field_name] = []
            else:
                defaults[field_name] = None
    return schema.model_construct(**defaults)


def run_task(
    task: ContinualLearningTask,
    system: ContinualLearningSystem,
    trace_recorder: Optional[Any] = None,
    show_progress: bool = True,
    verbose_logging: bool = False,
    rollout_label: Optional[str] = None,
    reset_system: bool = True,
    reset_between_instances: bool = False,
    phase: str = "rollout",
    initial_query: Optional[Query] = None,
) -> TaskResult:
    """
    Run a continual learning task with a given system.

    Implements the interaction loop:
    1. Task provides query
    2. System responds to query, optionally ingesting feedback from the prior step
    3. Task processes the response and provides feedback plus next query
    4. Runner attaches the observation as `feedback` on the next query

    Args:
        task: Task instance to evaluate on
        system: System to evaluate
        trace_recorder: Optional TraceRecorder to capture full interaction history
        show_progress: Whether to show progress bar (default: True)
        verbose_logging: Whether to emit per-instance progress logs.
        rollout_label: Optional rollout label for verbose logs.
        reset_system: Whether to reset the system before running (default: True).
        reset_between_instances: Whether to reset the system after each completed
            instance while keeping task state fixed. Used for the reward baseline.
        phase: Human-readable run phase for progress logging.
        initial_query: Optional pre-built first query. When provided, the runner
            skips ``task.reset()`` and starts from this query directly.

    Returns:
        TaskResult with evaluation metrics
    """
    pbar = None
    runtime_instance_outcomes: list[InstanceOutcome] = []

    def _call_system(
        query: Query,
    ) -> tuple[Response, float, str, str, list[dict[str, Any]]]:
        response_start_time = datetime.now().isoformat()
        response_start_perf = time.perf_counter()
        response = system.respond(query)
        response_elapsed_seconds = time.perf_counter() - response_start_perf
        response_end_time = datetime.now().isoformat()
        respond_events = serialize_usage_events(system.consume_usage_events())
        response_metadata = dict(response.metadata or {})
        response_metadata["usage"] = summarize_usage_events(respond_events)
        response.metadata = response_metadata
        return (
            response,
            response_elapsed_seconds,
            response_start_time,
            response_end_time,
            respond_events,
        )

    try:
        query = task.reset() if initial_query is None else initial_query
        query_context = (
            f"{type(task).__name__}.reset()"
            if initial_query is None
            else f"{type(task).__name__} initial query"
        )
        require_query_instance_identity(query, context=query_context)
        if reset_system:
            system.reset()
        system.consume_usage_events()
        logger.debug(
            "started",
            extra={
                "task_class": type(task).__name__,
                "system_class": type(system).__name__,
                "phase": phase,
                "reset_system": reset_system,
                "reset_between_instances": reset_between_instances,
            },
        )

        latest_outcome: Optional[dict[str, Any]] = None
        initial_outcomes = _upsert_instance_outcomes(
            runtime_instance_outcomes, task.get_instance_outcomes()
        )
        if initial_outcomes:
            latest_outcome = _copy_latest_outcome(initial_outcomes[-1])
            if trace_recorder is not None:
                trace_recorder.sync_instance_outcomes(initial_outcomes)

        if _query_is_terminal(query):
            return _finalize_task_result(
                task, system, trace_recorder, runtime_instance_outcomes
            )

        if show_progress and HAS_TQDM:
            pbar = tqdm(desc="Starting", bar_format="{desc}", dynamic_ncols=True)

        done = False
        step = 0
        completed_instances = 0
        prefix = f"[{rollout_label}] " if rollout_label else ""

        while not done:
            step += 1
            with bind_logging_context(
                phase=phase,
                step=step,
                instance_id=query.instance_id,
                instance_index=query.instance_index,
            ):
                logger.debug(
                    "interaction.started",
                    extra={
                        "metadata": query.metadata or {},
                        "has_feedback": query.feedback is not None,
                    },
                )

            if pbar is not None:
                metadata = query.metadata or {}
                desc_parts = [phase.title(), f"Interaction {step}"]

                if query.instance_index is not None:
                    desc_parts.append(f"Instance {query.instance_index + 1}")
                if query.instance_id:
                    desc_parts.append(f"ID {query.instance_id}")

                if "problem_id" in metadata:
                    desc_parts.append(f"Problem: {metadata['problem_id']}")
                    if "attempts" in metadata:
                        desc_parts.append(
                            f"Attempt: {metadata['attempts'] + 1}/{metadata.get('max_attempts', '?')}"
                        )
                elif "hand_num" in metadata:
                    desc_parts.append(f"Hand #{metadata['hand_num']}")
                    if "phase" in metadata:
                        desc_parts.append(f"Phase: {metadata['phase']}")
                elif "issue_id" in metadata:
                    desc_parts.append(f"Issue: {metadata['issue_id']}")
                    if "step" in metadata:
                        desc_parts.append(
                            f"Step: {metadata['step'] + 1}/{metadata.get('max_steps', '?')}"
                        )
                if latest_outcome is not None:
                    desc_parts.append(f"Latest reward {latest_outcome['reward']:+.4f}")

                pbar.set_description(" | ".join(desc_parts))
                pbar.update(1)

            step_result: Optional[TaskStepResult] = None
            try:
                (
                    response,
                    response_elapsed_seconds,
                    response_start_time,
                    response_end_time,
                    respond_events,
                ) = _call_system(query)
            except ProviderRefusalError as exc:
                logger.warning(
                    "provider.refusal.fallback",
                    extra={
                        "step": step,
                        "instance_id": query.instance_id,
                        "instance_index": query.instance_index,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "kind": exc.kind,
                        "prompt_digest": exc.prompt_digest,
                        "prompt_chars": exc.prompt_chars,
                    },
                )
                response = Response(
                    action=_make_fallback_action(query),
                    metadata={
                        "llm_error": {
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "kind": exc.kind,
                            "prompt_digest": exc.prompt_digest,
                            "prompt_chars": exc.prompt_chars,
                        }
                    },
                )
                response_elapsed_seconds = 0.0
                response_start_time = datetime.now().isoformat()
                response_end_time = response_start_time
                respond_events = []
            except Exception as exc:
                if not _is_recoverable_system_response_error(exc):
                    raise
                step_result = task.handle_system_error(query, exc)
                if step_result is None:
                    raise

                error_payload = _system_error_payload(exc)
                logger.warning(
                    "system.response.zero_credit",
                    extra={
                        "step": step,
                        "instance_id": query.instance_id,
                        "instance_index": query.instance_index,
                        "error_type": error_payload["error_type"],
                        "error_message": error_payload["error_message"],
                    },
                )
                print(
                    "  WARNING: system response failed; assigning zero credit "
                    f"for instance #{query.instance_index} and continuing: {exc}",
                    flush=True,
                )
                response = Response(
                    action=_make_fallback_action(query),
                    metadata={"llm_error": error_payload},
                )
                response_elapsed_seconds = 0.0
                response_start_time = datetime.now().isoformat()
                response_end_time = response_start_time
                respond_events = []

            if step_result is None:
                step_result = task.step(response)
            observed_outcomes = _collect_step_outcomes(
                task, step_result.instance_outcome
            )
            synced_outcomes = _upsert_instance_outcomes(
                runtime_instance_outcomes, observed_outcomes
            )

            system.observe(step_result.observation, step_result.next_query)
            observe_events = serialize_usage_events(system.consume_usage_events())

            if trace_recorder is not None:
                trace_recorder.record_interaction(
                    step,
                    query,
                    response,
                    step_result.observation,
                    step_result.done,
                    timing={
                        "response_start_time": response_start_time,
                        "response_end_time": response_end_time,
                        "response_latency_seconds": response_elapsed_seconds,
                        "response_latency_ms": round(
                            response_elapsed_seconds * 1000, 3
                        ),
                    },
                    usage={
                        "respond": {
                            **summarize_usage_events(respond_events),
                            "events": respond_events,
                        },
                        "observe": {
                            **summarize_usage_events(observe_events),
                            "events": observe_events,
                        },
                        "interaction": summarize_usage_events(
                            [*respond_events, *observe_events]
                        ),
                    },
                )

            if trace_recorder is not None:
                trace_recorder.sync_instance_outcomes(synced_outcomes)

            latest_outcome = _copy_latest_outcome(
                step_result.instance_outcome
                if step_result.instance_outcome is not None
                else (synced_outcomes[-1] if synced_outcomes else None)
            )

            if observation_marks_instance_complete(step_result.observation):
                completed_instances += 1
                (logger.info if verbose_logging else logger.debug)(
                    "instance.completed",
                    extra={
                        "step": step,
                        "completed_instances": completed_instances,
                        "instance_id": None
                        if latest_outcome is None
                        else latest_outcome.get("instance_id"),
                        "reward": None
                        if latest_outcome is None
                        else latest_outcome.get("reward"),
                    },
                )
                if verbose_logging:
                    extra_parts = [
                        f"instance {completed_instances}",
                        f"step {step}",
                        f"phase {phase}",
                        f"latency {response_elapsed_seconds:.2f}s",
                    ]
                    if latest_outcome is not None:
                        extra_parts.append(f"reward {latest_outcome['reward']:+.4f}")
                        extra_parts.append(f"id {latest_outcome['instance_id']}")
                    extra_parts.extend(_format_verbose_metadata(query.metadata))
                    print(f"{prefix}" + " | ".join(extra_parts))

            done = step_result.done
            logger.debug(
                "interaction.finished",
                extra={
                    "step": step,
                    "done": done,
                    "response_latency_seconds": round(response_elapsed_seconds, 6),
                    "synced_outcomes": len(synced_outcomes),
                    "observation_complete": observation_marks_instance_complete(
                        step_result.observation
                    ),
                },
            )
            if done:
                break

            next_query = step_result.next_query
            if next_query is None:
                raise RuntimeError("Task returned done=False but next_query=None.")
            require_query_instance_identity(
                next_query,
                context=f"{type(task).__name__}.step() next_query",
            )
            if reset_between_instances and observation_marks_instance_complete(
                step_result.observation
            ):
                system.reset()
                system.consume_usage_events()
            query = next_query

        return _finalize_task_result(
            task, system, trace_recorder, runtime_instance_outcomes
        )
    finally:
        with suppress(Exception):
            task.cleanup()
        if pbar is not None:
            pbar.close()
