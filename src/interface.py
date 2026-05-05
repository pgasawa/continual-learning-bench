"""
Generic interface for continual learning benchmark tasks.

This interface supports online learning where systems must efficiently integrate
new information from interactions and synthesize it with past knowledge.
"""

from __future__ import annotations

import inspect
import random as _random
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional

from pydantic import BaseModel

from .usage import UsageEvent


@dataclass
class Observation:
    """
    Observation or feedback from the task about the system's previous response.

    Attributes:
        content: Feedback content (e.g., "correct", "incorrect", reward signal, outcome)
        metadata: Optional task-specific metadata (e.g., ground truth, explanation).
            Standard keys:
                instance_complete (bool): Whether this observation marks the end of
                    a task instance (e.g., a poker hand, a coding issue, a problem).
                    Systems can use this inside `respond()` to decide when to learn
                    or clear per-instance state. Defaults to True for backward
                    compatibility.
    """

    content: str
    instance_complete: bool = True
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Query:
    """
    A task turn delivered to the system.

    Attributes:
        prompt: Natural language task prompt for the current timestep.
        response_schema: Pydantic BaseModel subclass describing the expected
            response structure for this turn.
        feedback: Optional observation about the previous action.
        instance_id: Stable benchmark instance identifier. Benchmark tasks are
            expected to populate this on every query.
        instance_index: Stable canonical benchmark instance index. Benchmark
            tasks are expected to populate this on every query.
        metadata: Optional task-specific metadata (e.g., timestep, episode_id, context)
    """

    prompt: str
    response_schema: type[BaseModel]
    feedback: Optional[Observation] = None
    instance_id: Optional[str] = None
    instance_index: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Response:
    """
    System's response to an interaction.

    Attributes:
        action: Structured response as a Pydantic BaseModel instance matching
            the query's response_schema.
        metadata: Optional system-specific metadata (e.g., confidence, reasoning trace)
    """

    action: BaseModel
    metadata: Optional[dict[str, Any]] = None


@dataclass
class TaskAgentBrief:
    """Shared benchmark framing that tasks surface to systems up front."""

    objective: str
    instance_unit: str
    reward_definition: str
    completion_definition: str
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstanceOutcome:
    """Normalized per-instance outcome used for reward-first benchmark math."""

    instance_id: str
    instance_index: int
    reward: float
    success: Optional[bool] = None
    raw_metric_name: Optional[str] = None
    raw_metric_value: Optional[float] = None
    raw_metric_higher_is_better: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class EvalMetrics:
    """
    Task-internal supplementary evaluation metrics.

    Each task computes its own loss metric (however the task defines it —
    e.g. regret vs optimal, MAE, error count). The *framework* computes the
    primary ``mean_gain`` (stateful minus stateless baseline reward) separately
    from ``InstanceOutcome.reward`` values.

    Attributes:
        loss_curve: Per-instance loss values (for plotting the learning trajectory).
            Cumulative loss is derivable as ``sum(loss_curve)``.
        optimal_performance: Best achievable performance across all instances (task-specific units)
        actual_performance: Actual performance achieved (task-specific units)
        extra: Optional task-specific metric breakdown
    """

    loss_curve: list[float]
    optimal_performance: float
    actual_performance: float
    extra: Optional[dict[str, float]] = None


@dataclass
class TaskResult:
    """
    Final evaluation result for a task.

    Attributes:
        metrics: Dictionary of additional metrics
        summary: Human-readable summary of performance
        eval_metrics: Standardized evaluation metrics (required)
        instance_outcomes: Per-instance outcomes used to compute score
    """

    metrics: dict[str, Any]
    summary: str
    eval_metrics: EvalMetrics
    instance_outcomes: list[InstanceOutcome] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean per-instance reward. Higher is always better."""
        if not self.instance_outcomes:
            return 0.0
        return statistics.mean(o.reward for o in self.instance_outcomes)


def standard_evaluate(
    outcomes: list[InstanceOutcome],
    *,
    optimal_per_instance: float = 1.0,
    schedule_id: str | None = None,
    extra_metrics: Optional[dict[str, Any]] = None,
) -> TaskResult:
    """Build a TaskResult from instance outcomes using standard reward-based scoring.

    score = mean reward across instances (computed as a property on TaskResult).
    loss_curve[i] = optimal_per_instance - reward[i] (lower is better).

    Args:
        outcomes: Per-instance outcomes accumulated during the run.
        optimal_per_instance: Best possible reward for a single instance (default 1.0).
        schedule_id: Optional schedule name to include in metrics.
        extra_metrics: Optional additional keys merged into the metrics dict.
    """
    rewards = [o.reward for o in outcomes]
    n = len(rewards)
    total = sum(rewards)
    optimal_total = optimal_per_instance * n
    loss_curve = [round(optimal_per_instance - r, 6) for r in rewards]

    metrics: dict[str, Any] = {"total_reward": round(total, 6), "num_instances": n}
    if schedule_id:
        metrics["schedule_id"] = schedule_id
    if extra_metrics:
        metrics.update(extra_metrics)

    mean = total / n if n else 0.0
    return TaskResult(
        metrics=metrics,
        summary=f"Mean reward: {mean:.4f} over {n} instance(s).",
        eval_metrics=EvalMetrics(
            loss_curve=loss_curve,
            optimal_performance=optimal_total,
            actual_performance=round(total, 6),
        ),
        instance_outcomes=list(outcomes),
    )


def _aggregate_scalars(values: list[float | int]) -> dict[str, float]:
    """Return mean, std, min, max for a list of numeric values."""
    floats = [float(v) for v in values]
    return {
        "mean": statistics.mean(floats),
        "std": statistics.stdev(floats) if len(floats) > 1 else 0.0,
        "min": min(floats),
        "max": max(floats),
    }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _aggregate_numeric_tree(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively aggregate numeric leaves across similarly-shaped dicts."""

    aggregated: dict[str, Any] = {}
    all_keys = set()
    for value in values:
        all_keys.update(value.keys())

    for key in sorted(all_keys):
        present = [value[key] for value in values if key in value]
        if present and all(_is_numeric(item) for item in present):
            aggregated[key] = _aggregate_scalars(present)
            continue
        if present and all(isinstance(item, dict) for item in present):
            nested = _aggregate_numeric_tree(present)
            if nested:
                aggregated[key] = nested
    return aggregated


@dataclass
class TaskResults:
    """Container for multiple TaskResult instances from repeated benchmark runs."""

    run_group_id: str
    results: list[TaskResult]
    execution_summaries: Optional[list[dict[str, Any]]] = None

    def aggregate(self) -> dict[str, Any]:
        """Aggregate scores, eval metrics, and consistently-numeric metrics."""
        agg: dict[str, Any] = {}

        agg["score"] = _aggregate_scalars([r.score for r in self.results])

        eval_fields = [
            "optimal_performance",
            "actual_performance",
        ]
        eval_agg: dict[str, Any] = {}
        for eval_field in eval_fields:
            vals = [getattr(r.eval_metrics, eval_field) for r in self.results]
            eval_agg[eval_field] = _aggregate_scalars(vals)
        # cumulative_loss is derivable: sum(r.eval_metrics.loss_curve) per run
        eval_agg["cumulative_loss"] = _aggregate_scalars(
            [sum(r.eval_metrics.loss_curve) for r in self.results]
        )
        agg["eval_metrics"] = eval_agg

        all_metric_keys = set()
        for r in self.results:
            all_metric_keys.update(r.metrics.keys())

        metrics_agg: dict[str, Any] = {}
        for key in sorted(all_metric_keys):
            vals = [r.metrics[key] for r in self.results if key in r.metrics]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                metrics_agg[key] = _aggregate_scalars(vals)
        if metrics_agg:
            agg["metrics"] = metrics_agg

        if self.execution_summaries:
            execution_agg = _aggregate_numeric_tree(self.execution_summaries)
            if execution_agg:
                agg["execution"] = execution_agg

        return agg


@dataclass
class TaskStepResult:
    """
    Result of advancing the task with one system response.

    Attributes:
        observation: Feedback about the response that was just processed.
        next_query: The next task turn to send to the system, if any.
        done: Whether the task run is complete.
    """

    observation: Observation
    next_query: Optional[Query]
    done: bool
    instance_outcome: Optional[InstanceOutcome] = None


def observation_marks_instance_complete(observation: Observation) -> bool:
    """Return the effective instance boundary flag with a legacy metadata fallback."""

    metadata = observation.metadata or {}
    if "instance_complete" in metadata and observation.instance_complete is True:
        return bool(metadata["instance_complete"])
    return observation.instance_complete


def serialize_instance_outcome(outcome: InstanceOutcome) -> dict[str, Any]:
    """Convert a normalized instance outcome into a JSON-friendly mapping."""
    return asdict(outcome)


def instance_outcome_identity(outcome: InstanceOutcome) -> tuple[str, int]:
    """Return the stable benchmark identity for one instance outcome."""
    return (str(outcome.instance_id), int(outcome.instance_index))


def require_query_instance_identity(query: Query, *, context: str) -> None:
    """Raise when a benchmark task query is missing stable instance identity."""
    if not isinstance(query.instance_id, str) or not query.instance_id:
        raise ValueError(f"{context} must set Query.instance_id to a non-empty string.")
    if not isinstance(query.instance_index, int) or isinstance(
        query.instance_index, bool
    ):
        raise ValueError(f"{context} must set Query.instance_index to an int.")
    if query.instance_index < 0:
        raise ValueError(f"{context} must set Query.instance_index >= 0.")


def serialize_task_agent_brief(
    brief: Optional[TaskAgentBrief],
) -> Optional[dict[str, Any]]:
    """Convert an optional task brief into a JSON-friendly mapping."""
    if brief is None:
        return None
    return brief.to_dict()


def format_task_agent_brief(brief: TaskAgentBrief) -> str:
    """Render the shared task brief as prompt text for the first turn of an instance."""
    lines = [
        "=== Brief ===",
        f"Objective: {brief.objective}",
        f"Task Unit: {brief.instance_unit}",
        f"Reward: {brief.reward_definition}",
        f"Completion: {brief.completion_definition}",
    ]
    if brief.constraints:
        lines.append("Constraints:")
        lines.extend(f"- {constraint}" for constraint in brief.constraints)
    lines.append("=======================")
    return "\n".join(lines)


class ContinualLearningTask(ABC):
    """Abstract base class for continual learning benchmark tasks.

    Tasks should model one run using a small lifecycle:

    1. :meth:`build_canonical_run_state` constructs the full canonical
       instance sequence and clears per-run state.
    2. :meth:`select_run_instances` optionally narrows that sequence to the
       active ordered subset for this run (the default baseline case is
       ``[instance_index]``).
    3. :meth:`build_current_query` renders the current active instance.

    Most tasks should implement those lifecycle methods and leave
    :meth:`reset` plus :meth:`reset_baseline_instance` alone.
    """

    has_setup: ClassVar[bool] = False
    """Set to True in task subclasses that override :meth:`setup` with
    non-trivial work (e.g. downloading datasets). Used by ``clbench setup``
    to show which tasks require one-time initialization."""

    baseline_task_params: ClassVar[dict[str, Any]] = {}
    """Extra ``__init__`` params merged into *task_params* for the stateless
    baseline run.  Subclasses that need different construction for the baseline
    (e.g. cleaning a Docker workspace) can set this at the class level.
    Unknown keys are silently dropped by ``run_single``'s signature filter."""

    parallel_safe: ClassVar[bool] = True
    """Whether multiple instances of this task class can run concurrently in
    separate processes without file or resource conflicts.

    True (default) means Docker containers use unique names, temp files use
    unique paths, and no fixed shared state is written.  Set to False if the
    task uses a fixed filesystem path or shared port that would clash when two
    instances run simultaneously.
    """

    r_max: ClassVar[float]
    """Mean per-instance maximum reward for the default evaluation schedule.

    This value must be on the same scale as ``InstanceOutcome.reward`` and is
    used to normalize cross-task reward and gain.  Equivalently, it is the
    run-level reference maximum reward divided by the number of instances in
    the default reported schedule.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            params = inspect.signature(cls.__init__).parameters
            if "num_instances" not in params:
                raise TypeError(
                    f"{cls.__name__} must define a 'num_instances' parameter in __init__. "
                    "This is the standard way to control run length across all tasks."
                )

    def reset(self) -> Query:
        """Reset to the active run sequence and return the first query."""
        self.build_canonical_run_state()
        self.select_run_instances(None)
        return self.build_current_query()

    def cleanup(self) -> None:
        """Release task-owned external resources after a run."""
        pass

    @abstractmethod
    def build_canonical_run_state(self) -> None:
        """Build the full canonical run sequence and clear per-run state.

        Implementations should populate ``self.instances`` with the canonical
        ordered instance list whenever possible.
        """
        pass

    @abstractmethod
    def step(self, response: Response) -> TaskStepResult:
        """
        Process a system response and return the next task step result.

        Args:
            response: System's response to the current query

        Returns:
            TaskStepResult with the observation, next query, and completion flag
        """
        pass

    @abstractmethod
    def evaluate(self) -> TaskResult:
        """
        Evaluate the system's performance on the task.

        Returns:
            TaskResult with score, metrics, and summary
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name identifier."""
        pass

    @property
    def description(self) -> str:
        """Human-readable task description."""
        return ""

    def get_agent_brief(self) -> Optional[TaskAgentBrief]:
        """Return the shared task brief shown to systems at instance start."""
        return None

    @classmethod
    def setup(cls, *, force: bool = False) -> None:
        """Perform one-time setup for this task (e.g., download datasets).

        Override in task subclasses that require external data or other
        one-time initialization before the task can be run. Also set
        ``has_setup = True`` on the class so ``clbench setup`` can surface it.

        Args:
            force: Re-run setup even when output artifacts already exist.
        """

    def reset_baseline_instance(self, instance_index: int) -> Query:
        """Reset to canonical state, activate one instance, and return its first query."""
        self.build_canonical_run_state()
        self.select_run_instances([instance_index])
        return self.build_current_query()

    def handle_system_error(
        self,
        query: Query,
        error: Exception,
    ) -> Optional[TaskStepResult]:
        """Optionally convert a recoverable system-response error into an outcome.

        The default is strict: tasks that cannot safely skip the active instance
        return ``None`` and the runner re-raises the error. Tasks with a clear
        zero-credit semantics can override this to record a failed instance and
        advance to the next query.
        """
        return None

    def select_run_instances(self, instance_indices: Optional[list[int]]) -> None:
        """Select the active ordered instance sequence for this run.

        ``None`` keeps the full canonical sequence. ``[i]`` is the standard
        baseline-worker case: run only canonical instance ``i``.

        Tasks that store their ordered instances in ``self.instances`` can rely
        on this default. Tasks with additional environment or metadata state
        should override it, typically by syncing that state and then calling
        ``super().select_run_instances(instance_indices)``.
        """
        instances = getattr(self, "instances", None)
        if not isinstance(instances, list):
            raise NotImplementedError(
                f"{type(self).__name__} must override select_run_instances() "
                "or maintain a list-valued self.instances."
            )

        if instance_indices is None:
            selected_indices = list(range(len(instances)))
        else:
            selected_indices = list(instance_indices)
            self.instances = [instances[i] for i in selected_indices]  # type: ignore[attr-defined]

        self._selected_instance_indices = selected_indices  # type: ignore[attr-defined]
        if hasattr(self, "num_instances"):
            self.num_instances = len(selected_indices)  # type: ignore[attr-defined]

    def canonical_instance_index(self, local_index: int) -> int:
        """Return the canonical run index for a local active-sequence index."""
        selected = getattr(self, "_selected_instance_indices", None)
        if isinstance(selected, list) and 0 <= local_index < len(selected):
            return int(selected[local_index])
        return local_index

    @abstractmethod
    def build_current_query(self) -> Query:
        """Return the query for the currently active instance."""
        pass

    def prepare_run(self, run_index: int) -> None:
        """Permute the task's instance order for *run_index*.

        Called by the runner after instantiation whenever ``run_index > 0``.
        The default implementation shuffles ``self.instances`` in-place if that
        attribute exists and ``self.seed`` is set, covering tasks that
        pre-generate a flat instance list at init time.

        Tasks with a different instance structure (e.g. poker's pre-dealt hands)
        should override this method.  ``run_index=0`` is the canonical order
        used by the baseline — both the default and any override must leave the
        task unchanged when called with ``run_index=0``.
        """
        if not run_index:
            return
        if hasattr(self, "instances") and hasattr(self, "seed"):
            shuffled = list(self.instances)  # type: ignore[attr-defined]
            _random.Random(self.seed + run_index).shuffle(shuffled)  # type: ignore[attr-defined]
            self.instances = shuffled  # type: ignore[attr-defined]

    def get_instance_outcomes(self) -> list[InstanceOutcome]:
        """Return any normalized instance outcomes accumulated so far.

        Tasks that stream outcomes during execution typically store them on
        ``self._instance_outcomes``. The default implementation returns that
        list when present so the runtime can keep live traces aligned even if a
        task advances through multiple instances between system turns.
        """
        outcomes = getattr(self, "_instance_outcomes", None)
        if isinstance(outcomes, list):
            return list(outcomes)
        return []


class ContinualLearningSystem(ABC):
    """
    Abstract base class for systems being evaluated on the benchmark.

    Systems should implement their own learning/update mechanisms internally.
    The interaction loop is: receive query (with optional feedback) → respond.

    Class attributes:
        supports_baseline: Set to False for systems that cannot be reset between
            instances (e.g. humans). When False the benchmark runner skips the
            baseline phase entirely; gain metrics will be unavailable but reward
            curves are still recorded in the viewer artifact.
    """

    supports_baseline: bool = True
    parallel_safe: ClassVar[bool] = True
    """Whether multiple instances of this system class can run concurrently in
    separate processes without resource conflicts.

    True (default) means the system uses only in-memory state, or isolates any
    file/container state per-instance (e.g. via ``tempfile.mkdtemp``).  Set to
    False if the system reads from stdin (Human) or writes to a fixed path that
    would clash across simultaneous instances.
    """

    @abstractmethod
    def respond(self, query: Query) -> Response:
        """
        Handle a task turn.

        Systems may choose to learn from `query.feedback` before, during, or after
        producing an action.

        Args:
            query: Current query from the task

        Returns:
            Response with action and optional metadata
        """
        pass

    def get_run_artifacts(self) -> Optional[dict[str, Any]]:
        """
        Optional hook to export run artifacts after task completion.

        Systems can return picklable debug artifacts such as learned memory
        snapshots. Returning None means there is nothing to export.
        """
        return None

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        """
        Observe task feedback after a response.

        Systems can use this hook to update internal state immediately after an
        interaction, before the next prompt is presented or checkpoints are saved.
        """
        return None

    def record_usage_event(self, event: UsageEvent) -> None:
        """Append a billable usage event to the system-local buffer."""
        if not hasattr(self, "_usage_events"):
            self._usage_events: list[UsageEvent] = []
        self._usage_events.append(event)

    def consume_usage_events(self) -> list[UsageEvent]:
        """Return and clear buffered usage events accumulated by the system."""
        if not hasattr(self, "_usage_events"):
            self._usage_events = []
        events = list(self._usage_events)
        self._usage_events.clear()
        return events

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the system state (if needed between tasks).

        Note: This is for resetting between different task instances,
        not for resetting learned knowledge within a task.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """System name identifier."""
        pass


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
    """Run a task with the shared runtime runner implementation."""
    from .runtime.runner import run_task as _run_task

    return _run_task(
        task=task,
        system=system,
        trace_recorder=trace_recorder,
        show_progress=show_progress,
        verbose_logging=verbose_logging,
        rollout_label=rollout_label,
        reset_system=reset_system,
        reset_between_instances=reset_between_instances,
        phase=phase,
        initial_query=initial_query,
    )
