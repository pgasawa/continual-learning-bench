"""
CLI interface for running benchmarks with dynamic parameter parsing.
"""

import argparse
import difflib
import gzip
import inspect
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from dotenv import load_dotenv

from .artifacts import save_artifacts
from .interface import ContinualLearningTask
from .live_dashboard import (
    build_live_manifest_path,
    build_live_output_dir,
    build_live_trace_path,
    start_live_dashboard_server,
    write_live_manifest,
    write_run_all_manifest,
)
from .logging_utils import (
    RUN_LOG_LEVEL,
    bind_logging_context,
    build_run_log_path,
    configure_logging,
)
from .registry import (
    get_system_class,
    get_task_class,
    get_class_params,
    list_systems,
    list_tasks,
)
from .run_ids import new_timestamp_run_id, safe_run_id_component
from .runs import RunMode, run_benchmark
from .runs.common import filter_init_params
from .system_manifest import build_system_manifest_entry
from .tasks.environments import (
    is_environment_backend_param,
    validate_environment_backend,
)
from .tasks.variants import list_task_variants
from .tasks.schedules import get_task_schedule, list_task_schedules
from .trace_metrics import (
    attach_baseline_to_outcomes,
    build_benchmark_aggregate,
    build_sequence_signature,
    summarize_instance_outcomes,
)
from .trace_storage import (
    BaselineSummaryEntry,
    RunSummary,
    RunSummaryRunEntry,
    save_viewer_artifact,
)

logger = logging.getLogger(__name__)
load_dotenv()


def _configure_cli_logging(
    debug_enabled: bool,
    *,
    verbosity: int = 0,
    log_file_path: Path | None = None,
) -> None:
    """Configure stderr logging for CLI diagnostics."""
    configure_logging(
        verbosity=verbosity,
        debug_console=debug_enabled,
        log_file_path=log_file_path,
    )


def _emit_cli_error(
    message: str,
    *,
    details: list[str] | None = None,
    hint: str | None = None,
    exc: BaseException | None = None,
    show_traceback: bool = False,
) -> None:
    """Print a user-facing CLI error to stderr, optionally with traceback in debug mode."""
    print(f"Error: {message}", file=sys.stderr)
    for detail in details or []:
        print(f"  {detail}", file=sys.stderr)
    if hint:
        print(f"Hint: {hint}", file=sys.stderr)
    if exc is not None and (show_traceback or logging.root.level <= logging.DEBUG):
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _exit_cli_error(
    message: str,
    *,
    details: list[str] | None = None,
    hint: str | None = None,
    exc: BaseException | None = None,
    show_traceback: bool = False,
    exit_code: int = 1,
) -> None:
    """Emit a CLI error and terminate."""
    _emit_cli_error(
        message,
        details=details,
        hint=hint,
        exc=exc,
        show_traceback=show_traceback,
    )
    sys.exit(exit_code)


def _exit_unknown_registry_entry(kind: str, name: str) -> None:
    """Emit a friendly error for unknown task/system names."""
    available = list_tasks() if kind == "task" else list_systems()
    matches = difflib.get_close_matches(name, available, n=3)
    details = [f"{kind.title()}: {name}"]
    if matches:
        details.append(f"Closest matches: {', '.join(matches)}")
    _exit_cli_error(
        f"Unknown {kind} '{name}'.",
        details=details,
        hint=f"Run 'clbench list {kind}s' to see available {kind}s.",
    )


def _parse_json_mapping_arg(raw_value: str, flag_name: str) -> dict[str, Any]:
    """Parse a CLI JSON object argument and exit with a helpful hint on failure."""
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        _exit_cli_error(
            f"Invalid JSON passed to {flag_name}.",
            details=[str(exc)],
            hint=(
                "Wrap the JSON in single quotes, for example "
                f'{flag_name} \'{{"model": "gpt-5.4"}}\'.'
            ),
            exc=exc,
        )

    if not isinstance(parsed, dict):
        _exit_cli_error(
            f"{flag_name} must be a JSON object.",
            details=[f"Received {type(parsed).__name__} instead."],
            hint=(f'Pass key/value pairs like {flag_name} \'{{"model": "gpt-5.4"}}\'.'),
        )

    return parsed


def format_schedule_summary(schedule: Any) -> str:
    """Format a concise schedule summary for CLI listing output."""
    run_defaults = schedule.run_defaults
    return (
        f"{schedule.id}: {schedule.display_name} - {schedule.description} "
        f"[stages={len(schedule.stages)}, runs={run_defaults.runs}, "
        f"mode={run_defaults.mode}, max_workers={run_defaults.max_workers}]"
    )


def format_cost_summary(summary: dict[str, Any]) -> str:
    """Format a usage summary for CLI display."""
    cost = float(summary.get("cost_usd", 0.0))
    suffix = "" if summary.get("pricing_complete", True) else "+ (partial pricing)"
    if summary.get("total_tokens_complete", True):
        token_text = f"{int(summary.get('total_tokens', 0))} tokens"
    elif int(summary.get("missing_total_token_count", 0)) == int(
        summary.get("call_count", 0)
    ):
        token_text = "tokens unavailable"
    else:
        token_text = f"{int(summary.get('total_tokens', 0))}+ tokens (partial)"
    return (
        f"${cost:.6f}{suffix} | "
        f"{int(summary.get('call_count', 0))} call(s) | "
        f"{token_text}"
    )


def _get_task_brief(
    task_class: type[ContinualLearningTask], task_params: dict[str, Any]
) -> dict[str, Any] | None:
    """Instantiate a task with the provided params to fetch its shared brief."""
    try:
        task = task_class(**filter_init_params(task_class, task_params))
        brief = task.get_agent_brief()
        return None if brief is None else brief.to_dict()
    except Exception:
        return None


def _build_display_task_params(
    task_class: type[ContinualLearningTask], task_params: dict[str, Any]
) -> dict[str, Any]:
    """Augment task params with resolved runtime metadata when available."""
    try:
        task = task_class(**filter_init_params(task_class, task_params))
    except Exception:
        return dict(task_params)

    display_params = dict(task_params)
    expected_instances = getattr(task, "num_instances", None)
    if isinstance(expected_instances, int) and expected_instances > 0:
        if "num_instances" in display_params:
            display_params["num_instances"] = expected_instances
        display_params["expected_num_instances"] = expected_instances
    return display_params


def _relpath(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_live_manifest_payload(
    *,
    run_group_id: str,
    status: str,
    phase: str,
    task_name: str,
    task_params: dict[str, Any],
    system_name: str,
    system_params: dict[str, Any],
    root_dir: Path,
    manifest_path: Path,
    task_brief: dict[str, Any] | None,
    baseline_trace_path: Path | None,
    run_trace_paths: dict[int, Path],
    summary_path: Path | None = None,
) -> dict[str, Any]:
    baseline_entry = None
    if baseline_trace_path is not None:
        baseline_entry = {"trace_path": _relpath(baseline_trace_path, root_dir)}
    runs = [
        {"run_index": run_index, "trace_path": _relpath(path, root_dir)}
        for run_index, path in sorted(run_trace_paths.items())
    ]
    payload: dict[str, Any] = {
        "kind": "live_manifest",
        "run_group_id": run_group_id,
        "status": status,
        "phase": phase,
        "poll_interval_ms": 2000,
        "task": {"name": task_name, "params": task_params},
        "system": build_system_manifest_entry(system_name, system_params),
        "task_brief": task_brief,
        "baseline": baseline_entry,
        "runs": runs,
        "manifest_path": _relpath(manifest_path, root_dir),
    }
    if summary_path is not None:
        payload["summary_path"] = _relpath(summary_path, root_dir)
    return payload


def _print_reward_first_results(
    *,
    aggregate: dict[str, Any],
    baseline_summary: dict[str, Any],
    baseline_execution: dict[str, Any],
    run_entries: list[RunSummaryRunEntry],
    runs: int,
) -> None:
    has_gain_data = any(
        value is not None for value in aggregate.get("mean_gain_by_index", [])
    )
    gain_values = [
        float(value)
        for value in aggregate.get("mean_gain_by_index", [])
        if isinstance(value, (int, float))
    ]
    cumulative_gain = round(sum(gain_values), 6) if has_gain_data else None
    reward_values = [
        float(value)
        for value in aggregate.get("mean_reward_by_index", [])
        if isinstance(value, (int, float))
    ]
    cumulative_reward = round(sum(reward_values), 6) if reward_values else 0.0

    print("\n" + "=" * 80)
    print("BENCHMARK")
    print("=" * 80)
    print(
        "\nFinal cumulative gain: "
        + (f"{cumulative_gain:+.4f}" if cumulative_gain is not None else "n/a")
    )
    print(f"Final cumulative reward: {cumulative_reward:+.4f}")
    print(
        f"Baseline reward total: {float(baseline_summary.get('total_reward', 0.0)):+.4f}"
    )
    baseline_total = baseline_execution.get("instances_total")
    baseline_completed = baseline_execution.get(
        "instances_completed", baseline_summary.get("instances_completed", 0)
    )
    baseline_blocked = int(baseline_execution.get("instances_blocked", 0) or 0)
    if baseline_total is not None:
        coverage_label = f"{baseline_completed}/{baseline_total}"
        if baseline_blocked:
            coverage_label += f" ({baseline_blocked} blocked)"
        print(f"Baseline coverage: {coverage_label}")

    if runs == 1 and run_entries:
        run_aggregate = run_entries[0].aggregate or {}
        print(f"Run total reward: {float(run_aggregate.get('total_reward', 0.0)):+.4f}")
        run_total_gain = run_aggregate.get("total_gain")
        print(
            "Run cumulative gain: "
            + (f"{float(run_total_gain):+.4f}" if run_total_gain is not None else "n/a")
        )
    elif run_entries:
        blocked_count = sum(1 for entry in run_entries if entry.status == "blocked")
        if blocked_count:
            print(
                f"\nRuns ({len(run_entries) - blocked_count} completed, "
                f"{blocked_count} blocked):"
            )
        else:
            print("\nRuns:")
        for entry in run_entries:
            run_aggregate = entry.aggregate or {}
            run_total_gain = run_aggregate.get("total_gain")
            status_tag = ""
            if entry.status == "blocked":
                reason = (entry.execution or {}).get("blocked_reason") or {}
                kind = (reason.get("kind") or "blocked").replace("_", " ")
                completed = run_aggregate.get("instances_completed", 0)
                status_tag = f" [BLOCKED:{kind}, partial: {completed} inst]"
            print(
                "  "
                f"Run {entry.run_index + 1}: "
                f"cumulative_gain="
                f"{f'{float(run_total_gain):+.4f}' if run_total_gain is not None else 'n/a'} "
                f"total_reward={float(run_aggregate.get('total_reward', 0.0)):+.4f} "
                f"trace={entry.trace_path}{status_tag}"
            )


def convert_type(value: str, type_hint: Any) -> Any:
    """
    Convert string value to the appropriate type based on type hint.

    Args:
        value: String value from CLI
        type_hint: Type annotation from parameter

    Returns:
        Converted value
    """
    if type_hint is None:
        return value

    origin = get_origin(type_hint)
    if origin in (UnionType, Union):
        non_none_args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        if len(non_none_args) == 1:
            return convert_type(value, non_none_args[0])
        return value

    # Handle string type
    if type_hint is str:
        return value

    # Handle int type
    if type_hint is int:
        return int(value)

    # Handle float type
    if type_hint is float:
        return float(value)

    # Handle bool type
    if type_hint is bool:
        if value.lower() in ("true", "1", "yes", "y"):
            return True
        elif value.lower() in ("false", "0", "no", "n"):
            return False
        else:
            raise ValueError(f"Invalid boolean value: {value}")

    if origin is list:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Invalid list value: {value}")
        return parsed

    if origin is dict:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid dict value: {value}")
        return parsed

    # Default to string if type is not recognized
    return value


def parse_namespace_args(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """
    Extract arguments with a specific namespace prefix.

    Args:
        args: Parsed arguments
        prefix: Namespace prefix (e.g., "system" or "task")

    Returns:
        Dict of parameter name -> value
    """
    result = {}
    prefix_with_dot = f"{prefix}."

    for key, value in vars(args).items():
        if key.startswith(prefix_with_dot):
            param_name = key[len(prefix_with_dot) :]
            # Convert hyphens to underscores for Python identifiers
            param_name = param_name.replace("-", "_")
            result[param_name] = value

    return result


def format_type_hint(type_hint: Any) -> str:
    """Format a type hint for CLI display."""
    if type_hint is None:
        return "Any"

    origin = get_origin(type_hint)
    if origin in (UnionType, Union):
        return " | ".join(format_type_hint(arg) for arg in get_args(type_hint))
    if origin is list:
        args = get_args(type_hint)
        inner = format_type_hint(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin is dict:
        args = get_args(type_hint)
        if len(args) == 2:
            return f"dict[{format_type_hint(args[0])}, {format_type_hint(args[1])}]"
        return "dict[Any, Any]"
    if type_hint is type(None):
        return "None"
    return getattr(type_hint, "__name__", str(type_hint).replace("typing.", ""))


def print_class_description(
    kind: str,
    name: str,
    cls: type,
    class_params: dict[str, Any],
) -> None:
    """Print a concise description of a task or system class."""
    print(f"{kind.title()}: {name}")
    print(f"Module: {cls.__module__}")

    doc = inspect.getdoc(cls)
    if doc:
        summary = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        if summary:
            print(f"Summary: {summary}")

    print("\nParameters:")
    if not class_params:
        print("  (none)")
    else:
        for param_name, param_info in class_params.items():
            line = f"  - {param_name}: {format_type_hint(param_info['type'])}"
            if param_info["required"]:
                line += " (required)"
            elif param_info["default"] != inspect.Parameter.empty:
                line += f" (default: {param_info['default']!r})"
            print(line)

    if kind == "task":
        schedules = list_task_schedules(name)
        if schedules:
            print("\nNamed schedules:")
            for schedule in schedules:
                print(f"  - {format_schedule_summary(schedule)}")

        variants = list_task_variants(name)
        if variants:
            print("\nNamed variants (reusable presets):")
            for variant in variants:
                print(
                    f"  - {variant.id}: {variant.display_name} - {variant.description}"
                )

        task_brief = _get_task_brief(cls, {})
        if task_brief:
            print("\nTask Brief:")
            for key in (
                "objective",
                "instance_unit",
                "reward_definition",
                "completion_definition",
            ):
                if key in task_brief:
                    label = key.replace("_", " ").title()
                    print(f"  {label}: {task_brief[key]}")
            constraints = task_brief.get("constraints", [])
            if constraints:
                print("  Constraints:")
                for constraint in constraints:
                    print(f"    - {constraint}")


def create_dynamic_parser(
    parser: argparse.ArgumentParser, class_params: dict[str, Any], prefix: str
) -> None:
    """
    Add dynamic arguments to parser based on class parameters.

    Args:
        parser: ArgumentParser to add arguments to
        class_params: Parameter metadata from get_class_params()
        prefix: Namespace prefix (e.g., "system" or "task")
    """
    for param_name, param_info in class_params.items():
        # Convert underscore to hyphen for CLI
        cli_name = param_name.replace("_", "-")
        arg_name = f"--{prefix}.{cli_name}"

        # Determine type for argparse
        param_type = param_info["type"]
        if param_type is bool:
            default_value = param_info.get("default", False)
            parser.add_argument(
                arg_name,
                type=str,
                help=f"{prefix} parameter: {param_name} (default: {default_value})",
            )
        else:
            # For other types, use type conversion
            help_text = f"{prefix} parameter: {param_name}"
            if param_info["default"] != inspect.Parameter.empty:
                help_text += f" (default: {param_info['default']})"

            parser.add_argument(
                arg_name,
                type=str,  # Parse as string, convert later
                help=help_text,
            )


def resolve_parameters(
    class_params: dict[str, Any],
    config_params: dict[str, Any],
    cli_params: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    """
    Resolve parameters using precedence: defaults < config < CLI.

    Args:
        class_params: Parameter metadata with defaults
        config_params: Parameters from config file
        cli_params: Parameters from CLI arguments

    Returns:
        Resolved parameters dict
    """
    result = {}
    unsupported_environment_keys = [
        key
        for key in {*config_params, *cli_params}
        if is_environment_backend_param(key) and key not in class_params
    ]
    if namespace == "task" and unsupported_environment_keys:
        raise ValueError(
            f"Task does not support {unsupported_environment_keys[0]}. Daytona is only available for shell/workspace tasks in this release."
        )

    for param_name, param_info in class_params.items():
        # Start with default
        if param_info["default"] != inspect.Parameter.empty:
            value = param_info["default"]
        else:
            value = None

        # Override with config
        if param_name in config_params:
            value = config_params[param_name]

        # Override with CLI (only if not None)
        if param_name in cli_params and cli_params[param_name] is not None:
            cli_value = cli_params[param_name]
            # Convert type if needed
            if isinstance(cli_value, str) and param_info["type"] is not None:
                try:
                    value = convert_type(cli_value, param_info["type"])
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid {namespace} parameter '{param_name}': {e}"
                    ) from e
            else:
                value = cli_value

        # Only include if not None or required
        if value is not None or param_info["required"]:
            if namespace == "task" and param_name == "environment_backend":
                validate_environment_backend(value)
            result[param_name] = value

    return result


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Config dict with 'system' and 'task' sections
    """
    path = Path(config_path)
    if not path.exists():
        _exit_cli_error(
            "Config file not found.",
            details=[f"Path: {config_path}"],
            hint="Check the path passed to --config and try again.",
        )

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _exit_cli_error(
            "Config file contains invalid JSON.",
            details=[f"Path: {config_path}", str(exc)],
            hint="Fix the JSON syntax or validate the file with a formatter.",
            exc=exc,
        )


def _normalize_task_fixture_params(task_params: dict[str, Any]) -> dict[str, Any]:
    """Map legacy task fixture keys onto the canonical schedule naming."""
    normalized = dict(task_params)
    legacy_schedule = normalized.pop("rollout_schedule", None)
    if legacy_schedule is not None and normalized.get("schedule") is None:
        normalized["schedule"] = legacy_schedule
    return normalized


def _cmd_list(args: list[str]) -> None:
    """clbench list [tasks|systems]"""
    parser = argparse.ArgumentParser(prog="clbench list")
    parser.add_argument(
        "kind",
        nargs="?",
        choices=["tasks", "systems"],
        default=None,
        help="What to list. Omit to list both.",
    )
    parsed = parser.parse_args(args)

    if parsed.kind in (None, "tasks"):
        tasks = list_tasks()
        print("Tasks:")
        for t in tasks:
            print(f"  {t}")

    if parsed.kind in (None, "systems"):
        if parsed.kind is None:
            print()
        systems = list_systems()
        print("Systems:")
        for s in systems:
            print(f"  {s}")


def _cmd_inspect(args: list[str]) -> None:
    """clbench inspect task|system <name>"""
    parser = argparse.ArgumentParser(prog="clbench inspect")
    parser.add_argument("kind", choices=["task", "system"])
    parser.add_argument("name")
    parsed = parser.parse_args(args)

    try:
        if parsed.kind == "task":
            cls = get_task_class(parsed.name)
            print_class_description("task", parsed.name, cls, get_class_params(cls))
        else:
            cls = get_system_class(parsed.name)
            print_class_description("system", parsed.name, cls, get_class_params(cls))
    except KeyError:
        _exit_unknown_registry_entry(parsed.kind, parsed.name)


_RUN_VALUE_FLAGS = {
    "--system",
    "--task",
    "--config",
    "--output",
    "--system-params",
    "--task-params",
    "--runs",
    "-k",
    "--max-workers",
    "--run-mode",
    "--run-group-id",
    "--verbosity",
}

_RUN_BOOLEAN_FLAGS = {
    "--dry-run",
    "--live-dashboard",
    "--no-live-dashboard",
    "--live-server",
    "--no-live-server",
    "--verbose-runs",
    "--no-verbose-runs",
    "-v",
    "-vv",
}


def _extract_run_verbosity(args: list[str]) -> int:
    """Best-effort verbosity extraction before full argparse setup."""
    verbosity = 0
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--verbosity" and i + 1 < len(args):
            try:
                verbosity = max(verbosity, int(args[i + 1]))
            except ValueError:
                pass
            i += 2
            continue
        if arg.startswith("--verbosity="):
            try:
                verbosity = max(verbosity, int(arg.split("=", 1)[1]))
            except ValueError:
                pass
        elif arg.startswith("-v") and set(arg[1:]) == {"v"}:
            verbosity = max(verbosity, len(arg) - 1)
        i += 1
    return verbosity


def _run_arg_expects_value(arg: str) -> bool:
    """Return True when the given run subcommand flag consumes the next token."""
    if "=" in arg or arg in _RUN_BOOLEAN_FLAGS:
        return False
    if arg in _RUN_VALUE_FLAGS:
        return True
    return arg.startswith("--system.") or arg.startswith("--task.")


def _cmd_run(args: list[str]) -> None:
    """clbench run <task> --system <sys> [--schedule <sched>] [--task.x v] [--system.x v]

    Translates the ergonomic positional+shorthand args into the full flag format
    and delegates to _cmd_run_impl() for parameter resolution and execution.
    """
    if "--debug" in args:
        _configure_cli_logging(True, verbosity=_extract_run_verbosity(args))

    translated: list[str] = []
    i = 0
    task_consumed = False
    expecting_flag_value = False

    while i < len(args):
        arg = args[i]
        if expecting_flag_value:
            translated.append(arg)
            expecting_flag_value = False
            i += 1
            continue
        # Positional task name (first non-flag token)
        if not arg.startswith("-") and not task_consumed:
            translated += ["--task", arg]
            task_consumed = True
            i += 1
            continue
        # --schedule <id> is shorthand for --task.schedule <id>
        if arg in ("--schedule",) and i + 1 < len(args):
            i += 1
            translated += ["--task.schedule", args[i]]
            i += 1
            continue
        translated.append(arg)
        if _run_arg_expects_value(arg):
            expecting_flag_value = True
        i += 1

    logger.debug("Translated run args: %s", translated)

    # Inject translated args and call the run implementation.
    original = sys.argv[:]
    sys.argv = [sys.argv[0]] + translated
    try:
        _cmd_run_impl()
    finally:
        sys.argv = original


def _cmd_smoke(args: list[str]) -> None:
    """clbench smoke <task> --system <sys> [--schedule <sched>]

    Runs one interaction end-to-end as a fast wiring check. Uses the first
    available schedule for the task, or falls back to num_instances=1 with
    no schedule. Skips the live dashboard and trace storage.
    """
    parser = argparse.ArgumentParser(
        prog="clbench smoke",
        description="Run one interaction end-to-end as a wiring check.",
    )
    parser.add_argument("task_name", help="Task to smoke-test.")
    parser.add_argument("--system", "-s", required=True, help="System to use.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mirror enabled logs to stderr and show Python tracebacks on failure.",
    )
    parser.add_argument(
        "-v",
        dest="verbosity_count",
        action="count",
        default=0,
        help="Increase run log verbosity. Repeat as -vv for full developer diagnostics.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Set run log verbosity explicitly (0=warnings/errors, 1=info, 2=debug).",
    )
    parser.add_argument(
        "--schedule", help="Schedule to use (defaults to first available)."
    )
    parser.add_argument(
        "--system.model",
        dest="system_model",
        default=None,
        help="Override system model (e.g. gpt-4o-mini).",
    )
    parsed, extra = parser.parse_known_args(args)
    requested_verbosity = max(parsed.verbosity, parsed.verbosity_count)
    if requested_verbosity < 0:
        parser.error("--verbosity must be >= 0")
    _configure_cli_logging(parsed.debug, verbosity=requested_verbosity)
    logger.debug("Smoke args: %s", vars(parsed))

    try:
        task_class = get_task_class(parsed.task_name)
    except KeyError:
        _exit_unknown_registry_entry("task", parsed.task_name)
    try:
        system_class = get_system_class(parsed.system)
    except KeyError:
        _exit_unknown_registry_entry("system", parsed.system)

    # Pick a schedule: use --schedule if given, else first available, else None.
    schedule_id = parsed.schedule
    if not schedule_id:
        from .tasks.schedules import list_task_schedules

        available = list_task_schedules(parsed.task_name)
        if available:
            schedule_id = available[0].id
            print(f"No --schedule specified; using first available: {schedule_id}")

    task_params: dict = {}
    if schedule_id:
        task_params["schedule"] = schedule_id
    else:
        task_params["num_instances"] = 1

    system_params: dict = {}
    if parsed.system_model:
        system_params["model"] = parsed.system_model

    print(f"Smoke test: {parsed.task_name} + {parsed.system}")
    if schedule_id:
        print(f"  schedule: {schedule_id}")
    print(f"  task params:   {task_params}")
    print(f"  system params: {system_params}")
    print()

    try:
        task = task_class(**task_params)
        system = system_class(**system_params)
        print("  [ok] task and system instantiate")

        query = task.reset()
        print("  [ok] task.reset() returned a Query")

        system.reset()
        print("  [ok] system.reset() succeeded")

        response = system.respond(query)
        print("  [ok] system.respond() returned a Response")

        result = task.step(response)
        print(f"  [ok] task.step() succeeded (done={result.done})")

        if not result.done:
            # Mirror the real runtime loop (see src/runtime/runner.py):
            # some systems rely on observe() to finalize per-turn state
            # before the next respond() call.
            system.observe(result.observation, result.next_query)
            response2 = system.respond(result.next_query)
            result2 = task.step(response2)
            print(f"  [ok] second step succeeded (done={result2.done})")

        print()
        print("Smoke test passed.")
    except NotImplementedError as e:
        _exit_cli_error(
            "Smoke test hit a NotImplementedError.",
            details=[f"Task: {parsed.task_name}", f"System: {parsed.system}", str(e)],
            hint="Implement the missing method shown above and run the smoke test again.",
            exc=e,
        )
    except Exception as e:
        _exit_cli_error(
            "Smoke test failed.",
            details=[
                f"Task: {parsed.task_name}",
                f"System: {parsed.system}",
                f"{type(e).__name__}: {e}",
            ],
            hint=(
                "Add --debug for extra CLI setup logs."
                if not parsed.debug
                else "Review the traceback above for the failure source."
            ),
            exc=e,
            show_traceback=True,
        )


def _find_clbench_cmd() -> list[str]:
    """Return a command prefix to invoke clbench in a subprocess."""
    import shutil

    clbench = shutil.which("clbench")
    if clbench:
        return [clbench]
    repo_root = Path(__file__).resolve().parent.parent
    run_script = repo_root / "run_benchmark.py"
    if run_script.exists():
        return [sys.executable, str(run_script)]
    _exit_cli_error(
        "Could not locate a clbench entry point.",
        hint="Run from an environment where 'clbench' is installed or run_benchmark.py is present.",
    )


def _extract_run_all_headline(summary_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull the four leaderboard metrics out of a per-task run summary."""
    aggregate = summary_dict.get("aggregate") or {}
    runs = summary_dict.get("runs") or []

    gain_values = [
        float(v)
        for v in aggregate.get("mean_gain_by_index", [])
        if isinstance(v, (int, float))
    ]
    cumulative_gain = round(sum(gain_values), 6) if gain_values else None

    total_rewards: list[float] = []
    total_costs: list[float] = []
    total_latencies: list[float] = []
    for run in runs:
        run_agg = (run or {}).get("aggregate") or {}
        if isinstance(run_agg.get("total_reward"), (int, float)):
            total_rewards.append(float(run_agg["total_reward"]))
        if isinstance(run_agg.get("total_cost"), (int, float)):
            total_costs.append(float(run_agg["total_cost"]))
        execution = (run or {}).get("execution") or {}
        usage = (execution.get("usage") or {}).get("interaction") or {}
        timing = execution.get("timing") or {}
        latency = timing.get("response_latency_seconds_total")
        if isinstance(latency, (int, float)):
            total_latencies.append(float(latency))
        if not total_costs and isinstance(usage.get("cost_usd"), (int, float)):
            total_costs.append(float(usage["cost_usd"]))

    mean_reward = (
        round(sum(total_rewards) / len(total_rewards), 6) if total_rewards else None
    )
    mean_cost = round(sum(total_costs) / len(total_costs), 10) if total_costs else None

    mean_latency_by_index = aggregate.get("mean_latency_by_index") or []
    latency_values = [
        float(v) for v in mean_latency_by_index if isinstance(v, (int, float))
    ]
    mean_latency = (
        round(sum(latency_values) / len(latency_values), 6) if latency_values else None
    )

    return {
        "cumulative_gain": cumulative_gain,
        "total_reward": mean_reward,
        "mean_latency_seconds": mean_latency,
        "total_cost_usd": mean_cost,
    }


def _read_log_tail(path: Path, max_lines: int = 60) -> str:
    """Return the trailing portion of a per-task log file for the failure block."""
    try:
        with open(path, "r") as fh:
            data = fh.read()
    except OSError:
        return ""
    lines = data.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _slugify_run_name(value: str) -> str:
    """Return a filesystem-safe run name that remains readable."""
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug.strip("._-")


def _is_gzip_path(path: Path) -> bool:
    return path.suffix == ".gz"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_gzip_path(path):
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        path.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
        return
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _persist_full_trace_outputs(
    *,
    task_name: str,
    run_group_id: str,
    baseline_trace_data: dict[str, Any] | None,
    run_trace_data_list: list[dict[str, Any]],
    root_dir: Path,
) -> tuple[str | None, list[str]]:
    """Write standalone trace JSON files and sidecar system artifacts.

    The viewer artifact remains the portable single-file bundle, but these files
    keep the full trace/artifact tree discoverable under ``results/<task>``.
    """
    run_id_component = safe_run_id_component(run_group_id)
    trace_dir = Path("results") / task_name / "traces" / run_id_component
    baseline_trace_path: str | None = None
    run_trace_paths: list[str] = []

    def _compact_trace_payload(trace_data: dict[str, Any]) -> dict[str, Any]:
        core_trace = dict(trace_data)
        core_trace["system_artifacts"] = None
        instance_traces = core_trace.get("instance_traces")
        if isinstance(instance_traces, list):
            core_trace["instance_traces"] = [
                {**trace, "system_artifacts": None}
                if isinstance(trace, dict)
                else trace
                for trace in instance_traces
            ]
        return core_trace

    def _drop_inline_system_artifacts(trace_data: dict[str, Any]) -> None:
        trace_data["system_artifacts"] = None
        instance_traces = trace_data.get("instance_traces")
        if not isinstance(instance_traces, list):
            return
        for trace in instance_traces:
            if isinstance(trace, dict):
                trace["system_artifacts"] = None

    def _write_trace_file(path: Path, trace_data: dict[str, Any]) -> str:
        raw_artifacts = trace_data.get("system_artifacts")
        _write_json(path, _compact_trace_payload(trace_data))
        save_artifacts(raw_artifacts, path)
        # Keep the later viewer artifact compact too; it shares these objects.
        _drop_inline_system_artifacts(trace_data)
        return _relpath(path, root_dir)

    if baseline_trace_data is not None:
        baseline_trace_path = _write_trace_file(
            trace_dir / "baseline.json",
            baseline_trace_data,
        )

    for i, trace_data in enumerate(run_trace_data_list):
        run_trace_paths.append(
            _write_trace_file(
                trace_dir / f"run_{i:04d}.json",
                trace_data,
            )
        )

    return baseline_trace_path, run_trace_paths


def _load_json_file(path: Path) -> dict[str, Any]:
    if _is_gzip_path(path):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        with open(path) as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _task_entry_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("task")): entry
        for entry in manifest.get("tasks", [])
        if isinstance(entry, dict) and entry.get("task")
    }


def _final_task_artifact_exists(
    *,
    final_run_dir: Path,
    task_name: str,
    task_entry: dict[str, Any] | None,
) -> bool:
    if task_entry is None:
        return False
    if task_entry.get("status") != "ok":
        return False
    task_dir = final_run_dir / "tasks"
    return (task_dir / f"{task_name}.json.gz").exists() or (
        task_dir / f"{task_name}.json"
    ).exists()


def _archive_existing_run_all_task_outputs(
    *,
    final_run_dir: Path,
    task_name: str,
    batch_id: str,
) -> None:
    """Move replaced task/log files aside so overwrite-task never destroys data."""
    archive_dir = final_run_dir / "archive" / batch_id / task_name
    moved = False
    for source in [
        final_run_dir / "tasks" / f"{task_name}.json.gz",
        final_run_dir / "tasks" / f"{task_name}.json",
        final_run_dir / "logs" / f"{task_name}.log",
    ]:
        if source.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive_dir / source.name))
            moved = True
    if moved:
        marker = {
            "task": task_name,
            "archived_at_batch": batch_id,
            "reason": "overwritten by clbench run-all --overwrite-task",
        }
        _write_json(archive_dir / "archive_metadata.json", marker)


def _cmd_run_all(args: list[str]) -> None:
    """clbench run-all --system <sys> [--skip TASK ...] [--task-parallelism N] [--per-task-parallelism N] [extra system args]

    Runs a system against every registered task that has a default.json schedule,
    executing tasks in parallel. Each per-task subprocess writes its live manifest
    and traces to disk; run-all hosts a single HTTP server that exposes them all
    through viewers/run_all_viewer.html. Per-task stdout/stderr is tee-ed to a log file
    under logs/run-all/<run_id>/<task>.log so you can tail individual tasks live.
    Tasks without a default.json schedule are skipped with a notice.
    Extra arguments (e.g. --system.model gpt-4o) are forwarded to every task run.

    Parallelism is controlled by two independent knobs:

    * ``--task-parallelism N`` (alias ``-j``, default 4) caps how many task
      subprocesses run concurrently here in run-all.
    * ``--per-task-parallelism N`` is forwarded to each inner ``clbench run``
      subprocess as ``--max-workers N``, capping concurrent rollout/baseline
      workers per task. Useful for staying under provider rate limits
      (e.g. Anthropic). Defaults to whatever the schedule specifies.
    """
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        prog="clbench run-all",
        description="Run a system on all tasks using each task's default.json schedule.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help=(
            "Required human-readable name for this suite run. The canonical "
            "artifact directory is final_results/runs/<name>."
        ),
    )
    parser.add_argument("--system", "-s", required=True, help="System to run.")
    parser.add_argument(
        "--task",
        "-t",
        nargs="*",
        default=None,
        metavar="TASK",
        help="Restrict the run to the specified task names (default: all tasks).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="TASK",
        help="Task names to skip.",
    )
    parser.add_argument(
        "--task-parallelism",
        "-j",
        dest="task_parallelism",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Max tasks to run in parallel as separate subprocesses (default: 4). "
            "Use -j 1 to run tasks sequentially."
        ),
    )
    parser.add_argument(
        "--per-task-parallelism",
        dest="per_task_parallelism",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Forwarded to each inner `clbench run` subprocess as `--max-workers`. "
            "Caps concurrent rollout/baseline workers per task (defaults to the "
            "schedule, otherwise 12). Useful for capping provider concurrency, "
            "e.g. Anthropic rate limits."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve params only; do not run."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show debug logs on failure."
    )
    parser.add_argument(
        "-v",
        dest="verbosity_count",
        action="count",
        default=0,
        help="Increase per-task run log verbosity. Repeat as -vv.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Set per-task run log verbosity explicitly.",
    )
    parser.add_argument(
        "--live-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Host an aggregate dashboard for the suite (default: enabled). Use --no-live-dashboard to skip the HTTP server and live writes entirely.",
    )
    parser.add_argument(
        "--final-results-dir",
        default="final_results/runs",
        help=(
            "Directory where named run-all artifacts are written "
            "(default: final_results/runs)."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append new task results into an existing named final-results run. "
            "Errors if any selected task already has a completed artifact."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "For an existing named run, run only selected/default tasks that do "
            "not already have completed task artifacts."
        ),
    )
    parser.add_argument(
        "--overwrite-task",
        action="store_true",
        help=(
            "Allow replacing selected task results in an existing named run. "
            "Previous task artifacts/logs are archived, not deleted."
        ),
    )
    parsed, extra = parser.parse_known_args(args)
    requested_verbosity = max(parsed.verbosity, parsed.verbosity_count)
    if requested_verbosity < 0:
        parser.error("--verbosity must be >= 0")
    run_name_slug = _slugify_run_name(parsed.name)
    if not run_name_slug:
        parser.error("--name must contain at least one filesystem-safe character")
    append_mode = bool(parsed.append or parsed.missing_only or parsed.overwrite_task)
    _configure_cli_logging(parsed.debug, verbosity=requested_verbosity)

    all_tasks = list_tasks()
    requested_tasks = parsed.task
    if parsed.overwrite_task and not requested_tasks:
        parser.error("--overwrite-task requires an explicit --task list")
    if requested_tasks:
        unknown = [name for name in requested_tasks if name not in all_tasks]
        if unknown:
            _exit_unknown_registry_entry("task", unknown[0])
        candidate_tasks = list(requested_tasks)
    else:
        candidate_tasks = all_tasks

    tasks_to_run: list[str] = []
    skipped: list[tuple[str, str]] = []

    for task_name in candidate_tasks:
        if task_name in (parsed.skip or []):
            skipped.append((task_name, "explicitly skipped via --skip"))
            continue
        available = list_task_schedules(task_name)
        if not any(s.id == "default" for s in available):
            skipped.append((task_name, "no default.json schedule"))
            continue
        tasks_to_run.append(task_name)

    if not tasks_to_run:
        print("No tasks with a default.json schedule found.")
        if skipped:
            print("Skipped:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")
        return

    task_parallelism = parsed.task_parallelism
    per_task_parallelism = parsed.per_task_parallelism
    clbench_cmd = _find_clbench_cmd()
    repo_root = Path(__file__).resolve().parent.parent

    run_id = new_timestamp_run_id("run-all")
    run_id_component = safe_run_id_component(run_id)
    started_at = datetime.now(timezone.utc).isoformat()

    final_results_root = Path(parsed.final_results_dir)
    if not final_results_root.is_absolute():
        final_results_root = repo_root / final_results_root
    final_run_dir = final_results_root / run_name_slug
    aggregate_manifest_path = final_run_dir / "manifest.json"
    existing_state: dict[str, Any] | None = None
    if final_run_dir.exists() and append_mode:
        if not aggregate_manifest_path.exists():
            parser.error(f"cannot append to {final_run_dir}: manifest.json is missing")
        try:
            existing_state = _load_json_file(aggregate_manifest_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"cannot read existing run manifest: {exc}")
        if existing_state.get("kind") != "run_all_manifest":
            parser.error(
                f"cannot append to non-run-all manifest: {aggregate_manifest_path}"
            )
        if existing_state.get("system") != parsed.system:
            parser.error(
                "cannot append with a different system: "
                f"existing={existing_state.get('system')!r}, requested={parsed.system!r}"
            )
        if list(existing_state.get("system_args") or []) != list(extra):
            parser.error(
                "cannot append with different forwarded system/task args. "
                f"existing={existing_state.get('system_args')!r}, requested={extra!r}"
            )
    elif final_run_dir.exists() and not parsed.dry_run:
        parser.error(
            f"final results run already exists: {final_run_dir}. Choose a new --name."
        )
    elif append_mode and not final_run_dir.exists():
        parser.error(
            f"cannot append: final results run does not exist: {final_run_dir}"
        )

    existing_entries = _task_entry_by_name(existing_state or {})
    existing_completed = {
        name
        for name, entry in existing_entries.items()
        if _final_task_artifact_exists(
            final_run_dir=final_run_dir, task_name=name, task_entry=entry
        )
    }
    if parsed.missing_only:
        tasks_to_run = [name for name in tasks_to_run if name not in existing_completed]
        for name in sorted(existing_completed):
            if name in candidate_tasks:
                skipped.append((name, "already present in final results"))
    elif append_mode and not parsed.overwrite_task:
        collisions = [name for name in tasks_to_run if name in existing_completed]
        if collisions:
            parser.error(
                "selected task(s) already have completed artifacts: "
                f"{', '.join(collisions)}. Use --missing-only or --overwrite-task."
            )

    if not tasks_to_run:
        print("No tasks to run.")
        if skipped:
            print("Skipped:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")
        return

    if parsed.dry_run:
        print("Dry run successful.")
        print(f"Run name: {parsed.name}")
        print(f"System: {parsed.system}")
        print(f"Final results dir: {final_run_dir}")
        print(f"Tasks to run ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
        if per_task_parallelism is not None:
            per_task_msg = f", up to {per_task_parallelism} worker(s) per task"
        else:
            per_task_msg = " (per-task workers from schedule)"
        print(
            f"Running {len(tasks_to_run)} task(s) with up to {task_parallelism} "
            f"parallel task subprocess(es){per_task_msg}..."
        )
        if append_mode:
            print(
                "Append mode: "
                + (
                    "overwrite-task"
                    if parsed.overwrite_task
                    else "missing-only"
                    if parsed.missing_only
                    else "append"
                )
            )
        if skipped:
            print("Skipped:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")
        return

    if not parsed.dry_run:
        final_run_dir.mkdir(parents=True, exist_ok=True)

    tasks_artifact_dir = final_run_dir / "tasks"
    logs_dir = final_run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks_artifact_dir.mkdir(parents=True, exist_ok=True)

    def _relpath(p: Path) -> str:
        try:
            return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return str(p)

    # Predict per-task live manifest paths (we control the run_group_id, so we
    # can compute these before each subprocess starts).
    task_run_ids: dict[str, str] = {name: run_id for name in tasks_to_run}
    task_live_manifests: dict[str, Path] = {
        name: build_live_manifest_path(name, task_run_ids[name])
        for name in tasks_to_run
    }
    task_log_paths: dict[str, Path] = {
        name: logs_dir / f"{name}.log" for name in tasks_to_run
    }
    batch_id = run_id_component
    if append_mode and not parsed.dry_run:
        for name in tasks_to_run:
            if (tasks_artifact_dir / f"{name}.json").exists() or (
                logs_dir / f"{name}.log"
            ).exists():
                _archive_existing_run_all_task_outputs(
                    final_run_dir=final_run_dir,
                    task_name=name,
                    batch_id=batch_id,
                )

    manifest_lock = threading.Lock()
    pending_entries = [
        {
            "task": name,
            "schedule": "default",
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "duration_seconds": None,
            "log_path": _relpath(task_log_paths[name]),
            "live_manifest_path": _relpath(task_live_manifests[name]),
            "summary_path": None,
            "final_artifact_path": _relpath(tasks_artifact_dir / f"{name}.json.gz"),
            "batch_id": batch_id,
            "headline": {
                "cumulative_gain": None,
                "total_reward": None,
                "mean_latency_seconds": None,
                "total_cost_usd": None,
            },
        }
        for name in tasks_to_run
    ]
    if existing_state is not None:
        aggregate_state = dict(existing_state)
        replaced_tasks = (
            set(tasks_to_run)
            if (parsed.overwrite_task or parsed.missing_only)
            else set()
        )
        retained_tasks = [
            entry
            for entry in aggregate_state.get("tasks", [])
            if not (
                replaced_tasks
                and isinstance(entry, dict)
                and entry.get("task") in replaced_tasks
            )
        ]
        aggregate_state.update(
            {
                "run_name": parsed.name,
                "run_slug": run_name_slug,
                "system": parsed.system,
                "system_args": extra,
                "finished_at": None,
                "status": "running",
                "poll_interval_ms": 2000,
                "manifest_path": _relpath(aggregate_manifest_path),
                "final_results_dir": _relpath(final_run_dir),
                "logs_dir": _relpath(logs_dir),
                "current_batch_id": batch_id,
            }
        )
        aggregate_state["tasks"] = retained_tasks + pending_entries
    else:
        aggregate_state = {
            "kind": "run_all_manifest",
            "run_name": parsed.name,
            "run_slug": run_name_slug,
            "run_group_id": run_id,
            "system": parsed.system,
            "system_args": extra,
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "poll_interval_ms": 2000,
            "manifest_path": _relpath(aggregate_manifest_path),
            "final_results_dir": _relpath(final_run_dir),
            "logs_dir": _relpath(logs_dir),
            "current_batch_id": batch_id,
            "tasks": pending_entries,
        }

    def _persist_manifest() -> None:
        write_run_all_manifest(aggregate_manifest_path, aggregate_state)

    def _update_task_entry(task_name: str, **fields: Any) -> None:
        with manifest_lock:
            current_batch_id = aggregate_state.get("current_batch_id")
            matching_entries = [
                entry
                for entry in aggregate_state["tasks"]
                if isinstance(entry, dict) and entry.get("task") == task_name
            ]
            target_entry = next(
                (
                    entry
                    for entry in matching_entries
                    if entry.get("batch_id") == current_batch_id
                ),
                matching_entries[0] if matching_entries else None,
            )
            if target_entry is not None:
                target_entry.update(fields)
            _persist_manifest()

    with manifest_lock:
        _persist_manifest()

    # Optional shared HTTP server.
    live_viewer_url: str | None = None
    if parsed.live_dashboard and not parsed.dry_run:
        try:
            _server, _thread, live_viewer_url = start_live_dashboard_server(
                root_dir=repo_root,
                manifest_path=aggregate_manifest_path,
                viewer_path="viewers/run_all_viewer.html",
            )
        except OSError as exc:
            print(f"Aggregate dashboard server could not start: {exc}")
        else:
            print(f"Live dashboard: {live_viewer_url}")

    run_flags: list[str] = []
    if parsed.live_dashboard:
        run_flags += ["--live-dashboard", "--no-live-server"]
    else:
        run_flags += ["--no-live-dashboard"]
    if parsed.dry_run:
        run_flags.append("--dry-run")
    if parsed.debug:
        run_flags.append("--debug")
    if requested_verbosity:
        run_flags.append(f"-{'v' * requested_verbosity}")
    if per_task_parallelism is not None:
        run_flags += ["--max-workers", str(per_task_parallelism)]
    run_flags.extend(extra)

    if per_task_parallelism is not None:
        per_task_msg = f", up to {per_task_parallelism} worker(s) per task"
    else:
        per_task_msg = " (per-task workers from schedule)"
    print(
        f"Running {len(tasks_to_run)} task(s) with up to {task_parallelism} parallel "
        f"task subprocess(es){per_task_msg}..."
    )
    print(f"Run id: {run_id}")
    print(f"Run name: {parsed.name}")
    print(f"Final results dir: {_relpath(final_run_dir)}")
    print(f"Aggregate manifest: {_relpath(aggregate_manifest_path)}")
    print(f"Logs dir: {_relpath(logs_dir)}")
    for name, reason in skipped:
        print(f"  [skip] {name}: {reason}")
    print()

    def _maybe_load_summary(task_name: str) -> Path | None:
        """Find the per-task viewer artifact written for this run group id."""
        # The subprocess writes results/<task>/viewer_artifact_<run_id>_<ts>.json.gz.
        # Legacy uncompressed .json artifacts are also accepted for older runs.
        results_dir = repo_root / "results" / task_name
        task_run_component = safe_run_id_component(task_run_ids[task_name])
        artifacts = sorted(
            {
                *results_dir.glob(f"viewer_artifact_{task_run_component}_*.json"),
                *results_dir.glob(f"viewer_artifact_{task_run_component}_*.json.gz"),
            }
        )
        return artifacts[-1] if artifacts else None

    def _run_one(task_name: str) -> tuple[str, int, float]:
        import time

        log_path = task_log_paths[task_name]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        _update_task_entry(
            task_name,
            status="running",
            started_at=started,
        )
        print(f"  [start] {task_name}  → {_relpath(log_path)}")
        t0 = time.monotonic()
        cmd = (
            clbench_cmd
            + [
                "run",
                task_name,
                "--system",
                parsed.system,
                "--schedule",
                "default",
                "--run-group-id",
                task_run_ids[task_name],
            ]
            + run_flags
        )
        with open(log_path, "w") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.monotonic() - t0

        finished = datetime.now(timezone.utc).isoformat()
        summary_path = _maybe_load_summary(task_name) if proc.returncode == 0 else None
        headline: dict[str, Any] = {
            "cumulative_gain": None,
            "total_reward": None,
            "mean_latency_seconds": None,
            "total_cost_usd": None,
        }
        if summary_path is not None:
            try:
                raw = _load_json_file(summary_path)
                payload = raw.get("summary", raw) if isinstance(raw, dict) else {}
                headline = _extract_run_all_headline(payload)
            except (OSError, json.JSONDecodeError, ValueError):
                pass

        _update_task_entry(
            task_name,
            status="ok" if proc.returncode == 0 else "failed",
            finished_at=finished,
            exit_code=proc.returncode,
            duration_seconds=round(elapsed, 2),
            summary_path=_relpath(summary_path) if summary_path else None,
            headline=headline,
        )
        return task_name, proc.returncode, elapsed

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=task_parallelism) as executor:
        futures = {executor.submit(_run_one, name): name for name in tasks_to_run}
        for future in as_completed(futures):
            task_name, code, elapsed = future.result()
            results[task_name] = code
            status = "ok  " if code == 0 else "FAIL"
            print(f"  [{status}] {task_name} ({elapsed:.0f}s)")

    finished_at = datetime.now(timezone.utc).isoformat()
    passed = [n for n in tasks_to_run if results[n] == 0]
    failed = [n for n in tasks_to_run if results[n] != 0]
    any_failed = bool(failed) or any(
        isinstance(entry, dict) and entry.get("status") == "failed"
        for entry in aggregate_state.get("tasks", [])
    )
    overall_status = "completed" if not any_failed else "completed_with_failures"
    with manifest_lock:
        aggregate_state["finished_at"] = finished_at
        aggregate_state["status"] = overall_status
        _persist_manifest()

    # Build per-task viewer artifacts and write them to a separate
    # task_artifacts.json alongside the manifest.  Keeping them out of
    # manifest.json avoids bloating it to hundreds of MB.
    existing_task_artifacts = aggregate_state.get("task_artifacts") or {}
    task_artifacts: dict[str, Any] = (
        dict(existing_task_artifacts)
        if isinstance(existing_task_artifacts, dict)
        else {}
    )
    for entry in aggregate_state["tasks"]:
        name = entry["task"]
        sp = entry.get("summary_path")
        if not sp:
            task_artifacts[name] = None
            continue
        try:
            source_path = Path(sp)
            if not source_path.is_absolute():
                source_path = repo_root / source_path
            task_artifacts[name] = _load_json_file(source_path)
            final_task_path = tasks_artifact_dir / f"{name}.json.gz"
            if source_path.resolve() != final_task_path.resolve():
                if _is_gzip_path(source_path):
                    shutil.copy2(source_path, final_task_path)
                else:
                    _write_json(final_task_path, task_artifacts[name])
            entry["raw_summary_path"] = sp
            entry["summary_path"] = _relpath(final_task_path)
            entry["final_artifact_path"] = _relpath(final_task_path)
        except (OSError, json.JSONDecodeError, shutil.Error, ValueError):
            task_artifacts[name] = None
    task_artifacts_path = final_run_dir / "task_artifacts.json.gz"
    _write_json(task_artifacts_path, task_artifacts)
    with manifest_lock:
        aggregate_state["task_artifacts"] = None
        aggregate_state.pop("task_artifacts_gz_b64", None)
        aggregate_state["task_artifacts_path"] = _relpath(task_artifacts_path)
        _persist_manifest()

    batch_record = {
        "batch_id": batch_id,
        "run_group_id": run_id,
        "mode": (
            "overwrite_task"
            if parsed.overwrite_task
            else "missing_only"
            if parsed.missing_only
            else "append"
            if parsed.append
            else "initial"
        ),
        "tasks_run": tasks_to_run,
        "tasks_passed": passed,
        "tasks_failed": failed,
        "tasks_skipped": [{"task": name, "reason": reason} for name, reason in skipped],
        "started_at": started_at,
        "finished_at": finished_at,
    }
    previous_batches: list[dict[str, Any]] = []
    metadata_path = final_run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            previous_metadata = _load_json_file(metadata_path)
            previous_batches = list(previous_metadata.get("batches") or [])
        except (OSError, json.JSONDecodeError, ValueError):
            previous_batches = []

    metadata = {
        "run_name": parsed.name,
        "run_slug": run_name_slug,
        "run_group_id": aggregate_state.get("run_group_id", run_id),
        "latest_batch_id": batch_id,
        "system": parsed.system,
        "system_args": extra,
        "tasks_requested": requested_tasks or "all",
        "tasks_run": [entry.get("task") for entry in aggregate_state.get("tasks", [])],
        "tasks_skipped": [{"task": name, "reason": reason} for name, reason in skipped],
        "started_at": aggregate_state.get("started_at", started_at),
        "finished_at": finished_at,
        "status": overall_status,
        "command_hint": "clbench run-all --name ... --system ...",
        "manifest_path": _relpath(aggregate_manifest_path),
        "batches": previous_batches + [batch_record],
    }
    aggregate_state["batches"] = metadata["batches"]
    aggregate_state["current_batch_id"] = batch_id
    _write_json(final_run_dir / "metadata.json", metadata)
    _persist_manifest()

    print(f"\n{'=' * 60}")
    print("RUN-ALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Passed ({len(passed)}):  {', '.join(passed) if passed else 'none'}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(n for n, _ in skipped)}")
    if live_viewer_url:
        print(f"Aggregate viewer: {live_viewer_url}")
    print(f"Aggregate manifest: {aggregate_manifest_path}")
    print(f"Final results dir: {final_run_dir}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            log_path = task_log_paths[name]
            tail = _read_log_tail(log_path)
            print(f"\n--- {name} (exit {results[name]}) — {_relpath(log_path)} ---")
            print(tail.strip() or "(no output)")
        # Give the browser a moment to pick up the final manifest before exit.
        if live_viewer_url:
            time.sleep(2)
        sys.exit(1)
    if live_viewer_url:
        time.sleep(2)


# Subcommand dispatch table.
_SUBCOMMANDS = {
    "list": _cmd_list,
    "run": _cmd_run,
    "run-all": _cmd_run_all,
    "inspect": _cmd_inspect,
    "smoke": _cmd_smoke,
    "init": None,  # loaded from src.commands.init
    "validate": None,  # loaded from src.commands.validate
    "doctor": None,  # loaded from src.commands.doctor
    "setup": None,  # loaded from src.commands.setup
}

_SUBCOMMAND_HELP = """\
usage: clbench <command> [options]

Commands:
  list      List available tasks and systems
  run       Run a benchmark
  run-all   Run a system on all tasks using each task's default.json schedule
  inspect   Show parameters, schedules, and variants for a task or system
  init      Scaffold a new task or system from a template
  validate  Validate a task or system implementation before running
  smoke     Run one interaction end-to-end as a wiring check
  doctor    Check environment prerequisites
  setup     Run one-time setup for a task (e.g. download datasets)

Debugging:
  Add --debug after 'run', 'run-all', or 'smoke' to show debug logs and Python tracebacks.

Run 'clbench <command> --help' for per-command help.
"""


def main():
    """Main CLI entry point — dispatches to subcommands or legacy mode."""
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        subcmd = sys.argv[1]
        rest = sys.argv[2:]

        if subcmd == "init":
            from .commands.init import cmd_init

            cmd_init(rest)
        elif subcmd == "validate":
            from .commands.validate import cmd_validate

            cmd_validate(rest)
        elif subcmd == "doctor":
            from .commands.doctor import cmd_doctor

            cmd_doctor(rest)
        elif subcmd == "setup":
            from .commands.setup import cmd_setup

            cmd_setup(rest)
        else:
            _SUBCOMMANDS[subcmd](rest)  # type: ignore[call-arg]
        return

    print(_SUBCOMMAND_HELP)
    sys.exit(1)


def _cmd_run_impl():
    """Core implementation for 'clbench run': resolves parameters and executes the benchmark."""
    # Create base parser
    parser = argparse.ArgumentParser(
        description="Run continual learning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (results auto-saved to results/<task>/)
  clbench run exploitable_poker --system human

  # Override parameters
  clbench run exploitable_poker --system icl --schedule quick_test \\
    --system.model gpt-4 --task.num-hands 100

  # Use config file
  clbench run --config configs/exploitable_poker/exploitable_poker_icl.json

  # Custom output location
  clbench run exploitable_poker --system icl --schedule quick_test \\
    --output results/my_experiment.json
        """,
    )

    # Core arguments
    parser.add_argument(
        "--system",
        type=str,
        help="System to use (e.g., 'human', 'icl')",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run (e.g., 'poker')",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mirror enabled logs to stderr and show Python tracebacks on failure.",
    )
    parser.add_argument(
        "-v",
        dest="verbosity_count",
        action="count",
        default=0,
        help="Increase run log verbosity. Repeat as -vv for full developer diagnostics.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Set run log verbosity explicitly (0=warnings/errors, 1=info, 2=debug).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve parameters and instantiate the selected system and task without running the benchmark.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output file path for results; only applies to single-run executions (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--system-params",
        type=str,
        help="System parameters as JSON string",
    )
    parser.add_argument(
        "--task-params",
        type=str,
        help="Task parameters as JSON string",
    )
    parser.add_argument(
        "--runs",
        "-k",
        type=int,
        default=None,
        help="Number of independent benchmark runs (defaults to the selected schedule, otherwise 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent benchmark-run workers (defaults to the selected schedule, otherwise 1)",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=[m.value for m in RunMode],
        default=None,
        help="Variance mode across repeated runs (defaults to the selected schedule, otherwise permute)",
    )
    parser.add_argument(
        "--verbose-runs",
        dest="verbose_runs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit incremental per-run progress logs during execution.",
    )
    parser.add_argument(
        "--live-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write live benchmark snapshots and start a local viewer server for progress polling (default: enabled). Use --no-live-dashboard to disable.",
    )
    parser.add_argument(
        "--live-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --live-dashboard is enabled, also start a local HTTP server for the viewer (default: enabled). Use --no-live-server to write live files without binding a port (e.g. when run-all hosts a single shared server).",
    )
    parser.add_argument(
        "--run-group-id",
        type=str,
        default=None,
        help="Override the auto-generated timestamp run group id (used by run-all to predict per-task live manifest paths).",
    )

    # Parse known args first to check for list commands
    args, unknown = parser.parse_known_args()
    requested_verbosity = max(args.verbosity, args.verbosity_count)
    _configure_cli_logging(args.debug, verbosity=requested_verbosity)
    logger.debug("Run args (pre-dynamic-parse): %s", vars(args))

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.debug("Loaded config from %s", args.config)

    system_name = args.system or config.get("system", {}).get("name")
    task_name = args.task or config.get("task", {}).get("name")

    if not system_name or not task_name:
        parser.error("Must specify --system and --task, or provide --config with both")

    # Get classes
    try:
        system_class = get_system_class(system_name)
    except KeyError:
        _exit_unknown_registry_entry("system", system_name)
    try:
        task_class = get_task_class(task_name)
    except KeyError:
        _exit_unknown_registry_entry("task", task_name)

    # Get parameter metadata
    system_params_meta = get_class_params(system_class)
    task_params_meta = get_class_params(task_class)

    # Add dynamic arguments
    create_dynamic_parser(parser, system_params_meta, "system")
    create_dynamic_parser(parser, task_params_meta, "task")

    # Parse all arguments now
    args = parser.parse_args()
    requested_verbosity = max(args.verbosity, args.verbosity_count)
    if requested_verbosity < 0:
        parser.error("--verbosity must be >= 0")

    # Extract parameters from different sources
    config_system_params = config.get("system", {}).get("params", {})
    config_task_params = _normalize_task_fixture_params(
        config.get("task", {}).get("params", {})
    )

    cli_system_params = parse_namespace_args(args, "system")
    cli_task_params = _normalize_task_fixture_params(parse_namespace_args(args, "task"))

    # Handle JSON params
    if args.system_params:
        json_params = _parse_json_mapping_arg(args.system_params, "--system-params")
        cli_system_params.update(json_params)

    if args.task_params:
        json_params = _parse_json_mapping_arg(args.task_params, "--task-params")
        cli_task_params.update(json_params)

    # Resolve parameters
    try:
        system_params = resolve_parameters(
            system_params_meta,
            config_system_params,
            cli_system_params,
            "system",
        )
        task_params = resolve_parameters(
            task_params_meta,
            config_task_params,
            cli_task_params,
            "task",
        )
    except ValueError as e:
        parser.error(str(e))

    selected_schedule_id = task_params.get("schedule")
    if not selected_schedule_id:
        _exit_cli_error(
            "A schedule is required to run this benchmark.",
            details=[f"Task: {task_name}"],
            hint=f"Pass --task.schedule <id> or run 'clbench inspect task {task_name}' to see available schedules.",
        )

    schedule_run_defaults = None
    if isinstance(selected_schedule_id, str) and selected_schedule_id:
        try:
            schedule_run_defaults = get_task_schedule(
                task_name, selected_schedule_id
            ).run_defaults
        except KeyError:
            schedule_run_defaults = None

    config_runs = config.get("runs")
    config_max_workers = config.get("max_workers")
    config_run_mode = config.get("run_mode", config.get("rollout_mode"))
    config_verbose_runs = config.get("verbose_runs")

    runs = (
        args.runs
        if args.runs is not None
        else config_runs
        if config_runs is not None
        else 1
        if schedule_run_defaults is None
        else schedule_run_defaults.runs
    )
    max_workers = (
        args.max_workers
        if args.max_workers is not None
        else config_max_workers
        if config_max_workers is not None
        else 12
        if schedule_run_defaults is None
        else schedule_run_defaults.max_workers
    )
    run_mode_value = (
        args.run_mode
        if args.run_mode is not None
        else config_run_mode
        if config_run_mode is not None
        else RunMode.PERMUTE.value
        if schedule_run_defaults is None
        else schedule_run_defaults.mode
    )
    run_mode = RunMode(run_mode_value)
    verbose_runs = (
        args.verbose_runs
        if args.verbose_runs is not None
        else bool(config_verbose_runs)
        if config_verbose_runs is not None
        else False
    )
    logger.debug(
        "Resolved run configuration: system=%s task=%s schedule=%s runs=%s max_workers=%s run_mode=%s verbose_runs=%s",
        system_name,
        task_name,
        selected_schedule_id,
        runs,
        max_workers,
        run_mode.value,
        verbose_runs,
    )
    task_brief = _get_task_brief(task_class, task_params)
    display_task_params = _build_display_task_params(task_class, task_params)

    if args.dry_run:
        print("Dry run successful.")
        print(
            f"System: {system_name} ({system_class.__module__}.{system_class.__name__})"
        )
        print(f"Task: {task_name} ({task_class.__module__}.{task_class.__name__})")
        print(f"Resolved system params: {system_params}")
        print(f"Resolved task params: {display_task_params}")
        print(f"Runs: {runs}, max workers: {max_workers}, mode: {run_mode.value}")
        print(f"Verbose runs: {verbose_runs}")
        print(f"Live dashboard: {args.live_dashboard}")
        if task_brief:
            print(f"Task brief: {task_brief}")
        print("Benchmark execution was skipped.")
        sys.exit(0)

    print(f"Running {system_name} on {task_name}...")
    print(f"System params: {system_params}")
    print(f"Task params: {display_task_params}")
    print()

    run_group_id = args.run_group_id or new_timestamp_run_id()
    run_log_path = build_run_log_path(task_name, run_group_id)
    _configure_cli_logging(
        args.debug,
        verbosity=requested_verbosity,
        log_file_path=run_log_path,
    )
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path: Path | None = None
    baseline_live_path: Path | None = None
    run_live_paths: dict[int, Path] = {}
    baseline_trace_data: dict[str, Any] | None = None
    live_viewer_url: str | None = None
    skip_baseline = not getattr(system_class, "supports_baseline", True)

    if skip_baseline:
        print(
            f"Note: {system_name} does not support baseline mode. "
            "Skipping baseline run; gain metrics will not be available."
        )
        print()

    if args.live_dashboard:
        build_live_output_dir(task_name, run_group_id).mkdir(
            parents=True, exist_ok=True
        )
        manifest_path = build_live_manifest_path(task_name, run_group_id)
        if not skip_baseline:
            baseline_live_path = build_live_trace_path(
                task_name, run_group_id, "baseline"
            )
        run_live_paths = {
            run_index: build_live_trace_path(
                task_name, run_group_id, f"run_{run_index + 1}"
            )
            for run_index in range(runs)
        }
        write_live_manifest(
            manifest_path,
            _build_live_manifest_payload(
                run_group_id=run_group_id,
                status="running",
                phase="baseline" if not skip_baseline else "runs",
                task_name=task_name,
                task_params=display_task_params,
                system_name=system_name,
                system_params=system_params,
                root_dir=repo_root,
                manifest_path=manifest_path,
                task_brief=task_brief,
                baseline_trace_path=baseline_live_path,
                run_trace_paths=run_live_paths,
            ),
        )
        if args.live_server:
            try:
                _server, _thread, live_viewer_url = start_live_dashboard_server(
                    root_dir=repo_root,
                    manifest_path=manifest_path,
                )
            except OSError as exc:
                print(f"Live dashboard server could not start: {exc}")
            else:
                print(f"Live dashboard: {live_viewer_url}")
                print()

    try:
        with bind_logging_context(
            run_group_id=run_group_id,
            task=task_name,
            system=system_name,
            phase="benchmark",
        ):
            logger.log(
                RUN_LOG_LEVEL,
                "started",
                extra={
                    "runs": runs,
                    "max_workers": max_workers,
                    "run_mode": run_mode.value,
                    "verbose_runs": verbose_runs,
                    "live_dashboard": args.live_dashboard,
                    "log_path": str(run_log_path),
                },
            )
            baseline_info, task_results, run_trace_data_list = run_benchmark(
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
                include_baseline=not skip_baseline,
                baseline_task_params=task_class.baseline_task_params or None,
                baseline_live_path=baseline_live_path,
                live_trace_paths=run_live_paths or None,
                output_path=Path(args.output) if args.output else None,
            )
            logger.log(RUN_LOG_LEVEL, "finished")

        if baseline_info is not None:
            _, _, baseline_trace_data, baseline_execution = baseline_info
        else:
            baseline_trace_data = None
            baseline_execution: dict[str, Any] = {}
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _exit_cli_error(
            "Benchmark execution failed.",
            details=[
                f"Task: {task_name}",
                f"System: {system_name}",
                f"Run group: {run_group_id}",
                f"Config: {args.config}" if args.config else "Config: (none)",
                (
                    f"Live dashboard: {live_viewer_url}"
                    if live_viewer_url
                    else "Live dashboard: disabled or unavailable"
                ),
                f"{type(e).__name__}: {e}",
            ],
            hint=(
                "Add --debug for extra CLI setup logs."
                if not args.debug
                else "Review the traceback above for the failure source."
            ),
            exc=e,
            show_traceback=True,
        )

    baseline_trace_path, persisted_run_trace_paths = _persist_full_trace_outputs(
        task_name=task_name,
        run_group_id=run_group_id,
        baseline_trace_data=baseline_trace_data,
        run_trace_data_list=run_trace_data_list,
        root_dir=repo_root,
    )

    baseline_outcomes = baseline_execution.get("instance_outcomes", [])
    baseline_summary = summarize_instance_outcomes(baseline_outcomes)
    allow_partial_baseline = bool(baseline_execution.get("instances_blocked", 0))

    run_entries: list[RunSummaryRunEntry] = []
    run_outcomes_for_aggregate: list[list[dict[str, Any]]] = []
    blocked_runs_count = 0
    for i in range(runs):
        execution_summary = (
            {}
            if not task_results.execution_summaries
            else task_results.execution_summaries[i]
        )
        raw_outcomes = execution_summary.get("instance_outcomes", [])
        run_status = execution_summary.get("status", "completed")
        if run_status == "blocked":
            blocked_runs_count += 1
        # Only attach baseline deltas when a baseline was actually run.
        # Tolerate partial rollouts (blocked mid-trajectory) by passing the
        # ``allow_partial_rollout`` flag whenever any rollout was blocked.
        enriched_outcomes = (
            attach_baseline_to_outcomes(
                baseline_outcomes,
                raw_outcomes,
                allow_partial_baseline=allow_partial_baseline,
                allow_partial_rollout=True,
            )
            if not skip_baseline
            else raw_outcomes
        )
        run_outcomes_for_aggregate.append(raw_outcomes)
        run_entries.append(
            RunSummaryRunEntry(
                run_index=i,
                score=task_results.results[i].score,
                trace_path=(
                    persisted_run_trace_paths[i]
                    if i < len(persisted_run_trace_paths)
                    else ""
                ),
                execution=execution_summary,
                status=run_status,
                instance_outcomes=enriched_outcomes,
                aggregate=summarize_instance_outcomes(enriched_outcomes),
            )
        )

    completed_runs_count = runs - blocked_runs_count
    # Aggregate cross-run statistics over completed rollouts only — partial
    # rollouts in PERMUTE mode see different instance subsets at different
    # trajectory positions, so including them would bias the mean curves.
    completed_run_outcomes = [
        run_outcomes_for_aggregate[i]
        for i in range(runs)
        if run_entries[i].status != "blocked"
    ]
    aggregate = {
        **task_results.aggregate(),
        **build_benchmark_aggregate(
            baseline_outcomes,
            completed_run_outcomes,
            allow_partial_baseline=allow_partial_baseline,
            allow_partial_rollout=True,
        ),
    }
    aggregate["runs_total"] = runs
    aggregate["runs_completed"] = completed_runs_count
    aggregate["runs_blocked"] = blocked_runs_count
    if blocked_runs_count:
        blocked_reasons = [
            entry.execution.get("blocked_reason")
            for entry in run_entries
            if entry.status == "blocked" and isinstance(entry.execution, dict)
        ]
        aggregate["blocked_runs"] = [r for r in blocked_reasons if r]

    _print_reward_first_results(
        aggregate=aggregate,
        baseline_summary=baseline_summary,
        baseline_execution=baseline_execution,
        run_entries=run_entries,
        runs=runs,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    if runs == 1:
        result = task_results.results[0]
        execution_summary = (
            {}
            if not task_results.execution_summaries
            else task_results.execution_summaries[0]
        )
        usage_summary = execution_summary.get("usage", {}).get("interaction", {})
        print(f"\nScore: {result.score:.4f}")
        if usage_summary:
            print(f"System compute: {format_cost_summary(usage_summary)}")
    else:
        score_agg = aggregate.get("score", {})
        if score_agg:
            print(
                f"\nLegacy score aggregate: mean={score_agg['mean']:.4f} "
                f"std={score_agg['std']:.4f}"
            )

    baseline_entry = None
    effective_task_brief = task_brief
    if not skip_baseline:
        baseline_entry = BaselineSummaryEntry(
            trace_path=baseline_trace_path or "",
            execution=baseline_execution,
            status=baseline_execution.get("status", "completed"),
            instance_outcomes=baseline_outcomes,
        )
        effective_task_brief = baseline_execution.get("task_brief") or task_brief

    # When baseline was skipped or completed with gaps, derive the sequence
    # signature from the first rollout so the viewer still knows instance order.
    if skip_baseline or allow_partial_baseline:
        first_run_outcomes = (
            run_outcomes_for_aggregate[0] if run_outcomes_for_aggregate else []
        )
        sequence_signature = build_sequence_signature(first_run_outcomes)
    else:
        sequence_signature = build_sequence_signature(baseline_outcomes)

    if completed_runs_count == 0 and blocked_runs_count:
        summary_status = "all_blocked"
    elif (
        blocked_runs_count or baseline_execution.get("status") == "completed_with_gaps"
    ):
        summary_status = "completed_with_gaps"
    else:
        summary_status = "completed"
    summary = RunSummary(
        run_group_id=task_results.run_group_id,
        system=build_system_manifest_entry(system_name, system_params),
        task={"name": task_name, "params": display_task_params},
        variant=task_params.get("variant"),
        schedule=task_params.get("schedule"),
        run_count=runs,
        max_workers=max_workers,
        run_mode=run_mode.value if runs > 1 else None,
        aggregate=aggregate,
        runs=run_entries,
        baseline=baseline_entry,
        status=summary_status,
        task_brief=effective_task_brief,
        sequence_signature=sequence_signature,
    )
    viewer_artifact_path = save_viewer_artifact(
        summary, task_name, baseline_trace_data, run_trace_data_list
    )
    print(f"\nLoad this in the viewer: {viewer_artifact_path}")
    print(
        "Full trace JSON files: "
        f"{Path('results') / task_name / 'traces' / safe_run_id_component(run_group_id)}"
    )
    if manifest_path is not None:
        write_live_manifest(
            manifest_path,
            _build_live_manifest_payload(
                run_group_id=run_group_id,
                status=summary.status,
                phase="completed",
                task_name=task_name,
                task_params=display_task_params,
                system_name=system_name,
                system_params=system_params,
                root_dir=repo_root,
                manifest_path=manifest_path,
                task_brief=summary.task_brief,
                baseline_trace_path=baseline_live_path,
                run_trace_paths=run_live_paths,
                summary_path=viewer_artifact_path,
            ),
        )
        if live_viewer_url is not None:
            print(f"Live viewer URL: {live_viewer_url}")
        # Give the browser time to fetch the completed manifest before the
        # daemon server thread is killed when this process exits.
        time.sleep(5)


if __name__ == "__main__":
    main()
