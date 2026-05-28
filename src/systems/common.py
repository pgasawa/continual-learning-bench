"""Shared helpers for CLI systems."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from html import escape as escape_xml_text
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTAINER_WORKSPACE = "/workspace"
DEFAULT_DOCKER_IMAGE = "node:22-slim"

_SKILL_STANDARD_FILE = "SKILL.md"
_SKILL_INVOCATION_TYPES = frozenset({"prepend", "skill"})
_SKILL_COMMAND_PREFIX_BY_AGENT = {"claude": "/", "codex": "$"}
_SKILL_HOME_DIR_BY_AGENT = {"claude": ".claude", "codex": ".codex"}


def create_run_workspace(cache_dir_name: str) -> str:
    """Create a per-run host workspace under the user's cache directory."""
    base_dir = Path.home() / ".cache" / cache_dir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="run_", dir=str(base_dir))


def resolve_seed_memory_dir(seed_memory_dir: str | None) -> Path | None:
    """Resolve and validate an optional seed memory directory."""
    if seed_memory_dir is None:
        return None
    seed_path = Path(seed_memory_dir).expanduser().resolve()
    if not seed_path.is_dir():
        raise FileNotFoundError(f"seed_memory_dir does not exist: {seed_path}")
    return seed_path


def copy_memory_seed(seed_memory_dir: Path | None, target_dir: Path) -> None:
    """Copy seed memory directory contents into a target memory directory."""
    if seed_memory_dir is None:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in seed_memory_dir.iterdir():
        target = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _extract_skill_frontmatter_name(contents: str) -> str | None:
    """Return a simple `name:` value from Agent Skills frontmatter, if present."""
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return None
        key, sep, value = line.partition(":")
        if sep and key.strip() == "name":
            name = value.strip().strip("\"'")
            return name or None
    return None


def _strip_skill_frontmatter(contents: str) -> str:
    """Strip Agent Skills frontmatter before inlining skill text into a prompt."""
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return contents

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).lstrip("\n")
    return contents


def _read_skill_location(location: str | Path) -> tuple[str, str, Path | None]:
    """Read a single-file or standard-directory skill location."""
    path = Path(location).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"skill does not exist: {path}")

    source_dir: Path | None = None
    if path.is_dir():
        skill_file = path / _SKILL_STANDARD_FILE
        if not skill_file.is_file():
            raise FileNotFoundError(
                f"skill directory does not contain {_SKILL_STANDARD_FILE}: {path}"
            )
        default_name = path.name
        source_dir = path
        contents = skill_file.read_text()
    elif path.is_file():
        contents = path.read_text()
        if path.name == _SKILL_STANDARD_FILE:
            default_name = path.parent.name
            source_dir = path.parent
        else:
            default_name = path.stem
    else:
        raise FileNotFoundError(f"skill does not exist: {path}")

    skill_name = _extract_skill_frontmatter_name(contents) or default_name
    if not skill_name:
        raise ValueError(f"skill has no usable name: {path}")
    return skill_name, contents, source_dir


def _format_skill_prompt_prefix(
    *,
    agent_name: str,
    command_prefix: str,
    skill_name: str,
) -> str:
    """Return the prompt prefix that invokes an installed native skill."""
    if agent_name == "claude":
        escaped_skill_name = escape_xml_text(skill_name, quote=False)
        return (
            f"<command-message>{escaped_skill_name}</command-message>\n"
            f"<command-name>{command_prefix}{escaped_skill_name}</command-name>\n"
        )
    return f"{command_prefix}{skill_name}\n"


def load_skill_prompt_prefix(
    location: str | Path | None,
    *,
    agent_name: str,
    workspace_dir: str | Path,
    invocation_type: str = "skill",
) -> str:
    """Load an optional skill and return the prompt prefix for using it.

    ``location`` may point to a markdown file, a skill directory containing
    ``SKILL.md``, or that ``SKILL.md`` file directly. ``invocation_type`` is
    ``"skill"`` to install the skill into the agent's native skill directory and
    return its command prefix, or ``"prepend"`` to inline the skill instructions
    directly into the prompt prefix.
    """
    if not isinstance(invocation_type, str):
        raise ValueError(
            "invocation_type must be one of "
            f"{sorted(_SKILL_INVOCATION_TYPES)}, got {invocation_type!r}"
        )
    normalized_invocation_type = invocation_type.strip().lower()
    if normalized_invocation_type not in _SKILL_INVOCATION_TYPES:
        raise ValueError(
            "invocation_type must be one of "
            f"{sorted(_SKILL_INVOCATION_TYPES)}, got {invocation_type!r}"
        )

    if not isinstance(agent_name, str):
        raise ValueError(
            "agent_name must be one of "
            f"{sorted(_SKILL_COMMAND_PREFIX_BY_AGENT)}, got {agent_name!r}"
        )
    normalized_agent_name = agent_name.strip().lower()
    try:
        command_prefix = _SKILL_COMMAND_PREFIX_BY_AGENT[normalized_agent_name]
        skill_home_dir = _SKILL_HOME_DIR_BY_AGENT[normalized_agent_name]
    except KeyError as exc:
        raise ValueError(
            "agent_name must be one of "
            f"{sorted(_SKILL_COMMAND_PREFIX_BY_AGENT)}, got {agent_name!r}"
        ) from exc

    if location is None:
        return ""

    skill_name, contents, source_dir = _read_skill_location(location)
    if normalized_invocation_type == "prepend":
        skill_text = _strip_skill_frontmatter(contents).strip()
        return f"{skill_text}\n\n" if skill_text else ""

    target_dir = (
        Path(workspace_dir).expanduser().resolve()
        / skill_home_dir
        / "skills"
        / skill_name
    )
    prompt_prefix = _format_skill_prompt_prefix(
        agent_name=normalized_agent_name,
        command_prefix=command_prefix,
        skill_name=skill_name,
    )
    if source_dir is not None and source_dir.resolve() == target_dir:
        return prompt_prefix

    if target_dir.exists():
        shutil.rmtree(target_dir)
    if source_dir is not None:
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / _SKILL_STANDARD_FILE).write_text(contents)
    return prompt_prefix


def read_memory_snapshot(
    memory_dirs: Sequence[Path],
    *,
    logger: logging.Logger,
) -> dict[str, str]:
    """Read memory files from directories as relpath -> contents."""
    snapshot: dict[str, str] = {}
    for host_dir in memory_dirs:
        if not host_dir.exists():
            continue
        for path in sorted(host_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(host_dir).as_posix()
            try:
                snapshot[rel] = path.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Could not read memory file %s: %s", path, exc)
    return snapshot


def _validate_other_memory_file_pattern(pattern: str) -> str:
    """Validate an opt-in workspace memory glob and return its stripped form."""
    if not isinstance(pattern, str):
        raise ValueError(
            "other_memory_files entries must be relative workspace glob strings, "
            f"got {pattern!r}"
        )
    normalized = pattern.strip()
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "other_memory_files entries must be relative paths/globs inside the "
            f"workspace, got {pattern!r}"
        )
    return normalized


def normalize_other_memory_file_patterns(
    patterns: Sequence[str] | None,
) -> list[str]:
    """Validate and normalize opt-in workspace memory globs."""
    if patterns is None:
        return []
    if isinstance(patterns, str):
        raise ValueError("other_memory_files must be a sequence of strings")
    return [_validate_other_memory_file_pattern(pattern) for pattern in patterns]


def _is_glob_pattern(pattern: str) -> bool:
    """Return whether a workspace memory pattern contains glob metacharacters."""
    return any(char in pattern for char in "*?[")


def _iter_workspace_memory_files(workspace_dir: Path, pattern: str) -> list[Path]:
    """Expand a relative file/dir/glob pattern to files under the workspace."""
    if _is_glob_pattern(pattern):
        candidates = workspace_dir.glob(pattern)
    else:
        candidates = [workspace_dir / pattern]

    files: dict[str, Path] = {}
    for candidate in candidates:
        if candidate.is_dir():
            nested = candidate.rglob("*")
        else:
            nested = [candidate]
        for path in nested:
            if path.is_file():
                files[path.as_posix()] = path
    return [files[key] for key in sorted(files)]


def read_other_memory_files(
    workspace_dir: str | Path,
    patterns: Sequence[str] | None,
    *,
    logger: logging.Logger,
) -> dict[str, str]:
    """Read opt-in workspace files as additional file-memory entries.

    ``patterns`` are relative workspace file paths, directory paths, or glob
    patterns. Matched file keys are always relative to ``workspace_dir`` so they
    can be merged directly into ``memory_files``.
    """
    normalized_patterns = normalize_other_memory_file_patterns(patterns)
    if not normalized_patterns:
        return {}

    workspace_path = Path(workspace_dir).expanduser().resolve()
    snapshot: dict[str, str] = {}
    for pattern in normalized_patterns:
        for path in _iter_workspace_memory_files(workspace_path, pattern):
            try:
                resolved = path.resolve()
                rel = resolved.relative_to(workspace_path).as_posix()
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Skipping other memory file outside workspace %s: %s", path, exc
                )
                continue
            try:
                snapshot[rel] = resolved.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Could not read other memory file %s: %s", resolved, exc)
    return snapshot


def _collision_memory_key(prefix: str, rel: str, existing: Mapping[str, str]) -> str:
    """Return a deterministic namespaced key that is absent from ``existing``."""
    candidate = f"{prefix}/{rel}"
    counter = 2
    while candidate in existing:
        candidate = f"{prefix}/{counter}/{rel}"
        counter += 1
    return candidate


def merge_memory_snapshots(
    native_files: Mapping[str, str],
    other_files: Mapping[str, str],
    *,
    native_source: str,
    other_source: str = "other_memory_file",
    collision_prefix: str = "other_memory",
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Merge native and opt-in memory snapshots without hiding native files.

    ``memory_files`` remains a single merged relpath -> contents view for traces
    and artifacts. If an opt-in workspace file has the same relpath as a native
    memory file, the native key is preserved and the opt-in file is emitted under
    ``<collision_prefix>/<relpath>``. The returned source map labels each merged
    key, and the returned other-key map links original opt-in paths to their
    merged keys.
    """
    merged = {str(key): value for key, value in native_files.items()}
    sources = {key: native_source for key in merged}
    other_key_map: dict[str, str] = {}

    for raw_key, contents in other_files.items():
        key = str(raw_key)
        merged_key = key
        if merged_key in merged:
            merged_key = _collision_memory_key(collision_prefix, key, merged)
        merged[merged_key] = contents
        sources[merged_key] = other_source
        other_key_map[key] = merged_key

    return merged, sources, other_key_map


def start_docker_container(
    *,
    name_prefix: str,
    host_workspace: str,
    docker_image: str,
    env: Mapping[str, str],
    logger: logging.Logger,
    container_workspace: str = CONTAINER_WORKSPACE,
) -> str:
    """Start a persistent Docker container for a per-run workspace."""
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        f"{name_prefix}_{Path(host_workspace).name}",
        "-v",
        f"{host_workspace}:{container_workspace}",
    ]
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(
        [
            "-w",
            container_workspace,
            docker_image,
            "sleep",
            "infinity",
        ]
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Docker container: {result.stderr}")
    container_id = result.stdout.strip()
    logger.info("Container started: %s", container_id[:12])
    return container_id


def docker_exec(
    container_id: str | None,
    command: str,
    *,
    timeout: int | None,
    default_timeout: int,
    stdout_idle_timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command inside a Docker container.

    Two timeouts run together: the wall-clock `timeout` raises TimeoutExpired,
    while `stdout_idle_timeout` kills the subprocess after that many seconds
    of silence and returns rc=124. The idle path works around claude code
    sometimes sitting in epoll_pwait2 on a dead connection — callers retry on
    non-zero exit, which gives a fresh `docker exec` and TCP connection.
    """
    if container_id is None:
        raise RuntimeError("Container not running")

    cmd = [
        "docker",
        "exec",
        container_id,
        "sh",
        "-c",
        command,
    ]
    overall_timeout = float(timeout or default_timeout)
    idle_timeout = float(stdout_idle_timeout)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    last_activity = [time.monotonic()]
    activity_lock = threading.Lock()

    def _drain(pipe: Any, sink: list[str]) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                sink.append(line)
                with activity_lock:
                    last_activity[0] = time.monotonic()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=_drain, args=(proc.stdout, stdout_chunks), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_drain, args=(proc.stderr, stderr_chunks), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    start = time.monotonic()
    poll_interval = 5.0
    while proc.poll() is None:
        time.sleep(poll_interval)
        now = time.monotonic()
        elapsed = now - start
        with activity_lock:
            idle_for = now - last_activity[0]
        if elapsed > overall_timeout:
            proc.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            raise subprocess.TimeoutExpired(
                cmd,
                overall_timeout,
                output="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
            )
        if idle_for > idle_timeout:
            logger.warning(
                "docker_exec: stdout idle for %.0fs (limit %.0fs); killing subprocess "
                "to trigger retry on fresh connection",
                idle_for,
                idle_timeout,
            )
            proc.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            return subprocess.CompletedProcess(
                cmd,
                returncode=124,
                stdout="".join(stdout_chunks),
                stderr=(
                    "".join(stderr_chunks)
                    + "\n[docker_exec] killed: stdout idle "
                    + f"{idle_for:.0f}s > {idle_timeout:.0f}s"
                ),
            )

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return subprocess.CompletedProcess(
        cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def stop_docker_container(container_id: str | None) -> None:
    """Stop and remove a Docker container if it is running."""
    if container_id is None:
        return
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        capture_output=True,
        check=False,
    )


def cleanup_run_workspace(tmp_dir: str) -> None:
    """Delete a per-run host workspace."""
    path = Path(tmp_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
