"""Phantom CLI system: Claude Code running inside a Phantom Docker container."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import build_usage_event
from ..common import (
    CONTAINER_WORKSPACE,
    cleanup_run_workspace,
    copy_memory_seed,
    create_run_workspace,
    docker_exec,
    load_skill_prompt_prefix,
    merge_memory_snapshots,
    normalize_other_memory_file_patterns,
    read_memory_snapshot,
    read_other_memory_files,
    resolve_seed_memory_dir,
    start_docker_container,
    stop_docker_container,
)
from ..utils.structured_output import (
    extract_json,
    schema_to_prompt_instruction,
    validate_with_coercion,
)
from .artifacts import ensure_registered as ensure_phantom_artifact_exporter_registered

ensure_phantom_artifact_exporter_registered()

logger = logging.getLogger(__name__)

_CONTAINER_CLAUDE_CONFIG = f"{CONTAINER_WORKSPACE}/.claude"
_CLAUDE_PROJECT_SLUG = "-workspace"
_CONTAINER_MEMORY_DIR = (
    f"{_CONTAINER_CLAUDE_CONFIG}/projects/{_CLAUDE_PROJECT_SLUG}/memory"
)
_DOCKER_IMAGE = "cl-bench/phantom"
_PHANTOM_SETTINGS_PATH = "/opt/phantom/settings.json"
_DEFAULT_PHANTOM_SPEC = (
    "/Users/yu.yan/code/phantom/src/phantom/agent-docs/PHANTOM_SPEC.md"
)

# Production Phantom's identity intro. Sent as the first user message of a session,
# then carried forward by claude's --resume conversation continuity.
_PHANTOM_INTRO_TEMPLATE = """# You are Phantom 👻

## Your Identity
- **Name:** Phantom
- **Role:** Browser Automation Agent
- **Emoji:** 👻

---

## Documentation Files (READ THESE FIRST)

You are currently running as the orchestrator agent. Before starting work, read these files for full context:

1. **Your Specification:** `cat /opt/phantom/agent-docs/PHANTOM_SPEC.md`
2. **Agent Protocol:** `cat /opt/phantom/agent-docs/AGENT_PROTOCOL.md`
3. **Slack Interface Docs:** `cat /opt/phantom/agent-docs/SLACK_INTERFACE.md`
4. **Workflow Docs:** `cat /opt/phantom/agent-docs/ORCHESTRATOR.md`
5. **Pipedream Integrations:** `cat /opt/phantom/agent-docs/PIPEDREAM_CONNECT.md` — connected app discovery, OAuth dashboard, and `tools/pdx.py` (`pdx`) CLI tools

---

## Your Memory

No previous memory. This is your first session.

---

## Current Task

"""

# Slack-bridge variant: same as the production intro but only references
# SLACK_INTERFACE.md (the other four production docs aren't relevant for CL-Bench).
# Memory and the task body are appended by _build_prompt_slack_bridge.
_PHANTOM_INTRO_TEMPLATE_SLACK = """# You are Phantom 👻

## Your Identity
- **Name:** Phantom
- **Role:** Browser Automation Agent
- **Emoji:** 👻

---

## Documentation Files (READ THESE FIRST)

You are currently running as the orchestrator agent. Before starting work, read these files for full context:

1. **Slack Interface Docs:** `cat /opt/phantom/agent-docs/SLACK_INTERFACE.md`

---
"""

# Mock-slack workspace plumbing. clbench_mock.py is staged into the workspace and
# autoloaded by the container's site-packages .pth file.
_CONTAINER_MOCK_DIR = f"{CONTAINER_WORKSPACE}/_clbench_mock"
_CONTAINER_PHANTOM_MOCK_STATE = f"{CONTAINER_WORKSPACE}/.phantom_mock"
_PHANTOM_SLACK_CHANNEL = "#benchmark"
_PHANTOM_SLACK_CHANNEL_ID = "C_BENCH"
_HOST_MOCK_MODULE = Path(__file__).parent / "clbench_mock.py"

# Production reads memory/<agent>_memory.md from its repo root; we expose the same
# filename at /workspace, which is bind-mounted so host reads see what claude wrote.
_CONTAINER_PHANTOM_MEMORY_FILE = f"{CONTAINER_WORKSPACE}/phantom_memory.md"
_HOST_PHANTOM_MEMORY_NAME = "phantom_memory.md"

# Matches production's empty-memory fallback wording.
_MEMORY_EMPTY_FALLBACK = "No previous memory. This is your first session."

# Production claude learns to update memory from a doc our intro strips, so we
# repeat the hint inline every turn.
_MEMORY_WRITE_HINT = (
    "\n\n## Memory Updates\n"
    f"Update your memory file (`{_CONTAINER_PHANTOM_MEMORY_FILE}`) when "
    "you learn something useful for future task instances — patterns "
    "in the data, useful tool invocations, traps to avoid. Use the "
    "Write or Edit tool. Future turns will read your updated memory."
)

# Routes the JSON answer through production phantom's slack_cli instead of stdout.
# slack_cli say takes only a positional message — no -c/--channel flag.
_SLACK_REPLY_INSTRUCTION = (
    "\n\n## Reply via Slack\n"
    "Instead of returning your answer on stdout, post your final JSON "
    "response to the Slack channel by running this exact Bash command:\n"
    "    python /opt/phantom/slack_cli say '<your-json-here>'\n"
    "Wrap the JSON in single quotes. Post exactly one message via Bash; "
    "the harness reads the last message in the channel as your answer."
)

_REASONING_EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}

_EMPTY_TOKENS: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cached_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}

_DEFAULT_MEMORY_INSTRUCTION = """You should learn and store information from each interaction, so use your memory. You have a persistent, file-based memory system at {memory_dir}. The MEMORY.md file at {memory_dir}/MEMORY.md is an index of individual memory files in that directory. Read it at the start of each turn and update both the index and individual memory files as you learn new information that will be useful in future turns."""


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line: %.120s", line)
    return events


def _extract_session_ids(events: list[dict[str, Any]]) -> list[str]:
    session_ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "system" or event.get("subtype") != "init":
            continue
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or session_id in seen:
            continue
        seen.add(session_id)
        session_ids.append(session_id)
    return session_ids


def _extract_session_id(events: list[dict[str, Any]]) -> str | None:
    session_ids = _extract_session_ids(events)
    return session_ids[0] if session_ids else None


def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
    if not text_parts:
        raise ValueError("No assistant message found in claude output")
    return "\n".join(text_parts)


def _extract_token_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    # input_tokens from claude excludes cache reads/writes; we fold them in here so
    # the bench sees a true input total but keep the cache buckets for pricing.
    totals = dict(_EMPTY_TOKENS)
    for event in events:
        if event.get("type") != "result":
            continue
        usage = event.get("usage", {})
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_writes = int(usage.get("cache_creation_input_tokens", 0) or 0)
        totals["input_tokens"] += (
            int(usage.get("input_tokens", 0) or 0) + cache_read + cache_writes
        )
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        totals["cached_input_tokens"] += cache_read
        totals["cache_creation_input_tokens"] += cache_writes
    return totals


def _extract_cli_cost_usd(events: list[dict[str, Any]]) -> float | None:
    costs: list[float] = []
    saw_result = False
    for event in events:
        if event.get("type") != "result":
            continue
        saw_result = True
        cost = event.get("total_cost_usd")
        if cost is None:
            return None
        try:
            costs.append(float(cost))
        except (TypeError, ValueError):
            return None
    if not saw_result:
        return None
    return sum(costs)


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                candidates.append(part)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(stripped[idx : idx + end])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_action_text(text: str, schema: type[Any]) -> Any:
    last_error: Exception | None = None
    for candidate in reversed(_json_candidates(text)):
        try:
            return validate_with_coercion(candidate, schema)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return validate_with_coercion(extract_json(text), schema)


@register_system("phantom")
class PhantomSystem(ContinualLearningSystem):
    """Claude Code running inside the Phantom container, talking via stdout."""

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        name: str = "phantom",
        seed_memory_dir: Optional[str] = None,
        timeout: int = 300,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
        docker_image: str = _DOCKER_IMAGE,
        version: str = "2.1.119",
        memory_instruction: str = _DEFAULT_MEMORY_INSTRUCTION,
        single_conversation: bool = True,
        preamble_file: Optional[str] = None,
        skill: Optional[str] = None,
        invocation_type: str = "skill",
        other_memory_files: Optional[list[str]] = None,
        auth_token: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = "low",
        settings_path: Optional[str] = _PHANTOM_SETTINGS_PATH,
        phantom_intro: bool = True,
        slack_bridge: bool = False,
    ):
        resolved_auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        if not resolved_auth_token:
            raise ValueError("ANTHROPIC_AUTH_TOKEN must be set")
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        if not resolved_base_url:
            raise ValueError("ANTHROPIC_BASE_URL must be set")
        if (
            reasoning_effort is not None
            and reasoning_effort not in _REASONING_EFFORT_VALUES
        ):
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_REASONING_EFFORT_VALUES)} "
                f"or None, got {reasoning_effort!r}"
            )

        if not isinstance(single_conversation, bool):
            raise ValueError(
                f"single_conversation must be a bool, got {single_conversation!r}"
            )

        seed_path = resolve_seed_memory_dir(seed_memory_dir)

        self._name = name
        self._model = model
        self._timeout = timeout
        self._max_turns = max_turns
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._auth_token = resolved_auth_token
        self._base_url = resolved_base_url
        self._settings_path = settings_path
        self._phantom_intro = phantom_intro
        self._slack_bridge = slack_bridge
        self._reasoning_effort = reasoning_effort
        self._memory_instruction = memory_instruction
        self._single_conversation = single_conversation
        self._preamble = self._read_preamble_file(preamble_file)
        self._skill = skill
        self._invocation_type = invocation_type
        self._skill_prompt_prefix = ""
        self._other_memory_files = normalize_other_memory_file_patterns(
            other_memory_files
        )
        self._seed_memory_dir = seed_path

        self._tmp_dir = create_run_workspace("phantom_bench")

        self._container_id: str | None = None
        self._docker_image = docker_image
        self._version = version
        self._conversation_id: str | None = None
        self._interaction_count: int = 0
        self._jsonl_log: list[dict[str, Any]] = []
        self._cumulative_tokens: dict[str, int] = dict(_EMPTY_TOKENS)
        self._memory_history: list[dict[str, Any]] = []
        self._claude_conversation_session_ids: set[str] = set()
        self._claude_conversation_history: list[dict[str, Any]] = []
        self._claude_instance_conversations: dict[
            tuple[str | None, int | None], dict[str, Any]
        ] = {}
        self._pending_feedback: str | None = None

        self._initialize_workspace()
        self._start_container()

    def _read_preamble_file(self, preamble_file: Optional[str]) -> str | None:
        if not preamble_file:
            return None
        path = Path(preamble_file).expanduser().resolve()
        if not path.is_file():
            logger.warning(
                "preamble_file %s does not exist; proceeding without preamble", path
            )
            return None
        content = path.read_text()
        logger.info("Loaded preamble from %s (%d chars)", path, len(content))
        return content

    def _start_container(self) -> None:
        self._container_id = start_docker_container(
            name_prefix="phantom_bench",
            host_workspace=self._tmp_dir,
            docker_image=self._docker_image,
            env={
                "ANTHROPIC_AUTH_TOKEN": self._auth_token,
                "ANTHROPIC_BASE_URL": self._base_url,
                "CLAUDE_CONFIG_DIR": _CONTAINER_CLAUDE_CONFIG,
                "IS_SANDBOX": "1",
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            logger=logger,
        )

        # `claude --version` round-trips to the API to validate the auth token,
        # so it can take a while under load. Match production's timeout.
        verify = self._docker_exec("claude --version", timeout=900)
        if verify.returncode != 0:
            raise RuntimeError(
                f"claude binary not available in phantom image: "
                f"{verify.stderr.strip() or verify.stdout.strip()}"
            )
        logger.info(
            "Phantom container ready: %s",
            verify.stdout.strip() or verify.stderr.strip(),
        )

    def _initialize_workspace(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        copy_memory_seed(self._seed_memory_dir, self.memory_dir)
        self._skill_prompt_prefix = load_skill_prompt_prefix(
            self._skill,
            agent_name="claude",
            workspace_dir=self._tmp_dir,
            invocation_type=self._invocation_type,
        )
        if self._slack_bridge:
            self._install_slack_bridge_workspace()

    def _install_slack_bridge_workspace(self) -> None:
        # Copy clbench_mock.py into the workspace; the container's .pth file picks
        # it up at every Python startup. .phantom_mock/ holds the inbox/outbox JSONL.
        workspace = Path(self._tmp_dir)
        mock_pkg_dir = workspace / "_clbench_mock"
        mock_pkg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_HOST_MOCK_MODULE, mock_pkg_dir / "clbench_mock.py")
        (workspace / ".phantom_mock").mkdir(parents=True, exist_ok=True)

    @property
    def _host_mock_state_dir(self) -> Path:
        return Path(self._tmp_dir) / ".phantom_mock"

    def _slack_bridge_reset_state(self) -> None:
        # Slack history doesn't accumulate across turns; each turn starts fresh.
        for name in ("inbox.jsonl", "outbox.jsonl"):
            path = self._host_mock_state_dir / name
            if path.is_file():
                path.unlink()
        self._host_mock_state_dir.mkdir(parents=True, exist_ok=True)

    def _slack_bridge_append_inbox(
        self,
        text: str,
        *,
        user: str = "U_BENCH",
    ) -> dict[str, Any]:
        self._host_mock_state_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "user": user,
            "text": text,
            "ts": f"{time.time():.6f}",
            "id": uuid.uuid4().hex,
        }
        with (self._host_mock_state_dir / "inbox.jsonl").open(
            "a", encoding="utf-8"
        ) as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    @property
    def _host_phantom_memory_path(self) -> Path:
        # /workspace is bind-mounted, so reading this path == reading the file
        # claude sees at _CONTAINER_PHANTOM_MEMORY_FILE.
        return Path(self._tmp_dir) / _HOST_PHANTOM_MEMORY_NAME

    def _read_phantom_memory(self) -> str:
        path = self._host_phantom_memory_path
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read phantom memory at %s: %s", path, exc)
            return ""

    def _wipe_phantom_memory(self) -> None:
        # Called on reset() only; memory persists across instances within a rollout.
        path = self._host_phantom_memory_path
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to wipe phantom memory at %s: %s", path, exc)

    def _slack_bridge_drain_outbox(self) -> list[dict[str, Any]]:
        path = self._host_mock_state_dir / "outbox.jsonl"
        if not path.is_file():
            return []
        messages: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed outbox line: %.120s", line)
        return messages

    def _docker_exec(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return docker_exec(
            self._container_id,
            command,
            timeout=timeout,
            default_timeout=self._timeout,
        )

    @property
    def memory_dir(self) -> Path:
        return (
            Path(self._tmp_dir)
            / ".claude"
            / "projects"
            / _CLAUDE_PROJECT_SLUG
            / "memory"
        )

    def _read_native_memory_snapshot(self) -> dict[str, str]:
        memory_dirs = [self.memory_dir]
        projects_dir = Path(self._tmp_dir) / ".claude" / "projects"
        if projects_dir.exists():
            for candidate in sorted(projects_dir.glob("*/memory")):
                if candidate not in memory_dirs:
                    memory_dirs.append(candidate)

        return read_memory_snapshot(memory_dirs, logger=logger)

    def _read_other_memory_snapshot(self) -> dict[str, str]:
        return read_other_memory_files(
            self._tmp_dir,
            self._other_memory_files,
            logger=logger,
        )

    def _read_memory_snapshot(self) -> dict[str, str]:
        native_memory_snapshot = self._read_native_memory_snapshot()
        other_memory_snapshot = self._read_other_memory_snapshot()
        memory_snapshot, _, _ = merge_memory_snapshots(
            native_memory_snapshot,
            other_memory_snapshot,
            native_source="claude_memory_file",
        )
        return memory_snapshot

    @property
    def _claude_projects_dir(self) -> Path:
        return Path(self._tmp_dir) / ".claude" / "projects"

    def _read_claude_conversation_jsonl_files(self) -> dict[str, str]:
        # Only export files whose stem matches a session_id we saw in stream-json,
        # to avoid pulling in subagent conversations.
        session_ids = set(self._claude_conversation_session_ids)
        if self._conversation_id is not None:
            session_ids.add(self._conversation_id)
        if not session_ids:
            return {}

        projects_dir = self._claude_projects_dir
        if not projects_dir.exists():
            return {}

        files: dict[str, str] = {}
        workspace_dir = Path(self._tmp_dir)
        for path in sorted(projects_dir.rglob("*.jsonl")):
            if not path.is_file() or path.stem not in session_ids:
                continue
            try:
                rel = path.relative_to(workspace_dir).as_posix()
            except ValueError:
                logger.warning("Skipping Claude JSONL outside workspace: %s", path)
                continue
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Could not read Claude conversation JSONL %s: %s", path, exc
                )
        return files

    @staticmethod
    def _summarize_claude_conversation_files(
        files: dict[str, str],
    ) -> tuple[list[str], dict[str, int]]:
        file_names = sorted(files)
        file_sizes = {name: len(files[name]) for name in file_names}
        return file_names, file_sizes

    def _record_claude_conversation_snapshot(
        self,
        query: Query,
        *,
        step: int,
    ) -> dict[str, Any]:
        conversation_files = self._read_claude_conversation_jsonl_files()
        file_names, file_sizes = self._summarize_claude_conversation_files(
            conversation_files
        )
        session_ids = sorted(self._claude_conversation_session_ids)
        summary = {
            "step": step,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "conversation_id": self._conversation_id,
            "session_ids": session_ids,
            "conversation_jsonl_files": file_names,
            "conversation_jsonl_file_sizes": file_sizes,
        }
        self._claude_conversation_history.append(summary)
        self._claude_instance_conversations[
            (query.instance_id, query.instance_index)
        ] = {
            **summary,
            "conversation_jsonl_files": conversation_files,
        }
        return summary

    def _sorted_claude_instance_conversations(self) -> list[dict[str, Any]]:
        def sort_key(snapshot: dict[str, Any]) -> tuple[int, int | str, str, int]:
            instance_index = snapshot.get("instance_index")
            if isinstance(instance_index, int):
                primary: tuple[int, int | str] = (0, instance_index)
            else:
                primary = (1, str(snapshot.get("instance_id") or ""))
            return (
                *primary,
                str(snapshot.get("instance_id") or ""),
                int(snapshot.get("step") or 0),
            )

        return sorted(self._claude_instance_conversations.values(), key=sort_key)

    def _build_cli_args(self, resume_mode: str) -> list[str]:
        # resume_mode is "fresh" (first call) or "resume-sid".
        args = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            self._model,
        ]
        if self._settings_path:
            args.extend(["--settings", self._settings_path])
        if self._reasoning_effort is not None:
            args.extend(["--effort", self._reasoning_effort])
        if resume_mode == "resume-sid":
            if self._conversation_id is None:
                raise RuntimeError("resume-sid requested with no session_id")
            args.extend(["--resume", self._conversation_id])
        elif resume_mode != "fresh":
            raise RuntimeError(f"unsupported Claude resume_mode: {resume_mode}")

        if self._max_turns is not None:
            args.extend(["--max-turns", str(self._max_turns)])
        if self._allowed_tools is not None:
            args.extend(["--allowedTools", ",".join(self._allowed_tools)])
        if self._disallowed_tools is not None:
            args.extend(["--disallowedTools", ",".join(self._disallowed_tools)])
        return args

    def _wrap_skill_invocation(self, prompt: str) -> str:
        if not self._skill_prompt_prefix:
            return prompt
        if self._invocation_type.strip().lower() != "skill":
            return f"{self._skill_prompt_prefix}{prompt}"
        return f"{self._skill_prompt_prefix}<command-args>{prompt}</command-args>"

    def _build_prompt(self, query: Query, *, include_skill: bool = True) -> str:
        if self._slack_bridge:
            return self._build_prompt_slack_bridge(query, include_skill=include_skill)
        return self._build_prompt_direct(query, include_skill=include_skill)

    def _build_prompt_direct(self, query: Query, *, include_skill: bool = True) -> str:
        # Query + schema embedded directly in the user message (stdout variant).
        parts: list[str] = []
        # Identity intro only fires on fresh-session turns (once per rollout in
        # stateful mode, once per instance in baseline mode after reset()).
        if include_skill and self._phantom_intro:
            parts.append(_PHANTOM_INTRO_TEMPLATE)
        if self._preamble:
            parts.append(f"PRE-AMBLE:\n{self._preamble}")
        if self._pending_feedback:
            parts.append(f"FEEDBACK FROM PREVIOUS ACTION: {self._pending_feedback}\n")
        parts.append(query.prompt)
        parts.append(schema_to_prompt_instruction(query.response_schema))
        parts.append(
            "\n\n" + self._memory_instruction.format(memory_dir=_CONTAINER_MEMORY_DIR)
        )
        prompt = "\n".join(parts)
        if include_skill:
            return self._wrap_skill_invocation(prompt)
        return prompt

    def _build_prompt_slack_bridge(
        self, query: Query, *, include_skill: bool = True
    ) -> str:
        # Slack variant: claude posts the JSON answer via slack_cli, not stdout.
        # Section ordering matches production phantom; memory + write hint repeat
        # every turn since our intro strips the doc that would otherwise teach them.
        parts: list[str] = []
        if include_skill and self._phantom_intro:
            parts.append(_PHANTOM_INTRO_TEMPLATE_SLACK)
        if self._preamble:
            parts.append(f"PRE-AMBLE:\n{self._preamble}")
        if self._pending_feedback:
            parts.append(f"FEEDBACK FROM PREVIOUS ACTION: {self._pending_feedback}\n")
        # Skip memory in baseline mode: each instance gets a fresh system, and
        # the empty section + write hint can push claude into text-only replies.
        baseline_mode = getattr(self, "_baseline_mode", False)
        if not baseline_mode:
            memory_content = self._read_phantom_memory().strip()
            memory_body = memory_content if memory_content else _MEMORY_EMPTY_FALLBACK
            parts.append(f"## Your Memory\n\n{memory_body}\n\n---\n")
        parts.append(f"## Current Task\n\n{query.prompt}")
        parts.append(schema_to_prompt_instruction(query.response_schema))
        prompt = "\n".join(parts) + _SLACK_REPLY_INSTRUCTION
        if not baseline_mode:
            prompt += _MEMORY_WRITE_HINT
        if include_skill:
            return self._wrap_skill_invocation(prompt)
        return prompt

    def _prepare_conversation(self, query: Query) -> dict[str, Any]:
        previous_conversation_id = self._conversation_id
        resumed_this_turn = self._conversation_id is not None
        return {
            "mode": "single_conversation"
            if self._single_conversation
            else "per_instance_conversation",
            "single_conversation": self._single_conversation,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "invocation_mode": "resume-sid" if resumed_this_turn else "fresh",
            "resumed_this_turn": resumed_this_turn,
            "conversation_id_before": previous_conversation_id,
            "conversation_id_for_turn": self._conversation_id,
        }

    def _build_repair_prompt(
        self,
        original_prompt: str,
        assistant_text: str,
        validation_error: str,
    ) -> str:
        return "\n".join(
            [
                "Your previous response did not match the required benchmark schema.",
                f"Validation error: {validation_error}",
                "Previous response:",
                assistant_text,
                "Return ONLY a corrected JSON object for the same task.",
                "Original prompt:",
                original_prompt,
            ]
        )

    def _run_claude(self, prompt: str, resume_mode: str) -> list[dict[str, Any]]:
        # Invoke claude once; retry once on non-zero exit.
        def _attempt(mode: str) -> subprocess.CompletedProcess[str]:
            args = self._build_cli_args(mode)
            command = " ".join([*args, "--", shlex.quote(prompt)])
            logger.info(
                "Calling phantom/claude (mode=%s, interaction=%d, prompt=%d chars)",
                mode,
                self._interaction_count + 1,
                len(prompt),
            )
            return self._docker_exec(command, timeout=self._timeout)

        try:
            result = _attempt(resume_mode)
        except subprocess.TimeoutExpired as exc:
            timeout = exc.timeout if exc.timeout is not None else self._timeout
            raise RuntimeError(
                f"LLM call failed: Claude timed out after {timeout} seconds"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            retry_mode = "resume-sid" if self._conversation_id is not None else "fresh"
            logger.warning(
                "Claude failed (rc=%d), retrying as %s: %s",
                result.returncode,
                retry_mode,
                detail[:300],
            )
            try:
                result = _attempt(retry_mode)
            except subprocess.TimeoutExpired as exc:
                timeout = exc.timeout if exc.timeout is not None else self._timeout
                raise RuntimeError(
                    f"LLM call failed: Claude timed out after {timeout} seconds during retry"
                ) from exc
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"Claude exited with code {result.returncode} after retry: {detail}"
                )

        events = _parse_events(result.stdout)
        logger.info("Claude returned %d events", len(events))
        self._jsonl_log.extend(events)
        session_ids = _extract_session_ids(events)
        self._claude_conversation_session_ids.update(session_ids)
        if session_ids:
            self._conversation_id = session_ids[0]
        return events

    def _retry_mode_after_empty_response(self) -> str:
        if self._conversation_id is not None:
            return "resume-sid"
        return "fresh"

    def _run_assistant_turn(
        self,
        prompt: str,
        *,
        resume_mode: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        events = self._run_claude(prompt, resume_mode=resume_mode)
        try:
            assistant_text = _extract_assistant_text(events)
        except ValueError as exc:
            retry_mode = self._retry_mode_after_empty_response()
            logger.warning(
                "Claude returned no assistant message, retrying as %s: %s",
                retry_mode,
                exc,
            )
            retry_events = self._run_claude(prompt, resume_mode=retry_mode)
            try:
                assistant_text = _extract_assistant_text(retry_events)
            except ValueError as retry_exc:
                raise RuntimeError(
                    f"LLM call failed: Claude returned no assistant message after retry: {retry_exc}"
                ) from retry_exc
            return assistant_text, [*events, *retry_events]
        return assistant_text, events

    def _run_action_turn_slack(
        self,
        prompt: str,
        response_schema: type[Any],
        *,
        resume_mode: str,
        repair_original_prompt: str,
    ) -> tuple[Any, list[dict[str, Any]], bool]:
        # Same shape as _run_action_turn, but the answer comes from the mock outbox
        # rather than stdout. The outbox is wiped first so an empty outbox after the
        # call unambiguously means "claude didn't post".
        self._slack_bridge_reset_state()
        events = self._run_claude(prompt, resume_mode=resume_mode)

        outbox = self._slack_bridge_drain_outbox()
        off_protocol_recovery = False
        if not outbox:
            # Claude may have answered in plain text instead of calling slack_cli.
            # Try to recover by parsing the assistant text directly.
            try:
                assistant_text = _extract_assistant_text(events)
                action = _parse_action_text(assistant_text, response_schema)
                logger.warning("Recovered JSON from assistant text; slack outbox empty")
                return action, events, True
            except Exception:
                pass
            retry_mode = "resume-sid" if self._conversation_id is not None else "fresh"
            logger.warning("Slack outbox empty; retrying as %s", retry_mode)
            retry_events = self._run_claude(prompt, resume_mode=retry_mode)
            events.extend(retry_events)
            outbox = self._slack_bridge_drain_outbox()
            off_protocol_recovery = True
            if not outbox:
                try:
                    assistant_text = _extract_assistant_text(retry_events)
                    action = _parse_action_text(assistant_text, response_schema)
                    logger.warning("Recovered JSON from retry assistant text")
                    return action, events, True
                except Exception:
                    pass
                raise RuntimeError(
                    "LLM call failed: slack outbox empty after retry "
                    "(claude did not post AND no parseable text fallback)"
                )

        assistant_text = outbox[-1].get("text", "")
        try:
            action = _parse_action_text(assistant_text, response_schema)
            return action, events, off_protocol_recovery
        except Exception as exc:
            # Schema-repair: ask claude to fix the JSON and re-post via slack.
            repair_prompt = (
                self._build_repair_prompt(
                    repair_original_prompt,
                    assistant_text,
                    str(exc),
                )
                + _SLACK_REPLY_INSTRUCTION
            )
            repair_resume_mode = (
                "resume-sid" if self._conversation_id is not None else "fresh"
            )
            self._slack_bridge_reset_state()
            repair_events = self._run_claude(
                repair_prompt, resume_mode=repair_resume_mode
            )
            events.extend(repair_events)
            outbox = self._slack_bridge_drain_outbox()
            if not outbox:
                # Same recovery as the first-call empty-outbox path.
                try:
                    repair_assistant_text = _extract_assistant_text(repair_events)
                    action = _parse_action_text(repair_assistant_text, response_schema)
                    logger.warning("Recovered JSON from repair assistant text")
                    return action, events, True
                except Exception:
                    pass
                raise RuntimeError(
                    "LLM call failed: slack outbox empty during JSON-repair retry"
                )
            action = _parse_action_text(outbox[-1].get("text", ""), response_schema)
            return action, events, True

    def _run_action_turn(
        self,
        prompt: str,
        response_schema: type[Any],
        *,
        resume_mode: str,
        repair_original_prompt: str,
    ) -> tuple[Any, list[dict[str, Any]], bool]:
        assistant_text, all_events = self._run_assistant_turn(
            prompt,
            resume_mode=resume_mode,
        )

        try:
            action = _parse_action_text(assistant_text, response_schema)
            return action, all_events, False
        except Exception as exc:
            repair_prompt = self._build_repair_prompt(
                repair_original_prompt,
                assistant_text,
                str(exc),
            )
            repair_resume_mode = (
                "resume-sid" if self._conversation_id is not None else "fresh"
            )
            repair_events = self._run_claude(
                repair_prompt,
                resume_mode=repair_resume_mode,
            )
            all_events.extend(repair_events)
            try:
                repaired_text = _extract_assistant_text(repair_events)
            except ValueError as repair_exc:
                raise RuntimeError(
                    f"LLM call failed: Claude repair returned no assistant message: {repair_exc}"
                ) from repair_exc
            action = _parse_action_text(repaired_text, response_schema)
            return action, all_events, True

    def _record_turn_usage(self, events: list[dict[str, Any]]) -> dict[str, int]:
        token_usage = _extract_token_usage(events)
        for key in self._cumulative_tokens:
            self._cumulative_tokens[key] += token_usage[key]

        usage_event = build_usage_event(
            model=self._model,
            provider="anthropic",
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
            cached_input_tokens=token_usage["cached_input_tokens"],
            cache_creation_input_tokens=token_usage["cache_creation_input_tokens"],
            call_type="completion",
        )
        cli_cost_usd = _extract_cli_cost_usd(events)
        if cli_cost_usd is not None:
            usage_event.cost_usd = cli_cost_usd
            usage_event.pricing_source = "claude_cli.total_cost_usd"
            usage_event.pricing_error = None
        self.record_usage_event(usage_event)

        logger.info(
            "Response parsed (in=%d out=%d)",
            token_usage["input_tokens"],
            token_usage["output_tokens"],
        )
        return token_usage

    def _record_memory_snapshot(self) -> dict[str, str]:
        memory_snapshot = self._read_memory_snapshot()
        if memory_snapshot or self._other_memory_files:
            self._memory_history.append(
                {
                    "step": self._interaction_count + 1,
                    "files": memory_snapshot,
                }
            )
        self._log_memory_snapshot(memory_snapshot)
        return memory_snapshot

    @staticmethod
    def _log_memory_snapshot(memory_files: dict[str, str]) -> None:
        if memory_files:
            logger.info(
                "Memory after step: %d file(s), %d chars",
                len(memory_files),
                sum(len(value) for value in memory_files.values()),
            )
            return
        logger.info("Memory after step: (empty)")

    def _finish_successful_interaction(self) -> None:
        self._pending_feedback = None
        self._interaction_count += 1

    def _build_response_metadata(
        self,
        *,
        continuity: dict[str, Any],
        token_usage: dict[str, int],
        repair_attempted: bool,
        memory_snapshot: dict[str, str],
        conversation_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "interaction_count": self._interaction_count,
            "system_type": "phantom",
            "model": self._model,
            "version": self._version,
            "conversation_id": self._conversation_id,
            "continuity": {
                **continuity,
                "conversation_id_after": self._conversation_id,
            },
            "token_usage": token_usage,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "repair_attempted": repair_attempted,
            "memory_files": memory_snapshot,
            "claude_conversation_session_ids": conversation_snapshot["session_ids"],
            "claude_conversation_jsonl_files": conversation_snapshot[
                "conversation_jsonl_files"
            ],
            "claude_conversation_jsonl_file_sizes": conversation_snapshot[
                "conversation_jsonl_file_sizes"
            ],
        }

    def respond(self, query: Query) -> Response:
        continuity = self._prepare_conversation(query)
        prompt_without_skill = self._build_prompt(query, include_skill=False)
        prompt = self._build_prompt(
            query,
            include_skill=continuity["invocation_mode"] == "fresh",
        )
        if self._slack_bridge:
            action, all_events, repair_attempted = self._run_action_turn_slack(
                prompt,
                query.response_schema,
                resume_mode=continuity["invocation_mode"],
                repair_original_prompt=prompt_without_skill,
            )
        else:
            action, all_events, repair_attempted = self._run_action_turn(
                prompt,
                query.response_schema,
                resume_mode=continuity["invocation_mode"],
                repair_original_prompt=prompt_without_skill,
            )
        token_usage = self._record_turn_usage(all_events)
        memory_snapshot = self._record_memory_snapshot()
        conversation_snapshot = self._record_claude_conversation_snapshot(
            query,
            step=self._interaction_count + 1,
        )
        self._finish_successful_interaction()

        return Response(
            action=action,
            metadata=self._build_response_metadata(
                continuity=continuity,
                token_usage=token_usage,
                repair_attempted=repair_attempted,
                memory_snapshot=memory_snapshot,
                conversation_snapshot=conversation_snapshot,
            ),
        )

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        _ = next_query
        if not self._single_conversation and observation_marks_instance_complete(
            observation
        ):
            self._conversation_id = None
        content = observation.content.strip()
        if not content:
            return
        self._record_observation_feedback(content)

    def _record_observation_feedback(self, content: str) -> None:
        self._pending_feedback = content

    def reset(self) -> None:
        self._clear_interaction_state()
        self._clear_conversation_state()
        self._clear_persistent_workspace_state()
        self._rebuild_workspace_state()

    def _clear_interaction_state(self) -> None:
        self._interaction_count = 0
        self._jsonl_log = []
        self._cumulative_tokens = dict(_EMPTY_TOKENS)
        self._memory_history = []
        self._claude_conversation_history = []
        self._claude_instance_conversations = {}
        self._pending_feedback = None

    def _clear_conversation_state(self) -> None:
        self._conversation_id = None
        self._claude_conversation_session_ids = set()

    def _clear_persistent_workspace_state(self) -> None:
        workspace = Path(self._tmp_dir)
        self._delete_claude_project_state(workspace)
        self._delete_watched_memory_files(workspace)
        # Next instance should start with an empty channel and no memory file.
        if self._slack_bridge:
            self._slack_bridge_reset_state()
            self._wipe_phantom_memory()

    @staticmethod
    def _delete_claude_project_state(workspace: Path) -> None:
        shutil.rmtree(workspace / ".claude" / "projects", ignore_errors=True)

    def _delete_watched_memory_files(self, workspace: Path) -> None:
        for relpath in self._read_other_memory_snapshot():
            path = workspace / relpath
            if path.is_file():
                path.unlink()

    def _rebuild_workspace_state(self) -> None:
        self._initialize_workspace()

    def get_run_artifacts(self) -> dict[str, Any]:
        memory_snapshot = self._read_memory_snapshot()
        conversation_jsonl_files = self._read_claude_conversation_jsonl_files()
        return {
            "artifact_type": "phantom",
            "memory_files": memory_snapshot,
            "memory_history": list(self._memory_history),
            "conversation_id": self._conversation_id,
            "claude_conversation_session_ids": sorted(
                self._claude_conversation_session_ids
            ),
            "claude_conversation_jsonl_files": conversation_jsonl_files,
            "claude_conversation_history": list(self._claude_conversation_history),
            "claude_instance_conversations": self._sorted_claude_instance_conversations(),
            "version": self._version,
            "interaction_count": self._interaction_count,
            "cumulative_tokens": dict(self._cumulative_tokens),
            "jsonl_events": self._jsonl_log,
        }

    @property
    def name(self) -> str:
        return self._name

    def _stop_container(self) -> None:
        if self._container_id is not None:
            stop_docker_container(self._container_id)
            self._container_id = None

    def __del__(self) -> None:
        if hasattr(self, "_container_id"):
            self._stop_container()
        if hasattr(self, "_tmp_dir"):
            cleanup_run_workspace(self._tmp_dir)


@register_system("phantom_slack")
class PhantomSlackSystem(PhantomSystem):
    """Phantom variant: claude posts the JSON answer via production phantom's
    slack_cli, and the host reads it from the mock outbox.

    Defaults differ from PhantomSystem to match production phantom: timeout=900,
    no --effort flag, slack-only intro, and a host-managed phantom_memory.md
    instead of the watched-memory-files mechanism.

    The __init__ signature is spelled out (rather than using **kwargs) so that
    CL-Bench's --system.X CLI flags work.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        name: str = "phantom_slack",
        seed_memory_dir: Optional[str] = None,
        timeout: int = 900,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
        docker_image: str = _DOCKER_IMAGE,
        version: str = "2.1.119",
        memory_instruction: str = "",
        single_conversation: bool = True,
        preamble_file: Optional[str] = None,
        skill: Optional[str] = None,
        invocation_type: str = "skill",
        other_memory_files: Optional[list[str]] = None,
        auth_token: Optional[str] = None,
        base_url: Optional[str] = None,
        # Production phantom doesn't pass --effort; None makes us skip it too.
        reasoning_effort: Optional[str] = None,
        settings_path: Optional[str] = _PHANTOM_SETTINGS_PATH,
        phantom_intro: bool = True,
        slack_bridge: bool = True,
        # Suppress the memory section + write hint. Use for baseline-only resumes
        # where memory is meaningless and the empty section is known to nudge
        # claude into plain-text replies that trip the print-mode hang detector.
        baseline_mode: bool = False,
    ):
        super().__init__(
            model=model,
            name=name,
            seed_memory_dir=seed_memory_dir,
            timeout=timeout,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            docker_image=docker_image,
            version=version,
            memory_instruction=memory_instruction,
            single_conversation=single_conversation,
            preamble_file=preamble_file,
            skill=skill,
            invocation_type=invocation_type,
            other_memory_files=other_memory_files,
            auth_token=auth_token,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            settings_path=settings_path,
            phantom_intro=phantom_intro,
            slack_bridge=slack_bridge,
        )
        self._baseline_mode = baseline_mode
