"""Phantom-specific artifact export helpers."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ...artifacts import register_artifact_exporter


def _as_event_list(artifacts: dict[str, Any]) -> list[Any]:
    return list(artifacts.get("jsonl_events", []))


def _baseline_instances(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    instances = artifacts.get("baseline_instances")
    if not isinstance(instances, list):
        return []
    return [instance for instance in instances if isinstance(instance, dict)]


def _instance_artifacts(instance: dict[str, Any]) -> dict[str, Any]:
    artifacts = instance.get("system_artifacts")
    return artifacts if isinstance(artifacts, dict) else {}


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _conversation_jsonl_files(artifacts: dict[str, Any]) -> dict[str, Any]:
    return _dict_value(artifacts.get("claude_conversation_jsonl_files"))


def _conversation_session_ids(artifacts: dict[str, Any]) -> list[str]:
    session_ids = artifacts.get("claude_conversation_session_ids")
    if not isinstance(session_ids, list):
        return []
    return [str(session_id) for session_id in session_ids]


def _instance_conversations(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = artifacts.get("claude_instance_conversations")
    if not isinstance(snapshots, list):
        return []
    return [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]


def _trace_instance_identity(
    trace: dict[str, Any],
    fallback_index: int,
) -> tuple[str | None, int]:
    """Return the instance id/index represented by a single-instance trace."""
    instance_id: str | None = None
    instance_index: int | None = None

    for outcome in trace.get("instance_outcomes", []) or []:
        if not isinstance(outcome, dict):
            continue
        raw_index = outcome.get("instance_index")
        if isinstance(raw_index, int):
            instance_index = raw_index
        raw_id = outcome.get("instance_id")
        if raw_id is not None:
            instance_id = str(raw_id)
        if instance_index is not None:
            return instance_id, instance_index

    execution = trace.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("run_index"), int):
        instance_index = execution["run_index"]

    for interaction in trace.get("interactions", []) or []:
        if not isinstance(interaction, dict):
            continue
        query = interaction.get("query")
        if not isinstance(query, dict):
            continue
        raw_index = query.get("instance_index")
        if instance_index is None and isinstance(raw_index, int):
            instance_index = raw_index
        raw_id = query.get("instance_id")
        if instance_id is None and raw_id is not None:
            instance_id = str(raw_id)
        if instance_index is not None:
            return instance_id, instance_index

    return instance_id, fallback_index if instance_index is None else instance_index


def build_claude_baseline_system_artifacts(
    trace_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Wrap per-worker Claude artifacts so the exporter can save baselines."""
    baseline_instances: list[dict[str, Any]] = []
    for fallback_index, trace in enumerate(trace_payloads):
        system_artifacts = trace.get("system_artifacts")
        if not isinstance(system_artifacts, dict):
            continue
        if system_artifacts.get("artifact_type") != "phantom":
            continue
        instance_id, instance_index = _trace_instance_identity(trace, fallback_index)
        baseline_instances.append(
            {
                "instance_id": instance_id,
                "instance_index": instance_index,
                "artifact_manifest": trace.get("artifacts", {}),
                "system_artifacts": system_artifacts,
            }
        )

    if not baseline_instances:
        return None
    return {
        "artifact_type": "phantom",
        "phase": "baseline",
        "baseline_instances": baseline_instances,
    }


def _safe_relative_path(raw_path: Any) -> PurePosixPath:
    """Normalize artifact keys so they cannot escape the artifact directory."""
    path = PurePosixPath(str(raw_path).replace("\\", "/"))
    parts = [part for part in path.parts if part not in ("", ".", "/")]
    safe_parts = ["_" if part == ".." else part for part in parts]
    if not safe_parts:
        safe_parts = ["unnamed"]
    return PurePosixPath(*safe_parts)


def _write_conversation_jsonl_files(
    files: dict[str, Any],
    artifact_dir: Path,
    relative_dir: str,
) -> tuple[Optional[str], list[str]]:
    if not files:
        return None, []

    base_dir = artifact_dir / relative_dir
    written_paths: list[str] = []
    for source_path, contents in sorted(files.items()):
        safe_rel = _safe_relative_path(source_path)
        target = base_dir / safe_rel.as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(contents or ""), encoding="utf-8")
        written_paths.append((PurePosixPath(relative_dir) / safe_rel).as_posix())
    return relative_dir, written_paths


def _build_claude_artifact_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    baseline_instances = _baseline_instances(artifacts)
    if baseline_instances:
        event_count = sum(
            len(_as_event_list(_instance_artifacts(instance)))
            for instance in baseline_instances
        )
        conversation_file_count = sum(
            len(_conversation_jsonl_files(_instance_artifacts(instance)))
            for instance in baseline_instances
        )
        return {
            "type": "phantom",
            "phase": "baseline",
            "baseline_instance_count": len(baseline_instances),
            "jsonl_event_count": event_count,
            "claude_conversation_jsonl_file_count": conversation_file_count,
        }

    events = _as_event_list(artifacts)
    memory_files = _dict_value(artifacts.get("memory_files"))
    conversation_files = _conversation_jsonl_files(artifacts)
    return {
        "type": "phantom",
        "conversation_id": artifacts.get("conversation_id"),
        "interaction_count": artifacts.get("interaction_count", 0),
        "jsonl_event_count": len(events),
        "memory_artifact_path": "memory" if memory_files else None,
        "memory_files": list(memory_files.keys()),
        "claude_conversation_session_ids": _conversation_session_ids(artifacts),
        "claude_conversation_jsonl_files": list(conversation_files.keys()),
        "claude_conversation_jsonl_file_count": len(conversation_files),
        "claude_instance_conversation_count": len(_instance_conversations(artifacts)),
    }


def _write_single_claude_artifacts(
    artifacts: dict[str, Any],
    artifact_dir: Path,
    *,
    include_instance_conversations: bool = True,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    events = _as_event_list(artifacts)
    with (artifact_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False, default=repr))
            fh.write("\n")
    (artifact_dir / "events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    (artifact_dir / "memory_history.json").write_text(
        json.dumps(
            artifacts.get("memory_history", []),
            indent=2,
            ensure_ascii=False,
            default=repr,
        ),
        encoding="utf-8",
    )

    memory_files = _dict_value(artifacts.get("memory_files"))
    memory_dir_artifact: Optional[Path] = None
    if memory_files:
        memory_dir_artifact = artifact_dir / "memory"
        memory_dir_artifact.mkdir(parents=True, exist_ok=True)
        for rel, contents in memory_files.items():
            target = memory_dir_artifact / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(contents or ""), encoding="utf-8")

    conversation_files = _conversation_jsonl_files(artifacts)
    conversation_dir, conversation_paths = _write_conversation_jsonl_files(
        conversation_files,
        artifact_dir,
        "claude_conversations/final",
    )
    conversation_history_path = "claude_conversation_history.json"
    (artifact_dir / conversation_history_path).write_text(
        json.dumps(
            artifacts.get("claude_conversation_history", []),
            indent=2,
            ensure_ascii=False,
            default=repr,
        ),
        encoding="utf-8",
    )

    instance_conversation_manifests: list[dict[str, Any]] = []
    if include_instance_conversations:
        for ordinal, snapshot in enumerate(_instance_conversations(artifacts)):
            instance_index = snapshot.get("instance_index")
            if not isinstance(instance_index, int):
                instance_index = ordinal
            instance_path = f"instances/instance_{instance_index:04d}"
            instance_dir = artifact_dir / instance_path
            snapshot_files = _dict_value(snapshot.get("conversation_jsonl_files"))
            snapshot_conversation_dir, snapshot_conversation_paths = (
                _write_conversation_jsonl_files(
                    snapshot_files,
                    artifact_dir,
                    f"{instance_path}/claude_conversations",
                )
            )
            metadata = {
                "instance_id": snapshot.get("instance_id"),
                "instance_index": snapshot.get("instance_index"),
                "step": snapshot.get("step"),
                "conversation_id": snapshot.get("conversation_id"),
                "session_ids": snapshot.get("session_ids") or [],
                "conversation_jsonl_files": list(snapshot_files.keys()),
                "conversation_jsonl_paths": snapshot_conversation_paths,
            }
            instance_dir.mkdir(parents=True, exist_ok=True)
            (instance_dir / "claude_conversation_manifest.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, default=repr),
                encoding="utf-8",
            )
            instance_conversation_manifests.append(
                {
                    **metadata,
                    "path": instance_path,
                    "claude_conversation_jsonl_dir": snapshot_conversation_dir,
                    "manifest_path": f"{instance_path}/claude_conversation_manifest.json",
                }
            )

    manifest = {
        "artifact_type": "phantom",
        "conversation_id": artifacts.get("conversation_id"),
        "version": artifacts.get("version"),
        "interaction_count": artifacts.get("interaction_count"),
        "cumulative_tokens": artifacts.get("cumulative_tokens"),
        "events_jsonl_path": "events.jsonl",
        "events_json_path": "events.json",
        "memory_history_path": "memory_history.json",
        "claude_conversation_history_path": conversation_history_path,
        "claude_conversation_session_ids": _conversation_session_ids(artifacts),
        "claude_conversation_jsonl_dir": conversation_dir,
        "claude_conversation_jsonl_files": list(conversation_files.keys()),
        "claude_conversation_jsonl_paths": conversation_paths,
        "claude_instance_conversations": instance_conversation_manifests,
        "memory_artifact_path": "memory" if memory_dir_artifact is not None else None,
        "memory_files": list(memory_files.keys()),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    return manifest


def _save_claude_artifacts(
    artifacts: dict[str, Any],
    trace_path: Path,
) -> Optional[Path]:
    artifact_dir = trace_path.parent / "artifacts" / trace_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    baseline_instances = _baseline_instances(artifacts)
    if baseline_instances:
        instance_manifests: list[dict[str, Any]] = []
        for ordinal, instance in enumerate(baseline_instances):
            instance_index = instance.get("instance_index")
            if not isinstance(instance_index, int):
                instance_index = ordinal
            instance_dir = artifact_dir / "instances" / f"instance_{instance_index:04d}"
            instance_artifacts = _instance_artifacts(instance)
            instance_manifest = _write_single_claude_artifacts(
                instance_artifacts,
                instance_dir,
                include_instance_conversations=False,
            )
            instance_manifests.append(
                {
                    "instance_index": instance_index,
                    "path": f"instances/instance_{instance_index:04d}",
                    "artifact_manifest": instance.get("artifact_manifest", {}),
                    "claude_manifest": instance_manifest,
                }
            )

        manifest = {
            "artifact_type": "phantom",
            "phase": "baseline",
            "trace_file": str(trace_path),
            "baseline_instance_count": len(instance_manifests),
            "claude_conversation_jsonl_file_count": sum(
                len(
                    (instance_manifest.get("claude_manifest") or {}).get(
                        "claude_conversation_jsonl_files"
                    )
                    or []
                )
                for instance_manifest in instance_manifests
            ),
            "instances": instance_manifests,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=repr),
            encoding="utf-8",
        )
        return artifact_dir

    single_manifest = _write_single_claude_artifacts(artifacts, artifact_dir)
    single_manifest["trace_file"] = str(trace_path)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(single_manifest, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    return artifact_dir


def ensure_registered() -> None:
    register_artifact_exporter(
        "phantom",
        build_manifest=_build_claude_artifact_manifest,
        save=_save_claude_artifacts,
        build_baseline=build_claude_baseline_system_artifacts,
    )
