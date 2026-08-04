"""Knowledge-card artifact export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ...artifacts import register_artifact_exporter


def _card_content(card: Any) -> str:
    if isinstance(card, dict):
        return str(card.get("content") or card.get("text") or "").strip()
    return str(card).strip()


def _build_knowledge_cards_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    cards = list(artifacts.get("knowledge_cards") or [])
    return {
        "type": "knowledge_cards",
        "knowledge_card_count": len(cards),
        "cards_stale": bool(artifacts.get("cards_stale", False)),
        "drop_stale_cards": bool(artifacts.get("drop_stale_cards", False)),
        "stateless": bool(artifacts.get("stateless", False)),
        "reflection_count": int(artifacts.get("reflection_count") or 0),
    }


def _save_knowledge_cards_artifacts(
    artifacts: dict[str, Any],
    trace_path: Path,
) -> Optional[Path]:
    artifact_dir = trace_path.parent / "artifacts" / trace_path.stem
    cards_dir = artifact_dir / "knowledge_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    raw_cards = list(artifacts.get("knowledge_cards") or [])
    card_paths: list[str] = []
    combined_parts: list[str] = []
    for index, card in enumerate(raw_cards, start=1):
        content = _card_content(card)
        filename = f"card_{index:04d}.md"
        (cards_dir / filename).write_text(content, encoding="utf-8")
        card_paths.append(f"knowledge_cards/{filename}")
        combined_parts.append(f"## Knowledge card {index}\n\n{content}")

    (artifact_dir / "knowledge_cards.md").write_text(
        "\n\n---\n\n".join(combined_parts),
        encoding="utf-8",
    )

    snapshots = list(artifacts.get("card_snapshots") or [])
    snapshots_dir = artifact_dir / "reflections"
    snapshot_paths: list[str] = []
    if snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        for snapshot in snapshots:
            index = int(snapshot.get("reflection_index") or 0)
            filename = f"reflection_{index:04d}.json"
            (snapshots_dir / filename).write_text(
                json.dumps(snapshot, indent=2, default=str),
                encoding="utf-8",
            )
            snapshot_paths.append(f"reflections/{filename}")

    manifest = {
        "artifact_type": "knowledge_cards",
        "trace_file": str(trace_path),
        "knowledge_cards_path": "knowledge_cards.md",
        "knowledge_card_paths": card_paths,
        "cards_stale": bool(artifacts.get("cards_stale", False)),
        "drop_stale_cards": bool(artifacts.get("drop_stale_cards", False)),
        "stateless": bool(artifacts.get("stateless", False)),
        "drift_notice_count": int(artifacts.get("drift_notice_count") or 0),
        "reflection_count": int(artifacts.get("reflection_count") or 0),
        "reflection_paths": snapshot_paths,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str),
        encoding="utf-8",
    )
    return artifact_dir


def ensure_registered() -> None:
    register_artifact_exporter(
        "knowledge_cards",
        build_manifest=_build_knowledge_cards_manifest,
        save=_save_knowledge_cards_artifacts,
    )
