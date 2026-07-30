"""Schema-card artifact export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ...artifacts import register_artifact_exporter


def _build_schema_card_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    cards = list(artifacts.get("schema_cards") or [])
    return {
        "type": "schema_card",
        "schema_card_count": len(cards),
        "cards_stale": bool(artifacts.get("cards_stale", False)),
        "drop_stale_cards": bool(artifacts.get("drop_stale_cards", False)),
        "stateless": bool(artifacts.get("stateless", False)),
        "reflection_count": int(artifacts.get("reflection_count") or 0),
    }


def _save_schema_card_artifacts(
    artifacts: dict[str, Any],
    trace_path: Path,
) -> Optional[Path]:
    artifact_dir = trace_path.parent / "artifacts" / trace_path.stem
    cards_dir = artifact_dir / "schema_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    cards = [str(card) for card in list(artifacts.get("schema_cards") or [])]
    card_paths: list[str] = []
    for index, card in enumerate(cards, start=1):
        filename = f"card_{index:04d}.md"
        (cards_dir / filename).write_text(card, encoding="utf-8")
        card_paths.append(f"schema_cards/{filename}")

    combined = "\n\n---\n\n".join(
        f"## Schema card {index}\n\n{card}" for index, card in enumerate(cards, start=1)
    )
    (artifact_dir / "schema_cards.md").write_text(combined, encoding="utf-8")

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
        "artifact_type": "schema_card",
        "trace_file": str(trace_path),
        "schema_cards_path": "schema_cards.md",
        "schema_card_paths": card_paths,
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
        "schema_card",
        build_manifest=_build_schema_card_manifest,
        save=_save_schema_card_artifacts,
    )
