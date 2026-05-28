"""Slack-backend mock for the phantom container.

Auto-installed via a .pth file at every Python startup. Patches SlackClient so
send_message writes to outbox.jsonl and get_channel_history reads inbox.jsonl,
both under PHANTOM_MOCK_DIR (default /workspace/.phantom_mock). No network calls.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mock_dir() -> Path:
    return Path(os.environ.get("PHANTOM_MOCK_DIR", "/workspace/.phantom_mock"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _patch_slack_client() -> None:
    try:
        from slack_interface import SlackClient
    except Exception:
        return

    if getattr(SlackClient, "_clbench_patched", False):
        return

    def send_message(
        self,
        token: str,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        username: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        ts = f"{time.time():.6f}"
        _append_jsonl(
            _mock_dir() / "outbox.jsonl",
            {
                "text": text,
                "channel": channel,
                "thread_ts": thread_ts,
                "username": username,
                "ts": ts,
            },
        )
        return {
            "ok": True,
            "channel": channel,
            "ts": ts,
            "message": {"text": text, "ts": ts, "user": "U_PHANTOM"},
        }

    def get_channel_history(
        self,
        token: str,
        channel: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        # Interleave inbox + outbox newest-first, matching real Slack semantics.
        inbox = _read_jsonl(_mock_dir() / "inbox.jsonl")
        outbox = _read_jsonl(_mock_dir() / "outbox.jsonl")
        for msg in outbox:
            msg.setdefault("user", "U_PHANTOM")
            msg.setdefault("bot_id", "B_PHANTOM")
        merged = inbox + outbox
        merged.sort(key=lambda m: float(m.get("ts", 0)))
        return list(reversed(merged))[:limit]

    def get_thread_replies(
        self,
        token: str,
        channel: str,
        thread_ts: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        msgs = _read_jsonl(_mock_dir() / "inbox.jsonl")
        return [m for m in msgs if m.get("thread_ts") == thread_ts][:limit]

    SlackClient.send_message = send_message
    SlackClient.get_channel_history = get_channel_history
    SlackClient.get_thread_replies = get_thread_replies
    SlackClient._clbench_patched = True


def install() -> None:
    _mock_dir().mkdir(parents=True, exist_ok=True)
    _patch_slack_client()


def write_inbound_message(text: str, user: str = "U_BENCH") -> Dict[str, Any]:
    entry = {
        "user": user,
        "text": text,
        "ts": f"{time.time():.6f}",
        "id": uuid.uuid4().hex,
    }
    _append_jsonl(_mock_dir() / "inbox.jsonl", entry)
    return entry


def read_outbound_messages() -> List[Dict[str, Any]]:
    return _read_jsonl(_mock_dir() / "outbox.jsonl")


def reset_state(mock_dir: Optional[Path] = None) -> None:
    target = mock_dir or _mock_dir()
    for name in ("inbox.jsonl", "outbox.jsonl"):
        path = target / name
        if path.is_file():
            path.unlink()


install()
