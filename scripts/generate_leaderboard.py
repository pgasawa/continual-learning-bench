#!/usr/bin/env python3
"""Generate website/leaderboard_data.json directly from final_results/runs artifacts.

Workflow for updating the website leaderboard:
  python scripts/generate_leaderboard.py

Then commit/deploy website/leaderboard_data.json so the site picks it up.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_final_results import build_analysis_payload  # noqa: E402

POKER_REFERENCE_DIR = REPO_ROOT / "src" / "tasks" / "exploitable_poker" / "references"

# Checked longest-first so "icl-notepad" matches before "icl".
SYSTEM_DEFS = [
    ("icl-notepad", "ICL Notepad", "icl-notepad"),
    ("icl", "ICL", "icl"),
    ("mem0", "Mem0", "mem0"),
    ("ace", "ACE", "ace"),
    ("codex", "Codex", "codex"),
    ("claude-code", "Claude Code", "claude-code"),
    ("ouroboros", "Ouroboros", "ouroboros"),
]

# Add entries here when a new model is introduced.
MODEL_DISPLAY = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-code-sonnet-4.6": "Claude Code Sonnet 4.6",
}


def parse_run_name(run_name: str) -> tuple[str, str, str, str]:
    """Return (system_label, system_class, model, display_name) for a run_name slug."""
    for prefix, label, cls in SYSTEM_DEFS:
        if run_name == prefix or run_name.startswith(prefix + "-"):
            model_slug = run_name[len(prefix) :].lstrip("-")
            model = MODEL_DISPLAY.get(model_slug, model_slug.replace("-", " ").title())
            display = f"{label} · {model}" if model else label
            return label, cls, model, display
    return run_name, run_name, "", run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default=REPO_ROOT / "final_results" / "runs", type=Path
    )
    parser.add_argument("--reference-dir", default=POKER_REFERENCE_DIR, type=Path)
    parser.add_argument(
        "--out", default=REPO_ROOT / "website" / "leaderboard_data.json", type=Path
    )
    parser.add_argument(
        "--run", action="append", dest="runs", help="Run directory name to include."
    )
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Include every run directory with task artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_analysis_payload(
        results_dir=args.results_dir,
        runs=args.runs,
        include_all_runs=args.include_all_runs,
        reference_dir=args.reference_dir,
    )

    for entry in analysis.get("leaderboard", []):
        label, cls, model, display = parse_run_name(entry["run_name"])
        entry["display_name"] = display
        entry["system_label"] = label
        entry["system_class"] = cls
        entry["model"] = model

    out = {
        "last_updated": date.today().isoformat(),
        "leaderboard": analysis.get("leaderboard", []),
        "task_summary": analysis.get("task_summary", []),
        "run_points": analysis.get("run_points", []),
        "warnings": analysis.get("warnings", {}),
        "normalization": analysis.get("normalization", {}),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")

    size_kb = args.out.stat().st_size / 1024
    n = len(out["leaderboard"])
    print(f"Wrote {n} leaderboard entries to {args.out}  ({size_kb:.0f} KB)")
    for entry in out["leaderboard"]:
        reward = entry["normalized_reward_mean"]
        print(
            f"  #{entry['rank']}  {reward:+.4f}  {entry['display_name']}  ({entry['tasks_included']} tasks)"
        )


if __name__ == "__main__":
    main()
