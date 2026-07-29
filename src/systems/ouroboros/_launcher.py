"""Ouroboros-side glue for the CL-Bench bridge — reuses the devtools `IsolatedServer`.

`OuroborosEngine` runs the REAL Ouroboros agent in full isolation (throwaway sub-clone + isolated data
root + free port — NEVER the live server/data), mirroring `devtools/benchmarks/evolve_smoke.py`:
- mode=stateless (E0): one ephemeral isolated server, evolution OFF, `memory_mode="empty"`.
- mode=stateful  (E1v2): one persistent isolated server, evolution ON, `memory_mode="forked"`;
  at each CL-Bench instance boundary: `reset_per_task_budget(confirm_isolated=True)` + `wait_for_absorb`.

`clbench` MUST run under the Ouroboros venv so `IsolatedServer.start()` (which spawns
`[sys.executable, "server.py"]`) uses an interpreter that has the agent's deps.

`SidecarLedger` keeps a best-effort audit trail under `$OUROBOROS_BENCH_RUNS_ROOT`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile

LIVE_SETTINGS = pathlib.Path.home() / "Ouroboros" / "data" / "settings.json"


def _runs_root() -> pathlib.Path:
    root = os.environ.get("OUROBOROS_BENCH_RUNS_ROOT") or str(pathlib.Path.home() / "cl_bench_runs")
    p = pathlib.Path(root).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- sidecar
class SidecarLedger:
    """Best-effort manifest + per-turn JSONL ledger. Any failure is swallowed."""

    def __init__(self, *, benchmark: str, system: str, model: str, mode: str, engine: str) -> None:
        self.ok = False
        try:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = _runs_root() / benchmark / f"{system}_{mode}_{engine}_{stamp}_{os.getpid()}"
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "run_manifest.json").write_text(json.dumps({
                "schema": "ouroboros.benchmark.run_manifest.v1", "benchmark": benchmark,
                "system": system, "model": model, "mode": mode, "engine": engine,
                "created": stamp, "pid": os.getpid(), "argv": list(sys.argv),
            }, indent=2), encoding="utf-8")
            self.turns = self.dir / "turns.jsonl"
            self.ok = True
        except Exception:
            self.ok = False

    def log_turn(self, **row) -> None:
        if not self.ok:
            return
        try:
            row = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), **row}
            with self.turns.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass


# --------------------------------------------------------- real Ouroboros engine
def _clone_root(ouroboros_repo: str) -> pathlib.Path:
    repo = pathlib.Path(ouroboros_repo or os.environ.get("OUROBOROS_BENCH_CLONE", "")).expanduser()
    if not (repo / "devtools" / "benchmarks" / "common" / "server_runner.py").exists():
        raise RuntimeError("ouroboros engine: set --system.ouroboros_repo (or $OUROBOROS_BENCH_CLONE) to an "
                           "Ouroboros CLONE containing devtools/benchmarks/common/ (NEVER the live repo).")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _git(args: list[str]) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True)


# One-time orientation: paid ONCE per server (front-loads the agent's exploration into shared memory),
# so subsequent per-turn tasks are lean. The agent must understand it is a thin per-turn action emitter
# with NO local DB/files/shell — there is nothing to explore, so it shouldn't churn through tool steps.
_PRIMER = (
    "You are now connected to an automated remote-work harness as a per-turn action emitter. PROTOCOL: you "
    "receive a stream of turns; each turn gives you a task/question plus the result of your previous action, "
    "and you must reply with EXACTLY ONE action as a single JSON object matching a schema you'll be shown each "
    "turn (e.g. a database action: run one SQL query, or give the final answer). An EXTERNAL system executes "
    "your action and returns the result on the next turn. Critically: you have NO local database, NO workspace, "
    "NO files, and NO shell relevant to this — there is nothing local to read or explore, so do NOT use tools, "
    "read files, or run commands; doing so wastes the whole turn. Orient yourself ONCE now, commit this "
    "protocol to memory, and from the next turn onward simply think briefly and output ONLY the next action "
    "JSON, nothing else. Acknowledge with the single word: READY."
)


class OuroborosEngine:
    """One isolated Ouroboros server, reused for an instance's turns (stateless) or the whole
    sequence (stateful). Mirrors evolve_smoke.py. Never touches the live server/data."""

    def __init__(self, *, ouroboros_repo: str, model: str, mode: str, evolution: bool,
                 steer: str = "", task_timeout_sec: int = 900, primer: str = _PRIMER,
                 cadence: str = "llm", extra_overrides: dict | None = None) -> None:
        # extra_overrides: extra settings.json overrides merged into build_isolated_settings
        # (default {} = no change to CL-Bench behaviour). The HarnessAudit adapter uses this to
        # inject MCP_ENABLED / MCP_SERVERS so the agent's only tools are the domain bank's.
        self._extra_overrides = dict(extra_overrides or {})
        self.repo = _clone_root(ouroboros_repo)
        self.primer = primer
        # All tasks share the isolated agent's memory so the one-time orientation (and, in stateful mode,
        # cross-instance learning) persists across turns — instead of re-exploring from a blank slate.
        self.mm = "shared"
        # CL-Bench passes litellm-style "openrouter/google/gemini-3.5-flash"; Ouroboros wants the bare
        # OpenRouter id "google/gemini-3.5-flash" (it prepends openrouter itself).
        self.model = model.split("openrouter/", 1)[-1] if model.startswith("openrouter/") else model
        self.mode = mode
        self.evolution = bool(evolution) and mode == "stateful"
        self.steer = steer
        self.cadence = cadence  # post_task evolution cadence: "llm"=native self-paced, "off"=frozen, "every_n:K"
        self.task_timeout = int(task_timeout_sec)
        # import the real devtools modules from the clone (now on sys.path)
        from devtools.benchmarks.common.server_runner import (  # noqa: E402
            IsolatedServer, build_isolated_settings, seed_owner_state, absorbed_cycles_done)
        from supervisor.state import reset_per_task_budget, ISOLATED_BENCHMARK_SENTINEL  # noqa: E402
        self._IsolatedServer = IsolatedServer
        self._build_isolated_settings = build_isolated_settings
        self._seed_owner_state = seed_owner_state
        self._absorbed_cycles_done = absorbed_cycles_done
        self._reset_per_task_budget = reset_per_task_budget
        self._SENTINEL = ISOLATED_BENCHMARK_SENTINEL
        self._server = None
        self._data_root = None
        self._run_root = None
        self._started = False

    # ----- read-only accessors (for measurement harnesses; never mutate state via these) -----
    @property
    def data_root(self):
        return self._data_root          # pathlib.Path | None (isolated OUROBOROS_DATA_DIR)

    @property
    def clone_path(self):
        return (self._run_root / "clone") if self._run_root is not None else None  # throwaway git repo

    @property
    def server(self):
        return self._server             # IsolatedServer | None

    def _start(self) -> None:
        if self._started:
            return
        self._run_root = pathlib.Path(tempfile.mkdtemp(prefix="obo_clbench_", dir=str(_runs_root())))
        clone = self._run_root / "clone"
        self._data_root = self._run_root / "data"
        self._data_root.mkdir(parents=True, exist_ok=True)
        # throwaway sub-clone, isolated branch, origin removed (an evolution self-mod can never push back)
        _git(["clone", "--no-hardlinks", "-q", str(self.repo), str(clone)])
        _git(["-C", str(clone), "checkout", "-B", "ouroboros"])
        subprocess.run(["git", "-C", str(clone), "remote", "remove", "origin"], capture_output=True)
        live_cfg = {}
        if LIVE_SETTINGS.exists():
            try:
                live_cfg = json.loads(LIVE_SETTINGS.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                live_cfg = {}
        m = self.model
        # Pin EVERY model slot to the chosen model — otherwise build_isolated_settings copies the live
        # slots (review/fallback/deep-review = gpt-5.5 / opus-4.8 / sonnet-4.6 …) and the agent silently
        # spends on expensive models. One model everywhere = cheap + provider-clean + fair.
        overrides = dict(
            OUROBOROS_RUNTIME_MODE="advanced",
            # Bench mode: skip the pre-push pytest gate. The full 278-file suite takes ~318s > the 300s
            # preflight cap (and has a pre-existing failing test), which blocks every evolution commit from
            # absorbing. We measure evolution lift, not ship code — so skip the gate. (off-switch: tools/git.py)
            OUROBOROS_PRE_PUSH_TESTS="0",
            OUROBOROS_POST_TASK_EVOLUTION="true" if self.evolution else "false",
            OUROBOROS_MODEL=m, OUROBOROS_MODEL_HEAVY=m, OUROBOROS_MODEL_LIGHT=m,
            OUROBOROS_MODEL_CODE=m, OUROBOROS_MODEL_CONSCIOUSNESS=m,
            OUROBOROS_MODEL_FALLBACK=m, OUROBOROS_MODEL_FALLBACKS=m,
            OUROBOROS_MODEL_DEEP_SELF_REVIEW=m,
            OUROBOROS_REVIEW_MODELS=",".join([m, m, m]),
            OUROBOROS_SCOPE_REVIEW_MODEL=m, OUROBOROS_SCOPE_REVIEW_MODELS=m,
            OUROBOROS_WEBSEARCH_MODEL=m, CLAUDE_CODE_MODEL=m,
        )
        if self.evolution:
            overrides["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"] = self.cadence  # "llm"=native self-paced; absorb at boundaries
            if self.steer:
                overrides["OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"] = self.steer
        overrides.update(self._extra_overrides)  # e.g. HarnessAudit MCP_ENABLED / MCP_SERVERS
        cfg = self._build_isolated_settings(live_cfg, **overrides)
        cfg.setdefault("TOTAL_BUDGET", 25.0)
        settings = self._data_root / "settings.json"
        settings.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self._seed_owner_state(self._data_root)
        os.environ["OUROBOROS_DATA_DIR"] = str(self._data_root)
        # PRE_PUSH_TESTS is read from os.environ by tools/git.py:744 (NOT settings.json), and the server
        # subprocess + its workers inherit our env — so set it here too, else the 318s>300s pytest gate
        # still times out every evolution commit and blocks absorption.
        os.environ["OUROBOROS_PRE_PUSH_TESTS"] = "0"
        (self._data_root / self._SENTINEL).write_text("isolated cl_bench data root\n", encoding="utf-8")
        self._server = self._IsolatedServer(clone, self._data_root, settings)
        self._server.start(ready_timeout=300)
        self._started = True
        # One-time orientation: let the agent read its run context / BIBLE ONCE and commit the
        # per-turn protocol to shared memory, so subsequent turns don't re-explore.
        if self.primer:
            try:
                pt = min(self.task_timeout, 300)
                pid = self._server.submit(self.primer, memory_mode=self.mm, timeout_sec=pt)
                self._server.wait_task(pid, timeout=pt + 120)
            except Exception:
                pass

    def run_turn(self, prompt: str) -> str:
        self._start()
        tid = self._server.submit(prompt, memory_mode=self.mm, timeout_sec=self.task_timeout)
        res = self._server.wait_task(tid, timeout=self.task_timeout + 300)
        if str(res.get("status") or "") == "timeout":
            self._server.cancel_task(tid)
            res = self._server.wait_task(tid, timeout=300)
        return str(res.get("result") or "")

    def on_instance_boundary(self) -> dict:
        """Stateful+evolution: reset per-task budget + await an absorbed self-evolution cycle."""
        if not self._started:
            return {}
        try:
            self._reset_per_task_budget(self._data_root, confirm_isolated=True)
        except Exception:
            pass
        if self.evolution:
            return self._server.wait_for_absorb(
                self._server.current_sha(), self._absorbed_cycles_done(self._data_root),
                timeout=self.task_timeout)
        return {}

    def force_evolution(self, objective: str, *, timeout: int = 1800) -> dict:
        """Force EXACTLY ONE post_task evolution cycle on demand. Writes the durable promotion request
        (apply_pending_request gates on enabled, NOT cadence — so this works with cadence='off'), then
        blocks for the absorbed cycle. Returns the wait_for_absorb dict {absorbed,new_sha,cycles,reason}."""
        prev_sha = self._server.current_sha()
        prev_absorbed = self._absorbed_cycles_done(self._data_root)
        req = {
            "schema_version": 1,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "objective": objective,
            "requires_plan_review": False,   # MUST be False (else apply_pending_request inserts a plan_task step)
            "backlog_id": "",
            "source": "epoch_boundary",
            "origin_task_id": "",
        }
        state_dir = self._data_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_dir / "post_task_evolution_request.json.tmp"
        tmp.write_text(json.dumps(req), encoding="utf-8")
        tmp.replace(state_dir / "post_task_evolution_request.json")  # atomic publish
        return self._server.wait_for_absorb(prev_sha, prev_absorbed, timeout=timeout)

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None
            self._started = False

    def __del__(self):
        self.close()
