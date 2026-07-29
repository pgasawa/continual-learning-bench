"""DockerOuroborosEngine — runs the isolated Ouroboros server INSIDE a Docker container
(leak-proof: the CL-Bench DB lives in the host shim, never mounted into the container).

Drop-in for ``_launcher.OuroborosEngine`` (same surface: ``_start`` / ``data_root`` /
``server.base_url`` / ``server.wait_for_health`` / ``mm`` / ``on_instance_boundary`` /
``close``), but boots via ``docker run --init`` instead of a host ``Popen``.

PROVEN recipe (see clbench_docker_plan.md):
  - ``--init`` is load-bearing: without it server.py is PID 1 and process_custody reads
    getppid()==1 as "orphaned" -> workers self-SIGKILL (signal 9, NOT OOM).
  - source git-cloned to run_root/clone (frozen; remote_work skill is INJECTED from clbench_skill/),
    mounted at /obo/repo; data at /obo/data; NO benchmark DB/dataset mounted.
  - host shim binds 127.0.0.1; the container reaches it via host.docker.internal (the agent's
    shim URL is rewritten by shim_url_for_agent()).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.parse

LIVE_SETTINGS = pathlib.Path.home() / "Ouroboros" / "data" / "settings.json"
# Stable skill source OUTSIDE the churning Ouroboros clone (its branch can switch under us and orphan a
# committed skill). The bootstrap seeds skills from the FILESYSTEM, so injecting an uncommitted copy into
# the per-run clone's skills/ before boot is robust and branch-independent.
_SKILL_SRC = pathlib.Path(__file__).resolve().parents[3] / "clbench_skill" / "remote_work"


def _runs_root() -> pathlib.Path:
    root = os.environ.get("OUROBOROS_BENCH_RUNS_ROOT") or str(pathlib.Path.home() / "cl_bench_runs")
    p = pathlib.Path(root).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clone_root(ouroboros_repo: str) -> pathlib.Path:
    repo = pathlib.Path(ouroboros_repo or os.environ.get("OUROBOROS_BENCH_CLONE", "")).expanduser()
    if not (repo / "devtools" / "benchmarks" / "common" / "server_runner.py").exists():
        raise RuntimeError("docker engine: set --ouroboros-repo (or $OUROBOROS_BENCH_CLONE) to an "
                           "Ouroboros CLONE (NEVER the live repo).")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _api(base: str, method: str, path: str, body=None, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}


_FINAL_STATUSES = {"completed", "failed", "cancelled", "rejected_duplicate"}


class _DockerServer:
    """Thin stand-in for IsolatedServer exposing the bits run_clbench_bridge_agent uses."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def wait_for_health(self, timeout: float = 180) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                st = _api(self.base_url, "GET", "/api/state", timeout=8)
                if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0:
                    return True
            except Exception:
                pass
            time.sleep(3)
        return False

    def healthy(self) -> bool:
        """Deep liveness: supervisor OK and at least one worker PROCESS alive right now.
        wait_for_health's workers_total>0 passes the boot-crash class where workers
        register and then die (partial repo mount -> ModuleNotFound cascade: the cohort
        baseline container kept accepting tasks into a dead queue and burned N x 900s);
        workers_alive is a live proc.is_alive() count and drops to 0 in that state."""
        try:
            st = _api(self.base_url, "GET", "/api/state", timeout=8)
        except Exception:
            return False
        if not st.get("supervisor_ready"):
            return False
        alive = st.get("workers_alive")
        if alive is None:  # older server without the field
            alive = st.get("workers_total")
        return int(alive or 0) > 0

    def task_status(self, task_id: str) -> str:
        """One-shot engine-side task status ('' on transport failure). Used by the
        live bridge's dead-vs-thinking watchdog (queued-stall fail-fast)."""
        try:
            rec = _api(self.base_url, "GET", "/api/tasks/" + urllib.parse.quote(task_id), timeout=15)
            return str(rec.get("status") or "")
        except Exception:
            return ""

    def current_sha(self) -> str:
        try:
            return str(_api(self.base_url, "GET", "/api/state", timeout=8).get("sha") or "")
        except Exception:
            return ""

    def submit(self, description: str, *, memory_mode: str = "shared", timeout_sec: int = 900,
               resume_from_task_id: str = "", resume_mode: str = "") -> str:
        """Submit one task to the in-container server (mirrors IsolatedServer.submit /api/tasks)."""
        body = {
            "description": description,
            "memory_mode": memory_mode,
            # Explicit full-toolset pin (CC tool parity): an absent key resolves to [] on the
            # server today, but pinning it here guards the per-action path against any future
            # default drift (the live path already passes DISABLED_TOOLS explicitly).
            "disabled_tools": [],
            "actor_id": "remote-driver",
            "source": "remote-driver",
            "timeout_sec": timeout_sec,
            "metadata": {"source": "remote-driver", "delegation_role": "root"},
        }
        if resume_from_task_id:
            body["resume_from_task_id"] = resume_from_task_id
            if resume_mode:
                body["resume_mode"] = resume_mode
        created = _api(self.base_url, "POST", "/api/tasks", body, timeout=60)
        return str(created.get("task_id") or "")

    def wait_task(self, task_id: str, timeout: float = 2400) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = _api(self.base_url, "GET", "/api/tasks/" + urllib.parse.quote(task_id), timeout=30)
                if str(result.get("status") or "") in _FINAL_STATUSES:
                    return result
            except Exception:
                pass  # transient (e.g. server re-exec restart) — keep polling
            time.sleep(3)
        return {"status": "timeout"}

    def cancel_task(self, task_id: str) -> None:
        try:
            _api(self.base_url, "POST",
                 "/api/tasks/" + urllib.parse.quote(task_id) + "/cancel", {}, timeout=30)
        except Exception:
            pass


# Objective for FORCED evolution windows (evolution_trigger="forced"). Generic and
# HOW-constrained only — no benchmark facts, no verifier knowledge (methodology rule);
# same envelope as the swe_bench_pro e1v2 steer. Overridable via persistent_objective.
_FORCED_EVOLUTION_OBJECTIVE = (
    "Review your recent task experience in this session's memory (scratchpad journal, "
    "reflections). Choose the single highest-leverage improvement to YOUR OWN operating "
    "machinery — prompts, SYSTEM.md guidance, or workflow discipline — that would raise "
    "your reliability on analytical question-answering work with tools. Implement it as "
    "ONE focused reviewed commit made with the commit_reviewed tool (shell git is "
    "blocked; no git config needed; a PATCH version bump is fine if the commit gate "
    "requires it). Do not touch unrelated files."
)


def _write_forced_evolution_request(data_root: pathlib.Path, objective: str) -> bool:
    """Durable promotion signal for a FORCED evolution window (bridge-side analog of
    post_task_evolution._write_request; the supervisor idle tick consumes it via
    apply_pending_request). Native maybe_promote runs in a post-task daemon thread that
    races the worker lifecycle at chunk boundaries AND honestly declines on an empty
    backlog — the forced file makes the treatment lane deterministic. Idempotence
    guards: skip when a request is already pending or a campaign is already enabled
    (apply_pending_request would leave/drop a second request anyway; don't hijack an
    in-flight cycle from a previous window). Returns True when a request was written."""
    state_dir = data_root / "state"
    req = state_dir / "post_task_evolution_request.json"
    try:
        if req.exists():
            return False
        st = {}
        try:
            st = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        if bool(st.get("evolution_mode_enabled")):
            return False  # a previous window's cycle is still in flight
        payload = {
            "schema_version": 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "objective": str(objective or "").strip() or _FORCED_EVOLUTION_OBJECTIVE,
            "requires_plan_review": True,
            "backlog_id": "",
            "source": "clbench_bridge_forced_window",
            "origin_task_id": "",
        }
        tmp = req.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(req)  # atomic publish — supervisor polls every tick
        return True
    except Exception as exc:
        print(f"[ouroboros-docker] forced evolution request write failed: {exc!r}",
              file=sys.stderr)
        return False


class DockerOuroborosEngine:
    def __init__(self, *, ouroboros_repo: str, model: str, mode: str, evolution: bool,
                 steer: str = "", task_timeout_sec: int = 900, primer: str = "",
                 cadence: str = "llm", image: str = "clbench-ouroboros:dev",
                 max_workers: int = 10, resume: bool | str = False,
                 evolution_trigger: str = "native") -> None:
        self.repo = _clone_root(ouroboros_repo)
        self.model = model.split("openrouter/", 1)[-1] if model.startswith("openrouter/") else model
        self.mode = mode
        self.evolution = bool(evolution) and mode == "stateful"
        self.evolution_trigger = str(evolution_trigger or "native").strip().lower()
        self.steer = steer
        self.cadence = cadence
        self.task_timeout = int(task_timeout_sec)
        self.mm = "shared"
        self.image = image
        self.max_workers = int(max_workers)
        self._run_root = None
        self._data_root = None
        self._server = None
        self._cname = None
        self._started = False
        self.runtime_attestation: dict = {}
        # chain conversation-resume across run_turn calls (CC single_conversation).
        # DEFAULT mode = continuation (true CC --resume analog: stored system verbatim,
        # append-only; 6q A/B 0.611 vs splice 0.144); "splice" is the explicit legacy opt-in.
        self.resume = bool(resume)
        _r = str(resume).strip().lower()
        self.resume_mode = ("splice" if _r == "splice" else "continuation") if self.resume else ""
        self._last_task_id = None
        from devtools.benchmarks.common.server_runner import (  # noqa: E402
            build_isolated_settings, seed_owner_state, free_port, absorbed_cycles_done)
        from supervisor.state import reset_per_task_budget, ISOLATED_BENCHMARK_SENTINEL  # noqa: E402
        self._build_isolated_settings = build_isolated_settings
        self._seed_owner_state = seed_owner_state
        self._free_port = free_port
        self._absorbed_cycles_done = absorbed_cycles_done
        self._reset_per_task_budget = reset_per_task_budget
        self._SENTINEL = ISOLATED_BENCHMARK_SENTINEL

    # ----- read-only accessors (mirror OuroborosEngine) -----
    @property
    def data_root(self):
        return self._data_root

    @property
    def server(self):
        return self._server

    def shim_url_for_agent(self, host_shim_url: str) -> str:
        """URL the CONTAINER should use for the host shim.

        A loopback-bound shim is only reachable through the host.docker.internal alias;
        a shim already bound to the docker bridge IP (CLBENCH_SHIM_BIND, the Linux
        rootful-daemon recipe) is reachable verbatim and must NOT be rewritten."""
        if os.environ.get("CLBENCH_SHIM_BIND", "").strip() not in ("", "127.0.0.1", "localhost"):
            return host_shim_url
        return host_shim_url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")

    def _overrides(self) -> dict:
        m = self.model
        ov = dict(
            # The BENCH key must win: container settings are merged over the LIVE app's
            # settings.json, whose OPENROUTER_API_KEY otherwise shadows the .env key at boot
            # (apply_settings_to_env writes settings->env over the docker -e value). The live
            # key hitting its per-key limit 402'd whole runs (qsr A/B, 2026-07-10).
            OPENROUTER_API_KEY=os.environ.get("OPENROUTER_API_KEY", ""),
            OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_PRE_PUSH_TESTS="0",
            OUROBOROS_POST_TASK_EVOLUTION="true" if self.evolution else "false",
            OUROBOROS_MODEL=m, OUROBOROS_MODEL_HEAVY=m, OUROBOROS_MODEL_LIGHT=m, OUROBOROS_MODEL_CODE=m,
            OUROBOROS_MODEL_CONSCIOUSNESS=m, OUROBOROS_MODEL_FALLBACK=m, OUROBOROS_MODEL_FALLBACKS=m,
            OUROBOROS_MODEL_DEEP_SELF_REVIEW=m, OUROBOROS_REVIEW_MODELS=",".join([m, m, m]),
            OUROBOROS_SCOPE_REVIEW_MODEL=m, OUROBOROS_SCOPE_REVIEW_MODELS=m, OUROBOROS_WEBSEARCH_MODEL=m,
            CLAUDE_CODE_MODEL=m,
            # Single-model claim closure: vision was the one slot the pin missed; even with the
            # vision tools disabled the slot must not resolve to a foreign owner default.
            OUROBOROS_MODEL_VISION=m,
            OUROBOROS_SERVER_HOST="0.0.0.0", OUROBOROS_SERVER_PORT="8765", OUROBOROS_HOST_SERVICE_PORT="8767",
        )
        if self.evolution:
            ov["OUROBOROS_POST_TASK_EVOLUTION_CADENCE"] = self.cadence
            # Opt-in lost-update guard (engine ≥ our 655 clone): at task-done, an
            # ahead HEAD descending from tx.base_head restores a clobbered commit
            # registration instead of no_op'ing a good reviewed commit. Safe here:
            # shell git is blocked in-container, so every ahead commit is reviewed.
            ov["OUROBOROS_EVOLUTION_TX_COMMIT_GUARD"] = "git_head"
            if self.steer:
                ov["OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"] = self.steer
        # OpenRouter provider routing (429 mitigation) — in BOTH settings (apply_settings_to_env)
        # and -e (forwarded in _start), so it survives whichever wins. Native in 6.50.0 (_resolve_or_provider).
        _or_pr = os.environ.get("OUROBOROS_OR_PROVIDER")
        if _or_pr:
            ov["OUROBOROS_OR_PROVIDER"] = _or_pr
        # Live rollout scope: one task spans many questions -> the default 200-round cap would
        # cut it mid-rollout. Host exports OUROBOROS_MAX_ROUNDS; forwarded via settings + -e below.
        _mr = os.environ.get("OUROBOROS_MAX_ROUNDS")
        if _mr:
            ov["OUROBOROS_MAX_ROUNDS"] = _mr
        # Reasoning effort — CC exposes a SINGLE effort knob (=low). Mirror that: apply ONE effort
        # UNIFORMLY to every Ouroboros task type (solve/evolution/review/scope/deep-review/consciousness),
        # defaulting to low for CC parity. All EFFORT_* keys are in apply_settings_to_env's allowlist, so
        # the settings.json path reaches the container.
        _eff = os.environ.get("OUROBOROS_EFFORT_TASK") or "low"
        for _k in ("OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_EVOLUTION", "OUROBOROS_EFFORT_REVIEW",
                   "OUROBOROS_EFFORT_SCOPE_REVIEW", "OUROBOROS_EFFORT_DEEP_SELF_REVIEW",
                   "OUROBOROS_EFFORT_CONSCIOUSNESS"):
            ov[_k] = _eff
        # Conversation-resume: tell the in-container agent to capture each task's final messages so the
        # next per-action submit can continue the conversation (the Ouroboros analog of CC --resume).
        if self.resume:
            ov["OUROBOROS_RESUME_CAPTURE"] = "1"
        # Review knobs: mirror the host env INTO the isolated settings at creation, not only
        # docker -e (adversarial-review finding, CONFIRMED for 663: upstream 1543c2f added the
        # two keys to the isolated-settings allowlist, so the DESKTOP's live policy gets copied
        # in and apply_settings_to_env stomps -e at boot; overrides land AFTER the allowlist
        # copy and win). UNSET host env pins the engine defaults explicitly — a desktop set to
        # required/blocking must never silently arm a control arm (verifier's residual leak).
        for _rk, _dflt in (("OUROBOROS_TASK_REVIEW_MODE", "auto"),
                           ("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")):
            ov[_rk] = os.environ.get(_rk, "").strip() or _dflt
        # Operator env overrides (campaign knobs, ported from the v6.81.0 bridge campaign).
        # Parity defaults above stay authoritative unless the host explicitly exports one of
        # these — then the export wins, so a campaign can pin e.g. runtime=pro / split
        # review efforts / max context without editing this file ("declared vs applied").
        for _k in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_REVIEW_MODELS",
                   "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
                   "OUROBOROS_SAFETY_MODE", "OUROBOROS_MAX_SUBAGENT_DEPTH",
                   "OUROBOROS_MAX_WORKERS", "OUROBOROS_CONTEXT_MODE",
                   "OUROBOROS_CONTEXT_MODE_AUTO_LOW"):
            _v = os.environ.get(_k)
            if _v:
                ov[_k] = _v
        # Per-question cost cap. CLBENCH_ prefix is load-bearing: launcher child-env scrubbing
        # strips OUROBOROS_* names, so the cap travels under a bench-scoped name and lands in
        # the container settings, which apply_settings_to_env projects back into env.
        _ptc = os.environ.get("CLBENCH_PER_TASK_COST_USD")
        if _ptc:
            ov["OUROBOROS_PER_TASK_COST_USD"] = _ptc
        return ov

    def _attest_runtime(self, clone: pathlib.Path) -> None:
        """Runtime attestation for the CLB *docker* path (v6.76.0, plan item D3).

        The host-engine path gets this for free: `_launcher.OuroborosEngine` boots a real
        `IsolatedServer`, whose `_wait_ready()` carries the shared attestation. THIS class
        is only a thin stand-in for `IsolatedServer` (own `_DockerServer`, own health gate)
        and never calls `_wait_ready`, so without this hook the supported docker path would
        run unattested. It sits right after the health+settle gate and BEFORE any paid task
        is dispatched, so a version skew stops the run instead of poisoning its numbers.

        Uses the SHARED devtools helper — no second attestation implementation:
            runtime_attestation(base_url, repo_dir, *, expected_version="", timeout=10)
        `repo_dir` is a REQUIRED POSITIONAL — it is how the local HEAD (the commit half of
        owner decision Q7=B) is reported — so the mounted clone is passed as the second
        argument. Requires an Ouroboros bench clone at v6.75.0 or newer, where the helper
        lands with that signature.

        Policy lives in the HELPER, not here: it already fails closed on skew / unreachable
        runtime / unknown commit, and already honours the named escape
        OBO_ALLOW_EVOLVED_VOLUME (a legitimately evolved /obo/repo volume changes VERSION on
        purpose), recording `overridden: true` when it was used. So this hook keeps no second
        copy of that decision: it labels the record, stores it, and prints it.
        """
        try:
            from devtools.benchmarks.common.manifests import (
                allow_evolved_volume, runtime_attestation)
        except ImportError as exc:
            raise RuntimeError(
                "CLB docker runtime attestation requires devtools/benchmarks/common/manifests."
                f"runtime_attestation (Ouroboros bench clone >= v6.75.0): {exc}"
            ) from exc
        expected = ""
        version_path = pathlib.Path(clone) / "VERSION"
        try:
            expected = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            expected = ""
        att = dict(runtime_attestation(self._server.base_url, pathlib.Path(clone),
                                       expected_version=expected, timeout=10) or {})
        att["expected_version"] = expected
        att["source"] = "clb_docker_launcher"
        att["allow_evolved_volume"] = allow_evolved_volume()
        self.runtime_attestation = att
        print(f"[ouroboros-docker] runtime attestation: "
              f"served={att.get('runtime_version') or '?'} clone_version={expected or '?'} "
              f"head={str(att.get('repo_head') or '?')[:12]} reason={att.get('reason') or 'ok'} "
              f"overridden={bool(att.get('overridden'))}", flush=True)

    def _start(self) -> None:
        if self._started:
            return
        self._run_root = pathlib.Path(tempfile.mkdtemp(prefix="obo_dockerclbench_", dir=str(_runs_root())))
        clone = self._run_root / "clone"
        self._data_root = self._run_root / "data"
        self._data_root.mkdir(parents=True, exist_ok=True)
        # frozen sub-clone; remote_work is injected below from the repo's clbench_skill/ dir;
        # writable so E1 evolution can commit to /obo/repo without touching the source.
        subprocess.run(["git", "clone", "--no-hardlinks", "-q", str(self.repo), str(clone)], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(clone), "remote", "remove", "origin"], capture_output=True)
        # Inject the remote_work skill from the stable source (branch-churn-proof). Bootstrap seeds
        # skills/ from the filesystem at boot, so an uncommitted copy is native-trusted + loaded.
        if not _SKILL_SRC.is_dir():
            raise RuntimeError(f"remote_work skill source missing: {_SKILL_SRC}")
        skill_dst = clone / "skills" / "remote_work"
        if skill_dst.exists():
            shutil.rmtree(skill_dst)
        shutil.copytree(_SKILL_SRC, skill_dst)
        shutil.rmtree(skill_dst / "__pycache__", ignore_errors=True)
        # COMMIT the injected skill (seed commit BEFORE boot; message kept neutral —
        # the agent reads its own git log, and naming the bench there primes
        # bench-scoped rather than portable evolution rules). An untracked
        # skills/ dir makes every worker boot flag uncommitted_changes, and — fatal
        # for evolution runs — the cycle-cleanup stash sweeps it away with the
        # worktree reset (evo-pair post-mortem: reset_to_base stashed the skill;
        # workers kept the loaded tools only until the next restart). Committed, the
        # skill is part of tx.base_head and survives every reset/rescue.
        # Clone-local git identity: the container has no global config, so the FIRST
        # commit_reviewed attempt died on bare `git commit` (evo-pair forensics:
        # GIT_ERROR at 09:41:48, fallback-identity retry succeeded but skipped
        # transaction registration → restart gate blocked → cycle no_op'd).
        subprocess.run(["git", "-C", str(clone), "config", "user.name", "Ouroboros"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(clone), "config", "user.email", "ouroboros@local"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(clone), "add", "skills/remote_work"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(clone),
                        "commit", "-q", "-m", "seed: add remote_work transport skill"],
                       capture_output=True)

        live_cfg = {}
        if LIVE_SETTINGS.exists():
            try:
                live_cfg = json.loads(LIVE_SETTINGS.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                live_cfg = {}
        cfg = self._build_isolated_settings(live_cfg, **self._overrides())
        # stateful conditions hold ONE server across the whole sequence -> the 25 default would halt
        # mid-run (evo ~$4/q x 40 = $160). Env-driven; raise for stateful runs.
        if os.environ.get("CLBENCH_SCOPE_CREDENTIALS") in ("1", "true", "yes"):
            _slots = " ".join(
                str(v) for k, v in cfg.items()
                if k.startswith("OUROBOROS_MODEL") or "REVIEW_MODEL" in k
            )
            _needs_claude_sdk = "claude_code_edit" not in os.environ.get("CLBENCH_SOLVE_DISABLED_TOOLS", "")
            _scoped = {
                "anthropic::": ("ANTHROPIC_API_KEY",) if not _needs_claude_sdk else (),
                "openai::": ("OPENAI_API_KEY",),
                "cloudru::": ("CLOUDRU_FOUNDATION_MODELS_API_KEY",),
                "gigachat::": ("GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD"),
            }
            for _pref, _names in _scoped.items():
                if _pref in _slots:
                    continue
                for _n in _names:
                    if cfg.get(_n):
                        cfg[_n] = ""
        cfg["TOTAL_BUDGET"] = float(os.environ.get("OUROBOROS_TOTAL_BUDGET") or 25.0)
        (self._data_root / "settings.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
        self._seed_owner_state(self._data_root)
        (self._data_root / self._SENTINEL).write_text("isolated docker cl_bench\n", encoding="utf-8")

        key = os.environ.get("OPENROUTER_API_KEY", "")
        env_pairs = [
            ("OUROBOROS_REPO_DIR", "/obo/repo"), ("OUROBOROS_APP_ROOT", "/obo"),
            ("OUROBOROS_DATA_DIR", "/obo/data"), ("OUROBOROS_SETTINGS_PATH", "/obo/data/settings.json"),
            ("PYTHONPATH", "/obo/repo"), ("OUROBOROS_SERVER_HOST", "0.0.0.0"),
            ("OUROBOROS_SERVER_PORT", "8765"), ("OUROBOROS_HOST_SERVICE_PORT", "8767"),
            ("OUROBOROS_WORKER_START_METHOD", "spawn"), ("OUROBOROS_MAX_WORKERS", str(self.max_workers)),
            ("OUROBOROS_PRE_PUSH_TESTS", "0"), ("OPENROUTER_API_KEY", key),
        ]
        if self.evolution:
            # MUST ride the docker -e path: custom (non-SETTINGS_DEFAULTS) keys in
            # settings.json never reach process env (the resume-port "-e flag, not
            # settings" gotcha) — the guard silently no-ops from settings alone.
            env_pairs.append(("OUROBOROS_EVOLUTION_TX_COMMIT_GUARD", "git_head"))
        # Experiment passthrough: host-set review-mode reaches the engine env directly
        # (get_task_review_mode reads os.environ). Unset host env -> engine default (auto).
        for _rk in ("OUROBOROS_TASK_REVIEW_MODE", "OUROBOROS_REVIEW_ENFORCEMENT"):
            _rv = os.environ.get(_rk, "").strip()
            if _rv:
                env_pairs.append((_rk, _rv))
        env_flags = []
        for k, v in env_pairs:
            env_flags += ["-e", f"{k}={v}"]
        # optional OpenRouter provider routing (429 mitigation) — inert unless set AND the llm.py patch is applied
        _or_pr = os.environ.get("OUROBOROS_OR_PROVIDER")
        if _or_pr:
            env_flags += ["-e", f"OUROBOROS_OR_PROVIDER={_or_pr}"]
        _mr = os.environ.get("OUROBOROS_MAX_ROUNDS")
        if _mr:
            env_flags += ["-e", f"OUROBOROS_MAX_ROUNDS={_mr}"]
        # Conversation-resume: must be a real container ENV var (not settings.json — it is not in
        # apply_settings_to_env's allowlist), so the in-container agent's capture gate sees it.
        if self.resume:
            env_flags += ["-e", "OUROBOROS_RESUME_CAPTURE=1"]
        repo_mount = f"{clone}:/obo/repo" + ("" if self.evolution else ":ro")  # rw for E1 commits
        boot_err = ""
        for attempt in (1, 2):   # HEALTH-GATE: respawn once on an unhealthy boot
            hostport = self._free_port()
            self._cname = f"clbench-obo-{os.getpid()}-{hostport}"
            subprocess.run(["docker", "rm", "-f", self._cname], capture_output=True)
            run = subprocess.run(
                ["docker", "run", "-d", "--rm", "--init", "--name", self._cname,
                 # Operator port (campaign v6.56.0, re-confirmed on the official path
                 # 2026-07-28): on a ROOTFUL daemon the container writes root-owned files
                 # into the bind-mounted data root and the host-side bridge dies with
                 # PermissionError on state/skills/remote_work/shim_target.txt. Running as
                 # the operator uid keeps the mount readable/writable from both sides.
                 "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/obo/data",
                 "--add-host=host.docker.internal:host-gateway",
                 "-p", f"127.0.0.1:{hostport}:8765",
                 "-v", repo_mount, "-v", f"{self._data_root}:/obo/data", "-w", "/obo/repo",
                 *env_flags, self.image, "python", "server.py"],
                capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"docker run failed: {run.stderr[:500]}")
            self._server = _DockerServer(f"http://127.0.0.1:{hostport}")
            ok = self._server.wait_for_health(timeout=300)
            if ok:
                # Settle re-check: workers can register and then crash at boot (ModuleNotFound
                # cascade on a broken mount) — wait_for_health alone accepts such a container
                # and every task dispatched into it burns the full watchdog window. Require
                # the deep probe (workers_alive>0) to hold across a short window.
                for _ in range(3):
                    time.sleep(3)
                    if not self._server.healthy():
                        ok = False
                        break
            if ok:
                self._attest_runtime(clone)
                self._started = True
                return
            logs = subprocess.run(["docker", "logs", "--tail", "30", self._cname],
                                  capture_output=True, text=True)
            boot_err = (logs.stdout + logs.stderr)[-1500:]
            subprocess.run(["docker", "rm", "-f", self._cname], capture_output=True)
            self._server = None
            print(f"[ouroboros-docker] boot attempt {attempt} unhealthy (workers dead or "
                  f"supervisor not ready){' — respawning fresh container' if attempt == 1 else ''}")
        self._cname = None
        raise RuntimeError(f"docker server never healthy after 2 boots. logs:\n{boot_err}")

    def _attempt_turn(self, prompt: str, rid: str) -> tuple[str, dict]:
        tid = self._server.submit(prompt, memory_mode=self.mm, timeout_sec=self.task_timeout,
                                  resume_from_task_id=rid, resume_mode=self.resume_mode)
        res = self._server.wait_task(tid, timeout=self.task_timeout + 300)
        if str(res.get("status") or "") == "timeout":
            self._server.cancel_task(tid)
            res = self._server.wait_task(tid, timeout=300)
        return tid, res

    def run_turn(self, prompt: str) -> str:
        """One per-action turn: submit the rendered prompt to the in-container Ouroboros server and
        return its text result. Mirrors _launcher.OuroborosEngine.run_turn (same /api/tasks protocol)."""
        self._start()
        rid = self._last_task_id if (self.resume and self._last_task_id) else ""
        tid, res = self._attempt_turn(prompt, rid)
        # CC-parity retry: the claude system retries exactly ONCE on a failed CLI call or an
        # empty assistant output (claude/system.py:601-643, 653-683). The retry resumes from
        # the SAME chain root (rid unchanged) — which also heals the replay hole a dead
        # action would otherwise leave: the second attempt rebuilds the turn in-conversation
        # instead of the benchmark just recording a fallback the transcript never shows.
        if str(res.get("status") or "") != "completed" or not str(res.get("result") or "").strip():
            print(f"[ouroboros-docker] action task {tid} unusable "
                  f"(status={res.get('status')!r}, empty={not str(res.get('result') or '').strip()}) "
                  f"— retrying once", file=sys.stderr)
            tid, res = self._attempt_turn(prompt, rid)
        if self.resume:
            # Advance the chain pointer ONLY on success: a killed/timed-out task never ran
            # capture, so re-rooting on it would silently COLD-RESTART the whole conversation
            # (load_resume_turns/load_continuation find no file and return []). Keeping the
            # previous id preserves the chain across a dead action.
            if str(res.get("status") or "") == "completed":
                self._last_task_id = tid
            else:
                print(f"[ouroboros-docker] task {tid} ended status={res.get('status')!r}; "
                      f"resume chain stays rooted at {self._last_task_id!r}", file=sys.stderr)
        return str(res.get("result") or "")

    def reset_conversation(self) -> None:
        """End the resume chain (called at a CL-Bench question boundary): the next run_turn starts a FRESH
        conversation. Cross-question continuity then flows ONLY through Ouroboros memory — not a carried
        transcript — so the rollout is memory-driven, not whole-rollout ICL (bounds cost/disk/time too)."""
        self._last_task_id = None

    def on_instance_boundary(self) -> dict:
        """Stateful+evolution: reset per-task budget + await an absorbed cycle (host-side data dir)."""
        if not self._started:
            return {}
        try:
            self._reset_per_task_budget(self._data_root, confirm_isolated=True)
        except Exception:
            pass
        if not self.evolution:
            return {}
        if self.evolution_trigger == "forced":
            # Deterministic window: file the promotion signal ourselves instead of
            # relying on the native post-task daemon thread (races the worker
            # lifecycle at boundaries; declines on an empty backlog even forced).
            wrote = _write_forced_evolution_request(self._data_root, self.steer)
            print(f"[ouroboros-docker] forced evolution window: request "
                  f"{'written' if wrote else 'skipped (pending/in-flight)'}", file=sys.stderr)
        prev_sha = self._server.current_sha()
        prev_abs = self._absorbed_cycles_done(self._data_root)
        deadline = time.time() + self.task_timeout
        req = self._data_root / "state" / "post_task_evolution_request.json"
        while time.time() < deadline:
            cur_abs = self._absorbed_cycles_done(self._data_root)
            cur_sha = self._server.current_sha()
            if cur_abs > prev_abs and cur_sha and cur_sha != prev_sha:
                return {"absorbed": True, "new_sha": cur_sha[:8], "cycles": cur_abs}
            # honest no-promotion early-out: nothing queued + queue idle
            if not req.exists():
                try:
                    st = _api(self._server.base_url, "GET", "/api/state", timeout=8)
                    if int(st.get("running_count") or 0) == 0:
                        return {"absorbed": False, "reason": "no_promotion", "cycles": cur_abs}
                except Exception:
                    pass
            time.sleep(8)
        return {"absorbed": False, "reason": "timeout", "cycles": self._absorbed_cycles_done(self._data_root)}

    def close(self) -> None:
        if self._cname:
            subprocess.run(["docker", "rm", "-f", self._cname], capture_output=True)
            self._cname = None
        self._server = None
        self._started = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
