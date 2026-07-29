"""clbench_step_shim.py — generic loopback HTTP shim around ONE ContinualLearningTask
instance for ONE question.

Domain-blind: it holds the live CL-Bench ``task`` + the current ``Query`` and
re-serializes the per-turn ``response_schema`` on every ``GET /observation`` so a
mid-instance schema swap (cohort_studies / sales_prediction) surfaces in-band.
The agent drives it through the ``remote_work`` skill:

    GET  /observation  -> {prompt, response_schema, last_observation, queries_used, budget, done}
    POST /step {action} -> validate vs LIVE response_schema -> task.step -> next observation
    GET  /_outcome      -> {reward, success, ...}   # HOST-ONLY, never a skill tool
    GET  /healthz       -> {ok, ready}

Reward/success are read ONLY off ``InstanceOutcome`` (the same official scoring
path the CL-Bench runner uses), so every domain stays comparable with no
per-domain code. The three runner helpers are imported (not re-implemented) so
the reward math is byte-identical to ``src/runtime/runner.py``.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from ...interface import (
    ContinualLearningTask,
    InstanceOutcome,
    Query,
    Response,
    TaskStepResult,
    observation_marks_instance_complete,
    require_query_instance_identity,
)
from ...runtime.runner import (
    _collect_step_outcomes,
    _make_fallback_action,
    _upsert_instance_outcomes,
)


def _infer_budget(query: Optional[Query], task: ContinualLearningTask) -> Optional[int]:
    """Surface a per-question budget to the agent (advisory; the task enforces its own)."""
    meta = (query.metadata if query is not None else None) or {}
    for key in ("budget", "max_steps", "remaining", "query_budget"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
    for attr in ("max_queries_per_question", "query_budget", "max_queries", "budget"):
        v = getattr(task, attr, None)
        if isinstance(v, int):
            return v
    return None


class TaskDriver:
    """Owns the single live task + its current query. ALL task touches (build,
    step, evaluate, get_instance_outcomes) run on ONE dedicated worker thread —
    CL-Bench tasks open thread-bound resources (e.g. database_exploration's SQLite
    connection), and ``ThreadingHTTPServer`` hands each request a fresh thread, so
    without this pinning ``task.step`` raises ``SQLite objects created in a thread
    can only be used in that same thread``. The agent is serial by design, so a
    single-worker executor adds no real latency. A lock guards the shared turn
    state read by ``/observation`` (which does NOT touch the task)."""

    def __init__(
        self,
        build: Callable[[], tuple[ContinualLearningTask, Query]],
        *,
        max_validation_retries: int = 8,
        continuous: bool = False,
    ) -> None:
        # continuous=True: ONE task runs the WHOLE instance sequence (rollout); a question's completion
        # pauses (self._done True for the agent) until the host calls advance_question() for the next
        # instance — so the schema-drift DB swap + NOTICE fire mid-sequence (idx crossing pre_drift_count).
        # continuous=False: a single sliced baseline instance (stateless), as before.
        self._continuous = bool(continuous)
        self._seq_done = False
        self._pending_next_query: Optional[Query] = None
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clbench-task")
        self._lock = threading.Lock()
        # Build the task + first query ON the worker thread (thread affinity).
        self._task, first_query = self._exec.submit(build).result()
        self._query: Optional[Query] = first_query
        require_query_instance_identity(
            first_query, context=f"{type(self._task).__name__}.reset()/reset_baseline_instance()"
        )
        self._last_obs: Optional[str] = None
        self._queries_used = 0
        self._done = False
        self._finalized = False
        self._outcomes: list[InstanceOutcome] = []
        self._retries = 0
        self._max_retries = int(max_validation_retries)
        # Seed any reset-time outcomes (mirrors the runner's initial upsert).
        _upsert_instance_outcomes(
            self._outcomes, self._on_worker(lambda: list(self._task.get_instance_outcomes()))
        )

    def _on_worker(self, fn: Callable[[], Any]) -> Any:
        """Run a task-touching callable on the single worker thread."""
        return self._exec.submit(fn).result()

    def close(self) -> None:
        self._exec.shutdown(wait=True)

    # -- observation (re-serialize the LIVE schema EVERY call) ----------------
    def observation(self) -> dict[str, Any]:
        with self._lock:
            return self._observation_locked()

    def _observation_locked(self) -> dict[str, Any]:
        """Build the observation dict; caller MUST already hold self._lock (Lock is non-reentrant)."""
        q = self._query
        schema = (
            q.response_schema.model_json_schema()
            if (q is not None and not self._done and q.response_schema is not None)
            else None
        )
        return {
            "prompt": None if q is None else q.prompt,
            "response_schema": schema,
            "last_observation": self._last_obs,
            "instance_id": None if q is None else q.instance_id,
            "instance_index": None if q is None else q.instance_index,
            "metadata": (None if q is None else q.metadata) or {},
            "queries_used": self._queries_used,
            "budget": _infer_budget(q, self._task),
            "done": self._done,
        }

    # -- step (validate vs LIVE schema; advance ONLY on success) --------------
    def step(self, action_json: Any) -> dict[str, Any]:
        with self._lock:
            if self._done or self._query is None:
                return {"ok": False, "error": "task_done", "done": True}
            schema: type[BaseModel] = self._query.response_schema
            try:
                action = schema.model_validate(action_json)
            except ValidationError as ve:
                self._retries += 1
                if self._retries > self._max_retries:
                    # Anti-no-op-loop: force a real task.step with a zero action.
                    action = _make_fallback_action(self._query)
                else:
                    return {
                        "ok": False,
                        "error": "validation_error",
                        "detail": ve.errors(),
                        "response_schema": schema.model_json_schema(),
                        "done": False,
                    }
            self._retries = 0
            result: TaskStepResult = self._on_worker(
                lambda: self._task.step(Response(action=action))
            )
            instance_complete = observation_marks_instance_complete(result.observation)
            self._advance(result)
            # PUSH model (mirrors CC's auto-fed "FEEDBACK FROM PREVIOUS ACTION"): return the FULL next
            # observation (next prompt+schema+budget+done) plus this action's result, so the agent never
            # needs a separate get_observation poll. observation() reads self._query (already advanced).
            obs = self._observation_locked()
            obs["ok"] = True
            obs["instance_complete"] = instance_complete
            return obs

    def _advance(self, result: TaskStepResult) -> None:
        self._queries_used += 1
        self._last_obs = result.observation.content
        _upsert_instance_outcomes(
            self._outcomes,
            self._on_worker(lambda: _collect_step_outcomes(self._task, result.instance_outcome)),
        )
        if self._continuous:
            # Per-question done for the AGENT; the whole-sequence done tracked separately. On a question
            # boundary HOLD the next query (don't surface the next question to the finishing agent) — the
            # host promotes it via advance_question(). The task already swapped the DB + set the NOTICE
            # (task._sync_stage_context) when the boundary crossed pre_drift_count, so the held next query
            # carries the drift NOTICE.
            instance_complete = observation_marks_instance_complete(result.observation)
            self._seq_done = bool(result.done)
            if instance_complete or self._seq_done:
                self._pending_next_query = result.next_query
                self._done = True
                if self._seq_done:
                    self._finalize()
            else:
                self._query = result.next_query
                self._done = False
        else:
            self._query = result.next_query
            self._done = bool(result.done)
            if self._done:
                self._finalize()

    def advance_question(self) -> dict[str, Any]:
        """HOST-ONLY (continuous rollout): promote the held next-instance query so the next agent task
        solves it. Returns {seq_done, next_instance_index}. No-op for sliced (stateless) mode."""
        with self._lock:
            if not self._continuous:
                return {"seq_done": self._done, "next_instance_index": None}
            if self._pending_next_query is not None:
                self._query = self._pending_next_query
                self._pending_next_query = None
            self._done = False
            self._queries_used = 0
            self._last_obs = None
            return {
                "seq_done": self._seq_done,
                "next_instance_index": (self._query.instance_index if self._query is not None else None),
            }

    def _finalize(self) -> None:
        """Merge evaluate()-time outcomes (mirrors runner._finalize_task_result) so any
        domain that only scores at evaluate-time still yields a reward. Idempotent;
        gated so nothing touches a destructive evaluate() twice. Runs on the worker."""
        if self._finalized:
            return
        self._finalized = True

        def _eval() -> list[InstanceOutcome]:
            tr = self._task.evaluate()
            return list(tr.instance_outcomes) or list(self._task.get_instance_outcomes())

        try:
            _upsert_instance_outcomes(self._outcomes, self._on_worker(_eval))
        except Exception:
            pass

    # -- host-only outcome (never a skill tool) -------------------------------
    def outcome(self) -> dict[str, Any]:
        with self._lock:
            # Finalize (destructive task.evaluate() — it CLOSES the shared DB connection) ONLY when the
            # whole task is genuinely complete, NEVER on a mid-rollout per-question /_outcome poll. The
            # standard runner calls evaluate() once at the end (_finalize_task_result); calling it on the
            # FIRST per-question poll here closed the ONE continuous-rollout connection after Q0, breaking
            # Q1..Q19 ("ERROR: Database connection not available") until the drift swap reopened it at Q20
            # — corrupting every stateful score. Per-question rewards already come from self._outcomes
            # (collected non-destructively in _advance via get_instance_outcomes); on normal completion
            # _advance(seq_done) already finalizes, so this stays a no-op on the happy path.
            seq_complete = self._seq_done if self._continuous else self._done
            if not self._finalized and seq_complete:
                self._finalize()
            last = self._outcomes[-1] if self._outcomes else None
            return {
                "reward": None if last is None else last.reward,
                "success": None if last is None else last.success,
                "raw_metric_name": None if last is None else last.raw_metric_name,
                "raw_metric_value": None if last is None else last.raw_metric_value,
                "done": self._done,
                "instance_outcomes": [
                    {
                        "instance_id": o.instance_id,
                        "instance_index": o.instance_index,
                        "reward": o.reward,
                        "success": o.success,
                    }
                    for o in self._outcomes
                ],
            }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _driver(self) -> TaskDriver:
        return self.server.driver  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"ok": True, "ready": self._driver._query is not None})
        elif self.path == "/observation":
            self._send(200, self._driver.observation())
        elif self.path == "/_outcome":  # HOST-ONLY
            self._send(200, self._driver.outcome())
        else:
            self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/step":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "bad_json"})
            return
        action = body.get("action", body) if isinstance(body, dict) else body
        self._send(200, self._driver.step(action))


def serve_task(
    build: Callable[[], tuple[ContinualLearningTask, Query]],
    *,
    max_validation_retries: int = 8,
    continuous: bool = False,
) -> tuple[ThreadingHTTPServer, str, TaskDriver]:
    """Bind 127.0.0.1:<free port>, attach a TaskDriver, serve in a daemon thread.

    ``build`` is a zero-arg callable returning ``(task, first_query)`` — it runs on
    the driver's single worker thread so the task's thread-bound resources (SQLite,
    etc.) are created on the same thread that later steps it. Typical use::

        def build():
            task = make_task(domain, num_instances)
            return task, task.reset_baseline_instance(i)
        server, url, driver = serve_task(build)

    The runner then writes ``url`` into the skill state dir. ``server.shutdown()``
    stops serving; call ``driver.close()`` to release the worker thread.
    Returns ``(server, url, driver)``."""
    driver = TaskDriver(build, max_validation_retries=max_validation_retries, continuous=continuous)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.driver = driver  # type: ignore[attr-defined]
    host, port = server.server_address
    url = f"http://{host}:{port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, url, driver
