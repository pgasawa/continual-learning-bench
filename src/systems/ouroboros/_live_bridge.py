"""Live-task bridge: skill mechanics under the OFFICIAL CL-Bench runner.

The runner keeps ownership of the task/DB/rewards and drives the standard
``System.respond()/observe()`` cycle. Inside, ONE live Ouroboros task per question
(``conversation="question"``) or per rollout (``conversation="rollout"``) owns the
exploration loop through the unchanged ``remote_work`` skill.

This module supplies the queue-backed shim: an HTTP server with ``clbench_step_shim``'s
transport surface (GET /observation, POST /step; HTTP 200 for all in-band errors;
bare-action acceptance; validation with retry counter and force-fallback after
``max_validation_retries``) whose agent-facing replies are CC-PROSE-SHAPED: each step
returns ``{ok, done, message}`` where ``message`` is assembled exactly like the claude
system's per-turn prompt (labeled feedback + question + schema instruction + memory
note — claude/system.py:550-564), backed by the runner's respond/observe cycle:

    agent POST /step {action} ── validate ──▶ action_q ──▶ respond() returns Response
    runner task.step(...) → observe(obs, next_q) ──▶ reply mailbox ──▶ /step HTTP reply

Concurrency model (post-review hardening):
  * The skill carries NO identity and re-reads the shim target on every call, so a
    stale agent's late /step is indistinguishable from the live agent's at the HTTP
    layer. The STRUCTURAL guarantee is therefore exclusivity: a new scope is opened
    only after the previous task is confirmed terminal (bounded cancel+wait loop).
  * Epoch counter + per-step reply mailbox close the remaining in-flight races:
    actions are (epoch, action) tuples discarded on mismatch; each accepted /step
    installs a fresh mailbox under the lock, so an orphaned handler can never steal
    the live step's reply.

Failure containment (the runner aborts the WHOLE run on any exception that escapes
respond()/observe() — src/runs/common.py:283-285): every runner-facing entry point
here catches everything and degrades to a zero-value action; after
``_MAX_CONSECUTIVE_FALLBACKS`` the fallback escalates to a TERMINAL action (ANSWER "")
so a persistently dead engine forfeits the question instead of livelocking the run.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

# How long the /step handler waits for the runner to deliver the next observation.
# The runner's turnaround (task.step on SQLite + bookkeeping) is milliseconds; this
# only trips if the run is wedged. Must stay below the skill's curl --max-time 120.
_STEP_REPLY_TIMEOUT_SEC = 280.0  # > sales/codebase docker-exec steps (120s) + eval-container spin; < skill curl 290 < tool 300
# Consecutive runner-facing fallbacks before escalating to a terminal action (#4).
_MAX_CONSECUTIVE_FALLBACKS = 3
# Liveness-aware patience (dead-vs-thinking): a fixed action timeout conflates a dead
# engine with an agent legitimately still working — a dead one still burns the full
# timeout (x3 escalation) while a slow-but-alive one gets its question forfeited.
_PROBE_CHUNK_SEC = 60.0
_TERMINAL_DRAIN_SEC = 2.0     # terminal task: short queue drain for an in-flight final action
_QUEUED_STALL_SEC = 240.0     # scope task never picked up by a worker -> workers dead, fail fast
_ACTIVITY_STALL_SEC = 600.0   # no engine log write this long -> engine dead (a single long
                              # LLM call can be silent for minutes; stay generous)
_ALIVE_PATIENCE_FACTOR = 3.0  # hard cap = factor x action_timeout while the engine shows life

# Host-network outage hold: when the HOST cannot even resolve the provider, an engine
# stall is environmental, not agentic — burning the question's timeout converts a net
# outage into a 0-reward (2026-07-20 night: DNS died for hours -> 4 domains spiraled at
# q0, every question forfeited). While the outage budget lasts, waiting time is NOT
# counted against the scope. 0 disables the gate.
_NET_OUTAGE_HOLD_SEC = float(os.environ.get("OUROBOROS_NET_OUTAGE_HOLD_SEC", "21600"))
_NET_PROBE_HOST = ("openrouter.ai", 443)


def _net_ok(timeout: float = 3.0) -> bool:
    """Cheap host-connectivity probe (DNS + TCP dial)."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(_NET_PROBE_HOST[0], _NET_PROBE_HOST[1])
        return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(None)


def _log(msg: str) -> None:
    print(f"[live-bridge] {msg}", file=sys.stderr, flush=True)


class LiveBridge:
    """Queue-backed replacement for clbench_step_shim.TaskDriver's agent surface.

    Runner side: ``begin(query)`` opens a new scope (bumps the epoch, installs a fresh
    reply mailbox); ``push(...)`` publishes each step result / boundary verdict
    (delivered to the CURRENT step's mailbox); ``take_action(timeout)`` blocks until
    the agent submits a valid action for the CURRENT epoch.
    Agent side (via HTTP): GET /observation (non-consuming snapshot), POST /step.
    """

    # Default 1 = CC's exact repair budget: initial attempt + ONE error-detail reply
    # (claude/system.py repair path is a single repair turn). The historic shim's 8 gave
    # materially more drift-adaptation chances than CC gets on schema_drift steps.
    def __init__(self, *, fallback_action_fn, max_validation_retries: int = 1,
                 memory_note: str = ""):
        self._fallback_action_fn = fallback_action_fn
        self._max_retries = int(max_validation_retries)
        # CC-parity memory-note cadence: the claude system re-sends its memory instruction
        # with EVERY action-turn prompt (claude/system.py:558); ours previously appeared only
        # once in the task header. Non-empty -> folded into every reply's prose message at
        # CC's position (last block).
        self._memory_note = str(memory_note or "").strip()
        self._retries = 0
        self._lock = threading.Lock()
        self._epoch = 0
        self._action_q: "queue.Queue[tuple[int, BaseModel]]" = queue.Queue()
        self._reply_q: "queue.Queue[dict]" = queue.Queue()   # CURRENT step's mailbox
        # /observation snapshot state (mirrors clbench_step_shim payload keys exactly)
        self._prompt: Optional[str] = None
        self._schema_cls: Optional[type[BaseModel]] = None
        self._last_obs: Optional[str] = None
        self._instance_id: Optional[str] = None
        self._instance_index: Optional[int] = None
        self._metadata: dict = {}
        self._queries_used = 0
        self._budget: Optional[int] = None
        self._done = False
        self._server: Optional[ThreadingHTTPServer] = None
        self.url: Optional[str] = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> str:
        """Bind <CLBENCH_SHIM_BIND or 127.0.0.1>:<free port> and serve in a daemon thread.

        Operator port (campaign v6.56.0, re-confirmed on the official path 2026-07-28):
        on Linux `--add-host host.docker.internal:host-gateway` resolves to the docker
        BRIDGE gateway, which cannot reach a host listener bound to loopback — every
        submit_action then dies with connection-refused and the evaluator scores
        `num_queries: 0` with a null action (observed: 30/30 questions, $43 burned for
        nothing). Bind to the bridge IP (e.g. 172.17.0.1) for rootful-daemon runs."""
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence per-request stderr noise
                pass

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/healthz":
                    self._send(200, {"ok": True, "ready": bridge._prompt is not None})
                elif self.path == "/observation":
                    self._send(200, bridge.observation())
                else:
                    self._send(404, {"ok": False, "error": "not_found"})

            def do_POST(self):
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
                self._send(200, bridge._step(action))

        _bind = os.environ.get("CLBENCH_SHIM_BIND", "127.0.0.1")
        self._server = ThreadingHTTPServer((_bind, 0), _Handler)
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.url

    def shutdown(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

    # ------------------------------------------------------------ runner side
    def begin(self, query: Any) -> None:
        """Open a new scope: bump the epoch (stale in-flight actions become inert),
        install a fresh reply mailbox, and seed the observation state."""
        with self._lock:
            self._epoch += 1
            self._reply_q = queue.Queue()
            try:
                while True:
                    self._action_q.get_nowait()
            except queue.Empty:
                pass
            self._queries_used = 0
            self._apply_query_locked(query)
            self._last_obs = None
            self._done = False
            self._retries = 0

    def push(self, observation_content: Optional[str], next_query: Any, *,
             done: bool, hold_prompt: bool = False,
             instance_complete: Optional[bool] = None) -> None:
        """Publish a step result to the agent (delivered to the CURRENT step's mailbox).

        observation_content is relayed VERBATIM (parity guarantee: exactly what the
        official runner handed to observe()). ``done=True`` with ``hold_prompt`` mirrors
        the old shim's continuous-boundary shape: the reply keeps the OLD question's
        prompt and a null schema so the finishing agent wraps up.
        """
        with self._lock:
            # Shim-parity counting (#11): every successful runner step counts, including
            # the final ANSWER step; the counter resets on question change below.
            self._queries_used += 1
            self._last_obs = observation_content
            if next_query is not None and not hold_prompt:
                self._apply_query_locked(next_query)
            self._done = bool(done)
            # instance_complete is accepted for API stability but no longer relayed as a
            # structured key — CC's model infers the boundary from the verdict prose.
            del instance_complete
            payload = self._observation_locked()
            payload["ok"] = True
            self._reply_q.put(payload)

    def take_action(self, timeout: float) -> Optional[BaseModel]:
        """Block until the agent submits a valid action for the CURRENT epoch;
        None on timeout. Stale-epoch actions are discarded (backstop guard)."""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                epoch, action = self._action_q.get(timeout=remaining)
            except queue.Empty:
                return None
            with self._lock:
                if epoch == self._epoch:
                    return action
            _log(f"discarding stale action from epoch {epoch} (current {self._epoch})")

    def note_query(self, query: Any) -> None:
        """Refresh bookkeeping from the runner's next query without emitting a reply."""
        with self._lock:
            self._apply_query_locked(query)

    # ------------------------------------------------------------- agent side
    def observation(self) -> dict:
        with self._lock:
            return self._observation_locked()

    def _step(self, action_json: Any) -> dict:
        with self._lock:
            if self._done or self._schema_cls is None:
                return {"ok": False, "error": "task_done", "done": True}
            epoch = self._epoch
            schema = self._schema_cls
            try:
                action = schema.model_validate(action_json)
            except ValidationError as ve:
                self._retries += 1
                if self._retries > self._max_retries:
                    # Anti-no-op-loop parity with clbench_step_shim: force a real step
                    # with a zero-value action (burns a step, resets the counter).
                    action = self._fallback_action_fn(schema)
                else:
                    # CC's exact repair-turn shape (claude/system.py _build_repair_prompt):
                    # error + previous response + the full original prompt (which already
                    # embeds the schema instruction and memory note).
                    return {
                        "ok": False,
                        "error": "validation_error",
                        "message": "\n".join([
                            "Your previous response did not match the required response schema.",
                            f"Validation error: {ve}",
                            "Previous response:",
                            json.dumps(action_json, ensure_ascii=False),
                            "Return ONLY a corrected JSON object for the same task.",
                            "Original prompt:",
                            self._message_locked(),
                        ]),
                        "done": False,
                    }
            self._retries = 0
            # Install THIS step's one-shot mailbox and enqueue under the SAME lock
            # acquisition that captured the epoch: an orphaned handler holds a dead
            # queue object and can never steal this step's reply (#1, #3).
            my_q: "queue.Queue[dict]" = queue.Queue()
            self._reply_q = my_q
            self._action_q.put((epoch, action))
        try:
            return my_q.get(timeout=_STEP_REPLY_TIMEOUT_SEC)
        except queue.Empty:
            return {
                "ok": False,
                "error": "host_timeout",
                "hint": "The run host did not deliver the step result in time. Call "
                        "get_observation to see the current state; do NOT resubmit this action.",
                "done": False,
            }

    # ---------------------------------------------------------------- helpers
    def _apply_query_locked(self, query: Any) -> None:
        new_iid = getattr(query, "instance_id", None)
        if new_iid is not None and new_iid != self._instance_id:
            # Question changed (rollout boundary applies the next query with no begin()):
            # reset the local step counter — the analog of the shim's advance_question reset.
            self._queries_used = 0
        self._prompt = getattr(query, "prompt", None)
        self._schema_cls = getattr(query, "response_schema", None)
        self._instance_id = new_iid
        self._instance_index = getattr(query, "instance_index", None)
        meta = getattr(query, "metadata", None) or {}
        self._metadata = meta
        qu = meta.get("queries_used")
        if isinstance(qu, int):
            self._queries_used = qu       # authoritative override when the task provides it
        self._budget = None
        for key in ("budget", "max_steps", "remaining", "query_budget"):
            v = meta.get(key)
            if isinstance(v, int):
                self._budget = v
                break

    def _message_locked(self) -> str:
        """Render the CC-shaped per-turn message: the SAME components, order and joins as
        the claude system's _build_prompt (claude/system.py:550-564) — labeled feedback,
        the re-rendered question (its prose already carries the queries-used counter),
        the byte-identical schema instruction, then the memory note last. The earlier
        typed payload exposed the same data as JSON keys, which was arguably EASIER for
        the model than CC's prose (machine-readable schema, no parsing) — prose removes
        that last structural divergence. Task metadata (difficulty / question_id /
        db_path) is not rendered at all: CC's model receives none."""
        from . import run_clbench_bridge_agent as _rb  # lazy: avoid import cycles
        parts: list[str] = []
        if self._last_obs:
            parts.append(f"FEEDBACK FROM PREVIOUS ACTION: {self._last_obs}\n")
        if self._prompt and not self._done:
            parts.append(self._prompt)
            if self._schema_cls is not None:
                schema = json.dumps(self._schema_cls.model_json_schema(), indent=2)
                parts.append(f"{_rb._SCHEMA_PROMPT_HEAD}```json\n{schema}\n```{_rb._SCHEMA_PROMPT_TAIL}")
        if self._memory_note:
            parts.append("\n\n" + self._memory_note)
        return "\n".join(parts)

    def _observation_locked(self) -> dict:
        # {ok, done, message}: `done` is the push-protocol termination signal our agent
        # needs because IT drives the loop (CC's model is driven by the runner and never
        # needs one); everything informational rides in the CC-shaped prose message.
        return {"message": self._message_locked(), "done": self._done}


class LiveController:
    """Per-System orchestration: engine + bridge + one live Ouroboros task per scope.

    scope = "question": a task per CL-Bench question (skill-bridge mechanics, proven 0.608).
    scope = "rollout":  ONE task spans every question; after an ANSWER the agent's final
                        /step reply carries the verdict AND the next question (the genuine
                        growing-conversation analog of CC's single_conversation).
    """

    def __init__(self, *, engine, conversation: str, memory_mode: Optional[str],
                 fallback_action_fn, action_timeout_sec: float = 600.0,
                 task_timeout_sec: int = 900, discipline: str = "",
                 chunk_questions: int = 0, chunk_resume_mode: str = "continuation",
                 final_answer_delivery: bool = False, final_coerce_fn=None):
        if conversation not in ("question", "rollout"):
            raise ValueError(f"conversation must be question|rollout, got {conversation!r}")
        self.engine = engine
        self.conversation = conversation
        self.memory_mode = memory_mode
        self.discipline = discipline
        # Chunked rollout: end the live task every N question boundaries so POST-TASK
        # evolution gets a mid-benchmark promotion window (maybe_promote fires only on
        # completed tasks; an unchunked rollout offers exactly one, after Q40 — too late),
        # then stitch the next chunk to the previous task via conversation-resume so the
        # rollout's ICL continuity survives the seam. 0 = off (one task, the scored default).
        self.chunk_questions = int(chunk_questions or 0)
        self.chunk_resume_mode = chunk_resume_mode
        self._chunk_boundaries = 0
        self._resume_root: Optional[str] = None
        self._fallback = fallback_action_fn
        self.action_timeout_sec = float(action_timeout_sec)
        self.task_timeout_sec = int(task_timeout_sec)
        from . import run_clbench_bridge_agent as _rb  # lazy: avoid import cycles
        self.bridge = LiveBridge(fallback_action_fn=fallback_action_fn,
                                 memory_note=_rb._MEMORY_NOTES.get(memory_mode or "", ""))
        # Hybrid delivery (question scope): exploration rides submit_action as usual,
        # but the FINALIZING action arrives as the task's FINAL ANSWER text (CC-style),
        # parsed by the bridge. That routes the scored answer through task finalization,
        # so acceptance review (required/blocking) gates it BEFORE submission.
        self.final_delivery = bool(final_answer_delivery)
        self._final_coerce_fn = final_coerce_fn
        self._await_feedback = False
        self._last_delivered_blob: Optional[dict] = None
        # True once the CURRENT item saw at least one protocol step (submit_action
        # result observed). Scopes the echo-guard: a relayed-verdict echo is only
        # plausible BEFORE any real exploration on the item — after it, an identical
        # answer is a legitimate repeat (review finding: legit repeats were swallowed).
        self._item_had_steps = False
        # (kind, text): kind="step"  -> result of an action on the SAME, still-open item;
        #               kind="verdict" -> closing verdict of the PREVIOUS item (new item next).
        # One shared header for both poisoned qsh_code: the agent read its own
        # "echo COMPLETE" step output relayed as ITEM feedback and concluded the fresh
        # item was already done (7 empty-patch zeros).
        self._pending_feedback: Optional[tuple] = None
        self._task_id: Optional[str] = None
        self._published = False
        self._boundary_pending = False   # question scope: previous question just ended
        self._reopen_pending = False     # dead-task recovery: reopen scope in ANY conversation mode
        self._evolution_window_due = False  # set ONLY at legit boundaries, never on abandon
        self._turns = 0
        self._consec_fallbacks = 0
        self._fallback_instance: Optional[int] = None
        # Set by _fallback_or_terminal on forfeit escalation; read-and-cleared by
        # system.py into Response.metadata (the benchmark's latency_timeout convention).
        self.forfeit_metadata: Optional[dict] = None
        # Engine-task usage harvest (v6.81 submission port): CL-Bench UsageEvents otherwise
        # only see the tiny structured-output calls, under-reporting real cost by ~3 orders
        # of magnitude (colleague's official manifest: $0.54 vs ~$530 actual). Each terminal
        # engine task contributes one usage row; system.py drains this buffer into
        # record_usage_event. Dedup by task_id — a task may hit several terminal paths.
        self.usage_buffer: list[dict] = []
        self._usage_harvested: set[str] = set()
        self._usage_pending: dict[str, str] = {}

    def _harvest_task_usage(self, tid: Optional[str], status: str, *, block: bool = False,
                            budget_sec: float = 25.0) -> None:
        """Record one terminal engine task's cost into the usage buffer (idempotent).

        Review contract (codex adapter review, 2026-07-28):
        - authoritative cost = cost_usd_with_children (subtree rollup) when present,
          else cost_usd; a row is FINAL only when the record says so (cost_final=True
          or cost_accounting_status == "available");
        - the recorded status comes from the TASK RECORD, never from a synthetic
          caller label; a non-final read must never clobber a pending retry;
        - hot path (block=False) performs exactly ONE short-socket GET (<=3s), no
          polling, no sleeps (terminal fail-fast contract);
        - block=True obeys one wall-clock budget shared by every HTTP call inside;
          on budget exhaustion the last seen cost is recorded with cost_final=False
          rather than silently dropped."""
        if not tid or tid in self._usage_harvested:
            return
        from . import run_clbench_bridge_agent as _rb  # lazy: import cycle
        base = self.engine.server.base_url
        deadline = time.time() + (budget_sec if block else 0.0)
        rec: dict = {}
        while True:
            try:
                rec = _rb._api(base, "GET", "/api/tasks/" + tid, timeout=3) or {}
            except Exception as exc:  # noqa: BLE001 — harvest must never break the run
                _log(f"usage harvest read failed for {tid}: {type(exc).__name__}: {str(exc)[:120]}")
                rec = rec or {}
            cost_children = rec.get("cost_usd_with_children")
            cost_own = rec.get("cost_usd")
            cost = cost_children if cost_children is not None else cost_own
            final = bool(rec.get("cost_final")) or (
                str(rec.get("cost_accounting_status") or "") == "available")
            if final and cost is not None:
                break
            if not block or time.time() + 3.0 >= deadline:
                if not block:
                    # park for close()/finalize_usage() to finish with retries
                    self._usage_pending.setdefault(tid, str(rec.get("status") or status))
                    return
                break  # blocking budget exhausted: record what we have, marked non-final
            time.sleep(min(3.0, max(0.5, deadline - time.time())))
        self._usage_harvested.add(tid)
        self._usage_pending.pop(tid, None)
        self.usage_buffer.append({
            "task_id": tid,
            "status": str(rec.get("status") or status),
            "cost_usd": (rec.get("cost_usd_with_children")
                         if rec.get("cost_usd_with_children") is not None
                         else rec.get("cost_usd")),
            "cost_final": bool(rec.get("cost_final")) or (
                str(rec.get("cost_accounting_status") or "") == "available"),
        })

    def finalize_usage(self, budget_sec: float = 25.0) -> None:
        """Terminal-cost finalization before control returns to the runner (the runner
        never calls reset()/close() on the success path — review BLOCKER 1). One shared
        wall-clock budget across the current task and every parked pending tid."""
        deadline = time.time() + budget_sec
        if self._task_id:
            self._harvest_task_usage(self._task_id, self._task_status(), block=True,
                                     budget_sec=max(1.0, deadline - time.time()))
        for _tid, _st in list(self._usage_pending.items()):
            self._harvest_task_usage(_tid, _st, block=True,
                                     budget_sec=max(1.0, deadline - time.time()))

    # ------------------------------------------------------------- runner side
    def respond(self, query: Any):
        """Return the agent's next action for this query. NEVER raises."""
        try:
            action = self._respond_inner(query)   # real agent action, or None on timeout
        except Exception as exc:  # noqa: BLE001 — an escape aborts the entire run
            _log(f"respond failed: {type(exc).__name__}: {str(exc)[:200]}")
            action = None
        # Fallback + escalation live in ONE place so the consecutive-fallback counter
        # accumulates correctly on the timeout path too (a timeout returns None here, it
        # does NOT reset the counter — the reset only happens on a genuine agent action).
        if action is None:
            return self._fallback_or_terminal(query)
        self._consec_fallbacks = 0
        self.forfeit_metadata = None
        return action

    def observe(self, observation: Any, next_query: Any) -> None:
        """Relay the runner's observation verbatim to the agent. NEVER raises."""
        try:
            self._observe_inner(observation, next_query)
        except Exception as exc:  # noqa: BLE001
            _log(f"observe failed (continuing): {type(exc).__name__}: {str(exc)[:200]}")

    def close(self) -> None:
        try:
            self.finalize_usage()
        except Exception:
            pass
        try:
            self.bridge.shutdown()
        except Exception:
            pass

    def stats(self) -> dict:
        return {"live_turns": self._turns, "conversation": self.conversation,
                "task_id": self._task_id}

    # ---------------------------------------------------------------- internals
    def _respond_inner(self, query: Any):
        self._turns += 1
        if self.bridge.url is None:
            self.bridge.start()
        if self._task_id is None or self._reopen_pending or (
                self._boundary_pending and
                (self.conversation == "question" or self.chunk_questions)):
            # New scope. STRUCTURAL exclusivity guarantee (#1/#2/#5): the previous task
            # must be terminal before the new scope opens — a stale agent's /step is
            # otherwise indistinguishable from the new agent's.
            had_prev = self._task_id is not None
            if had_prev and self.chunk_questions:
                # Stitch root for the next chunk: the wrap-up task normally COMPLETES
                # (capture fires) — if it had to be cancelled, the capture is missing and
                # resume falls back to a fresh conversation (graceful degradation).
                self._resume_root = self._task_id
            # grace=180: give the wrap-up its natural completion window (final answer +
            # memory post-mortem rounds) so the resume capture survives; cancels only in
            # the last 60s of the bound.
            self._ensure_task_dead(bounded=240.0, grace=180.0)
            if had_prev and self._evolution_window_due:
                # Only a REAL chunk/question boundary earns an evolution window.
                # An abandon-recovery scope reopen must NOT open one: during e.g. a
                # network outage every failed task would otherwise file a forced
                # request -> cycle -> restart -> kill the retry task -> abandon ->
                # another window (self-sustaining restart storm, observed live).
                try:
                    # Mirrors the per-action path's boundary hook (#6): per-task budget
                    # reset (docker: best-effort no-op) + evolution absorb wait if enabled.
                    self.engine.on_instance_boundary()
                except Exception as exc:  # noqa: BLE001
                    _log(f"on_instance_boundary failed (continuing): {exc}")
            self._evolution_window_due = False
            self.bridge.begin(query)
            self._submit_task(query)
            self._boundary_pending = False
            self._reopen_pending = False
        else:
            self.bridge.note_query(query)

        action = self._take_action_patient(query)
        if action is None:
            self._abandon_task()
            return None   # respond() applies the fallback + accumulates the escalation counter
        return action

    def _take_action_patient(self, query: Any = None):
        """take_action with liveness-aware patience: fail FAST when the engine is provably
        dead (scope task never picked up / terminal without an action / logs silent), and
        extend PAST the base timeout — up to _ALIVE_PATIENCE_FACTOR x — while the engine
        demonstrably keeps working. Replaces the fixed wait that both burned the full
        timeout on dead containers AND forfeited questions under slow-but-alive agents."""
        base = self.action_timeout_sec
        hard = base * _ALIVE_PATIENCE_FACTOR
        start = time.time()
        outage_held = 0.0
        while True:
            if _NET_OUTAGE_HOLD_SEC > 0 and outage_held < _NET_OUTAGE_HOLD_SEC and not _net_ok():
                # Environmental stall: shift the scope clock so the outage does not
                # count against the question; log at a low cadence.
                if int(outage_held) % 300 == 0:
                    _log(f"host network DOWN — outage hold {outage_held:.0f}s/"
                         f"{_NET_OUTAGE_HOLD_SEC:.0f}s (task={self._task_id}); question clock paused")
                time.sleep(30)
                start += 30
                outage_held += 30
                continue
            elapsed = time.time() - start
            if elapsed >= hard:
                _log(f"no action after hard cap {hard:.0f}s (task={self._task_id}); "
                     f"abandoning scope (respond() will fall back/escalate)")
                return None
            # Terminal check BEFORE the probe block: a task that is already dead must
            # not burn a full _PROBE_CHUNK_SEC per turn (the code_reroll wedge ground
            # 60s/step for 80+ min on one dead rollout task). Short drain first — a
            # final action posted right before termination may still be in the queue.
            status = self._task_status()
            if status in ("completed", "failed", "cancelled", "rejected_duplicate"):
                self._harvest_task_usage(self._task_id, status)
                action = self.bridge.take_action(_TERMINAL_DRAIN_SEC)
                if action is not None:
                    return action
                # Salvage a typed FINAL ANSWER even from a FAILED/cancelled task: a task can
                # produce a valid final answer and only THEN die on a post-phase transient
                # (observed: 179 APIConnectionError in one db run flipped answered tasks to
                # 'failed' during the memory/review post-rounds — task 3bd8e178 had emitted
                # `FINAL ANSWER: {"content":"10.22"}` yet was abandoned to a zero). The typed
                # field survives on the task record regardless of terminal status; if it is
                # empty (failed before answering) _final_answer_action returns None and we
                # abandon as before, so this only ever recovers real, already-produced answers.
                if (self.final_delivery and self.conversation == "question"
                        and status in ("completed", "failed", "cancelled")
                        and query is not None):
                    final_action = self._final_answer_action(query)
                    if final_action is not None:
                        if status != "completed":
                            _log(f"task {self._task_id} terminal ({status}) but a typed FINAL "
                                 f"ANSWER survived — salvaged instead of abandoning")
                        return final_action
                _log(f"task {self._task_id} terminal ({status}) without a protocol action — "
                     f"abandoning scope")
                return None
            if status in ("queued", "pending") and elapsed >= _QUEUED_STALL_SEC:
                _log(f"task {self._task_id} still {status!r} after {elapsed:.0f}s — workers "
                     f"presumed dead; abandoning early (fail-fast)")
                return None
            action = self.bridge.take_action(min(_PROBE_CHUNK_SEC, hard - elapsed))
            if action is not None:
                return action
            elapsed = time.time() - start
            if elapsed < base:
                continue
            age = self._engine_activity_age()
            if age is not None and age < _ACTIVITY_STALL_SEC:
                _log(f"no action for {elapsed:.0f}s (base {base:.0f}s) but engine active "
                     f"{age:.0f}s ago (task={self._task_id}, status={status or '?'}) — extending wait")
                continue
            _log(f"no action within {elapsed:.0f}s and no recent engine activity "
                 f"({'unknown' if age is None else f'{age:.0f}s ago'}, task={self._task_id}); "
                 f"abandoning scope (respond() will fall back/escalate)")
            return None

    def _final_answer_action(self, query: Any):
        """CC-style delivery: parse the completed task's FINAL ANSWER into the query's
        action schema. Source order (re-emission fix): (1) the task record's typed
        `final_answer` field — the engine's marker-extracted or latch-recovered payload,
        which SURVIVES a post-review prose re-emission that rewrites the final text;
        (2) the raw final text (last JSON blob), the legacy path and fallback. Strict
        parse -> model_validate; unparseable finals fall to the coercion net / ordinary
        fallback path. On success the scope is consumed: the next respond() opens a fresh
        task for the next item, and the runner's verdict is relayed into that task's
        description (_pending_feedback)."""
        tid = self._task_id
        schema = getattr(query, "response_schema", None)
        if not tid or schema is None:
            return None
        try:
            res = self.engine.server.wait_task(tid, timeout=45)
            raw = str(res.get("result") or "")
        except Exception:
            return None
        typed = str(res.get("final_answer") or "").strip()
        if not raw.strip() and not typed:
            return None
        from .system import _extract_last_json  # lazy: avoid module cycle
        # Candidate ladder (adversarial-review finding: a bad typed blob must never
        # SHADOW a good text one): each source is tried strictly in order; a candidate
        # that fails JSON-shape, echo-guard, or schema falls through to the next
        # source, and only after both fail does the coercion net see the raw text.
        candidates = [("typed final_answer", _extract_last_json(typed))] if typed else []
        candidates.append(("final text", _extract_last_json(raw)))
        action = source = seen_blob = None
        echoed = failed = None
        for src, blob in candidates:
            if not isinstance(blob, dict):
                continue
            blob.pop("notes", None)
            if (self._last_delivered_blob is not None and blob == self._last_delivered_blob
                    and not self._item_had_steps):
                echoed = src
                continue
            try:
                action = schema.model_validate(blob)
            except Exception as exc:  # noqa: BLE001
                failed = f"{src}: {type(exc).__name__}"
                continue
            source, seen_blob = src, blob
            break
        if action is None:
            why = (f"echo of previous action ({echoed})" if echoed
                   else (f"schema {failed}" if failed else "no JSON"))
            coerced = self._coerce_final(raw, schema, tid, why)
            if coerced is not None:
                return coerced
            repaired = self._format_repair_action(query, schema, tid, why)
            if repaired is not None:
                return repaired
            _log(f"final-answer delivery: task {tid} no deliverable answer ({why}) — "
                 f"falling back")
            return None
        _log(f"final-answer delivery: task {tid} finalized item "
             f"q_index={getattr(query, 'instance_index', '?')} via {source}")
        self._boundary_pending = True   # scope consumed; next respond() opens fresh
        self._await_feedback = True     # runner's verdict gets relayed, not pushed to the dead task
        self._last_delivered_blob = dict(seen_blob)
        return action

    def _coerce_final(self, raw: str, schema: Any, tid: str, why: str):
        """Coercion safety net for the final-answer channel: the strict parse failed, but
        the final text may still CONTAIN the answer as prose (qsh_code lost 3 solved
        issues to perfect engineering REPORTS with no JSON). The System-supplied coerce
        callback (its standard general-purpose light-LLM plumbing — no task knowledge)
        recovers it; a coerce that returns a schema-empty action is treated as failure
        so the ordinary fallback/escalation semantics stay intact."""
        if self._final_coerce_fn is None:
            return None
        try:
            action = self._final_coerce_fn(raw, schema)
        except Exception:  # noqa: BLE001
            return None
        if action is None:
            return None
        try:
            dumped = action.model_dump()
        except Exception:  # noqa: BLE001
            return None
        # Emptiness that reasoning-chaff can't defeat: judge by PAYLOAD fields (a coerce
        # that filled only 'thought' but no command/content is a failed coerce).
        _CHAFF = {"thought", "reasoning", "rationale", "notes"}
        payload = {k: v for k, v in dumped.items() if k not in _CHAFF} or dumped
        if all(not str(v or "").strip() for v in payload.values()):
            return None
        # Echo-guard parity with the strict path (same exploration scoping): a coerced
        # action must neither repeat the previously delivered blob nor leave the guard
        # stale for the next strict parse.
        if (self._last_delivered_blob is not None and dumped == self._last_delivered_blob
                and not self._item_had_steps):
            _log(f"final-answer delivery: task {tid} coerce reproduced the PREVIOUS "
                 f"delivered action — falling back")
            return None
        _log(f"final-answer delivery: task {tid} strict parse failed ({why}) — "
             f"COERCED from final text")
        self._boundary_pending = True
        self._await_feedback = True
        self._last_delivered_blob = dict(dumped)
        return action

    def _format_repair_action(self, query: Any, schema: Any, tid: str, why: str):
        """Last-resort format repair: the task finished with the ANSWER DATA living only
        in the agent's own workspace/memory (observed: a sales task finalized with a
        prose report — '75 predictions submitted, saved to data/history.json' — while
        the numbers never reached the final message, so neither strict parse nor the
        text-only coercion net could recover them). Submit ONE tiny follow-up task to
        the SAME container (same data root, so the agent sees its own notes/files) that
        asks to re-emit strictly the JSON payload. Mirrors v6.71's own in-engine
        'second send is an extraction/format repair' idea at the bridge boundary.
        Env: OUROBOROS_FORMAT_REPAIR=0 disables; one attempt per question, never loops."""
        if os.environ.get("OUROBOROS_FORMAT_REPAIR", "1").strip() == "0":
            return None
        try:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
        except Exception:  # noqa: BLE001
            return None
        prompt = (
            "FORMAT REPAIR (mechanical, no new work): your previous task in this workspace "
            "produced an answer but finalized with a prose summary instead of the required "
            "JSON payload. Recover the ACTUAL answer data you already computed — check your "
            "memory notes, scratchpad and files you saved during that work — and re-emit it. "
            "Do NOT redo the analysis and do NOT invent values you cannot find. If the data "
            "is genuinely absent, emit exactly: FINAL ANSWER: {}\n\n"
            "Required schema:\n```json\n" + schema_json + "\n```\n\n"
            "Finalize with a single line: FINAL ANSWER: <one JSON object matching the schema>."
        )
        body = {
            "description": prompt,
            "memory_mode": self.engine.mm,
            "disabled_tools": _rb.DISABLED_TOOLS,
            "actor_id": "remote-driver",
            "source": "remote-driver",
            "timeout_sec": 420,
            "metadata": {"source": "remote-driver", "delegation_role": "root",
                         # a mechanical re-emission must not arm a review wave
                         "budget_profile": {"max_improvement_passes": 0}},
            "answer_protocol": "final_answer_line",
        }
        try:
            created = _rb._api(self.engine.server.base_url, "POST", "/api/tasks", body, timeout=60)
            rep_tid = str(created.get("task_id") or "")
            if not rep_tid:
                return None
            _log(f"format-repair: task {tid} ({why}) -> repair task {rep_tid} submitted")
            deadline = time.time() + 480
            res = None
            while time.time() < deadline:
                res = self.engine.server.wait_task(rep_tid, timeout=30)
                if str(res.get("status") or "") in ("completed", "failed", "cancelled",
                                                    "rejected_duplicate"):
                    break
                time.sleep(5)
            if not res:
                return None
        except Exception:  # noqa: BLE001
            _log(f"format-repair: task {tid} repair submission failed — giving up")
            return None
        from .system import _extract_last_json  # lazy: avoid module cycle
        for src_label, source_text in (("typed", str(res.get("final_answer") or "")),
                                       ("text", str(res.get("result") or ""))):
            blob = _extract_last_json(source_text)
            if not isinstance(blob, dict) or not blob:
                continue
            blob.pop("notes", None)
            try:
                action = schema.model_validate(blob)
            except Exception:  # noqa: BLE001
                continue
            # empty-payload guard (parity with coercion): repair that carries no data fails
            dumped = action.model_dump()
            _CHAFF = {"thought", "reasoning", "rationale", "notes"}
            payload = {k: v for k, v in dumped.items() if k not in _CHAFF} or dumped
            if all(not str(v or "").strip() for v in payload.values()):
                return None
            if (self._last_delivered_blob is not None and dumped == self._last_delivered_blob
                    and not self._item_had_steps):
                return None
            _log(f"format-repair: task {tid} REPAIRED via {src_label} of repair task")
            self._boundary_pending = True
            self._await_feedback = True
            self._last_delivered_blob = dict(dumped)
            return action
        _log(f"format-repair: task {tid} repair produced no usable payload — falling back")
        return None

    def _task_status(self) -> str:
        """Engine-side status of the live task ('' when unknowable — non-docker engines
        may not expose a one-shot status probe; '' disables the fail-fast branches)."""
        server = getattr(self.engine, "server", None)
        probe = getattr(server, "task_status", None)
        if not self._task_id or probe is None:
            return ""
        try:
            return str(probe(self._task_id) or "")
        except Exception:
            return ""

    def _engine_activity_age(self) -> Optional[float]:
        """Seconds since the engine last wrote any log line (host-side bind mount of the
        container's data root); None if unknowable. A single long LLM call writes nothing
        for its whole duration, so callers must use generous thresholds."""
        try:
            root = pathlib.Path(str(self.engine.data_root))
        except Exception:
            return None
        now = time.time()
        ages = []
        for rel in ("logs/events.jsonl", "logs/tools.jsonl", "logs/server.log"):
            try:
                ages.append(now - (root / rel).stat().st_mtime)
            except OSError:
                continue
        return min(ages) if ages else None

    def _observe_inner(self, observation: Any, next_query: Any) -> None:
        content = getattr(observation, "content", None)
        instance_complete = bool(getattr(observation, "instance_complete", False))
        seq_done = next_query is None
        # Consume-once: the flag belongs to the delivery that just happened; NEVER let it
        # leak into a later boundary (review: stale flag suppressed the wrap-up push and
        # wedged the live agent in /step for 280s).
        was_final_delivery = self._await_feedback
        self._await_feedback = False
        if not instance_complete:
            self._item_had_steps = True     # real protocol step on the CURRENT item
            if was_final_delivery:
                # The delivered "final" action did NOT finish the item (the note allows
                # misjudging finality). The task is already terminal — continue the SAME
                # question in a fresh scope, relaying the executed action's result.
                txt = str(content or "")
                self._pending_feedback = ("step", txt) if txt else None
                self._boundary_pending = True
                return
            # Ordinary step result: relay verbatim; next_query is the re-rendered question
            # (updated "Queries used: X/N" line) — the agent sees both, exactly like the
            # old shim's push model.
            self.bridge.push(content, next_query, done=False)
            return
        # Question boundary: content is the official verdict text.
        self._item_had_steps = False        # next item starts unexplored
        if self.conversation == "rollout" and not seq_done:
            self._chunk_boundaries += 1
            if self.chunk_questions and self._chunk_boundaries % self.chunk_questions == 0:
                # Chunk seam: let the agent wrap this task up (post-task evolution window
                # opens on completion); the runner's next respond() re-opens the scope with
                # the SAME next question and resume-stitches to this task.
                _log(f"chunk boundary after {self._chunk_boundaries} questions — ending live "
                     f"task {self._task_id} for a post-task evolution window")
                self.bridge.push(content, None, done=True, hold_prompt=True)
                self._boundary_pending = True
                self._evolution_window_due = True   # REAL chunk boundary: window earned
                return
            # Same live task continues: verdict + next question in one reply.
            self.bridge.push(content, next_query, done=False, instance_complete=True)
            return
        # question scope boundary, or end of the whole sequence: agent must wrap up.
        if was_final_delivery:
            # Final-answer delivery consumed this scope: the task is already terminal, a
            # wrap-up push has no reader. Relay the verdict into the NEXT item's task
            # description instead (verbatim observation relay).
            txt = str(content or "")
            self._pending_feedback = ("verdict", txt) if txt else None
            self._boundary_pending = True
            self._evolution_window_due = True
            if seq_done:
                self._ensure_task_dead(bounded=60.0)
            return
        self.bridge.push(content, None, done=True, hold_prompt=True)
        self._boundary_pending = True
        self._evolution_window_due = True   # legit boundary (question scope / sequence end)
        if seq_done:
            self._ensure_task_dead(bounded=120.0)

    def _submit_task(self, query: Any) -> None:
        from . import run_clbench_bridge_agent as _rb  # lazy: avoid import cycles

        self.engine._start()  # idempotent (guarded by engine._started)
        obs = {"prompt": getattr(query, "prompt", "") or "",
               "response_schema": (query.response_schema.model_json_schema()
                                   if getattr(query, "response_schema", None) is not None else None)}
        prompt = _rb._build_prompt(obs, memory_mode=self.memory_mode,
                                   steer_note=_rb.resolve_steer_note(self.discipline),
                                   action_note=(_rb._ACTION_NOTE_FINAL_DELIVERY
                                                if (self.final_delivery
                                                    and self.conversation == "question")
                                                else None))
        _fed_feedback = bool(self._pending_feedback)
        if self._pending_feedback:
            kind, txt = self._pending_feedback
            if kind == "step":
                hdr = ("RESULT OF YOUR PREVIOUS ACTION ON THIS ITEM — the item is NOT "
                       "finished; it continues below:")
            else:
                hdr = ("VERDICT FOR YOUR PREVIOUS ITEM — that item is CLOSED; the item "
                       "below is a NEW, separate item you have not started yet:")
            prompt = f"{hdr}\n{txt}\n\n{prompt}"
        if self.conversation == "rollout":
            # Domain-neutral (never names a domain-specific action — the schema in the prompt
            # defines the terminal one), BUT keeps the load-bearing affordance explicit:
            # you may take several actions per item and gather information step by step BEFORE
            # committing to the finalizing one (the earlier reword dropped this anchor and the
            # agent drifted toward answering blind — the live-bridge analog of the cont40 bug).
            prompt += (
                "\n\nThis is a CONTINUOUS session spanning multiple items. For the current item "
                "you may take several actions: use non-final actions to gather the information "
                "you need — each one's result is returned to you before you decide — and only "
                "then submit the action that finalizes the item. When you do, submit_action "
                "returns that item's outcome together with the next item, so keep working in the "
                "same session until the response reports done=true."
            )
        if not self._published:
            _rb._publish_target(self.engine.data_root, self._agent_url())
            self._published = True
        if self.conversation == "question":
            # final-answer delivery routes the SCORED answer through task finalization,
            # where required+blocking acceptance review (and one improvement pass) runs
            # BEFORE the answer is collectible — give the absolute deadline headroom so
            # a review never kills the task mid-finalization (review finding).
            timeout = (max(self.task_timeout_sec, 1800) if self.final_delivery
                       else self.task_timeout_sec)
        else:
            timeout = max(self.task_timeout_sec, 7200)
        # POST /api/tasks directly (NOT engine.server.submit) so the proven task body shape
        # is preserved (DISABLED_TOOLS is now empty = CC's unrestricted-toolset parity; the
        # key stays for protocol stability). Retry transient 503s (server re-exec windows).
        body = {
            "description": prompt,
            "memory_mode": self.engine.mm,
            "disabled_tools": _rb.DISABLED_TOOLS,
            "actor_id": "remote-driver",
            "source": "remote-driver",
            "timeout_sec": timeout,
            "metadata": {"source": "remote-driver", "delegation_role": "root"},
        }
        if (os.environ.get("OUROBOROS_TASK_REVIEW_MODE", "").strip() == "required"
                and os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT", "").strip() == "blocking"):
            # v6.64 made the required+blocking improvement loop count-UNBOUNDED (iterates
            # until the deadline reserve; 663 default was ONE bounded pass). Pin the 663
            # semantics explicitly: only a per-task budget_profile cap binds that lane on
            # 664+ (the env default is dead there), and pre-6.64 contracts ignore the key.
            # OUROBOROS_REVIEW_MAX_PASSES overrides for ablations: "unbounded" leaves the
            # upstream convergence loop uncapped (v6.71+ terminates by reviewer agreement);
            # an integer pins that count; unset keeps the validated 1-pass pin.
            _pin = os.environ.get("OUROBOROS_REVIEW_MAX_PASSES", "").strip() or "1"
            if _pin != "unbounded":
                try:
                    body["metadata"]["budget_profile"] = {"max_improvement_passes": int(_pin)}
                except ValueError:
                    body["metadata"]["budget_profile"] = {"max_improvement_passes": 1}
        if self.final_delivery and self.conversation == "question":
            # v6.60+ engines: declare the marker protocol as a task contract — the engine
            # itself injects the FINAL ANSWER doctrine into the task's runtime context and
            # arms the marker nudge on finalization (SYSTEM.md lost the global doctrine in
            # v6.60.0). Pre-6.60 gateways parse the body as a plain dict and ignore the
            # unknown key; there the doctrine still lives in SYSTEM.md + our action note.
            # Question-scope gate mirrors the note/consumption gates (review finding: a
            # direct-constructed rollout controller must not inject a no-consumer doctrine).
            body["answer_protocol"] = "final_answer_line"
        if self._resume_root and self.chunk_resume_mode != "none":
            # Chunk stitching: replay the prior chunk's conversation into this task.
            # continuation = verbatim frozen system (cache-hot, but an evolved SYSTEM.md /
            # identity stays INVISIBLE to later chunks); splice = FRESH system incl. evolved
            # prompts + updated identity/memory sections, prior turns spliced in (cache-cold
            # seam); none = no replay at all — cross-chunk continuity rides ONLY on native
            # memory (fresh system auto-injects the scratchpad), the question-scope story at
            # chunk granularity.
            body["resume_from_task_id"] = self._resume_root
            body["resume_mode"] = self.chunk_resume_mode
        # Pre-submit connectivity gate: never launch a question into a host-network
        # outage (the engine would stall silently and the scope timeout would zero it).
        _held = 0.0
        while _NET_OUTAGE_HOLD_SEC > 0 and _held < _NET_OUTAGE_HOLD_SEC and not _net_ok():
            if int(_held) % 300 == 0:
                _log(f"host network DOWN — holding submit {_held:.0f}s/{_NET_OUTAGE_HOLD_SEC:.0f}s")
            time.sleep(30)
            _held += 30
        created, last_exc = None, None
        for _attempt in range(6):
            try:
                created = _rb._api(self.engine.server.base_url, "POST", "/api/tasks", body, timeout=60)
                break
            except Exception as exc:  # noqa: BLE001 — transient 503/URLError during recovery
                last_exc = exc
                time.sleep(5)
        if created is None:
            raise RuntimeError(f"/api/tasks submit failed after retries: {last_exc}")
        self._task_id = str(created.get("task_id") or "") or None
        if self._task_id is None:
            raise RuntimeError(f"no task_id from /api/tasks: {created!r}")
        if _fed_feedback:
            self._pending_feedback = None   # consumed only once the task actually exists
        _log(f"submitted live task {self._task_id} (scope={self.conversation}, "
             f"timeout={timeout}s, q_index={getattr(query, 'instance_index', '?')})")

    def _agent_url(self) -> str:
        url = self.bridge.url or ""
        fn = getattr(self.engine, "shim_url_for_agent", None)
        return fn(url) if callable(fn) else url

    def _ensure_task_dead(self, *, bounded: float, grace: float = 0.0) -> None:
        """Cancel + wait until the previous task is TERMINAL (bounded). Blocking respond()
        is safe (the runner has no timeouts); an alive predecessor is the one thing the
        bridge cannot tolerate. If the bound expires, log loudly and proceed — the epoch
        guard limits the damage to in-flight requests.

        ``grace``: initial window where we wait for NATURAL completion without issuing
        any cancel. A wrap-up task that completes on its own writes its resume capture;
        a CANCELLED task never does (the worker loop dies before note_final_msg), so the
        next chunk restarts cold. The old fixed ~10s pre-cancel window silently cancelled
        every 8-question chunk's wrap-up (evo-pair post-mortem: all chunk tasks
        'cancelled', state/resume/ empty in BOTH lanes)."""
        tid = self._task_id
        if tid is None:
            return
        deadline = time.time() + bounded
        grace_end = time.time() + max(0.0, min(grace, bounded))
        terminal = False
        while time.time() < deadline:
            try:
                if time.time() >= grace_end:
                    self.engine.server.cancel_task(tid)   # re-issue: a single cancel can be lost
                res = self.engine.server.wait_task(tid, timeout=10)
                if str(res.get("status") or "") not in ("", "timeout"):
                    terminal = True
                    break
            except Exception:
                break
        if not terminal:
            try:
                self.engine.server.cancel_task(tid)
            except Exception:
                pass
            _log(f"previous task {tid} still not terminal after {bounded:.0f}s — proceeding "
                 f"(epoch guard active); stale steps may be rejected as task_done")
        self._harvest_task_usage(tid, "terminal" if terminal else "cancelled")
        self._task_id = None

    def _abandon_task(self) -> None:
        tid = self._task_id
        self._boundary_pending = True   # force a fresh scope on the next respond()
        # Dead-task recovery must work in EVERY conversation mode. Bare rollout
        # (no chunking) used to have NO reopen path at all: one degenerate agent
        # completion (e.g. an empty no-tool-call LLM round) mid-sequence left every
        # later respond() polling the same dead task — a 60s/step grind to the
        # watchdog with zero instances (code_reroll wedge, 2026-07-09).
        self._reopen_pending = True
        if tid is None:
            return
        if self.conversation == "rollout" and not self.chunk_questions:
            # Stitch the reopened scope to the dead task's capture when one exists
            # (natural completion writes it; a cancelled task degrades to a fresh
            # conversation server-side) — same mechanism as chunk seams.
            self._resume_root = tid
        try:
            self.engine.server.cancel_task(tid)
        except Exception:
            pass
        # keep _task_id set: the next respond()'s _ensure_task_dead confirms termination

    def _fallback_or_terminal(self, query: Any):
        """Zero-value action; escalates to a FORFEIT after _MAX_CONSECUTIVE_FALLBACKS on
        the same instance, so a dead engine forfeits the instance instead of livelocking
        the run (#4). DOMAIN-AGNOSTIC by construction (no hardcoded action names): the
        action is always the schema-derived valid-but-empty _fallback_action, and the
        forfeit signal rides on Response.metadata["latency_timeout"] — the benchmark's
        OWN zero-credit timed-out-answer convention (e.g. database_exploration
        task.py:325, blind_spectrum_monitoring task.py:714); tasks that don't read it
        simply score the empty action. system.py merges forfeit_metadata into the
        Response and clears it."""
        idx = getattr(query, "instance_index", None)
        if idx != self._fallback_instance:
            self._fallback_instance = idx
            self._consec_fallbacks = 0
        self._consec_fallbacks += 1
        schema = getattr(query, "response_schema", None)
        if self._consec_fallbacks >= _MAX_CONSECUTIVE_FALLBACKS:
            self.forfeit_metadata = {"latency_timeout": True,
                                     "forfeit_reason": "engine_unresponsive"}
        return self._fallback(schema)
