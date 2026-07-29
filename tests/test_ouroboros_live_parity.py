"""Parity + safety tests for the live-task bridge (run from the repo root:
``.venv/bin/python -m pytest src/systems/ouroboros/test_live_parity.py -q``).

1. The schema instruction our agent sees is BYTE-IDENTICAL to the one CC sees
   (src/systems/utils/structured_output.schema_to_prompt_instruction).
2. observe() content reaches the agent VERBATIM (no rewriting).
3. Validation semantics mirror clbench_step_shim (retry counter, force-fallback after max).
4. The runner-facing surface NEVER raises (an escaped exception aborts the whole run).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from src.systems.utils.structured_output import schema_to_prompt_instruction
from src.tasks.database_exploration.task import DatabaseAction
from src.systems.ouroboros import run_clbench_bridge_agent as rb
from src.systems.ouroboros._live_bridge import LiveBridge, LiveController
from src.systems.ouroboros.system import _fallback_action


def _q(prompt="Question 1/1\nHow many rows?", queries_used=0, idx=0):
    return SimpleNamespace(
        prompt=prompt, response_schema=DatabaseAction, instance_id=f"q{idx}",
        instance_index=idx,
        metadata={"queries_used": queries_used, "query_budget": 15, "question_num": idx + 1},
    )


# ---------------------------------------------------------------- 1. prompt parity
def test_schema_instruction_byte_identity():
    schema = DatabaseAction.model_json_schema()
    ours = f"{rb._SCHEMA_PROMPT_HEAD}```json\n{json.dumps(schema, indent=2)}\n```{rb._SCHEMA_PROMPT_TAIL}"
    cc = schema_to_prompt_instruction(DatabaseAction)
    assert ours == cc, "bridge schema instruction drifted from CC's schema_to_prompt_instruction"


def test_build_prompt_contains_question_verbatim():
    question = "Question 3/40\nWhat is the total?\nQueries used so far this question: 0/15"
    obs = {"prompt": question, "response_schema": DatabaseAction.model_json_schema()}
    prompt = rb._build_prompt(obs, memory_mode="tools")
    assert question in prompt
    assert schema_to_prompt_instruction(DatabaseAction) in prompt


# ---------------------------------------------------------- 2. verbatim passthrough
def test_observation_passthrough_verbatim():
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    bridge.begin(_q())
    content = "Query result (1/15 queries used, 14 remaining):\n\n weird ©†ext  \n"
    bridge.push(content, _q(queries_used=1), done=False)
    reply = bridge._reply_q.get(timeout=1)
    # CC delivers feedback as a labeled prose line before the question (system.py:554-555)
    assert reply["message"].startswith(f"FEEDBACK FROM PREVIOUS ACTION: {content}\n")
    assert content in bridge.observation()["message"]      # byte-for-byte inside the prose
    assert reply["ok"] is True and reply["done"] is False
    # nothing structural beyond transport + protocol termination (CC's model gets prose only)
    assert set(reply) == {"ok", "done", "message"}


def test_step_message_byte_matches_cc_prompt_render():
    # THE parity pin: our per-step message must be byte-identical to the claude
    # system's _build_prompt composition (parts joined with "\n"; schema instruction
    # via the shared schema_to_prompt_instruction; memory note last) — modulo the
    # note text pointing at Ouroboros-native memory tools instead of MEMORY.md.
    q = _q(prompt="Question 2/40\nHow many rows?\nQueries used so far this question: 0/15")
    bridge = LiveBridge(fallback_action_fn=_fallback_action,
                        memory_note=rb._MEMORY_NOTES["tools"])
    bridge.begin(q)
    content = "Query result (1/15 queries used, 14 remaining):\n\n42"
    bridge.push(content, q, done=False)
    reply = bridge._reply_q.get(timeout=1)
    expected = "\n".join([
        f"FEEDBACK FROM PREVIOUS ACTION: {content}\n",
        q.prompt,
        schema_to_prompt_instruction(DatabaseAction),
        "\n\n" + rb._MEMORY_NOTES["tools"].strip(),
    ])
    assert reply["message"] == expected


def test_boundary_hold_prompt_shape():
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    bridge.begin(_q())
    verdict = "Question 1: INCORRECT.\nYour answer: 5\nCorrect answer: 8\nExploratory queries used: 2"
    bridge.push(verdict, None, done=True, hold_prompt=True)
    reply = bridge._reply_q.get(timeout=1)
    assert reply["done"] is True
    # wrap-up reply: labeled verdict only — no re-rendered question, no schema
    assert reply["message"] == f"FEEDBACK FROM PREVIOUS ACTION: {verdict}\n"
    assert "You MUST respond" not in reply["message"]


# ------------------------------------------------------- 3. validation semantics
def test_step_validation_retries_then_fallback():
    bridge = LiveBridge(fallback_action_fn=_fallback_action, max_validation_retries=3)
    bridge.begin(_q())
    for _ in range(3):
        r = bridge._step({"action": "NOPE", "bogus": 1})
        assert r["ok"] is False and r["error"] == "validation_error"
    # 4th consecutive invalid -> forced fallback burns a real step (mirrors shim)
    got = {}

    def run_step():
        got["reply"] = bridge._step({"action": "NOPE"})

    t = threading.Thread(target=run_step)
    t.start()
    action = bridge.take_action(timeout=5)
    assert action is not None and isinstance(action, DatabaseAction)
    bridge.push("result", _q(queries_used=1), done=False)
    t.join(timeout=5)
    assert got["reply"]["ok"] is True


def test_step_accepts_bare_action_and_valid_action():
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    bridge.begin(_q())
    result = {}

    def run_step():
        result["reply"] = bridge._step({"action": "QUERY", "content": "SELECT 1"})

    t = threading.Thread(target=run_step)
    t.start()
    action = bridge.take_action(timeout=5)
    assert isinstance(action, DatabaseAction) and action.action == "QUERY"
    bridge.push("Query result ...", _q(queries_used=1), done=False)
    t.join(timeout=5)
    assert result["reply"]["ok"] is True


def test_step_after_done_reports_task_done():
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    bridge.begin(_q())
    bridge.push("verdict", None, done=True, hold_prompt=True)
    bridge._reply_q.get(timeout=1)
    r = bridge._step({"action": "QUERY", "content": "SELECT 1"})
    assert r == {"ok": False, "error": "task_done", "done": True}


# ------------------------------------------------------------- 4. never raises
class _BrokenEngine:
    mm = "shared"
    data_root = "/nonexistent"

    def _start(self):
        raise RuntimeError("engine exploded")

    server = None


def test_controller_respond_never_raises():
    ctl = LiveController(engine=_BrokenEngine(), conversation="question",
                         memory_mode="tools", fallback_action_fn=_fallback_action,
                         action_timeout_sec=2, task_timeout_sec=5)
    action = ctl.respond(_q())
    assert isinstance(action, DatabaseAction)      # degraded to fallback, no exception
    ctl.observe(SimpleNamespace(content="x", instance_complete=False), _q(queries_used=1))
    ctl.close()


def test_controller_action_timeout_returns_fallback():
    class _NoopServer:
        base_url = "http://127.0.0.1:1"   # never contacted (submit is monkeypatched)

        def wait_task(self, *a, **k):
            return {"status": "cancelled"}

        def cancel_task(self, *a, **k):
            pass

    class _IdleEngine:
        mm = "shared"
        data_root = "/tmp"
        server = _NoopServer()

        def _start(self):
            pass

        def on_instance_boundary(self):
            return {}

    ctl = LiveController(engine=_IdleEngine(), conversation="question",
                         memory_mode=None, fallback_action_fn=_fallback_action,
                         action_timeout_sec=0.5, task_timeout_sec=5)
    ctl._submit_task = lambda query: setattr(ctl, "_task_id", "tid-1")  # skip real HTTP submit
    t0 = time.time()
    action = ctl.respond(_q())
    assert isinstance(action, DatabaseAction)
    assert time.time() - t0 < 30
    # abandoned task forces a fresh scope; _ensure_task_dead confirms termination fast
    action2 = ctl.respond(_q(idx=1))
    assert isinstance(action2, DatabaseAction)
    ctl.close()


def test_repeated_timeouts_escalate_to_terminal_no_livelock():
    """Regression for the counter-reset livelock: repeated take_action timeouts on the
    SAME instance must ESCALATE to a FORFEIT after N (domain-generic: schema-derived
    empty action + latency_timeout metadata), not reset the counter each time (which
    livelocked db_live40's last baseline instance)."""
    class _NoopServer:
        base_url = "http://127.0.0.1:1"

        def wait_task(self, *a, **k):
            return {"status": "cancelled"}

        def cancel_task(self, *a, **k):
            pass

    class _WedgedEngine:
        mm = "shared"
        data_root = "/tmp"
        server = _NoopServer()

        def _start(self):
            pass

        def on_instance_boundary(self):
            return {}

    ctl = LiveController(engine=_WedgedEngine(), conversation="question", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=0.2, task_timeout_sec=1)  # agent never acts -> timeout
    ctl._submit_task = lambda query: setattr(ctl, "_task_id", "tid-x")
    q = _q(idx=7)
    a1 = ctl.respond(q)
    assert ctl.forfeit_metadata is None                 # not yet escalated
    a2 = ctl.respond(q); a3 = ctl.respond(q)
    assert all(isinstance(a, DatabaseAction) for a in (a1, a2, a3))
    # 3rd consecutive -> DOMAIN-GENERIC forfeit: schema-derived empty action + the
    # benchmark's own latency_timeout metadata convention (no hardcoded action names)
    assert ctl.forfeit_metadata and ctl.forfeit_metadata.get("latency_timeout") is True
    ctl.close()


def test_stale_epoch_action_discarded():
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    bridge.begin(_q())
    stale = DatabaseAction.model_validate({"action": "ANSWER", "content": "stale"})
    fresh = DatabaseAction.model_validate({"action": "QUERY", "content": "SELECT 1"})
    bridge._action_q.put((bridge._epoch - 1, stale))   # from a previous scope
    bridge._action_q.put((bridge._epoch, fresh))
    got = bridge.take_action(timeout=1)
    assert got is fresh                                  # stale one silently dropped


def test_consecutive_fallbacks_escalate_to_forfeit_metadata():
    class _Dead:
        mm = "shared"
        data_root = "/nonexistent"
        server = None

        def _start(self):
            raise RuntimeError("dead engine")

    ctl = LiveController(engine=_Dead(), conversation="question", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    q = _q()
    a1 = ctl.respond(q)
    a2 = ctl.respond(q)
    a3 = ctl.respond(q)                                  # 3rd consecutive -> forfeit
    assert isinstance(a1, DatabaseAction) and isinstance(a3, DatabaseAction)
    assert ctl.forfeit_metadata and ctl.forfeit_metadata.get("latency_timeout") is True
    ctl.close()


# ------------------------------------------------------ 6. memory-note CC cadence
def test_memory_note_cc_cadence_in_every_payload():
    # CC re-sends its memory instruction with EVERY action-turn prompt
    # (claude/system.py:558-560, appended last); the bridge folds the note into
    # every reply's prose message at the same position.
    note = rb._MEMORY_NOTES["tools"].strip()
    noted = LiveBridge(fallback_action_fn=_fallback_action,
                       memory_note=rb._MEMORY_NOTES["tools"])
    noted.begin(_q())
    assert noted.observation()["message"].endswith("\n\n" + note)
    noted.push("Query result (1/15 queries used)", _q(queries_used=1), done=False)
    reply = noted._reply_q.get(timeout=1)
    assert reply["message"].endswith("\n\n" + note)
    # CC's repair prompt embeds the ORIGINAL prompt (note included) — ours matches
    err = noted._step({"action": "BOGUS", "content": 1})
    assert err["error"] == "validation_error" and note in err["message"]
    # and with no memory_mode the note is absent everywhere (A/B toggle intact)
    bare = LiveBridge(fallback_action_fn=_fallback_action)
    bare.begin(_q())
    assert "learn and store" not in bare.observation()["message"]


# --------------------------------------------------- 7. CC-comparability clamps
def test_no_structured_task_data_reaches_agent():
    # CC's model receives ZERO structured task metadata (claude/system.py:550-564):
    # difficulty/question_id/db_path must never reach the agent in any form.
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    q = _q()
    q.metadata = {"queries_used": 2, "query_budget": 15, "difficulty": "hard",
                  "question_id": "q_007", "db_path": "/tmp/secret.db"}
    bridge.begin(q)
    obs = bridge.observation()
    assert set(obs) == {"message", "done"}
    for leak in ("difficulty", "q_007", "secret.db", "instance_id", "metadata"):
        assert leak not in obs["message"]
    assert q.prompt in obs["message"]                      # the question itself does arrive


def test_validation_retry_default_matches_cc_repair_budget():
    # CC gets initial attempt + ONE repair turn; our default must match (was 8),
    # and the repair reply mirrors CC's _build_repair_prompt (error + previous
    # response + the full original prompt).
    bridge = LiveBridge(fallback_action_fn=_fallback_action)
    assert bridge._max_retries == 1
    bridge.begin(_q())
    r1 = bridge._step({"action": "BOGUS", "content": 1})
    assert r1["error"] == "validation_error"              # the single repair chance
    assert r1["message"].startswith("Your previous response did not match")
    assert "Original prompt:" in r1["message"] and _q().prompt in r1["message"]
    # 2nd invalid attempt -> forced schema-derived fallback burns a real step
    def _drain():
        act = bridge.take_action(timeout=5)
        bridge.push("burned", _q(queries_used=1), done=False)
        assert isinstance(act, DatabaseAction)
    t = threading.Thread(target=_drain, daemon=True); t.start()
    r2 = bridge._step({"action": "BOGUS", "content": 1})
    t.join(timeout=5)
    assert r2.get("ok") is True                           # fallback stepped, not retried


# ------------------------------------------------- 7b. discipline steer (opt-in)
def test_discipline_steer_off_by_default_and_resolvable():
    # default OFF: the validated CC-parity prompt must stay byte-stable
    assert rb.resolve_steer_note("") == ""
    assert rb.resolve_steer_note(None) == ""
    # built-in key resolves to the unit-1 note; unknown text passes through literally
    note = rb.resolve_steer_note("unit1")
    assert note.startswith("\n\nOperating discipline")
    assert "re-inspect the live metadata" in note and "arbitrate" in note
    assert rb.resolve_steer_note("custom steer line") == "\n\ncustom steer line"
    # zero task knowledge: no domain nouns may creep into the built-in note
    for banned in ("SQL", "poker", "cohort", "prc", "schema_drift", "tablib", "pytest"):
        assert banned not in note, f"domain noun {banned!r} leaked into the generic steer"
    # controller pass-through
    class _Idle:
        mm = "shared"; data_root = "/nonexistent"; server = None
    ctl = LiveController(engine=_Idle(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action, discipline="unit1",
                         action_timeout_sec=1, task_timeout_sec=5)
    assert ctl.discipline == "unit1"
    ctl.close()
    # POSITION pin: the steer must sit BEFORE the action note — the how-to-act mechanics
    # stay the prompt tail. The first placement (steer appended last) displaced the
    # tool-invocation instruction and the agent narrated actions as text instead of
    # calling submit_action (33/30/43 completed-without-action abandons).
    p = rb._build_prompt({"prompt": "Q1", "response_schema": DatabaseAction.model_json_schema()},
                         memory_mode="tools", steer_note=rb.resolve_steer_note("unit1"))
    assert p.index("Operating discipline") < p.index("Submit each action")
    assert p.rstrip().endswith("until it reports done.")


# ------------------------------------------------- 7c. chunked rollout (evolution windows)
def test_chunked_rollout_seam_marks_boundary_and_stitches_resume():
    # chunk_questions=2 in rollout scope: after the 2nd question boundary the live task is
    # wrapped up (done=True, prompt held) and the next respond() must open a NEW scope whose
    # task body resume-stitches to the previous task id.
    class _Idle:
        mm = "shared"; data_root = "/nonexistent"; server = None
    ctl = LiveController(engine=_Idle(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action, chunk_questions=2,
                         action_timeout_sec=1, task_timeout_sec=5)
    ctl._task_id = "task-chunk-1"           # pretend the first chunk task is live
    ctl.bridge.begin(_q(idx=0))
    # boundary 1 of 2: mid-chunk — fused verdict+next reply, task continues
    ctl._observe_inner(SimpleNamespace(content="Question 1: CORRECT", instance_complete=True),
                       _q(idx=1))
    r1 = ctl.bridge._reply_q.get(timeout=1)
    assert r1["done"] is False and ctl._boundary_pending is False
    # boundary 2 of 2: chunk seam — wrap-up reply, boundary flagged
    ctl._observe_inner(SimpleNamespace(content="Question 2: CORRECT", instance_complete=True),
                       _q(idx=2))
    r2 = ctl.bridge._reply_q.get(timeout=1)
    assert r2["done"] is True
    assert ctl._boundary_pending is True
    ctl.close()


def test_chunk_resume_fields_enter_task_body_and_default_off():
    class _Idle:
        mm = "shared"; data_root = "/nonexistent"; server = None
    # default: chunking off — rollout boundary never wraps up mid-sequence
    off = LiveController(engine=_Idle(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    off._task_id = "t0"; off.bridge.begin(_q(idx=0))
    for i in range(1, 6):
        off._observe_inner(SimpleNamespace(content=f"Question {i}: CORRECT",
                                           instance_complete=True), _q(idx=i))
        assert off.bridge._reply_q.get(timeout=1)["done"] is False
    assert off._boundary_pending is False
    off.close()
def test_resume_verdict_feedback_carries_across_boundary_once():
    # CC carries the finished question's verdict as FEEDBACK into the NEXT question's
    # first prompt (claude/system.py:843-857) and shows each feedback exactly once
    # (cleared after a successful respond, :775-777). Our per-action resume path used
    # to wipe it at the instance boundary — the agent never saw CORRECT/INCORRECT.
    from src.systems.ouroboros.system import OuroborosSystem

    prompts: list[str] = []

    class _StubEngine:
        def run_turn(self, prompt):
            prompts.append(prompt)
            return '{"action": "ANSWER", "content": "42"}'

        def on_instance_boundary(self):
            return {}

    # docker=True only satisfies the ctor's resume fail-fast; the engine is lazy and the
    # stub below replaces it before any respond, so no container is ever started.
    s = OuroborosSystem(engine="ouroboros", mode="stateful", loop="action",
                        resume=True, docker=True)
    s._engine = _StubEngine()

    q1 = SimpleNamespace(prompt="Question 1/2\nHow many rows?", response_schema=DatabaseAction,
                         instance_id="i1", instance_index=0, metadata={})
    q2 = SimpleNamespace(prompt="Question 2/2\nWhat is the max?", response_schema=DatabaseAction,
                         instance_id="i2", instance_index=1, metadata={})
    s.respond(q1)
    verdict = "Question 1: INCORRECT.\nYour answer: 42\nCorrect answer: 58\nExploratory queries used: 1"
    s.observe(SimpleNamespace(content=verdict, instance_complete=True), q2)
    s.respond(q2)
    # the verdict crossed the boundary into Q2's first prompt (CC behavior)
    assert prompts[-1].startswith(f"FEEDBACK FROM PREVIOUS ACTION: {verdict}")
    assert "Question 2/2" in prompts[-1]
    # ...and is shown exactly once: a further respond without a new observation is clean
    s.respond(q2)
    assert "INCORRECT" not in prompts[-1]


def test_memory_note_resolved_from_controller_memory_mode():
    class _Idle:
        mm = "shared"
        data_root = "/nonexistent"
        server = None

    ctl = LiveController(engine=_Idle(), conversation="rollout", memory_mode="tools",
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    assert ctl.bridge._memory_note == rb._MEMORY_NOTES["tools"].strip()
    off = LiveController(engine=_Idle(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    assert off.bridge._memory_note == ""
    ctl.close(); off.close()


def test_forced_evolution_request_writer(tmp_path):
    """evolution_trigger=forced: the bridge files the durable promotion request with the
    engine-native schema; idempotence guards skip on pending request / in-flight campaign."""
    from src.systems.ouroboros._docker_launcher import (
        _FORCED_EVOLUTION_OBJECTIVE, _write_forced_evolution_request,
    )
    (tmp_path / "state").mkdir()
    # 1. fresh state → writes; steer empty → default generic objective
    assert _write_forced_evolution_request(tmp_path, "") is True
    req = json.loads((tmp_path / "state" / "post_task_evolution_request.json").read_text())
    assert req["objective"] == _FORCED_EVOLUTION_OBJECTIVE
    assert req["requires_plan_review"] is True
    assert req["source"] == "clbench_bridge_forced_window"
    # 2. pending request → skip (never clobber an unconsumed signal)
    assert _write_forced_evolution_request(tmp_path, "custom") is False
    # 3. consumed request but campaign in flight → skip
    (tmp_path / "state" / "post_task_evolution_request.json").unlink()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"evolution_mode_enabled": True}))
    assert _write_forced_evolution_request(tmp_path, "custom") is False
    # 4. campaign done → writes the steer text as the objective
    (tmp_path / "state" / "state.json").write_text(json.dumps({"evolution_mode_enabled": False}))
    assert _write_forced_evolution_request(tmp_path, "custom objective") is True
    req = json.loads((tmp_path / "state" / "post_task_evolution_request.json").read_text())
    assert req["objective"] == "custom objective"


def test_abandon_reopen_does_not_open_evolution_window():
    """Abandon-recovery scope reopen must NOT call on_instance_boundary (forced
    evolution window): during a provider outage every failed task would file a
    request -> cycle -> restart -> kill the retry -> another abandon (observed
    restart storm). Only a legit chunk/question boundary earns the window."""
    calls = []

    class _SpyEngine:
        mm = "shared"
        def __init__(self):
            self.server = self
        def cancel_task(self, tid):
            return {}
        def wait_task(self, tid, timeout=10):
            return {"status": "cancelled"}
        def on_instance_boundary(self):
            calls.append(1)
            return {}

    ctl = LiveController(engine=_SpyEngine(), conversation="rollout", memory_mode=None,
                         chunk_questions=8, chunk_resume_mode="splice",
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    ctl._task_id = "dead-task"
    ctl._abandon_task()             # abandon: boundary_pending=True, window NOT due
    assert ctl._boundary_pending is True
    assert ctl._evolution_window_due is False, "abandon must not earn an evolution window"
    # legit chunk boundary DOES arm it
    ctl._evolution_window_due = True
    assert ctl._evolution_window_due is True
    ctl.close()


def test_dead_rollout_task_reopens_scope_with_resume_stitch():
    """Regression for the code_reroll wedge (2026-07-09): in BARE rollout (no chunking)
    a live task that goes terminal WITHOUT a protocol action (e.g. a degenerate empty
    no-tool-call LLM round completes the whole-sequence task) must (1) be abandoned
    FAST — terminal status is probed BEFORE the 60s take_action block — and (2) the
    NEXT respond() must reopen a fresh scope resume-stitched to the dead task, instead
    of polling the dead task forever (the 60s/step grind to the watchdog, 0 instances)."""
    class _DeadTaskServer:
        base_url = "http://127.0.0.1:1"   # never contacted (submit is monkeypatched)

        def task_status(self, tid):
            return "completed"            # terminal from the bridge's point of view

        def wait_task(self, *a, **k):
            return {"status": "completed"}

        def cancel_task(self, *a, **k):
            pass

    class _Engine:
        mm = "shared"
        data_root = "/tmp"
        server = _DeadTaskServer()

        def _start(self):
            pass

        def on_instance_boundary(self):
            return {}

    ctl = LiveController(engine=_Engine(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         # generous timeouts: the terminal fail-fast must beat them
                         action_timeout_sec=30, task_timeout_sec=60)
    submits = []

    def _fake_submit(query):
        # record the stitch root AT SUBMIT TIME (a later abandon overwrites it)
        submits.append((getattr(query, "instance_index", None), ctl._resume_root))
        ctl._task_id = f"tid-{len(submits)}"

    ctl._submit_task = _fake_submit

    t0 = time.time()
    a1 = ctl.respond(_q())                      # scope 1: task dies without an action
    assert isinstance(a1, DatabaseAction)       # degraded to fallback, no exception
    assert time.time() - t0 < 15, "terminal task must fail fast, not burn the 60s probe"

    a2 = ctl.respond(_q(queries_used=1))        # recovery: fresh scope on the SAME question
    assert isinstance(a2, DatabaseAction)
    assert len(submits) == 2, "dead bare-rollout task must reopen a fresh scope"
    assert submits[1][1] == "tid-1", "reopened scope must stitch to the dead task"
    ctl.close()


def test_submit_task_stitches_resume_root_without_chunking():
    """The reopen stitch must reach the task body in BARE rollout: resume_from_task_id
    rides on _resume_root alone (the old guard also required chunk_questions and
    silently dropped the stitch for unchunked rollouts)."""
    class _Server:
        base_url = "http://127.0.0.1:1"

    class _Engine:
        mm = "shared"
        data_root = "/tmp"
        server = _Server()

        def _start(self):
            pass

    ctl = LiveController(engine=_Engine(), conversation="rollout", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=1, task_timeout_sec=5)
    ctl._published = True                       # skip skill-state publish (needs a data root)
    ctl._resume_root = "dead-task"
    bodies = []
    real_api = rb._api
    rb._api = lambda base, method, path, body=None, timeout=60: (
        bodies.append(body) or {"task_id": "tid-2"})
    try:
        ctl._submit_task(_q(queries_used=1))
    finally:
        rb._api = real_api
    assert bodies and bodies[0].get("resume_from_task_id") == "dead-task"
    assert bodies[0].get("resume_mode") == "continuation"


def _final_delivery_controller(final_result: str, final_answer: str = "",
                               terminal_status: str = "completed"):
    """Controller with a fake engine whose live task is already terminal (completed)
    with ``final_result`` as its final answer text — the hybrid delivery scenario.
    ``final_answer`` fakes the engine's typed marker-latch field (re-emission fix).
    ``terminal_status`` fakes a task that died AFTER answering (failed/cancelled) —
    the network-blip salvage path."""
    class _Server:
        base_url = "http://127.0.0.1:1"

        def task_status(self, tid):
            return terminal_status

        def wait_task(self, *a, **k):
            return {"status": terminal_status, "result": final_result,
                    "final_answer": final_answer}

        def cancel_task(self, *a, **k):
            pass

    class _Engine:
        mm = "shared"
        data_root = "/tmp"
        server = _Server()

        def _start(self):
            pass

        def on_instance_boundary(self):
            return {}

    ctl = LiveController(engine=_Engine(), conversation="question", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=30, task_timeout_sec=60,
                         final_answer_delivery=True)
    submits = []

    def _fake_submit(query):
        submits.append(getattr(query, "instance_index", None))
        ctl._task_id = f"tid-{len(submits)}"

    ctl._submit_task = _fake_submit
    return ctl, submits


def test_final_answer_delivery_parses_terminal_result():
    """Hybrid channel: a completed task whose final text carries a schema-valid JSON
    delivers THAT action (not the fallback), consumes the scope, and relays the
    runner's verdict into the NEXT item's task description."""
    ctl, submits = _final_delivery_controller(
        'Analysis done.\n\n{"action": "ANSWER", "content": "42"}')
    t0 = time.time()
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert a1.action.value == "ANSWER" if hasattr(a1.action, "value") else str(a1.action) == "ANSWER"
    assert a1.content == "42"                      # the agent's answer, not a fallback
    assert time.time() - t0 < 20
    # verdict relay: observe() stashes; the REAL _submit_task prepends the header
    ctl.observe(SimpleNamespace(content="Correct! Ground truth: 42",
                                instance_complete=True, metadata={}), _q(idx=1))
    assert ctl._pending_feedback and ctl._pending_feedback[0] == "verdict"
    assert "Ground truth" in ctl._pending_feedback[1]
    bodies = []
    real_api = rb._api
    rb._api = lambda base, method, path, body=None, timeout=60: (
        bodies.append(body) or {"task_id": "tid-next"})
    ctl._published = True
    try:
        # restore the REAL _submit_task (the fake consumed by _final_delivery_controller)
        ctl._submit_task = type(ctl)._submit_task.__get__(ctl)
        ctl._submit_task(_q(idx=1))
    finally:
        rb._api = real_api
    assert bodies and bodies[0]["description"].startswith("VERDICT FOR YOUR PREVIOUS ITEM")
    assert "Ground truth: 42" in bodies[0]["description"]
    assert ctl._pending_feedback is None            # consumed after successful POST
    ctl.close()


def test_final_answer_delivery_garbage_falls_back():
    """Unparseable final text must NOT crash the runner: ordinary fallback action."""
    ctl, submits = _final_delivery_controller("I think I finished but here is prose only.")
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)          # schema-empty fallback
    assert (a1.content or "") == ""                # not an invented answer
    ctl.close()


def test_final_answer_delivery_off_by_default():
    """Flag off -> old behavior: terminal-without-action is a fallback even when the
    final text WOULD parse (no silent behavior change for existing configs)."""
    class _Server:
        base_url = "http://127.0.0.1:1"

        def task_status(self, tid):
            return "completed"

        def wait_task(self, *a, **k):
            return {"status": "completed", "result": '{"action": "ANSWER", "content": "42"}'}

        def cancel_task(self, *a, **k):
            pass

    class _Engine:
        mm = "shared"
        data_root = "/tmp"
        server = _Server()

        def _start(self):
            pass

    ctl = LiveController(engine=_Engine(), conversation="question", memory_mode=None,
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=30, task_timeout_sec=60)
    ctl._submit_task = lambda q: setattr(ctl, "_task_id", "tid-1")
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert (a1.content or "") == ""                # fallback, NOT the parseable "42"
    ctl.close()


def test_final_delivery_action_note_swaps_in_prompt():
    """final_answer_delivery=True must swap the action-note tail; default note otherwise."""
    obs = {"prompt": "Q?", "response_schema": None}
    p_default = rb._build_prompt(obs)
    p_final = rb._build_prompt(obs, action_note=rb._ACTION_NOTE_FINAL_DELIVERY)
    assert "Continue until it reports done." in p_default
    assert "FINAL ANSWER:" in p_final               # native marker protocol (re-emission fix)
    assert "finalizes the item" in p_final



def test_final_answer_delivery_requires_question_scope():
    """Constructor fail-fast: the hybrid channel is meaningless (and was proven harmful)
    in rollout scope — one task spans all questions."""
    from src.systems.ouroboros.system import OuroborosSystem
    try:
        OuroborosSystem(conversation="rollout", final_answer_delivery=True,
                        loop="live", docker=True, mode="stateful")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_final_delivery_nonterminal_action_continues_same_question():
    """A delivered 'final' that does NOT complete the instance must open a fresh scope
    for the SAME question with the executed result relayed — never leak _await_feedback
    into a later boundary (stale-flag wedge from the adversarial review)."""
    ctl, submits = _final_delivery_controller(
        'Let me check one thing.\n\n{"action": "QUERY", "content": "SELECT 1"}')
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction) and a1.content == "SELECT 1"
    # runner executes it; instance NOT complete
    ctl.observe(SimpleNamespace(content="Query result: 1", instance_complete=False,
                                metadata={}), _q(queries_used=1))
    assert ctl._await_feedback is False             # consume-once: no stale flag
    assert ctl._pending_feedback == ("step", "Query result: 1")   # SAME-item header kind
    assert ctl._boundary_pending is True            # fresh scope for the same question
    ctl.close()


def test_final_delivery_echo_of_previous_blob_falls_back():
    """The relayed verdict may quote the previous answer JSON; an agent echoing it as its
    final text must NOT deliver the stale action again (echo-guard)."""
    ctl, submits = _final_delivery_controller(
        'Done.\n\n{"action": "ANSWER", "content": "42"}')
    a1 = ctl.respond(_q())
    assert a1.content == "42"
    ctl.observe(SimpleNamespace(content='Your answer: {"action": "ANSWER", "content": "42"}',
                                instance_complete=True, metadata={}), _q(idx=1))
    # next task echoes the SAME blob as its final text -> guard refuses, fallback
    a2 = ctl.respond(_q(idx=1))
    assert isinstance(a2, DatabaseAction)
    assert (a2.content or "") == ""                 # fallback, not the stale 42
    ctl.close()



def test_relay_headers_disambiguate_step_vs_verdict():
    """The qsh_code poisoning regression: a relayed STEP result must carry the
    'item is NOT finished' header, never the item-verdict header (the shared header
    made the agent read its own echo output as item completion -> empty patches)."""
    ctl, _ = _final_delivery_controller(
        'Continuing.\n\n{"action": "QUERY", "content": "SELECT 1"}')
    a1 = ctl.respond(_q())
    ctl.observe(SimpleNamespace(content="<output>COMPLETE</output>", instance_complete=False,
                                metadata={}), _q(queries_used=1))
    assert ctl._pending_feedback == ("step", "<output>COMPLETE</output>")
    bodies = []
    real_api = rb._api
    rb._api = lambda base, method, path, body=None, timeout=60: (
        bodies.append(body) or {"task_id": "tid-x"})
    ctl._published = True
    try:
        ctl._submit_task = type(ctl)._submit_task.__get__(ctl)
        ctl._submit_task(_q(queries_used=1))
    finally:
        rb._api = real_api
    d = bodies[0]["description"]
    assert d.startswith("RESULT OF YOUR PREVIOUS ACTION ON THIS ITEM")
    assert "NOT finished" in d.split("\n")[0]
    assert "VERDICT FOR YOUR PREVIOUS ITEM" not in d
    ctl.close()


def test_final_delivery_coercion_net_recovers_prose_report():
    """A perfect engineering report with no JSON must be recoverable via the
    System-supplied coerce callback (the 3 lost solved issues of qsh_code)."""
    ctl, _ = _final_delivery_controller(
        "All 138 tests pass. The fix is complete. What was implemented: ...")
    calls = []

    def _coerce(raw, schema):
        calls.append(raw)
        return schema.model_validate({"action": "ANSWER", "content": "recovered"})

    ctl._final_coerce_fn = _coerce
    a1 = ctl.respond(_q())
    assert calls, "coerce callback must be invoked on strict-parse failure"
    assert isinstance(a1, DatabaseAction) and a1.content == "recovered"
    ctl.close()


def test_final_delivery_coercion_empty_result_falls_back():
    """A coerce that degrades to a schema-empty action must NOT be delivered as the
    answer — ordinary fallback semantics (escalation counter) stay intact."""
    ctl, _ = _final_delivery_controller("prose only, nothing recoverable")
    ctl._final_coerce_fn = lambda raw, schema: schema.model_validate(
        {"action": "", "content": ""})
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert (a1.content or "") == ""                 # fallback path, not a "delivered" empty
    assert ctl._await_feedback is False             # scope NOT marked as delivered
    ctl.close()


def test_coerced_delivery_respects_and_updates_echo_guard():
    """Coerce path must have echo-guard parity with the strict path (skeptic finding):
    reproducing the previously delivered blob is rejected; a successful coerce updates
    the guard for the NEXT strict parse."""
    ctl, _ = _final_delivery_controller("prose report, no json")
    ctl._last_delivered_blob = {"action": "ANSWER", "content": "42"}
    ctl._final_coerce_fn = lambda raw, schema: schema.model_validate(
        {"action": "ANSWER", "content": "42"})      # reproduces the previous delivery
    a1 = ctl.respond(_q())
    assert (a1.content or "") == ""                  # rejected -> fallback
    ctl.close()


def test_coerce_chaff_only_result_rejected():
    """A coerce that filled only reasoning fields (thought) but no payload must be
    treated as failed — reasoning chaff must not defeat the emptiness check."""
    from pydantic import BaseModel

    class Bashish(BaseModel):
        thought: str = ""
        command: str = ""

    ctl, _ = _final_delivery_controller("prose only")
    ctl._final_coerce_fn = lambda raw, schema: Bashish(thought="I could not find a command",
                                                       command="")
    q = _q(); q.response_schema = Bashish
    a1 = ctl.respond(q)
    # falls back through the ordinary path (fallback builds schema-empty Bashish)
    assert isinstance(a1, Bashish) and (a1.command or "") == ""
    assert ctl._await_feedback is False              # NOT marked as delivered
    ctl.close()


# ---------------------------------------------------------------- re-emission fix (typed field)
def test_final_delivery_prefers_typed_field_over_prose_text():
    """THE qsh2 re-emission regression: after a blocking review the agent re-emits its
    already-committed answer as PROSE (no JSON in the final text). The engine's typed
    final_answer field (marker latch) still holds the original payload — the bridge
    must deliver from IT, not die on the text parse."""
    ctl, _ = _final_delivery_controller(
        "All review findings addressed; the submitted patch stands as reviewed.",
        final_answer='{"action": "ANSWER", "content": "42"}')
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert a1.content == "42"                      # recovered from the typed field
    assert ctl._await_feedback is True             # delivered, scope consumed
    ctl.close()


def test_final_delivery_typed_garbage_falls_back_to_text():
    """A junk typed field (prose latch, tier token) must not mask a parseable final
    text: source order is typed-first, text-second."""
    ctl, _ = _final_delivery_controller(
        'Done.\n\n{"action": "ANSWER", "content": "7"}',
        final_answer="see my patch description above")
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert a1.content == "7"                       # text path still wins when typed is junk
    ctl.close()


def test_final_delivery_typed_field_respects_echo_guard():
    """Echo-guard parity on the typed path: a typed payload identical to the PREVIOUS
    delivered action (relayed-verdict echo) must fall back, not re-deliver stale JSON."""
    ctl, _ = _final_delivery_controller(
        "prose only", final_answer='{"action": "ANSWER", "content": "42"}')
    a1 = ctl.respond(_q())
    assert a1.content == "42"
    ctl.observe(SimpleNamespace(content='Verdict: {"action": "ANSWER", "content": "42"}',
                                instance_complete=True, metadata={}), _q(idx=1))
    a2 = ctl.respond(_q(idx=1))                    # same typed blob again -> echo
    assert isinstance(a2, DatabaseAction)
    assert (a2.content or "") == ""                # fallback, not the stale 42
    ctl.close()


def test_final_delivery_action_note_marker_protocol():
    """The note must teach the engine's NATIVE marker (outcomes.FINAL_ANSWER_MARKER)
    plus the always-rule (user decision: positive 'always end with the line', incl.
    verbatim repetition after reviews/tool work — closes the latch's discard window)."""
    n = rb._ACTION_NOTE_FINAL_DELIVERY
    assert "FINAL ANSWER:" in n
    assert "EVERY reply" in n                      # always-rule, not just first emission
    assert "verbatim" in n                         # unchanged answer -> byte-identical line
    assert "submit_action" in n                    # exploration channel untouched


def test_submit_body_declares_answer_protocol_only_for_final_delivery():
    """final_delivery submits must declare the v6.60+ answer_protocol contract (pre-6.60
    gateways ignore the key); ordinary live submits must NOT (CC-parity body stays pinned)."""
    ctl, _ = _final_delivery_controller("x")
    bodies = []
    real_api = rb._api
    rb._api = lambda base, method, path, body=None, timeout=60: (
        bodies.append(body) or {"task_id": "tid-x"})
    ctl._published = True
    try:
        ctl._submit_task = type(ctl)._submit_task.__get__(ctl)
        ctl._submit_task(_q())
        ctl2, _ = _final_delivery_controller("x")
        ctl2.final_delivery = False                # same controller, flag off
        ctl2._published = True
        ctl2._submit_task = type(ctl2)._submit_task.__get__(ctl2)
        ctl2._submit_task(_q())
    finally:
        rb._api = real_api
    assert bodies[0].get("answer_protocol") == "final_answer_line"
    assert "answer_protocol" not in bodies[1]
    ctl.close(); ctl2.close()


def test_final_delivery_bad_typed_does_not_shadow_valid_text():
    """Ladder repair (adversarial finding): a typed blob that fails SCHEMA validation
    must fall through to the strict text parse, not straight to coercion/fallback."""
    ctl, _ = _final_delivery_controller(
        'Done.\n\n{"action": "ANSWER", "content": "9"}',
        final_answer='{"action": "NO_SUCH_ACTION", "bogus": 1}')
    a1 = ctl.respond(_q())
    assert isinstance(a1, DatabaseAction)
    assert a1.content == "9"                       # recovered from text despite bad typed
    ctl.close()


def test_final_delivery_typed_echo_falls_through_to_text():
    """Ladder repair: a typed ECHO of the previous answer must not abort delivery when
    the final text carries a DIFFERENT schema-valid answer."""
    ctl, _ = _final_delivery_controller(
        'Revised.\n\n{"action": "ANSWER", "content": "13"}',
        final_answer='{"action": "ANSWER", "content": "42"}')
    a1 = ctl.respond(_q())
    assert a1.content == "42"                      # first item: typed wins
    ctl.observe(SimpleNamespace(content='Verdict: {"action": "ANSWER", "content": "42"}',
                                instance_complete=True, metadata={}), _q(idx=1))
    a2 = ctl.respond(_q(idx=1))
    assert a2.content == "13"                      # typed echoed -> text answer delivered
    ctl.close()


def test_final_delivery_legit_repeat_after_exploration_not_swallowed():
    """Echo-guard scoping (adversarial finding): after REAL protocol steps on the item,
    an answer identical to the previous item's is a legitimate repeat, not an echo."""
    ctl, _ = _final_delivery_controller(
        "prose", final_answer='{"action": "ANSWER", "content": "42"}')
    a1 = ctl.respond(_q())
    assert a1.content == "42"
    ctl.observe(SimpleNamespace(content="Verdict: correct (42)",
                                instance_complete=True, metadata={}), _q(idx=1))
    a2 = ctl.respond(_q(idx=1))                    # unexplored: identical -> echo, fallback
    assert (a2.content or "") == ""
    ctl.observe(SimpleNamespace(content="Query result: it is 42 again",
                                instance_complete=False, metadata={}), _q(idx=1, queries_used=1))
    a3 = ctl.respond(_q(idx=1, queries_used=1))    # explored: identical -> legit repeat
    assert a3.content == "42"
    ctl.close()


def test_final_delivery_salvages_typed_answer_from_failed_task():
    """Reliability (smoke finding): a task that emitted a valid typed FINAL ANSWER and
    only THEN died on a transient (APIConnectionError in a post-round) must still deliver
    the answer, not abandon to a zero. 179 such errors flipped answered tasks to 'failed'
    in one db run; the typed field survives on the record regardless of terminal status."""
    ctl, _ = _final_delivery_controller(
        "prose report, no JSON",
        final_answer='{"action": "ANSWER", "content": "10.22"}',
        terminal_status="failed")
    a = ctl.respond(_q())
    assert isinstance(a, DatabaseAction)
    assert a.content == "10.22"                     # salvaged from failed task, not abandoned
    ctl.close()


def test_final_delivery_failed_task_with_no_answer_still_abandons():
    """The salvage must NOT manufacture an answer: a task that failed BEFORE producing a
    typed answer (empty field, no JSON in text) still abandons to the fallback action."""
    ctl, _ = _final_delivery_controller(
        "connection died mid-exploration", final_answer="", terminal_status="failed")
    a = ctl.respond(_q())
    assert (a.content or "") == ""                   # fallback empty action, not a fabricated one
    ctl.close()
