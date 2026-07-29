"""Contract tests for the v6.81 submission ports (codex adapter review, 2026-07-28).

Covers the three review-mandated contracts:
1. engine-task cost harvest: delayed/final cost recorded exactly once with the subtree
   total, survives the runner's real lifecycle (no reset()/close() on success), never
   blocks the hot path;
2. campaign env-override loop: explicit env wins over parity defaults, AUTO_LOW stays
   tri-state;
3. single-model pin: a foreign live vision slot never survives into isolated settings.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from src.systems.ouroboros import run_clbench_bridge_agent as rb
from src.systems.ouroboros._live_bridge import LiveController
from src.systems.ouroboros.system import _fallback_action


class _CostServer:
    """Task record server: cost is absent on the first read, final on the next."""

    base_url = "http://cost-fake"

    def __init__(self):
        self.reads = 0
        self.cancelled = []

    def record(self):
        if self.reads <= 2:
            return {"status": "completed"}  # terminal, cost not checkpointed yet
        return {"status": "completed", "cost_usd": 1.0,
                "cost_usd_with_children": 3.0, "cost_final": True,
                "cost_accounting_status": "available"}

    def wait_task(self, *a, **k):
        return {"status": "completed"}

    def cancel_task(self, tid):
        self.cancelled.append(tid)


def _controller(server):
    ctl = LiveController(engine=SimpleNamespace(server=server, mm="shared",
                                                on_instance_boundary=lambda: None),
                         conversation="question", memory_mode="tools",
                         fallback_action_fn=_fallback_action,
                         action_timeout_sec=2, task_timeout_sec=5)
    return ctl


def test_engine_usage_final_delayed_cost_is_recorded_once_with_subtree_total(monkeypatch):
    srv = _CostServer()
    ctl = _controller(srv)

    def fake_api(base, method, path, body=None, timeout=30.0):
        assert timeout <= 3.5, "hot-path harvest must use a short socket timeout"
        srv.reads += 1
        return srv.record()

    monkeypatch.setattr(rb, "_api", fake_api)

    # Hot path: one non-blocking read sees no cost -> parked, nothing recorded, no sleep.
    t0 = time.time()
    ctl._harvest_task_usage("tid-final", "completed", block=False)
    assert time.time() - t0 < 2, "non-blocking harvest must not poll"
    assert ctl.usage_buffer == []
    assert "tid-final" in ctl._usage_pending
    assert "tid-final" not in ctl._usage_harvested

    # A synthetic 'terminal' label must NOT clobber the pending retry with a null row.
    ctl._harvest_task_usage("tid-final", "terminal", block=False)
    assert ctl.usage_buffer == [] and "tid-final" in ctl._usage_pending

    # Finalization (what observe(next_query=None) triggers): reads the FINAL record and
    # records the SUBTREE total exactly once.
    ctl.finalize_usage(budget_sec=10)
    rows = [r for r in ctl.usage_buffer if r["task_id"] == "tid-final"]
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 3.0, "must prefer cost_usd_with_children"
    assert rows[0]["cost_final"] is True
    assert rows[0]["status"] == "completed", "status must come from the task record"

    # Idempotence: further terminal paths must not duplicate the event.
    ctl._harvest_task_usage("tid-final", "completed", block=True)
    ctl.close()
    assert len([r for r in ctl.usage_buffer if r["task_id"] == "tid-final"]) == 1


def test_engine_usage_budget_exhaustion_records_nonfinal_not_lost(monkeypatch):
    ctl = _controller(_CostServer())
    monkeypatch.setattr(rb, "_api",
                        lambda *a, **k: {"status": "failed", "cost_usd": 0.42})
    ctl._harvest_task_usage("tid-fail", "failed", block=True, budget_sec=1.0)
    rows = [r for r in ctl.usage_buffer if r["task_id"] == "tid-fail"]
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 0.42, "failed tasks with billable calls must be recorded"
    assert rows[0]["cost_final"] is False, "non-final read must be marked as such"


def test_docker_overrides_explicit_campaign_knobs_win_and_auto_low_stays_tristate(monkeypatch):
    from src.systems.ouroboros._docker_launcher import DockerOuroborosEngine

    def overrides(env: dict) -> dict:
        for k in ("OUROBOROS_RUNTIME_MODE", "OUROBOROS_TASK_REVIEW_MODE",
                  "OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_EFFORT_TASK",
                  "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
                  "OUROBOROS_SAFETY_MODE", "OUROBOROS_CONTEXT_MODE",
                  "OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_MAX_SUBAGENT_DEPTH"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        eng = DockerOuroborosEngine.__new__(DockerOuroborosEngine)
        eng.model = "anthropic/claude-sonnet-4.6"
        eng.evolution = False
        eng.resume = False
        eng.cadence = ""
        eng.steer = ""
        return eng._overrides()

    ov = overrides({})
    assert ov["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert ov["OUROBOROS_TASK_REVIEW_MODE"] == "auto"
    assert ov["OUROBOROS_REVIEW_ENFORCEMENT"] == "advisory"
    assert "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in ov, "absent env must stay UNKNOWN"
    assert ov["OUROBOROS_MODEL_VISION"] == "anthropic/claude-sonnet-4.6"

    ov = overrides({"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_TASK_REVIEW_MODE": "required",
                    "OUROBOROS_REVIEW_ENFORCEMENT": "blocking", "OUROBOROS_SAFETY_MODE": "off",
                    "OUROBOROS_EFFORT_TASK": "high", "OUROBOROS_EFFORT_REVIEW": "low",
                    "OUROBOROS_EFFORT_SCOPE_REVIEW": "low", "OUROBOROS_CONTEXT_MODE": "max",
                    "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
                    "OUROBOROS_MAX_SUBAGENT_DEPTH": "0"})
    assert ov["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert ov["OUROBOROS_TASK_REVIEW_MODE"] == "required"
    assert ov["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert ov["OUROBOROS_SAFETY_MODE"] == "off"
    assert ov["OUROBOROS_EFFORT_TASK"] == "high"
    assert ov["OUROBOROS_EFFORT_REVIEW"] == "low", "split review effort must survive"
    assert ov["OUROBOROS_EFFORT_SCOPE_REVIEW"] == "low"
    assert ov["OUROBOROS_CONTEXT_MODE"] == "max"
    assert ov["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert ov["OUROBOROS_MAX_SUBAGENT_DEPTH"] == "0"


def test_single_model_pin_overrides_foreign_live_vision_slot(monkeypatch):
    from src.systems.ouroboros._docker_launcher import DockerOuroborosEngine

    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE", raising=False)
    eng = DockerOuroborosEngine.__new__(DockerOuroborosEngine)
    eng.model = "anthropic/claude-sonnet-4.6"
    eng.evolution = False
    eng.resume = False
    eng.cadence = ""
    eng.steer = ""
    ov = eng._overrides()
    live = {"OUROBOROS_MODEL_VISION": "openai::foreign-model", "TOTAL_BUDGET": 60.0}
    merged = {**live, **ov}  # build_isolated_settings applies overrides after live settings
    assert merged["OUROBOROS_MODEL_VISION"] == "anthropic/claude-sonnet-4.6"
