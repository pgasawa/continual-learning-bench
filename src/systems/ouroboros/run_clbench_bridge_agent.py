#!/usr/bin/env python3
"""CL-Bench runner: ONE Ouroboros agentic run per CL-Bench question, via an
in-process shim the agent drives with curl (run_command). Mirrors run_cu_bridge_agent.py
(one_run_per_task) but drives a ContinualLearningTask instead of DesktopEnv, and
reuses ``_launcher.OuroborosEngine`` for the isolation core.

Why curl instead of a dedicated skill: a ``net``/extension skill is re-reviewed to
"pending" by the server bootstrap on every boot (wiping the host native-trust stamp),
and the worker pool caches its tool catalog at spawn — so an extension enabled after
boot never reaches the worker. The bundled ``run_command`` tool is always available to
workers, so the agent reaches the shim over HTTP with curl. Same whole-task shape, so
self-evolution can still absorb in-task.

  host:  reset_baseline_instance(i) -> serve_task(build) -> shim URL in the prompt
         -> submit ONE Ouroboros task -> poll to terminal -> /_outcome reward
  agent (one run, full memory): curl /observation -> curl POST /step -> ... -> done

Run under the venv that has BOTH CL-Bench and the Ouroboros clone deps:

  cd ~/continual-learning-bench
  OUROBOROS_BENCH_CLONE=~/ouroboros-bench-src .venv/bin/python \
      -m src.systems.ouroboros.run_clbench_bridge_agent \
      --domain database_exploration --phases e0 --num-instances 1 \
      --model openrouter/anthropic/claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# --- package wiring (run as `-m src.systems.ouroboros.run_clbench_bridge_agent`) -----
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../continual-learning-bench
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.registry import get_task_class  # noqa: E402
from src.systems.ouroboros import _launcher, _docker_launcher, clbench_step_shim  # noqa: E402

DOMAINS = (
    "database_exploration", "exploitable_poker", "codebase_adaptation",
    "cohort_studies", "blind_spectrum_monitoring", "sales_prediction",
)

# CC-parity: the claude leaderboard run passed NO --allowedTools/--disallowedTools (full native
# toolset, WebSearch/WebFetch available in-container), so we run the FULL native Ouroboros toolset
# too — nothing disabled. Bench data is synthetic (gold answers not on the web), which is why
# unrestricted tools are legal for every system. Isolation stays the JOB OF DOCKER (DB lives on the
# host, outside the container), not of tool restriction. The earlier web/vision self-handicap
# (vlm_query/screenshot/browse_page/web_search/...) is retired; the task-body key stays for
# protocol stability.
# Operator port (v6.81.0 campaign): honor CLBENCH_SOLVE_DISABLED_TOOLS from env so a
# no-swarm/no-web submittable run can declare its tool exclusions from the launcher.
# Empty env -> [] (CC unrestricted-toolset parity, the previous hardcoded default).
# The live path consumes this module-level list too (_live_bridge reads _rb.DISABLED_TOOLS).
DISABLED_TOOLS: list[str] = [
    t.strip() for t in os.environ.get("CLBENCH_SOLVE_DISABLED_TOOLS", "").split(",") if t.strip()
]


_SCHEMA_PROMPT_HEAD = "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
_SCHEMA_PROMPT_TAIL = "\nRespond ONLY with the JSON object, no other text."
# Our system's neutral action-delivery note — the analog of CC emitting JSON directly; replaces CC's
# MEMORY.md instruction (Ouroboros memory is native). PUSH model: submit_action returns the next
# observation, so no get_observation poll and no "schema may change" hint is needed.
_ACTION_NOTE = (
    "\n\nSubmit each action via the remote_work skill's submit_action(action=...) tool. "
    "It returns the result of your action and the next question. Continue until it reports done."
)
# Hybrid (final-answer delivery, question scope): exploration rides the tool as usual, but the
# FINALIZING action is the task's FINAL ANSWER — parsed by the harness, gated by task-acceptance
# review before it is submitted. Domain-generic: names no domains, no action types.
# v2 (re-emission fix): the answer travels on the engine's NATIVE `FINAL ANSWER:` marker
# protocol (outcomes.FINAL_ANSWER_MARKER) so the loop latches it and stores the typed
# `final_answer` payload on the task record — the bridge reads THAT field first, so a
# post-review prose reply can no longer drop an already-committed answer. The always-rule
# (repeat the line verbatim after reviews/tool work) keeps the latch fresh on the one
# trajectory the latch legitimately discards (new tool calls after the marker).
_ACTION_NOTE_FINAL_DELIVERY = (
    "\n\nFor actions whose RESULTS you need before deciding your next step, use the remote_work "
    "skill's submit_action(action=...) tool — it returns each action's outcome. When you have "
    "determined the action that FINALIZES this item (your final committed answer), do NOT send it "
    "via the tool: end your reply with one last line of the exact form\n"
    "FINAL ANSWER: <one JSON object matching the schema>\n"
    "with nothing after that line. EVERY reply that concludes your work on this item must end "
    "with that FINAL ANSWER line — including replies made after running further commands and "
    "after addressing review findings: if your committed action is unchanged, repeat the exact "
    "same line verbatim; if it changed, emit the full corrected JSON on that line. A prose "
    "summary is never a substitute for the FINAL ANSWER line, and if any reminder asks for a "
    "bare short answer, still emit the full JSON object on that line. If you cannot know in "
    "advance whether an action finalizes the item, send it via the tool."
)
# Opt-in behavioral discipline steer (--system.discipline unit1; default OFF so the validated
# CC-parity prompt stays byte-stable). DOMAIN-GENERIC by construction — every line pre-bakes a
# lesson the agent PROVABLY self-learns mid-session but too late for the unit-1-scored metric
# (measured: codebase adopted run-the-real-tests at instance 5 after 3 zeros; sales spent 76% of
# its regret unlearning priors in the first 5 instances; DB learned the re-audit/arbitrate lessons
# only after repeated INCORRECT verdicts). Zero task knowledge: no domain nouns, no trap values.
# Legality class: system-owned prompt framing, same lane as ace's injected playbook / CC's system
# prompt; disclosed in METHODOLOGY.md and visible verbatim in every trace's task description.
_STEER_NOTE_UNIT1 = (
    "\n\nOperating discipline (applies to EVERY item, including the very first):\n"
    "- If you receive any notice or hint that the environment, schema, or data may have changed, "
    "re-inspect the live metadata/state before relying on earlier conclusions — a stale assumption "
    "costs far more than one verification step.\n"
    "- Before committing a final/terminal answer, verify it when the budget allows: derive it a "
    "second, independent way. If two candidate answers disagree, never submit either — spend one "
    "more step to arbitrate the disagreement first.\n"
    "- When an authoritative verification mechanism exists (a test suite, a checker, ground-truth "
    "feedback), run it before declaring success; your own reproduction is not authoritative.\n"
    "- In multi-part answers, give every sub-part its own analysis from the first item onward; "
    "never fill a sub-part with a default or baseline value without checking it.\n"
    "- Parse graded feedback fully (each sub-entry, not just the headline) and re-anchor on the "
    "revealed ground truth; refine your approach incrementally each round."
)
_STEER_NOTES = {"unit1": _STEER_NOTE_UNIT1}


def resolve_steer_note(discipline: str) -> str:
    """'' -> no steer (default); a known key -> the built-in note; any other
    non-empty string -> literal custom steer text (prefixed for prompt joining)."""
    d = (discipline or "").strip()
    if not d:
        return ""
    if d in _STEER_NOTES:
        return _STEER_NOTES[d]
    return "\n\n" + d
# Ouroboros-native analog of CC's MEMORY.md instruction (parity: CC is explicitly told to learn+store
# across turns; without this our agent writes ZERO memory and the rollout can't beat stateless). Points
# to Ouroboros's OWN memory tools, not CC's MEMORY.md. Toggled by --memory-instruction for the A/B.
# generic = CC-near-verbatim (no Ouroboros tool names). tools = + explicit native-tool pointers.
_MEMORY_NOTE_GENERIC = (
    "\n\nYou should learn and store information from each interaction, so use your memory. You have a "
    "persistent, file-based memory system. Read it at the start of each turn and update your memory as you "
    "learn new information that will be useful in future turns."
)
_MEMORY_NOTE_TOOLS = (
    "\n\nYou should learn and store information from each interaction, so use your memory. You have a "
    "persistent, file-based memory system. Read it at the start of each turn and update your memory (using "
    "update_scratchpad for working notes and knowledge_write for durable knowledge) as you learn new "
    "information that will be useful in future turns."
)
_MEMORY_NOTES = {"generic": _MEMORY_NOTE_GENERIC, "tools": _MEMORY_NOTE_TOOLS}


def _build_prompt(obs: dict, *, memory_mode=None, steer_note: str = "",
                  action_note: str | None = None) -> str:
    """CC-MIRRORED prompt: the SAME components CC assembles — the task's question (query.prompt, verbatim)
    + a schema instruction byte-identical to CC's shared schema_to_prompt_instruction — plus our neutral
    action-delivery note. NO benchmark framing, NO 'be economical', NO 'schema may change' hint. Built per
    question from the live first observation. memory_hint=True appends the Ouroboros-native memory
    instruction (parity with CC's MEMORY.md instruction) — placed where CC puts its memory note."""
    question = (obs.get("prompt") or "").rstrip()
    schema = obs.get("response_schema")
    schema_instr = ""
    if schema is not None:
        # NOTE: _SCHEMA_PROMPT_TAIL starts with "\n" — no extra newline after the fence, so the
        # rendered block is BYTE-IDENTICAL to CC's schema_to_prompt_instruction (test_live_parity).
        schema_instr = f"{_SCHEMA_PROMPT_HEAD}```json\n{json.dumps(schema, indent=2)}\n```{_SCHEMA_PROMPT_TAIL}"
    memory_instr = _MEMORY_NOTES.get(memory_mode or "", "")
    # steer_note (opt-in) sits BEFORE the action note: the how-to-act mechanics must stay
    # the prompt's tail. Appending the steer after them (the first placement) displaced the
    # tool-invocation instruction from the end and the agent started NARRATING actions as
    # text instead of calling submit_action (measured: 33/30/43 completed-without-action
    # abandons in the first validation attempt; the seed with the old tail was clean).
    return f"{question}{schema_instr}{memory_instr}{steer_note}{action_note or _ACTION_NOTE}"


def _publish_target(data_dir, shim_url: str) -> None:
    """Write the shim URL where the remote_work skill's 3-tier resolver reads it (per question)."""
    from ouroboros.skill_loader import skill_state_dir
    sd = Path(skill_state_dir(data_dir, "remote_work"))
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "shim_target.txt").write_text(shim_url, encoding="utf-8")


# ----------------------------------------------------------------- HTTP helper
def _api(base: str, method: str, path: str, body: Optional[dict] = None, timeout: float = 60.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    return json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}


# ----------------------------------------------------------- per-question run
def _make_build(domain: str, i: int, *, run_index=None):
    """Build the task + its i-th baseline instance (sliced). schema_drift variant via schedule='default'
    (40 questions = 20 pre + 20 post-drift), matching CC. run_index selects the per-seed question shuffle
    (None == order 0 == CC's baseline order)."""
    def build():
        cls = get_task_class(domain)
        task = cls(schedule="default", run_index=run_index)
        return task, task.reset_baseline_instance(i)
    return build


def _wait_health(url: str, timeout_sec: int) -> None:
    deadline = time.time() + max(5, timeout_sec)
    while time.time() < deadline:
        try:
            if _api(url, "GET", "/healthz", timeout=5).get("ok"):
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("shim never became healthy")


def _poll_terminal(base: str, task_id: str, timeout_sec: int) -> dict:
    final = {"completed", "failed", "cancelled", "rejected_duplicate"}
    deadline = time.time() + max(60, timeout_sec)
    latest: dict = {}
    while True:
        if time.time() >= deadline:
            try:
                _api(base, "POST", f"/api/tasks/{task_id}/cancel", {})
            except Exception:
                pass
            return {"status": "timeout"}
        try:
            latest = _api(base, "GET", "/api/tasks/" + task_id, timeout=30)
        except Exception:
            time.sleep(5)
            continue
        if str(latest.get("status") or "") in final:
            return latest
        time.sleep(8)


def run_one_question(engine, domain: str, i: int, num_instances: int, run_dir: Path, args) -> dict:
    """ONE Ouroboros task solves ONE CL-Bench question i (agent curls the shim). Returns {reward, ...}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    server, url, driver = clbench_step_shim.serve_task(
        _make_build(domain, i, run_index=getattr(args, "run_index", None)),
        max_validation_retries=args.max_validation_retries)
    try:
        _wait_health(url, args.startup_timeout_sec)
        # Publish the URL the AGENT uses to reach the shim. Host mode: 127.0.0.1 (loopback).
        # Docker mode: host.docker.internal (the container reaches the host-loopback shim).
        agent_shim_url = engine.shim_url_for_agent(url) if hasattr(engine, "shim_url_for_agent") else url
        _publish_target(engine.data_root, agent_shim_url)   # skill resolves the shim from shim_target.txt
        first_obs = _api(url, "GET", "/observation", timeout=30)  # host-side fetch: the actual question + schema
        prompt = _build_prompt(first_obs, memory_mode=getattr(args, "memory_instruction", None))  # CC-mirrored
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        base = engine.server.base_url
        # The server can be transiently 503 right after an evolution boundary (budget reset / re-exec),
        # so wait for health + retry the submit (mirrors the old bridge's post-boundary wait_for_health).
        try:
            engine.server.wait_for_health(timeout=args.startup_timeout_sec)
        except Exception:
            pass
        body = {
            "description": prompt,
            "memory_mode": engine.mm,
            "disabled_tools": DISABLED_TOOLS,
            "actor_id": "remote-driver",
            "source": "remote-driver",
            "timeout_sec": args.task_timeout_sec,
            "metadata": {"source": "remote-driver", "delegation_role": "root"},
        }
        created, last_exc = None, None
        for attempt in range(6):
            try:
                created = _api(base, "POST", "/api/tasks", body, timeout=60)
                break
            except Exception as exc:  # transient 503 / URLError during server recovery
                last_exc = exc
                time.sleep(5)
        if created is None:
            raise RuntimeError(f"/api/tasks submit failed after retries: {last_exc}")
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"no task_id from /api/tasks: {created!r}")

        final = _poll_terminal(base, task_id, args.task_timeout_sec)
        (run_dir / "ouroboros_task_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

        outcome = _api(url, "GET", "/_outcome", timeout=30)
        row = {
            "domain": domain, "instance_index": i, "reward": outcome.get("reward"),
            "success": outcome.get("success"), "ouroboros_status": str(final.get("status") or ""),
            "task_id": task_id, "queries_used": _api(url, "GET", "/observation").get("queries_used"),
            "cost_usd": final.get("cost_usd"),
        }
        (run_dir / "task_outcome.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    finally:
        server.shutdown()
        driver.close()


# -------------------------------------------------------- E0/E1v2 orchestration
def _new_engine(args, *, mode: str, evolution: bool):
    if getattr(args, "docker", False):
        return _docker_launcher.DockerOuroborosEngine(
            ouroboros_repo=args.ouroboros_repo, model=args.model, mode=mode,
            evolution=evolution, cadence=args.cadence, steer=args.persistent_objective,
            task_timeout_sec=args.task_timeout_sec, primer="",
            image=args.docker_image, max_workers=args.max_workers)
    return _launcher.OuroborosEngine(
        ouroboros_repo=args.ouroboros_repo, model=args.model, mode=mode,
        evolution=evolution, cadence=args.cadence, steer=args.persistent_objective,
        task_timeout_sec=args.task_timeout_sec, primer="",  # NO per-turn "don't use tools" primer
    )


def _stateless_one(args, out_root: Path, i: int):
    """One stateless question: fresh engine (own container/port/data-dir) + own shim. Independent —
    safe to run concurrently. Returns (i, reward); reward=None on infra failure."""
    eng = _new_engine(args, mode="stateless", evolution=False)
    try:
        eng._start()
        rd = out_root / "stateless" / args.domain / f"q{i:03d}"
        row = run_one_question(eng, args.domain, i, args.num_instances, rd, args)
        print(f"[stateless] q{i}: reward={row['reward']} queries={row['queries_used']} "
              f"cost=${row.get('cost_usd')}", flush=True)
        return i, row["reward"]
    except Exception as exc:  # noqa: BLE001
        print(f"[stateless] q{i}: FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return i, None
    finally:
        eng.close()


def run_stateless(args, out_root: Path) -> list:
    """STATELESS (E0 baseline): a FRESH isolated server per question, evolution OFF, no cross-question
    memory carry. == CL-Bench stateless baseline; compare to CC baseline. Questions are INDEPENDENT, so
    --concurrency N runs N at a time (each in its own container/port/shim)."""
    conc = max(1, int(getattr(args, "concurrency", 1)))
    start = max(0, int(getattr(args, "start_instance", 0)))
    idxs = list(range(start, args.num_instances))
    results: dict = {}
    if conc == 1:
        for i in idxs:
            _, r = _stateless_one(args, out_root, i)
            results[i] = r
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[stateless] running questions {start}..{args.num_instances-1} at concurrency={conc}", flush=True)
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(_stateless_one, args, out_root, i) for i in idxs]
            for f in as_completed(futs):
                i, r = f.result()
                results[i] = r
    return [results.get(i) for i in idxs]


def _make_continuous_build(domain: str, *, run_index=None):
    """Continuous rollout build: start the FULL instance sequence (NOT sliced) so the schema_drift DB swap
    + NOTICE fire mid-sequence (idx crossing pre_drift_count), exactly as CC's rollout experiences them."""
    def build():
        cls = get_task_class(domain)
        task = cls(schedule="default", run_index=run_index)
        return task, task.reset()
    return build


def _submit_and_poll(engine, prompt: str, args) -> dict:
    """Submit ONE Ouroboros agent task (shared memory) to the persistent server; poll to terminal."""
    base = engine.server.base_url
    try:
        engine.server.wait_for_health(timeout=args.startup_timeout_sec)
    except Exception:
        pass
    body = {
        "description": prompt, "memory_mode": engine.mm, "disabled_tools": DISABLED_TOOLS,
        "actor_id": "remote-driver", "source": "remote-driver",
        "timeout_sec": args.task_timeout_sec, "metadata": {"source": "remote-driver", "delegation_role": "root"},
    }
    created, last_exc = None, None
    for _ in range(6):
        try:
            created = _api(base, "POST", "/api/tasks", body, timeout=60)
            break
        except Exception as exc:  # transient 503 during server recovery
            last_exc = exc
            time.sleep(5)
    if created is None:
        raise RuntimeError(f"/api/tasks submit failed after retries: {last_exc}")
    task_id = str(created.get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"no task_id from /api/tasks: {created!r}")
    return _poll_terminal(base, task_id, args.task_timeout_sec)


def run_stateful(args, out_root: Path, *, evolution: bool) -> list:
    """STATEFUL ROLLOUT (CC-comparable): ONE persistent Ouroboros server + ONE CONTINUOUS task across all
    instances (the schema_drift migration fires mid-sequence, as CC's rollout experiences it). One agent
    task solves each question against the shared continuous shim; memory carries across questions via shared
    memory; the host advances the rollout between questions. evolution=False -> memory-only headline
    (== CC rollout); True -> + post-task self-evolution (research-only, not a CC-gain)."""
    label = "stateful_evo" if evolution else "stateful_noevo"
    eng = _new_engine(args, mode="stateful", evolution=evolution)
    rewards: dict = {}
    try:
        eng._start()
        server, url, driver = clbench_step_shim.serve_task(
            _make_continuous_build(args.domain, run_index=getattr(args, "run_index", None)),
            max_validation_retries=args.max_validation_retries, continuous=True)
        try:
            _wait_health(url, args.startup_timeout_sec)
            agent_shim_url = eng.shim_url_for_agent(url) if hasattr(eng, "shim_url_for_agent") else url
            _publish_target(eng.data_root, agent_shim_url)
            steps = 0
            while True:
                obs = _api(url, "GET", "/observation", timeout=30)
                idx = obs.get("instance_index")
                if obs.get("done") or idx is None:
                    break
                rd = out_root / label / args.domain / f"q{idx:03d}"
                rd.mkdir(parents=True, exist_ok=True)
                prompt = _build_prompt(obs, memory_mode=getattr(args, "memory_instruction", None))
                (rd / "prompt.txt").write_text(prompt, encoding="utf-8")
                final = _submit_and_poll(eng, prompt, args)
                (rd / "ouroboros_task_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
                oc = _api(url, "GET", "/_outcome", timeout=30)
                this = next((o for o in reversed(oc.get("instance_outcomes") or [])
                             if o.get("instance_index") == idx), None)
                rew = this.get("reward") if this else None
                rewards[idx] = rew
                (rd / "task_outcome.json").write_text(json.dumps({
                    "domain": args.domain, "instance_index": idx, "reward": rew,
                    "success": (this or {}).get("success"), "ouroboros_status": str(final.get("status") or ""),
                    "cost_usd": final.get("cost_usd"),
                }, indent=2), encoding="utf-8")
                absorb = eng.on_instance_boundary()   # budget reset; waits for absorb ONLY if evolution
                (rd / "absorb.json").write_text(json.dumps(absorb), encoding="utf-8")
                print(f"[{label}] q{idx}: reward={rew} absorb={absorb.get('absorbed')}/{absorb.get('reason')}", flush=True)
                adv = driver.advance_question()
                steps += 1
                # full rollout = 40 (variant size); --num-instances < 40 caps it for a smoke (drift at q20)
                if adv.get("seq_done") or steps >= max(1, args.num_instances):
                    break
        finally:
            server.shutdown()
            driver.close()
    finally:
        eng.close()
    n = (max(rewards) + 1) if rewards else 0
    return [rewards.get(i) for i in range(n)]


def _aggregate(summary: dict) -> dict:
    """Means per condition + the decomposed effects (memory vs evolution) + CC reference."""
    def _mean(xs):
        xs = [x for x in (xs or []) if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 4) if xs else None
    s = dict(summary)
    m_sl = _mean(summary.get("reward_stateless"))
    m_sn = _mean(summary.get("reward_stateful_noevo"))
    m_se = _mean(summary.get("reward_stateful_evo"))
    s["mean_stateless"], s["mean_stateful_noevo"], s["mean_stateful_evo"] = m_sl, m_sn, m_se
    if m_sl is not None and m_sn is not None:
        s["memory_effect"] = round(m_sn - m_sl, 4)        # stateful_noevo - stateless
    if m_sn is not None and m_se is not None:
        s["evolution_effect"] = round(m_se - m_sn, 4)     # stateful_evo - stateful_noevo  (the research question)
    if m_sl is not None and m_se is not None:
        s["total_gain"] = round(m_se - m_sl, 4)
    s["cc_reference_database_exploration"] = {"baseline": 0.205, "rollout": 0.551, "model": "claude-sonnet-4.6"}
    return s


def main() -> int:
    p = argparse.ArgumentParser(description="CL-Bench via host-side Ouroboros shim bridge (one run per question).")
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--phases", choices=["stateless", "stateful_noevo", "stateful_evo", "all"],
                   default="stateless", help="which condition(s) to run; 'all' = the 3-way comparison")
    p.add_argument("--num-instances", type=int, default=1)
    p.add_argument("--ouroboros-repo", default=os.environ.get("OUROBOROS_BENCH_CLONE", ""),
                   help="Ouroboros CLONE for the isolated server (NEVER the live repo)")
    p.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4.6")  # match CC (claude-sonnet-4-6)
    p.add_argument("--run-index", type=int, default=None,
                   help="per-seed question-shuffle index (schema_drift). None == order 0 == CC baseline order")
    p.add_argument("--memory-instruction", choices=["generic", "tools"], default=None,
                   help="append a memory instruction (parity with CC's MEMORY.md note): 'generic' = "
                        "CC-near-verbatim, 'tools' = + explicit update_scratchpad/knowledge_write pointers. "
                        "Default off. A/B this to measure whether memory engages + helps.")
    p.add_argument("--cadence", default="every_n:1")
    p.add_argument("--persistent-objective", default="")
    p.add_argument("--result-dir", default=os.path.expanduser("~/cl_bench_runs/clbench_bridge"))
    p.add_argument("--task-timeout-sec", type=int, default=900)
    p.add_argument("--startup-timeout-sec", type=int, default=120)
    p.add_argument("--max-validation-retries", type=int, default=8)
    p.add_argument("--docker", action="store_true",
                   help="Run the Ouroboros server inside a Docker container (leak-proof: DB on host only)")
    p.add_argument("--docker-image", default="clbench-ouroboros:dev")
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=1,
                   help="stateless ONLY: run N questions concurrently (each own container/port/shim)")
    p.add_argument("--start-instance", type=int, default=0,
                   help="stateless ONLY: first question index to run (append to an existing result-dir; "
                        "safe because schema_drift is OFF -> instances=pre_questions[:N] is order-stable)")
    args = p.parse_args()
    if not args.ouroboros_repo:
        p.error("--ouroboros-repo (or $OUROBOROS_BENCH_CLONE) is required")

    out_root = Path(args.result_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"domain": args.domain, "model": args.model, "num_instances": args.num_instances}

    conditions = (["stateless", "stateful_noevo", "stateful_evo"] if args.phases == "all" else [args.phases])
    for cond in conditions:
        if cond == "stateless":
            summary["reward_stateless"] = run_stateless(args, out_root)
        elif cond == "stateful_noevo":
            summary["reward_stateful_noevo"] = run_stateful(args, out_root, evolution=False)
        elif cond == "stateful_evo":
            summary["reward_stateful_evo"] = run_stateful(args, out_root, evolution=True)
        # incremental write: a crash mid-comparison still preserves completed conditions
        (out_root / "gain_summary.json").write_text(json.dumps(_aggregate(summary), indent=2), encoding="utf-8")

    print(json.dumps(_aggregate(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
