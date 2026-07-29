"""Ouroboros bridge: run the Ouroboros self-modifying agent as a CL-Bench system.

Mapping (see plan):
- CL-Bench *stateless baseline* (fresh process per instance, memoryless) == Ouroboros **E0**
  (`--memory-mode empty`, evolution OFF).
- CL-Bench *stateful rollout* (one persistent system across the sequence)  == Ouroboros **E1v2**
  (persistent IsolatedServer, `memory_mode=forked`, evolution ON).  [M2 — scaffolded here]
- CL-Bench `mean_gain` (stateful - stateless) == externally-validated **E1v2 - E0 lift**.

The bridge is pure output-format plumbing: it renders `query.prompt` (+ accumulated feedback) into a
reasoning prompt, runs the chosen ENGINE, and returns a `Response(action=<response_schema instance>)`.
No task-specific answers or verifier knowledge live here (CL-Bench methodology rule).

Engines (`--system.engine`):
- ``llm``       : answer each turn directly via CL-Bench's `completion_with_structured_output`.
                  This proves the *bridge plumbing* round-trips end-to-end and is the M1 smoke engine.
- ``ouroboros`` : shell out to a real Ouroboros agent (`python -m ouroboros.cli run`) per turn.
                  Requires an installed Ouroboros venv (see _launcher); this is the real-agent path (M2).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from pydantic import BaseModel

from ...interface import ContinualLearningSystem, Observation, Query, Response
from ...usage import UsageEvent
from ...registry import register_system
from ..utils import completion_with_structured_output
from . import _launcher, _docker_launcher, _live_bridge


@register_system("ouroboros")
class OuroborosSystem(ContinualLearningSystem):
    """Bridge the Ouroboros agent into CL-Bench."""

    # stateless mode gives a clean memoryless baseline -> gain is defined. Env-gated OFF switch
    # (OUROBOROS_BENCH_SKIP_BASELINE=1) for matrix experiments where the baseline arm is duplicated
    # across configs (baselines ignore conversation mode, so N configs x same model = N identical
    # baseline sets); the official runner then skips the baseline phase first-class (cli.py
    # skip_baseline path) and still flushes run artifacts. A baseline.json can be produced later by
    # a dedicated run and placed alongside run_0000.json. Default ON — leaderboard run-all needs it.
    supports_baseline: bool = os.environ.get("OUROBOROS_BENCH_SKIP_BASELINE", "") != "1"
    parallel_safe: bool = True       # state is per-object/in-memory (stateful holds its own isolated server)

    def __init__(
        self,
        model: str = "openrouter/google/gemini-3.5-flash",
        engine: str = "llm",                # "llm" (M1 plumbing smoke) | "ouroboros" (real agent)
        mode: str = "stateless",            # "stateless" (E0) | "stateful" (E1v2)
        ouroboros_repo: str = "",           # path to an Ouroboros clone (NEVER the live repo); ouroboros engine only
        evolution: Optional[bool] = None,   # default: True iff mode == "stateful"
        persistent_objective: str = "",     # evolution steer (stateful only)
        task_timeout_sec: int = 900,
        docker: bool = False,               # run the Ouroboros AGENT inside Docker (leak-proof: task+DB stay host-side)
        docker_image: str = "clbench-ouroboros:dev",
        max_workers: int = 10,              # Ouroboros server worker pool (10 = engine default; enables
                                            #   subagents. The skill-import race that forced 1 is fixed in
                                            #   the core, ec1cec1/c11f080 — pid-namespaced staging + scoped
                                            #   sweep. NOTE: does NOT speed the benchmark (runner drives one
                                            #   task per container); only lets the agent spawn subagents.
        cadence: str = "every_n:1",         # post-task evolution cadence (stateful only)
        evolution_trigger: str = "native",  # evolution window trigger: "native" (post-task daemon
                                            #   maybe_promote — stochastic at chunk boundaries, declines
                                            #   on empty backlog) | "forced" (bridge files the durable
                                            #   promotion request at each chunk boundary — deterministic;
                                            #   docker only)
        resume: bool | str = False,         # CC-style --resume: True/"continuation" = continuation (DEFAULT:
                                            #   stored system verbatim, append-only — the true CC analog) |
                                            #   "splice" = legacy splice-into-fresh-frame | False = off
        loop: str = "action",               # "action" (one Ouroboros task per action) | "live" (one LIVE task
                                            #   owns the exploration loop via the remote_work skill)
        conversation: str = "question",     # live loop scope: "question" (skill mechanics, memory carries)
                                            #   | "rollout" (ONE growing session across all questions, = CC shape)
        memory_instruction: str = "tools",  # live loop: memory note variant ("tools" | "generic" | "")
        system_prompt: str = "",
        discipline: str = "",               # opt-in behavioral steer for the live loop: "unit1"
                                            #   (built-in unit-1 discipline note) | literal text | "" off.
                                            #   Appended to the live task description; visible in traces.
        final_answer_delivery: bool = False,  # hybrid: finalizing action = task final answer (question scope)
        chunk_questions: int = 0,           # rollout chunking: end the live task every N question
                                            #   boundaries (post-task evolution window mid-benchmark)
                                            #   and resume-stitch the next chunk. 0 = off (default).
        chunk_resume_mode: str = "continuation",  # chunk seam mode: continuation (cache-hot, evolved
                                            #   SYSTEM.md/identity invisible) | splice (fresh system
                                            #   incl. evolved prompts + updated identity/memory,
                                            #   cache-cold seam) | none (no replay — continuity via
                                            #   native memory only)
        name: str = "ouroboros",
    ):
        self._name = name
        self.model = model
        self.engine = engine
        self.mode = mode
        self.ouroboros_repo = ouroboros_repo
        self.evolution = (mode == "stateful") if evolution is None else bool(evolution)
        self.persistent_objective = persistent_objective
        self.task_timeout_sec = int(task_timeout_sec)
        self.docker = bool(docker)
        self.docker_image = docker_image
        self.max_workers = int(max_workers)
        self.cadence = cadence
        self.evolution_trigger = str(evolution_trigger or "native").strip().lower()
        # Normalize resume ("continuation" | True | False): CLI params arrive as STRINGS, so
        # --system.resume false would otherwise be truthy. Unknown strings fail loudly.
        _r = str(resume).strip().lower()
        if _r in ("true", "1", "yes", "y", "continuation"):
            self.resume: bool | str = "continuation"   # continuation IS the default resume
        elif _r in ("false", "0", "no", "n", "none", ""):
            self.resume = False
        elif _r == "splice":
            self.resume = "splice"                     # explicit legacy opt-in
        else:
            raise ValueError(f"resume must be true|false|continuation|splice, got {resume!r}")
        self.loop = loop
        self.conversation = conversation
        self.memory_instruction = memory_instruction
        self.system_prompt = system_prompt
        self.discipline = discipline
        self.final_answer_delivery = bool(final_answer_delivery)
        if self.final_answer_delivery and str(conversation) != "question":
            raise ValueError(
                "final_answer_delivery requires conversation='question' (the finalizing "
                "action rides task finalization — a rollout task spans ALL questions, so "
                "its final answer cannot deliver per-question actions)")
        self.chunk_questions = int(chunk_questions or 0)
        self.chunk_resume_mode = str(chunk_resume_mode or "continuation")
        self._live: Optional[_live_bridge.LiveController] = None
        if self.loop == "live" and engine == "ouroboros" and not self.docker:
            # Fail fast at construction (review #8): the host OuroborosEngine never injects
            # the remote_work skill and runs a poisoning primer — live loop needs docker.
            raise ValueError("loop='live' requires --system.docker=true (the host engine "
                             "has no remote_work skill injection)")
        if self.resume and engine == "ouroboros" and not self.docker:
            # Same fail-fast for resume: _launcher.OuroborosEngine has no resume support, so
            # the delta-only resume prompts would reach an engine that replays NOTHING.
            raise ValueError("resume requires --system.docker=true (the host engine has no "
                             "conversation-resume chaining)")

        # Within-instance conversation (mirrors the ICL baseline). For the stateless baseline CL-Bench
        # spawns a fresh object per instance, so this never leaks across baseline instances; reset()
        # also clears it. For the stateful rollout the runner does NOT reset between instances, so the
        # accumulating history is part of the carried state (alongside Ouroboros memory/evolution in M2).
        self._messages: list[dict[str, str]] = []
        self._cur_instance: Optional[int] = None
        self._turns = 0
        self._sidecar = _launcher.SidecarLedger(benchmark="cl_bench", system=name, model=model,
                                                mode=mode, engine=engine)
        self._engine = None   # lazy _launcher.OuroborosEngine (engine="ouroboros")
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear per-instance state + tear down any isolated Ouroboros server."""
        if getattr(self, "_live", None) is not None:
            self._live.close()
            self._drain_engine_usage()
            self._live = None
        if getattr(self, "_engine", None) is not None:
            self._engine.close()
            self._engine = None
        self._messages = []
        self._cur_instance = None
        self._turns = 0
        self._question = ""        # OSWorld-style: the question, re-supplied each turn (not the full transcript)
        self._last_feedback = ""   # the most recent observation (the last action's result)
        self._notes = ""           # the agent's compact carried working memory (OSWorld 'notes' pattern)

    @property
    def name(self) -> str:
        return self._name

    def observe(self, observation: Observation, next_query: Optional[Query] = None) -> None:
        """Fold task feedback into the running conversation (so the next turn sees it)."""
        if self.engine == "ouroboros" and self.loop == "live" and self._live is not None:
            # Live loop: relay the official observation VERBATIM to the in-session agent
            # (this is the parity guarantee — no rewriting). Bookkeeping below still runs.
            self._live.observe(observation, next_query)
            if next_query is None:
                # End of the sequence: the runner consumes usage right after observe and
                # NEVER calls reset()/close() on the success path (review BLOCKER 1) —
                # finalize terminal engine-task costs NOW, before control returns.
                try:
                    self._live.finalize_usage()
                except Exception:
                    pass
            self._drain_engine_usage()
        content = (observation.content or "").strip()
        if content:
            self._last_feedback = content   # OSWorld-style: only the latest result is re-supplied next turn
            self._messages.append({"role": "user", "content": f"FEEDBACK: {content}"})

    def get_run_artifacts(self) -> dict[str, Any]:
        art = {"artifact_type": "ouroboros_bridge", "mode": self.mode, "engine": self.engine,
               "model": self.model, "turns": self._turns, "messages": list(self._messages)}
        if self._live is not None:
            art["live"] = self._live.stats()
        return art

    # ------------------------------------------------------------------ main turn
    def _drain_engine_usage(self) -> None:
        """Convert terminal engine-task costs harvested by the live bridge into CL-Bench
        UsageEvents. Without this, the official manifest under-reports real spend by ~3
        orders of magnitude (structured-output calls only)."""
        live = getattr(self, "_live", None)
        buf = getattr(live, "usage_buffer", None)
        if not buf:
            return
        for row in buf:
            self.record_usage_event(UsageEvent(
                call_type="ouroboros_engine_task",
                model=self.model,
                cost_usd=row.get("cost_usd"),
                pricing_source="ouroboros_internal_ledger",
                metadata={"task_id": row.get("task_id"), "task_status": row.get("status")},
            ))
        buf.clear()

    def respond(self, query: Query) -> Response:
        self._turns += 1
        if self.engine == "ouroboros" and self.loop == "live":
            resp = self._respond_live(query)
            # Drain AFTER the live turn: rows created by this very call (previous task
            # went terminal while opening the new scope) land in THIS interaction's
            # usage, which the runner collects right after respond() returns.
            self._drain_engine_usage()
            return resp
        if self._cur_instance is not None and query.instance_index != self._cur_instance:
            # The previous CL-Bench instance just ended. For stateful E1v2, this is where post-task
            # evolution is given a chance to absorb (budget reset + wait_for_absorb).
            if self.engine == "ouroboros" and self.mode == "stateful" and self._engine is not None:
                absorb = self._engine.on_instance_boundary()
                self._sidecar.log_turn(event="instance_boundary", prev_instance=self._cur_instance,
                                       absorb=absorb)
                # Do NOT reset the resumed conversation here. The rollout is ONE continuous conversation
                # across ALL questions (whole-rollout), matching CC's single_conversation (system.py
                # observe() resets the conversation only when single_conversation=False). The accumulated
                # prior-question context IS the in-context continual-learning signal the benchmark rewards;
                # resetting it throws that signal away and would rely on Ouroboros memory alone. Ouroboros
                # memory still persists as an additional channel. The growing transcript's cost is bounded
                # by OUROBOROS_TOTAL_BUDGET and mitigated by prompt-caching on the stable system prefix.
            # OSWorld-style: each question starts fresh; within a question we carry only a compact
            # 'notes' field forward (not an ever-growing transcript).
            self._messages = []
            self._question = ""
            # CC-parity: _last_feedback is deliberately NOT cleared here. The boundary
            # observation IS the finished question's verdict ("Question N: CORRECT/INCORRECT
            # ..."), and CC carries it as FEEDBACK into the NEXT question's first prompt
            # (claude/system.py:843-857 records it regardless of instance completion; it is
            # cleared only after a successful respond). Clearing it here blinded the resume
            # path to the one continual-learning signal the rollout metric rewards.
            self._notes = ""
        self._cur_instance = query.instance_index
        # Refresh EVERY turn (not just the first): the task's query.prompt carries the running
        # "Queries used: X/budget" line + the "explore up to N queries / you'll need to explore the
        # schema" framing, and CC re-sends query.prompt verbatim each turn. Caching only the first
        # turn's prompt dropped that framing on continuing actions and drove under-exploration.
        self._question = query.prompt or "(no content)"

        self._messages.append({"role": "user", "content": query.prompt or "(no content)"})

        if self.engine == "llm":
            action = self._engine_llm(self._build_messages(), query.response_schema)
        elif self.engine == "ouroboros":
            action = self._engine_ouroboros(query.response_schema)   # OSWorld-style notes step loop
        else:
            raise ValueError(f"unknown engine {self.engine!r} (use 'llm' or 'ouroboros')")

        self._messages.append({"role": "assistant", "content": action.model_dump_json()})
        self._sidecar.log_turn(instance_id=query.instance_id, instance_index=query.instance_index,
                               turn=self._turns, action=action.model_dump())
        return Response(action=action, metadata={"engine": self.engine, "mode": self.mode,
                                                 "turn": self._turns})

    # ------------------------------------------------------------------ live loop
    def _respond_live(self, query: Query) -> Response:
        """Live-task loop: ONE in-session Ouroboros agent owns the exploration (via the
        remote_work skill) while the official runner keeps the task/DB/rewards. The
        controller never raises (an escaped exception aborts the entire run)."""
        if self._live is None:
            if self.mode == "stateful" and self.conversation == "rollout":
                # One live task spans every question: the default 200-round cap would cut
                # it mid-rollout (review #12). Real caps remain TOTAL_BUDGET + task timeout.
                os.environ.setdefault("OUROBOROS_MAX_ROUNDS", "10000")
            if self._engine is None:
                if self.docker:
                    self._engine = _docker_launcher.DockerOuroborosEngine(
                        ouroboros_repo=self.ouroboros_repo, model=self.model, mode=self.mode,
                        evolution=self.evolution, cadence=self.cadence, steer=self.persistent_objective,
                        evolution_trigger=self.evolution_trigger,
                        task_timeout_sec=self.task_timeout_sec, primer="",
                        image=self.docker_image, max_workers=self.max_workers,
                        # chunked rollout needs in-container capture at chunk-task end so the
                        # next chunk can resume-stitch; the engine's own run_turn chaining is
                        # unused on the live path. mode "none" carries nothing -> no capture.
                        # Bare rollout needs capture too: dead-task recovery (bridge reopen)
                        # stitches the fresh scope to the dead task's capture. Baselines
                        # (stateless) stay capture-free — question scope never stitches.
                        resume=(self.chunk_resume_mode
                                if (self.chunk_resume_mode != "none"
                                    and (self.chunk_questions
                                         or (self.mode != "stateless"
                                             and self.conversation == "rollout")))
                                else False))
                else:
                    self._engine = _launcher.OuroborosEngine(
                        ouroboros_repo=self.ouroboros_repo, model=self.model, mode=self.mode,
                        evolution=self.evolution, steer=self.persistent_objective,
                        task_timeout_sec=self.task_timeout_sec)
            self._live = _live_bridge.LiveController(
                engine=self._engine,
                # NOTE (review #9): baseline instances receive the SAME system params as the
                # rollout (incl. mode/conversation) — the runner never tells a System its phase.
                # That's benign here: on a fresh single-question System both scopes behave
                # identically (submit -> solve -> seq_done), and E0 memorylessness comes from
                # the fresh per-instance engine, exactly as in the per-action path.
                conversation=("question" if self.mode == "stateless" else self.conversation),
                memory_mode=(self.memory_instruction or None),
                discipline=self.discipline,
                # Baselines are fresh single-question Systems: chunking never engages there
                # (a chunk seam needs >N completed questions in ONE System's lifetime).
                chunk_questions=(0 if self.mode == "stateless" else self.chunk_questions),
                chunk_resume_mode=self.chunk_resume_mode,
                final_answer_delivery=self.final_answer_delivery,
                final_coerce_fn=(self._coerce_final_answer if self.final_answer_delivery else None),
                fallback_action_fn=_fallback_action,
                action_timeout_sec=float(max(self.task_timeout_sec, 600)),
                task_timeout_sec=self.task_timeout_sec)
        self._cur_instance = query.instance_index
        action = self._live.respond(query)
        self._sidecar.log_turn(instance_id=query.instance_id, instance_index=query.instance_index,
                               turn=self._turns, action=action.model_dump())
        # Forfeit escalation (dead engine): merge the benchmark's own latency_timeout
        # convention into the Response metadata — domain-generic zero-credit forfeit.
        _forfeit = self._live.forfeit_metadata or {}
        self._live.forfeit_metadata = None
        return Response(action=action, metadata={"engine": "ouroboros-live", "mode": self.mode,
                                                 "conversation": self.conversation,
                                                 "turn": self._turns, **_forfeit})

    # ------------------------------------------------------------------ engines
    def _engine_llm(self, messages: list[dict], response_schema: type[BaseModel]) -> BaseModel:
        """M1 plumbing engine: structured-output LLM call (validated instance straight back)."""
        action, usage = completion_with_structured_output(
            model=self.model, messages=messages, response_schema=response_schema)
        self.record_usage_event(usage)
        return action

    def _engine_ouroboros(self, response_schema: type[BaseModel]) -> BaseModel:
        """Real-agent engine, OSWorld-style step loop (mirrors devtools osworld/run_step_agent): each turn is
        an otherwise-STATELESS Ouroboros task; we re-supply the question + the last result + the agent's OWN
        compact 'notes' (not the full transcript), and carry forward the 'notes' it returns each turn."""
        if self._engine is None:
            if self.docker:
                # Leak-proof: the AGENT runs inside a Docker container; the CL-Bench task + DB stay
                # host-side in the runner. Same run_turn surface as the host engine.
                self._engine = _docker_launcher.DockerOuroborosEngine(
                    ouroboros_repo=self.ouroboros_repo, model=self.model, mode=self.mode,
                    evolution=self.evolution, cadence=self.cadence, steer=self.persistent_objective,
                    task_timeout_sec=self.task_timeout_sec, primer="",
                    image=self.docker_image, max_workers=self.max_workers, resume=self.resume)
            else:
                self._engine = _launcher.OuroborosEngine(
                    ouroboros_repo=self.ouroboros_repo, model=self.model, mode=self.mode,
                    evolution=self.evolution, steer=self.persistent_objective,
                    task_timeout_sec=self.task_timeout_sec)
        if self.resume:
            # Full prior conversation is replayed in-container (CC-style --resume) -> send only the delta.
            prompt = _render_resume_prompt(self._question, self._last_feedback, response_schema,
                                           memory_instruction=self.memory_instruction)
        else:
            prompt = _render_notes_prompt(self._question, self._last_feedback, self._notes, response_schema)
        # CC-parity feedback lifecycle: each feedback is shown exactly ONCE (CC clears
        # _pending_feedback after a successful respond, claude/system.py:775-777). Without
        # this, an empty mid-instance observation would re-show a stale feedback line.
        self._last_feedback = ""
        raw = self._engine.run_turn(prompt)
        action, notes = self._parse_action_notes(raw, response_schema)
        if notes:
            self._notes = notes
        return action

    def _parse_action_notes(self, raw: str, response_schema: type[BaseModel]):
        """Extract the agent's action + its compact 'notes' (carried forward). 'notes' is popped before
        validating the action against the (notes-free) CL-Bench response schema."""
        blob = _extract_last_json(raw)
        notes = ""
        if isinstance(blob, dict):
            notes = str(blob.pop("notes", "") or "").strip()
            try:
                return response_schema.model_validate(blob), notes
            except Exception:
                pass
        return self._parse_action(raw, response_schema), notes

    def _coerce_final_answer(self, raw: str, response_schema: type[BaseModel]):
        """Bridge-side coercion for the final-answer channel. Differs from _parse_action:
        (1) ESCAPE HATCH — the model is told to return ALL-EMPTY fields when the text does
        not contain a clearly committed final action (progress notes / accidental endings
        must NOT be inventively closed — adversarial finding: coerce could hallucinate a
        terminal submit where the old fallback correctly left the item open);
        (2) returns None on ANY failure instead of a schema-filled fallback, so the
        bridge's rejection guards stay live for Literal/enum schemas;
        (3) caps the input (reports can be many KB; the committed answer lives at the
        edges) to bound the extra scored-path LLM call."""
        capped = raw if len(raw) <= 8000 else (raw[:2000] + "\n...\n" + raw[-6000:])
        try:
            action, usage = completion_with_structured_output(
                model=self.model,
                messages=[{"role": "user",
                           "content": "The text below is an agent's final message. If it contains a "
                                      "clearly committed final action/answer, convert it into the "
                                      "required structured response, verbatim where possible. If it "
                                      "does NOT contain a committed final action (e.g. it is a "
                                      "progress note, a plan, or an accidental ending), return every "
                                      f"field empty.\n\n{capped}"}],
                response_schema=response_schema)
            self.record_usage_event(usage)
            return action
        except Exception as exc:  # noqa: BLE001
            print(f"[ouroboros-bridge] final coerce failed: {type(exc).__name__}: "
                  f"{str(exc)[:160]}", file=sys.stderr)
            return None

    def _parse_action(self, raw: str, response_schema: type[BaseModel]) -> BaseModel:
        """Extract the agent's structured answer; fall back to a light schema-coercion call."""
        blob = _extract_last_json(raw)
        if blob is not None:
            try:
                return response_schema.model_validate(blob)
            except Exception:
                pass
        # Coerce free-form agent output into the required schema (general-purpose; no task answers).
        try:
            action, usage = completion_with_structured_output(
                model=self.model,
                messages=[{"role": "user",
                           "content": "Convert the following agent output into the required structured "
                                      f"response, verbatim where possible:\n\n{raw}"}],
                response_schema=response_schema)
            self.record_usage_event(usage)
            return action
        except Exception as exc:
            # A transient provider/transport failure during this light coerce (e.g. an SSL EOF that exhausts
            # retries, or a litellm error-mapping crash: `APIError.__init__() missing 3 args`) must NOT
            # propagate — it kills the baseline WORKER process -> BrokenProcessPool -> the WHOLE run aborts
            # (see baseline.py: any instance exception re-raises RuntimeError). Degrade to an empty (wrong)
            # action so this turn scores 0 and the run continues.
            print(f"[ouroboros-bridge] coerce failed, using empty action: "
                  f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
            return _fallback_action(response_schema)

    # ------------------------------------------------------------------ helpers
    def _build_messages(self) -> list[dict[str, str]]:
        sys = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return [*sys, *self._messages]


def _fallback_action(schema: type[BaseModel]) -> BaseModel:
    """A valid-but-empty action, returned when the light coerce call fails on a transient provider/transport
    error. Lets that turn score 0 (a wrong/empty action) instead of letting the exception kill the baseline
    worker — a worker crash breaks the spawn ProcessPool and aborts the ENTIRE run. General-purpose: fills
    each field with a benign empty (last enum/Literal member for action-type fields; "" / 0 / False otherwise)."""
    import enum as _enum
    import typing as _t
    vals: dict = {}
    for name, fld in schema.model_fields.items():
        ann = fld.annotation
        origin, args = _t.get_origin(ann), _t.get_args(ann)
        if origin is _t.Literal and args:
            vals[name] = args[-1]                                  # terminal action usually the last literal
        elif isinstance(ann, type) and issubclass(ann, _enum.Enum):
            vals[name] = list(ann)[-1].value
        elif ann is int:
            vals[name] = 0
        elif ann is float:
            vals[name] = 0.0
        elif ann is bool:
            vals[name] = False
        else:
            vals[name] = ""                                        # str + anything else
    try:
        return schema.model_validate(vals)
    except Exception:
        return schema.model_construct(**vals)


def _render_single_prompt(messages: list[dict], response_schema: type[BaseModel]) -> str:
    """Flatten the conversation + a strict output contract into one prompt for the agent engine."""
    convo = "\n\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
    schema = json.dumps(response_schema.model_json_schema(), indent=2)
    return ("This is ONE turn of the benchmark protocol. There is nothing local to explore — do NOT use "
            "tools, read files, or run commands; just reason briefly from the conversation below and emit "
            "the next action.\n\n"
            f"{convo}\n\n"
            "Emit ONLY your action as a SINGLE fenced ```json block matching this JSON schema exactly, with "
            f"nothing before or after it:\n```json-schema\n{schema}\n```")


def _render_resume_prompt(question: str, last_feedback: str,
                          response_schema: type[BaseModel],
                          memory_instruction: str = "tools") -> str:
    """Resume-mode step prompt — MIRRORS the CC reference's `_build_prompt` (claude/system.py)
    COMPONENT FOR COMPONENT: [feedback prefix] + the task's OWN prompt (`query.prompt`) VERBATIM
    + a schema instruction BYTE-IDENTICAL to CC's schema_to_prompt_instruction + the memory
    instruction (Ouroboros-native analog of CC's MEMORY.md note). NO tool suppression — CC never
    suppresses its agent's tools, and the old "nothing local to explore via tools" line was
    measured as anti-agentic pragmatics (the audit's gap #2: CC nudges memory each action, we
    nudged inaction). NO reframing ("reason briefly"/"ONE turn") either — that stripped the task's
    own exploration guidance and drove under-exploration. The full prior conversation is replayed
    in-container (resume), exactly as CC's --resume replays it."""
    from . import run_clbench_bridge_agent as _rb  # lazy: avoid import cycles
    schema = json.dumps(response_schema.model_json_schema(), indent=2)
    schema_instr = f"{_rb._SCHEMA_PROMPT_HEAD}```json\n{schema}\n```{_rb._SCHEMA_PROMPT_TAIL}"
    feedback = f"FEEDBACK FROM PREVIOUS ACTION: {last_feedback}\n\n" if last_feedback else ""
    memory_instr = _rb._MEMORY_NOTES.get(memory_instruction or "", "")
    # Protocol affordance (domain-neutral, DISCLOSED deviation from CC's bare prompt): without
    # it the agent treats each task as answer-now and emits blind guesses (measured: 6/6
    # straight final answers, zero intermediate actions). It replaces the old suppression
    # line's load-bearing half without its anti-tool half and without naming any domain action.
    # The environment-fact sentence is a FACT, not a tool prohibition: the leak-proof container
    # holds no task data, and without saying so the agent burns 5-8 LLM rounds per action on
    # run_command exploration of an empty environment (measured on db_cont40b: 15 run_command
    # calls across ~16 actions).
    protocol_note = (
        "\n\nEach reply is ONE action in a multi-turn protocol: a non-terminal action is executed "
        "and its result arrives as the next FEEDBACK, so you can gather information step by step "
        "before committing to a final/terminal action. The local execution environment contains "
        "no task data — task information is only obtainable through protocol actions."
    )
    return f"{feedback}{question}{schema_instr}{protocol_note}{memory_instr}"


def _render_notes_prompt(question: str, last_feedback: str, notes: str,
                         response_schema: type[BaseModel]) -> str:
    """OSWorld-style lean step prompt (mirrors osworld/run_step_agent's stateless steps + carried notes):
    re-supply the question + the latest result + the agent's OWN compact running notes — NOT the full
    transcript. The agent returns its action PLUS a 'notes' string we carry to its next turn."""
    schema = json.dumps(response_schema.model_json_schema(), indent=2)
    parts = ["This is ONE turn of a step-based benchmark. There is nothing local to explore — do NOT use "
             "tools, read files, or run commands; reason briefly and emit the next action.",
             "", f"QUESTION: {question}"]
    if last_feedback:
        parts += ["", f"RESULT OF YOUR LAST ACTION: {last_feedback}"]
    if notes:
        parts += ["", f"YOUR RUNNING NOTES (your own memory, carried for you across turns):\n{notes}"]
    parts += ["",
              "Emit ONLY a SINGLE fenced ```json block. It MUST contain the action fields below AND a "
              "\"notes\" string — your compact running memory (schema, values, partial findings you'll need "
              "later). The full turn history is NOT replayed to you, so put everything you must remember into "
              "\"notes\"; it is carried forward verbatim. Nothing before or after the block.",
              f"```json-schema\n{schema}\n```"]
    return "\n".join(parts)


def _extract_last_json(text: str) -> Optional[dict]:
    """Return the last well-formed top-level JSON object found in `text`, else None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    found = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found = obj
                    i += end
                    continue
            except json.JSONDecodeError:
                pass
        i += 1
    return found
