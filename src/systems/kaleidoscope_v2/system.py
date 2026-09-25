"""
Kaleidoscope memory system for the continual learning benchmark, version 2.

In plain terms: before every answer the system still looks up what it has
remembered and shows it to the model, exactly as version 1 does. The model can
now also read and write that memory itself, with Kaleidoscope's own two tools
as they ship. And after every turn's feedback, a note-taker reads the whole
task instance so far, together with the stored memories most like it, and
creates, corrects or deletes memories with the same ``remember`` tool.

Per turn:

1. **Automatic search, unchanged from version 1.** Version 1's shingle-diff key
   (``_SearchKey``) searches the private vault for up to ``top_k`` (10)
   memories with ``kscope call ... search --json``, and they go into the
   prompt as mem0-style lines under ``=== RELEVANT MEMORIES ===``.
2. **The model's tool loop, on the Responses API.** One ``litellm.responses``
   call at a time. Azure refuses function tools together with
   ``reasoning_effort`` for gpt-5.6-luna on ``/v1/chat/completions``, and the
   start-up hook sets ``reasoning.effort`` high on this route. Each call goes
   through ICL's own ``ProviderTurnClient``, as ICL's responses route does:
   ``_litellm_responses_with_retry`` makes it (with ICL's retries),
   ``_capture_openai_response`` keeps the chain, and
   ``_usage_from_provider_response`` builds the usage event. Three tools:

   - ``search`` and ``remember``: name, description and parameters from the
     shipped surface (``mcp_surface.json``, checked against its pinned sha256
     when the system is built). **One adaptation:** a parameters schema's
     top-level ``oneOf``, ``anyOf``, ``allOf``, ``enum``, ``const`` and
     ``not`` are dropped, only at the top level, because Azure refuses them
     there ("schema must have type 'object' and not have ... at the top
     level"). Of the shipped schemas this removes exactly one keyword: the
     ``search`` schema's top-level ``oneOf`` (exactly one of ``query`` and
     ``memory_id``). kscope still enforces that rule itself, and its refusal
     goes back to the model. Everything else, descriptions included, is
     verbatim. Both are sent with ``"strict": false``: MCP clients do not use
     strict mode, and strict validation makes a model fill every optional
     field. ``ks2_surface`` records what was dropped and the ``strict`` each
     tool is sent with, on every response; ``ks2_answer``'s ``tools_strict``
     records what the provider echoed back.
   - ``submit_answer``: its parameters are the query's own response schema,
     through the same adaptation. It is sent without ``strict``, so it keeps
     the provider's default.
   - A ``search`` or ``remember`` call runs the pinned ``kscope call <vault>
     <op>`` with the model's argument string as its stdin, in version 1's
     private environment. The tool result is kscope's plain CLI output,
     verbatim: the display text of a search and the short receipt of a
     remember, which is what an agent sees through MCP. A refusal (exit 2)
     comes back the same way, its envelope unchanged, so the model can
     correct the call and resend it. Any other failure returns a short
     ``error: ...`` string and is counted as a failure. The proof fields are
     read from that text: a search's status, the number of memories it served
     (its ``Memory N`` lines) and the ids it names (only an addressed lookup
     names one), a remember's created, updated and deleted ids (its ``Created
     |``, ``Updated |``, ``Deleted |`` and ``Item N |`` lines). An execution's
     outcome is ``ok``, ``refused`` or ``error``; a batch kscope wrote only in
     part (exit 3) is ``ok``, with the items it refused in ``refused_items``.
     Output of a shape kscope was not seen to print is shown to the model as
     it is but counted as a failure, so it cannot pass unseen.
   - ``submit_answer`` is validated with the benchmark's own
     ``validate_with_coercion``. A validation error goes back as the tool
     result; the third rejection in a turn fails the answer.
   - At most 8 ``search`` and ``remember`` executions a turn (any other
     non-submit call counts too). After the 8th, and after a plain-text reply,
     ``tool_choice`` forces ``submit_answer`` for the rest of the turn. A
     forced call after a plain-text reply has nothing else to send, so it
     carries the one-line ``FORCE_SUBMIT_NOTE``.
   - A failed call raises ``RuntimeError("LLM call failed: answer: ...")``, as
     in version 1, and a provider refusal passes through as
     ``ProviderRefusalError``.
   - **The conversation** is stored by the provider (``store=True``) and
     chained with ``previous_response_id`` for the whole instance: the prompts
     with their memory blocks, the function calls and their outputs (every
     call's output is sent with the next request, ``submit_answer``'s
     included), the submitted answer and the feedback, as a user message. A
     fresh chain starts when an instance completes, and on ``reset()``.
     **What bounds it:** one instance, and the model's own context window.
     Nothing is trimmed, so an instance that outgrew the window would fail its
     answer call visibly instead of losing context silently.
3. **The note-taker, after every answered turn's feedback.** It sees the
   instance's transcript so far as plain text, without the injected memory
   blocks and capped to its most recent 30,000 tokens (oldest turns go first),
   and the 10 stored memories one ``--json`` search with this turn's key
   returns, each with its ``memory_id``, ``version_id``, title and body. Its
   instructions are one neutral sentence and the surface's instructions,
   verbatim; the context ends with one neutral line asking it to decide what
   to remember. Its only tool is ``remember`` (adapted as above, which leaves
   it unchanged), at most 4 executions a run, with refusals returned to it,
   on a fresh chain of its own for every run. It never raises: a failure, of
   the model call or of kscope, is recorded and counted in
   ``ks_failures_total``.
4. **Proof in every response's metadata:** ``ks2_auto_search``,
   ``ks2_agent_tools`` (every search and remember the model ran, with its
   outcome and ``ids``, a search's full arguments, status and served count, a
   remember's ids by effect), ``ks2_answer`` (the tool loop),
   ``ks2_notetaker`` (the note-taker
   runs since the previous response), ``ks2_surface``, and
   ``ks_failures_total`` and ``ks_binary`` as in version 1. The run artifacts
   add every memory written, with its text, and what no response reported.
5. **Reset and isolation, as version 1:** a fresh vault in a
   ``kscope_clbench_*`` temp dir on ``reset()``, deleted with the process.

Version 1 is imported read-only and never changed: the pinned binary, the
private environment, the vault, the search key, the search classification and
retries, and the mem0-format memory lines all come from it.

Disclosed: the only text not taken from the shipped product or the task is the
two neutral sentences and the note-taker's closing line (``LUNA_PREAMBLE``,
``NOTETAKER_PREAMBLE``, ``NOTETAKER_TASK``), ``submit_answer``'s description,
the short results and the one note the loop itself needs (``SUBMITTED``,
``NOT_EXECUTED``, ``ALREADY_SUBMITTED``, ``FORCE_SUBMIT_NOTE``, ``error:
...``) and the transcript's section labels. Nothing refers to any benchmark
task.

Version 2.0.1 changes only how the note-taker's transcript is counted and cut.
tiktoken's byte-pair encoding takes time quadratic in the length of one
pre-tokenizer piece, and a run of one character class is one piece, so a
query result whose separator line and padding ran to megabytes kept 2.0's
note-taker counting for hours. ``_transcript_text`` now calls ``_count`` and
``_tail``: on a text with no run of whitespace or of non-whitespace longer
than 20,000 characters (whitespace as the tokenizer's ``\\s`` means it,
Unicode's White_Space) they make exactly 2.0's calls; on any other they
encode it in 2,000-character chunks. ``get_run_artifacts()`` records the
version and this file's sha256 as ``ks2_code``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from typing import Any

import litellm
from pydantic import BaseModel

from ...errors import as_provider_refusal
from ...interface import (
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import UsageEvent
from ..kaleidoscope import system as v1
from ..utils import count_tokens
from ..utils.provider_adapters import ProviderTurnClient, _object_value
from ..utils.structured_output import validate_with_coercion

logger = logging.getLogger(__name__)

# As in version 1 and mem0.
litellm.drop_params = True

# ---- the shipped surface ------------------------------------------------------

# A copy of experiments/clbench-2026-09/v2-shipped-surface/mcp_surface.json from
# the kaleidoscope worktree: the pinned binary's `initialize` instructions and
# `tools/list`, unedited. Read when the system is built, refused unless its
# bytes hash to this.
_SURFACE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mcp_surface.json"
)
_SURFACE_SHA256 = "c2b7c9ea32c6f7cfba163cab677240fda75eb0da533ce5392bc35bd879ee0fc1"
_KSCOPE_TOOLS = ("search", "remember")
# Refused by Azure at the top level of a function's parameters schema; dropped
# there and nowhere else.
_ROOT_KEYWORDS_DROPPED = ("oneOf", "anyOf", "allOf", "enum", "const", "not")
# The `strict` each function tool is sent with. MCP clients do not use strict
# mode, and strict validation makes a model fill every optional field, so the
# kscope tools are sent non-strict; submit_answer is sent without the field
# and gets the provider's default, as it always has (None: not sent).
_TOOL_STRICT = {"search": False, "remember": False}

# ---- the only text that is ours -----------------------------------------------

LUNA_PREAMBLE = "You are an agent working on a benchmark task."
NOTETAKER_PREAMBLE = (
    "You maintain Kaleidoscope memory for an agent that is working through a "
    "sequence of related benchmark tasks."
)
NOTETAKER_TASK = (
    "Decide what, if anything, from this interaction should be remembered, and "
    "write it with `remember`; update or delete a memory above if the "
    "interaction shows it is wrong."
)
SUBMIT_TOOL = "submit_answer"
SUBMIT_DESCRIPTION = (
    "Submit your response to the current prompt. Its arguments are the "
    "response itself, in the task's response format."
)
SUBMITTED = "Submitted."
NOT_EXECUTED = "Not executed: the tool-call limit for this turn has been reached."
ALREADY_SUBMITTED = f"Not used: an answer was already submitted with {SUBMIT_TOOL}."
FORCE_SUBMIT_NOTE = f"Submit your response with {SUBMIT_TOOL}."
TRANSCRIPT_HEADING = "The current task instance so far:"
RELATED_HEADING = "Memories already stored that may be related:"
OMITTED = "(The first {n} turn(s) of this instance are left out to fit the limit.)"

# ---- limits -------------------------------------------------------------------

# The model's search and remember executions per turn; then submit is forced.
_MAX_TOOL_EXECUTIONS = 8
# Rejected submit_answer calls a turn may retry before the answer fails.
_MAX_SUBMIT_RETRIES = 2
# A turn's model calls: the executions, a plain-text reply, every submit
# attempt, and one spare. Only a provider that ignores a forced tool_choice
# can reach it.
_MAX_MODEL_CALLS = _MAX_TOOL_EXECUTIONS + _MAX_SUBMIT_RETRIES + 3
# The note-taker's remember executions per run.
_MAX_NOTETAKER_EXECUTIONS = 4
# The note-taker's transcript, in the benchmark's own token count.
_TRANSCRIPT_TOKENS = 30_000
_DETAIL_CHARS = 200
# 2.0.1: a text holding a run longer than this is counted and cut in chunks of
# _CHUNK_CHARS characters (_count, _tail); any other gets 2.0's exact calls.
_GIANT_RUN_CHARS = 20_000
_CHUNK_CHARS = 2_000

# ---- provenance ---------------------------------------------------------------


def _own_sha256() -> str:
    """This file's sha256, from its bytes on disk."""
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


_CODE_VERSION = "2.0.1"
_SYSTEM_PY_SHA256 = _own_sha256()

# A tool execution's outcome, as PLAN-v2.md slice A item 4 names them.
_OUTCOMES = ("ok", "refused", "error")

# ---- kscope's plain output, as the pinned binary prints it --------------------

# A remember receipt: one "<Label> | mem_..." line for a single write, or an
# "Item N | <Label>" block with "Memory | mem_..." for each item of a batch;
# each written record has a "Version | ver_..." line, and a replayed request
# a "Replayed | ..." line. Labels the binary carries: Created, Updated,
# Deleted, Merged, Unfolded (memory kept), Refused (nothing written), Partial.
_RECEIPT_EFFECTS = {
    "Created": "created",
    "Unfolded (memory kept)": "created",
    "Updated": "updated",
    "Deleted": "deleted",
    "Merged": "consolidated",
}
_RECEIPT_SINGLE = re.compile(
    r"^(Created|Updated|Deleted|Merged|Unfolded \(memory kept\)|Partial) \| "
    r"(mem_[\w-]+)$"
)
_RECEIPT_ITEM = re.compile(r"^Item (\d+) \| (.+)$")
_RECEIPT_MEMORY = re.compile(r"^Memory \| (mem_[\w-]+)$")
_RECEIPT_VERSION = re.compile(r"^Version \| (ver_[\w-]+)$")
# A ranked search's display text opens with this line and names a tag, new in
# every response, that starts each served memory's "<tag> | Memory N | ..."
# line; an addressed one opens with "Memory | mem_..." and, when the memory
# is not live, says "Status: <status>".
_SEARCH_HEADER = "Kaleidoscope memory context"
_SEARCH_TAG = re.compile(r"^Lines beginning `(kscope:[0-9a-f]+) \|`", re.MULTILINE)
_SEARCH_ADDRESSED = re.compile(r"^Memory \| (mem_[\w-]+)$")
_SEARCH_STATUS = re.compile(r"^Status: (\w+)$", re.MULTILINE)


def _load_surface(path: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """The shipped surface and its sha256, refused unless its bytes are the
    pinned ones."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ValueError(f"the shipped surface {path!r} cannot be read: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"the shipped surface {path!r} has sha256 {actual}, "
            f"not the pinned {expected_sha256}"
        )
    surface = json.loads(raw.decode("utf-8"))
    names = {tool.get("name") for tool in surface.get("tools") or []}
    if not isinstance(surface.get("instructions"), str) or set(_KSCOPE_TOOLS) - names:
        raise ValueError(f"the shipped surface {path!r} lacks instructions or a tool")
    return surface, actual


def _adapted_parameters(parameters: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The parameters schema without the top-level keywords Azure refuses
    there, and which of them were dropped. Nothing below the top level, and
    nothing else at it, changes."""
    kept = copy.deepcopy(parameters)
    dropped = [key for key in _ROOT_KEYWORDS_DROPPED if key in kept]
    for key in dropped:
        del kept[key]
    return kept, dropped


def _function_tool(
    name: str, description: str, parameters: dict[str, Any], strict: bool | None = None
) -> tuple[dict[str, Any], list[str]]:
    """One Responses API function tool, and the keywords its schema lost.
    ``strict`` is sent only when it is not None."""
    kept, dropped = _adapted_parameters(parameters)
    tool = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": kept,
    }
    if strict is not None:
        tool["strict"] = strict
    return tool, dropped


def _function_calls(response: Any) -> list[dict[str, str]]:
    """The reply's function calls, in order.

    litellm 1.81.6 returns them in ``response.output`` as
    ``ResponseFunctionToolCall`` items whose ``arguments`` is the model's
    JSON as a string, answered by ``call_id``.
    """
    calls = []
    for item in _object_value(response, "output") or []:
        if _object_value(item, "type") != "function_call":
            continue
        arguments = _object_value(item, "arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(
            {
                "call_id": str(
                    _object_value(item, "call_id") or _object_value(item, "id") or ""
                ),
                "name": str(_object_value(item, "name") or ""),
                "arguments": arguments,
            }
        )
    return calls


def _output_text(response: Any) -> str:
    """The reply's plain text, if it wrote any."""
    parts = []
    for item in _object_value(response, "output") or []:
        if _object_value(item, "type") != "message":
            continue
        for part in _object_value(item, "content") or []:
            text = _object_value(part, "text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _echoed_strict(response: Any) -> dict[str, Any]:
    """Each tool's ``strict`` as the provider echoed it back: whether strict
    schema validation applied to the model's arguments."""
    strict = {}
    for tool in _object_value(response, "tools") or []:
        name = _object_value(tool, "name")
        if isinstance(name, str):
            strict[name] = _object_value(tool, "strict")
    return strict


def _function_output(call_id: str, output: str) -> dict[str, str]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _json_object(text: str | None) -> dict[str, Any]:
    try:
        value = json.loads(text or "")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _arguments_record(arguments: str) -> Any:
    """The model's arguments as sent: the JSON object, or the raw text."""
    try:
        value = json.loads(arguments)
    except ValueError:
        return arguments
    return value if isinstance(value, dict) else arguments


def _search_result(text: str) -> tuple[str | None, int, list[str]]:
    """A search's status, served count and the ids its output names, read off
    its display text; a status of None is output this reader does not
    recognise. A ranked search's display text names no memory ids; an
    addressed one names the memory it found."""
    body = (text or "").strip()
    lines = body.splitlines()
    first = lines[0] if lines else ""
    if first == _SEARCH_HEADER:
        tag = _SEARCH_TAG.search(body)
        prefix = f"{tag.group(1)} | Memory " if tag else None
        served = sum(line.startswith(prefix) for line in lines) if prefix else 0
        return ("ok" if served else "empty"), served, []
    if match := _SEARCH_ADDRESSED.match(first):
        status = _SEARCH_STATUS.search(body)
        if status:
            return status.group(1), 0, []
        return "found", 1, [match.group(1)]
    return None, 0, []


def _receipt_records(text: str) -> list[dict[str, Any]]:
    """One record per write a remember receipt reports."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in (text or "").splitlines():
        line = line.rstrip()
        if match := _RECEIPT_ITEM.match(line):
            current = {"item": int(match.group(1)), "label": match.group(2)}
        elif current is None and (match := _RECEIPT_SINGLE.match(line)):
            current = {"item": None, "label": match.group(1)}
            current["memory_id"] = match.group(2)
        elif current is not None:
            if current["item"] is not None and (match := _RECEIPT_MEMORY.match(line)):
                current.setdefault("memory_id", match.group(1))
            elif match := _RECEIPT_VERSION.match(line):
                current.setdefault("version_id", match.group(1))
            elif line.startswith("Replayed |"):
                current["replayed"] = True
            continue
        else:
            continue
        records.append(current)
    return records


def _remember_effects(text: str, arguments: str) -> dict[str, Any]:
    """What one remember call changed, read from its receipt.

    ``versions`` and ``items`` map each id written to its new version and to
    the item that wrote it, so the run artifacts can carry its text;
    ``unknown`` lists the labels this reader does not know.
    """
    request = _json_object(arguments)
    items = request.get("items") if isinstance(request.get("items"), list) else None
    effects: dict[str, Any] = {
        "created": [],
        "updated": [],
        "deleted": [],
        "consolidated": 0,
        "replayed": 0,
        "refused_items": 0,
        "unfolded": 0,
        "unknown": [],
        "records": 0,
        "versions": {},
        "items": {},
    }
    for record in _receipt_records(text):
        effects["records"] += 1
        label, memory_id = record["label"], record.get("memory_id")
        kind = _RECEIPT_EFFECTS.get(label)
        if record.get("replayed") or label == "Replayed":
            effects["replayed"] += 1
        elif label.startswith("Refused"):
            effects["refused_items"] += 1
        elif kind == "consolidated":
            effects["consolidated"] += 1
        elif kind is None or not memory_id:
            effects["unknown"].append(label)
        else:
            effects[kind].append(memory_id)
            effects["unfolded"] += label.startswith("Unfolded")
            effects["versions"][memory_id] = record.get("version_id")
            position = record["item"]
            if position is not None and items is not None:
                item = items[position - 1] if 0 < position <= len(items) else {}
            else:
                item = request
            effects["items"][memory_id] = item if isinstance(item, dict) else {}
    return effects


def _related_entry(hit: dict[str, Any]) -> str:
    """One stored memory as the note-taker sees it: ids, title and body."""
    first, _, rest = hit["content_md"].strip().partition("\n")
    if first.startswith("# "):
        title, body = v1._flat(first[2:]), v1._flat(rest)
    else:
        title, body = "", v1._flat(hit["content_md"])
    return (
        f"memory_id: {hit['memory_id']}\n"
        f"version_id: {hit.get('version_id') or 'unknown'}\n"
        f"title: {title}\n"
        f"body: {body}"
    )


def _render_turn(turn: dict[str, Any]) -> str:
    """One turn of the transcript, as plain text, without the memory block."""
    parts = [f"TURN {turn['number']}", f"QUESTION:\n{turn['question']}"]
    parts.extend(turn["events"])
    if turn["answer"] is not None:
        parts.append(f"SUBMITTED ANSWER:\n{turn['answer']}")
    else:
        parts.append("SUBMITTED ANSWER:\n(none: the answer call failed)")
    parts.append(f"FEEDBACK:\n{turn['feedback'] or '(none)'}")
    return "\n\n".join(parts)


def _tool_event(name: str, arguments: str, result: str) -> str:
    return f"TOOL CALL {name}:\n{arguments}\n\nTOOL RESULT:\n{result}"


def _tail_tokens(model: str, text: str, limit: int) -> str:
    """The last ``limit`` tokens of ``text``, in litellm's count for ``model``."""
    try:
        tokens = litellm.encode(model=model, text=text)
        if len(tokens) <= limit:
            return text
        return litellm.decode(model=model, tokens=tokens[-limit:])
    except Exception:
        return text[-limit * 4 :]


# Unicode's White_Space, which is what the tokenizer's pre-tokenizer means by
# ``\s``. Python's str.isspace() and re's ``\s`` also count U+001C..U+001F,
# which the tokenizer keeps inside a run of punctuation, so neither is used.
_WHITESPACE = (
    r"\t\n\x0b\x0c\r \x85\xa0"
    r"\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
)
# Each match is one whole run, of whitespace or of anything else.
_RUNS = re.compile(rf"[{_WHITESPACE}]+|[^{_WHITESPACE}]+")


def _longest_run(text: str) -> int:
    """The length of the longest run of whitespace or of non-whitespace in
    ``text``, in one linear pass. A tokenizer piece spans at most one such
    run, a whitespace character before it and newlines after it."""
    runs = (match.end() - match.start() for match in _RUNS.finditer(text))
    return max(runs, default=0)


def _count(model: str, text: str) -> int:
    """``count_tokens(model, text)``, unless ``text`` holds a run longer than
    ``_GIANT_RUN_CHARS``: then the sum of ``count_tokens`` over its
    consecutive ``_CHUNK_CHARS``-character chunks."""
    if _longest_run(text) <= _GIANT_RUN_CHARS:
        return count_tokens(model, text)
    return sum(
        count_tokens(model, text[start : start + _CHUNK_CHARS])
        for start in range(0, len(text), _CHUNK_CHARS)
    )


def _tail(model: str, text: str, limit: int) -> str:
    """``_tail_tokens(model, text, limit)``, unless ``text`` holds a run longer
    than ``_GIANT_RUN_CHARS``: then its ``_CHUNK_CHARS``-character chunks are
    encoded from the end until they hold at least ``limit`` tokens, and the
    last ``limit`` of them are decoded."""
    if _longest_run(text) <= _GIANT_RUN_CHARS:
        return _tail_tokens(model, text, limit)
    try:
        chunks: list[list[int]] = []
        held, end = 0, len(text)
        while end > 0 and held < limit:
            start = max(0, end - _CHUNK_CHARS)
            chunks.append(litellm.encode(model=model, text=text[start:end]))
            held += len(chunks[-1])
            end = start
        if end == 0 and held <= limit:
            return text
        tokens = [token for chunk in reversed(chunks) for token in chunk]
        return litellm.decode(model=model, tokens=tokens[-limit:])
    except Exception:
        return text[-limit * 4 :]


def _new_counters() -> dict[str, int]:
    counters = {
        # Version 1's own search and vault code updates these four.
        "searches": 0,
        "search_degraded_retries": 0,
        "search_licence_retries": 0,
        "vaults_failed": 0,
        "related_searches": 0,
        "agent_model_calls": 0,
        "agent_chained_turns": 0,
        "agent_forced_tool_cap": 0,
        "agent_forced_plain_text": 0,
        "agent_force_notes": 0,
        "agent_submit_rejected": 0,
        "agent_not_executed": 0,
        "agent_unknown_tools": 0,
        "notetaker_runs": 0,
        "notetaker_skipped": 0,
        "notetaker_model_calls": 0,
        "notetaker_model_error": 0,
        "notetaker_internal_error": 0,
        "notetaker_not_executed": 0,
        "notetaker_unknown_tools": 0,
        "tool_licence_retries": 0,
        "tool_output_unrecognised": 0,
        "transcript_turns_omitted": 0,
        "memories_created": 0,
        "memories_updated": 0,
        "memories_deleted": 0,
    }
    for status in v1._SEARCH_STATUSES:
        counters[f"search_{status}"] = 0
        counters[f"related_{status}"] = 0
    for who in ("agent", "notetaker"):
        for tool in _KSCOPE_TOOLS:
            for outcome in _OUTCOMES:
                counters[f"{who}_{tool}_{outcome}"] = 0
    return counters


def _new_notes() -> dict[str, Any]:
    """The note-taker runs since the previous response, as ``ks2_notetaker``.

    ``calls`` counts remember executions and ``model_calls`` the note-taker's
    model calls; ``refused`` counts remember calls kscope refused and
    ``partial`` those it wrote only in part (returned to the note-taker, not
    failures);
    ``errors`` counts failures, split into ``model_errors`` and
    ``kscope_errors`` (a failed related-memory search included); ``related``
    has one entry per run: the related search's status and served ids.
    """
    return {
        "ran": False,
        "runs": 0,
        "skipped": 0,
        "model_calls": 0,
        "calls": 0,
        "created_ids": [],
        "updated_ids": [],
        "deleted_ids": [],
        "refused": 0,
        "partial": 0,
        "errors": 0,
        "model_errors": 0,
        "kscope_errors": 0,
        "not_executed": 0,
        "unknown_tools": 0,
        "related": [],
    }


@register_system("kaleidoscope_v2")
class KaleidoscopeV2System(v1.KaleidoscopeSystem):
    """
    Kaleidoscope-backed memory system, version 2: version 1's automatic search,
    the shipped ``search`` and ``remember`` tools in the model's own hands, and
    a note-taker that sees the instance so far and can correct memories.
    """

    def __init__(
        self,
        model: str = "openai/gpt-5",
        extraction_model: str = "",
        system_prompt: str = "",
        top_k: int = 10,
        *,
        kscope_binary: str,
        kscope_sha256: str,
        kscope_entitlement_src: str,
        kscope_timeout_s: float = 120.0,
    ):
        """
        Args:
            model: LiteLLM model for task responses.
            extraction_model: LiteLLM model for the note-taker (defaults to model).
            system_prompt: Additional system prompt content, as in version 1.
            top_k: Memories per automatic and related search (10, as mem0).
            kscope_binary: Path of the pinned kscope executable (no PATH fallback).
            kscope_sha256: The binary's expected SHA-256.
            kscope_entitlement_src: Directory holding ``api-key`` and
                ``verdict-v1.json``. Each vault gets its own copy.
            kscope_timeout_s: Per-process timeout for kscope.
        """
        # Before anything else, and before any kscope process starts.
        surface, self._surface_sha256 = _load_surface(_SURFACE_PATH, _SURFACE_SHA256)
        shipped = {tool["name"]: tool for tool in surface["tools"]}
        self._kscope_tools = []
        self._dropped: dict[str, list[str]] = {}
        for name in _KSCOPE_TOOLS:
            tool, dropped = _function_tool(
                name,
                shipped[name]["description"],
                shipped[name]["inputSchema"],
                strict=_TOOL_STRICT[name],
            )
            self._kscope_tools.append(tool)
            if dropped:
                self._dropped[name] = dropped
        self._remember_tool = self._kscope_tools[_KSCOPE_TOOLS.index("remember")]
        self._instructions = surface["instructions"]
        super().__init__(
            model=model,
            extraction_model=extraction_model,
            system_prompt=system_prompt,
            top_k=top_k,
            kscope_binary=kscope_binary,
            kscope_sha256=kscope_sha256,
            kscope_entitlement_src=kscope_entitlement_src,
            kscope_timeout_s=kscope_timeout_s,
        )

    # ---- state ----

    def _clear_state(self) -> None:
        """Everything ``reset`` zeroes, apart from the vault itself."""
        super()._clear_state()
        self._counters = _new_counters()
        # The model's chain for the current instance, and what the next call
        # must send along it: owed function-call outputs, feedback, a prompt.
        self._answer_client = ProviderTurnClient(model=self.model)
        self._notes_client = ProviderTurnClient(model=self.extraction_model)
        self._pending: list[dict[str, Any]] = []
        self._transcript: list[dict[str, Any]] = []
        self._turn_key = ""
        self._memories: dict[str, dict[str, Any]] = {}
        self._unreported_notes = _new_notes()

    def _luna_instructions(self) -> str:
        parts = [f"{LUNA_PREAMBLE}\n\n{self._instructions}"]
        if self.user_system_prompt:
            parts.append(self.user_system_prompt)
        return "\n\n".join(parts)

    def _surface_record(self, extra: dict[str, list[str]] | None = None) -> dict:
        """``ks2_surface``: the surface's hash, what the adaptation dropped, and
        the ``strict`` each tool is sent with (None: not sent, so the
        provider's default applies; the provider's echo is in ``ks2_answer``'s
        ``tools_strict``)."""
        dropped = {**self._dropped, **(extra or {})}
        return {
            "sha256_16": self._surface_sha256[:16],
            "adapted": bool(dropped),
            "root_keywords_dropped": copy.deepcopy(dropped),
            "strict": {**_TOOL_STRICT, SUBMIT_TOOL: None},
        }

    # ---- model calls ----

    def _call_model(
        self,
        kind: str,
        client: ProviderTurnClient,
        input_items: list[dict[str, Any]],
        instructions: str,
        tools: list[dict[str, Any]],
        tool_choice: Any,
    ) -> tuple[Any, UsageEvent]:
        """One Responses API call on ``client``'s chain, made, chained and
        billed exactly as ICL's responses route makes, chains and bills its
        own. A provider refusal is raised as ``ProviderRefusalError``; any
        other failure as ``RuntimeError("LLM call failed: <kind>: ...")``."""
        kwargs: dict[str, Any] = {
            "model": client.model,
            "input": input_items,
            "instructions": instructions,
            "tools": tools,
            "tool_choice": tool_choice,
            "store": True,
        }
        if client.state.previous_response_id:
            kwargs["previous_response_id"] = client.state.previous_response_id
        try:
            response = client._litellm_responses_with_retry(**kwargs)
        except Exception as exc:
            refusal = as_provider_refusal(
                exc, provider="openai", model=client.model, messages=input_items
            )
            if refusal is not None:
                raise refusal from exc
            raise RuntimeError(
                f"LLM call failed: {kind}: {type(exc).__name__}: {exc}"
            ) from exc
        client._capture_openai_response(response)
        event = client._usage_from_provider_response(response, "openai")
        self.record_usage_event(event)
        return response, event

    # ---- kscope tools ----

    def _tool_error(
        self, who: str, record: dict[str, Any], detail: str
    ) -> tuple[str, dict[str, Any]]:
        detail = " ".join(str(detail).split())[:_DETAIL_CHARS]
        record.update(outcome="error", detail=detail)
        self._counters[f"{who}_{record['tool']}_error"] += 1
        self._failures_total += 1
        logger.warning("kscope %s %s error: %s", who, record["tool"], detail)
        return f"error: {detail}", record

    def _execute(
        self, who: str, name: str, arguments: str
    ) -> tuple[str, dict[str, Any]]:
        """Run one ``search`` or ``remember`` the model asked for, on this vault.

        The model's argument string is kscope's stdin, unchanged, and kscope
        validates it. Returns the tool result the model sees (kscope's plain
        output, a refusal included, or a short error string) and the record
        for the metadata. A licence refusal is retried once, as version 1
        retries its own. Never raises.
        """
        record: dict[str, Any] = {"tool": name, "outcome": "error", "ids": []}
        if name == "search":
            record["arguments"] = _arguments_record(arguments)
        else:
            request = _json_object(arguments)
            items = request.get("items")
            record.update(
                mode=request.get("mode"),
                batch_items=len(items) if isinstance(items, list) else None,
            )
        if self._vault is None:
            return self._tool_error(who, record, "no vault")
        licence = 0
        while True:
            try:
                proc = subprocess.run(
                    [self.kscope_binary, "call", *self._vault.address, name],
                    input=arguments,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.kscope_timeout_s,
                    env=v1._private_env(self._vault.dir),
                )
            except subprocess.TimeoutExpired:
                return self._tool_error(
                    who, record, f"timed out after {self.kscope_timeout_s}s"
                )
            except Exception as exc:
                return self._tool_error(who, record, f"{type(exc).__name__}: {exc}")
            if proc.returncode == 4 and licence < v1._LICENCE_RETRIES:
                licence += 1
                self._counters["tool_licence_retries"] += 1
                time.sleep(v1._LICENCE_RETRY_DELAY_S)
                continue
            break

        code = proc.returncode
        if code == 2:
            record.update(outcome="refused", detail=v1._refusal_message(proc))
            self._counters[f"{who}_{name}_refused"] += 1
            return (proc.stdout if proc.stdout.strip() else proc.stderr), record
        if code == 4:
            return self._tool_error(
                who, record, f"licence refused: {v1._first_line(proc.stderr)}"
            )
        if code not in (0, 3):
            return self._tool_error(
                who, record, f"exit {code}: {v1._first_line(proc.stderr)}"
            )
        if name == "search":
            status, served, ids = _search_result(proc.stdout)
            record.update(outcome="ok", status=status, served=served, ids=ids)
            recognised = status is not None
        else:
            effects = _remember_effects(proc.stdout, arguments)
            # Exit 3 is a batch kscope wrote only in part: its outcome is ok,
            # and refused_items counts the items it refused.
            record.update(
                outcome="ok",
                ids=effects["created"] + effects["updated"] + effects["deleted"],
                created=effects["created"],
                updated=effects["updated"],
                deleted=effects["deleted"],
                consolidated=effects["consolidated"],
                replayed=effects["replayed"],
                refused_items=effects["refused_items"],
                unfolded=effects["unfolded"],
            )
            self._note_memories(effects, who)
            recognised = effects["records"] > 0 and not effects["unknown"]
        if not recognised:
            # kscope acted, so the model sees what it printed; the proof
            # cannot read it, so it counts as a failure.
            record.update(outcome="error", detail=f"exit {code}: output not recognised")
            self._counters["tool_output_unrecognised"] += 1
            self._counters[f"{who}_{name}_error"] += 1
            self._failures_total += 1
            logger.warning("kscope %s %s output not recognised", who, name)
            return proc.stdout, record
        self._counters[f"{who}_{name}_{record['outcome']}"] += 1
        return proc.stdout, record

    def _note_memories(self, effects: dict[str, Any], who: str) -> None:
        """Keep every written memory's current text for the run artifacts."""
        for kind in ("created", "updated", "deleted"):
            self._counters[f"memories_{kind}"] += len(effects[kind])
            for memory_id in effects[kind]:
                entry = self._memories.setdefault(
                    memory_id,
                    {
                        "memory_id": memory_id,
                        "created_by": None,
                        "title": "",
                        "body": "",
                    },
                )
                if kind == "created":
                    entry["created_by"] = who
                entry.update(
                    version_id=effects["versions"].get(memory_id),
                    status="deleted" if kind == "deleted" else "live",
                    last_written_by=who,
                )
                if kind != "deleted":
                    item = effects["items"].get(memory_id) or {}
                    delta = item.get("semantic_delta")
                    title = delta.get("title") if isinstance(delta, dict) else None
                    entry["title"] = str(title or "")
                    entry["body"] = str(item.get("content_md") or "")

    # ---- answering ----

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1
        # Set again only once this turn is answered, so a failed answer is
        # never paired with the next observation's feedback.
        self._last_query = None
        self._last_response = None

        # 1. The automatic search, exactly as version 1.
        key = self._search_key.build(query.prompt)
        status, hits = self._search(key)
        self._counters["searches"] += 1
        self._counters[f"search_{status}"] += 1
        if status not in v1._SEARCH_NON_FAILURES:
            self._failures_total += 1
        lines = [v1._memory_line(hit) for hit in hits]
        parts = []
        if lines:
            parts.append(
                "=== RELEVANT MEMORIES ===\n"
                + "\n".join(lines)
                + "\n========================="
            )
        parts.append(query.prompt)

        self._turn_key = key
        turn: dict[str, Any] = {
            "number": len(self._transcript) + 1,
            "question": query.prompt,
            "events": [],
            "answer": None,
            "feedback": None,
            "block": None,
            "tokens": None,
        }
        self._transcript.append(turn)
        self._pending.append({"role": "user", "content": "\n\n".join(parts)})

        # 2. The tool loop.
        submit, submit_dropped = _function_tool(
            SUBMIT_TOOL, SUBMIT_DESCRIPTION, query.response_schema.model_json_schema()
        )
        tools = [*self._kscope_tools, submit]
        instructions = self._luna_instructions()
        loop: dict[str, Any] = {
            "model_calls": 0,
            "chained": self._answer_client.state.previous_response_id is not None,
            "response_ids": [],
            "forced": None,
            "force_notes": 0,
            "submit_rejected": 0,
            "plain_text_replies": 0,
            "not_executed": 0,
            "unknown_tools": 0,
            "tools_strict": None,
        }
        self._counters["agent_chained_turns"] += loop["chained"]
        agent_tools: list[dict[str, Any]] = []
        executions = 0
        answer: BaseModel | None = None
        last_rejection = ""
        while answer is None:
            if loop["model_calls"] >= _MAX_MODEL_CALLS:
                raise RuntimeError(
                    f"LLM call failed: answer: no accepted {SUBMIT_TOOL} after "
                    f"{_MAX_MODEL_CALLS} model calls"
                )
            if not self._pending:
                # Only after a plain-text reply: a request must carry input.
                self._pending.append({"role": "user", "content": FORCE_SUBMIT_NOTE})
                loop["force_notes"] += 1
                self._counters["agent_force_notes"] += 1
            choice = (
                {"type": "function", "name": SUBMIT_TOOL} if loop["forced"] else "auto"
            )
            response, _ = self._call_model(
                "answer",
                self._answer_client,
                self._pending,
                instructions,
                tools,
                choice,
            )
            # Sent: the chain now holds it.
            self._pending = []
            loop["model_calls"] += 1
            self._counters["agent_model_calls"] += 1
            loop["response_ids"].append(_object_value(response, "id"))
            if loop["tools_strict"] is None:
                loop["tools_strict"] = _echoed_strict(response)
            text = _output_text(response)
            if text:
                turn["events"].append(f"ASSISTANT:\n{text}")
            calls = _function_calls(response)
            if not calls:
                loop["plain_text_replies"] += 1
                if not loop["forced"]:
                    loop["forced"] = "plain_text"
                    self._counters["agent_forced_plain_text"] += 1
                continue
            # Every call gets its output, sent with the next request.
            for call in calls:
                name, arguments = call["name"], call["arguments"]
                if name == SUBMIT_TOOL and answer is None:
                    try:
                        answer = validate_with_coercion(
                            arguments, query.response_schema
                        )
                    except Exception as exc:
                        loop["submit_rejected"] += 1
                        self._counters["agent_submit_rejected"] += 1
                        result = last_rejection = f"{type(exc).__name__}: {exc}"
                        turn["events"].append(_tool_event(name, arguments, result))
                    else:
                        result = SUBMITTED
                        turn["answer"] = answer.model_dump_json()
                elif name == SUBMIT_TOOL:
                    result = ALREADY_SUBMITTED
                    turn["events"].append(_tool_event(name, arguments, result))
                elif executions >= _MAX_TOOL_EXECUTIONS:
                    loop["not_executed"] += 1
                    self._counters["agent_not_executed"] += 1
                    result = NOT_EXECUTED
                    turn["events"].append(_tool_event(name, arguments, result))
                else:
                    executions += 1
                    if name in _KSCOPE_TOOLS:
                        result, outcome = self._execute("agent", name, arguments)
                        agent_tools.append(outcome)
                    else:
                        loop["unknown_tools"] += 1
                        self._counters["agent_unknown_tools"] += 1
                        result = f"error: there is no tool named {name!r}"
                    turn["events"].append(_tool_event(name, arguments, result))
                self._pending.append(_function_output(call["call_id"], result))
            if answer is None and loop["submit_rejected"] > _MAX_SUBMIT_RETRIES:
                raise RuntimeError(
                    f"LLM call failed: answer: {SUBMIT_TOOL} was rejected "
                    f"{loop['submit_rejected']} times: {last_rejection}"
                )
            if executions >= _MAX_TOOL_EXECUTIONS and not loop["forced"]:
                loop["forced"] = "tool_cap"
                self._counters["agent_forced_tool_cap"] += 1

        # 4. Proof, then the response.
        notes, self._unreported_notes = self._unreported_notes, _new_notes()
        surface = self._surface_record(
            {SUBMIT_TOOL: submit_dropped} if submit_dropped else None
        )
        response = Response(
            action=answer,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "kaleidoscope_v2",
                "model": self.model,
                "ks2_auto_search": {
                    "key": key,
                    "status": status,
                    "served_ids": [hit["memory_id"] for hit in hits],
                    "served_count": len(hits),
                    "served_bytes": len("\n".join(lines).encode("utf-8")),
                },
                "ks2_agent_tools": agent_tools,
                "ks2_answer": loop,
                "ks2_notetaker": notes,
                "ks2_surface": surface,
                "ks_failures_total": self._failures_total,
                "ks_binary": dict(self._binary_info),
            },
        )
        self._last_query = query
        self._last_response = response
        return response

    # ---- the note-taker ----

    def _transcript_text(self) -> str:
        """The instance so far, oldest turn first, capped to its most recent
        ``_TRANSCRIPT_TOKENS`` tokens: whole turns leave oldest first, and a
        single turn still over the cap keeps only its end."""
        model = self.extraction_model
        turns = self._transcript
        for turn in turns:
            if turn["tokens"] is None:
                turn["block"] = _render_turn(turn)
                turn["tokens"] = _count(model, turn["block"])
        first = 0
        while (
            first < len(turns) - 1
            and sum(t["tokens"] + 2 for t in turns[first:]) + 32 > _TRANSCRIPT_TOKENS
        ):
            first += 1
        self._counters["transcript_turns_omitted"] += first
        blocks = [OMITTED.format(n=first)] if first else []
        text = "\n\n".join(blocks + [t["block"] for t in turns[first:]])
        if _count(model, text) > _TRANSCRIPT_TOKENS:
            text = _tail(model, text, _TRANSCRIPT_TOKENS - 16)
        return text

    def _run_notetaker(self) -> None:
        """One note-taker run after a turn's feedback. Never raises."""
        notes = self._unreported_notes
        notes["ran"] = True
        notes["runs"] += 1
        self._counters["notetaker_runs"] += 1
        try:
            self._take_notes(notes)
        except Exception as exc:
            notes["errors"] += 1
            self._counters["notetaker_internal_error"] += 1
            self._failures_total += 1
            logger.warning("note-taker failed: %s: %s", type(exc).__name__, exc)

    def _take_notes(self, notes: dict[str, Any]) -> None:
        status, hits = self._search(self._turn_key)
        self._counters["related_searches"] += 1
        self._counters[f"related_{status}"] += 1
        notes["related"].append(
            {"status": status, "served_ids": [hit["memory_id"] for hit in hits]}
        )
        if status not in v1._SEARCH_NON_FAILURES:
            notes["errors"] += 1
            notes["kscope_errors"] += 1
            self._failures_total += 1
        related = "\n\n".join(_related_entry(hit) for hit in hits) or "(none)"
        instructions = f"{NOTETAKER_PREAMBLE}\n\n{self._instructions}"
        pending: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"{TRANSCRIPT_HEADING}\n\n{self._transcript_text()}\n\n"
                f"{RELATED_HEADING}\n\n{related}\n\n{NOTETAKER_TASK}",
            }
        ]
        tools = [self._remember_tool]
        # A fresh chain for every run.
        self._notes_client.reset()
        executions = 0
        # Each call that returns tool calls uses at least one execution, so
        # this ends by the cap's own count, plus the call that says it is done.
        for _ in range(_MAX_NOTETAKER_EXECUTIONS + 1):
            try:
                response, _ = self._call_model(
                    "notetaker",
                    self._notes_client,
                    pending,
                    instructions,
                    tools,
                    "auto",
                )
            except Exception as exc:
                notes["errors"] += 1
                notes["model_errors"] += 1
                self._counters["notetaker_model_error"] += 1
                self._failures_total += 1
                logger.warning("notetaker_error: %s: %s", type(exc).__name__, exc)
                return
            pending = []
            notes["model_calls"] += 1
            self._counters["notetaker_model_calls"] += 1
            calls = _function_calls(response)
            if not calls:
                return
            for call in calls:
                name = call["name"]
                if executions >= _MAX_NOTETAKER_EXECUTIONS:
                    notes["not_executed"] += 1
                    self._counters["notetaker_not_executed"] += 1
                    result = NOT_EXECUTED
                elif name == "remember":
                    executions += 1
                    notes["calls"] += 1
                    result, outcome = self._execute(
                        "notetaker", name, call["arguments"]
                    )
                    notes["created_ids"] += outcome.get("created", [])
                    notes["updated_ids"] += outcome.get("updated", [])
                    notes["deleted_ids"] += outcome.get("deleted", [])
                    if outcome["outcome"] == "refused":
                        notes["refused"] += 1
                    elif outcome["outcome"] == "ok" and outcome["refused_items"]:
                        notes["partial"] += 1
                    elif outcome["outcome"] == "error":
                        notes["errors"] += 1
                        notes["kscope_errors"] += 1
                else:
                    executions += 1
                    notes["unknown_tools"] += 1
                    self._counters["notetaker_unknown_tools"] += 1
                    result = f"error: there is no tool named {name!r}"
                pending.append(_function_output(call["call_id"], result))
            if executions >= _MAX_NOTETAKER_EXECUTIONS:
                return

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        instance_complete = observation_marks_instance_complete(observation)
        content = observation.content.strip()

        if self._transcript and self._transcript[-1]["feedback"] is None:
            self._transcript[-1]["feedback"] = content
        if self._last_query is not None and self._last_response is not None:
            action = self._last_response.action
            action_str = (
                action.model_dump_json()
                if isinstance(action, BaseModel)
                else str(action)
            )
            self._search_key.record_turn(action_str, content)
            self._run_notetaker()
            self._last_query = None
            self._last_response = None
        elif self._transcript:
            # A turn whose answer failed: its feedback is on the runner's
            # fallback action, which this system never chose.
            self._unreported_notes["skipped"] += 1
            self._counters["notetaker_skipped"] += 1

        if content:
            self._pending.append({"role": "user", "content": f"FEEDBACK: {content}"})

        if instance_complete:
            # A fresh chain for the next instance.
            self._answer_client.reset()
            self._pending = []
            self._transcript = []
            self._search_key.end_instance()

    @property
    def name(self) -> str:
        return "kaleidoscope_v2"

    def get_run_artifacts(self) -> dict[str, Any] | None:
        memories = [dict(memory) for memory in self._memories.values()]
        return {
            "artifact_type": "kaleidoscope_v2",
            "memories": memories,
            "memory_count": sum(m.get("status") == "live" for m in memories),
            "deleted_count": sum(m.get("status") == "deleted" for m in memories),
            "counters": dict(self._counters),
            "ks_binary": dict(self._binary_info),
            "ks2_surface": self._surface_record(),
            "ks_failures_total": self._failures_total,
            # The note-taker runs after the last response (the final
            # observe's), which no response's metadata reports.
            "ks2_notetaker_unreported": copy.deepcopy(self._unreported_notes),
            # Which code ran: its version and this file's sha256.
            "ks2_code": {
                "version": _CODE_VERSION,
                "system_py_sha256": _SYSTEM_PY_SHA256,
            },
        }
