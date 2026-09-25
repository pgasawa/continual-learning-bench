"""
Kaleidoscope memory system for the continual learning benchmark.

It follows the ``mem0`` system turn for turn: the same system prompt, the same
message handling and token budget, the same ``RELEVANT MEMORIES`` frame, and
``top_k=10``. Before each answer it searches a private Kaleidoscope vault.
After each observation, one extraction call, made with the task model and
billed like any other call, turns the finished turn into memories, which are
written to the vault. ``reset`` starts a fresh, empty vault, so the no-memory
pass sees none.

It is not mem0 with only the memory backend swapped. The differences, all
disclosed:

- **Our extraction prompt.** mem0 extracts inside its SDK with its own prompt
  of about 33 KB. Kaleidoscope has no model of its own, so this system makes
  the extraction call itself, with the short ``EXTRACTION_SYSTEM_PROMPT`` below
  and the structured schema ``_Extraction``.
- **Create-only writes.** Memories are only ever created; duplicates are merged
  by kscope's own admission. mem0 can also update and delete.
- **The search key, adapted to kscope's limit.** kscope refuses a search over
  1,024 bytes, or one containing any control character, newline included.
  mem0 searches with the whole previous turn and the whole current prompt.
  Here the key is one line of at most 1,024 bytes that puts the distinctive
  part of the current prompt first; ``_SearchKey`` has the rule.
- **Plain-line injection.** Each served memory becomes one ``- <title>: <body>``
  line inside mem0's frame, built from the search's JSON hits. None of
  Kaleidoscope's display framing, tags or footers reaches the model.

How it runs:

- **A pinned binary.** ``kscope_binary`` is hashed when the system is built and
  must match ``kscope_sha256``. Its version and the first 16 hex digits of its
  hash are in every response's metadata (``ks_binary``).
- **A private environment.** Every ``kscope`` process gets exactly five
  variables: ``PATH``, ``HOME`` and ``TMPDIR`` (the vault's temp dir),
  ``KSCOPE_NO_RESIDENT=1``, and ``KSCOPE_ENTITLEMENT_HOME``, the vault's own
  copy of the licence taken from ``kscope_entitlement_src``. No provider key and
  no inherited variable reaches it.
- **kscope failures are recorded, never raised.** Each search and each write is
  classified and reported in the response metadata (``ks_search_status``,
  ``ks_writes``, ``ks_failures_total``), so a run with any failure can be
  rejected at scoring time instead of aborting a paid rollout. A degraded
  search is retried twice; a licence refusal (exit 4), on a search or a write,
  once after 2 s, and is recorded as ``licence`` only when the retry is
  refused too.
- **The answer call fails as in ICL.** A failed answer call raises
  ``RuntimeError("LLM call failed: answer: ...")``, and a provider refusal
  passes through untouched, exactly as in ``src/systems/icl/system.py``.
- **A failed extraction call is recorded, never raised.** The runner has no
  handler around ``observe``, so a raise there would end the whole rollout.
  The failure, a provider refusal included, is counted as
  ``extraction_error`` in ``ks_writes`` and in ``ks_failures_total``; that
  turn's memories are lost and the rollout goes on.
- **Configuration errors raise at construction**, before any model is called: a
  hash mismatch, a licence source without its two files, or a binary that
  cannot report its version.
- **Isolation.** Each vault lives in its own ``kscope_clbench_*`` temp dir,
  deleted by ``reset`` and, for the last one, when the process exits.
"""

import copy
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import weakref
from datetime import datetime, timezone
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field

from ...errors import ProviderRefusalError
from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ..utils import (
    TokenBudgetTracker,
    completion_with_structured_output,
)

logger = logging.getLogger(__name__)

litellm.drop_params = True

# Identical to the mem0 system's prompt, so the two systems differ only in
# which memory store fills the RELEVANT MEMORIES block.
RESPONSE_SYSTEM_PROMPT = """\
You are an agent working on a continual learning benchmark task. You have \
access to a memory store of facts extracted from past interactions. Use \
these memories to inform your response to the current query.

Memories are retrieved by relevance and may come from previous task \
instances. Apply any relevant patterns, strategies, or facts you find."""

# Generic on purpose: nothing here refers to any benchmark task.
EXTRACTION_SYSTEM_PROMPT = """\
You maintain a long-term memory for an agent that repeats related tasks.
You are shown one completed interaction turn: the query the agent received,
the action it took, and the feedback it got back.

Write down only what would help the agent do better on FUTURE, similar tasks:
durable facts about the environment, rules that were revealed, patterns in
how the environment or other parties behave, what worked, what failed and why.
Do not restate the query. Do not record one-off values that will not recur.
If this turn teaches nothing new, return an empty list.

Each memory must be self-contained and specific:
- title: one sentence that states the fact itself.
- body: two to five sentences with the specifics (names, numbers, conditions).
- facts: one to five (subject, predicate, object) triples. Every subject and
  object must also appear in entities. Use short predicates such as uses,
  causes, depends_on, part_of, avoids, measured_at, decided.
- entities: each with a short name, a kind, and a one-line gloss."""


class _Entity(BaseModel):
    name: str = Field(description="Short name used in facts")
    kind: Literal[
        "artifact",
        "capability",
        "concept",
        "doc",
        "event",
        "metric",
        "model",
        "org",
        "person",
        "place",
        "practice",
        "tool",
    ]
    gloss: str = Field(description="One-line description of what this is")


class _Fact(BaseModel):
    subject: str
    predicate: str
    object: str


class _Memory(BaseModel):
    title: str
    body: str
    memory_type: Literal["fact", "procedure", "outcome", "decision"]
    facts: list[_Fact]
    entities: list[_Entity]


class _Extraction(BaseModel):
    memories: list[_Memory] = Field(default_factory=list, max_length=3)


# Relations the store emits itself and refuses when authored.
_RESERVED_PREDICATES = {
    "contradicts",
    "justified_by",
    "not_same_as",
    "possibly_same_as",
    "same_as",
    "supersedes",
}

# ---- kscope's contract, as the pinned binary enforces it --------------------

# A search query over 1,024 bytes is refused (exit 2).
_KEY_BYTES = 1024
# The current prompt's share of the key, and the shortest distinctive part
# worth searching with before the prompt's tail is used instead.
_CURRENT_BYTES = 720
_MIN_DISTINCT_BYTES = 64
# A word of the current prompt inside any run of this many consecutive words
# that the reference prompt also has is shared, not distinctive; the gap
# marker (a space, U+2026 and a space) stands where shared words were dropped
# between two kept ones.
_SHINGLE_WORDS = 4
_GAP = " … "
# The pinned binary refuses a search containing any control character (Unicode
# category Cc: newline and tab, but also ESC, NUL, DEL and the C1 range), and
# a lone surrogate cannot be encoded at all.
_UNACCEPTED = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")

_VAULT_PREFIX = "kscope_clbench_"
_LICENCE_FILES = ("api-key", "verdict-v1.json")
_PATH = "/usr/bin:/bin"

# A degraded search is retried this many times before it is recorded.
_DEGRADED_RETRIES = 2
_DEGRADED_RETRY_DELAY_S = 0.2
# A licence refusal (exit 4) is retried once, on a search or a write: a
# refused check revalidates in the background, and the pinned binary's own
# exit-4 message says the next attempt often succeeds.
_LICENCE_RETRIES = 1
_LICENCE_RETRY_DELAY_S = 2.0

_SEARCH_STATUSES = (
    "ok",
    "empty",
    "degraded",
    "refused",
    "licence",
    "error",
    "no_vault",
)
_WRITE_STATUSES = ("ok", "partial", "error", "licence", "no_vault")
# Only these searches, and a write that exits 0, are not failures.
_SEARCH_NON_FAILURES = ("ok", "empty")


def _flat(text: str) -> str:
    """One line kscope accepts: control characters become spaces, then every
    run of whitespace collapses to one space."""
    return " ".join(_UNACCEPTED.sub(" ", text or "").split())


def _fit_bytes(text: str, limit: int) -> str:
    """Keep at most the first ``limit`` UTF-8 bytes, without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[: max(0, limit)].decode("utf-8", errors="ignore").rstrip()


def _fit_words(text: str, limit: int) -> str:
    """Keep at most the first ``limit`` UTF-8 bytes, cut at a space. A first
    word longer than ``limit`` is cut between two characters instead."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    cut = raw.rfind(b" ", 0, limit + 1)
    if cut <= 0:
        return _fit_bytes(text, limit)
    return raw[:cut].decode("utf-8").rstrip()


def _tail_bytes(text: str, limit: int) -> str:
    """Keep at most the last ``limit`` UTF-8 bytes, without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[len(raw) - max(0, limit) :].decode("utf-8", errors="ignore").lstrip()


def _distinct(cur: str, ref: str | None) -> str:
    """The words of ``cur`` that ``ref`` does not share, in their order.

    Both are flattened, so words are what single spaces separate. A word of
    ``cur`` is dropped when it lies inside any run of ``_SHINGLE_WORDS``
    consecutive words of ``cur`` that ``ref`` also has, anywhere; ``_GAP``
    stands wherever words were dropped between two kept ones.
    """
    if not ref:
        return cur
    words, ref_words, n = cur.split(), ref.split(), _SHINGLE_WORDS
    shared = {tuple(ref_words[i : i + n]) for i in range(len(ref_words) - n + 1)}
    dropped = [False] * len(words)
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) in shared:
            dropped[i : i + n] = [True] * n
    kept_runs = [
        " ".join(word for word, _ in run)
        for is_dropped, run in itertools.groupby(
            zip(words, dropped, strict=True), key=lambda pair: pair[1]
        )
        if not is_dropped
    ]
    return _GAP.join(kept_runs)


class _SearchKey:
    """The search key: the words that set the current prompt apart, first.

    ``cur`` is the current prompt, flattened, and ``ref`` the flattened prompt
    it is compared with: within an instance, the previous turn's prompt; at an
    instance's first turn, the previous instance's first prompt; at the run's
    first turn, none. ``distinct`` is ``cur`` without every word that lies in
    a run of four consecutive words ``ref`` also has, with `` … `` wherever
    words were dropped between two kept ones (``_distinct``); when that
    leaves fewer than 64 bytes, the last 1,024 bytes of ``cur`` stand in for
    it. The key is ``CURRENT: <distinct>``, then ``FEEDBACK: <previous
    feedback>``, then ``PREVIOUS: <previous response>``, all flattened;
    ``distinct`` is cut at a word boundary to at most 720 bytes, and so is the
    whole key to at most 1,024.

    ``ref`` reaches back across instances, because what prompts share does:
    every blind_spectrum_monitoring prompt of a run opens with the same 2,199
    bytes, so a key cut from the start of the prompt was the same for all 90.
    An instance's first prompt is compared with the previous instance's first
    prompt, which is built from the same frame, not with the previous
    instance's last turn, which is not. Against the last turn nearly the whole
    first prompt counted as distinct, so the key kept only the fixed text at
    the prompt's start: replaying the published mem0-gpt-5.4 prompts, every
    rollout had 2 distinct first-turn keys for 19 instances of
    codebase_adaptation, 1 for 20 of cohort_studies and 1 for 12 of
    sales_prediction.

    Shared runs are dropped wherever they are, not only as a shared prefix and
    suffix, because prompts differ in several places with shared text between.
    Cut from its first difference to its last, a sales_prediction prompt kept
    one digit of its year and then shared boilerplate, and a
    codebase_adaptation prompt after a one-off opening line shared no prefix
    with the next and kept the fixed brief: in every rollout, 2 of 12 and 2 of
    19 first-turn keys were shared. Replaying the same prompts with this rule,
    no first-turn key of a rollout is shared, in any task (T6).

    The previous feedback and response belong to the previous turn of the
    current instance and are cleared when an instance completes, as mem0
    clears the previous turn it searches with. No rule here refers to any task.
    """

    def __init__(self) -> None:
        self._prev: str | None = None
        self._first: str | None = None
        self._first_turn = True
        self._feedback = ""
        self._response = ""

    def build(self, prompt: str) -> str:
        cur = _flat(prompt)
        if self._first_turn:
            ref, self._first = self._first, cur
            self._first_turn = False
        else:
            ref = self._prev
        distinct = _distinct(cur, ref)
        if len(distinct.encode("utf-8")) < _MIN_DISTINCT_BYTES:
            distinct = _tail_bytes(cur, _KEY_BYTES)
        self._prev = cur
        parts = [f"CURRENT: {_fit_words(distinct, _CURRENT_BYTES)}"]
        if self._feedback:
            parts.append(f"FEEDBACK: {self._feedback}")
        if self._response:
            parts.append(f"PREVIOUS: {self._response}")
        return _fit_words(" ".join(parts), _KEY_BYTES)

    def record_turn(self, response: str, feedback: str) -> None:
        """Keep a finished turn's response and feedback for the next key."""
        self._response = _fit_bytes(_flat(response), _KEY_BYTES)
        self._feedback = _fit_bytes(_flat(feedback), _KEY_BYTES)

    def end_instance(self) -> None:
        """Forget the previous turn; the next prompt is an instance's first."""
        self._response = ""
        self._feedback = ""
        self._first_turn = True


def _first_line(text: str | None) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def _refusal_message(proc: subprocess.CompletedProcess) -> str:
    """The message of a refusal envelope on stdout, else stderr's first line."""
    try:
        message = json.loads(proc.stdout).get("message")
    except Exception:
        message = None
    return str(message)[:200] if message else _first_line(proc.stderr)


def _private_env(home: str) -> dict[str, str]:
    """The whole environment of a kscope process. Nothing is inherited."""
    return {
        "PATH": _PATH,
        "HOME": home,
        "TMPDIR": home,
        "KSCOPE_NO_RESIDENT": "1",
        "KSCOPE_ENTITLEMENT_HOME": os.path.join(home, "entitlement"),
    }


def _run_kscope(
    binary: str,
    args: list[str],
    home: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
) -> subprocess.CompletedProcess:
    """Run one kscope process in the private environment rooted at ``home``."""
    return subprocess.run(
        [binary, *args],
        input=None if payload is None else json.dumps(payload),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        env=_private_env(home),
    )


def _check_binary(path: str, expected_sha256: str) -> str:
    """Hash the binary and raise ValueError unless it is the pinned one."""
    expected = (expected_sha256 or "").strip().lower()
    if not path:
        raise ValueError(
            "kscope_binary is required: the path of the pinned kscope executable"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(
            "kscope_sha256 is required: the pinned binary's SHA-256, 64 hex digits"
        )
    try:
        with open(path, "rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError(f"kscope_binary {path!r} cannot be read: {exc}") from exc
    if actual != expected:
        raise ValueError(
            f"kscope_binary {path!r} has sha256 {actual}, not the pinned {expected}"
        )
    return actual


def _check_licence_source(path: str) -> str:
    """Require a licence source holding both files; only their presence is checked."""
    if not path:
        raise ValueError(
            "kscope_entitlement_src is required: the directory holding "
            + " and ".join(_LICENCE_FILES)
        )
    missing = [
        name for name in _LICENCE_FILES if not os.path.isfile(os.path.join(path, name))
    ]
    if missing:
        raise ValueError(f"kscope_entitlement_src {path!r} lacks {', '.join(missing)}")
    return path


def _read_version(binary: str, timeout_s: float) -> str:
    """``kscope --version``, run once, in a private environment of its own."""
    with tempfile.TemporaryDirectory(prefix=_VAULT_PREFIX) as home:
        proc = _run_kscope(binary, ["--version"], home, timeout_s)
    version = (proc.stdout or "").strip()
    if proc.returncode != 0 or not version:
        raise RuntimeError(
            f"{binary} --version exited {proc.returncode}: {_first_line(proc.stderr)}"
        )
    return version


class _Vault:
    """One private vault in its own temp dir.

    The dir holds the vault, the vault's own copy of the licence, and is HOME
    and TMPDIR for every kscope process run against it. It is deleted by
    ``close``, when the object is collected, or when the process exits,
    whichever comes first.
    """

    def __init__(self, binary: str, timeout_s: float, licence_source: str):
        self.binary = binary
        self.timeout_s = timeout_s
        self.dir = tempfile.mkdtemp(prefix=_VAULT_PREFIX)
        self._delete = weakref.finalize(
            self, shutil.rmtree, self.dir, ignore_errors=True
        )
        try:
            self._copy_licence(licence_source)
            created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            proc = self._run(
                ["init", os.path.join(self.dir, "vault"), created, "process-local"]
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"kscope init exited {proc.returncode}: {_first_line(proc.stderr)}"
                )
            ids = json.loads(proc.stdout)
            self.address = [
                ids["root"],
                ids["workspace_id"],
                ids["principal_id"],
                ids["journal"],
            ]
        except BaseException:
            self.close()
            raise

    def _copy_licence(self, source: str) -> None:
        """Copy the licence into a 0700 folder of this vault's own. Never read out."""
        target = os.path.join(self.dir, "entitlement")
        os.mkdir(target, 0o700)
        os.chmod(target, 0o700)
        for name in _LICENCE_FILES:
            copied = os.path.join(target, name)
            shutil.copyfile(os.path.join(source, name), copied)
            os.chmod(copied, 0o600)

    def _run(
        self, args: list[str], payload: dict[str, Any] | None = None
    ) -> subprocess.CompletedProcess:
        return _run_kscope(self.binary, args, self.dir, self.timeout_s, payload)

    def call(
        self, operation: str, payload: dict[str, Any]
    ) -> subprocess.CompletedProcess:
        return self._run(["call", *self.address, operation, "--json"], payload)

    def close(self) -> None:
        self._delete()


def _is_hit(hit: Any) -> bool:
    return (
        isinstance(hit, dict)
        and isinstance(hit.get("memory_id"), str)
        and isinstance(hit.get("content_md"), str)
    )


def _memory_line(hit: dict[str, Any]) -> str:
    """One served memory as mem0 shows one: ``- <title>: <body>``, on one line.

    Every memory is written with markdown ``# <title>`` then the body, and a
    search hit carries that markdown back, so the title is read off our own
    heading.
    """
    first, _, rest = hit["content_md"].strip().partition("\n")
    if first.startswith("# "):
        title, body = _flat(first[2:]), _flat(rest)
    else:
        title, body = "", _flat(hit["content_md"])
    return f"- {title}: {body}" if title and body else f"- {title or body}"


def _new_counters() -> dict[str, int]:
    counters = {
        "searches": 0,
        "search_degraded_retries": 0,
        "search_licence_retries": 0,
        "remember_calls": 0,
        "remember_licence_retries": 0,
        "memories_committed": 0,
        "memories_consolidated": 0,
        "memories_replayed": 0,
        "items_refused": 0,
        "items_unfolded": 0,
        "extractions_empty": 0,
        "extraction_error": 0,
        "vaults_failed": 0,
    }
    counters.update({f"search_{status}": 0 for status in _SEARCH_STATUSES})
    counters.update({f"remember_{status}": 0 for status in _WRITE_STATUSES})
    return counters


def _new_writes() -> dict[str, Any]:
    """The outcomes of the extractions and remember calls since the previous response.

    ``calls`` counts remember calls by status. ``items`` counts their items:
    ``committed`` (new memories, whose ids are ``committed_ids``),
    ``consolidated`` (merged into a memory the vault holds), ``replayed`` (an
    item sent again, which kscope answers with the id it already committed and
    does not write), ``refused`` and ``unfolded``. ``extractions_empty``
    counts extractions that found nothing to write, and ``extraction_error``
    extraction calls that failed.
    """
    return {
        "calls": {status: 0 for status in _WRITE_STATUSES},
        "items": {
            "committed": 0,
            "consolidated": 0,
            "replayed": 0,
            "refused": 0,
            "unfolded": 0,
        },
        "committed_ids": [],
        "extractions_empty": 0,
        "extraction_error": 0,
    }


@register_system("kaleidoscope")
class KaleidoscopeSystem(ContinualLearningSystem):
    """
    Kaleidoscope-backed memory system, structured like ``mem0``.

    After each observation the finished turn is distilled into memories and
    written with ``kscope call ... remember``. At query time ``kscope call ...
    search`` serves the relevant memories, which are injected into the prompt.
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
            extraction_model: LiteLLM model that writes memories (defaults to model).
            system_prompt: Additional system prompt content.
            top_k: Memories requested per search (mem0's default is 10).
            kscope_binary: Path of the pinned kscope executable. Required; there
                is no PATH fallback.
            kscope_sha256: The binary's expected SHA-256. A mismatch raises
                ValueError before anything else happens.
            kscope_entitlement_src: Directory holding ``api-key`` and
                ``verdict-v1.json``. Each vault gets its own copy.
            kscope_timeout_s: Per-process timeout for kscope.
        """
        self.kscope_sha256 = _check_binary(kscope_binary, kscope_sha256)
        self.kscope_binary = kscope_binary
        self.kscope_entitlement_src = _check_licence_source(kscope_entitlement_src)
        self.kscope_timeout_s = kscope_timeout_s
        self._binary_info = {
            "version": _read_version(kscope_binary, kscope_timeout_s),
            "sha256_16": self.kscope_sha256[:16],
        }

        self.model = model
        self.extraction_model = extraction_model or model
        self.user_system_prompt = system_prompt
        self.top_k = top_k

        self._token_budget = TokenBudgetTracker()
        self._vault: _Vault | None = None
        self._clear_state()
        self._vault = self._open_vault()

    # ---- state and vaults ----

    def _clear_state(self) -> None:
        """Everything ``reset`` zeroes, apart from the vault itself."""
        self.messages: list[dict[str, str]] = []
        self._token_budget.reset()
        self.interaction_count = 0
        self._last_query: Query | None = None
        self._last_response: Response | None = None
        self._search_key = _SearchKey()
        self._written: list[dict[str, str]] = []
        self._counters = _new_counters()
        self._failures_total = 0
        self._unreported_writes = _new_writes()

    def _open_vault(self) -> _Vault | None:
        """A fresh vault. A failure is recorded as ``vault_failed``, never raised,
        and every later call reports ``no_vault`` until the next reset."""
        try:
            return _Vault(
                self.kscope_binary, self.kscope_timeout_s, self.kscope_entitlement_src
            )
        except Exception as exc:
            self._counters["vaults_failed"] += 1
            self._failures_total += 1
            logger.warning("kscope vault_failed: %s: %s", type(exc).__name__, exc)
            return None

    def _close_vault(self) -> None:
        if self._vault is not None:
            self._vault.close()
            self._vault = None

    # ---- model calls ----

    @staticmethod
    def _complete(
        kind: str, model: str, messages: list[dict[str, str]], schema: type[BaseModel]
    ):
        """One model call, failing exactly as the ICL system does: a provider
        refusal passes through, and anything else becomes ``RuntimeError("LLM
        call failed: <kind>: ...")``. The answer call lets either reach the
        runner; the extraction call records it (``_extract_and_remember``)."""
        try:
            return completion_with_structured_output(
                model=model,
                messages=messages,
                response_schema=schema,
            )
        except ProviderRefusalError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"LLM call failed: {kind}: {type(exc).__name__}: {exc}"
            ) from exc

    # ---- retrieval ----

    def _search(self, key: str) -> tuple[str, list[dict[str, Any]]]:
        """One search, retried while degraded and once after a licence refusal.
        Returns the status and the selected hits; never raises."""
        if self._vault is None:
            return "no_vault", []
        payload = {"query": key, "top_k": self.top_k}
        degraded_retries = licence_retries = 0
        while True:
            status, hits, detail = self._search_once(payload)
            if status == "degraded" and degraded_retries < _DEGRADED_RETRIES:
                degraded_retries += 1
                self._counters["search_degraded_retries"] += 1
                time.sleep(_DEGRADED_RETRY_DELAY_S)
            elif status == "licence" and licence_retries < _LICENCE_RETRIES:
                licence_retries += 1
                self._counters["search_licence_retries"] += 1
                time.sleep(_LICENCE_RETRY_DELAY_S)
            else:
                break
        if status not in _SEARCH_NON_FAILURES:
            logger.warning("kscope search %s: %s", status, detail)
        return status, hits

    def _search_once(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], str]:
        try:
            proc = self._vault.call("search", payload)
        except subprocess.TimeoutExpired:
            return "error", [], f"timed out after {self.kscope_timeout_s}s"
        except Exception as exc:
            return "error", [], f"{type(exc).__name__}: {exc}"
        if proc.returncode == 2:
            return "refused", [], _refusal_message(proc)
        if proc.returncode == 4:
            return "licence", [], _first_line(proc.stderr)
        if proc.returncode != 0:
            return "error", [], f"exit {proc.returncode}: {_first_line(proc.stderr)}"
        try:
            body = json.loads(proc.stdout)
        except ValueError:
            return "error", [], "the response is not JSON"
        status = body.get("status") if isinstance(body, dict) else None
        if status == "degraded":
            return "degraded", [], str(body.get("reason") or "")[:200]
        hits = body.get("selected_hits") if status == "compiled" else None
        if not isinstance(hits, list) or not all(_is_hit(hit) for hit in hits):
            return "error", [], f"unexpected response with status {status!r}"
        return ("ok" if hits else "empty"), hits, ""

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1
        # Set again only once this turn is answered, so a failed answer is
        # never paired with the next observation's feedback.
        self._last_query = None
        self._last_response = None

        key = self._search_key.build(query.prompt)
        status, hits = self._search(key)
        self._counters["searches"] += 1
        self._counters[f"search_{status}"] += 1
        if status not in _SEARCH_NON_FAILURES:
            self._failures_total += 1
        lines = [_memory_line(hit) for hit in hits]

        parts = []
        if lines:
            parts.append(
                "=== RELEVANT MEMORIES ===\n"
                + "\n".join(lines)
                + "\n========================="
            )
        parts.append(query.prompt)
        query_content = "\n\n".join(parts)

        self._add_message("user", query_content)

        llm_messages = self.messages.copy()
        sys_parts = [RESPONSE_SYSTEM_PROMPT]
        if self.user_system_prompt:
            sys_parts.append(self.user_system_prompt)
        llm_messages.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})

        parsed, usage_event = self._complete(
            "answer", self.model, llm_messages, query.response_schema
        )
        assistant_record = parsed.model_dump_json()

        if usage_event is not None:
            self._note_prompt_token_usage(usage_event.input_tokens)
            self.record_usage_event(usage_event)

        self._add_message("assistant", assistant_record)

        writes, self._unreported_writes = self._unreported_writes, _new_writes()
        response = Response(
            action=parsed,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "kaleidoscope",
                "model": self.model,
                "ks_search_key": key,
                "ks_search_status": status,
                "ks_served_ids": [hit["memory_id"] for hit in hits],
                "ks_served_count": len(hits),
                "ks_served_bytes": len("\n".join(lines).encode("utf-8")),
                "ks_writes": writes,
                "ks_failures_total": self._failures_total,
                "ks_binary": dict(self._binary_info),
            },
        )
        self._last_query = query
        self._last_response = response
        return response

    # ---- writing ----

    def _extract_and_remember(self, messages: list[dict[str, str]]) -> None:
        transcript = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        extraction_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        try:
            parsed, usage_event = self._complete(
                "extraction", self.extraction_model, extraction_messages, _Extraction
            )
        except Exception as exc:
            # Raised from observe(), this would end the rollout: the runner
            # has no handler around observe. It costs this turn's memories.
            self._counters["extraction_error"] += 1
            self._unreported_writes["extraction_error"] += 1
            self._failures_total += 1
            logger.warning("extraction_error: %s: %s", type(exc).__name__, exc)
            return
        if usage_event is not None:
            self.record_usage_event(usage_event)

        items = []
        for mem in parsed.memories:
            payload = self._to_remember_payload(mem)
            if payload is not None:
                items.append(payload)
        if not items:
            self._counters["extractions_empty"] += 1
            self._unreported_writes["extractions_empty"] += 1
            return
        self._remember(items)

    def _remember(self, items: list[dict[str, Any]]) -> None:
        """Create ``items`` in one call and record the outcome; never raises."""
        self._counters["remember_calls"] += 1
        if self._vault is None:
            self._record_write("no_vault")
            return
        for attempt in range(_LICENCE_RETRIES + 1):
            try:
                proc = self._vault.call("remember", {"mode": "create", "items": items})
            except subprocess.TimeoutExpired:
                self._record_write(
                    "error", detail=f"timed out after {self.kscope_timeout_s}s"
                )
                return
            except Exception as exc:
                self._record_write("error", detail=f"{type(exc).__name__}: {exc}")
                return
            if proc.returncode != 4 or attempt == _LICENCE_RETRIES:
                break
            self._counters["remember_licence_retries"] += 1
            time.sleep(_LICENCE_RETRY_DELAY_S)
        if proc.returncode == 4:
            self._record_write("licence", detail=_first_line(proc.stderr))
            return
        if proc.returncode not in (0, 3):
            self._record_write(
                "error", detail=f"exit {proc.returncode}: {_refusal_message(proc)}"
            )
            return
        try:
            results = json.loads(proc.stdout)["results"]
            if not isinstance(results, list):
                raise TypeError("results is not a list")
        except Exception:
            self._record_write(
                "error", detail=f"exit {proc.returncode} with an unreadable response"
            )
            return

        committed: list[str] = []
        consolidated = replayed = refused = unfolded = 0
        for index, outcome in enumerate(results):
            outcome = outcome if isinstance(outcome, dict) else {}
            memory_id = outcome.get("memory_id")
            effect = outcome.get("canonical_effect")
            if effect == "no_write":
                consolidated += 1
            elif effect == "replayed":
                # An item sent before: kscope answers with the id it committed
                # then, with memory_kept true, and writes nothing.
                replayed += 1
            elif effect == "committed" and isinstance(memory_id, str) and memory_id:
                committed.append(memory_id)
                unfolded += outcome.get("status") == "unfolded"
                position = outcome.get("item_index")
                if not isinstance(position, int) or not 0 <= position < len(items):
                    position = min(index, len(items) - 1)
                item = items[position]
                self._written.append(
                    {
                        "memory_id": memory_id,
                        "title": item["semantic_delta"]["title"],
                        "body": item["content_md"],
                    }
                )
            else:
                refused += 1
        refused += max(0, len(items) - len(results))
        self._record_write(
            "ok" if proc.returncode == 0 else "partial",
            committed,
            consolidated,
            replayed,
            refused,
            unfolded,
        )

    def _record_write(
        self,
        status: str,
        committed: list[str] | tuple[str, ...] = (),
        consolidated: int = 0,
        replayed: int = 0,
        refused: int = 0,
        unfolded: int = 0,
        detail: str = "",
    ) -> None:
        writes = self._unreported_writes
        writes["calls"][status] += 1
        writes["items"]["committed"] += len(committed)
        writes["items"]["consolidated"] += consolidated
        writes["items"]["replayed"] += replayed
        writes["items"]["refused"] += refused
        writes["items"]["unfolded"] += unfolded
        writes["committed_ids"].extend(committed)
        self._counters[f"remember_{status}"] += 1
        self._counters["memories_committed"] += len(committed)
        self._counters["memories_consolidated"] += consolidated
        self._counters["memories_replayed"] += replayed
        self._counters["items_refused"] += refused
        self._counters["items_unfolded"] += unfolded
        if status != "ok":
            self._failures_total += 1
            logger.warning(
                "kscope remember %s: %s",
                status,
                detail or f"{refused} item(s) not written",
            )

    @staticmethod
    def _to_remember_payload(mem: "_Memory") -> dict[str, Any] | None:
        title = _fit_bytes(_flat(mem.title), 300)
        if not title:
            return None
        entities = {}
        for ent in mem.entities:
            name = _fit_bytes(_flat(ent.name), 120)
            if name and name not in entities:
                entities[name] = {
                    "n": name,
                    "kind": ent.kind,
                    "is": _fit_bytes(_flat(ent.gloss), 300) or name,
                }
        facts = []
        for fact in mem.facts:
            pred = _fit_bytes(_flat(fact.predicate).lower().replace(" ", "_"), 60)
            subj = _fit_bytes(_flat(fact.subject), 120)
            obj = _fit_bytes(_flat(fact.object), 120)
            if not pred or pred in _RESERVED_PREDICATES:
                continue
            for surface in (subj, obj):
                if surface and surface not in entities:
                    entities[surface] = {"n": surface, "kind": "concept", "is": surface}
            if subj and obj:
                facts.append({"subject": subj, "predicate": pred, "object": obj})
        if not facts:
            return None
        return {
            "content_md": f"# {title}\n\n{mem.body.strip()}",
            "semantic_delta": {
                "memory_type": mem.memory_type,
                "title": title,
                "facts": facts[:32],
                "entities": list(entities.values())[:32],
            },
        }

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        instance_complete = observation_marks_instance_complete(observation)
        content = observation.content.strip()

        if self._last_query is not None and self._last_response is not None:
            action = self._last_response.action
            action_str = (
                action.model_dump_json()
                if isinstance(action, BaseModel)
                else str(action)
            )
            messages = [
                {"role": "user", "content": self._last_query.prompt},
                {"role": "assistant", "content": action_str},
            ]
            if content:
                messages.append({"role": "user", "content": f"FEEDBACK: {content}"})
            self._extract_and_remember(messages)
            self._search_key.record_turn(action_str, content)

        if content:
            self._add_message("user", f"FEEDBACK: {content}")

        if instance_complete:
            self.messages = []
            self._search_key.end_instance()

    def reset(self) -> None:
        self._close_vault()
        self._clear_state()
        self._vault = self._open_vault()

    @property
    def name(self) -> str:
        return "kaleidoscope"

    def get_run_artifacts(self) -> dict[str, Any] | None:
        return {
            "artifact_type": "kaleidoscope",
            "memories": list(self._written),
            "memory_count": len(self._written),
            "counters": dict(self._counters),
            "ks_binary": dict(self._binary_info),
            "ks_failures_total": self._failures_total,
            # Writes made after the last response (by the final observe), which
            # no response's metadata reports.
            "ks_writes_unreported": copy.deepcopy(self._unreported_writes),
        }

    # ---- internal (identical to mem0) ----

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        if not messages:
            return 0
        try:
            return litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            total = sum(4 + len(m.get("content", "")) // 4 for m in messages) + 2
            return total

    def _count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        return self._token_budget.count(messages, self._estimate_message_tokens)

    def _note_prompt_token_usage(self, input_tokens: int | None) -> None:
        self._token_budget.note_usage(
            messages=self.messages,
            input_tokens=input_tokens,
            estimate_fn=self._estimate_message_tokens,
        )
