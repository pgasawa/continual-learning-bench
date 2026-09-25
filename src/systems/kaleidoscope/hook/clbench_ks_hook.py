"""Start-up hook for the Kaleidoscope experiment on Continual Learning Bench.

``sitecustomize.py`` in this directory imports this module and calls
``install()`` when ``CLBENCH_KS_HOOK=1``. It does five things and nothing else.

1. **Pins Luna's model entry.** It sets ``LITELLM_LOCAL_MODEL_COST_MAP=True``
   before litellm is imported, so litellm reads the price map bundled with it
   instead of downloading the live one with a 5-second timeout. The bundled map
   has no Luna, so the hook checks ``luna_model_entry.json`` against
   ``CLBENCH_KS_MODEL_ENTRY_SHA256``, registers its two entries with
   ``litellm.register_model``, and asserts that
   ``litellm.get_model_info("gpt-5.6-luna")`` reports the pinned costs and a
   positive context window. litellm is imported with ``LITELLM_MODE=PRODUCTION``
   set for the import only, so this early import does not load a ``.env`` found
   above the working directory, which the benchmark would never have read.
2. **Sets reasoning effort high on both routes.** It wraps ``litellm.completion``
   and ``litellm.responses``, the two attributes the benchmark calls
   (``src/systems/utils/structured_output.py`` and ``provider_adapters.py``;
   nothing under ``src/`` calls ``acompletion`` or ``aresponses``). For a model
   whose name ends with ``gpt-5.6-luna``, completion gets
   ``reasoning_effort="high"`` and responses gets
   ``reasoning={**existing, "effort": "high"}``. Nothing else about a call
   changes.
3. **Logs every attempt of every wrapped call**, Luna or not, as one JSON line
   appended to ``$CLBENCH_KS_CALL_LOG_DIR/calls-<pid>.jsonl`` once litellm
   returns or raises:

   - ``ts``: UTC time the attempt was made
   - ``attempt``: 1 for a call's first attempt, 2 for its first retry, and so
     on
   - ``retry_wait_s``: the seconds slept before this attempt, 0 for the first
   - ``pid``, ``route`` (``completion`` or ``responses``), ``model`` as passed
   - ``reasoning_effort`` (completion) or ``reasoning`` (responses): the value
     handed to litellm after the hook set it, or null
   - ``ok``, and when litellm raised, ``error`` (the exception's class name)
     and ``status_code``
   - ``response_id``: the returned response's id, which is the id the
     benchmark's usage event records, so each usage event can be matched to
     its line
   - ``response_model``: the model the provider named in its response body,
     so the line shows which deployment served the call, not only which
     name the call asked for; null when the response names none or litellm
     raised
   - ``usage``: the response's token counts, read the way src/usage.py reads
     them, as ``input_tokens``, ``cached_input_tokens``, ``output_tokens`` and
     ``reasoning_tokens`` (each null when the response lacks it); null when
     the response has no usage or litellm raised. A response the benchmark
     discards and re-sends (a 200 that fails to parse) records no usage event
     but is billed, and this is what prices it.
   - ``headers``: the provider's rate-limit and request-id headers, and no
     other: every ``x-ratelimit-*``, ``retry-after``, ``retry-after-ms``,
     ``x-request-id`` and ``apim-request-id`` present, names lower-cased,
     taken from the response, or from the exception when litellm raised.
     Under ``litellm.completion`` the OpenAI SDK retries a 429 itself unless
     the server sends ``x-should-retry: false`` (three requests for one call,
     probed on 1.81.6), so a failed completion line can stand for several 429
     responses and carries the last one's headers.

   Nothing of the request or the response body is logged: no prompt, no
   output, no key.

   A line that cannot be written raises ``CallLogError`` from the call, and
   the call is not retried.
4. **Fails closed.** Any exception during ``install()`` writes one line to
   stderr and ends the process with ``os._exit(78)``. Python would otherwise
   print a sitecustomize error and carry on without the hook.
5. **Retries a Luna call that failed for a transient reason**, inside item 2's
   wrapper, before the benchmark sees the failure. Azure's limit for the
   deployment (500 requests and 500,000 tokens a minute) must delay a call,
   never fail it, and the same way on both routes and for both arms; the
   benchmark's own loops give up after six tries (about eight minutes).

   - **Transient** means the call got no HTTP response because it timed out
     (``httpx.TimeoutException``) or its connection failed or dropped
     (``httpx.NetworkError``, ``httpx.RemoteProtocolError``), or the response's
     status was 429, 500, 502, 503 or 504. The status is the one httpx
     received, the ``httpx.HTTPStatusError`` in the exception's chain, not
     litellm's class: litellm 1.81.6 raises ``RateLimitError`` for any error
     whose text says "rate limit" or a bare "429", a 400 included, raises
     ``Timeout`` for a 504 and ``InternalServerError`` for a refused
     connection (probed on both routes). Everything else is raised at once:
     every other 4xx, 408 included, a 200 that litellm could not read, and any
     exception that is not an ``Exception``.
   - **The wait** before the next attempt is the failed response's
     ``retry-after-ms``, else its ``retry-after`` (seconds, or an HTTP date),
     read from ``exc.litellm_response_headers`` as the call log reads them.
     ``retry-after-ms`` comes first, as in the OpenAI SDK: in the 2026-09-23
     pilot Azure sent both on every 429, and on the completion route they
     disagreed (``retry-after`` 30, ``retry-after-ms`` 19 to 6664). A server's
     wait is honoured as sent, above 60 s too. With neither header, the wait is
     exponential backoff with full jitter, uniform on
     [0, min(60, 5 x 2^(n-1))] seconds before the n-th retry, drawn from the
     OS so the benchmark's seeded ``random`` state never moves. Every wait is
     rounded to the millisecond, and the value logged is the value slept.
   - **Caps**, per logical call (one call of the wrapped function): at most
     ``CLBENCH_KS_RETRY_MAX_WAIT_S`` seconds of waiting in total (default 600),
     so a wait that would pass it is not slept, and at most
     ``CLBENCH_KS_RETRY_MAX_ATTEMPTS`` attempts (default 30). At either cap the
     last attempt's exception is re-raised unchanged, and the benchmark's own
     handling applies; each of its own retries is a new logical call here.
   - Every attempt passes the same arguments, effort included. A call to any
     other model is made once, as before.

   Beneath this, under ``litellm.completion`` only, the OpenAI SDK retries a
   408, 409, 429, 5xx, timeout or failed connection twice itself, unless the
   server sends ``x-should-retry: false``, waiting what the server asks for
   when that is 60 s or less and at most 8 s otherwise; ``litellm.responses``
   sends one request per attempt (both probed on 1.81.6). So one completion
   attempt can be three requests, and on that route a call can wait longer
   in total than the cap.

Environment, required when the hook is on:

- ``CLBENCH_KS_MODEL_ENTRY_SHA256``: sha256 of ``luna_model_entry.json``
- ``CLBENCH_KS_CALL_LOG_DIR``: an existing absolute directory

and optional, each read once by ``install()``; a value that is set but is not
valid (an empty one included) stops the process like any other set-up error:

- ``CLBENCH_KS_RETRY_MAX_WAIT_S``: a finite number of seconds, 0 or more
  (default 600)
- ``CLBENCH_KS_RETRY_MAX_ATTEMPTS``: a whole number, 1 or more (default 30;
  1 turns retrying off)

The effort, pricing and retry behaviour was verified against litellm 1.81.6 (the
checkout's pin); any other installed version stops the hook until it has been
re-verified and ``VERIFIED_LITELLM_VERSION`` changed.

``python clbench_ks_hook.py build <map.json> <url> <fetched_at> [<etag>]`` writes
``luna_model_entry.json`` from one downloaded copy of litellm's live model map.
"""

from __future__ import annotations

import copy
import email.utils
import functools
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping

EXIT_CODE = 78
EFFORT = "high"
LUNA_SUFFIX = "gpt-5.6-luna"
LUNA_KEY = "gpt-5.6-luna"
ENTRY_KEYS = ("gpt-5.6-luna", "azure/gpt-5.6-luna")
SOURCE_KEY = "_source"
EXPECTED_COSTS = {
    "input_cost_per_token": 2e-07,
    "cache_read_input_token_cost": 2e-08,
    "output_cost_per_token": 1.2e-06,
}
VERIFIED_LITELLM_VERSION = "1.81.6"
ENTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "luna_model_entry.json"
)

# The keyword each route's effort travels in, and where `model` sits when it
# is passed positionally: litellm.completion(model, ...) and
# litellm.responses(input, model, ...).
_EFFORT_FIELD = {"completion": "reasoning_effort", "responses": "reasoning"}
_MODEL_POSITION = {"completion": 0, "responses": 1}

# Token counts, in the order src/usage.py (_resolve_usage_counts and
# _resolve_usage_details) tries them, restricted to the two shapes litellm
# 1.81.6 returns here: litellm.types.utils.Usage on the completion route
# (prompt_tokens, prompt_tokens_details.cached_tokens, ...) and
# ResponseAPIUsage on the responses route (input_tokens,
# input_tokens_details.cached_tokens, ...).
_USAGE_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "input_tokens": (("prompt_tokens",), ("input_tokens",)),
    "cached_input_tokens": (
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ),
    "output_tokens": (("completion_tokens",), ("output_tokens",)),
    "reasoning_tokens": (
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ),
}
# The only response headers the log keeps.
_HEADER_PREFIX = "x-ratelimit-"
_HEADER_NAMES = frozenset(
    {"retry-after", "retry-after-ms", "x-request-id", "apim-request-id"}
)

# Retrying (item 5 of the module docstring).
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 60.0
DEFAULT_MAX_WAIT_S = 600.0
DEFAULT_MAX_ATTEMPTS = 30

_log_dir: str | None = None
_log_lock = threading.Lock()
_installed = False
_max_wait_s = DEFAULT_MAX_WAIT_S
_max_attempts = DEFAULT_MAX_ATTEMPTS
# Jitter comes from the OS, never from the random module's shared generator,
# which the benchmark may have seeded.
_jitter = random.SystemRandom()


class HookError(RuntimeError):
    """Set-up found something wrong; ``install()`` turns it into exit 78."""


class CallLogError(RuntimeError):
    """A call-log line could not be written.

    Deliberately not an OSError: completion_with_structured_output retries
    OSError, and would pay for the same call up to five more times.
    """


def entries_sha256(entries: dict[str, Any]) -> str:
    """Hash of the model entries, independent of the file's formatting."""
    blob = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_entries(path: str, expected_sha256: str) -> dict[str, dict[str, Any]]:
    """Read the pinned entries, refusing a file whose bytes are not the pinned ones."""
    with open(path, "rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise HookError(
            f"{os.path.basename(path)} has sha256 {actual}, "
            f"CLBENCH_KS_MODEL_ENTRY_SHA256 is {expected_sha256}"
        )
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or set(document) != {SOURCE_KEY, *ENTRY_KEYS}:
        raise HookError(
            f"{os.path.basename(path)} must hold exactly {[SOURCE_KEY, *ENTRY_KEYS]}"
        )
    entries = {key: document[key] for key in ENTRY_KEYS}
    if not all(isinstance(entry, dict) for entry in entries.values()):
        raise HookError(f"{os.path.basename(path)}: every entry must be an object")
    source = document[SOURCE_KEY]
    recorded = source.get("entries_sha256") if isinstance(source, dict) else None
    if recorded != entries_sha256(entries):
        raise HookError(
            f"{os.path.basename(path)}: entries do not match _source.entries_sha256"
        )
    return entries


def _pin_model_entry(litellm: Any, entries: dict[str, dict[str, Any]]) -> None:
    # register_model probes each unknown name with get_model_info, and litellm
    # prints a "Provider List" banner to stdout for every miss unless told not
    # to. Silence only that, only here.
    quiet = litellm.suppress_debug_info
    litellm.suppress_debug_info = True
    try:
        litellm.register_model(copy.deepcopy(entries))
    finally:
        litellm.suppress_debug_info = quiet
    for key, entry in entries.items():
        if litellm.model_cost.get(key) != entry:
            raise HookError(
                f"after register_model, litellm.model_cost[{key!r}] is not the pinned entry"
            )
    info = litellm.get_model_info(LUNA_KEY)
    for field, expected in EXPECTED_COSTS.items():
        if info.get(field) != expected:
            raise HookError(
                f"get_model_info({LUNA_KEY!r})[{field!r}] is {info.get(field)!r}, "
                f"expected {expected!r}"
            )
    context = info.get("max_input_tokens")
    if not isinstance(context, int) or context <= 0:
        raise HookError(
            f"get_model_info({LUNA_KEY!r}) has no positive context size: {context!r}"
        )
    provider = litellm.get_llm_provider(LUNA_KEY)[1]
    if provider != "openai":
        raise HookError(f"{LUNA_KEY!r} routes to {provider!r}, expected 'openai'")


def _append(record: dict[str, Any]) -> None:
    line = json.dumps(record, default=repr, separators=(",", ":"), ensure_ascii=False)
    data = (line + "\n").encode("utf-8")
    path = os.path.join(str(_log_dir), f"calls-{os.getpid()}.jsonl")
    try:
        with _log_lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
            finally:
                os.close(fd)
    except OSError as exc:
        raise CallLogError(
            f"clbench_ks_hook could not append to {path}: {exc.strerror}"
        ) from None


def _field(obj: Any, name: str) -> Any:
    """``obj.name``, else ``obj[name]`` for a dict, as src/usage.py's _usage_value; None if absent.

    Never raises: what a call returned is only read here, and a read that
    fails must not turn a billed, successful call into an error.
    """
    try:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
    except Exception:
        pass
    return None


def _count(value: Any) -> int | None:
    """src/usage.py's _coerce_optional_int."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def usage_of(result: Any) -> dict[str, int | None] | None:
    """The four token counts of a returned response, or None when it has no usage."""
    usage = _field(result, "usage")
    if usage is None:
        return None
    counts: dict[str, int | None] = {}
    for key, paths in _USAGE_PATHS.items():
        value = None
        for path in paths:
            value = usage
            for name in path:
                value = _field(value, name)
            if value is not None:
                break
        counts[key] = _count(value)
    return counts


def served_model(result: Any) -> str | None:
    """The model a returned response names in its body, or None.

    litellm 1.81.6 copies the body's ``model`` verbatim on both routes, to
    ``ModelResponse.model`` on the completion route
    (litellm_core_utils/llm_response_utils/convert_dict_to_response.py,
    convert_to_model_response_object) and to ``ResponsesAPIResponse.model`` on
    the responses route (llms/openai/responses/transformation.py,
    transform_response_api_response). Probed on 1.81.6: a server answering
    ``gpt-5.6-luna`` with another model's name shows that name here, on both
    routes, whether the call named the model bare or as ``openai/...``.
    """
    model = _field(result, "model")
    return model if isinstance(model, str) else None


def kept_headers(*sources: Any) -> dict[str, str]:
    """The rate-limit and request-id headers in the given header maps, names lower-cased.

    The first source to carry a header wins; every other header is dropped.
    """
    kept: dict[str, str] = {}
    for source in sources:
        try:
            items = list(source.items()) if source is not None else []
        except Exception:
            continue
        for raw_name, value in items:
            name = str(raw_name).strip().lower()
            if name in kept:
                continue
            if name.startswith(_HEADER_PREFIX) or name in _HEADER_NAMES:
                kept[name] = str(value)
    return dict(sorted(kept.items()))


def response_headers(result: Any) -> dict[str, str]:
    """The kept headers of a returned response.

    litellm 1.81.6 hands the provider's raw headers back in
    ``ModelResponse._response_headers`` on the completion route
    (llms/openai/openai.py, convert_to_model_response_object) and in
    ``ResponsesAPIResponse._hidden_params["headers"]`` on the responses route
    (llms/openai/responses/transformation.py, transform_response_api_response).
    """
    hidden = _field(result, "_hidden_params")
    return kept_headers(
        _field(result, "_response_headers"),
        hidden.get("headers") if isinstance(hidden, dict) else None,
    )


def error_headers(exc: BaseException) -> dict[str, str]:
    """The kept headers of the response an exception came from.

    litellm 1.81.6 raises every provider error through exception_type(), which
    sets ``litellm_response_headers`` from the provider exception's own headers
    or response (litellm_core_utils/exception_mapping_utils.py,
    _get_response_headers). The mapped exception's ``response`` is a stub
    without headers.
    """
    return kept_headers(_field(exc, "litellm_response_headers"))


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception, then each one it was raised from or while handling.

    ``__context__`` is followed even when suppressed: the OpenAI SDK raises its
    status errors ``from None``, which hides httpx's HTTPStatusError from a
    traceback but keeps it here.
    """
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        yield link
        link = link.__cause__ or link.__context__


def is_transient(exc: BaseException) -> bool:
    """Whether a failed call may succeed if sent again (module docstring, item 5)."""
    import httpx  # loaded by litellm; imported here so this module needs neither

    if not isinstance(exc, Exception):
        return False
    links = list(_chain(exc))
    for link in links:
        if isinstance(link, httpx.HTTPStatusError):
            return link.response.status_code in RETRY_STATUSES
    no_response = (
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    )
    return any(isinstance(link, no_response) for link in links)


def _seconds(text: str, per_second: float = 1.0) -> float | None:
    """A header's number of seconds, or None unless it is finite and not negative."""
    try:
        value = float(text) / per_second
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def server_wait_s(headers: Mapping[str, str]) -> float | None:
    """The wait a failed response asked for, in seconds, or None when it asked for none.

    ``retry-after-ms``, else ``retry-after`` as seconds or an HTTP date, the
    order the OpenAI SDK reads them in (openai/_base_client.py,
    _parse_retry_after_header). A value that is neither counts as absent; a
    date already past asks for no wait. ``headers`` are ``kept_headers``
    output, so the names are lower-case.
    """
    if "retry-after-ms" in headers:
        wait = _seconds(headers["retry-after-ms"], per_second=1000.0)
        if wait is not None:
            return wait
    text = headers.get("retry-after")
    if text is None:
        return None
    wait = _seconds(text)
    if wait is not None:
        return wait
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def backoff_s(
    retry: int, draw: Callable[[float, float], float] = _jitter.uniform
) -> float:
    """Exponential backoff with full jitter before the ``retry``-th retry (1-based)."""
    ceiling = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** min(retry - 1, 16))
    return draw(0.0, ceiling)


def retry_wait_s(headers: Mapping[str, str], retry: int) -> float:
    """The wait before the ``retry``-th retry: the server's, else the backoff, to the ms."""
    wait = server_wait_s(headers)
    if wait is None:
        wait = backoff_s(retry)
    return round(wait, 3)


def _wrap(route: str, call: Callable[..., Any]) -> Callable[..., Any]:
    field = _EFFORT_FIELD[route]
    position = _MODEL_POSITION[route]

    @functools.wraps(call)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = args[position] if len(args) > position else None
        luna = isinstance(model, str) and model.endswith(LUNA_SUFFIX)
        if luna:
            if route == "completion":
                kwargs["reasoning_effort"] = EFFORT
            else:
                existing = kwargs.get("reasoning")
                if existing is None:
                    existing = {}
                if not isinstance(existing, dict):
                    raise TypeError(
                        f"clbench_ks_hook: reasoning must be a dict, got {type(existing).__name__}"
                    )
                kwargs["reasoning"] = {**existing, "effort": EFFORT}
        attempt = 0
        wait = waited = 0.0
        while True:
            attempt += 1
            record: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "pid": os.getpid(),
                "route": route,
                "model": model,
                field: copy.deepcopy(kwargs.get(field)),
                "attempt": attempt,
                "retry_wait_s": wait,
            }
            try:
                result = call(*args, **kwargs)
            except BaseException as exc:
                status = getattr(exc, "status_code", None)
                headers = error_headers(exc)
                record.update(
                    ok=False,
                    error=type(exc).__name__,
                    status_code=status if isinstance(status, int) else None,
                    response_id=None,
                    response_model=None,
                    usage=None,
                    headers=headers,
                )
                _append(record)
                if not (luna and attempt < _max_attempts and is_transient(exc)):
                    raise
                wait = retry_wait_s(headers, attempt)
                if waited + wait > _max_wait_s:
                    raise
                waited += wait
            else:
                response_id = getattr(result, "id", None)
                record.update(
                    ok=True,
                    error=None,
                    status_code=None,
                    response_id=response_id if isinstance(response_id, str) else None,
                    response_model=served_model(result),
                    usage=usage_of(result),
                    headers=response_headers(result),
                )
                _append(record)
                return result
            # Outside the except clause, so the next attempt's exception is
            # not chained to this one's.
            time.sleep(wait)

    wrapper.__clbench_ks_hook__ = route  # type: ignore[attr-defined]
    return wrapper


def _env_limit(
    name: str, default: float, parse: Callable[[str], float], minimum: float, what: str
) -> Any:
    """An optional retry cap from the environment; set but not valid is a HookError."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = parse(raw)
        valid = math.isfinite(value) and value >= minimum
    except (ValueError, OverflowError):
        valid = False
    if not valid:
        raise HookError(f"{name} must be {what}, got {raw!r}")
    return value


def _install() -> None:
    global _log_dir, _max_wait_s, _max_attempts
    expected = os.environ.get("CLBENCH_KS_MODEL_ENTRY_SHA256", "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise HookError(
            "CLBENCH_KS_MODEL_ENTRY_SHA256 must be the lowercase hex sha256 of "
            "luna_model_entry.json"
        )
    log_dir = os.environ.get("CLBENCH_KS_CALL_LOG_DIR", "")
    if not os.path.isabs(log_dir) or not os.path.isdir(log_dir):
        raise HookError(
            f"CLBENCH_KS_CALL_LOG_DIR must name an existing absolute directory: {log_dir!r}"
        )
    if not os.access(log_dir, os.W_OK | os.X_OK):
        raise HookError(f"CLBENCH_KS_CALL_LOG_DIR is not writable: {log_dir!r}")
    max_wait_s = _env_limit(
        "CLBENCH_KS_RETRY_MAX_WAIT_S",
        DEFAULT_MAX_WAIT_S,
        float,
        0.0,
        "a finite number of seconds, 0 or more",
    )
    max_attempts = _env_limit(
        "CLBENCH_KS_RETRY_MAX_ATTEMPTS",
        DEFAULT_MAX_ATTEMPTS,
        int,
        1,
        "a whole number, 1 or more",
    )
    if "litellm" in sys.modules:
        raise HookError(
            "litellm was imported before the hook, too late for LITELLM_LOCAL_MODEL_COST_MAP"
        )
    from importlib.metadata import version

    installed = version("litellm")
    if installed != VERIFIED_LITELLM_VERSION:
        raise HookError(
            f"litellm {installed} is installed; the hook was verified on "
            f"{VERIFIED_LITELLM_VERSION} only"
        )
    entries = load_entries(ENTRY_PATH, expected)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # litellm calls dotenv.load_dotenv() on import. This early, __main__ has no
    # __file__ yet, so python-dotenv would search upward from the working
    # directory rather than from litellm's own folder, and could load a .env
    # the benchmark never reads. LITELLM_MODE=PRODUCTION skips only that call,
    # and is set only while litellm is imported. The benchmark's own
    # load_dotenv() in src/cli.py is untouched.
    mode = os.environ.get("LITELLM_MODE")
    os.environ["LITELLM_MODE"] = "PRODUCTION"
    try:
        import litellm
    finally:
        if mode is None:
            os.environ.pop("LITELLM_MODE", None)
        else:
            os.environ["LITELLM_MODE"] = mode

    _pin_model_entry(litellm, entries)
    _log_dir = log_dir
    _max_wait_s, _max_attempts = float(max_wait_s), int(max_attempts)
    litellm.completion = _wrap("completion", litellm.completion)
    litellm.responses = _wrap("responses", litellm.responses)


def _fail(reason: str) -> None:
    line = "clbench_ks_hook: FATAL, exiting 78: " + " ".join(str(reason).split())
    try:
        os.write(2, (line[:2000] + "\n").encode("utf-8", "replace"))
    finally:
        os._exit(EXIT_CODE)


def install() -> None:
    """Pin the entry, read the retry caps, wrap both routes and open the call log, or exit 78."""
    global _installed
    if _installed:
        return
    try:
        _install()
    except BaseException as exc:
        _fail(f"{type(exc).__name__}: {exc}")
    _installed = True


def build(
    map_path: str, url: str, fetched_at: str, etag: str | None = None
) -> dict[str, Any]:
    """The pinned-entry document, copied verbatim from a downloaded model map."""
    with open(map_path, "rb") as handle:
        raw = handle.read()
    live = json.loads(raw.decode("utf-8"))
    entries = {key: live[key] for key in ENTRY_KEYS}
    source = {
        "url": url,
        "fetched_at": fetched_at,
        "http_etag": etag,
        "map_sha256": hashlib.sha256(raw).hexdigest(),
        "map_entry_count": len(live),
        "keys": list(ENTRY_KEYS),
        "copied": "both entries verbatim from the map at url, long-context "
        "(_above_272k_tokens) fields included",
        "entries_sha256": entries_sha256(entries),
        "entries_sha256_method": "sha256 of the UTF-8 bytes of json.dumps(<the "
        "entries keyed by name>, sort_keys=True, separators=(',', ':'), "
        "ensure_ascii=False); clbench_ks_hook.entries_sha256",
    }
    return {SOURCE_KEY: source, **entries}


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6) or sys.argv[1] != "build":
        sys.exit(
            "usage: python clbench_ks_hook.py build <map.json> <url> <fetched_at> [<etag>]"
        )
    document = build(*sys.argv[2:])
    with open(ENTRY_PATH, "w", encoding="utf-8") as out:
        out.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    with open(ENTRY_PATH, "rb") as written:
        print(hashlib.sha256(written.read()).hexdigest())
