"""Shared utility for calling LLMs with structured output, with fallback for unsupported models."""

import json
import logging
import re
import time
from typing import Any, Callable, get_args, get_origin

import litellm
from pydantic import BaseModel, ValidationError

from ...errors import as_provider_refusal
from ...usage import UsageEvent, build_usage_event_from_response

logger = logging.getLogger(__name__)

_RETRYABLE_LITELLM_EXCEPTIONS = (
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
    litellm.RateLimitError,
)
_RETRYABLE_LOCAL_EXCEPTIONS = (
    ConnectionError,
    OSError,
    ValidationError,
)
_RETRYABLE_BAD_REQUEST_SUBSTRINGS = (
    "could not parse the json body of your request",
    "request contains an invalid argument",
)
# Policy: every 5xx status is treated as transient, plus the three 4xx codes
# that the HTTP RFCs and major providers explicitly mark as retryable.
#
# 408 Request Timeout      — server didn't get a complete request in time.
# 425 Too Early            — replay-protection rejection, safe to resend.
# 429 Too Many Requests    — rate limited, retry after backoff.
# 500-599                  — server-side errors. By definition the failure is
#                            on the origin, edge, or proxy and a fresh request
#                            may succeed. This blanket coverage avoids the
#                            historical whack-a-mole pattern where each new
#                            provider-specific 5xx (Cloudflare 520-526,
#                            Anthropic 529 "overloaded", future codes) had to
#                            be added explicitly after a failure.
#
# All other 4xx codes (400, 401, 403, 404, 405, 410, 422, ...) are client
# errors that retrying cannot fix and are left non-retryable.
RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 425, 429}) | frozenset(
    range(500, 600)
)
_HTTP_STATUS_IN_MESSAGE_RE = re.compile(
    r"(?:status[_ ]?code|http(?:\s+error)?|error\s+code)[\s:=]+(\d{3})",
    re.IGNORECASE,
)


def extract_http_status(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from a raised exception.

    Different SDKs (litellm, openai, anthropic, httpx, urllib) attach the
    status code in different places. Falls back to a regex over ``str(exc)``
    so we still catch cases where the code is only embedded in the message.
    """
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    match = _HTTP_STATUS_IN_MESSAGE_RE.search(str(exc))
    if match:
        try:
            value = int(match.group(1))
        except ValueError:
            return None
        if 100 <= value <= 599:
            return value
    return None


_GRAMMAR_TOO_LARGE_SUBSTRINGS = (
    "compiled grammar is too large",
    "simplify your tool schemas",
)
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0
_RATE_LIMIT_BACKOFF_BASE = 15.0
# A single bounded retry for content-policy refusals.  Some 400s on reasoning
# models are stochastic; appending a benign reminder catches a meaningful
# fraction without changing semantics.  We deliberately do *not* retry more
# than once: a real policy violation should not be papered over.
_REFUSAL_RETRY_REMINDER = (
    "\n\nReminder: respond with valid JSON matching the requested schema. "
    "Keep your reasoning factual and on-task."
)
_SCHEMA_ECHO_RETRY_REMINDER = (
    "\n\nYour previous response copied JSON Schema metadata "
    "(keys like type, description, default, items, title, properties). "
    "Those describe the format — they are not the answer. "
    "Respond again with a JSON object of real DATA VALUES only."
)
_JSON_SCHEMA_META_KEYS = frozenset(
    {
        "type",
        "description",
        "default",
        "items",
        "title",
        "properties",
        "$defs",
        "$ref",
        "anyOf",
        "oneOf",
        "allOf",
        "required",
    }
)


class _EmptyContentError(Exception):
    """Raised when the LLM returns None content."""


def is_retryable_llm_exception(exc: Exception) -> bool:
    """Return True when a failed LLM call should be retried."""

    if isinstance(exc, _RETRYABLE_LITELLM_EXCEPTIONS):
        return True
    if isinstance(exc, _RETRYABLE_LOCAL_EXCEPTIONS):
        return True
    if isinstance(exc, _EmptyContentError):
        return True
    if isinstance(exc, json.JSONDecodeError):
        # The model produced structurally invalid JSON (bad delimiters, unescaped
        # quotes in string values, empty response that slipped past content checks).
        # These are stochastic — a fresh sample will usually parse correctly.
        return True
    if isinstance(exc, litellm.BadRequestError):
        message = str(exc).lower()
        return any(snippet in message for snippet in _RETRYABLE_BAD_REQUEST_SUBSTRINGS)
    # Catches transient HTTP errors that don't surface as one of litellm's typed
    # exception classes — most importantly Cloudflare 520-526, which providers
    # like Anthropic/Together can emit as a generic litellm.APIError.
    status = extract_http_status(exc)
    if status is not None and status in RETRYABLE_HTTP_STATUSES:
        return True
    return False


def _model_supports_response_format(model: str) -> bool:
    """Check if the model supports response_format parameter.

    Gemini models are excluded: their structured-output API intermittently
    rejects valid schemas with INVALID_ARGUMENT (400).  The prompt-based
    fallback is more reliable for Gemini.
    """
    model_lower = model.lower()
    if "gemini" in model_lower:
        return False
    try:
        supported_params = litellm.get_supported_openai_params(model=model)
        return supported_params is not None and "response_format" in supported_params
    except Exception:
        return False


def _annotation_example(annotation: Any) -> Any:
    """Build a tiny placeholder value for prompt examples."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is str:
        return "example"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if origin is list:
        item = _annotation_example(args[0]) if args else "example"
        return [item]
    if origin is dict:
        return {"key": "value"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return example_values_for_schema(annotation)
    if origin is not None:
        # Optional/Union: prefer the first non-None arg.
        for arg in args:
            if arg is not type(None):
                return _annotation_example(arg)
    return "example"


def example_values_for_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Build an example data payload (values, not JSON Schema metadata)."""
    extra = getattr(schema, "model_config", {}).get("json_schema_extra")
    if isinstance(extra, dict):
        examples = extra.get("examples")
        if isinstance(examples, list) and examples and isinstance(examples[0], dict):
            return dict(examples[0])

    example: dict[str, Any] = {}
    for name, field_info in schema.model_fields.items():
        if field_info.examples:
            example[name] = field_info.examples[0]
            continue
        if not field_info.is_required():
            try:
                example[name] = field_info.get_default(call_default_factory=True)
                continue
            except Exception:
                pass
        example[name] = _annotation_example(field_info.annotation)
    return example


def looks_like_json_schema_fragment(value: Any) -> bool:
    """True when a value looks like a copied JSON Schema node, not data."""
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    if "type" not in keys:
        return False
    return bool(keys & (_JSON_SCHEMA_META_KEYS - {"type"})) or keys == {"type"}


def payload_looks_like_schema_echo(
    data: Any,
    schema: type[BaseModel],
) -> bool:
    """Detect responses that echoed JSON Schema property defs as field values."""
    if not isinstance(data, dict):
        return False
    for name in schema.model_fields:
        if name in data and looks_like_json_schema_fragment(data[name]):
            return True
    return False


def schema_to_prompt_instruction(schema: type[BaseModel]) -> str:
    """Convert a Pydantic model's JSON schema into a prompt instruction.

    Includes a concrete value example. Some models (notably Gemini on the
    prompt-based structured-output path) otherwise copy JSON Schema metadata
    keys like ``type`` / ``description`` into the response as if they were
    field values.
    """
    json_schema = schema.model_json_schema()
    example = example_values_for_schema(schema)
    return (
        "\n\nYou MUST respond with a JSON object of DATA VALUES that satisfy "
        "this schema.\n"
        "The schema uses metadata keys (type, description, default, items, "
        "title, properties). Those keys describe the format — do NOT copy them "
        "into your response. Fill in real values (strings, arrays, numbers, "
        "objects) instead.\n"
        f"Schema:\n```json\n{json.dumps(json_schema, indent=2)}\n```\n"
        "Example of a correctly shaped response (replace with your real "
        "values):\n"
        f"```json\n{json.dumps(example, indent=2)}\n```\n"
        "Respond ONLY with the JSON object of values, no other text."
    )


def _is_basemodel_type(annotation: Any) -> bool:
    """Check if an annotation is or contains a BaseModel subclass."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return True
    return False


def coerce_stringified_fields(data: dict, schema: type[BaseModel]) -> dict:
    """Coerce string-valued fields to dicts when the schema expects a BaseModel."""
    coerced = dict(data)
    for field_name, field_info in schema.model_fields.items():
        if field_name not in coerced:
            continue
        value = coerced[field_name]
        if isinstance(value, str) and _is_basemodel_type(field_info.annotation):
            try:
                coerced[field_name] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
    return coerced


def extract_content(response) -> str | None:
    """Extract structured content from an LLM response, checking multiple locations.

    For Anthropic models, litellm implements ``response_format`` via a forced
    tool call (``json_tool_call``).  The structured JSON lives in
    ``tool_calls`` while ``message.content`` holds free-text reasoning that
    usually lacks the full schema.  We therefore check tool_calls *first* so
    the structured payload is always preferred over free-text.
    """
    message = response.choices[0].message

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        args = tool_calls[0].function.arguments
        if isinstance(args, str):
            return args
        if isinstance(args, dict):
            return json.dumps(args)

    # OpenAI parsed field
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, BaseModel):
            return parsed.model_dump_json()
        if isinstance(parsed, dict):
            return json.dumps(parsed)

    if message.content:
        return message.content

    return None


def _is_grammar_too_large(exc: Exception) -> bool:
    """Return True when the provider rejects the schema as too complex."""
    if not isinstance(exc, litellm.BadRequestError):
        return False
    msg = str(exc).lower()
    return any(s in msg for s in _GRAMMAR_TOO_LARGE_SUBSTRINGS)


def _prompt_based_kwargs(
    model: str,
    messages: list[dict],
    response_schema: type[BaseModel],
) -> dict:
    """Build completion kwargs with schema injected into the last user message."""
    messages = [m.copy() for m in messages]
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] += schema_to_prompt_instruction(response_schema)
            break
    return dict(model=model, messages=messages)


def completion_with_structured_output(
    model: str,
    messages: list[dict],
    response_schema: type[BaseModel],
) -> tuple[BaseModel, UsageEvent]:
    """
    Call LLM and parse response into a Pydantic model.

    If the model supports response_format, uses it directly.
    Otherwise, injects schema instructions into the prompt and parses the response.
    Retries automatically on transient network/server errors.

    Returns:
        Parsed Pydantic model instance.
    """
    use_response_format = _model_supports_response_format(model)

    if use_response_format:
        completion_kwargs = dict(
            model=model,
            messages=messages,
            response_format=response_schema,
        )
    else:
        logger.info(
            "Model %s does not support response_format, using prompt-based fallback",
            model,
        )
        completion_kwargs = _prompt_based_kwargs(model, messages, response_schema)

    refusal_retry_used = False
    schema_echo_retry_used = False
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = litellm.completion(**completion_kwargs)
            content = extract_content(response)
            if content is None:
                raise _EmptyContentError("LLM returned empty content")

            payload = content if use_response_format else extract_json(content)
            if not schema_echo_retry_used and _content_is_schema_echo(
                payload, response_schema
            ):
                schema_echo_retry_used = True
                logger.warning(
                    "Structured output looked like JSON Schema echo on %s; "
                    "retrying once with value-only reminder",
                    model,
                )
                completion_kwargs = _append_user_reminder(
                    completion_kwargs,
                    _SCHEMA_ECHO_RETRY_REMINDER
                    + "\nExample:\n"
                    + json.dumps(example_values_for_schema(response_schema)),
                )
                time.sleep(1.0)
                continue

            return (
                validate_with_coercion(payload, response_schema),
                build_usage_event_from_response(
                    model=model,
                    response=response,
                    call_type="completion",
                ),
            )
        except Exception as exc:
            refusal = as_provider_refusal(
                exc,
                model=model,
                messages=completion_kwargs.get("messages"),
            )
            if refusal is not None:
                if not refusal_retry_used:
                    refusal_retry_used = True
                    logger.warning(
                        "Provider refusal on %s; retrying once with benign reminder",
                        model,
                    )
                    completion_kwargs = _append_user_reminder(
                        completion_kwargs,
                        _REFUSAL_RETRY_REMINDER,
                    )
                    time.sleep(1.0)
                    continue
                raise refusal from exc
            if (
                isinstance(exc, ValidationError)
                and not schema_echo_retry_used
                and _validation_error_looks_like_schema_echo(exc)
            ):
                schema_echo_retry_used = True
                logger.warning(
                    "Validation failed with JSON Schema echo on %s; "
                    "retrying once with value-only reminder",
                    model,
                )
                completion_kwargs = _append_user_reminder(
                    completion_kwargs,
                    _SCHEMA_ECHO_RETRY_REMINDER
                    + "\nExample:\n"
                    + json.dumps(example_values_for_schema(response_schema)),
                )
                time.sleep(1.0)
                continue
            if _is_grammar_too_large(exc) and use_response_format:
                logger.warning(
                    "Grammar too large for response_format on %s; "
                    "switching to prompt-based fallback",
                    model,
                )
                use_response_format = False
                completion_kwargs = _prompt_based_kwargs(
                    model, messages, response_schema
                )
                continue
            if not is_retryable_llm_exception(exc) or attempt == _MAX_RETRIES:
                raise
            is_overloaded = "overloaded" in str(exc).lower()
            is_rate_limit = isinstance(exc, litellm.RateLimitError)
            base = (
                _RATE_LIMIT_BACKOFF_BASE
                if (is_rate_limit or is_overloaded)
                else _BACKOFF_BASE
            )
            wait = base * (2**attempt)
            logger.warning(
                "%s (attempt %d/%d), retrying in %.1fs: %s",
                "Rate limited" if is_rate_limit else "Transient error",
                attempt + 1,
                _MAX_RETRIES + 1,
                wait,
                exc,
            )
            time.sleep(wait)


def _append_user_reminder(completion_kwargs: dict, reminder: str) -> dict:
    """Copy completion kwargs and append a reminder to the last user message."""
    updated = dict(completion_kwargs)
    perturbed_messages = [m.copy() for m in updated.get("messages", [])]
    for i in range(len(perturbed_messages) - 1, -1, -1):
        if perturbed_messages[i].get("role") == "user":
            perturbed_messages[i]["content"] = (
                str(perturbed_messages[i].get("content", "")) + reminder
            )
            break
    updated["messages"] = perturbed_messages
    return updated


def _content_is_schema_echo(content: str, schema: type[BaseModel]) -> bool:
    try:
        data = json.loads(extract_json(content))
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return payload_looks_like_schema_echo(data, schema)


def _validation_error_looks_like_schema_echo(exc: ValidationError) -> bool:
    for err in exc.errors():
        input_value = err.get("input")
        if looks_like_json_schema_fragment(input_value):
            return True
    return False


def maybe_unwrap_parameter_envelope(json_str: str) -> str:
    """Unwrap ``{"parameter": {…}}`` envelopes produced by some providers.

    Anthropic's structured-output implementation sometimes wraps the
    entire response payload in a ``"parameter"`` key.  If the outer dict
    contains *only* that key and its value is itself a dict, re-serialize
    the inner value so downstream validation sees the real payload.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError, TypeError):
        return json_str
    if (
        isinstance(data, dict)
        and list(data.keys()) == ["parameter"]
        and isinstance(data["parameter"], dict)
    ):
        return json.dumps(data["parameter"])
    return json_str


def validate_with_coercion(json_str: str, schema: type[BaseModel]) -> BaseModel:
    """Parse JSON and coerce stringified nested BaseModel fields before validation."""
    json_str = maybe_unwrap_parameter_envelope(json_str)
    try:
        return schema.model_validate_json(json_str)
    except Exception:
        data = _parse_with_fallbacks(json_str, schema)
        coerced = coerce_stringified_fields(data, schema)
        try:
            return schema.model_validate(coerced)
        except ValidationError:
            recovered = _walk_for_object_with_expected_keys(
                coerced,
                set(schema.model_fields.keys()),
            )
            if recovered is None:
                raise
            return schema.model_validate(recovered)


def _parse_with_fallbacks(
    json_str: str,
    schema: type[BaseModel],
) -> Any:
    """Run progressively more tolerant JSON parses, returning the first success.

    Each strategy must either return the parsed object or raise
    ``json.JSONDecodeError`` / ``ValueError`` to defer to the next one.
    Strategies are ordered from cheapest/strictest to most permissive:

    1. ``json.loads`` on the raw content (fastest path for clean input).
    2. ``json.loads`` after ``extract_json`` (handles markdown fences,
       leading/trailing prose).
    3. ``decode_first_json_object`` (tolerates trailing JSON / text after
       a complete object).
    4. Same as (3) after repairing LLM JSON (raw control chars inside
       strings + invalid backslash escapes).
    5. ``_scan_for_embedded_object`` — schema-aware scan that walks every
       ``{`` and returns the first dict mentioning any expected field
       (e.g. for stringified-payload wrappers like ``{"thought": "<JSON>"}``
       where the outer string is malformed), on both raw and repaired text.

    If every strategy raises, the last error is re-raised so callers see
    the most informative traceback.
    """
    extracted = extract_json(json_str)
    repaired = _sanitize_invalid_escapes(_sanitize_unescaped_controls(extracted))
    strategies: list[Callable[[], Any]] = [
        lambda: json.loads(json_str),
        lambda: json.loads(extracted),
        lambda: decode_first_json_object(extracted),
        lambda: decode_first_json_object(repaired),
        lambda: _scan_or_raise(extracted, schema),
        lambda: _scan_or_raise(repaired, schema),
    ]
    last_error: Exception | None = None
    for strategy in strategies:
        try:
            return strategy()
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _scan_or_raise(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Adapter: ``_scan_for_embedded_object`` returns ``None`` on miss; convert
    that to a ``JSONDecodeError`` so it fits the fallback chain alongside
    the other strategies."""
    obj = _scan_for_embedded_object(text, schema)
    if obj is None:
        raise json.JSONDecodeError(
            "no embedded JSON object matching schema fields found", text, 0
        )
    return obj


def decode_first_json_object(text: str) -> Any:
    """Decode just the first JSON value from text that may contain trailing data."""
    decoder = json.JSONDecoder()
    text = text.lstrip()
    obj, _ = decoder.raw_decode(text)
    return obj


def _dict_contains_expected(obj: Any, expected: set[str]) -> bool:
    """True iff any expected key appears in obj or any nested dict value."""
    if isinstance(obj, dict):
        if any(k in obj for k in expected):
            return True
        return any(_dict_contains_expected(v, expected) for v in obj.values())
    return False


def _walk_for_object_with_expected_keys(
    obj: Any,
    expected: set[str],
) -> dict | None:
    """Walk a parsed JSON tree for the dict that carries the expected keys.

    Recurses through nested dicts, lists, and JSON-encoded strings (so
    markdown-fenced or stringified JSON inside a ``"thought"`` field can
    be reached).  Returns the first dict at any depth that contains any
    of the expected schema field names.  Returns ``None`` if nothing
    matches — callers re-raise the original validation error in that
    case so genuine schema misses aren't silently swallowed.
    """
    if isinstance(obj, str):
        try:
            obj = json.loads(extract_json(obj))
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(obj, dict):
        if any(k in obj for k in expected):
            return obj
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None

    for child in children:
        found = _walk_for_object_with_expected_keys(child, expected)
        if found is not None:
            return found
    return None


def _scan_for_embedded_object(
    text: str,
    schema: type[BaseModel],
) -> dict | None:
    """Walk every ``{`` in ``text`` and return the first parseable dict
    that mentions any of ``schema``'s top-level field names.

    Recovery path for models (e.g. Gemini under the prompt-based
    structured-output fallback) that wrap their answer inside a
    stringified JSON value, producing payloads like
    ``{"thought": "I will submit: { \\"voc_3_6__s12\\": 0.4, ... }"}``
    where the standard ``json.loads`` path blows up on the first
    inconsistently escaped quote.

    Tries the raw text first, then a heuristic ``\\" -> "`` and ``\\\\ -> \\``
    unescape pass to surface the inner object.
    """
    expected = set(schema.model_fields.keys())
    if not expected:
        return None

    candidates = [text]
    if '\\"' in text or "\\\\" in text:
        candidates.append(text.replace('\\"', '"').replace("\\\\", "\\"))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        pos = 0
        while True:
            idx = candidate.find("{", pos)
            if idx == -1:
                break
            try:
                obj, end = decoder.raw_decode(candidate, idx)
            except json.JSONDecodeError:
                pos = idx + 1
                continue
            if isinstance(obj, dict) and _dict_contains_expected(obj, expected):
                return obj
            pos = max(end, idx + 1)
    return None


# Matches a backslash not followed by a valid JSON escape character.
# Valid single-char escapes: " \ / b f n r t
# Valid multi-char escape:   uXXXX (handled by excluding 'u' from the pattern)
_INVALID_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')

_CONTROL_ESCAPE = {
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _sanitize_unescaped_controls(text: str) -> str:
    """Escape raw U+0000–U+001F characters inside JSON string literals.

    Models often put multiline shell/python in a ``command`` field with
    literal newlines instead of ``\\n``. JSON forbids raw controls in
    strings; this walk repairs them without touching structural whitespace
    outside strings, and without disturbing already-valid ``\\n`` escapes.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if in_string and ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ord(ch) < 0x20:
            out.append(_CONTROL_ESCAPE.get(ch, f"\\u{ord(ch):04x}"))
            continue
        out.append(ch)
    return "".join(out)


def _sanitize_invalid_escapes(text: str) -> str:
    """Double-escape backslashes that are not part of a valid JSON escape sequence.

    LLMs sometimes emit raw strings inside JSON values — Windows paths like
    ``C:\\Users\\name`` or regex patterns like ``\\s+`` — where the backslash
    is not properly escaped.  This converts ``\\p`` → ``\\\\p`` so the JSON
    parser can accept the response.
    """
    return _INVALID_ESCAPE_RE.sub(r"\\\\\1", text)


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown fences or surrounding text."""
    text = text.strip()

    # Try to extract from markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside fences
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                return part

    # If it starts with { or [, assume it's raw JSON
    if text.startswith("{") or text.startswith("["):
        return text

    # Last resort: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text
