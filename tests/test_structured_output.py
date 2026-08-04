"""Tests for structured output parsing utilities (no LLM calls required)."""

import json

import pytest
from pydantic import BaseModel

from src.systems.utils import structured_output
from src.systems.utils.structured_output import (
    RETRYABLE_HTTP_STATUSES,
    coerce_stringified_fields,
    decode_first_json_object,
    completion_with_structured_output,
    extract_content,
    extract_http_status,
    extract_json,
    is_retryable_llm_exception,
    maybe_unwrap_parameter_envelope,
    validate_with_coercion,
)
from src.usage import UsageEvent


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    name: str
    value: int


class NestedInner(BaseModel):
    x: int
    y: int


class SchemaWithNested(BaseModel):
    label: str
    inner: NestedInner


class OptionalFieldSchema(BaseModel):
    name: str
    score: float = 0.0


class RetryTestSchema(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_object(self):
        raw = '{"name": "foo", "value": 1}'
        assert extract_json(raw) == raw

    def test_plain_json_array(self):
        raw = "[1, 2, 3]"
        assert extract_json(raw) == raw

    def test_markdown_fence_json(self):
        raw = 'Here is the result:\n```json\n{"name": "a", "value": 2}\n```\nDone.'
        assert json.loads(extract_json(raw)) == {"name": "a", "value": 2}

    def test_markdown_fence_no_lang_tag(self):
        raw = '```\n{"name": "b", "value": 3}\n```'
        assert json.loads(extract_json(raw)) == {"name": "b", "value": 3}

    def test_surrounded_by_text(self):
        raw = 'The answer is {"name": "c", "value": 4} as shown.'
        result = json.loads(extract_json(raw))
        assert result["name"] == "c"
        assert result["value"] == 4

    def test_whitespace_padding(self):
        raw = '   \n  {"name": "d", "value": 5}  \n  '
        assert json.loads(extract_json(raw)) == {"name": "d", "value": 5}


# ---------------------------------------------------------------------------
# decode_first_json_object
# ---------------------------------------------------------------------------


class TestDecodeFirstJsonObject:
    def test_single_object(self):
        text = '{"name": "a", "value": 1}'
        assert decode_first_json_object(text) == {"name": "a", "value": 1}

    def test_two_concatenated_objects(self):
        text = '{"name": "a", "value": 1}\n{"name": "b", "value": 2}'
        assert decode_first_json_object(text) == {"name": "a", "value": 1}

    def test_object_with_trailing_text(self):
        text = '{"name": "a", "value": 1}\nSome extra text here'
        assert decode_first_json_object(text) == {"name": "a", "value": 1}

    def test_leading_whitespace(self):
        text = '  \n  {"name": "a", "value": 1}\n{"ignored": true}'
        assert decode_first_json_object(text) == {"name": "a", "value": 1}

    def test_array(self):
        text = "[1, 2, 3]\n[4, 5]"
        assert decode_first_json_object(text) == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            decode_first_json_object("not json at all")


# ---------------------------------------------------------------------------
# coerce_stringified_fields
# ---------------------------------------------------------------------------


class TestCoerceStringifiedFields:
    def test_stringified_nested_model(self):
        data = {"label": "test", "inner": '{"x": 1, "y": 2}'}
        coerced = coerce_stringified_fields(data, SchemaWithNested)
        assert coerced["inner"] == {"x": 1, "y": 2}

    def test_already_dict_unchanged(self):
        data = {"label": "test", "inner": {"x": 1, "y": 2}}
        coerced = coerce_stringified_fields(data, SchemaWithNested)
        assert coerced["inner"] == {"x": 1, "y": 2}

    def test_non_model_string_untouched(self):
        data = {"name": "hello", "value": 42}
        coerced = coerce_stringified_fields(data, SimpleSchema)
        assert coerced == data

    def test_missing_field_ignored(self):
        data = {"label": "test"}
        coerced = coerce_stringified_fields(data, SchemaWithNested)
        assert "inner" not in coerced

    def test_invalid_json_string_left_as_is(self):
        data = {"label": "test", "inner": "not-json"}
        coerced = coerce_stringified_fields(data, SchemaWithNested)
        assert coerced["inner"] == "not-json"


# ---------------------------------------------------------------------------
# validate_with_coercion
# ---------------------------------------------------------------------------


class TestValidateWithCoercion:
    def test_clean_json(self):
        raw = '{"name": "foo", "value": 42}'
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_json_with_trailing_object(self):
        """The GPT 5.4 failure case: valid JSON followed by another JSON object."""
        raw = '{"name": "foo", "value": 42}\n{"name": "bar", "value": 99}'
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_json_with_trailing_text(self):
        raw = '{"name": "foo", "value": 42}\nHere is some explanation text.'
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_markdown_fenced_json(self):
        raw = 'Result:\n```json\n{"name": "foo", "value": 42}\n```'
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_stringified_nested_field(self):
        raw = json.dumps({"label": "test", "inner": '{"x": 10, "y": 20}'})
        result = validate_with_coercion(raw, SchemaWithNested)
        assert result.label == "test"
        assert result.inner.x == 10
        assert result.inner.y == 20

    def test_default_fields_filled(self):
        raw = '{"name": "foo"}'
        result = validate_with_coercion(raw, OptionalFieldSchema)
        assert result.name == "foo"
        assert result.score == 0.0

    def test_completely_invalid_raises(self):
        with pytest.raises(Exception):
            validate_with_coercion("not json at all", SimpleSchema)

    def test_wrong_types_raises(self):
        raw = '{"name": 123, "value": "not_an_int"}'
        with pytest.raises(Exception):
            validate_with_coercion(raw, SimpleSchema)

    def test_multiple_trailing_objects(self):
        """Three concatenated JSON objects — should parse the first."""
        obj1 = {"name": "first", "value": 1}
        obj2 = {"name": "second", "value": 2}
        obj3 = {"name": "third", "value": 3}
        raw = "\n".join(json.dumps(o) for o in [obj1, obj2, obj3])
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "first"
        assert result.value == 1

    def test_recovers_object_embedded_in_malformed_thought_string(self):
        """Gemini failure case: 108-key payload stringified inside `thought`,
        with inconsistent quote escaping that breaks ``json.loads`` partway.

        The recovery scanner should still locate the inner ``{... "name": "foo" ...}``
        and validate from it.  We simulate the malformed prefix that Pydantic /
        ``json.loads`` chokes on by leaving an unescaped quote inside the
        outer string."""
        # Outer "thought" string contains an unescaped quote, then a valid
        # inner JSON object that satisfies SimpleSchema.
        raw = (
            '{"thought":"I will submit my answer below.'
            ' Here it is: "intermediate thought" '
            '{"name": "foo", "value": 42}"}'
        )
        # Sanity check: standard json.loads must really fail on this input,
        # otherwise the recovery path isn't being exercised.
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_recovers_object_from_markdown_fenced_string_value(self):
        """Gemini failure mode: the answer is properly-escaped markdown-fenced
        JSON inside a string value.  Outer ``json.loads`` parses cleanly into
        ``{"thought": "<long fenced string>"}``; only post-validation recovery
        catches it."""
        inner = {"name": "foo", "value": 42}
        thought = (
            "I will provide my answer in the JSON block below.\n"
            f"```json\n{json.dumps(inner)}\n```"
        )
        raw = json.dumps({"thought": thought})
        # Outer JSON is well-formed; the failure is in schema validation,
        # so this exercises the post-validate recovery path specifically.
        json.loads(raw)
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_recovers_object_from_raw_json_string_value(self):
        """Same shape but without the markdown fence — string value is bare JSON."""
        raw = json.dumps({"thought": '{"name": "foo", "value": 42}'})
        result = validate_with_coercion(raw, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_recovery_scanner_does_not_match_garbage(self):
        """If no embedded object satisfies the schema, recovery must not
        silently swallow the error — the original parse failure should
        surface."""
        raw = '{"thought":"nothing valid here", "broken": '
        with pytest.raises(Exception):
            validate_with_coercion(raw, SimpleSchema)

    def test_recovers_raw_newlines_inside_string_values(self):
        """Gemini/sales_prediction failure: multiline bash/python in ``command``
        with literal newlines (and tabs) inside the JSON string. JSON forbids
        raw U+0000–U+001F in strings; Pydantic raises ``json_invalid`` /
        control-character errors and the old fallbacks could not repair it.
        """

        class BashCommandSchema(BaseModel):
            thought: str
            command: str

        # Deliberately invalid JSON: raw newline + tab inside "command".
        raw = (
            '{"thought": "Write a python script.",'
            ' "command": "python - <<\'PY\'\n'
            "import json\n"
            "print(target_items)\n"
            'PY"}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        result = validate_with_coercion(raw, BashCommandSchema)
        assert result.thought.startswith("Write a python")
        assert "print(target_items)" in result.command
        assert "\n" in result.command

    def test_valid_escaped_newlines_unchanged(self):
        """Proper ``\\n`` escapes must still round-trip (no double-escaping)."""

        class BashCommandSchema(BaseModel):
            thought: str
            command: str

        raw = json.dumps({"thought": "ok", "command": "python -c 'print(1)\nprint(2)'"})
        result = validate_with_coercion(raw, BashCommandSchema)
        assert result.command == "python -c 'print(1)\nprint(2)'"


class FakeFunction:
    def __init__(self, arguments: str | dict):
        self.name = "json_tool_call"
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, arguments: str | dict):
        self.function = FakeFunction(arguments)


class FakeMessage:
    def __init__(self, content: str | None, *, tool_calls=None, parsed=None):
        self.content = content
        self.tool_calls = tool_calls
        self.parsed = parsed


class FakeChoice:
    def __init__(self, message: "FakeMessage"):
        self.message = message


class FakeResponse:
    def __init__(self, content: str, *, tool_calls=None, parsed=None):
        self.choices = [
            FakeChoice(FakeMessage(content, tool_calls=tool_calls, parsed=parsed))
        ]
        self.usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


class FakeBadRequestError(Exception):
    pass


class TestRetryPolicy:
    def test_retryable_bad_request_signature_is_explicit(self, monkeypatch):
        monkeypatch.setattr(
            structured_output.litellm, "BadRequestError", FakeBadRequestError
        )
        exc = FakeBadRequestError(
            "OpenAIException - We could not parse the JSON body of your request."
        )
        assert is_retryable_llm_exception(exc) is True

    def test_other_bad_request_errors_fail_fast(self, monkeypatch):
        monkeypatch.setattr(
            structured_output.litellm, "BadRequestError", FakeBadRequestError
        )
        exc = FakeBadRequestError("BadRequestError: unsupported response_format")
        assert is_retryable_llm_exception(exc) is False

    def test_cloudflare_5xx_status_attribute_is_retryable(self):
        exc = Exception("Anthropic edge error")
        exc.status_code = 520
        assert is_retryable_llm_exception(exc) is True

    def test_cloudflare_5xx_status_in_message_is_retryable(self):
        # litellm sometimes raises a generic APIError whose status code only
        # appears inside the stringified payload.
        exc = Exception("APIError: status_code: 522 origin connection timed out")
        assert is_retryable_llm_exception(exc) is True

    def test_request_timeout_status_is_retryable(self):
        exc = Exception("HTTP Error 408: Request Timeout")
        assert is_retryable_llm_exception(exc) is True

    def test_unrelated_4xx_is_not_retryable(self):
        exc = Exception("HTTP Error 401: Unauthorized")
        assert is_retryable_llm_exception(exc) is False

    def test_anthropic_overloaded_529_is_retryable(self):
        """Regression: Anthropic emits HTTP 529 with body
        ``{"type":"error","error":{"type":"overloaded_error",...}}`` when the
        cluster is over capacity. It must be classified as transient at every
        retry surface."""
        exc = Exception("HTTP Error 529: Too Many Requests / overloaded")
        exc.status_code = 529
        assert is_retryable_llm_exception(exc) is True


class TestRetryableHttpStatusesPolicy:
    """Pin the retry-status set to a *policy*, not a hand-curated list.

    Whack-a-mole guard: every time we forgot a transient code (520-526, 408,
    529, ...) a real benchmark run died. The policy is now ``5xx + the
    canonical transient 4xx``; these tests exist so future edits that
    silently narrow it (e.g. drop a 5xx, drop 425) are caught immediately.
    """

    EXPLICIT_TRANSIENT_4XX = {408, 425, 429}
    KNOWN_NON_RETRYABLE_4XX = {400, 401, 403, 404, 405, 406, 410, 422}

    def test_set_matches_policy_exactly(self):
        expected = frozenset(self.EXPLICIT_TRANSIENT_4XX) | frozenset(range(500, 600))
        assert RETRYABLE_HTTP_STATUSES == expected

    def test_every_5xx_is_retryable(self):
        for status in range(500, 600):
            assert status in RETRYABLE_HTTP_STATUSES, (
                f"{status} should be retryable (all 5xx are transient)"
            )

    def test_anthropic_overloaded_529_in_set(self):
        assert 529 in RETRYABLE_HTTP_STATUSES

    def test_cloudflare_edge_codes_in_set(self):
        for status in range(520, 527):
            assert status in RETRYABLE_HTTP_STATUSES

    def test_explicit_transient_4xx_in_set(self):
        for status in self.EXPLICIT_TRANSIENT_4XX:
            assert status in RETRYABLE_HTTP_STATUSES

    def test_known_non_retryable_4xx_excluded(self):
        for status in self.KNOWN_NON_RETRYABLE_4XX:
            assert status not in RETRYABLE_HTTP_STATUSES, (
                f"{status} is a client error and must NOT be retried"
            )


class TestExtractHttpStatus:
    def test_status_code_attribute(self):
        exc = Exception("boom")
        exc.status_code = 503
        assert extract_http_status(exc) == 503

    def test_response_status_code_attribute(self):
        # httpx-style: status code lives on exc.response.status_code.
        class FakeResp:
            status_code = 524

        exc = Exception("upstream timeout")
        exc.response = FakeResp()
        assert extract_http_status(exc) == 524

    def test_message_regex(self):
        exc = Exception("Anthropic request failed: error code: 520")
        assert extract_http_status(exc) == 520

    def test_no_status_returns_none(self):
        assert extract_http_status(ValueError("nothing http here")) is None

    def test_out_of_range_status_ignored(self):
        exc = Exception("status_code: 999 totally bogus")
        assert extract_http_status(exc) is None

    def test_completion_retries_known_parse_body_bad_request(self, monkeypatch):
        monkeypatch.setattr(
            structured_output, "_model_supports_response_format", lambda _model: False
        )
        monkeypatch.setattr(
            structured_output,
            "build_usage_event_from_response",
            lambda **kwargs: UsageEvent(call_type="completion", model="test-model"),
        )
        monkeypatch.setattr(structured_output.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            structured_output.litellm, "BadRequestError", FakeBadRequestError
        )

        calls = {"count": 0}

        def fake_completion(**_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FakeBadRequestError(
                    "OpenAIException - We could not parse the JSON body of your request."
                )
            return FakeResponse('{"answer":"ok"}')

        monkeypatch.setattr(structured_output.litellm, "completion", fake_completion)

        parsed, _usage = completion_with_structured_output(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            response_schema=RetryTestSchema,
        )

        assert parsed.answer == "ok"
        assert calls["count"] == 2

    def test_completion_does_not_retry_other_bad_requests(self, monkeypatch):
        monkeypatch.setattr(
            structured_output, "_model_supports_response_format", lambda _model: False
        )
        monkeypatch.setattr(structured_output.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            structured_output.litellm, "BadRequestError", FakeBadRequestError
        )

        calls = {"count": 0}

        def fake_completion(**_kwargs):
            calls["count"] += 1
            raise FakeBadRequestError("BadRequestError: unsupported response_format")

        monkeypatch.setattr(structured_output.litellm, "completion", fake_completion)

        with pytest.raises(FakeBadRequestError):
            completion_with_structured_output(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                response_schema=RetryTestSchema,
            )

        assert calls["count"] == 1


# ---------------------------------------------------------------------------
# extract_content – Anthropic tool-call priority
# ---------------------------------------------------------------------------


class NotepadUpdate(BaseModel):
    tracks_to_upsert: list[dict] = []
    delete_track_ids: list[str] = []


class AnthropicResponseSchema(BaseModel):
    """Mimics the merged schema with an Optional notepad_update field."""

    transmitters: list[dict] = []
    notepad_update: NotepadUpdate | None = None


class TestExtractContent:
    """Verify extract_content prefers structured tool_calls over free-text."""

    TOOL_JSON = json.dumps(
        {
            "transmitters": [{"center_freq": 30.0}],
            "notepad_update": {
                "tracks_to_upsert": [{"track_id": "T1"}],
                "delete_track_ids": [],
            },
        }
    )

    FREE_TEXT_WITH_PARTIAL_JSON = (
        "Based on the scan data I observe a peak at 30 MHz.\n"
        '{"transmitters": [{"center_freq": 30.0}]}'
    )

    def test_content_only_returns_content(self):
        """OpenAI / Gemini path: JSON in content, no tool_calls."""
        resp = FakeResponse(self.TOOL_JSON)
        assert extract_content(resp) == self.TOOL_JSON

    def test_tool_calls_only_returns_arguments(self):
        """Anthropic path when no text block is emitted."""
        resp = FakeResponse(None, tool_calls=[FakeToolCall(self.TOOL_JSON)])
        assert extract_content(resp) == self.TOOL_JSON

    def test_tool_calls_preferred_over_free_text(self):
        """Anthropic path: text reasoning in content, structured JSON in tool_calls.

        This is the core bug: litellm puts Claude's free-text reasoning in
        message.content and the real structured payload in tool_calls.
        extract_content must return the tool_call arguments so fields like
        notepad_update are not silently dropped.
        """
        resp = FakeResponse(
            self.FREE_TEXT_WITH_PARTIAL_JSON,
            tool_calls=[FakeToolCall(self.TOOL_JSON)],
        )
        result = extract_content(resp)
        parsed = json.loads(result)
        assert parsed.get("notepad_update") is not None, (
            "extract_content returned free-text content instead of tool_call "
            "arguments — notepad_update is lost"
        )
        assert parsed["notepad_update"]["tracks_to_upsert"] == [{"track_id": "T1"}]

    def test_tool_calls_dict_arguments(self):
        """tool_calls[0].function.arguments may be a dict instead of str."""
        args_dict = {"transmitters": [], "notepad_update": None}
        resp = FakeResponse(None, tool_calls=[FakeToolCall(args_dict)])
        result = extract_content(resp)
        assert json.loads(result) == args_dict

    def test_parsed_field_fallback(self):
        """OpenAI parsed field is used when content and tool_calls are absent."""
        resp = FakeResponse(None, parsed={"transmitters": [], "notepad_update": None})
        result = extract_content(resp)
        assert json.loads(result) == {"transmitters": [], "notepad_update": None}

    def test_empty_response_returns_none(self):
        resp = FakeResponse(None)
        assert extract_content(resp) is None


# ---------------------------------------------------------------------------
# maybe_unwrap_parameter_envelope
# ---------------------------------------------------------------------------


class TestMaybeUnwrapParameterEnvelope:
    """Anthropic sometimes wraps structured output in {"parameter": {…}}."""

    def test_unwraps_parameter_key(self):
        wrapped = json.dumps({"parameter": {"name": "foo", "value": 42}})
        assert json.loads(maybe_unwrap_parameter_envelope(wrapped)) == {
            "name": "foo",
            "value": 42,
        }

    def test_leaves_normal_json_alone(self):
        normal = json.dumps({"name": "foo", "value": 42})
        assert maybe_unwrap_parameter_envelope(normal) == normal

    def test_leaves_parameter_with_siblings_alone(self):
        has_siblings = json.dumps({"parameter": {"a": 1}, "extra": True})
        assert maybe_unwrap_parameter_envelope(has_siblings) == has_siblings

    def test_leaves_non_dict_parameter_alone(self):
        scalar = json.dumps({"parameter": 42})
        assert maybe_unwrap_parameter_envelope(scalar) == scalar

    def test_non_json_passthrough(self):
        assert maybe_unwrap_parameter_envelope("not json") == "not json"


class TestValidateWithCoercionParameterEnvelope:
    """End-to-end: wrapped payloads should validate correctly."""

    def test_wrapped_simple_schema(self):
        wrapped = json.dumps({"parameter": {"name": "foo", "value": 42}})
        result = validate_with_coercion(wrapped, SimpleSchema)
        assert result.name == "foo"
        assert result.value == 42

    def test_wrapped_with_optional_field(self):
        inner = {"name": "bar", "score": 3.14}
        wrapped = json.dumps({"parameter": inner})
        result = validate_with_coercion(wrapped, OptionalFieldSchema)
        assert result.name == "bar"
        assert result.score == 3.14


class TestSchemaEchoHandling:
    def test_prompt_instruction_includes_value_example(self):
        text = structured_output.schema_to_prompt_instruction(SimpleSchema)
        assert "DATA VALUES" in text
        assert "do NOT copy" in text
        assert '"name": "example"' in text
        assert '"value": 0' in text

    def test_detects_schema_property_echo(self):
        echoed = {
            "name": {"description": "Name", "type": "string"},
            "value": {"description": "Value", "type": "integer"},
        }
        assert structured_output.payload_looks_like_schema_echo(echoed, SimpleSchema)
        assert not structured_output.payload_looks_like_schema_echo(
            {"name": "ok", "value": 1},
            SimpleSchema,
        )

    def test_completion_retries_once_on_schema_echo(self, monkeypatch):
        monkeypatch.setattr(
            structured_output, "_model_supports_response_format", lambda _model: False
        )
        monkeypatch.setattr(
            structured_output,
            "build_usage_event_from_response",
            lambda **kwargs: UsageEvent(call_type="completion", model="test-model"),
        )
        monkeypatch.setattr(structured_output.time, "sleep", lambda _seconds: None)

        echo = json.dumps(
            {
                "answer": {
                    "description": "The answer",
                    "type": "string",
                }
            }
        )
        calls = {"count": 0, "messages": None}

        def fake_completion(**kwargs):
            calls["count"] += 1
            calls["messages"] = kwargs.get("messages")
            if calls["count"] == 1:
                return FakeResponse(echo)
            return FakeResponse('{"answer":"ok"}')

        monkeypatch.setattr(structured_output.litellm, "completion", fake_completion)

        parsed, _usage = completion_with_structured_output(
            model="gemini/gemini-3.6-flash",
            messages=[{"role": "user", "content": "reflect"}],
            response_schema=RetryTestSchema,
        )

        assert parsed.answer == "ok"
        assert calls["count"] == 2
        user_content = calls["messages"][-1]["content"]
        assert "JSON Schema metadata" in user_content
