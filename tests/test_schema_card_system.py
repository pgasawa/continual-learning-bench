"""Tests for the schema_card system."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import BaseModel

from src.artifacts import save_artifacts
from src.interface import Observation, Query
from src.registry import get_system_class, list_systems
from src.systems.schema_card.system import (
    DRIFT_MARKER,
    MEMORY_HEADER,
    STALE_CARDS_WARNING,
    TRUST_CARDS_REMINDER,
    SchemaCardEntry,
    SchemaCardNotebook,
    SchemaCardSystem,
    _feedback_marked_correct,
    _merge_cards_with_confidence,
)
from src.usage import UsageEvent


class DummyAction(BaseModel):
    action_type: str = "ANSWER"
    answer: str = "x"


def _make_query(prompt: str = "How many rows?") -> Query:
    return Query(
        prompt=prompt,
        response_schema=DummyAction,
        instance_id="q1",
        instance_index=0,
    )


def _make_observation(content: str, instance_complete: bool = True) -> Observation:
    return Observation(
        content=content, metadata={"instance_complete": instance_complete}
    )


def _dummy_action():
    return DummyAction(action_type="ANSWER", answer="x")


def _dummy_usage() -> UsageEvent:
    return UsageEvent(
        call_type="completion",
        model="test-model",
        cost_usd=0.0,
        pricing_source="test",
    )


def _capture_messages(system: SchemaCardSystem, query: Query) -> list[dict]:
    with patch(
        "src.systems.schema_card.system.completion_with_structured_output",
        return_value=(_dummy_action(), _dummy_usage()),
    ) as mock_llm:
        system.respond(query)
        return list(
            mock_llm.call_args.kwargs.get("messages") or mock_llm.call_args[0][0]
        )


class TestSchemaCardRegistry:
    def test_registered(self):
        assert "schema_card" in list_systems()
        assert get_system_class("schema_card").__name__ == "SchemaCardSystem"


class TestConfidenceHelpers:
    def test_feedback_correct_detection(self):
        assert _feedback_marked_correct("CORRECT: 42") is True
        assert _feedback_marked_correct("Question 1: CORRECT! Your answer: 1") is True
        assert _feedback_marked_correct("INCORRECT. Correct answer: 1") is False
        assert _feedback_marked_correct("no signal") is False

    def test_merge_upvotes_and_carries_exact_matches(self):
        prior = [
            SchemaCardEntry(content="tables: items_g1", confidence=2),
            SchemaCardEntry(content="old fact", confidence=4),
        ]
        merged = _merge_cards_with_confidence(
            prior,
            ["tables: items_g1", "brand lives in attrs"],
            upvote=True,
        )
        assert merged == [
            SchemaCardEntry(content="tables: items_g1", confidence=3),
            SchemaCardEntry(content="brand lives in attrs", confidence=1),
        ]

    def test_merge_without_upvote_preserves_scores(self):
        prior = [SchemaCardEntry(content="tables: items_g1", confidence=2)]
        merged = _merge_cards_with_confidence(
            prior,
            ["tables: items_g1", "new"],
            upvote=False,
        )
        assert merged[0].confidence == 2
        assert merged[1].confidence == 0


class TestSchemaCardBehavior:
    def test_no_mid_run_card_field_and_reflects_at_instance_end(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.reset()

        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(_dummy_action(), _dummy_usage()),
        ) as mock_llm:
            system.respond(_make_query("Q1"))
            schema = mock_llm.call_args.kwargs["response_schema"]
            assert schema is DummyAction
            assert "schema_card_update" not in schema.model_fields

        assert system.schema_cards == []

        notebook = SchemaCardNotebook(
            schema_cards=["g2.prc is cents; use COALESCE(prc_usd, prc/100)"],
            change_summary="added price encoding",
        )
        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(notebook, _dummy_usage()),
        ) as mock_reflect:
            system.observe(_make_observation("CORRECT: 42"))
            assert mock_reflect.called
            reflect_messages = (
                mock_reflect.call_args.kwargs.get("messages")
                or mock_reflect.call_args[0][0]
            )
            assert any(m["role"] == "system" for m in reflect_messages)

        assert system.schema_cards == [
            SchemaCardEntry(
                content="g2.prc is cents; use COALESCE(prc_usd, prc/100)",
                confidence=1,
            )
        ]
        assert system.reflection_count == 1
        assert system.messages == []
        assert "Do NOT rediscover" in system.system_prompt

        messages = _capture_messages(system, _make_query("Q2"))
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert MEMORY_HEADER in messages[1]["content"]
        assert TRUST_CARDS_REMINDER in messages[1]["content"]
        assert "confidence: 1" in messages[1]["content"]
        assert "g2.prc is cents" in messages[1]["content"]

    def test_correct_feedback_upvotes_surviving_cards(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.reset()
        system.schema_cards = [
            SchemaCardEntry(content="g1 is Office Products", confidence=2)
        ]
        system._episode = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "{}"},
        ]
        notebook = SchemaCardNotebook(
            schema_cards=["g1 is Office Products", "fdbk_g1.ts is ms"],
            change_summary="kept office mapping; added ts unit",
        )
        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(notebook, _dummy_usage()),
        ):
            system.observe(_make_observation("CORRECT! Your answer: 3"))
        assert system.schema_cards == [
            SchemaCardEntry(content="g1 is Office Products", confidence=3),
            SchemaCardEntry(content="fdbk_g1.ts is ms", confidence=1),
        ]
        assert system.card_snapshots[-1]["feedback_correct"] is True

    def test_incorrect_feedback_does_not_upvote(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.reset()
        system.schema_cards = [
            SchemaCardEntry(content="g1 is Office Products", confidence=2)
        ]
        system._episode = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "{}"},
        ]
        notebook = SchemaCardNotebook(
            schema_cards=["g1 is Office Products"],
            change_summary="unchanged",
        )
        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(notebook, _dummy_usage()),
        ):
            system.observe(_make_observation("INCORRECT. Correct answer: 1"))
        assert system.schema_cards == [
            SchemaCardEntry(content="g1 is Office Products", confidence=2)
        ]
        assert system.card_snapshots[-1]["feedback_correct"] is False

    def test_reflector_can_remove_cards(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.reset()
        system.schema_cards = [
            SchemaCardEntry(content="g1 is Electronics", confidence=1),
            SchemaCardEntry(content="g1 is Office Products", confidence=3),
        ]
        system._episode = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "{}"},
        ]
        notebook = SchemaCardNotebook(
            schema_cards=["g1 is Office Products"],
            change_summary="removed contradictory electronics claim",
        )
        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(notebook, _dummy_usage()),
        ):
            system.observe(_make_observation("INCORRECT. Correct answer: 1"))
        assert system.schema_cards == [
            SchemaCardEntry(content="g1 is Office Products", confidence=3)
        ]

    def test_stateless_skips_cards_and_reflection(self):
        system = SchemaCardSystem(
            model="gpt-5",
            provider_mode="litellm_chat",
            stateless=True,
        )
        system.reset()

        with patch(
            "src.systems.schema_card.system.completion_with_structured_output",
            return_value=(_dummy_action(), _dummy_usage()),
        ) as mock_llm:
            system.respond(_make_query("Q1"))
            messages = mock_llm.call_args.kwargs["messages"]
            assert all(MEMORY_HEADER not in str(m.get("content")) for m in messages)

        with patch(
            "src.systems.schema_card.system.completion_with_structured_output"
        ) as mock_llm:
            system.observe(_make_observation("CORRECT"))
            mock_llm.assert_not_called()
        assert system.schema_cards == []
        assert system.reflection_count == 0

    def test_drift_drops_cards_by_default(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.reset()
        system.schema_cards = [SchemaCardEntry(content="old table map", confidence=5)]
        system._at_instance_boundary = True

        messages = _capture_messages(
            system, _make_query(f"{DRIFT_MARKER}\n\nNew question")
        )
        assert system.drop_stale_cards is True
        assert system.schema_cards == []
        assert system.cards_stale is False
        assert "old table map" not in messages[1]["content"]
        assert system.drift_notice_count == 1

    def test_drift_keep_stale_cards_when_disabled(self):
        system = SchemaCardSystem(
            model="gpt-5",
            provider_mode="litellm_chat",
            drop_stale_cards=False,
        )
        system.reset()
        system.schema_cards = [SchemaCardEntry(content="old table map", confidence=5)]
        system._at_instance_boundary = True

        messages = _capture_messages(
            system, _make_query(f"{DRIFT_MARKER}\n\nNew question")
        )
        assert system.cards_stale is True
        assert STALE_CARDS_WARNING in messages[1]["content"]
        assert "old table map" in messages[1]["content"]
        assert "confidence: 5" in messages[1]["content"]

    def test_artifacts_written_to_disk(self):
        system = SchemaCardSystem(model="gpt-5", provider_mode="litellm_chat")
        system.schema_cards = [
            SchemaCardEntry(content="card one", confidence=2),
            SchemaCardEntry(content="card two", confidence=0),
        ]
        system.card_snapshots = [
            {
                "reflection_index": 1,
                "prior_cards": [],
                "schema_cards": [{"content": "card one", "confidence": 1}],
                "change_summary": "init",
                "feedback_correct": True,
            }
        ]
        artifacts = system.get_run_artifacts()
        assert artifacts["artifact_type"] == "schema_card"
        assert artifacts["schema_cards"][0]["confidence"] == 2

        with TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "run.json"
            trace_path.write_text("{}", encoding="utf-8")
            out = save_artifacts(artifacts, trace_path)
            assert out is not None
            assert (out / "schema_cards.md").exists()
            card_text = (out / "schema_cards" / "card_0001.md").read_text(
                encoding="utf-8"
            )
            assert "confidence: 2" in card_text
            assert "card one" in card_text
            assert (out / "reflections" / "reflection_0001.json").exists()
