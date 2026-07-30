"""Schema-card system tuned for database_exploration continual learning.

Stateful runs keep a living list of schema cards. During an instance the model
only answers; after the instance completes a reflector rebuilds the card
notebook (add/edit/remove) from the episode + prior cards. Cards are injected
immediately after the system prompt as durable memory. Disabled when
``stateless=True``. On schema-drift NOTICE, cards are cleared by default so
the notebook rebuilds from post-drift evidence (optional keep+STALE mode).
"""

from __future__ import annotations

import json
from typing import Any

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
    ProviderTurnClient,
    TokenBudgetTracker,
    completion_with_structured_output,
    count_tokens,
    resolve_context_token_limit,
)
from .artifacts import ensure_registered as ensure_schema_card_artifacts_registered

ensure_schema_card_artifacts_registered()

DRIFT_MARKER = "NOTICE: The live database schema or contents may have changed"

DEFAULT_SYSTEM_PROMPT = """\
You are answering natural-language questions about an unknown SQLite database.
You may issue exploratory QUERY actions, then submit an ANSWER.

Durable schema memory is provided immediately after these instructions as
SCHEMA CARDS. Treat that block as trusted environment knowledge (tables,
columns, types, joins, encodings) — stronger than a casual chat note. Prefer
using it over rediscovering the same facts with sqlite_master / PRAGMA unless
the cards are marked STALE or a query error contradicts them.

Do not invent schema facts. If a needed fact is missing, explore with SQL, then
continue answering. Schema cards are updated automatically after each question
from your episode and feedback; you do not write cards during the question.
"""

REFLECTION_SYSTEM_PROMPT = """\
You maintain a notebook of durable schema cards for a SQLite database agent.
Given the prior cards and one completed episode (SQL, results, feedback),
produce a FULL replacement notebook.

Rules:
- Keep only durable environment facts: tables/columns, types, encodings \
(cents vs dollars, timestamp units), joins, group identities, missing tables, \
migration/legacy/soft-delete notes.
- Do NOT store questions, SQL text, result row dumps, submitted/correct answers, \
or ephemeral plans.
- You may add, edit, merge, or remove cards. Newer episode evidence overrides \
older cards when they conflict. Prefer fewer, consistent cards.
- If feedback shows an incorrect answer caused by a bad schema assumption, \
fix or remove that assumption.
- If cards were stale after a migration, rewrite them from episode evidence.
"""

STALE_CARDS_WARNING = (
    "WARNING: A schema/content change was noticed. Prior schema cards may be "
    "STALE. Re-verify facts (sqlite_master / PRAGMA / sample values) before "
    "trusting them. Prefer fresh evidence over pre-notice cards until the "
    "notebook is rebuilt after this question."
)

MEMORY_HEADER = (
    "SCHEMA CARDS (trusted durable memory — authoritative unless marked STALE)"
)


class SchemaCardNotebook(BaseModel):
    """Reflector output: full rebuilt card notebook."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_cards": [
                        "Table products has columns id INTEGER, main_cat TEXT",
                        "g2.prc is cents; use COALESCE(prc_usd, prc/100.0)",
                    ],
                    "change_summary": "Added verified columns and price encoding",
                }
            ]
        }
    }

    schema_cards: list[str] = Field(
        description=(
            "Complete replacement list of durable schema card strings after "
            "reconciling prior cards with this episode. Must be a JSON array "
            "of strings, not a schema object."
        ),
        examples=[
            [
                "Table products has columns id INTEGER, main_cat TEXT",
                "g2.prc is cents; use COALESCE(prc_usd, prc/100.0)",
            ]
        ],
    )
    change_summary: str = Field(
        default="",
        description="Brief note on what was added, edited, or removed.",
        examples=["Added verified columns and price encoding"],
    )


def _format_schema_cards_block(cards: list[str], *, stale: bool) -> str:
    lines = [f"=== {MEMORY_HEADER} ==="]
    if stale:
        lines.append(STALE_CARDS_WARNING)
        lines.append("")
    if not cards:
        lines.append("(no schema cards yet)")
    else:
        for index, card in enumerate(cards, start=1):
            lines.append(f"--- card {index} ---")
            lines.append(card.strip())
            lines.append("")
    lines.append("=== END SCHEMA CARDS ===")
    return "\n".join(lines).strip()


def _format_episode_for_reflection(episode: list[dict[str, str]]) -> str:
    if not episode:
        return "(empty episode)"
    parts: list[str] = []
    for index, turn in enumerate(episode, start=1):
        role = turn.get("role", "?")
        content = turn.get("content", "")
        parts.append(f"[{index}] {role}:\n{content}")
    return "\n\n".join(parts)


@register_system("schema_card")
class SchemaCardSystem(ContinualLearningSystem):
    """ICL-style system with end-of-instance reflected schema cards."""

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int | None = None,
        system_prompt: str = "",
        name: str = "schema_card",
        reserve_tokens: int = 500,
        clear_context_between_instances: bool = True,
        stateless: bool = False,
        drop_stale_cards: bool = True,
        provider_mode: str = "auto",
        openai_store: bool = True,
        openai_include_encrypted_reasoning: bool = False,
        anthropic_max_tokens: int | None = None,
    ):
        """
        Args:
            model: LiteLLM model identifier.
            max_tokens: Context limit before FIFO truncation of dialogue.
            system_prompt: Optional extra instructions appended to the fixed
                system prompt. Empty uses the default alone.
            name: System identifier.
            reserve_tokens: Tokens reserved for system prompt + response.
            clear_context_between_instances: Clear dialogue at instance end
                (cards are preserved unless stateless).
            stateless: Baseline mode — no cards / no reflection.
            drop_stale_cards: When a drift NOTICE is seen, clear prior cards so
                the notebook rebuilds from post-drift evidence (default True).
                Set False to keep cards and only mark them STALE.
            provider_mode: ``auto`` or ``litellm_chat``.
            openai_store: OpenAI Responses server-side state.
            openai_include_encrypted_reasoning: Request encrypted reasoning.
            anthropic_max_tokens: Native Anthropic output budget.
        """
        self.stateless = stateless
        self.drop_stale_cards = drop_stale_cards
        if stateless:
            clear_context_between_instances = True

        self._name = name
        self.model = model
        self.max_tokens = resolve_context_token_limit(model, max_tokens)
        self.reserve_tokens = reserve_tokens
        self.clear_context_between_instances = clear_context_between_instances

        if system_prompt.strip():
            self.system_prompt = (
                f"{DEFAULT_SYSTEM_PROMPT.rstrip()}\n\n{system_prompt.strip()}"
            )
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.provider_mode = provider_mode
        self._provider_client = ProviderTurnClient(
            model=model,
            system_prompt=self.system_prompt,
            provider_mode=provider_mode,  # type: ignore[arg-type]
            openai_store=openai_store,
            openai_include_encrypted_reasoning=openai_include_encrypted_reasoning,
            anthropic_max_tokens=anthropic_max_tokens,
        )

        self.messages: list[dict[str, str]] = []
        self._token_budget = TokenBudgetTracker()
        self.schema_cards: list[str] = []
        self.cards_stale: bool = False
        self.drift_notice_count: int = 0
        self.reflection_count: int = 0
        self.card_snapshots: list[dict[str, Any]] = []
        self._episode: list[dict[str, str]] = []

        self.truncation_count: int = 0
        self.has_truncated_flag: bool = False
        self.interaction_count: int = 0
        self._at_instance_boundary: bool = True
        self._pending_feedback: str | None = None

    @property
    def cards_enabled(self) -> bool:
        return not self.stateless

    def respond(self, query: Query) -> Response:
        if self.cards_enabled:
            self._maybe_handle_drift(query.prompt or "")

        query_parts: list[str] = []
        if self._pending_feedback and self._at_instance_boundary:
            query_parts.append(
                f"FEEDBACK FROM PREVIOUS INSTANCE:\n{self._pending_feedback}"
            )
            self._pending_feedback = None

        if query.prompt:
            query_parts.append(query.prompt)
        query_content = "\n\n".join(query_parts) if query_parts else "(no content)"

        self.interaction_count += 1
        self._at_instance_boundary = False
        self._add_message("user", query_content)

        try:
            task_schema = query.response_schema
            prefix = [*self._system_messages(), *self._memory_messages()]
            self._truncate_context(
                prefix_messages=prefix,
                extra_tokens=self._response_schema_tokens(task_schema),
            )
            llm_messages = [*prefix, *self.messages]
            if self._provider_client.state.provider == "litellm":
                action, usage_event = completion_with_structured_output(
                    model=self.model,
                    messages=llm_messages,
                    response_schema=task_schema,
                )
                usage_events = [usage_event]
                assistant_record = action.model_dump_json()
            else:
                provider_result = self._provider_client.respond_structured(
                    messages=llm_messages,
                    response_schema=task_schema,
                )
                action = provider_result.action
                usage_events = provider_result.usage_events
                assistant_record = provider_result.assistant_record

            for usage_event in usage_events:
                self._note_prompt_token_usage(usage_event.input_tokens)
                self.record_usage_event(usage_event)
        except ProviderRefusalError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        self._add_message("assistant", assistant_record)
        self._sync_provider_visible_count()

        return Response(
            action=action,
            metadata={
                "interaction_count": self.interaction_count,
                "system_type": "schema_card",
                "model": self.model,
                "context_tokens": self._count_message_tokens(
                    [*self._memory_messages_plain(), *self.messages]
                ),
                "has_truncated": self.has_truncated_flag,
                "truncation_count": self.truncation_count,
                "schema_card_count": len(self.schema_cards),
                "cards_stale": self.cards_stale,
                "stateless": self.stateless,
                "drop_stale_cards": self.drop_stale_cards,
                "reflection_count": self.reflection_count,
                "provider_state": self._provider_client.state_metadata(),
            },
        )

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        instance_complete = observation_marks_instance_complete(observation)
        content = observation.content.strip()

        if content and not (self.stateless and instance_complete):
            if instance_complete and self.clear_context_between_instances:
                self._pending_feedback = content
            else:
                self._add_message("user", f"FEEDBACK: {content}")

        if instance_complete:
            if self.cards_enabled:
                # Include terminal feedback in the episode used for reflection.
                if content:
                    self._episode.append({"role": "feedback", "content": content})
                self._reflect_and_rebuild_cards()
            if self.clear_context_between_instances:
                self.messages = []
                self._episode = []
                self._provider_client.reset()
            if self.stateless:
                self.schema_cards = []
                self.cards_stale = False
            self._at_instance_boundary = True
        else:
            self._at_instance_boundary = False

    def reset(self) -> None:
        self.messages = []
        self._episode = []
        self.schema_cards = []
        self.cards_stale = False
        self.drift_notice_count = 0
        self.reflection_count = 0
        self.card_snapshots = []
        self._token_budget.reset()
        self.truncation_count = 0
        self.has_truncated_flag = False
        self.interaction_count = 0
        self._at_instance_boundary = True
        self._pending_feedback = None
        self._provider_client.reset()

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "schema_card",
            "schema_cards": list(self.schema_cards),
            "schema_card_count": len(self.schema_cards),
            "cards_stale": self.cards_stale,
            "drop_stale_cards": self.drop_stale_cards,
            "drift_notice_count": self.drift_notice_count,
            "reflection_count": self.reflection_count,
            "card_snapshots": list(self.card_snapshots),
            "stateless": self.stateless,
            "messages": list(self.messages),
            "message_count": len(self.messages),
            "interaction_count": self.interaction_count,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "context_tokens": self._count_message_tokens(
                [*self._memory_messages_plain(), *self.messages]
            ),
            "has_truncated": self.has_truncated_flag,
            "truncation_count": self.truncation_count,
            "provider_state": self._provider_client.state_metadata(),
        }

    def _maybe_handle_drift(self, prompt: str) -> None:
        if DRIFT_MARKER not in prompt:
            return
        self.drift_notice_count += 1
        if self.drop_stale_cards:
            self.schema_cards = []
            self.cards_stale = False
        else:
            self.cards_stale = True

    def _reflect_and_rebuild_cards(self) -> None:
        prior_cards = list(self.schema_cards)
        episode_text = _format_episode_for_reflection(self._episode)
        cards_text = _format_schema_cards_block(prior_cards, stale=self.cards_stale)
        user_prompt = "\n\n".join(
            [
                "Prior schema cards:",
                cards_text,
                "Completed episode (newest evidence):",
                episode_text,
                "Return the full updated notebook as DATA VALUES, for example:",
                json.dumps(
                    {
                        "schema_cards": [
                            "Table products has columns id INTEGER, main_cat TEXT"
                        ],
                        "change_summary": "Added verified columns",
                    }
                ),
                "schema_cards must be an array of strings; change_summary a string.",
            ]
        )
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            notebook, usage_event = completion_with_structured_output(
                model=self.model,
                messages=messages,
                response_schema=SchemaCardNotebook,
            )
            self.record_usage_event(usage_event)
        except ProviderRefusalError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Schema card reflection failed: {exc}") from exc

        rebuilt = [
            card.strip()
            for card in list(notebook.schema_cards)
            if isinstance(card, str) and card.strip()
        ]
        self.schema_cards = rebuilt
        self.cards_stale = False
        self.reflection_count += 1
        self.card_snapshots.append(
            {
                "reflection_index": self.reflection_count,
                "interaction_count": self.interaction_count,
                "prior_cards": prior_cards,
                "schema_cards": list(self.schema_cards),
                "change_summary": notebook.change_summary,
            }
        )

    def _is_anthropic_model(self) -> bool:
        model_lower = self.model.lower()
        return "anthropic/" in model_lower or "claude" in model_lower

    def _system_messages(self) -> list[dict]:
        if self._is_anthropic_model():
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        return [{"role": "system", "content": self.system_prompt}]

    def _memory_messages_plain(self) -> list[dict[str, str]]:
        if not self.cards_enabled:
            return []
        # Always inject the memory slot when stateful so the model knows the
        # channel exists even before the first reflection.
        return [
            {
                "role": "user",
                "content": _format_schema_cards_block(
                    self.schema_cards, stale=self.cards_stale
                ),
            }
        ]

    def _memory_messages(self) -> list[dict]:
        plain = self._memory_messages_plain()
        if not plain:
            return []
        if not self._is_anthropic_model():
            return plain
        # Elevate cards right after the system prompt; allow cache on the block
        # until the notebook changes.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": plain[0]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

    def _estimate_message_tokens(self, messages: list[dict[str, str]]) -> int:
        if not messages:
            return 0
        try:
            return litellm.token_counter(model=self.model, messages=messages)
        except Exception:
            total = 0
            for message in messages:
                total += 4
                total += count_tokens(self.model, message["role"])
                content = message["content"]
                if isinstance(content, list):
                    content = " ".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                total += count_tokens(self.model, str(content))
            total += 2
            return total

    def _response_schema_tokens(self, schema: type[BaseModel] | None) -> int:
        if schema is None:
            return 0
        return count_tokens(self.model, json.dumps(schema.model_json_schema()))

    def _count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        return self._token_budget.count(messages, self._estimate_message_tokens)

    def _note_prompt_token_usage(self, input_tokens: int | None) -> None:
        self._token_budget.note_usage(
            messages=[*self._memory_messages_plain(), *self.messages],
            input_tokens=input_tokens,
            estimate_fn=self._estimate_message_tokens,
        )

    def _sync_provider_visible_count(self) -> None:
        self._provider_client.state.sent_message_count = len(
            [*self._system_messages(), *self._memory_messages(), *self.messages]
        )

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._episode.append({"role": role, "content": content})
        self._truncate_context(prefix_messages=self._memory_messages_plain())

    def _truncate_context(
        self,
        *,
        prefix_messages: list[dict[str, str]] | None = None,
        extra_tokens: int = 0,
    ) -> None:
        while len(self.messages) > 1:
            current_tokens = self._count_message_tokens(
                [*(prefix_messages or []), *self.messages]
            ) + max(0, extra_tokens)
            available = self.max_tokens - self.reserve_tokens
            if current_tokens <= available:
                break
            self.messages.pop(0)
            self.truncation_count += 1
            self.has_truncated_flag = True
