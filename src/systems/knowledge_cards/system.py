"""Knowledge-cards continual learning system.

Stateful runs keep a living list of knowledge cards. During an instance the
model only answers; after the instance completes a reflector rebuilds the card
notebook (add/edit/remove) from the episode + prior cards. Cards are injected
immediately after the system prompt as trusted memory learned this run.
Disabled when ``stateless=True``. On schema-drift NOTICE (database_exploration),
cards are cleared by default so the notebook rebuilds from post-drift evidence
(optional keep+STALE mode).
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
from .artifacts import ensure_registered as ensure_knowledge_cards_artifacts_registered

ensure_knowledge_cards_artifacts_registered()

DRIFT_MARKER = "NOTICE: The live database schema or contents may have changed"

DEFAULT_SYSTEM_PROMPT = """\
You are solving a sequence of task instances. Durable KNOWLEDGE CARDS appear
immediately after these instructions. They were learned from earlier instances
in this run. Treat them as trusted memory — not as casual notes.

Prefer applying facts already stated on cards instead of rediscovering them,
unless the card block is empty, marked STALE, or new evidence contradicts a
card. When cards are cleared after an environment change, rediscovery is
expected until new cards are written.

Do not invent facts. Knowledge cards are updated automatically after each
instance from your episode and feedback; you do not write cards during the
instance.
"""

DEFAULT_REFLECTION_PROMPT = """\
You maintain a notebook of durable knowledge cards for an agent that will face
later instances in the same run. Given the prior cards and one completed
episode (actions, observations, feedback), produce a FULL replacement notebook.

Prefer concrete reusable state the later agent can apply without re-deriving:
entities and identities, parameters and encodings, relationships, counts or
coverage, and other durable environment facts from the episode. Prefer state
over vague strategy slogans.

Do NOT store low-value cards: response-schema or field-name reminders the task
already provides, generic advice without concrete facts, full episode
transcripts or raw dumps, submitted answers as the main content, or ephemeral
one-instance plans.

Write each card for the next acting agent. If a card is a list or inventory,
keep it complete enough to apply in full. Start each card with a short
"Use when:" line stating when the card applies (situation, entity, or cue),
then the durable content.

Hygiene (resolve in this reflection step): prefer fewer, consistent cards;
merge duplicates; when cards conflict, keep the version supported by newer
episode evidence and remove or rewrite the stale claim. Drop weak one-off
hypotheses that never recur. If coverage is unknown, say so briefly — do not
invent facts.
"""

STALE_CARDS_WARNING = (
    "WARNING: An environment change was noticed. Prior knowledge cards may be "
    "STALE. Re-verify facts before trusting them. Prefer fresh evidence over "
    "pre-notice cards until the notebook is rebuilt after this instance."
)

MEMORY_HEADER = "KNOWLEDGE CARDS (learned this run — trust them)"

TRUST_CARDS_REMINDER = (
    "These cards were earned from prior instances in this run. Prefer them "
    "over rediscovering covered facts unless STALE or new evidence contradicts "
    "a card."
)


class KnowledgeCardNotebook(BaseModel):
    """Reflector output: full rebuilt card notebook (text only)."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "knowledge_cards": [
                        "Entity X uses field Y for status; missing values mean inactive",
                        "Timestamps in table Z are epoch milliseconds",
                    ],
                    "change_summary": "Added status encoding and timestamp unit",
                }
            ]
        }
    }

    knowledge_cards: list[str] = Field(
        description=(
            "Complete replacement list of durable knowledge card strings after "
            "reconciling prior cards with this episode. Must be a JSON array "
            "of strings, not a schema object."
        ),
        examples=[
            [
                "Entity X uses field Y for status; missing values mean inactive",
                "Timestamps in table Z are epoch milliseconds",
            ]
        ],
    )
    change_summary: str = Field(
        default="",
        description="Brief note on what was added, edited, or removed.",
        examples=["Added status encoding and timestamp unit"],
    )


def _format_knowledge_cards_block(
    cards: list[str],
    *,
    stale: bool,
) -> str:
    lines = [f"=== {MEMORY_HEADER} ==="]
    if stale:
        lines.append(STALE_CARDS_WARNING)
        lines.append("")
    elif cards:
        lines.append(TRUST_CARDS_REMINDER)
        lines.append("")
    if not cards:
        lines.append("(no knowledge cards yet)")
    else:
        for index, card in enumerate(cards, start=1):
            content = str(card).strip()
            lines.append(f"--- card {index} ---")
            lines.append(content)
            lines.append("")
    lines.append("=== END KNOWLEDGE CARDS ===")
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


@register_system("knowledge_cards")
class KnowledgeCardsSystem(ContinualLearningSystem):
    """ICL-style system with end-of-instance reflected knowledge cards."""

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int | None = None,
        system_prompt: str = "",
        reflection_prompt: str = "",
        name: str = "knowledge_cards",
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
            reflection_prompt: Reflector system prompt. Empty uses the generic
                default; non-empty fully replaces it (for task-tuned A/B runs).
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

        if reflection_prompt.strip():
            self.reflection_prompt = reflection_prompt.strip()
        else:
            self.reflection_prompt = DEFAULT_REFLECTION_PROMPT

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
        self.knowledge_cards: list[str] = []
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
                "system_type": "knowledge_cards",
                "model": self.model,
                "context_tokens": self._count_message_tokens(
                    [*self._memory_messages_plain(), *self.messages]
                ),
                "has_truncated": self.has_truncated_flag,
                "truncation_count": self.truncation_count,
                "knowledge_card_count": len(self.knowledge_cards),
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
                self.knowledge_cards = []
                self.cards_stale = False
            self._at_instance_boundary = True
        else:
            self._at_instance_boundary = False

    def reset(self) -> None:
        self.messages = []
        self._episode = []
        self.knowledge_cards = []
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
            "artifact_type": "knowledge_cards",
            "knowledge_cards": list(self.knowledge_cards),
            "knowledge_card_count": len(self.knowledge_cards),
            "cards_stale": self.cards_stale,
            "drop_stale_cards": self.drop_stale_cards,
            "drift_notice_count": self.drift_notice_count,
            "reflection_count": self.reflection_count,
            "card_snapshots": list(self.card_snapshots),
            "stateless": self.stateless,
            "reflection_prompt": self.reflection_prompt,
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
            self.knowledge_cards = []
            self.cards_stale = False
        else:
            self.cards_stale = True

    def _reflect_and_rebuild_cards(self) -> None:
        prior_cards = list(self.knowledge_cards)
        episode_text = _format_episode_for_reflection(self._episode)
        cards_text = _format_knowledge_cards_block(prior_cards, stale=self.cards_stale)
        user_prompt = "\n\n".join(
            [
                "Prior knowledge cards:",
                cards_text,
                "Completed episode (newest evidence):",
                episode_text,
                "Return the full updated notebook as DATA VALUES, for example:",
                json.dumps(
                    {
                        "knowledge_cards": [
                            "Entity X uses field Y for status; missing values "
                            "mean inactive. Related column Z plays the same "
                            "role when Y is null. Coverage unknown beyond this "
                            "episode unless noted."
                        ],
                        "change_summary": "Added status encoding",
                    }
                ),
                "knowledge_cards must be an array of strings (each card can be "
                "a short paragraph); change_summary a string.",
            ]
        )
        messages = [
            {"role": "system", "content": self.reflection_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            notebook, usage_event = completion_with_structured_output(
                model=self.model,
                messages=messages,
                response_schema=KnowledgeCardNotebook,
            )
            self.record_usage_event(usage_event)
        except ProviderRefusalError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Knowledge card reflection failed: {exc}") from exc

        self.knowledge_cards = [
            card.strip()
            for card in list(notebook.knowledge_cards)
            if isinstance(card, str) and card.strip()
        ]
        self.cards_stale = False
        self.reflection_count += 1
        self.card_snapshots.append(
            {
                "reflection_index": self.reflection_count,
                "interaction_count": self.interaction_count,
                "prior_cards": prior_cards,
                "knowledge_cards": list(self.knowledge_cards),
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
                "content": _format_knowledge_cards_block(
                    self.knowledge_cards, stale=self.cards_stale
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
