import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import litellm
from pydantic import BaseModel

from src.cli import format_cost_summary
from src.interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    InstanceOutcome,
    Observation,
    Query,
    EvalMetrics,
    Response,
    TaskResult,
    TaskStepResult,
)
from src.runs import RunMode, run_benchmark_runs
from src.runtime.runner import run_task
from src.trace_storage import TraceRecorder
from src.usage import UsageEvent, build_usage_event, summarize_usage_events


class UsageAction(BaseModel):
    value: str = "ok"


class OneStepUsageTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))

    def build_current_query(self) -> Query:
        return Query(
            prompt="start",
            response_schema=UsageAction,
            instance_id="usage-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        return TaskStepResult(
            observation=Observation(
                content="done", metadata={"instance_complete": True}
            ),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0],
                optimal_performance=1.0,
                actual_performance=1.0,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="usage-0", instance_index=0, reward=1.0)
            ],
        )

    @property
    def name(self) -> str:
        return "usage_task"


class CostTrackingSystem(ContinualLearningSystem):
    def respond(self, query: Query) -> Response:
        self.record_usage_event(
            build_usage_event(
                model="gpt-5",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                call_type="completion",
            )
        )
        return Response(action=UsageAction())

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        self.record_usage_event(
            build_usage_event(
                model="text-embedding-3-small",
                provider="openai",
                input_tokens=25,
                output_tokens=0,
                call_type="embedding",
            )
        )

    def reset(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "cost_tracking_system"


class SelfReportedCostSystem(ContinualLearningSystem):
    def respond(self, query: Query) -> Response:
        self.record_usage_event(
            UsageEvent(
                call_type="completion",
                model="local-model",
                provider="local",
                cost_usd=0.25,
                pricing_source="self_reported",
            )
        )
        return Response(action=UsageAction())

    def reset(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "self_reported_cost_system"


class UsageTrackingTests(unittest.TestCase):
    def setUp(self):
        pricing = {
            "gpt-5.4": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 2.5e-6,
                "output_cost_per_token": 15e-6,
                "cache_read_input_token_cost": 0.25e-6,
                "input_cost_per_token_above_272k_tokens": 5e-6,
                "output_cost_per_token_above_272k_tokens": 22.5e-6,
                "cache_read_input_token_cost_above_272k_tokens": 0.5e-6,
            },
            "gpt-5.4-mini": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.75e-6,
                "output_cost_per_token": 4.5e-6,
                "cache_read_input_token_cost": 0.075e-6,
            },
        }
        self._pricing_patch = patch.dict(litellm.model_cost, pricing)
        self._pricing_patch.start()
        self.addCleanup(self._pricing_patch.stop)

    def test_usage_summary_allows_missing_token_counts(self):
        summary = summarize_usage_events(
            [
                UsageEvent(
                    call_type="completion",
                    model="local-model",
                    provider="local",
                    cost_usd=0.25,
                    pricing_source="self_reported",
                )
            ]
        )

        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["cost_usd"], 0.25)
        self.assertEqual(summary["total_tokens"], 0)
        self.assertFalse(summary["total_tokens_complete"])
        self.assertEqual(summary["missing_total_token_count"], 1)
        self.assertIn("tokens unavailable", format_cost_summary(summary))

    def test_trace_recorder_captures_respond_and_observe_usage(self):
        recorder = TraceRecorder(
            system_name="cost_tracking_system",
            task_name="usage_task",
            system_params={},
            task_params={},
        )

        result = run_task(
            OneStepUsageTask(),
            CostTrackingSystem(),
            trace_recorder=recorder,
            show_progress=False,
        )
        trace = recorder.finalize(result)

        interaction = trace["interactions"][0]
        self.assertEqual(interaction["response"]["metadata"]["usage"]["call_count"], 1)
        self.assertEqual(interaction["usage"]["respond"]["call_count"], 1)
        self.assertEqual(interaction["usage"]["observe"]["call_count"], 1)
        self.assertEqual(interaction["usage"]["interaction"]["call_count"], 2)
        self.assertGreater(trace["execution"]["usage"]["interaction"]["cost_usd"], 0.0)
        self.assertTrue(trace["execution"]["usage"]["interaction"]["pricing_complete"])

    def test_trace_recorder_preserves_self_reported_cost_without_tokens(self):
        recorder = TraceRecorder(
            system_name="self_reported_cost_system",
            task_name="usage_task",
            system_params={},
            task_params={},
        )

        result = run_task(
            OneStepUsageTask(),
            SelfReportedCostSystem(),
            trace_recorder=recorder,
            show_progress=False,
        )
        trace = recorder.finalize(result)

        usage = trace["execution"]["usage"]["interaction"]
        self.assertEqual(usage["call_count"], 1)
        self.assertEqual(usage["cost_usd"], 0.25)
        self.assertFalse(usage["total_tokens_complete"])
        self.assertEqual(usage["missing_total_token_count"], 1)

    def test_build_usage_event_discounts_cached_input_tokens(self):
        event = build_usage_event(
            model="gpt-5.4-mini",
            provider="openai",
            input_tokens=1000,
            cached_input_tokens=800,
            output_tokens=100,
            call_type="completion",
        )

        self.assertAlmostEqual(event.cost_usd or 0.0, 0.00066)
        self.assertEqual(event.cached_input_tokens, 800)

    def test_build_usage_event_under_long_context_threshold_uses_base_rates(self):
        event = build_usage_event(
            model="gpt-5.4",
            provider="openai",
            input_tokens=100_000,
            output_tokens=1_000,
            call_type="completion",
        )

        # Base rates: input $2.50/M, output $15/M.
        expected = 100_000 * 2.5e-6 + 1_000 * 1.5e-5
        self.assertAlmostEqual(event.cost_usd or 0.0, expected)
        self.assertEqual(event.pricing_source, "litellm.cost_per_token")

    def test_build_usage_event_above_long_context_threshold_applies_gpt5_tier(self):
        event = build_usage_event(
            model="gpt-5.4",
            provider="openai",
            input_tokens=400_000,
            output_tokens=1_000,
            call_type="completion",
        )

        # gpt-5.4 above 272k tier: input $5/M, output $22.5/M.
        expected = 400_000 * 5e-6 + 1_000 * 2.25e-5
        self.assertAlmostEqual(event.cost_usd or 0.0, expected)
        self.assertEqual(event.pricing_source, "cl_benchmark.tiered")

    def test_build_usage_event_applies_openai_long_context_multiplier(self):
        event = build_usage_event(
            model="gpt-5.4",
            provider="openai",
            input_tokens=300_000,
            output_tokens=1_000,
            call_type="completion",
        )

        expected = 300_000 * 5e-6 + 1_000 * 2.25e-5
        self.assertAlmostEqual(event.cost_usd or 0.0, expected)
        self.assertEqual(event.pricing_source, "cl_benchmark.tiered")

    def test_build_usage_event_above_threshold_applies_anthropic_tier(self):
        event = build_usage_event(
            model="claude-sonnet-4-5",
            provider="anthropic",
            input_tokens=300_000,
            output_tokens=1_000,
            call_type="completion",
        )

        # claude-sonnet-4-5 above 200k tier: input $6/M, output $22.5/M.
        expected = 300_000 * 6e-6 + 1_000 * 2.25e-5
        self.assertAlmostEqual(event.cost_usd or 0.0, expected)
        self.assertEqual(event.pricing_source, "cl_benchmark.tiered")

    def test_build_usage_event_long_context_with_cached_input(self):
        event = build_usage_event(
            model="gpt-5.4",
            provider="openai",
            input_tokens=400_000,
            cached_input_tokens=300_000,
            output_tokens=1_000,
            call_type="completion",
        )

        # Above 272k: input tier $5/M, cache_read tier $0.50/M, output tier $22.5/M.
        # Uncached input = 400k - 300k = 100k.
        expected = 100_000 * 5e-6 + 300_000 * 5e-7 + 1_000 * 2.25e-5
        self.assertAlmostEqual(event.cost_usd or 0.0, expected)
        self.assertEqual(event.pricing_source, "cl_benchmark.tiered")

    def test_run_rollouts_aggregate_execution_usage(self):
        orig_cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            task_results, _ = run_benchmark_runs(
                task_class=OneStepUsageTask,
                task_params={},
                system_class=CostTrackingSystem,
                system_params={},
                runs=2,
                max_workers=1,
                system_name="cost_tracking_system",
                task_name="usage_task",
                run_mode=RunMode.REPLICATE,
            )
            aggregate = task_results.aggregate()
            interaction_usage = aggregate["execution"]["usage"]["interaction"]
            self.assertEqual(interaction_usage["call_count"]["mean"], 2.0)
            self.assertGreater(interaction_usage["cost_usd"]["mean"], 0.0)
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
