"""Tests for the sales prediction continual learning task."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from src.tasks.sales_prediction.dgp import (
    HierarchicalPanelConfig,
    generate_hierarchical_panel,
    generate_locations,
    load_furniture_catalog,
)
from src.tasks.sales_prediction.evaluator import (
    parse_predictions,
    score_predictions,
    score_predictions_for_year,
)
from src.tasks.sales_prediction.corpus import (
    build_output_schema,
    slice_annual_ground_truth,
    slice_historical_csv,
)
from src.tasks.sales_prediction.prompts import get_system_prompt, render_template
from src.tasks.variants import clear_task_variant_cache, list_task_variants
from src.tasks.schedules import (
    clear_task_schedule_cache,
    list_task_schedules,
)


class DGPTests(unittest.TestCase):
    def setUp(self):
        self.furniture = load_furniture_catalog(n_per_type=2)
        self.locations = generate_locations()

    def test_catalog_loads_with_prices(self):
        self.assertEqual(len(self.furniture), 100)
        types = {f["furniture_type"] for f in self.furniture}
        self.assertEqual(len(types), 50)
        self.assertTrue(all(float(f["furniture_price"]) > 0 for f in self.furniture))

    def test_locations_has_three_entries(self):
        self.assertEqual(len(self.locations), 3)

    def test_hierarchical_panel_row_count(self):
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2028, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        rows = result["rows"]
        n_clustered = sum(
            1
            for f in self.furniture
            if f["furniture_type"]
            in {
                "SECTIONAL_SOFA",
                "CHESTERFIELD_SOFA",
                "LOUNGE_CHAIR",
                "ACCENT_CHAIR",
                "COFFEE_TABLE",
                "DINING_ROOM_TABLE",
                "CONSOLE_TABLE",
                "END_TABLE",
                "BOOKCASE",
                "DRESSER",
                "DISPLAY_CABINET",
                "SIDEBOARD_BUFFET",
                "KING_BED_FRAME",
                "QUEEN_BED_FRAME",
                "NIGHTSTAND",
                "UPHOLSTERED_BED",
                "WRITING_DESK",
                "EXECUTIVE_DESK",
                "DESK_CHAIR",
            }
        )
        expected = n_clustered * len(self.locations) * 3  # 3 years
        self.assertEqual(len(rows), expected)

    def test_hierarchical_panel_columns_present(self):
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2027, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        row = result["rows"][0]
        required = {
            "furniture_id",
            "location_id",
            "locality",
            "state",
            "furniture_type",
            "furniture_name",
            "furniture_price",
            "year",
            "date",
            "bucket_id",
            "expected_items_sold",
            "items_sold",
        }
        self.assertTrue(required.issubset(set(row.keys())))

    def test_hierarchical_panel_positive_expected(self):
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2028, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        for row in result["rows"]:
            self.assertGreater(row["expected_items_sold"], 0)

    def test_hierarchical_panel_nonnegative_sold(self):
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2028, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        for row in result["rows"]:
            self.assertGreaterEqual(row["items_sold"], 0)

    def test_hierarchical_panel_growth_shared_within_cluster(self):
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2028, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        cluster_params = result["cluster_params"]
        self.assertIn("seating", cluster_params)
        self.assertIn("tables", cluster_params)
        for cluster, (ceiling, growth) in cluster_params.items():
            self.assertGreaterEqual(growth, 1.10)
            self.assertLessEqual(growth, 1.25)

    def test_hierarchical_panel_config_round_trips(self):
        cfg = HierarchicalPanelConfig(seed=99, start_year=2020, end_year=2025)
        d = cfg.to_dict()
        cfg2 = HierarchicalPanelConfig.from_dict(d)
        self.assertEqual(cfg2.seed, 99)
        self.assertEqual(cfg2.start_year, 2020)
        self.assertEqual(cfg2.end_year, 2025)

    def test_hierarchical_panel_price_separation(self):
        """Cheaper items should have higher expected demand."""
        cfg = HierarchicalPanelConfig(start_year=2026, end_year=2026, seed=42)
        result = generate_hierarchical_panel(self.furniture, self.locations, config=cfg)
        by_price: dict[str, list[float]] = {"low": [], "high": []}
        for row in result["rows"]:
            bucket = "low" if row["furniture_price"] < 1000 else "high"
            by_price[bucket].append(row["expected_items_sold"])
        if by_price["low"] and by_price["high"]:
            self.assertGreater(np.mean(by_price["low"]), np.mean(by_price["high"]))


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.schema = [
            {"locality": "SF", "furniture_name": "A", "year": 2010},
            {"locality": "SF", "furniture_name": "B", "year": 2010},
            {"locality": "NY", "furniture_name": "A", "year": 2010},
            {"locality": "SF", "furniture_name": "A", "year": 2011},
            {"locality": "SF", "furniture_name": "B", "year": 2011},
            {"locality": "NY", "furniture_name": "A", "year": 2011},
        ]
        self.ground_truth = [
            {
                "locality": "SF",
                "furniture_name": "A",
                "year": 2010,
                "items_sold": 100,
                "expected_items_sold": 95,
            },
            {
                "locality": "SF",
                "furniture_name": "B",
                "year": 2010,
                "items_sold": 200,
                "expected_items_sold": 190,
            },
            {
                "locality": "NY",
                "furniture_name": "A",
                "year": 2010,
                "items_sold": 150,
                "expected_items_sold": 145,
            },
            {
                "locality": "SF",
                "furniture_name": "A",
                "year": 2011,
                "items_sold": 110,
                "expected_items_sold": 105,
            },
            {
                "locality": "SF",
                "furniture_name": "B",
                "year": 2011,
                "items_sold": 210,
                "expected_items_sold": 200,
            },
            {
                "locality": "NY",
                "furniture_name": "A",
                "year": 2011,
                "items_sold": 160,
                "expected_items_sold": 155,
            },
        ]

    def test_perfect_predictions(self):
        predictions = [
            {"locality": "SF", "furniture_name": "A", "year": 2010, "items_sold": 100},
            {"locality": "SF", "furniture_name": "B", "year": 2010, "items_sold": 200},
            {"locality": "NY", "furniture_name": "A", "year": 2010, "items_sold": 150},
            {"locality": "SF", "furniture_name": "A", "year": 2011, "items_sold": 110},
            {"locality": "SF", "furniture_name": "B", "year": 2011, "items_sold": 210},
            {"locality": "NY", "furniture_name": "A", "year": 2011, "items_sold": 160},
        ]
        result = score_predictions(
            json.dumps(predictions),
            self.ground_truth,
            self.schema,
        )
        self.assertTrue(result.file_created)
        self.assertTrue(result.format_valid)
        self.assertEqual(result.score, 1.0)

    def test_empty_output(self):
        result = score_predictions(
            "",
            self.ground_truth,
            self.schema,
        )
        self.assertFalse(result.file_created)
        self.assertEqual(result.score, 0.0)

    def test_invalid_json(self):
        result = score_predictions(
            "not json",
            self.ground_truth,
            self.schema,
        )
        self.assertTrue(result.file_created)
        self.assertFalse(result.format_valid)

    def test_wrong_count(self):
        predictions = [
            {"locality": "SF", "furniture_name": "A", "year": 2010, "items_sold": 100},
        ]
        result = score_predictions(
            json.dumps(predictions),
            self.ground_truth,
            self.schema,
        )
        self.assertFalse(result.format_valid)

    def test_year_feedback_score_uses_elapsed_year_only(self):
        predictions = [
            {"locality": "SF", "furniture_name": "A", "year": 2010, "items_sold": 100},
            {"locality": "SF", "furniture_name": "B", "year": 2010, "items_sold": 200},
            {"locality": "NY", "furniture_name": "A", "year": 2010, "items_sold": 150},
            {"locality": "SF", "furniture_name": "A", "year": 2011, "items_sold": 0},
            {"locality": "SF", "furniture_name": "B", "year": 2011, "items_sold": 0},
            {"locality": "NY", "furniture_name": "A", "year": 2011, "items_sold": 0},
        ]
        full_result = score_predictions(
            json.dumps(predictions),
            self.ground_truth,
            self.schema,
        )
        year_result = score_predictions_for_year(
            json.dumps(predictions),
            self.schema,
            self.ground_truth,
            year=2010,
        )
        self.assertTrue(year_result.format_valid)
        self.assertGreater(year_result.score, full_result.score)
        self.assertGreater(year_result.score, 0.9)

    def test_parse_predictions_missing_field(self):
        predictions = [
            {"locality": "SF", "furniture_name": "A", "year": 2010},
            {"locality": "SF", "furniture_name": "B", "year": 2010, "items_sold": 1},
            {"locality": "NY", "furniture_name": "A", "year": 2010, "items_sold": 1},
        ]
        report, errors = parse_predictions(json.dumps(predictions), self.schema[:3])
        self.assertTrue(report is None or len(errors) > 0)


class CorpusSliceTests(unittest.TestCase):
    def setUp(self):
        self.panel = [
            {
                "furniture_id": 1,
                "location_id": "LOC01",
                "locality": "SF",
                "furniture_type": "SOFA",
                "furniture_name": "Modern Sofa",
                "year": 2010,
                "month": 1,
                "date": "2010-01-01",
                "items_sold": 10,
                "expected_items_sold": 9.5,
            },
            {
                "furniture_id": 1,
                "location_id": "LOC01",
                "locality": "SF",
                "furniture_type": "SOFA",
                "furniture_name": "Modern Sofa",
                "year": 2011,
                "month": 1,
                "date": "2011-01-01",
                "items_sold": 12,
                "expected_items_sold": 11.0,
            },
            {
                "furniture_id": 2,
                "location_id": "LOC01",
                "locality": "SF",
                "furniture_type": "DESK",
                "furniture_name": "Classic Desk",
                "year": 2011,
                "month": 1,
                "date": "2011-01-01",
                "items_sold": 8,
                "expected_items_sold": 7.5,
            },
        ]

    def test_slice_historical_csv_filters_by_year(self):
        csv = slice_historical_csv(self.panel, before_year=2011)
        self.assertIn("2010", csv)
        self.assertNotIn("2011", csv)

    def test_slice_annual_ground_truth_aggregates(self):
        gt = slice_annual_ground_truth(self.panel, year=2011)
        self.assertEqual(len(gt), 2)
        names = {r["furniture_name"] for r in gt}
        self.assertIn("Modern Sofa", names)
        self.assertIn("Classic Desk", names)
        self.assertNotIn("year", gt[0])

    def test_slice_annual_ground_truth_multi_year(self):
        gt = slice_annual_ground_truth(self.panel, years=[2010, 2011])
        self.assertEqual(len(gt), 3)
        self.assertTrue(all("year" in r for r in gt))

    def test_slice_annual_ground_truth_with_type_filter(self):
        gt = slice_annual_ground_truth(self.panel, year=2011, target_types=["SOFA"])
        self.assertEqual(len(gt), 1)
        self.assertEqual(gt[0]["furniture_name"], "Modern Sofa")

    def test_slice_annual_ground_truth_with_target_entities(self):
        gt = slice_annual_ground_truth(
            self.panel,
            year=2011,
            target_entities=[{"locality": "SF", "furniture_name": "Classic Desk"}],
        )
        self.assertEqual(len(gt), 1)
        self.assertEqual(gt[0]["furniture_name"], "Classic Desk")

    def test_build_output_schema(self):
        furniture = [
            {"furniture_id": 1, "furniture_type": "SOFA", "furniture_name": "A"},
            {"furniture_id": 2, "furniture_type": "DESK", "furniture_name": "B"},
        ]
        locations = [
            {"location_id": "L1", "locality": "SF", "state": "CA"},
        ]
        schema = build_output_schema(furniture, locations)
        self.assertEqual(len(schema), 2)
        self.assertNotIn("year", schema[0])

        schema_filtered = build_output_schema(
            furniture, locations, target_types=["SOFA"]
        )
        self.assertEqual(len(schema_filtered), 1)

        schema_entities = build_output_schema(
            furniture,
            locations,
            target_entities=[{"locality": "SF", "furniture_name": "B"}],
            years=[2010, 2011],
        )
        self.assertEqual(len(schema_entities), 2)
        self.assertEqual(
            schema_entities,
            [
                {"locality": "SF", "furniture_name": "B", "year": 2010},
                {"locality": "SF", "furniture_name": "B", "year": 2011},
            ],
        )

    def test_build_output_schema_multi_year(self):
        furniture = [
            {"furniture_id": 1, "furniture_type": "SOFA", "furniture_name": "A"},
            {"furniture_id": 2, "furniture_type": "DESK", "furniture_name": "B"},
        ]
        locations = [
            {"location_id": "L1", "locality": "SF", "state": "CA"},
        ]
        schema = build_output_schema(furniture, locations, years=[2010, 2011])
        self.assertEqual(len(schema), 4)
        self.assertTrue(all("year" in s for s in schema))


class VariantAndRolloutTests(unittest.TestCase):
    def tearDown(self):
        clear_task_variant_cache()
        clear_task_schedule_cache()

    def test_list_variants(self):
        variants = list_task_variants("sales_prediction")
        ids = sorted(v.id for v in variants)
        self.assertEqual(
            ids,
            [
                "disrupted_products",
                "full_portfolio",
                "seasonal_products",
                "trending_products",
            ],
        )

    def test_variant_has_target_types(self):
        variants = list_task_variants("sales_prediction")
        for v in variants:
            self.assertIn("target_types", v.config)
            self.assertTrue(len(v.config["target_types"]) > 0)

    def test_list_rollouts(self):
        rollouts = list_task_schedules("sales_prediction")
        self.assertIn("default", {r.id for r in rollouts})

    def test_rollout_has_three_stages(self):
        rollouts = list_task_schedules("sales_prediction")
        spec = rollouts[0]
        self.assertEqual(len(spec.stages), 3)
        total = sum(int(s.schedule.get("num_instances", 0)) for s in spec.stages)
        self.assertEqual(total, 12)


class PromptsTests(unittest.TestCase):
    def test_system_prompt_renders(self):
        prompt = get_system_prompt()
        self.assertIn("JSON object", prompt)
        self.assertIn("thought", prompt)
        self.assertIn("command", prompt)

    def test_first_instance_template(self):
        prompt = render_template(
            "instance.j2",
            target_year=2027,
            round_num=1,
            target_entity_count=10,
            required_pairs=[
                {"locality": "SF", "furniture_name": "A", "year": 2027},
                {"locality": "NY", "furniture_name": "B", "year": 2027},
            ],
        )
        self.assertIn("SF / A / 2027", prompt)
        self.assertIn("NY / B / 2027", prompt)
        self.assertIn("institutional knowledge", prompt)
        self.assertIn("10 product-location combinations", prompt)
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", prompt)
        self.assertIn("This is year 2027", prompt)
        self.assertIn("through 2045", prompt)
        self.assertNotIn("previous_round_feedback.json", prompt)
        self.assertNotIn("output.json", prompt)

    def test_subsequent_instance_template(self):
        feedback_payload = json.dumps(
            {
                "feedback_year": 2014,
                "composite_score": 0.85,
                "entries": [
                    {
                        "locality": "SF",
                        "furniture_name": "A",
                        "predicted": 50.0,
                        "actual": 55,
                        "error": 5.0,
                    },
                ],
            },
            indent=2,
        )
        prompt = render_template(
            "instance.j2",
            target_year=2015,
            round_num=4,
            prev_year=2014,
            prev_round_feedback_json=feedback_payload,
            target_entity_count=5,
            required_pairs=[],
        )
        self.assertNotIn("previous_round_feedback.json", prompt)
        self.assertNotIn("previous_year_actuals.json", prompt)
        self.assertIn("composite_score", prompt)
        self.assertIn("0.85", prompt)
        self.assertIn("previous workspace", prompt)
        self.assertIn("2014", prompt)
        self.assertIn("institutional knowledge", prompt)


class ForecastHorizonTests(unittest.TestCase):
    def test_instance_forecast_years(self):
        from src.tasks.sales_prediction.task import _PredictionInstance

        inst = _PredictionInstance(
            instance_idx=0,
            stage_idx=0,
            variant_id=None,
            target_year=2010,
            target_types=[],
            forecast_horizon=5,
        )
        self.assertEqual(inst.forecast_years, [2010, 2011, 2012, 2013, 2014])

    def test_default_forecast_horizon(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42)
        self.assertEqual(task.forecast_horizon, 5)

    def test_custom_forecast_horizon(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42, forecast_horizon=3)
        self.assertEqual(task.forecast_horizon, 3)

    def test_schedule_alias_sets_rollout_schedule(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42, schedule="default")
        self.assertEqual(task.schedule, "default")

    def test_evaluate_uses_eval_metrics_shape(self):
        from src.interface import InstanceOutcome
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42)
        task.instance_history = [
            {
                "instance_idx": 0,
                "target_year": 2010,
                "forecast_years": [2010, 2011, 2012, 2013, 2014],
                "variant_id": "trending_products",
                "stage_idx": 0,
                "steps": 3,
                "score": 0.72,
                "format_valid": True,
                "errors": [],
            }
        ]
        task._instance_outcomes = [
            InstanceOutcome(
                instance_id="sales_prediction:years:2010-2014",
                instance_index=0,
                reward=0.72,
            )
        ]

        result = task.evaluate()

        self.assertEqual(result.score, 0.72)
        self.assertEqual(result.eval_metrics.loss_curve, [0.28])
        self.assertEqual(result.eval_metrics.optimal_performance, 1.0)
        self.assertEqual(result.eval_metrics.actual_performance, 0.72)
        self.assertEqual(result.eval_metrics.extra["cumulative_regret"], 0.28)


class FrozenCorpusLoadingTests(unittest.TestCase):
    def test_default_schedule_loads_shipped_frozen_corpus(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(schedule="default")
        task.build_canonical_run_state()

        self.assertEqual(task._resolved_corpus_id, "sales_lifecycle")
        self.assertIsNotNone(task._resolved_panel_path)
        self.assertEqual(task._resolved_panel_path.name, "sales_lifecycle_panel.jsonl")
        self.assertIsNone(task._generated_rollout_metadata)
        self.assertEqual(len(task.instances), 12)
        self.assertEqual(task.instances[0].target_year, 2027)
        self.assertEqual(task.instances[-1].target_year, 2038)

    def test_default_schedule_missing_corpus_raises_without_dgp_fallback(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(schedule="default")
        with (
            patch(
                "src.tasks.sales_prediction.task.resolve_corpus_paths",
                return_value=None,
            ),
            patch(
                "src.tasks.sales_prediction.task.build_rollout_corpus",
                side_effect=AssertionError("DGP fallback should not run"),
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "sales_lifecycle"):
                task.build_canonical_run_state()


class WorkspacePersistenceTests(unittest.TestCase):
    def test_default_workspace_persists(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42)
        self.assertFalse(task.clean_workspace_between_instances)

    def test_clean_workspace_flag_accepted(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(
            num_instances=1, seed=42, clean_workspace_between_instances=True
        )
        self.assertTrue(task.clean_workspace_between_instances)

    def test_clean_agent_workspace_command(self):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(
            num_instances=1, seed=42, clean_workspace_between_instances=True
        )
        commands_run: list[str] = []

        def mock_execute(command: str) -> dict:
            commands_run.append(command)
            return {"output": "", "returncode": 0, "exception_info": None}

        task._execute = mock_execute  # type: ignore[assignment]
        task._clean_agent_workspace()

        self.assertEqual(len(commands_run), 1)
        cmd = commands_run[0]
        self.assertIn("find /app", cmd)
        self.assertIn("! -name data", cmd)
        self.assertIn("-exec rm -rf", cmd)


class DataRoomTests(unittest.TestCase):
    """Tests for data_room.py: schema profiles, distractors, visibility filtering."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.furniture = [
            {
                "furniture_id": "F01",
                "furniture_name": "Modern Sofa",
                "furniture_type": "SOFA",
                "furniture_price": 800,
            },
            {
                "furniture_id": "F02",
                "furniture_name": "Classic Desk",
                "furniture_type": "DESK",
                "furniture_price": 500,
            },
            {
                "furniture_id": "F03",
                "furniture_name": "Cozy Armchair",
                "furniture_type": "ARMCHAIR",
                "furniture_price": 300,
            },
        ]
        self.locations = [
            {"location_id": "LOC01", "locality": "San Francisco", "state": "CA"},
            {"location_id": "LOC02", "locality": "New York", "state": "NY"},
            {"location_id": "LOC03", "locality": "Chicago", "state": "IL"},
        ]
        self.panel = [
            {
                "furniture_id": "F01",
                "location_id": "LOC01",
                "locality": "San Francisco",
                "state": "CA",
                "furniture_type": "SOFA",
                "furniture_name": "Modern Sofa",
                "furniture_price": 800,
                "year": 2010,
                "date": "2010",
                "items_sold": 15,
                "expected_items_sold": 14.0,
            },
            {
                "furniture_id": "F02",
                "location_id": "LOC02",
                "locality": "New York",
                "state": "NY",
                "furniture_type": "DESK",
                "furniture_name": "Classic Desk",
                "furniture_price": 500,
                "year": 2010,
                "date": "2010",
                "items_sold": 10,
                "expected_items_sold": 9.5,
            },
            {
                "furniture_id": "F03",
                "location_id": "LOC03",
                "locality": "Chicago",
                "state": "IL",
                "furniture_type": "ARMCHAIR",
                "furniture_name": "Cozy Armchair",
                "furniture_price": 300,
                "year": 2010,
                "date": "2010",
                "items_sold": 8,
                "expected_items_sold": 7.5,
            },
            {
                "furniture_id": "F01",
                "location_id": "LOC01",
                "locality": "San Francisco",
                "state": "CA",
                "furniture_type": "SOFA",
                "furniture_name": "Modern Sofa",
                "furniture_price": 800,
                "year": 2011,
                "date": "2011",
                "items_sold": 12,
                "expected_items_sold": 11.0,
            },
        ]

    def test_schema_profiles_exist(self):
        from src.tasks.sales_prediction.data_room import SCHEMA_PROFILES

        for name in ("west_coast", "east_coast", "national", "clean"):
            self.assertIn(name, SCHEMA_PROFILES)

    def test_get_schema_profile_default(self):
        from src.tasks.sales_prediction.data_room import get_schema_profile

        profile = get_schema_profile(None)
        self.assertEqual(profile.name, "clean")

    def test_build_data_room_clean_profile(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="clean",
            seed=0,
        )
        self.assertIn("furniture_id", dr.sales_csv)
        self.assertIn("location_id", dr.sales_csv)
        self.assertIn("items_sold", dr.sales_csv)
        self.assertNotIn("2011", dr.sales_csv)
        self.assertEqual(dr.schema_profile_name, "clean")

    def test_build_data_room_west_coast_schema(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="west_coast",
            seed=0,
        )
        self.assertIn("product_id", dr.sales_csv)
        self.assertIn("store_id", dr.sales_csv)
        self.assertIn("sale_year", dr.sales_csv)
        self.assertIn("quantity", dr.sales_csv)
        self.assertNotIn("furniture_id", dr.sales_csv)

    def test_build_data_room_east_coast_date_format(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="east_coast",
            seed=0,
        )
        self.assertIn("item_code", dr.sales_csv)
        self.assertIn("txn_year", dr.sales_csv)
        self.assertIn("2010", dr.sales_csv)
        self.assertNotIn("2010-03-01", dr.sales_csv)

    def test_visibility_filters_types(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            visible_types=["SOFA"],
            seed=0,
        )
        self.assertIn("F01", dr.sales_csv)
        self.assertNotIn("F02", dr.sales_csv)
        furn = json.loads(dr.furniture_json)
        types = {f["furniture_type"] for f in furn}
        self.assertEqual(types, {"SOFA"})

    def test_visibility_filters_locations(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            visible_locations=["LOC01"],
            seed=0,
        )
        self.assertIn("LOC01", dr.sales_csv)
        self.assertNotIn("LOC02", dr.sales_csv)
        locs = json.loads(dr.locations_json)
        self.assertEqual(len(locs), 1)
        self.assertEqual(locs[0]["locality"], "San Francisco")

    def test_distractors_add_columns(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            distractors=["weather_index", "foot_traffic"],
            seed=42,
        )
        self.assertIn("weather_index", dr.sales_csv)
        self.assertIn("foot_traffic", dr.sales_csv)
        self.assertEqual(
            dr.sales_csv_columns,
            [
                "furniture_id",
                "location_id",
                "year",
                "items_sold",
                "weather_index",
                "foot_traffic",
            ],
        )

    def test_no_distractors_for_clean_profile(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="clean",
            seed=0,
        )
        self.assertEqual(
            dr.sales_csv_columns,
            ["furniture_id", "location_id", "year", "items_sold"],
        )

    def test_furniture_json_uses_schema_keys(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="east_coast",
            seed=0,
        )
        furn = json.loads(dr.furniture_json)
        self.assertIn("item_code", furn[0])
        self.assertIn("description", furn[0])
        self.assertNotIn("furniture_id", furn[0])

    def test_locations_json_uses_schema_keys(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            schema_profile="west_coast",
            seed=0,
        )
        locs = json.loads(dr.locations_json)
        self.assertIn("store_id", locs[0])
        self.assertIn("city", locs[0])
        self.assertNotIn("location_id", locs[0])

    def test_distractor_deterministic(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr1 = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            distractors=["weather_index"],
            seed=99,
        )
        dr2 = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            distractors=["weather_index"],
            seed=99,
        )
        self.assertEqual(dr1.sales_csv, dr2.sales_csv)

    def test_location_open_year_filters_history(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2012,
            location_open_year={"LOC02": 2011},
            seed=0,
        )
        # LOC02 (NY) row in 2010 should be excluded (opened 2011)
        self.assertNotIn("LOC02", dr.sales_csv.split("\n")[1])
        # LOC01 (SF) 2010 and 2011 rows should be present
        self.assertIn("LOC01", dr.sales_csv)

    def test_location_open_year_no_effect_on_established(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr_without = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            seed=0,
        )
        dr_with = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2011,
            location_open_year={"LOC99": 2005},
            seed=0,
        )
        self.assertEqual(dr_without.sales_csv, dr_with.sales_csv)

    def test_scoring_unaffected_by_schema(self):
        """Scoring uses target_types (not visible_types), so it shouldn't change."""
        gt = [
            {
                "locality": "San Francisco",
                "furniture_name": "Modern Sofa",
                "year": 2010,
                "items_sold": 15,
                "expected_items_sold": 14.0,
            },
        ]
        schema = [
            {
                "locality": "San Francisco",
                "furniture_name": "Modern Sofa",
                "year": 2010,
            },
        ]
        predictions = json.dumps(
            [
                {
                    "locality": "San Francisco",
                    "furniture_name": "Modern Sofa",
                    "year": 2010,
                    "items_sold": 15,
                },
            ]
        )
        result = score_predictions(predictions, gt, schema)
        self.assertTrue(result.format_valid)
        self.assertEqual(result.score, 1.0)


class RetentionAndAssortmentTests(unittest.TestCase):
    """Tests for retention window, per-location assortments, and per-location schemas."""

    def setUp(self):
        self.furniture = [
            {
                "furniture_id": "F01",
                "furniture_name": "Modern Sofa",
                "furniture_type": "SOFA",
                "furniture_price": 800,
            },
            {
                "furniture_id": "F02",
                "furniture_name": "Classic Desk",
                "furniture_type": "DESK",
                "furniture_price": 500,
            },
            {
                "furniture_id": "F03",
                "furniture_name": "Cozy Armchair",
                "furniture_type": "ARMCHAIR",
                "furniture_price": 300,
            },
        ]
        self.locations = [
            {"location_id": "LOC01", "locality": "San Francisco", "state": "CA"},
            {"location_id": "LOC02", "locality": "New York", "state": "NY"},
        ]
        self.panel = []
        for year in range(2010, 2020):
            for loc, loc_name, state in [
                ("LOC01", "San Francisco", "CA"),
                ("LOC02", "New York", "NY"),
            ]:
                for fid, fname, ftype, fprice in [
                    ("F01", "Modern Sofa", "SOFA", 800),
                    ("F02", "Classic Desk", "DESK", 500),
                    ("F03", "Cozy Armchair", "ARMCHAIR", 300),
                ]:
                    self.panel.append(
                        {
                            "furniture_id": fid,
                            "location_id": loc,
                            "locality": loc_name,
                            "state": state,
                            "furniture_type": ftype,
                            "furniture_name": fname,
                            "furniture_price": fprice,
                            "year": year,
                            "date": str(year),
                            "items_sold": 10 + year - 2010,
                            "expected_items_sold": 10.0 + year - 2010,
                        }
                    )

    def test_retention_window_filters_old_years(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            retention_years=3,
            seed=0,
        )
        for year in range(2010, 2015):
            self.assertNotIn(str(year), dr.sales_csv)
        for year in range(2015, 2018):
            self.assertIn(str(year), dr.sales_csv)
        self.assertNotIn("2018", dr.sales_csv)

    def test_per_location_csvs_with_schema_map(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            visible_locations=["LOC01", "LOC02"],
            location_schema_map={"LOC01": "west_coast", "LOC02": "east_coast"},
            seed=0,
        )
        self.assertEqual(dr.sales_csv, "")
        self.assertIn("sales_san_francisco.csv", dr.extra_files)
        self.assertIn("sales_new_york.csv", dr.extra_files)
        sf_csv = dr.extra_files["sales_san_francisco.csv"]
        ny_csv = dr.extra_files["sales_new_york.csv"]
        self.assertIn("product_id", sf_csv)
        self.assertIn("quantity", sf_csv)
        self.assertIn("item_code", ny_csv)
        self.assertIn("units_sold", ny_csv)
        self.assertNotIn("LOC02", sf_csv)
        self.assertNotIn("LOC01", ny_csv)

    def test_per_location_assortment_filters_types(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            visible_locations=["LOC01", "LOC02"],
            location_schema_map={"LOC01": "west_coast", "LOC02": "east_coast"},
            location_assortments={"LOC01": ["SOFA"], "LOC02": ["DESK", "ARMCHAIR"]},
            seed=0,
        )
        sf_csv = dr.extra_files["sales_san_francisco.csv"]
        ny_csv = dr.extra_files["sales_new_york.csv"]
        self.assertIn("F01", sf_csv)
        self.assertNotIn("F02", sf_csv)
        self.assertNotIn("F03", sf_csv)
        self.assertNotIn("F01", ny_csv)
        self.assertIn("F02", ny_csv)
        self.assertIn("F03", ny_csv)

    def test_location_csv_columns_populated(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            visible_locations=["LOC01", "LOC02"],
            location_schema_map={"LOC01": "west_coast", "LOC02": "east_coast"},
            seed=0,
        )
        self.assertIn("sales_san_francisco.csv", dr.location_csv_columns)
        self.assertIn("sales_new_york.csv", dr.location_csv_columns)
        sf_cols = dr.location_csv_columns["sales_san_francisco.csv"]
        ny_cols = dr.location_csv_columns["sales_new_york.csv"]
        self.assertIn("product_id", sf_cols)
        self.assertIn("item_code", ny_cols)

    def test_retention_plus_per_location_combined(self):
        from src.tasks.sales_prediction.data_room import build_data_room_files

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            retention_years=3,
            visible_locations=["LOC01", "LOC02"],
            location_schema_map={"LOC01": "west_coast", "LOC02": "national"},
            seed=0,
        )
        sf_csv = dr.extra_files["sales_san_francisco.csv"]
        for year in range(2010, 2015):
            self.assertNotIn(str(year), sf_csv)
        self.assertIn("2015", sf_csv)

    def test_furniture_catalog_uses_national_schema_in_per_location_mode(self):
        from src.tasks.sales_prediction.data_room import (
            build_data_room_files,
            get_schema_profile,
        )

        dr = build_data_room_files(
            self.panel,
            self.furniture,
            self.locations,
            before_year=2018,
            visible_locations=["LOC01", "LOC02"],
            location_schema_map={"LOC01": "west_coast", "LOC02": "east_coast"},
            seed=0,
        )
        national = get_schema_profile("national")
        furn = json.loads(dr.furniture_json)
        self.assertIn(national.furniture_id_key, furn[0])
        self.assertEqual(dr.schema_profile_name, "national")


class SchemaTemplateTests(unittest.TestCase):
    """Test that extra template vars (schema profile columns, etc.) don't break rendering."""

    def test_template_renders_with_extra_schema_vars(self):
        prompt = render_template(
            "instance.j2",
            target_year=2027,
            round_num=1,
            target_entity_count=0,
            required_pairs=[],
        )
        self.assertIn("institutional knowledge", prompt)
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", prompt)


class PredictionInstanceDataRoomTests(unittest.TestCase):
    """Test that _PredictionInstance carries data room config."""

    def test_instance_data_room_fields(self):
        from src.tasks.sales_prediction.task import _PredictionInstance

        inst = _PredictionInstance(
            instance_idx=0,
            stage_idx=0,
            variant_id="trending_products",
            target_year=2010,
            target_types=["SOFA", "DESK"],
            forecast_horizon=5,
            visible_types=["SOFA", "DESK"],
            visible_locations=["LOC01", "LOC04"],
            schema_profile="west_coast",
            distractors=["weather_index"],
        )
        self.assertEqual(inst.schema_profile, "west_coast")
        self.assertEqual(inst.visible_locations, ["LOC01", "LOC04"])
        self.assertEqual(inst.distractors, ["weather_index"])

    def test_from_corpus_preserves_data_room(self):
        from src.tasks.sales_prediction.corpus import FrozenCorpusInstance
        from src.tasks.sales_prediction.task import _PredictionInstance

        ci = FrozenCorpusInstance(
            instance_idx=5,
            stage_idx=1,
            variant_id="seasonal_products",
            target_year=2015,
            target_types=["SOFA"],
            forecast_horizon=5,
            visible_types=["SOFA", "DESK"],
            visible_locations=["LOC02", "LOC05"],
            schema_profile="east_coast",
            distractors=["online_search_volume", "marketing_spend"],
            location_open_year={"LOC02": 2012, "LOC05": 2012},
            target_entities=[{"locality": "New York", "furniture_name": "Modern Sofa"}],
            retention_years=3,
            location_assortments={"LOC02": ["SOFA"], "LOC05": ["DESK"]},
            location_schema_map={"LOC02": "east_coast", "LOC05": "national"},
            generator_kind="rich",
            dgp_params={"stage_template": "stage1_axis_mix"},
        )
        inst = _PredictionInstance.from_corpus(ci)
        self.assertEqual(inst.schema_profile, "east_coast")
        self.assertEqual(inst.visible_locations, ["LOC02", "LOC05"])
        self.assertEqual(inst.distractors, ["online_search_volume", "marketing_spend"])
        self.assertEqual(inst.visible_types, ["SOFA", "DESK"])
        self.assertEqual(inst.location_open_year, {"LOC02": 2012, "LOC05": 2012})
        self.assertEqual(
            [(e.locality, e.furniture_name) for e in inst.target_entities],
            [("New York", "Modern Sofa")],
        )
        self.assertEqual(inst.retention_years, 3)
        self.assertEqual(
            inst.location_assortments, {"LOC02": ["SOFA"], "LOC05": ["DESK"]}
        )
        self.assertEqual(
            inst.location_schema_map, {"LOC02": "east_coast", "LOC05": "national"}
        )
        self.assertEqual(inst.generator_kind, "rich")
        self.assertEqual(inst.dgp_params, {"stage_template": "stage1_axis_mix"})


class StructuredPredictionTests(unittest.TestCase):
    """Tests for PredictionEntry, PredictionResponse, build_prediction_schema, and scoring."""

    def test_prediction_entry_defaults_to_none(self):
        from src.tasks.sales_prediction.task import PredictionEntry

        entry = PredictionEntry(locality="SF", furniture_name="Sofa", year=2010)
        self.assertIsNone(entry.items_sold)

    def test_prediction_entry_accepts_value(self):
        from src.tasks.sales_prediction.task import PredictionEntry

        entry = PredictionEntry(
            locality="SF", furniture_name="Sofa", year=2010, items_sold=42.0
        )
        self.assertEqual(entry.items_sold, 42.0)

    def test_build_prediction_schema_constrains_values(self):
        from src.tasks.sales_prediction.task import (
            PredictionResponse,
            build_prediction_schema,
        )

        pairs = [
            {"locality": "SF", "furniture_name": "Sofa", "year": 2010},
            {"locality": "NY", "furniture_name": "Desk", "year": 2011},
        ]
        schema_cls = build_prediction_schema(pairs)
        self.assertTrue(issubclass(schema_cls, PredictionResponse))

        valid = schema_cls(
            predictions=[
                {
                    "locality": "SF",
                    "furniture_name": "Sofa",
                    "year": 2010,
                    "items_sold": 10.0,
                },
                {
                    "locality": "NY",
                    "furniture_name": "Desk",
                    "year": 2011,
                    "items_sold": 20.0,
                },
            ]
        )
        self.assertEqual(len(valid.predictions), 2)

    def test_build_prediction_schema_rejects_invalid_locality(self):
        from pydantic import ValidationError
        from src.tasks.sales_prediction.task import build_prediction_schema

        pairs = [{"locality": "SF", "furniture_name": "Sofa", "year": 2010}]
        schema_cls = build_prediction_schema(pairs)
        with self.assertRaises(ValidationError):
            schema_cls(
                predictions=[
                    {
                        "locality": "INVALID",
                        "furniture_name": "Sofa",
                        "year": 2010,
                        "items_sold": 1.0,
                    },
                ]
            )

    def test_score_structured_predictions_perfect(self):
        from src.tasks.sales_prediction.evaluator import score_structured_predictions
        from src.tasks.sales_prediction.task import PredictionEntry

        predictions = [
            PredictionEntry(
                locality="SF", furniture_name="A", year=2010, items_sold=100
            ),
            PredictionEntry(
                locality="NY", furniture_name="B", year=2010, items_sold=200
            ),
        ]
        gt = [
            {
                "locality": "SF",
                "furniture_name": "A",
                "year": 2010,
                "items_sold": 100,
                "expected_items_sold": 95,
            },
            {
                "locality": "NY",
                "furniture_name": "B",
                "year": 2010,
                "items_sold": 200,
                "expected_items_sold": 190,
            },
        ]
        result = score_structured_predictions(predictions, gt)
        self.assertTrue(result.format_valid)
        self.assertEqual(result.score, 1.0)

    def test_score_structured_predictions_none_items(self):
        from src.tasks.sales_prediction.evaluator import score_structured_predictions
        from src.tasks.sales_prediction.task import PredictionEntry

        predictions = [
            PredictionEntry(
                locality="SF", furniture_name="A", year=2010, items_sold=None
            ),
            PredictionEntry(
                locality="NY", furniture_name="B", year=2010, items_sold=200
            ),
        ]
        gt = [
            {
                "locality": "SF",
                "furniture_name": "A",
                "year": 2010,
                "items_sold": 100,
                "expected_items_sold": 95,
            },
            {
                "locality": "NY",
                "furniture_name": "B",
                "year": 2010,
                "items_sold": 200,
                "expected_items_sold": 190,
            },
        ]
        # WAPE-skill: |0-100| + |200-200| = 100; sum|y| = 300; score = 1 - 100/300
        result = score_structured_predictions(predictions, gt)
        self.assertTrue(result.format_valid)
        self.assertAlmostEqual(result.score, 1.0 - 100.0 / 300.0, places=6)

    def test_score_structured_predictions_all_none(self):
        from src.tasks.sales_prediction.evaluator import score_structured_predictions
        from src.tasks.sales_prediction.task import PredictionEntry

        predictions = [
            PredictionEntry(
                locality="SF", furniture_name="A", year=2010, items_sold=None
            ),
        ]
        gt = [
            {
                "locality": "SF",
                "furniture_name": "A",
                "year": 2010,
                "items_sold": 100,
                "expected_items_sold": 95,
            },
        ]
        # All-zero forecast over a 100-item truth: WAPE-skill = 1 - 100/100 = 0.
        result = score_structured_predictions(predictions, gt)
        self.assertEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()


class TestSandboxLoss(unittest.TestCase):
    """The task sandbox can disappear under a running instance (container exit and
    ``--rm`` removal); ``docker exec`` then fails as an ordinary rc=1 command with a
    daemon error rather than an exception. ``_execute`` must recreate the sandbox,
    restore the data room and retry once instead of scoring the loss against the agent."""

    def _task_with_env(self, outputs: list[dict]):
        from src.tasks.sales_prediction.task import SalesPredictionTask

        task = SalesPredictionTask(num_instances=1, seed=42)
        calls: dict[str, int] = {"execute": 0, "start": 0, "push": 0}

        class _Env:
            def execute(self, action):
                calls["execute"] += 1
                return outputs.pop(0)

        task._env = _Env()  # type: ignore[assignment]
        task._start_container = lambda: calls.__setitem__("start", calls["start"] + 1)  # type: ignore[method-assign]
        task._push_data_room = lambda: calls.__setitem__("push", calls["push"] + 1)  # type: ignore[method-assign]
        return task, calls

    def test_lost_container_is_recreated_and_the_command_retried_once(self):
        task, calls = self._task_with_env([
            {"output": "Error response from daemon: No such container: 0377c137c8a7", "returncode": 1},
            {"output": "ok", "returncode": 0},
        ])
        result = task._execute("ls /app/data")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["output"], "ok")
        self.assertEqual(calls, {"execute": 2, "start": 1, "push": 1})

    def test_ordinary_failure_does_not_recreate(self):
        task, calls = self._task_with_env([
            {"output": "cat: data/missing.csv: No such file or directory", "returncode": 1},
        ])
        result = task._execute("cat data/missing.csv")
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(calls, {"execute": 1, "start": 0, "push": 0})

    def test_second_loss_is_reported_not_looped(self):
        task, calls = self._task_with_env([
            {"output": "Error response from daemon: No such container: 0377c137c8a7", "returncode": 1},
            {"output": "Error response from daemon: No such container: 9a1b2c3d4e5f", "returncode": 1},
        ])
        result = task._execute("ls /app/data")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("No such container", result["output"])
        self.assertEqual(calls, {"execute": 2, "start": 1, "push": 1})
