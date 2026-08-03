import unittest

from src.tasks.schedules import clear_task_schedule_cache, list_task_schedules
from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
from src.tasks.exploitable_poker.task import Poker


class TaskRolloutTests(unittest.TestCase):
    def tearDown(self):
        clear_task_schedule_cache()

    def test_list_task_schedules_returns_expected_poker_schedules(self):
        schedules = list_task_schedules("exploitable_poker")
        self.assertEqual(
            [schedule.id for schedule in schedules],
            [
                "calling_station_only",
                "calling_station_then_fit_or_fold",
                "default",
                "fit_or_fold_only",
                "loose_aggressive_only",
                "quick_test",
            ],
        )

    def test_bsm_rollout_reset_rebinds_first_stage(self):
        task = BlindSpectrumMonitoringTask(
            schedule="default",
            corpus_id="mixed_grid_lifecycle",
        )
        task.reset()
        # Simulate reaching the last stage by activating it directly.
        last_stage_idx = len(task._schedule_stages) - 1
        task._activate_schedule_stage(last_stage_idx)
        self.assertEqual(task.variant, "full_grid_active")

        task.reset()

        self.assertEqual(task.variant, "five_ch_wide")
        self.assertEqual(task.W, 15.0)
        self.assertEqual(task.G, 9.0)

    def test_poker_rollout_num_instances(self):
        task = Poker(schedule="quick_test")
        self.assertEqual(task.num_instances, 5)


if __name__ == "__main__":
    unittest.main()
