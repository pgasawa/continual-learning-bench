import unittest

from src.registry import (
    get_system_class,
    get_task_class,
    list_systems,
    list_tasks,
    register_task,
)


class RegistryDiscoveryTests(unittest.TestCase):
    def test_list_tasks_discovers_task_modules_from_filesystem(self):
        tasks = list_tasks()
        self.assertIn("blind_spectrum_monitoring", tasks)
        self.assertIn("cohort_studies", tasks)
        self.assertIn("codebase_adaptation", tasks)
        self.assertIn("exploitable_poker", tasks)

    def test_list_systems_discovers_system_modules_from_filesystem(self):
        systems = list_systems()
        self.assertIn("ace", systems)
        self.assertIn("human", systems)
        self.assertIn("icl", systems)
        self.assertIn("icl_notepad", systems)
        self.assertIn("schema_card", systems)

    def test_get_task_class_imports_discovered_module(self):
        self.assertEqual(get_task_class("exploitable_poker").__name__, "Poker")

    def test_get_system_class_imports_discovered_module(self):
        self.assertEqual(get_system_class("human").__name__, "Human")

    def test_register_task_requires_r_max(self):
        with self.assertRaises(TypeError):

            @register_task("missing_reference_for_test")
            class MissingReferenceTask:
                pass


if __name__ == "__main__":
    unittest.main()
