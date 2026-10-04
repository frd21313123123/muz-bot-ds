"""Regressions for the failure observed in the submitted V1 model."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import laya_training_v2 as balanced


class BalancedTrainingTests(unittest.TestCase):
    def test_class_and_task_mass_is_independent_of_row_counts(self):
        items = []
        for task, classes in (("action", 8), ("planner", 2)):
            for label in range(classes):
                for group in range(3):
                    for _ in range(100 if label == 0 else 1):
                        items.append({"task": task, "qid": "q", "label": label, "group_id": f"g{group}"})
        order = balanced.balanced_indices(items, 3200, 42)
        counts = Counter((items[i]["task"], items[i]["label"]) for i in order)
        self.assertEqual({counts["action", label] for label in range(8)}, {200})
        self.assertEqual({counts["planner", label] for label in range(2)}, {800})
        self.assertEqual(order, balanced.balanced_indices(items, 3200, 42))

    def test_duplicate_augmentation_does_not_dominate_family_sampling(self):
        items = [{"task": "t", "qid": "q", "label": 0, "group_id": "large"} for _ in range(1000)]
        items.append({"task": "t", "qid": "q", "label": 0, "group_id": "small"})
        order = balanced.balanced_indices(items, 2000, 17)
        small = sum(items[i]["group_id"] == "small" for i in order)
        self.assertGreater(small, 850)
        self.assertLess(small, 1150)

    def test_selection_rejects_majority_class_collapse(self):
        collapsed = {"task": {"accuracy": 0.84, "macro_f1": 0.057, "nll": 0.4,
                              "per_class": {"unknown": {"support": 1506, "recall": 1.}, "play": {"support": 82, "recall": 0.}}}}
        useful = {"task": {"accuracy": 0.80, "macro_f1": 0.78, "nll": 0.5,
                           "per_class": {"unknown": {"support": 1506, "recall": .80}, "play": {"support": 82, "recall": .75}}}}
        self.assertGreater(balanced.selection_score(useful), balanced.selection_score(collapsed))
        self.assertFalse(balanced.metric_gate(collapsed)["passed"])

    def test_gate_checks_each_class_despite_high_macro_f1(self):
        metrics = {"task": {"macro_f1": .95, "per_class": {"respond": {"support": 301, "recall": 4 / 301},
                                                            "ignore": {"support": 1491, "recall": .99}}}}
        self.assertFalse(balanced.metric_gate(metrics)["passed"])
        metrics["task"]["per_class"]["respond"]["recall"] = .9
        self.assertTrue(balanced.metric_gate(metrics)["passed"])


if __name__ == "__main__":
    unittest.main()
