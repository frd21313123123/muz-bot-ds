"""Offline tests for reviewed-data ingestion and leakage-free Laya partitions."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("laya_training", ROOT / "scripts/laya_training.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.directory / name
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_predictions_are_not_gold_and_explicit_review_overrides_them(self):
        path = self.write("logs.jsonl", [{"event_id": "e1", "schema_version": 1,
            "state": {"message": "пауза", "canonical_message": "wrong hint"},
            "suggested_intent": "skip", "validated_intent": {"action": "skip"},
            "outcome": "changed", "gold": None, "label_status": "unreviewed"}])
        with self.assertRaisesRegex(ValueError, "No manually reviewed"):
            training.load_commands([path], intent_labels=["pause", "skip"])
        reviews = self.write("reviewed.jsonl", [{"event_id": "e1", "intent": "pause"}])
        rows = training.load_commands([path], reviews, ["pause", "skip"])
        self.assertEqual(rows[0]["intent"], "pause")
        self.assertEqual(rows[0]["message"], "пауза")
        self.assertNotIn("canonical_message", rows[0])

    def test_existing_export_and_direct_rows_are_supported(self):
        path = self.write("export.jsonl", [{"source_event_id": "e2", "state": json.dumps({"message": "тише"}),
            "questions": "{}", "gold": json.dumps({"intent": "volume_down"}), "tags": ["human-reviewed"]}])
        rows = training.load_commands([path], intent_labels=["volume_down"])
        self.assertEqual(rows[0]["intent"], "volume_down")
        direct = self.write("direct.jsonl", [{"event_id": "e3", "message": "pause", "intent": "pause", "reviewed": True}])
        self.assertEqual(training.load_commands([direct], intent_labels=["pause"])[0]["message"], "pause")

    def test_automatic_labels_require_opt_in_and_explicit_provenance(self):
        row = {"event_id": "s1", "message": "пауза", "intent": "pause", "reviewed": False,
               "origin": "synthetic", "label_status": "rule_verified", "label_source": "synthetic_template_checked"}
        path = self.write("synthetic.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "No manually reviewed"):
            training.load_commands([path], intent_labels=["pause"])
        accepted = training.load_commands([path], intent_labels=["pause"], allow_automatic_labels=True)
        self.assertEqual(accepted[0]["origin"], "synthetic")
        row["label_source"] = "model_prediction"
        path = self.write("predictions.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "No manually reviewed"):
            training.load_commands([path], intent_labels=["pause"], allow_automatic_labels=True)

    def test_joint_task_grouping_keeps_contextual_copies_in_one_partition(self):
        runtime = [{"event_id": f"r{i}", "group_id": f"family-{i}", "message": f"command {i}",
                    "canonical": f"command {i}", "intent": "pause"} for i in range(30)]
        planner = [{"event_id": f"p{i}", "group_id": f"family-{i}", "message": f"command {i}",
                    "canonical": f"command {i}", "intent": "respond|player_control|pause", "task": "planner"}
                   for i in range(30)]
        splits = training.grouped_split(runtime + planner)
        location = {r["event_id"]: name for name, rows in splits.items() for r in rows}
        for i in range(30):
            self.assertEqual(location[f"r{i}"], location[f"p{i}"])

    def test_duplicate_ids_and_missing_reviews_fail(self):
        row = {"event_id": "e1", "message": "pause", "intent": "pause", "reviewed": True}
        path = self.write("duplicates.jsonl", [row, row])
        with self.assertRaisesRegex(ValueError, "unique"):
            training.load_commands([path], intent_labels=["pause"])
        path = self.write("single.jsonl", [row])
        reviews = self.write("reviews.jsonl", [{"event_id": "missing", "intent": "pause"}])
        with self.assertRaisesRegex(ValueError, "not found"):
            training.load_commands([path], reviews, ["pause"])

    def test_canonical_and_context_variants_cannot_leak(self):
        rows = [{"event_id": f"e{i}", "group_id": f"g{i}", "message": f"phrase {i}",
                 "canonical": f"canonical {i}", "intent": "pause" if i % 2 else "unknown"} for i in range(60)]
        rows += [{**rows[0], "event_id": "alias", "group_id": "other", "message": rows[0]["canonical"]},
                 {**rows[0], "event_id": "context", "group_id": "third", "intent": "pause"}]
        splits = training.grouped_split(rows)
        self.assertEqual(splits, training.grouped_split(rows))
        location = {r["event_id"]: split for split, records in splits.items() for r in records}
        self.assertEqual(location["e0"], location["alias"])
        self.assertEqual(location["e0"], location["context"])
        self.assertEqual(len(location), len(rows))
        for records in splits.values():
            self.assertEqual({r["intent"] for r in records}, {"pause", "unknown"})

    def test_missing_label_families_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            training.grouped_split([{"event_id": "e1", "group_id": "g", "message": "pause", "canonical": "pause", "intent": "pause"}])

    def test_reviewed_intent_is_converted_to_the_runtime_action_schema(self):
        worker_spec = importlib.util.spec_from_file_location("training_worker_contract", ROOT / "scripts/voice_worker.py")
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        commands = [{"event_id": "e1", "canonical": "next", "intent": "skip"},
                    {"event_id": "e2", "canonical": "включи Numb", "intent": "play"},
                    {"event_id": "e3", "canonical": "включи автоплей", "intent": "autoplay_on"}]
        records = training.runtime_records(commands, worker)
        self.assertEqual(len(records), 6)
        self.assertEqual([r["gold"]["action"] for r in records], ["skip", "skip", "unknown", "unknown", "unknown", "unknown"])
        self.assertTrue(all(isinstance(r["state"], str) and set(r["questions"]) == {"action"} for r in records))
        self.assertEqual(records[0]["state"], "Next")

    def test_notebook_snapshot_bridge_matches_live_contract(self):
        notebook = json.loads((ROOT / "notebooks/laya_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        snapshot = None
        for cell in notebook["cells"]:
            source = cell["source"]
            self.assertIsInstance(source, str)
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                tree = ast.parse(source)
                if source.startswith("SNAPSHOT ="):
                    snapshot = json.loads(ast.literal_eval(tree.body[0].value.args[0]))
        self.assertIsNotNone(snapshot)
        for path, expected in snapshot["source_sha256"].items():
            self.assertEqual(training.sha256_file(ROOT / path), expected, f"Regenerate notebook: {path}")
        self.assertEqual(snapshot["helper"], (ROOT / "scripts/laya_training.py").read_text(encoding="utf-8"))
        script = self.directory / "bridge.mjs"
        script.write_text(snapshot["bridge"], encoding="utf-8")
        rows = [
            {"message": "next", "decision": {"action": "skip", "confidence": 0.99}},
            {"message": "поставь на паузу", "decision": {"action": "resume", "confidence": 0.99}},
            {"message": "включи Numb", "decision": {"action": "stop", "confidence": 0.99}},
            {"message": "не выключай музыку", "decision": {"action": "stop", "confidence": 0.99}},
            {"message": "включи музыку", "player": {"paused": True}, "decision": {"action": "resume", "confidence": 0.99}},
            {"message": "громче на 50", "decision": {"action": "volume_up", "confidence": 0.99}},
            {"message": "включи автоплей", "decision": {"action": "unknown", "confidence": 0.99}},
            {"message": "https://youtu.be/dQw4w9WgXcQ", "decision": {"action": "unknown", "confidence": 0.1}},
        ]
        result = training.node_bridge(rows, script)
        self.assertEqual([r["final_action"] for r in result], ["skip", "unknown", "play", "unknown", "resume", "unknown", "autoplay_on", "play"])
        self.assertEqual(result[0]["canonical"], "Следующий трек")
        self.assertFalse(result[2]["model_required"])

    def test_planner_cannot_learn_execution_without_owner_or_channel(self):
        notebook = json.loads((ROOT / "notebooks/laya_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
        source = next(c["source"] for c in notebook["cells"] if c["source"].startswith("SNAPSHOT ="))
        snapshot = json.loads(ast.literal_eval(ast.parse(source).body[0].value.args[0]))
        bridge = self.directory / "planner-bridge.mjs"
        bridge.write_text(snapshot["bridge"], encoding="utf-8")
        questions = training.planner_questions({"intent": {"type": "choice", "instructions": "intent", "criteria": {"unknown": "unknown", "pause": "pause", "play": "play"}}})
        row = {"event_id": "p1", "group_id": "g", "reviewed": True,
               "state": {"message": "пауза", "phase": "awaiting", "voice_enabled": True,
                         "in_bot_channel": True, "is_owner": False, "wake_name": "Муза", "player": None},
               "gold": {"should_respond": "respond", "next_tool": "player_control", "intent": "pause"}}
        path = self.write("planner.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            training.load_planner([path], questions, bridge)
        row["gold"] = {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}
        path = self.write("planner.jsonl", [row])
        self.assertEqual(len(training.load_planner([path], questions, bridge)), 1)
        row["state"].update(phase="idle", message="Muza")
        row["gold"] = {"should_respond": "respond", "next_tool": "wake_ack", "intent": "unknown"}
        path = self.write("planner.jsonl", [row])
        self.assertEqual(len(training.load_planner([path], questions, bridge)), 1)


if __name__ == "__main__":
    unittest.main()
