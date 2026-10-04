"""V3 regressions: semantic/policy separation and augmentation-neutral sampling."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import laya_training_v3 as v3


class SemanticTrainingTests(unittest.TestCase):
    def test_class_task_and_negative_subtype_mass(self):
        items = []
        for task in ("primary", "short"):
            for label in range(8):
                for category in (["negation", "question", "music", "other"] if label == 0 else ["positive"]):
                    for i in range(100 if category == "music" else 1):
                        items.append({"task": task, "qid": "action", "label": label,
                            "labels": ["unknown"] + [str(n) for n in range(1, 8)],
                            "group_id": "family", "text_key": "text", "category": category})
        order = v3.balanced_indices(items, 16000, 17)
        counts = Counter((items[i]["task"], items[i]["label"]) for i in order)
        self.assertEqual(set(counts.values()), {1000})
        negatives = Counter(items[i]["category"] for i in order if items[i]["label"] == 0)
        self.assertTrue(all(400 < n < 600 for n in negatives.values()))
        self.assertEqual(order, v3.balanced_indices(items, 16000, 17))

    def test_punctuation_duplicates_do_not_inflate_text(self):
        common = {"task": "semantic", "qid": "intent", "label": 0, "labels": ["pause"],
                  "group_id": "g", "category": "positive"}
        items = [{**common, "text_key": "oversampled"} for _ in range(1000)]
        items.append({**common, "text_key": "one"})
        order = v3.balanced_indices(items, 2000, 2)
        count = sum(items[i]["text_key"] == "one" for i in order)
        self.assertTrue(850 < count < 1150)

    def test_policy_never_calls_model_when_blocked_or_waking(self):
        def forbidden(text): self.fail("Model was called in a blocked/wake session")
        state = {"voice_enabled": True, "in_bot_channel": True, "is_owner": True, "phase": "awaiting"}
        probe = {"canonical": "Стоп", "wake_match": False, "music_kind": None}
        for patch in ({"voice_enabled": False}, {"in_bot_channel": False}, {"is_owner": False}, {"phase": "busy"}):
            self.assertEqual(v3.policy_decision({**state, **patch}, forbidden, probe)["next_tool"], "no_tool")
        wake = v3.policy_decision({**state, "phase": "idle", "is_owner": False}, forbidden, {**probe, "wake_match": True})
        self.assertEqual(wake, {"should_respond": "respond", "next_tool": "wake_ack", "intent": "unknown"})

    def test_semantic_intent_in_eligible_session_and_invalid_output(self):
        state = {"voice_enabled": True, "in_bot_channel": True, "is_owner": True, "phase": "awaiting"}
        probe = {"canonical": "Поставь музыку НА ПАУЗУ!", "wake_match": False, "music_kind": None}
        seen = []
        def model(text): seen.append(text); return "pause"
        self.assertEqual(v3.policy_decision(state, model, probe)["intent"], "pause")
        self.assertEqual(seen, ["Поставь музыку на паузу"])
        self.assertEqual(v3.policy_decision(state, lambda _: "unexpected", probe)["next_tool"], "no_tool")
        self.assertEqual(v3.policy_decision(state, lambda _: "play", {**probe, "music_kind": "video"})["next_tool"], "direct_youtube_video")

    def test_actual_embedded_data_has_no_contradictory_model_input(self):
        import laya_training as base
        from build_laya_notebook import snapshot
        root = Path(__file__).resolve().parents[1]
        data = root / ".runtime/datasets/lia-semantic-v3"
        rows = v3.load_commands(data / "semantic-commands.jsonl", data / "verification_parser.mjs",
                                snapshot()["intent_questions"]["intent"]["criteria"])
        seen = {}
        for row in rows:
            text = v3.runtime_text(row["canonical"])
            self.assertTrue(text not in seen or seen[text] == row["intent"])
            seen[text] = row["intent"]
            self.assertFalse(any(k in row for k in ("phase", "is_owner", "model_decision", "gold")))
        ids = __import__("json").loads((data / "split-manifest.json").read_text())
        log_ids = {r["event_id"] for r in rows if r["origin"] == "bot_log"}
        self.assertEqual(len(log_ids), 31)
        self.assertTrue(log_ids <= set(ids["train"]))
        self.assertEqual(len(base.read_jsonl([data / "bot-logs-unreviewed.jsonl"])), 44)
        self.assertEqual(len(base.read_jsonl([data / "bot-review-queue.jsonl"])), 13)

    def test_joint_split_keeps_families_and_all_effective_task_labels(self):
        import laya_training as base
        from build_laya_notebook import snapshot
        import json
        root = Path(__file__).resolve().parents[1]
        data = root / ".runtime/datasets/lia-semantic-v3"
        rows = []
        for task in ("action", "semantic"):
            loaded = v3.load_commands(data / (task + "-commands.jsonl"), data / "verification_parser.mjs",
                                     snapshot()["intent_questions"]["intent"]["criteria"])
            rows.extend({**r, "task": task} for r in loaded)
        by_id = {r["event_id"]: r for r in rows}
        ids = json.loads((data / "split-manifest.json").read_text())
        self.assertEqual(set(by_id), {i for part in ids.values() for i in part})
        locations = {}
        for name, keys in ids.items():
            coverage = {task: {} for task in ("action", "semantic")}
            for key in keys:
                row = by_id[key]
                for family in [("family", row["group_id"])] + [("text", base.normalized(row[k])) for k in ("message", "canonical")]:
                    self.assertTrue(family not in locations or locations[family] == name)
                    locations[family] = name
                label = row["intent"] if row["task"] == "semantic" or row["intent"] in v3.CONTROLS else "unknown"
                coverage[row["task"]].setdefault(label, set()).add(base.normalized(row["canonical"]))
                if row["origin"] == "bot_log": self.assertEqual(name, "train")
            self.assertEqual(len(coverage["action"]), 8)
            self.assertEqual(len(coverage["semantic"]), 16)
            self.assertTrue(all(len(texts) >= 2 for classes in coverage.values() for texts in classes.values()))

    def test_candidate_zip_marks_failure_and_diagnostics_survive_empty_globals(self):
        import json, shutil, tempfile, zipfile
        import laya_training as base
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="v3-artifact-test-", dir=root / ".runtime") as temporary:
            work = Path(temporary)
            directory = work / "candidates" / "unit-test"
            directory.mkdir(parents=True)
            (directory / "toy-test.txt").write_text("unit test, no model", encoding="utf-8")
            env = {"Path": Path, "shutil": shutil, "training": base, "v3": v3, "SMOKE_TEST": False, "WORKDIR": work}
            exec((root / "scripts/kaggle_v3_workflow.py").read_text(encoding="utf-8"), env)
            filename = env["package_candidate"](directory, "unit-test", False)
            self.assertTrue(filename.endswith("-CANDIDATE.zip"))
            with zipfile.ZipFile(filename) as archive:
                self.assertIsNone(archive.testzip())
                status = json.loads(archive.read("unit-test/qualification.json"))
                self.assertFalse(status["qualified_on_synthetic_holdouts"])
                hashes = json.loads(archive.read("unit-test/sha256.json"))
                self.assertEqual(hashes["toy-test.txt"], base.sha256_file(directory / "toy-test.txt"))
            v3.atomic_json(work / "resume" / "action.json", {"completed_epoch": 1, "qualification": "not evaluated"})
            finish = (root / "scripts/kaggle_v3_finish.py").read_text(encoding="utf-8").replace('"/kaggle/working/lia-v3"', repr(str(work)))
            exec(finish, {})
            self.assertTrue((work / "lia-v3-diagnostics.zip").exists())


if __name__ == "__main__": unittest.main()
