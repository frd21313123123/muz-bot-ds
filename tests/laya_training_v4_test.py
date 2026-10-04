"""Behavioral checks for pairing, provenance, leakage and quality selection."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import laya_training as base
import laya_training_v3 as v3
import laya_training_v4 as v4


class ActionV4Tests(unittest.TestCase):
    def test_pair_sampler_balances_classes_and_bounds_historical_mass(self):
        items = []
        labels = sorted(v4.LABELS)
        for label_index, label in enumerate(labels):
            for role in ("authored", "legacy_replay"):
                for category in (["music", "negation", "question", "other"] if label == "unknown" else ["positive"]):
                    for i in range(2):
                        event = f"{label}-{role}-{category}-{i}"
                        for task in ("action_primary", "action_short"):
                            items.append({"event_id": event, "task": task, "labels": labels, "label": label_index,
                                "source_role": role, "category": category, "group_id": str(i), "text_key": event})
        order = v4.paired_indices(items, 16000, 73)
        self.assertEqual(order, v4.paired_indices(items, 16000, 73))
        counts = Counter((items[i]["task"], items[i]["label"]) for i in order)
        self.assertEqual(set(counts.values()), {1000})
        replay = Counter(items[i]["label"] for i in order if items[i]["source_role"] == "legacy_replay")
        self.assertEqual(set(replay.values()), {400})
        for a, b in zip(order[::2], order[1::2]):
            self.assertEqual(items[a]["event_id"], items[b]["event_id"])
            self.assertNotEqual(items[a]["task"], items[b]["task"])
        subtype = Counter(items[i]["category"] for i in order if items[i]["labels"][items[i]["label"]] == "unknown")
        self.assertEqual(set(subtype.values()), {500})

    def test_consistency_loss_penalizes_disagreement_and_has_finite_gradients(self):
        import torch
        same = torch.tensor([[4., 0., 0., 0., 0., 0., 0., 0.]] * 2, requires_grad=True)
        y = torch.tensor([0, 0])
        loss, _, js = v4.paired_loss(same, y)
        self.assertLess(abs(js.item()), 1e-6)
        loss.backward(); self.assertTrue(torch.isfinite(same.grad).all())
        different = torch.tensor([[4., 0., 0., 0., 0., 0., 0., 0.], [0., 4., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
        loss, _, js = v4.paired_loss(different, y)
        self.assertGreater(js.item(), .1)
        loss.backward(); self.assertTrue(torch.isfinite(different.grad).all())
        with self.assertRaises(ValueError): v4.paired_loss(different, torch.tensor([0, 1]))

    def test_selection_rejects_average_improvement_that_collapses_pause(self):
        def metrics(macro, pause):
            return {task: {"macro_f1": macro, "nll": 1., "per_class": {
                label: {"support": 10, "recall": pause if label == "pause" else .9} for label in v4.LABELS}}
                for task in ("primary", "short")}
        balanced, collapsed = metrics(.82, .8), metrics(.88, 0.)
        self.assertGreater(v4.selection_score(balanced), v4.selection_score(collapsed))
        # A candidate passing every model gate takes priority over one with
        # higher recall but failing prompt macro-F1.
        self.assertGreater(v4.selection_score(metrics(.82, .8)), v4.selection_score(metrics(.78, .9)))

    def test_real_worker_thresholds_primary_unknown_and_fallback(self):
        def p(label, confidence): return {"prediction": label, "confidence": confidence}
        self.assertEqual(v4.worker_decision(p("unknown", .2), p("pause", .95))["action"], "unknown")
        self.assertEqual(v4.worker_decision(p("pause", .60), p("stop", .95))["action"], "pause")
        self.assertEqual(v4.worker_decision(p("pause", .59999), p("stop", .95))["action"], "pause")
        self.assertEqual(v4.worker_decision(p("pause", .59), p("resume", .75))["action"], "resume")
        self.assertEqual(v4.worker_decision(p("pause", .59), p("resume", .749))["action"], "unknown")

    def test_dataset_provenance_integrity_unique_inputs_and_logs(self):
        data = ROOT / ".runtime/datasets/lia-action-v4"
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["file_sha256"].items(): self.assertEqual(base.sha256_file(data / name), digest)
        rows = v4.load_commands(data / "commands.jsonl", data / "verification_parser.mjs")
        by_id = {r["event_id"]: r for r in rows}
        ids = json.loads((data / "split-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(by_id), {i for part in ids.values() for i in part})
        self.assertEqual(sum(map(len, ids.values())), len(rows))
        positions, gold = {}, {}
        for fold, event_ids in ids.items():
            self.assertEqual({by_id[i]["intent"] for i in event_ids}, v4.LABELS)
            for ident in event_ids:
                row = by_id[ident]
                for key in [("family", row["group_id"]), ("raw", base.normalized(row["message"])), ("input", v3.runtime_text(row["canonical"]))]:
                    self.assertTrue(key not in positions or positions[key] == fold); positions[key] = fold
                text = v3.runtime_text(row["canonical"])
                self.assertTrue(text not in gold or gold[text] == row["intent"]); gold[text] = row["intent"]
                if fold != "train": self.assertEqual(row["source_role"], "authored")
        self.assertEqual(sum(r["origin"] == "bot_log" for r in rows), 31)
        for name in ("bot-logs-unreviewed.jsonl", "bot-review-queue.jsonl", "bot-label-review.csv"):
            self.assertEqual((data / name).read_bytes(), (ROOT / ".runtime/datasets/lia-semantic-v3" / name).read_bytes())
        with zipfile.ZipFile(data.with_suffix(".zip")) as archive: self.assertIsNone(archive.testzip())

    def test_model_hints_are_refused_and_finish_only_cannot_skip_training(self):
        data = ROOT / ".runtime/datasets/lia-action-v4"
        row = base.read_jsonl([data / "commands.jsonl"])[0]
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as name:
            path = Path(name) / "bad.jsonl"
            path.write_text(json.dumps({**row, "suggested_intent": "pause"}), encoding="utf-8")
            with self.assertRaises(ValueError): v4.load_commands(path, data / "verification_parser.mjs")
            progress = Path(name) / "progress.json"
            cfg = {"epochs": 12, "patience": 4}
            progress.write_text(json.dumps({"config": cfg, "completed_epoch": 1, "no_improvement": 0}), encoding="utf-8")
            with self.assertRaises(ValueError): v4.finished_info(progress, cfg)

    def test_candidate_export_and_final_cell_without_training_globals(self):
        import shutil
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as name:
            work = Path(name); directory = work / "candidates/laya-muz-bot-controls-v4"
            directory.mkdir(parents=True)
            (directory / "toy-test.txt").write_text("unit test; no weights", encoding="utf-8")
            env = {"Path": Path, "shutil": shutil, "training": base, "v3": v3, "SMOKE_TEST": False, "WORKDIR": work, "zipfile": zipfile}
            exec((ROOT / "scripts/kaggle_v4_workflow.py").read_text(encoding="utf-8"), env)
            filename = env["package_candidate"](directory, False)
            self.assertTrue(filename.endswith("-CANDIDATE.zip"))
            with zipfile.ZipFile(filename) as archive:
                self.assertIsNone(archive.testzip())
                status = json.loads(archive.read(directory.name + "/qualification.json"))
                self.assertFalse(status["qualified_on_synthetic_holdouts"])
                hashes = json.loads(archive.read(directory.name + "/sha256.json"))
                self.assertEqual(hashes["toy-test.txt"], base.sha256_file(directory / "toy-test.txt"))
            source = (ROOT / "scripts/kaggle_v4_finish.py").read_text(encoding="utf-8")
            source = source.replace('default_root / "lia-action-v4"', f"Path({str(work)!r})")
            exec(source, {})
            self.assertTrue((work / "lia-action-v4-diagnostics.zip").exists())


if __name__ == "__main__": unittest.main()
