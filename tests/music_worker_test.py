"""Offline worker-contract tests: no Whisper, Laya or model weights required."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import os
import tempfile

spec = importlib.util.spec_from_file_location("voice_worker", Path(__file__).resolve().parents[1] / "scripts/voice_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def candidates(count=5):
    return [{"index": i, "title": f"Track {i}", "artist": "Artist", "duration": "3:00"} for i in range(count)]


class Agent:
    def __init__(self, versions=None, selected=0):
        self.versions = versions or {}
        self.selected = selected
        self.calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.calls.append((states, questions))
        return [{"answers": {"version": {"choice": "live" if item["index"] in self.versions else "original",
                 "probabilities": {"live": self.versions.get(item["index"], 0.1)}}}} for item in states]

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions))
        if "best_track" in questions:
            assert str(self.selected) in questions["best_track"]["criteria"]
            return {"answers": {"best_track": {"choice": str(self.selected), "answer_confidence": 0.95}}}
        return {"answers": {"next_tool": {"choice": "search", "answer_confidence": 0.9},
                            "search_result_policy": {"choice": "live", "answer_confidence": 0.9}}}


class V3Agent:
    cfg = {"model_name": "laya-muz-bot-ds-v3"}

    def __init__(self, choice="c4", source="parsed_search_query"):
        self.calls = []
        self.choice, self.source = choice, source

    def predict_batch(self, states, questions, **kwargs):
        return [{"answers": {"version": {"choice": "live" if item["index"] == 4 else "original",
                 "probabilities": {"live": 0.99 if item["index"] == 4 else 0.01}}}} for item in states]

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions))
        if "next_tool" in questions:
            return {"answers": {"next_tool": {"choice": "youtube_music_search", "answer_confidence": 0.99},
                                "query_source": {"choice": self.source, "answer_confidence": 0.98}}}
        if "search_result_policy" in questions:
            return {"answers": {"search_result_policy": {"choice": "rerank_results", "answer_confidence": 0.99}}}
        return {"answers": {"best_track": {"choice": self.choice, "answer_confidence": 0.97}}}


class MusicWorkerTests(unittest.TestCase):
    def test_stt_upgrade_defaults_and_existing_manifests_keep_the_wake_model_explicit(self):
        with patch.object(worker, 'PROFILE', 'heavy'), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(worker.whisper_names({}, True), ('large-v3-turbo', 'small'))
        with patch.object(worker, 'PROFILE', 'heavy'), patch.dict(os.environ, {'WHISPER_MODEL': 'large-v3', 'WHISPER_WAKE_MODEL': 'base'}):
            self.assertEqual(worker.whisper_names({}, True), ('large-v3', 'base'))
            self.assertEqual(worker.whisper_names({'whisper': 'small'}), ('small', 'small'))
            self.assertEqual(worker.whisper_names({'whisper': 'large-v3-turbo', 'whisper_wake': 'small'}),
                             ('large-v3-turbo', 'small'))

    def test_light_uses_one_cpu_model_and_ignores_heavy_overrides(self):
        with patch.object(worker, 'PROFILE', 'light'), patch.dict(os.environ, {'WHISPER_MODEL': 'large-v3', 'VOICE_DEVICE': 'cuda'}, clear=True):
            self.assertEqual(worker.whisper_names({}, True), ('small', 'small'))
            for name in ('tiny', 'base', 'small'):
                with patch.dict(os.environ, {'WHISPER_LIGHT_MODEL': name}):
                    self.assertEqual(worker.whisper_names({}, True), (name, name))
            with patch.dict(os.environ, {'WHISPER_LIGHT_MODEL': 'large-v3'}):
                with self.assertRaises(ValueError):
                    worker.whisper_names({}, True)

    def test_v3_route_uses_parsed_request_and_real_player_state(self):
        agent = V3Agent()
        state = {"message": "включи Numb", "selected_track": None,
                 "parsed_request": {"kind": "play", "search_query": "Numb", "request_variant": None},
                 "player": {"connected": True, "playing": True, "paused": False, "autoplay": False, "queue_length": 3}}
        decision = worker.music_route(agent, state)
        self.assertTrue(decision["defer_result_policy"])
        self.assertEqual(decision["confidence"], 0.98)
        model_state = agent.calls[0][0]
        self.assertEqual(model_state["phase"], "request")
        self.assertEqual(model_state["parsed_request"], state["parsed_request"])
        self.assertEqual(model_state["player"], state["player"])
        self.assertEqual(worker.music_route(V3Agent(source="selected_track"), state)["next_tool"], "unknown")

    def test_v3_policy_normalizes_aliases_without_mutating_candidates(self):
        agent = V3Agent()
        items = candidates()
        items[4]["title"] = "Numb концертная версия"
        decision = worker.music_result_policy(agent, "Numb концертная версия", items)
        self.assertEqual(decision["search_result_policy"], "rerank_results")
        state = agent.calls[0][0]
        self.assertEqual(state["phase"], "tool_result")
        self.assertEqual(state["query"], "Numb live")
        self.assertEqual(state["request_variant"], "live")
        self.assertEqual(state["results"][4]["id"], "c4")
        self.assertEqual(state["results"][4]["duration"], 180)
        self.assertEqual(items[4]["title"], "Numb концертная версия")

    def test_v3_selection_translates_candidate_ids_and_explicit_none(self):
        for choice, expected in (("c4", {"best_track": 4, "confidence": 0.97}),
                                 ("none", {"best_track": -1, "no_match": True, "confidence": 0.97})):
            agent = V3Agent(choice)
            self.assertEqual(worker.music_rerank(agent, "Numb Live", candidates()), expected)
            state, questions = agent.calls[0]
            self.assertEqual(state["phase"], "rerank")
            self.assertEqual(state["query"], "Numb live")
            self.assertEqual(set(questions["best_track"]["criteria"]), {"c4", "none"})
        with self.assertRaises(ValueError):
            worker.music_rerank(V3Agent("c9"), "Numb live", candidates())

    def test_local_model_path_overrides_remote_revision_and_missing_path_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LAYA_MODEL_PATH": directory}):
                self.assertEqual(worker.laya_source({"laya_revision": "old"}), (str(Path(directory).resolve()), None))
            with patch.dict(os.environ, {"LAYA_MODEL_PATH": str(Path(directory) / "missing")}):
                with self.assertRaises(FileNotFoundError):
                    worker.laya_source({})

    def test_single_matching_version_retains_original_index(self):
        agent = Agent({4: 0.8})
        self.assertEqual(worker.music_rerank(agent, "Numb live", candidates()), {"best_track": 4, "confidence": 0.8})
        self.assertEqual(len(agent.calls), 1)
        self.assertNotIn("query", agent.calls[0][0][0])

    def test_multiple_matching_versions_use_final_model_choice(self):
        agent = Agent({1: 0.7, 3: 0.8}, selected=3)
        self.assertEqual(worker.music_rerank(agent, "Numb live", candidates()), {"best_track": 3, "confidence": 0.8})
        self.assertEqual(set(agent.calls[1][1]["best_track"]["criteria"]), {"1", "3"})

    def test_no_matching_version_and_weak_match_trigger_fallback(self):
        for versions in ({}, {1: 0.59}):
            self.assertEqual(worker.music_rerank(Agent(versions), "Numb live", candidates()), {"best_track": 0, "confidence": 0})

    def test_without_explicit_version_uses_direct_candidate_choice(self):
        agent = Agent(selected=2)
        self.assertEqual(worker.music_rerank(agent, "Numb", candidates()), {"best_track": 2, "confidence": 0.95})
        self.assertEqual(len(agent.calls), 1)

    def test_route_normalizes_alias_without_changing_original_state(self):
        state = {"message": "включи Numb концертная версия", "selected_track": None}
        agent = Agent()
        decision = worker.music_route(agent, state)
        self.assertEqual(decision["next_tool"], "youtube_music_search")
        self.assertEqual(decision["search_result_policy"], "rerank_results")
        self.assertEqual(agent.calls[0][0]["message"], "включи Numb live")
        self.assertEqual(state["message"], "включи Numb концертная версия")

    def test_invalid_worker_input_is_rejected(self):
        for items in ([], candidates(6), [{"index": 9, "title": "Bad", "artist": "", "duration": "?"}],
                      [{"index": 0, "title": "x" * 241, "artist": "", "duration": "?"}]):
            with self.assertRaises(ValueError):
                worker.music_rerank(Agent(), "Numb live", items)
        with self.assertRaises(ValueError):
            worker.music_route(Agent(), {"message": "Numb", "selected_track": {"title": "Old"}})


class PreparedWhisperTests(unittest.TestCase):
    def test_runtime_uses_prepared_path_instead_of_a_new_environment_override(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(worker, 'PROFILE', 'light'):
            directory = Path(temporary)
            for name in ('model.bin', 'config.json', 'tokenizer.json'):
                (directory / name).touch()
            with patch.dict(os.environ, {'WHISPER_LIGHT_MODEL_PATH': '/missing-model'}):
                self.assertEqual(worker.light_whisper_path({'whisper_path': temporary}), directory.resolve())
                self.assertIsNone(worker.light_whisper_path({}))
                with self.assertRaises(ValueError):
                    worker.light_whisper_path({}, prepare=True)

    def test_incomplete_model_fails_before_attempting_online_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(worker, 'PROFILE', 'light'):
            (Path(temporary) / 'model.bin').touch()
            with self.assertRaises(ValueError):
                worker.light_whisper_path({'whisper_path': temporary})

    def test_heavy_profile_ignores_light_model_override(self):
        with patch.object(worker, 'PROFILE', 'heavy'), patch.dict(os.environ, {'WHISPER_LIGHT_MODEL_PATH': '/missing-model'}):
            self.assertIsNone(worker.light_whisper_path({'whisper_path': '/missing-model'}))
            self.assertIsNone(worker.light_whisper_path({}, prepare=True))


if __name__ == "__main__":
    unittest.main()
