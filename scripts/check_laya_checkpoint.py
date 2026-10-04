"""Offline checkpoint review using the production classifier and parser snapshot."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import laya_training as training
import voice_worker as worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    import torch
    import laya
    from build_laya_notebook import snapshot
    torch.set_num_threads(2)
    started = time.monotonic()
    model = args.checkpoint.resolve()
    output = model.parent / "review"
    output.mkdir(exist_ok=True)
    contract = snapshot()
    bridge = output / "current-bot-parser.mjs"
    bridge.write_text(contract["bridge"], encoding="utf-8")
    report = json.loads((model / "training_report.json").read_text(encoding="utf-8"))
    print("Loading trained checkpoint", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya.load(str(model), device=device)
    probes = [("Пауза", "pause"), ("Поставь музыку на паузу", "pause"),
              ("Останови воспроизведение на время", "pause"), ("Продолжи музыку", "resume"),
              ("Сними с паузы", "resume"), ("Возобнови воспроизведение", "resume"),
              ("Следующий трек", "skip"), ("Пропусти эту песню", "skip"),
              ("Переключи на следующую песню", "skip"), ("Стоп", "stop"),
              ("Останови музыку", "stop"), ("Выключи музыку и выйди", "stop"),
              ("Сделай громче", "volume_up"), ("Увеличь громкость", "volume_up"),
              ("Прибавь звук", "volume_up"), ("Сделай тише", "volume_down"),
              ("Уменьши громкость", "volume_down"), ("Убавь звук", "volume_down"),
              ("Громкость 50 процентов", "volume_set"), ("Поставь громкость на 25", "volume_set"),
              ("Громкость сто процентов", "volume_set"), ("Включи Басту", "play"),
              ("Поставь песню группы Кино", "play"), ("Найди музыку для работы", "play"),
              ("Включи повтор", "loop_on"), ("Выключи повтор", "loop_off"),
              ("Включи автоплей", "autoplay_on"), ("Выключи автоплей", "autoplay_off"),
              ("Очисти очередь", "queue_clear"), ("Включи голос", "voice_on"),
              ("Выключи голос", "voice_off"), ("Как дела?", "unknown"),
              ("Не ставь музыку на паузу", "unknown"), ("Кто поёт эту песню?", "unknown"),
              ("Завтра у меня экзамен", "unknown"), ("Не выключай музыку", "unknown")]
    commands = training.prepare_commands([
        {"event_id": f"review-{i}", "message": text, "intent": gold,
         "player": {"connected": True, "playing": True, "paused": False, "autoplay": False, "queue_length": 4}}
        for i, (text, gold) in enumerate(probes)], bridge)
    probe_metrics, probe_rows = training.runtime_evaluation(agent, commands, worker, bridge)
    probe_details = [{**row, "message": commands[i]["message"], "canonical": commands[i]["canonical"]}
                     for i, row in enumerate(probe_rows)]
    print("Manual probes", json.dumps(probe_metrics), flush=True)
    dataset = ROOT / ".runtime/datasets/lia-starter-v1"
    rows = training.load_commands([dataset / "commands-synthetic.jsonl", dataset / "bot-commands-labeled.jsonl"],
                                 intent_labels=contract["intent_questions"]["intent"]["criteria"], allow_automatic_labels=True)
    test_ids = {row["event_id"] if isinstance(row, dict) else row for row in report["split_manifest"]["test"]}
    tests = training.prepare_commands([row for row in rows if row["event_id"] in test_ids], bridge)
    assert len(tests) == report["pipeline_metrics"]["n"]
    pipeline_metrics, pipeline_rows = training.runtime_evaluation(agent, tests, worker, bridge)
    print("Reproduced held-out pipeline", json.dumps(pipeline_metrics), flush=True)
    questions = training.planner_questions(contract["intent_questions"])
    planner = []
    for text, intent, tool in [("Муза", "unknown", "wake_ack"), ("Пауза", "pause", "player_control"),
                               ("Включи Басту", "play", "youtube_music_search"),
                               ("Громкость 50 процентов", "volume_set", "player_control"),
                               ("Следующий трек", "skip", "player_control"),
                               ("Включи повтор", "loop_on", "player_control")]:
        for variant in ("eligible", "other_channel", "disabled", "busy"):
            eligible = variant == "eligible"
            state = {"message": text, "phase": "idle" if tool == "wake_ack" else "awaiting",
                     "voice_enabled": variant != "disabled", "in_bot_channel": variant != "other_channel",
                     "is_owner": True, "wake_name": "Муза",
                     "player": {"connected": True, "playing": True, "paused": False, "autoplay": False, "queue_length": 4}}
            if variant == "busy": state["phase"] = "busy"
            gold = {"should_respond": "respond" if eligible else "ignore",
                    "next_tool": tool if eligible else "no_tool", "intent": intent if eligible else "unknown"}
            answers = agent.predict(state, questions)["answers"]
            actual = {k: answer["choice"] for k, answer in answers.items()}
            planner.append({"state": state, "gold": gold, "prediction": actual,
                            "correct": actual == gold})
    result = {"checkpoint": str(model), "device": device, "load": "passed",
              "source_contract_matches_report": {k: v == report["source_sha256"].get(k) for k, v in contract["source_sha256"].items()},
              "reported_pipeline_metrics": report["pipeline_metrics"], "reproduced_pipeline_metrics": pipeline_metrics,
              "heldout_pipeline_errors": [r for r in pipeline_rows if r["prediction"] != r["gold"]],
              "manual_probe_metrics": probe_metrics, "manual_probes": probe_details,
              "planner_probes": planner, "planner_all_fields_correct": sum(r["correct"] for r in planner),
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    (output / "model_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REVIEW", output / "model_review.json", flush=True)
    print("Planner exact matches", result["planner_all_fields_correct"], "/", len(planner), flush=True)


if __name__ == "__main__":
    main()
