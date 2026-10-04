"""Private V2 corpus: preserve logged speech and add declared, contract-checked families."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import laya_training as training
from build_laya_notebook import snapshot
from build_lia_dataset import automatic_row, player, variants

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".runtime/datasets/lia-starter-v1"
OUT = ROOT / ".runtime/datasets/lia-balanced-v2"
SEED = 20261002

EXTRA = {
    "pause": ["поставь текущую песню на паузу", "приостанови этот трек", "приостанови текущую песню",
              "приостанови аудио", "приостанови текущий трек", "приостановите воспроизведение",
              "приостанови проигрыватель", "включи паузу сейчас", "поставь на паузу сейчас",
              "приостановить текущую песню", "приостанови композицию", "приостановите проигрывание"],
    "resume": ["продолжи текущую песню", "продолжи этот трек", "продолжи проигрыватель",
               "возобнови этот трек", "возобнови текущую песню", "возобновите воспроизведение",
               "продолжите воспроизведение", "продолжи аудио", "возобнови аудио",
               "продолжить воспроизведение", "продолжи текущий трек", "возобновить проигрывание"],
    "skip": ["пропусти этот трек", "пропусти эту музыку", "пропусти текущую песню",
             "пропустить эту песню", "пропустите текущий трек", "пропусти аудио",
             "пропусти композицию", "пропустите песню", "пропусти трек сейчас",
             "пропусти эту песню сейчас", "следующая песня", "следующий сейчас"],
    "stop": ["останови проигрыватель", "останови текущую музыку", "остановите воспроизведение",
             "останови этот трек", "останови аудио", "остановить проигрывание",
             "выключи проигрыватель", "останови текущую песню", "остановите проигрывание",
             "отключи музыку", "отключи плеер", "выйди из канала"],
    "volume_up": ["увеличь громкость плеера", "сделай эту песню громче", "сделай этот трек громче",
                  "увеличь громкость трека", "сделай проигрывание громче", "увеличить громкость плеера",
                  "сделайте музыку громче", "увеличьте громкость музыки", "сделай аудио громче",
                  "увеличь громкость воспроизведения", "сделай текущую музыку громче", "увеличить громкость музыки"],
    "volume_down": ["уменьши громкость плеера", "сделай эту песню тише", "сделай этот трек тише",
                    "уменьши громкость трека", "сделай проигрывание тише", "уменьшить громкость плеера",
                    "сделайте музыку тише", "уменьшите громкость музыки", "сделай аудио тише",
                    "уменьши громкость воспроизведения", "сделай текущую музыку тише", "уменьшить громкость музыки"],
    "unknown": ["не приостанавливай музыку", "не увеличивай громкость", "не уменьшай громкость",
                "не возобновляй воспроизведение", "почему ты пропустил песню?", "как поставить на паузу?",
                "когда музыка остановится?", "не пропускай эту песню", "не очищай очередь",
                "не включай голосовое управление", "пауза и выключи повтор", "громкость двести процентов",
                "выключи кондиционер", "выключи камеру", "уменьши яркость", "не выходи из канала"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot()
    bridge = OUT / "verification_parser.mjs"
    bridge.write_text(snap["bridge"], encoding="utf-8")
    commands = training.read_jsonl([OLD / "commands-synthetic.jsonl"])
    planner = training.read_jsonl([OLD / "planner-synthetic.jsonl"])
    candidates = []
    for label, messages in EXTRA.items():
        for i, text in enumerate(messages):
            for variant in variants(text)[:4]:
                candidates.append(automatic_row(variant, label, f"v2:{label}:semantic-{i}", current_player=player(i)))
    results = training.node_bridge([{**r, "decision": {"action": r["intent"], "confidence": 1.0}} for r in candidates], bridge)
    accepted, excluded = [], []
    existing = {(r["message"], json.dumps(r["player"], sort_keys=True)) for r in commands}
    for row, parsed in zip(candidates, results, strict=True):
        if parsed["final_action"] != row["intent"]:
            excluded.append({"message": row["message"], "declared_intent": row["intent"], "rules_action": parsed["final_action"]})
            continue
        signature = row["message"], json.dumps(row["player"], sort_keys=True)
        if signature in existing:
            continue
        existing.add(signature)
        commands.append(row)
        accepted.append(row)
        context = {"message": row["message"], "phase": "awaiting", "voice_enabled": True, "in_bot_channel": True,
                   "is_owner": True, "wake_name": "Муза", "player": row["player"]}
        cases = [(context, {"should_respond": "respond", "next_tool": "no_tool" if row["intent"] == "unknown" else "player_control", "intent": row["intent"]})]
        for change in ({"is_owner": False}, {"in_bot_channel": False}, {"voice_enabled": False, "phase": "disabled"}, {"phase": "busy"}, {"phase": "idle"}):
            cases.append(({**context, **change}, {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}))
        for j, (state, gold) in enumerate(cases):
            planner.append({"event_id": "planner-" + row["event_id"] + f"-{j}", "group_id": row["group_id"], "reviewed": False,
                            "origin": "synthetic", "label_status": "rule_verified", "label_source": "synthetic_template_checked", "state": state, "gold": gold})
    # More independent wake-name families, including explicit wrong-name/channel contexts.
    for i, name in enumerate(("Вега", "Орион", "Сириус", "Аврора", "Эхо", "Ирис", "Лира", "Альтаир", "Робот", "Ника", "Эра", "Терра")):
        for text in (name, name.upper(), name + "!"):
            context = {"message": text, "phase": "idle", "voice_enabled": True, "in_bot_channel": True,
                       "is_owner": False, "wake_name": name, "player": player(i)}
            cases = [(context, {"should_respond": "respond", "next_tool": "wake_ack", "intent": "unknown"})]
            for change in ({"wake_name": "Муза"}, {"in_bot_channel": False}, {"voice_enabled": False}, {"phase": "busy"}):
                cases.append(({**context, **change}, {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}))
            for j, (state, gold) in enumerate(cases):
                key = hashlib.sha256(json.dumps(state, ensure_ascii=False).encode()).hexdigest()[:24]
                planner.append({"event_id": "v2-wake-" + key, "group_id": f"v2:wake:name-{i}", "reviewed": False,
                    "origin": "synthetic", "label_status": "rule_verified", "label_source": "synthetic_template_checked", "state": state, "gold": gold})
    for name in ("bot-commands-labeled.jsonl", "bot-logs-unreviewed.jsonl", "bot-review-queue.jsonl", "bot-label-review.csv"):
        shutil.copyfile(OLD / name, OUT / name)
    for name, rows in (("commands-synthetic.jsonl", commands), ("planner-synthetic.jsonl", planner), ("excluded-v2-candidates.jsonl", excluded)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    loaded = training.prepare_commands(training.load_commands([OUT / "commands-synthetic.jsonl", OUT / "bot-commands-labeled.jsonl"],
        intent_labels=snap["intent_questions"]["intent"]["criteria"], allow_automatic_labels=True), bridge)
    planned = training.load_planner([OUT / "planner-synthetic.jsonl"], training.planner_questions(snap["intent_questions"]), bridge, True)
    for row, parsed in zip(planned, training.node_bridge([{"message": r["message"], "player": r["state"]["player"]} for r in planned], bridge), strict=True):
        row["canonical"] = parsed["canonical"]
    # Select only by corpus labels/group constraints, never by model quality.
    # Automatic bot logs and every connected phrase/context component must be train-only.
    for split_seed in range(SEED, SEED + 20):
        splits = training.grouped_split(loaded + planned, seed=split_seed)
        if all(r["origin"] != "bot_log" for name in training.SPLITS[1:] for r in splits[name]):
            break
    else:
        raise ValueError("Cannot reserve bot logs for train with class coverage")
    split_manifest = {name: [r["event_id"] for r in rows] for name, rows in splits.items()}
    (OUT / "split-manifest.json").write_text(json.dumps(split_manifest, indent=1), encoding="utf-8")
    manifest = {"dataset": "lia-balanced-v2", "seed": split_seed, "training_seed": SEED, "human_reviewed": False,
        "synthetic_commands": len(commands), "planner_scenarios": len(planner), "bot_log_rows": 44, "bot_labeled_rows": 31, "bot_pending_rows": 13,
        "new_command_rows": len(accepted), "new_semantic_families": len({r["group_id"] for r in accepted}), "excluded_v2_rows": len(excluded),
        "source_sha256": snap["source_sha256"], "parent_manifest_sha256": training.sha256_file(OLD / "manifest.json"),
        "automatic_bot_logs_train_only": True,
        "split_counts": {name: {"action_events": sum(r.get("task") != "planner" for r in rows), "planner_events": sum(r.get("task") == "planner" for r in rows)} for name, rows in splits.items()},
        "intent_counts": dict(Counter(r["intent"] for r in loaded)),
        "notes": ["V1 development test was already inspected; V2 is not a fresh real-user benchmark", "Raw corpus is preserved; training sampler balances task/class/family, not duplicate row counts",
                  "Automatic/synthetic labels are not human gold", "Ambiguous bot logs never enter training", "No invented real-log planner context"],
        "file_sha256": {p.name: training.sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("file_sha256", "source_sha256")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
