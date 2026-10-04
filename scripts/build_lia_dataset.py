"""Build a private starter corpus from declared templates and local bot logs.

Never copies model predictions/outcomes into labels. Generated/automatic labels
are explicitly unreviewed by a human, and require opt-in in the notebook.
All data artifacts stay in ignored .runtime; no logged speech enters Git.
"""
from collections import Counter
from copy import deepcopy
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/datasets/lia-starter-v1"
SEED = 42


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


training = module("dataset_training", ROOT / "scripts/laya_training.py")
builder = module("dataset_notebook_builder", ROOT / "scripts/build_laya_notebook.py")

SEEDS = {
    "skip": ["пропусти трек", "пропусти текущий трек", "пропусти эту песню", "пропусти песню",
             "пропустить трек", "пропусти играющий трек", "следующий", "следующая", "next", "skip",
             "следующий трек", "следующую песню", "пропустите трек", "пропустить текущую песню",
             "пропусти эту композицию", "пропусти текущую композицию", "пропусти воспроизводимый трек",
             "пропусти музыку", "следующий трек сейчас", "пропусти этот трек сейчас", "пропустить эту композицию"],
    "pause": ["поставь на паузу", "приостанови музыку", "приостанови воспроизведение", "приостановить музыку",
              "приостановить воспроизведение", "пауза", "pause", "включи паузу", "поставь режим паузы",
              "на паузу", "приостановите музыку", "приостанови плеер", "приостанови звук",
              "приостанови музыку сейчас", "приостановить плеер", "приостанови текущее воспроизведение",
              "приостанови проигрывание", "приостановить проигрывание", "приостанови воспроизведение сейчас",
              "приостановите плеер", "приостанови музыкальный плеер"],
    "resume": ["продолжи музыку", "продолжить музыку", "продолжай музыку", "возобнови музыку",
               "возобновить воспроизведение", "продолжи воспроизведение", "сними с паузы", "resume", "continue",
               "выключи паузу", "отключи режим паузы", "продолжите музыку", "возобнови проигрывание",
               "возобнови работу плеера", "продолжи музыку сейчас", "продолжай воспроизведение",
               "возобновите музыку", "возобновить музыку", "возобнови текущее воспроизведение",
               "продолжи проигрывание", "продолжить проигрывание"],
    "stop": ["стоп", "stop", "останови музыку", "остановить музыку", "останови воспроизведение",
             "выключи музыку", "отключи бота", "отключись", "выйди", "хватит играть",
             "останови плеер", "остановите музыку", "остановить воспроизведение", "выключи звук",
             "останови музыкальный плеер", "остановить плеер", "отключи музыкального бота",
             "выключи воспроизведение", "останови воспроизведение сейчас", "остановите плеер", "выйти"],
    "volume_up": ["громче", "погромче", "сделай громче", "увеличь громкость", "увеличить громкость",
                  "сделай музыку громче", "увеличь звук", "сделай звук громче", "сделать громче",
                  "увеличьте громкость", "сделай воспроизведение громче", "сделай громкость выше",
                  "сделай музыку погромче", "увеличить звук", "громче пожалуйста", "увеличь громкость музыки",
                  "сделай плеер громче", "увеличьте звук", "громче сейчас", "сделай чуть громче", "погромче музыку"],
    "volume_down": ["тише", "потише", "сделай тише", "уменьши громкость", "уменьшить громкость",
                    "сделай музыку тише", "уменьши звук", "сделай звук тише", "сделать тише",
                    "уменьшите громкость", "сделай воспроизведение тише", "сделай громкость ниже",
                    "сделай музыку потише", "уменьшить звук", "тише пожалуйста", "уменьши громкость музыки",
                    "сделай плеер тише", "уменьшите звук", "тише сейчас", "сделай чуть тише", "потише музыку"],
    "volume_set": ["громкость {n}", "установи громкость {n}", "поставь громкость {n}", "сделай громкость {n}",
                   "увеличь громкость до {n}", "уменьши громкость до {n}", "включи громкость {n}",
                   "установить громкость {n}", "громкость {n} процентов", "установи звук {n}",
                   "поставить громкость {n}", "сделать громкость {n}", "установите громкость {n}",
                   "увеличить громкость до {n}", "уменьшить громкость до {n}", "установи громкость на {n}",
                   "сделай громкость {n} процентов", "установить звук {n}", "громкость музыки {n}",
                   "громкость плеера {n}", "установи громкость плеера {n}"],
    "queue_clear": ["очисти очередь", "очистить очередь", "сбрось очередь", "сбросить очередь",
                    "удали все песни из очереди", "удали все треки из очереди", "очисти очередь песен",
                    "очисти очередь треков", "сбрось очередь песен", "сбросить очередь треков",
                    "очистить очередь песен", "очистить очередь треков", "сбрось очередь треков",
                    "сбросить очередь песен", "удали песни из очереди", "удали треки из очереди"],
    "unknown": ["не пропускай трек", "не ставь на паузу", "не выключай музыку", "не делай громче",
                "не делай тише", "если я попрошу останови музыку", "почему музыка на паузе?",
                "кто исполняет эту песню?", "расскажи о музыке", "я поставлю это на паузу сам",
                "мне нравится эта музыка", "привет", "спасибо", "выключи свет", "отключи микрофон",
                "громкость 0", "громкость 151", "громкость минус 10", "громкость -10", "громкость 5.5",
                "громкость 20 или 30", "громче на 50", "тише на двадцать", "пауза и следующий трек",
                "останови музыку потом включи Numb", "включи эту", "включи музыку", "кто такой Баста?",
                "не включай эту песню", "не включи автоплей", "если включи повтор трека",
                "отключи повтор и очисти очередь", "включи неизвестный режим", "поставь режим телепортации",
                "громкость девяносто девяносто", "громкость +20", "громкость 50 60", "увеличь громкость на десять",
                "очисти очередь и останови музыку", "поставь на паузу?", "выключи телевизор",
                "потом пропусти трек", "не останавливай воспроизведение", "пропусти и сделай громче",
                "найди погоду", "я обсуждаю песню", "мне нравится Numb", "как включить музыку?"],
}

ON = ["включи", "включить", "включай", "ключи", "запусти", "запустить", "поставь", "поставить", "активируй"]
OFF = ["выключи", "выключить", "отключи", "отключить", "отключай", "деактивируй", "убери", "останови"]
for label, targets in {
    "autoplay": ["автоплей", "бесконечный режим", "режим автоподбора", "рекомендации YouTube", "автоподбор песен",
                 "автоматический режим", "бесконечное воспроизведение", "музыку бесконечно"],
    "loop": ["повтор трека", "повтор песни", "режим повтора", "повторение текущего трека", "повтор этой песни", "repeat", "loop"],
    "voice": ["голосовое управление", "голосовой режим", "режим голосового управления", "прослушивание"],
}.items():
    for direction, verbs in (("on", ON), ("off", OFF)):
        # Different verb/target families, not a giant single class-sized group.
        SEEDS[label + "_" + direction] = [f"{verb} {target}" for verb in verbs for target in targets]

SEARCH_TEMPLATES = ["включи {q}", "поставь песню {q}", "сыграй {q}", "воспроизведи {q}",
                    "запусти {q}", "хочу послушать {q}", "послушаем {q}", "проиграй {q}",
                    "можешь включить {q}", "ключи {q}", "включай {q}", "поставить трек {q}"]
QUERIES = ["Кино Группа крови", "Земфира", "Linkin Park Numb", "Queen Don't Stop Me Now", "Сплин Выхода нет",
           "Rammstein Sonne", "Монеточка", "Баста", "Би-2", "Ночные Снайперы", "jazz for studying",
           "электронную музыку без слов", "музыку для отдыха", "песню из Лунтика", "Numb live", "Sonne remix",
           "кавер на Группу крови", "акустическую версию Numb", "инструментальную версию Sonne", "Бесконечный режим"]


def player(index=0):
    return {"connected": True, "playing": index % 3 != 0, "paused": False,
            "autoplay": bool(index % 2), "queue_length": index % 5}


def variants(message):
    return [message, message[:1].upper() + message[1:] + ".", message.upper(),
            "пожалуйста, " + message, "ну, " + message, "слушай, " + message]


def words(number):
    units = "ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split()
    tens = {20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят", 60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто"}
    if number >= 100:
        return "сто" + (" " + words(number - 100) if number > 100 else "")
    if number < 20:
        return units[number]
    return tens[number // 10 * 10] + (" " + units[number % 10] if number % 10 else "")


def automatic_row(message, intent, group, origin="synthetic", source_event_id=None, current_player=None):
    key = hashlib.sha256((origin + "|" + group + "|" + message + "|" + json.dumps(current_player, sort_keys=True)).encode()).hexdigest()[:24]
    row = {"event_id": origin + "-" + key, "message": message, "intent": intent,
           "player": current_player or player(), "group_id": group, "origin": origin,
           "reviewed": False, "label_status": "rule_verified",
           "label_source": "synthetic_template_checked" if origin == "synthetic" else "bot_transcript_rules"}
    if source_event_id:
        row["source_event_id"] = source_event_id
    return row


def save_jsonl(name, rows):
    path = OUT / name
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = builder.snapshot()
    bridge = OUT / "verification_parser.mjs"
    bridge.write_text(snap["bridge"], encoding="utf-8")
    command_candidates, planner_candidates = [], []
    for label, seeds in SEEDS.items():
        for i, seed in enumerate(seeds):
            target = planner_candidates if i % 3 == 2 else command_candidates
            group = f"synthetic:{label}:template-{i:03d}"
            texts = [seed] if "{n}" not in seed else [seed.format(n=n) for n in [1, 10, 30, 50, 70, 100, 125, 150, words(1), words(50), words(100), words(150)]]
            for text in texts:
                for variant in variants(text):
                    target.append(automatic_row(variant, label, group, current_player=player(i)))
    for i, template in enumerate(SEARCH_TEMPLATES):
        target = planner_candidates if i % 3 == 2 else command_candidates
        for query in QUERIES:
            for text in variants(template.format(q=query))[:4]:
                target.append(automatic_row(text, "play", f"synthetic:play:search-template-{i}", current_player=player(i)))
    for i in range(12):
        target = planner_candidates if i % 3 == 2 else command_candidates
        url = f"https://www.youtube.com/watch?v=Example{i:04d}"
        for template in ("{q}", "включи {q}", "поставь {q}"):
            target.append(automatic_row(template.format(q=url), "play", f"synthetic:play:video-{i}", current_player=player(i)))
    # Context-sensitive command is deliberately labelled from paused state.
    for text in ("включи музыку", "включи", "включи воспроизведение", "продолжай"):
        state = player(); state["paused"] = True; state["playing"] = False
        command_candidates.append(automatic_row(text, "resume", "synthetic:resume:paused-context", current_player=state))

    def verified(candidates, deduplicate=True):
        probes = [{**row, "decision": {"action": row["intent"] if row["intent"] in snap["intent_questions"]["intent"]["criteria"] and row["intent"] != "play" else "unknown", "confidence": 1.0}} for row in candidates]
        results = training.node_bridge(probes, bridge)
        accepted, excluded = [], []
        seen = set()
        for row, result in zip(candidates, results, strict=True):
            if result["final_action"] != row["intent"]:
                excluded.append({"message": row["message"], "declared_intent": row["intent"], "rules_action": result["final_action"], "group_id": row["group_id"]})
                continue  # Do not relabel a rejected positive as a negative.
            signature = (row["message"], json.dumps(row["player"], sort_keys=True))
            if not deduplicate or signature not in seen:
                seen.add(signature)
                accepted.append(row)
        return accepted, excluded

    commands, exclusions = verified(command_candidates)
    planner_positive, more_exclusions = verified(planner_candidates)
    exclusions += more_exclusions

    raw_logs, labelled_logs, pending, log_sources = [], [], [], []
    for source in sorted((ROOT / ".runtime/nli").glob("commands-*.jsonl")):
        blob = source.read_bytes()
        rows = [training.json_object(line) for line in blob.decode("utf-8-sig").splitlines() if line.strip()]
        log_sources.append({"file": source.name, "sha256": hashlib.sha256(blob).hexdigest(), "rows": len(rows)})
        probes = training.node_bridge([{"message": r["state"]["message"], "player": r["state"].get("player"),
                                       "decision": {"action": "volume_set", "confidence": 1.0}} for r in rows], bridge)
        for row, result in zip(rows, probes, strict=True):
            message = row["state"]["message"]
            state = row["state"].get("player")
            safe = {"event_id": row["event_id"], "recorded_at": row["recorded_at"], "source_file": source.name,
                    "origin": "bot_log", "reviewed": False, "label_status": "unreviewed", "gold": None,
                    "state": {"phase": "request", "message": message, "player": state}, "group_id": "bot-session:" + source.stem}
            raw_logs.append(safe)
            norm, intent = training.normalized(result["canonical"]), None
            if result["direct_action"] not in (None, "unknown"):
                intent = result["direct_action"]
            elif result["music_kind"] and not result["model_required"] and not result["unsupported"]:
                intent = "play"
            elif norm == "следующий трек":
                intent = "skip"
            elif norm in {"стоп", "останови музыку", "остановить музыку", "отключись", "выйди", "хватит играть"}:
                intent = "stop"
            elif re.fullmatch(r"громкость [\d ]+", norm) and result["final_action"] == "volume_set":
                intent = "volume_set"
            elif message.strip().endswith("?"):
                intent = "unknown"
            elif norm in {"баста"}:
                intent = "play"  # Known bare artist; no model prediction consulted.
            if intent:
                labelled = automatic_row(message, intent, safe["group_id"], "bot_log", row["event_id"], state)
                labelled["event_id"] = row["event_id"]
                labelled["recorded_at"] = row["recorded_at"]
                labelled_logs.append(labelled)
            else:
                pending.append({**safe, "intent": None, "review_reason": "Недостаточно контекста / возможная ошибка STT; нужна проверка исходной просьбы"})
    labelled_logs, log_exclusions = verified(labelled_logs, deduplicate=False)
    if log_exclusions:
        raise ValueError("Automatic log labels contradict current rules; inspect manually")

    planner = []
    names = ["Муза", "Лия", "Лайя", "Бот", "Луна", "Нота", "Астра", "Ритм"]
    planner_parsed = training.node_bridge(planner_positive, bridge)
    for i, (row, probe) in enumerate(zip(planner_positive, planner_parsed, strict=True)):
        message, label = row["message"], row["intent"]
        tool = ("direct_youtube_video" if probe["music_kind"] == "video" else "youtube_music_search") if label == "play" else "no_tool" if label == "unknown" else "player_control"
        state = {"message": message, "phase": "awaiting", "voice_enabled": True, "in_bot_channel": True,
                 "is_owner": True, "wake_name": names[i % len(names)], "player": row["player"]}
        cases = [(state, {"should_respond": "respond", "next_tool": tool, "intent": label})]
        for change in ({"is_owner": False}, {"in_bot_channel": False}, {"phase": "idle"},
                       {"phase": "busy"}, {"phase": "disabled", "voice_enabled": False}):
            cases.append(({**state, **change}, {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}))
        for j, (context, gold) in enumerate(cases):
            planner.append({"event_id": "planner-" + row["event_id"] + f"-{j}", "group_id": row["group_id"],
                            "reviewed": False, "origin": "synthetic", "label_status": "rule_verified",
                            "label_source": "synthetic_template_checked", "state": context, "gold": gold})
    similar = ["музыка", "липа", "лавка", "вот", "лунка", "ноты", "астрагал", "ритмы"]
    for i, name in enumerate(names):
        for text in (name, name.upper(), name + "."):
            state = {"message": text, "phase": "idle", "voice_enabled": True, "in_bot_channel": True,
                     "is_owner": False, "wake_name": name, "player": player(i)}
            planner.append({"event_id": f"wake-{i}-" + hashlib.sha256(text.encode()).hexdigest()[:8],
                            "group_id": f"synthetic:wake:name-{i}", "reviewed": False, "origin": "synthetic",
                            "label_status": "rule_verified", "label_source": "synthetic_template_checked", "state": state,
                            "gold": {"should_respond": "respond", "next_tool": "wake_ack", "intent": "unknown"}})
        state = {"message": similar[i], "phase": "idle", "voice_enabled": True, "in_bot_channel": True,
                 "is_owner": False, "wake_name": name, "player": player(i)}
        planner.append({"event_id": f"false-wake-{i}", "group_id": f"synthetic:wake:name-{i}", "reviewed": False,
                        "origin": "synthetic", "label_status": "rule_verified", "label_source": "synthetic_template_checked",
                        "state": state, "gold": {"should_respond": "ignore", "next_tool": "no_tool", "intent": "unknown"}})

    rng = random.Random(SEED)
    for rows in (commands, labelled_logs, planner):
        rng.shuffle(rows)
    files = {
        "commands-synthetic.jsonl": commands,
        "bot-commands-labeled.jsonl": labelled_logs,
        "bot-logs-unreviewed.jsonl": raw_logs,
        "bot-review-queue.jsonl": pending,
        "planner-synthetic.jsonl": planner,
        "excluded-synthetic-candidates.jsonl": exclusions,
    }
    for name, rows in files.items():
        save_jsonl(name, rows)
    manifest = {"dataset": "lia-starter-v1", "seed": SEED, "human_reviewed": False,
                "source_sha256": snap["source_sha256"], "synthetic_commands": len(commands),
                "bot_log_rows": len(raw_logs), "bot_labeled_rows": len(labelled_logs), "bot_pending_rows": len(pending),
                "planner_scenarios": len(planner), "excluded_synthetic_candidates": len(exclusions),
                "bot_log_sources": log_sources,
                "intent_counts": dict(Counter(r["intent"] for r in commands + labelled_logs)),
                "planner_answer_counts": {k: dict(Counter(r["gold"][k] for r in planner)) for k in ("should_respond", "next_tool", "intent")},
                "file_sha256": {name: training.sha256_file(OUT / name) for name in files},
                "limitations": ["Template-generated data, not a representative speech benchmark",
                                "Automatic labels require explicit opt-in and later human review",
                                "Local log records do not cover all actions",
                                "No audio or idle/owner flags recorded in bot logs; no invented real planner context",
                                "Synthetic YouTube URLs express routing shape, not verified playable videos"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = training.load_commands([OUT / "commands-synthetic.jsonl", OUT / "bot-commands-labeled.jsonl"],
                                    intent_labels=snap["intent_questions"]["intent"]["criteria"], allow_automatic_labels=True)
    loaded = training.prepare_commands(loaded, bridge)
    planner_loaded = training.load_planner([OUT / "planner-synthetic.jsonl"], training.planner_questions(snap["intent_questions"]),
                                          bridge, allow_automatic_labels=True)
    parsed = training.node_bridge([{"message": r["message"], "player": r["state"]["player"]} for r in planner_loaded], bridge)
    for row, result in zip(planner_loaded, parsed, strict=True):
        row["canonical"] = result["canonical"]
    all_splits = training.grouped_split(loaded + planner_loaded, seed=SEED)
    split_stats = {name: {"runtime_commands": sum(r.get("task") != "planner" for r in rows),
                         "planner_scenarios": sum(r.get("task") == "planner" for r in rows),
                         "bot_log_commands": sum(r["origin"] == "bot_log" for r in rows)} for name, rows in all_splits.items()}
    for rows in all_splits.values():
        available = {r["intent"] if r["intent"] in {"skip", "pause", "resume", "stop", "volume_set", "volume_up", "volume_down"} else "unknown"
                     for r in rows if r.get("task") != "planner"}
        if len(available) != 8:
            raise ValueError("Dataset does not cover all runtime classes in every split")
    manifest["validated_grouped_split"] = split_stats
    manifest["runtime_normalized_texts"] = len({training.normalized(r["canonical"]) for r in loaded})
    manifest["runtime_declared_families"] = len({r["group_id"] for r in loaded})
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "split-manifest.json").write_text(json.dumps({name: [r["event_id"] for r in rows] for name, rows in all_splits.items()}, indent=2), encoding="utf-8")
    labels_by_event = {r["event_id"]: r["intent"] for r in labelled_logs}
    with (OUT / "bot-label-review.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["event_id", "message", "proposed_intent", "approved_intent", "review_reason"])
        writer.writeheader()
        for row in raw_logs:
            writer.writerow({"event_id": row["event_id"], "message": row["state"]["message"],
                             "proposed_intent": labels_by_event.get(row["event_id"], ""), "approved_intent": "",
                             "review_reason": "Автоматическая разметка; подтвердите смысл" if row["event_id"] in labels_by_event else "Недостаточно контекста / ошибка STT"})
    notebook = json.loads((ROOT / "notebooks/laya_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
    ready = deepcopy(notebook)
    for cell in ready["cells"]:
        if "parameters" in cell["metadata"].get("tags", []):
            cell["source"] = cell["source"].replace("ALLOW_AUTOMATIC_LABELS = False", "ALLOW_AUTOMATIC_LABELS = True")
            cell["source"] = cell["source"].replace("AUTO_LOAD_PLANNER = False", "AUTO_LOAD_PLANNER = True")
    upload_code = '''import zipfile
if not DATA_PATHS:
    roots = [p for p in (Path("/content"), Path("/kaggle/input")) if p.exists()]
    has_data = any(any(root.rglob("commands-synthetic.jsonl")) for root in roots)
    if not has_data:
        archives = [p for root in roots for p in root.rglob("lia-starter-v1.zip")]
        if not archives and Path("/content").exists():
            from google.colab import files
            files.upload()  # Выберите полученный архив lia-starter-v1.zip.
            archives = list(Path("/content").glob("lia-starter-v1*.zip"))
        if len(archives) != 1:
            raise FileNotFoundError("Add the dataset files or a single lia-starter-v1.zip; alternatively fill DATA_PATHS")
        destination = WORKDIR.parent / "lia-dataset"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                (destination / member.filename).resolve().relative_to(destination.resolve())
            archive.extractall(destination)
        DATA_PATHS = [str(p) for name in ("commands-synthetic.jsonl", "bot-commands-labeled.jsonl") for p in destination.rglob(name)]
        if AUTO_LOAD_PLANNER:
            PLANNER_PATHS = [str(p) for p in destination.rglob("planner-synthetic.jsonl")]
print("Dataset files are ready; automatic labels are explicitly enabled")
'''
    ast.parse(upload_code)
    parameter_index = next(i for i, c in enumerate(ready["cells"]) if "parameters" in c["metadata"].get("tags", []))
    ready["cells"][parameter_index + 1:parameter_index + 1] = [
        {"cell_type": "markdown", "id": "dataset-upload-description", "metadata": {}, "source": "### Загрузить полученный ZIP\n\nВ Colab выберите lia-starter-v1.zip. В Kaggle добавьте Dataset с файлами или этим ZIP. Разметка синтетики и логов автоматическая, не проверенная человеком.\n"},
        {"cell_type": "code", "id": "dataset-upload", "metadata": {"tags": ["dataset-upload"]}, "source": upload_code, "execution_count": None, "outputs": []},
    ]
    (OUT / "lia_dataset_training_colab_kaggle.ipynb").write_text(json.dumps(ready, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    description = f'''# LIA: стартовый датасет и логи бота

В архиве {len(commands):,} синтетических команд, {len(planner):,} контекстных сценариев planner и все {len(raw_logs)} записи из журналов бота.
Для {len(labelled_logs)} реальных записей подготовлена автоматическая разметка по исходной расшифровке и правилам; {len(pending)} неоднозначных записей оставлены без обучающей метки.
Предсказания Laya, outcome и исправленный canonical_message из журналов не использовались как ответы.

## Запуск

1. Откройте `lia_dataset_training_colab_kaggle.ipynb` в Colab или Kaggle и включите GPU.
2. В Colab запустите ячейку «Загрузить полученный ZIP» и выберите этот архив. В Kaggle добавьте Dataset с файлами либо ZIP.
3. Запускайте ячейки по порядку. В этой копии ноутбука `ALLOW_AUTOMATIC_LABELS=True` и `AUTO_LOAD_PLANNER=True` уже выставлены.
4. Чтобы обучать только действующий классификатор action, задайте `AUTO_LOAD_PLANNER=False` до ячейки загрузки архива и оставьте `PLANNER_PATHS=[]`.

## Файлы

- `commands-synthetic.jsonl`: шаблонные команды, цифры и числа словами, режимы, запросы музыки, отрицания, вопросы, составные/недопустимые просьбы.
- `bot-commands-labeled.jsonl`: {len(labelled_logs)} реальные записи с автоматическими метками, сохранёнными event_id и player.
- `bot-logs-unreviewed.jsonl`: все {len(raw_logs)} реальные записи, исходная речь и состояние плеера; без модельных подсказок. Этот файл не загружается в обучение автоматически.
- `bot-review-queue.jsonl`: {len(pending)} неоднозначных записей. Их нельзя исправить по аудио: журнал не содержит аудио.
- `bot-label-review.csv`: все реальные записи для проверки; proposed_intent — подсказка, approved_intent оставлен пустым.
- `planner-synthetic.jsonl`: реагировать/игнорировать, wake_ack, поиск, видео, управление, no_tool; владелец сессии, чужой пользователь, другой канал, idle/busy/disabled.
- `manifest.json`: объёмы, происхождение, классы, SHA-256 исходников/данных, проверенное разделение.
- `split-manifest.json`: event_id в train/validation/calibration/test, воспроизводимые при seed=42.
- `excluded-synthetic-candidates.jsonl`: объявленные команды, которые текущий код не принимает. Они исключены, а не переименованы в unknown.
- `verification_parser.mjs`: снимок чистых TypeScript-функций для проверки контракта.

## Что проверено и что означают метрики

JSONL и метки проверены загрузчиками ноутбука и правилами бота. Runtime и planner разделяются совместно:
родственные шаблоны, исходные/приведённые совпадения и разные контексты одной фразы находятся в одной части.
Все восемь runtime-классов присутствуют в каждой части. Фразы из одного файла журнала сохраняются вместе.
При текущем seed все {len(labelled_logs)} размеченные реальные записи попадают в train. Validation/calibration/test здесь синтетические.

Корпус содержит {manifest['runtime_declared_families']} объявленных runtime-семейств и {manifest['runtime_normalized_texts']} разных нормализованных runtime-текстов.
Часть строк — регистр, пунктуация и вежливые вводные той же фразы, а не независимые реальные наблюдения.
Метрики этого набора не являются точностью на реальных пользователях. Для такой оценки соберите отдельные новые записи и проверьте метки вручную.

Метки имеют `reviewed=false`, `origin` и `label_source`: это синтетика / автоматическая разметка, не человеческий gold.
Для ручной проверки заполните approved_intent в CSV, перенесите подтверждённые event_id/intent в `reviewed.jsonl`
и экспортируйте через `npm run export:nli -- reviewed.jsonl reviewed-nli.jsonl`. При обучении на этом экспорте
исключите `bot-commands-labeled.jsonl` из DATA_PATHS, чтобы не дублировать те же event_id.

Текущий бот использует только action; будущие should_respond/next_tool/intent требуют отдельного подключения.
В логах нет owner/voice_enabled/idle, поэтому реальный контекст planner не выдумывается из этих записей.
Синтетические URL показывают формат запроса конкретного видео; существование видео и выдача YouTube не проверялись.
Архив и распознанные реплики хранятся в игнорируемой `.runtime/datasets/`.
'''
    (OUT / "README.md").write_text(description, encoding="utf-8")
    archive_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(OUT.name) / path.relative_to(OUT)))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print("Archive:", archive_path)
    print(json.dumps({k: manifest[k] for k in ("synthetic_commands", "bot_log_rows", "bot_labeled_rows", "bot_pending_rows", "planner_scenarios", "intent_counts")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    build()
