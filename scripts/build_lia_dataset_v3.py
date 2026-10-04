"""V3: semantic requests and session-policy fixtures are separate datasets."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import laya_training as base
from build_laya_notebook import snapshot
from build_lia_dataset import player, variants, words
from laya_training_v3 import joint_split

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".runtime/datasets/lia-balanced-v2"
OUT = ROOT / ".runtime/datasets/lia-semantic-v3"
OBJECTS = ["музыку", "трек", "песню", "воспроизведение", "звук", "плеер", "композицию", "проигрывание"]
PATTERNS = {
    "pause": ["поставь {x} на паузу", "поставить {x} на паузу", "приостанови {x}", "приостановить {x}",
              "приостановите {x}", "поставь на паузу {x}", "поставьте {x} на паузу", "приостанови текущую {x}",
              "приостанови {x} сейчас", "поставь {x} на паузу сейчас", "нажми паузу для {x}", "сейчас приостанови {x}"],
    "resume": ["продолжи {x}", "возобнови {x}", "продолжить {x}", "возобновить {x}",
               "продолжайте {x}", "возобновите {x}", "продолжи {x} сейчас", "возобнови {x} сейчас",
               "сними {x} с паузы", "сними с паузы {x}", "снова продолжи {x}", "продолжи воспроизведение {x}"],
    "stop": ["останови {x}", "остановить {x}", "остановите {x}", "выключи {x}", "отключи {x}",
             "останови {x} сейчас", "выключи {x} сейчас", "останови воспроизведение {x}",
             "отключи воспроизведение {x}", "останови {x} полностью", "прекрати играть {x}", "полностью останови {x}"],
    "skip": ["пропусти {x}", "пропустить {x}", "пропустите {x}", "пропусти {x} сейчас",
             "пропусти текущий {x}", "пропусти этот {x}", "переключи {x}", "переключи на следующий {x}",
             "следующий {x}", "включи следующий {x}", "пропусти играющий {x}", "перейди к следующему {x}"],
    "volume_up": ["сделай {x} громче", "сделать {x} громче", "увеличь громкость {x}", "увеличить громкость {x}",
                  "сделай {x} погромче", "сделайте {x} громче", "увеличьте громкость {x}", "увеличь звук {x}",
                  "сделай громкость {x} выше", "добавь громкости {x}", "сделай {x} чуть громче", "подними громкость {x}"],
    "volume_down": ["сделай {x} тише", "сделать {x} тише", "уменьши громкость {x}", "уменьшить громкость {x}",
                    "сделай {x} потише", "сделайте {x} тише", "уменьшите громкость {x}", "уменьши звук {x}",
                    "сделай громкость {x} ниже", "убавь громкости {x}", "сделай {x} чуть тише", "снизь громкость {x}"],
}
# Each object is paired with the appropriate grammatical case; do not synthesize
# malformed "громкость музыку", "текущий песню", etc. merely to inflate row counts.
GENITIVES = ["музыки", "трека", "песни", "воспроизведения", "звука", "плеера", "композиции", "проигрывания"]


def key(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def make(message, intent, group, state=None):
    return {"event_id": "v3-synthetic-" + key([message, intent, group, state]), "message": message, "intent": intent,
            "player": state or player(), "group_id": group, "origin": "synthetic", "reviewed": False,
            "label_status": "template_declared", "label_source": "declared_semantics"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot()
    bridge = OUT / "verification_parser.mjs"
    bridge.write_text(snap["bridge"], encoding="utf-8")
    # Only eligible awaiting examples contribute semantic gold; blocked contexts are fixtures.
    semantic = base.read_jsonl([OLD / "commands-synthetic.jsonl", OLD / "bot-commands-labeled.jsonl"])
    policy = base.read_jsonl([OLD / "planner-synthetic.jsonl"])
    for row in policy:
        if row["state"]["phase"] == "awaiting" and row["gold"]["should_respond"] == "respond":
            semantic.append({"event_id": "semantic-" + row["event_id"], "message": row["state"]["message"],
                "player": row["state"]["player"], "intent": row["gold"]["intent"], "group_id": row["group_id"],
                "origin": row["origin"], "reviewed": False, "label_status": "rule_verified", "label_source": "synthetic_template_checked"})
    fresh = []
    for label, patterns in PATTERNS.items():
        for i, pattern in enumerate(patterns):
            # Avoid awkward case/agreement patterns; retain explicit manually declared phrases below instead.
            if any(term in pattern for term in ("текущую", "текущий", "этот", "играющий", "следующему", "следующий", "следующий")):
                continue
            objects = GENITIVES if any(term in pattern for term in ("громкость {x}", "громкости {x}", "звук {x}", "воспроизведение {x}", "громкость {x}")) else OBJECTS
            if pattern.startswith(("сделай громкость",)):
                objects = GENITIVES
            if label == "skip":
                objects = ["трек", "песню", "композицию"]
            for noun in objects:
                sentence = pattern.format(x=noun)
                group = f"v3:{label}:pattern-{i}"
                for text in variants(sentence)[:4]: fresh.append(make(text, label, group))
                # A prohibition is a clear unknown; policy-blocked contexts are not.
                fresh.append(make("не " + sentence, "unknown", group))
    explicit = {
        "pause": ["нажми паузу", "поставь трек на паузу", "поставь музыку на паузу", "приостанови текущую песню"],
        "resume": ["возобнови проигрывание", "продолжи текущую песню", "сними с паузы", "продолжи музыку после паузы"],
        "skip": ["следующий трек", "следующая песня", "включи следующий трек", "перейди к следующей песне"],
        "stop": ["стоп", "останови музыку и выйди", "выйди из голосового канала", "отключись от канала"],
        "unknown": ["как поставить музыку на паузу?", "почему музыка остановилась?", "почему стало тише?",
                    "как увеличить громкость?", "не останавливай музыку", "пауза и следующий трек",
                    "сделай громче и пропусти песню", "выключи свет", "выключи камеру", "уменьши яркость",
                    "кто исполняет эту песню?", "какая сейчас громкость?", "поставь таймер", "найди погоду",
                    "громкость 0", "громкость 151", "громкость -30", "громкость 7.5", "громкость 30 или 70",
                    "тише на пятьдесят", "громче на сорок", "я обсуждаю эту песню", "завтра у меня экзамен"],
    }
    for label, texts in explicit.items():
        for i, sentence in enumerate(texts):
            for text in variants(sentence)[:4]: fresh.append(make(text, label, f"v3:{label}:explicit-{i}"))
    for i, template in enumerate(("громкость {n} процентов", "установи громкость на {n}", "сделай громкость {n}", "поставь громкость {n} процентов")):
        for n in (5, 15, 35, 65, 95, 115, 135, 145):
            for value in (str(n), words(n)):
                for text in variants(template.format(n=value))[:4]: fresh.append(make(text, "volume_set", f"v3:volume_set:pattern-{i}"))
    semantic += fresh
    # Remove only exact duplicates and ambiguous imperative questions from semantic learning.
    dedup, seen, quarantined = [], set(), []
    for row in semantic:
        signature = row["message"], json.dumps(row["player"], sort_keys=True), row["intent"]
        if signature in seen and row["origin"] != "bot_log": continue
        seen.add(signature)
        if row["intent"] == "unknown" and row["message"].strip().lower() == "поставь на паузу?":
            quarantined.append(row); continue
        dedup.append(row)
    semantic = dedup
    parsed = base.node_bridge([{**r, "decision": {"action": r["intent"] if r["intent"] != "play" else "unknown", "confidence": 1.0}} for r in semantic], bridge)
    action, limitations = [], []
    for row, probe in zip(semantic, parsed, strict=True):
        if probe["final_action"] == row["intent"]:
            item = deepcopy(row)
            item["label_status"] = "rule_verified"
            if item["origin"] == "synthetic": item["label_source"] = "synthetic_template_checked"
            action.append(item)
        else:
            limitations.append({"message": row["message"], "semantic_intent": row["intent"], "current_bot_action": probe["final_action"], "group_id": row["group_id"]})
    # Distinct task IDs, same families/texts => same joint split.
    prepared_action = [{**r, **p} for r, p in zip(action, base.node_bridge(action, bridge), strict=True)]
    prepared_semantic = [{**r, "event_id": "semantic:" + r["event_id"], "task": "semantic", "canonical": p["canonical"]}
                         for r, p in zip(semantic, base.node_bridge(semantic, bridge), strict=True)]
    corpus = prepared_action + prepared_semantic
    split_seed = 20261010
    splits = joint_split(corpus, split_seed)
    for row in semantic: row["event_id"] = "semantic:" + row["event_id"]
    files = {"action-commands.jsonl": action, "semantic-commands.jsonl": semantic, "policy-fixtures.jsonl": policy,
             "parser-limitations.jsonl": limitations, "synthetic-review-queue.jsonl": quarantined}
    for name, rows in files.items():
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for name in ("bot-logs-unreviewed.jsonl", "bot-review-queue.jsonl", "bot-label-review.csv"):
        shutil.copyfile(OLD / name, OUT / name)
    ids = {name: [r["event_id"] for r in rows] for name, rows in splits.items()}
    (OUT / "split-manifest.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
    manifest = {"dataset": "lia-semantic-v3", "split_seed": split_seed, "human_reviewed": False,
        "action_events": len(action), "semantic_events": len(semantic), "policy_fixtures": len(policy),
        "fresh_declared_rows": len(fresh), "fresh_families": len({r["group_id"] for r in fresh}),
        "automatic_bot_log_events": 31, "all_bot_log_events": 44, "pending_bot_log_events": 13,
        "new_contract": "semantic intention on text; eligibility/tool mapping by deterministic policy",
        "blocked_contexts_are_training_examples": False, "automatic_logs_train_only": True,
        "semantic_intent_counts": dict(Counter(r["intent"] for r in semantic)),
        "split_counts": {name: {"action": sum(r.get("task") != "semantic" for r in rows), "semantic": sum(r.get("task") == "semantic" for r in rows)} for name, rows in splits.items()},
        "source_sha256": snap["source_sha256"], "parent_dataset": base.sha256_file(OLD / "manifest.json"),
        "limitations": ["Synthetic and automatic labels; not independently human-reviewed speech accuracy", "V1/V2 test has been observed; development corpus contains those phrases",
                        "Current bot parser has additional documented limits; semantic model integration is separate", "No invented owner/channel context for real logs"],
        "file_sha256": {p.name: base.sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("file_sha256", "source_sha256")}, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
