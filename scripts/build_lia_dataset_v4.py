"""Package literal Codex-authored action examples with bounded historical replay.

Synthetic source text is public repository data. Private logs/artifacts stay in
.runtime. A parser check can exclude a declaration, never create/change its label.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import laya_training as base
from laya_training_v3 import runtime_text, negative_kind
from laya_training_v4 import category_of
from build_laya_notebook import snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/lia_action_v4_authored.txt"
OUT = ROOT / ".runtime/datasets/lia-action-v4"
OLD = ROOT / ".runtime/datasets/lia-semantic-v3"
LABELS = {"pause", "resume", "skip", "stop", "volume_set", "volume_up", "volume_down", "unknown"}


def authored_rows():
    rows, section = [], None
    for line_number, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("["):
            fold, family = line[1:-1].split(":", 1)
            assert fold in base.SPLITS
            section = fold, "v4-authored:" + family
            continue
        label, sentences = line.split("|", 1)
        assert section and label in LABELS
        for index, sentence in enumerate(sentences.split(" || ")):
            message = sentence.strip()
            assert message and "{" not in message and "}" not in message
            ident = f"v4-authored:{line_number}:{index}"
            rows.append({"event_id": ident, "message": message, "intent": label,
                "group_id": section[1], "declared_split": section[0], "origin": "synthetic",
                "reviewed": False, "label_status": "assistant_authored", "label_source": "codex_explicit_declaration",
                "source_role": "authored", "source_line": line_number,
                "player": {"connected": True, "playing": True, "paused": False, "autoplay": False, "queue_length": 2}})
    return rows


def write_jsonl(name, rows):
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot()
    bridge = OUT / "verification_parser.mjs"
    bridge.write_text(snap["bridge"], encoding="utf-8")
    declared = authored_rows()
    prepared = base.prepare_commands(declared, bridge)
    oracle = base.node_bridge([{**r, "decision": {"action": r["intent"], "confidence": 1.}} for r in declared], bridge)
    authored, rejected, seen = [], [], {}
    for row, probe in zip(prepared, oracle, strict=True):
        text = runtime_text(row["canonical"])
        if row["intent"] != "unknown" and probe["final_action"] != row["intent"]:
            rejected.append({**row, "reason": "current_parser_rejects_declared_control", "parser_action": probe["final_action"]})
            continue
        if text in seen:
            previous = seen[text]
            if previous["intent"] != row["intent"]: raise ValueError(f"Contradictory input: {text}")
            rejected.append({**row, "reason": "duplicate_runtime_input", "first_event_id": previous["event_id"]})
            continue
        seen[text] = row
        row["category"] = category_of(row)
        row["pipeline_gold"] = probe["final_action"]
        authored.append(row)
    # Never move holdout declarations into train based on a model result.
    protected_texts = {runtime_text(r["canonical"]) for r in authored if r["declared_split"] != "train"}
    historical = base.read_jsonl([OLD / "action-commands.jsonl"])
    historical = base.prepare_commands(historical, bridge)
    contaminated_families = {r["group_id"] for r in historical if runtime_text(r["canonical"]) in protected_texts}
    assert not any(r["origin"] == "bot_log" and r["group_id"] in contaminated_families for r in historical), "Real log intersects fresh holdout; review before packaging"
    replay, replay_texts, replay_exclusions = [], set(), Counter()
    for row in historical:
        row["category"] = negative_kind(row)
        if row["intent"] not in LABELS:
            row["historical_semantic_intent"] = row["intent"]
            row["intent"] = "unknown"
        row["category"] = category_of(row)
        text = runtime_text(row["canonical"])
        if row["origin"] != "bot_log":
            if row["group_id"] in contaminated_families:
                replay_exclusions["family_intersects_holdout"] += 1; continue
            if text in seen or text in replay_texts:
                replay_exclusions["duplicate_input"] += 1; continue
        if text in seen and seen[text]["intent"] != row["intent"]: raise ValueError(f"Legacy label conflicts with declaration: {text}")
        replay_texts.add(text)
        replay.append({**deepcopy(row), "declared_split": "train", "source_role": "legacy_replay"})
    rows = authored + replay
    # Families/texts cannot cross folds, even after TypeScript and worker normalization.
    owners, targets = {}, {}
    for r in rows:
        for key in [("family", r["group_id"]), ("text", runtime_text(r["canonical"])), ("raw", base.normalized(r["message"]))]:
            assert key not in owners or owners[key] == r["declared_split"], (key, "cross-fold leakage")
            owners[key] = r["declared_split"]
        text = runtime_text(r["canonical"])
        assert text not in targets or targets[text] == r["intent"], text
        targets[text] = r["intent"]
    splits = {name: [r["event_id"] for r in rows if r["declared_split"] == name] for name in base.SPLITS}
    summary = {}
    for name in base.SPLITS:
        part = [r for r in rows if r["declared_split"] == name]
        per_class = {label: len({runtime_text(r["canonical"]) for r in part if r["intent"] == label}) for label in sorted(LABELS)}
        assert min(per_class.values()) >= (10 if name == "calibration" else 12), (name, per_class)
        summary[name] = {"rows": len(part), "unique_inputs_per_class": per_class,
                         "families": len({r["group_id"] for r in part}), "authored": sum(r["source_role"] == "authored" for r in part),
                         "automatic_logs": sum(r["origin"] == "bot_log" for r in part)}
        if name != "train": assert all(r["source_role"] == "authored" for r in part)
    assert sum(r["origin"] == "bot_log" for r in rows) == 31
    write_jsonl("commands.jsonl", rows)
    write_jsonl("authored-declarations.jsonl", declared)
    write_jsonl("parser-and-duplicate-exclusions.jsonl", rejected)
    (OUT / "split-manifest.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    shutil.copyfile(SOURCE, OUT / SOURCE.name)
    for filename in ("bot-logs-unreviewed.jsonl", "bot-review-queue.jsonl", "bot-label-review.csv"):
        shutil.copyfile(OLD / filename, OUT / filename)
    guidance = """# Action V4: авторская синтетика

Codex явно составил каждую новую фразу и её метку 2026-10-03. Исходник —
lia_action_v4_authored.txt; это синтетические декларации ассистента, а не человеческая разметка.
Контрастные семейства заранее распределены по train / validation / calibration / test.
Совпадающие после нормализации входы исключены, родственные семьи не переходят между выборками.
TypeScript проверяет совместимость положительного примера с ботом и НЕ создаёт метку.
Исключения доступны в parser-and-duplicate-exclusions.jsonl, их нельзя скрывать в accuracy.

Исторический V3 — только train replay. Его метрики не являются новым независимым тестом.
Новые holdout семьи имеют собственные декларации; общие языковые конструкции остаются.
Это проверка парафразов разработки, не новых пользователей, звука или речи Whisper.
31 автоматически размеченная реальная запись — только train. 44 исходных записи сохранены
без изменения; 13 неоднозначных записей не обучаются. Предсказания модели не служат gold.
Логи содержат приватную речь: не публикуйте весь ZIP/ноутбук без собственной проверки.

Сэмплер ограничивает долю исторического replay до 20% каждой группы класса;
пунктуационные дубликаты не увеличивают вероятность. Два вопроса worker учатся на каждой фразе вместе.
Unknown включает поиск музыки, режимы, вопросы, запреты, чужие устройства и составные команды.
Модель action выбирает только семь управлений или unknown; функции режимов/доступ — код бота.
"""
    (OUT / "README.md").write_text(guidance, encoding="utf-8")
    manifest = {"dataset": "lia-action-v4", "authored_by": "Codex", "authored_date": "2026-10-03",
        "human_reviewed": False, "new_literal_declarations": len(declared), "authored_accepted": len(authored),
        "authored_families": len({r["group_id"] for r in authored}), "parser_exclusions": dict(Counter(r["reason"] for r in rejected)),
        "historical_replay_rows": len(replay), "historical_exclusions": dict(replay_exclusions), "rows": len(rows),
        "all_bot_log_events": 44, "automatic_bot_log_events": 31, "pending_bot_log_events": 13,
        "replay_fraction": .20, "historical_used_for_holdouts": False,
        "summary": summary, "source_sha256": snap["source_sha256"],
        "authored_source_sha256": base.sha256_file(SOURCE), "parent_manifest_sha256": base.sha256_file(OLD / "manifest.json"),
        "limitations": ["Assistant synthetic declarations, not human labels", "No independent real-user accuracy",
                        "Holdouts are related-language development paraphrases, not unseen speakers", "Parser limits excluded explicitly"],
        "file_sha256": {p.name: base.sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file(): z.write(p, p.name)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_sha256", "file_sha256")}, ensure_ascii=False, indent=2))
    print("ZIP", archive_path)


if __name__ == "__main__": main()
