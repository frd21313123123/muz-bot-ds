"""Generate a private self-contained V3 Kaggle notebook with semantic data and resume."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile
from build_laya_notebook import snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/datasets/lia-semantic-v3"
OUTPUT = DATA.parent / "lia_kaggle_v3_embedded.ipynb"


def main():
    snap = snapshot()
    for key, filename in (("previous_helper", "laya_training_v2.py"), ("v3_helper", "laya_training_v3.py")):
        snap[key] = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        snap["source_sha256"]["scripts/" + filename] = hashlib.sha256(snap[key].encode()).hexdigest()
    for filename in ("kaggle_v3_workflow.py", "kaggle_v3_finish.py", "build_lia_kaggle_v3.py", "build_lia_dataset_v3.py"):
        snap["source_sha256"]["scripts/" + filename] = hashlib.sha256((ROOT / "scripts" / filename).read_bytes()).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(DATA.iterdir()):
            if path.is_file(): archive.write(path, path.name)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    chunks = "\n".join(repr(encoded[i:i + 120]) for i in range(0, len(encoded), 120))
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "id": f"v3-{len(cells):02d}", "source": textwrap.dedent(source).strip() + "\n"})
    def code(source, tag):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "id": f"v3-{len(cells):02d}", "source": source, "execution_count": None, "outputs": []})
    md('''
    # LIA / Laya V3 — Kaggle, все данные встроены

    Импортируйте этот `.ipynb`, включите **GPU T4/P100 и Internet**. Оставьте `SMOKE_TEST=False`.
    Запускайте **Save Version → Save & Run All**. Такой запуск сохраняет результат `/kaggle/working` в Output версии;
    интерактивная среда сама по себе не служит резервной копией. [Документация Kaggle](https://www.kaggle.com/docs/notebooks).
    После окончания скачайте ZIP модели и `lia-v3-diagnostics.zip`. Для продолжения нужны также `resume/` и `candidates/`.
    Дополнительные датасеты, исходники бота и предыдущие веса загружать не нужно. На T4×2 используется одна GPU.

    V3 исправляет конфликт прежнего planner: **смысл команды определяется по тексту**, а владелец, канал,
    фаза сессии, имя активации и право ответа проверяются кодом. Команда в запрещённой сессии больше не учит
    модель считать её смысл `unknown`. Выбор инструмента вычисляется из разрешённого смыслового действия.
    Это соответствует текущему устройству бота; новую semantic-модель пока нельзя подставлять вместо action.

    Внутри находятся новый синтетический корпус, все **44 записи бота** и проверяемые сценарии правил.
    **31** запись с автоматической разметкой используется только в train, **13** неоднозначных оставлены для проверки.
    Предсказания прошлых моделей не становятся правильными метками. Метки синтетики и логов не проверены человеком.
    Сохраняйте ноутбук приватным: внутри есть реальные расшифровки. Это набор для разработки, не независимая оценка пользователей.

    Обе модели начинают с закреплённой исходной multilingual-модели. Балансируются классы, типы негативов,
    семьи и уникальные тексты. Обучаются голова и четыре последних слоя encoder; validation выбирает лучшую эпоху,
    calibration настраивает уверенность, test используется для заключительной проверки. Проверки качества сохраняются.
    Изменения устраняют обнаруженные проблемы процесса; улучшение точности нужно подтвердить полным запуском.
    ''')
    md('''
    ## 1. Настройки и продолжение

    Для первого запуска оставьте параметры. При OOM: `BATCH_SIZE=1`, `ACCUMULATION=16`, затем новая папка WORKDIR.
    Не уменьшайте пороги качества. После эпохи сохраняются веса, optimizer, scheduler, GradScaler и RNG.
    [PyTorch: сохранение состояния обучения](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

    Для продолжения после потери среды добавьте сохранённый Output прошлой версии как Input и задайте
    `RESUME_FROM` путём к папке `lia-v3` внутри `/kaggle/input/...`. Параметры и этот ноутбук должны совпадать.
    Если Output не сохранён, удалённые веса восстановить невозможно; данные в этом ноутбуке остаются доступны.
    ''')
    code('''
    from pathlib import Path
    import os, sys, json, hashlib, shutil
    WORKDIR = Path("/kaggle/working/lia-v3")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HOME", "/kaggle/temp/lia-v3-hf")
    BASE_MODEL = "convaiinnovations/laya-multilingual"
    BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    SEED = 20261010
    EPOCHS = 10
    ACTION_DRAWS = 4096
    SEMANTIC_DRAWS = 4096
    BATCH_SIZE = 2
    ACCUMULATION = 8
    HEAD_LR = 5e-5
    ENCODER_LR = 5e-6
    LAST_ENCODER_LAYERS = 4
    MIN_RECALL = .80
    MIN_MACRO_F1 = .80
    TRAIN_SEMANTIC = True
    SMOKE_TEST = False
    RESUME = True
    RESUME_FROM = ""  # e.g. /kaggle/input/<saved-notebook-output>/lia-v3
    if RESUME_FROM:
        previous_output = Path(RESUME_FROM).resolve()
        if not (previous_output / "resume").is_dir() or not (previous_output / "candidates").is_dir():
            raise FileNotFoundError("RESUME_FROM must contain resume/ AND candidates/")
        for name in ("resume", "candidates"):
            if (WORKDIR / name).exists():
                raise FileExistsError("Do not overwrite existing training state; clear RESUME_FROM to resume local WORKDIR")
            shutil.copytree(previous_output / name, WORKDIR / name)
    ''', "parameters")
    md("## 2. Встроенный датасет\n\nSHA-256 проверяет архив и каждый файл. Непроверенные логи и сценарии доступа не попадают в обучение NLI.")
    code(f'''import base64, io, zipfile
EMBEDDED_DATASET_BASE64 = (
{chunks}
)
blob = base64.b64decode(EMBEDDED_DATASET_BASE64, validate=True)
assert hashlib.sha256(blob).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
DATA_DIR = WORKDIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    assert archive.testzip() is None
    for member in archive.infolist():
        (DATA_DIR / member.filename).resolve().relative_to(DATA_DIR.resolve())
    archive.extractall(DATA_DIR)
DATA_MANIFEST = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
for name, digest in DATA_MANIFEST["file_sha256"].items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest, name
print({{k: DATA_MANIFEST[k] for k in ("action_events", "semantic_events", "policy_fixtures", "all_bot_log_events", "pending_bot_log_events")}})
''', "embedded-data")
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    md("## 3. Зависимости\n\nСохраняем CUDA PyTorch среды Kaggle. Установка и первое скачивание базовых весов требуют Internet.")
    code('''
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "laya==0.3.22", "transformers==5.17.0",
                    "huggingface-hub==1.33.0", "matplotlib>=3.9,<4"], check=True)
    if not shutil.which("node"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "nodejs"], check=True)
    ''', "install")
    code('''
    import importlib.util, importlib.metadata, random, gc, platform
    import numpy as np
    import torch, laya
    import matplotlib.pyplot as plt
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE != "cuda" and not SMOKE_TEST: raise RuntimeError("Enable GPU in Kaggle settings")
    assert importlib.metadata.version("laya") == "0.3.22"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    print({"device": DEVICE, "GPU": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "torch": torch.__version__, "python": platform.python_version()})
    ''', "environment")
    md('''
    ## 4. Контракт бота

    Action: восемь классов и оба вопроса реального worker с его обработкой текста и порогами.
    Semantic: шестнадцать смысловых действий по каноническому тексту, без owner/channel/phase и меток внутри входа.
    Число громкости, ограничения 1–150, распознавание имени и доступ остаются в коде. Whisper и TTS не обучаются.
    ''')
    code("SNAPSHOT = json.loads(" + repr(json.dumps(snap, ensure_ascii=False)) + ")\n" + textwrap.dedent('''
    contract_dir = WORKDIR / "contract" / "scripts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    for key, name in (("helper", "laya_training.py"), ("previous_helper", "laya_training_v2.py"),
                      ("v3_helper", "laya_training_v3.py"), ("worker", "voice_worker.py"), ("bridge", "bot_parser.mjs")):
        (contract_dir / name).write_text(SNAPSHOT[key], encoding="utf-8")
    (contract_dir / "laya_music_v3.json").write_text(json.dumps(SNAPSHOT["music_schema"]), encoding="utf-8")
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    training = load_module("laya_training", contract_dir / "laya_training.py")
    legacy = load_module("laya_training_v2", contract_dir / "laya_training_v2.py")
    v3 = load_module("laya_training_v3", contract_dir / "laya_training_v3.py")
    worker = load_module("lia_worker_v3", contract_dir / "voice_worker.py")
    bridge_path = contract_dir / "bot_parser.mjs"
    SEMANTIC_QUESTIONS = SNAPSHOT["intent_questions"]
    ''').lstrip(), "contract")
    md('''
    ## 5. Проверить данные и зафиксированное разделение

    Точные исходные/канонические тексты и семьи держатся вместе для обеих моделей.
    Автоматические логи используются только в train. Test содержит синтетические данные разработки,
    в том числе ранее просмотренные фразы V1/V2; свежая реальная оценка требует новых ручных меток.
    ''')
    code('''
    labels = SEMANTIC_QUESTIONS["intent"]["criteria"]
    action_rows = v3.load_commands(DATA_DIR / "action-commands.jsonl", bridge_path, labels)
    semantic_rows = v3.load_commands(DATA_DIR / "semantic-commands.jsonl", bridge_path, labels)
    for row in semantic_rows: row["task"] = "semantic"
    all_rows = action_rows + semantic_rows
    by_id = {r["event_id"]: r for r in all_rows}
    assert len(by_id) == len(all_rows)
    ids = json.loads((DATA_DIR / "split-manifest.json").read_text(encoding="utf-8"))
    assert set(by_id) == {i for part in ids.values() for i in part}
    assert sum(map(len, ids.values())) == len(by_id)
    all_splits = {name: [by_id[i] for i in ids[name]] for name in training.SPLITS}
    locations, semantics = {}, {}
    for name, rows in all_splits.items():
        for row in rows:
            for key in [("family", row["group_id"])] + [("text", training.normalized(row[k])) for k in ("message", "canonical")]:
                assert key not in locations or locations[key] == name, "Cross-fold leakage"
                locations[key] = name
            if row.get("task") == "semantic":
                text = v3.runtime_text(row["canonical"])
                assert text not in semantics or semantics[text] == row["intent"], "Contradictory semantic labels"
                semantics[text] = row["intent"]
    action_splits = {name: [r for r in rows if r.get("task") != "semantic"] for name, rows in all_splits.items()}
    semantic_splits = {name: [r for r in rows if r.get("task") == "semantic"] for name, rows in all_splits.items()}
    split_summary = {}
    action_labels = set(worker.QUESTIONS["action"]["criteria"])
    for name in training.SPLITS:
        assert {r["intent"] if r["intent"] in action_labels else "unknown" for r in action_splits[name]} == action_labels
        assert {r["intent"] for r in semantic_splits[name]} == set(labels)
        split_summary[name] = {"action": len(action_splits[name]), "semantic": len(semantic_splits[name]),
            "semantic_unique_texts": len({v3.runtime_text(r["canonical"]) for r in semantic_splits[name]}),
            "semantic_families": len({r["group_id"] for r in semantic_splits[name]}),
            "bot_logs_per_model": sum(r["origin"] == "bot_log" for r in action_splits[name])}
        if name != "train": assert all(r["origin"] != "bot_log" for r in all_splits[name])
    assert split_summary["train"]["bot_logs_per_model"] == 31
    assert sum(r["origin"] == "bot_log" for r in semantic_splits["train"]) == 31
    print(json.dumps(split_summary, indent=2))
    v3.atomic_json(WORKDIR / "diagnostics" / "split-summary.json", split_summary)
    if SMOKE_TEST:
        EPOCHS = 1; ACTION_DRAWS = 32; SEMANTIC_DRAWS = 32
        for corpus in (action_splits, semantic_splits):
            for name, rows in corpus.items():
                selected = {}
                for row in rows:
                    if row["intent"] not in selected: selected[row["intent"]] = row
                corpus[name] = list(selected.values())
    action_records = {name: v3.records(rows, worker) for name, rows in action_splits.items()}
    semantic_records = {name: v3.records(rows, worker, SEMANTIC_QUESTIONS) for name, rows in semantic_splits.items()}
    ''', "data-loading")
    md('''
    ## 6. Правила сессии и выбора функции

    Следующая проверка использует правильный intent как заглушку, чтобы проверить только код доступа/маршрутизации.
    Её результат **не является точностью модели**. В запрещённых сессиях и при отдельном имени активации NLI не вызывается.
    ''')
    code('''
    fixtures = training.read_jsonl([DATA_DIR / "policy-fixtures.jsonl"])
    probes = training.node_bridge([{**r["state"], "message": r["state"]["message"]} for r in fixtures], bridge_path)
    model_calls = 0
    for row, probe in zip(fixtures, probes, strict=True):
        calls = []
        def oracle(text):
            calls.append(text)
            return row["gold"]["intent"]
        decision = v3.policy_decision(row["state"], oracle, probe)
        assert decision == row["gold"], (row["event_id"], decision, row["gold"])
        expected_call = row["state"]["phase"] == "awaiting" and all(row["state"][k] for k in ("voice_enabled", "in_bot_channel", "is_owner"))
        assert bool(calls) == expected_call
        model_calls += len(calls)
    policy_report = {"fixtures": len(fixtures), "passed": True, "eligible_model_calls": model_calls,
                     "measures": "deterministic policy with oracle intent; NOT learned model accuracy"}
    v3.atomic_json(WORKDIR / "diagnostics" / "policy-check.json", policy_report)
    print(policy_report)
    ''', "policy-check")
    md('''
    ## 7. Обучение и проверка экспорта

    NLI обучается supervised cross entropy, отдельно для action и semantic. Семейства и типы негативов
    балансируются; варианты регистра/пунктуации не повышают вес одной фразы. Act/escalate head заморожен.
    Проверяется наличие градиентов encoder и число реальных шагов optimizer. Пропущенные AMP-шаги видны в отчёте.

    Лучший checkpoint выбирается по validation macro-F1, затем средней/минимальной recall и NLL.
    Допуск требует macro-F1 ≥ 0,80 и recall каждого класса ≥ 0,80; action дополнительно проверяется реальным
    worker + TypeScript: model_required accuracy ≥ 0,85 и false control rate ≤ 0,02. Уверенность калибруется отдельно.
    Semantic проверяется также с порогом отказа 0,60. Ни test, ни calibration не выбирают эпоху.

    После каждой эпохи в `resume/` сохраняется состояние. После каждого этапа формируется ZIP:
    прошедшие проверки веса либо явно обозначенный **CANDIDATE** для анализа, плюс диагностика.
    Это позволяет прислать неудачный результат для проверки. CANDIDATE не подключайте к боту.
    ''')
    code((ROOT / "scripts/kaggle_v3_workflow.py").read_text(encoding="utf-8"), "stage-functions")
    md("## 8. Action — checkpoint для текущего worker")
    code('''
    reports = {}
    reports["action"] = run_stage("action", action_records, ACTION_DRAWS)
    ''', "train-action")
    md("## 9. Semantic — отдельный checkpoint\n\nРаспознаёт play, управление, режимы и unknown. Для использования этой модели нужен отдельный этап интеграции с ботом.")
    code('''
    if TRAIN_SEMANTIC:
        reports["semantic"] = run_stage("semantic", semantic_records, SEMANTIC_DRAWS)
    ''', "train-semantic")
    md("## 10. Графики и реальные шаги обучения\n\nСмотрите macro-F1 и полноту каждого класса. Высокая общая accuracy не компенсирует провал pause, play или unknown.")
    code('''
    for stage, report in reports.items():
        history = report["training"]["history"]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot([r["epoch"] for r in history], [r["validation_score"][0] for r in history], "s-", label="Macro-F1")
        ax.plot([r["epoch"] for r in history], [r["validation_score"][2] for r in history], "o-", label="Минимальная recall")
        ax.axhline(MIN_RECALL, ls="--", color="#555555", label="Порог")
        ax.set(xlabel="Эпоха", ylabel="Доля", ylim=(0, 1.04), title=stage + ": validation")
        ax.set_xticks([r["epoch"] for r in history]); ax.legend(); fig.tight_layout(); plt.show()
        for task, values in report["metrics"]["test"].items():
            labels = list(values["per_class"])
            fig, ax = plt.subplots(figsize=(11, max(3, len(labels) * .32)))
            bars = ax.barh(labels, [values["per_class"][label]["recall"] for label in labels],
                           color=["#245b91" if values["per_class"][label]["recall"] >= MIN_RECALL else "#b24d32" for label in labels])
            ax.axvline(MIN_RECALL, ls="--", color="#555555")
            for bar, label in zip(bars, labels):
                ax.text(min(.90, bar.get_width() + .02), bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.0%}; n={values["per_class"][label]["support"]}', va="center")
            ax.set(xlabel="Recall на синтетическом test", xlim=(0, 1.17), title=task + f'; macro-F1={values["macro_f1"]:.3f}')
            fig.tight_layout(); plt.show()
        print(stage, "QUALIFIED:", report["qualified_on_synthetic_holdouts"])
        print([{k: r[k] for k in ("epoch", "optimizer_updates", "skipped_updates", "amp_scale")} for r in history])
    ''', "plots")
    md('''
    ## 11. Результаты и восстановление переменных

    Ячейка ниже работает после потери переменных, если файлы сохранились в `/kaggle/working/lia-v3`.
    Output сохранённой версии содержит модели, данные, `candidates/`, `resume/` и диагностику.
    Из-за размеров сохраняются только лучший checkpoint и последнее состояние обучения, не все эпохи.
    ''')
    code((ROOT / "scripts/kaggle_v3_finish.py").read_text(encoding="utf-8"), "diagnostics")
    md('''
    ## 12. Что можно подключить

    `laya-muz-bot-controls-v3.zip` без CANDIDATE/SMOKE-ONLY — action, прошедший заданные проверки синтетического корпуса.
    До использования проверьте новые реальные расшифровки. Распакуйте модель локально, задайте `LAYA_MODEL_PATH`
    и перезапустите бот. `laya-muz-bot-semantic-v3.zip` — отдельная будущая модель: заменять ею action нельзя.
    Ноутбук не меняет сервер, `.env` и slash-команды. Старое имя `laya-muz-bot-ds-v3` активирует другой профиль — не используйте его.

    `parser-limitations.jsonl` показывает просьбы, чей смысл расходится с текущими TypeScript-правилами;
    они учат semantic, но не выдаются за корректные примеры action pipeline. Например, парсер может принять
    «поставь музыку на паузу» за поиск. Одними весами это ограничение не исправляется.
    Точность Whisper/аудио, работа Discord/YouTube и задержка этим ноутбуком не измеряются.
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}}, "cells": cells}
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
    print("cells", len(cells), "bytes", OUTPUT.stat().st_size)


if __name__ == "__main__": main()
