"""Build a single Kaggle notebook with private embedded V2 data and separate models."""
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
DATA = ROOT / ".runtime/datasets/lia-balanced-v2"
OUTPUT = DATA.parent / "lia_kaggle_v2_embedded.ipynb"


def main():
    snap = snapshot()
    snap["balanced_helper"] = (ROOT / "scripts/laya_training_v2.py").read_text(encoding="utf-8")
    snap["source_sha256"]["scripts/laya_training_v2.py"] = hashlib.sha256(snap["balanced_helper"].encode()).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(DATA.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    chunks = "\n".join(repr(encoded[i:i + 120]) for i in range(0, len(encoded), 120))
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "id": f"v2-{len(cells):02d}", "source": textwrap.dedent(source).strip() + "\n"})
    def code(source, tag):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {"tags": [tag]}, "id": f"v2-{len(cells):02d}", "source": source, "execution_count": None, "outputs": []})
    md('''
    # LIA / Laya V2 — Kaggle, датасет внутри

    В Kaggle импортируйте этот файл, включите **GPU T4/P100 и Internet**, затем Run All.
    Дополнительные ZIP, JSONL и предыдущие веса не нужны. Используется закреплённая исходная multilingual-модель;
    неудачный checkpoint V1 не продолжаем обучать. На T4×2 используется одна GPU, модели обучаются последовательно.

    Action и planner получают **отдельные веса**. Балансируется задача → класс → семейство фраз;
    дообучаются голова и два последних encoder-слоя. Лучший checkpoint выбирается сначала по минимальной
    полноте классов, затем macro-F1, а не по общей accuracy/NLL. Calibration отделена от validation и test.

    Экспорт для бота создаётся только после проверок классов, фактического worker с порогами уверенности и повторной загрузки.
    Если проверки не пройдены, сохраняется диагностический отчёт; успешное завершение обучения само по себе не означает пригодность.
    Настройки исправляют выявленные проблемы процесса, но не гарантируют требуемую точность до выполнения обучения.

    Встроены 3792 синтетические команды, 12212 planner-сценариев и все 44 записи бота: 31 с автоматическими метками,
    13 без меток для ручной проверки. 58 новых семейств прошли проверку правил бота. Метки не подтверждены человеком.
    В файле есть реальные расшифровки: сохраняйте ноутбук приватным. Ранее просмотренный V1 test частично входит в корпус;
    V2 test — диагностический синтетический набор, а не свежая оценка на реальных пользователях.
    ''')
    md('''
    ## 1. Параметры

    Для обычного обучения оставьте SMOKE_TEST=False. TRAIN_PLANNER=True обучает второй экспериментальный checkpoint;
    он пока не подключён к боту. Проверки владельца, канала, имени, таймеров и отмены остаются в коде независимо от модели.
    Не снижайте пороги качества, чтобы скрыть провал классов. При OOM уменьшите BATCH_SIZE до 1, увеличив ACCUMULATION до 16.
    ''')
    code('''
    from pathlib import Path
    import os, json, sys
    WORKDIR = Path("/kaggle/working/lia-v2")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HOME", str(WORKDIR / "hf-cache"))
    BASE_MODEL = "convaiinnovations/laya-multilingual"
    BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    SEED = 20261002
    EPOCHS = 6
    ACTION_DRAWS = 3072
    PLANNER_DRAWS = 6144
    BATCH_SIZE = 2
    ACCUMULATION = 8
    HEAD_LR = 5e-5
    ENCODER_LR = 3e-6
    LAST_ENCODER_LAYERS = 2
    TRAIN_PLANNER = True
    MIN_RECALL = 0.80
    MIN_MACRO_F1 = 0.80
    SMOKE_TEST = False
    ''', "parameters")
    md("## 2. Распаковать встроенные данные\n\nПроверяется целостность архива и каждого файла. Неоднозначные логи доступны для просмотра, но не загружаются в обучение.")
    code(f'''import base64, hashlib, io, zipfile
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
print({{k: DATA_MANIFEST[k] for k in ("synthetic_commands", "planner_scenarios", "bot_log_rows", "bot_pending_rows")}})
''', "embedded-data")
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    md("## 3. Зависимости\n\nCUDA PyTorch, установленный Kaggle, сохраняется. Для чистого первого запуска нужен Internet.")
    code('''
    import subprocess, shutil
    subprocess.run([sys.executable, "-m", "pip", "install", "laya==0.3.22", "transformers==5.17.0",
                    "huggingface-hub==1.33.0", "matplotlib>=3.9,<4"], check=True)
    if not shutil.which("node"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "nodejs"], check=True)
    ''', "install")
    code('''
    import importlib.util, importlib.metadata, random, gc, platform, shutil
    import numpy as np
    import torch, laya
    import matplotlib.pyplot as plt
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE != "cuda" and not SMOKE_TEST:
        raise RuntimeError("Enable GPU in Kaggle Notebook settings")
    assert importlib.metadata.version("laya") == "0.3.22"
    plt.rcParams.update({"figure.figsize": (10, 4), "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    print({"device": DEVICE, "GPU": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "torch": torch.__version__, "python": platform.python_version()})
    ''', "environment")
    md('''
    ## 4. Контракт и загрузчики

    Action обучается на той же строке и обоих вопросах primary/short, что получает реальный worker.
    Режимы и поиск музыки не добавляются в восемь классов action; они обрабатываются TypeScript.
    Модель выбирает действие, а число громкости разбирает код. Whisper и TTS здесь не обучаются.
    ''')
    code("SNAPSHOT = json.loads(" + repr(json.dumps(snap, ensure_ascii=False)) + ")\n" + textwrap.dedent('''
    contract_dir = WORKDIR / "contract" / "scripts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    for key, name in (("helper", "laya_training.py"), ("balanced_helper", "laya_training_v2.py"), ("worker", "voice_worker.py"), ("bridge", "bot_parser.mjs")):
        (contract_dir / name).write_text(SNAPSHOT[key], encoding="utf-8")
    (contract_dir / "laya_music_v3.json").write_text(json.dumps(SNAPSHOT["music_schema"]), encoding="utf-8")
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    training = load_module("laya_training", contract_dir / "laya_training.py")
    balanced = load_module("laya_training_v2", contract_dir / "laya_training_v2.py")
    worker = load_module("lia_worker_v2", contract_dir / "voice_worker.py")
    bridge_path = contract_dir / "bot_parser.mjs"
    PLANNER_QUESTIONS = training.planner_questions(SNAPSHOT["intent_questions"])
    ''').lstrip(), "contract")
    md('''
    ## 5. Совместное разделение, раздельное обучение

    Одна фраза и её контекстные варианты остаются в одной части одновременно для action и planner.
    Разделение зафиксировано до обучения, по семействам и совпадениям исходного/канонического текста.
    Логи одного файла сохраняются вместе; принадлежность логов train/test показывается явно.
    Сохраняются отдельные train / validation / calibration / test. Ни calibration, ни test не выбирают эпоху.
    ''')
    code('''
    commands = training.prepare_commands(training.load_commands(
        [DATA_DIR / "commands-synthetic.jsonl", DATA_DIR / "bot-commands-labeled.jsonl"],
        intent_labels=SNAPSHOT["intent_questions"]["intent"]["criteria"], allow_automatic_labels=True), bridge_path)
    planner = training.load_planner([DATA_DIR / "planner-synthetic.jsonl"], PLANNER_QUESTIONS, bridge_path, True)
    parsed = training.node_bridge([{"message": r["message"], "player": r["state"]["player"]} for r in planner], bridge_path)
    for row, probe in zip(planner, parsed, strict=True): row["canonical"] = probe["canonical"]
    all_rows = commands + planner
    by_id = {r["event_id"]: r for r in all_rows}
    assert len(by_id) == len(all_rows)
    ids = json.loads((DATA_DIR / "split-manifest.json").read_text())
    assert set(by_id) == {i for part in ids.values() for i in part}
    assert sum(map(len, ids.values())) == len(by_id)
    all_splits = {name: [by_id[i] for i in ids[name]] for name in training.SPLITS}
    locations = {}
    for name, rows in all_splits.items():
        for row in rows:
            keys = [("family", row["group_id"])] + [("text", training.normalized(row[k])) for k in ("message", "canonical")]
            for key in keys:
                assert key not in locations or locations[key] == name, "Leakage: " + str(key)
                locations[key] = name
    command_splits = {name: [r for r in rows if r.get("task") != "planner"] for name, rows in all_splits.items()}
    planner_splits = {name: [r for r in rows if r.get("task") == "planner"] for name, rows in all_splits.items()}
    action_labels = set(worker.QUESTIONS["action"]["criteria"])
    for name in training.SPLITS:
        assert {r["intent"] if r["intent"] in action_labels else "unknown" for r in command_splits[name]} == action_labels
        for qid, question in PLANNER_QUESTIONS.items():
            assert {r["gold"][qid] for r in planner_splits[name]} == set(question["criteria"]), (name, qid)
    assert all(r["origin"] != "bot_log" for name in training.SPLITS[1:] for r in command_splits[name])
    print({name: {"commands": len(command_splits[name]), "planner": len(planner_splits[name]),
                  "bot_logs": sum(r["origin"] == "bot_log" for r in command_splits[name])} for name in ids})
    if SMOKE_TEST:
        # Only bounded code verification, never a usable model or quality measurement.
        EPOCHS = 1; ACTION_DRAWS = 32; PLANNER_DRAWS = 48; LAST_ENCODER_LAYERS = 2
        for corpus in (command_splits, planner_splits):
            for name, rows in corpus.items():
                selected = {}
                for row in rows:
                    label = row["intent"] if corpus is command_splits else "|".join(row["gold"].values())
                    if len(selected.setdefault(label, [])) < 1: selected[label].append(row)
                corpus[name] = [r for bucket in selected.values() for r in bucket]
    action_records = {name: training.runtime_records(rows, worker) for name, rows in command_splits.items()}
    planner_records = {name: [{"event_id": r["event_id"], "task": "planner", "state": r["state"],
                               "questions": PLANNER_QUESTIONS, "gold": r["gold"]} for r in rows] for name, rows in planner_splits.items()}
    groups = {r["event_id"]: r["group_id"] for r in all_rows}
    ''', "data-loading")
    md('''
    ## 6. Обучение, калибровка и допуск экспорта

    В каждой эпохе balanced sampler выбирает равные доли вопросов и классов, затем случайное семейство и вариант.
    Поток аудитов показывает фактическое распределение. Encoder частично обучается с gradient checkpointing;
    head act/escalate заморожен, поскольку для него нет gold. Gradient accumulation учитывает неполное последнее окно.

    Проверки требуют macro-F1 ≥ 0,80 и recall каждого класса ≥ 0,80 для обоих action-вопросов и всех planner-вопросов.
    Дополнительно action проверяется через реальный worker с порогами: recall модельных классов ≥ 0,80,
    точность model_required ≥ 0,85 и доля ошибочного управления на негативных просьбах ≤ 0,02.
    Эти инженерные пороги заданы заранее, не доказывают качество на реальных пользователях и не подбираются по test.
    ''')
    code('''
    def runtime_gate(agent, rows):
        metrics, predictions = training.runtime_evaluation(agent, rows, worker, bridge_path)
        failures = []
        selected = [r for r in predictions if r["model_required"]]
        recalls = {}
        for label in sorted({r["gold"] for r in selected}):
            relevant = [r for r in selected if r["gold"] == label]
            recalls[label] = sum(r["prediction"] == label for r in relevant) / len(relevant)
            if recalls[label] < MIN_RECALL: failures.append(f"pipeline/{label} recall={recalls[label]:.3f}")
        if metrics["model_required_accuracy"] is not None and metrics["model_required_accuracy"] < 0.85:
            failures.append("pipeline model_required_accuracy < 0.85")
        if metrics["false_control_rate"] > 0.02: failures.append("pipeline false_control_rate > 0.02")
        return {"passed": not failures, "failures": failures, "per_class_recall": recalls, "metrics": metrics}

    def run_stage(stage, records, draws):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Each stage starts from the pinned base; no shared trained heads or optimizer.
        agent = laya.load(BASE_MODEL, device=DEVICE, revision=BASE_REVISION)
        balanced.install_temperatures(agent, {"temperature": [1., 1., 1.], "temperature_by_options": {}})
        training.batches.pad_id = agent.tok.pad_token_id
        items = {name: balanced.encode(agent, rows, groups) for name, rows in records.items()}
        for name, rows in items.items():
            support = training.Counter((r["task"], r["qid"], r["labels"][r["label"]]) for r in rows)
            print(stage, name, dict(support))
        model_name = "laya-muz-bot-controls-v2" if stage == "action" else "laya-muz-bot-planner-v2"
        if SMOKE_TEST: model_name += "-SMOKE-ONLY"
        directory = WORKDIR / "candidates" / model_name
        # A failed rerun must not leave a previous approved ZIP or stale digests.
        stale_archive = WORKDIR / (model_name + ".zip")
        if stale_archive.exists(): stale_archive.unlink()
        stale_digests = directory / "sha256.json"
        if stale_digests.exists(): stale_digests.unlink()
        stale_report = directory / "training_report.json"
        if stale_report.exists(): stale_report.unlink()
        info = balanced.train(agent, items["train"], items["validation"], directory, model_name,
            epochs=EPOCHS, draws=draws, batch_size=BATCH_SIZE, accumulation=ACCUMULATION,
            head_lr=HEAD_LR, encoder_lr=ENCODER_LR, last_layers=LAST_ENCODER_LAYERS, seed=SEED)
        temperatures = balanced.calibrate(training.raw_logits(agent, items["calibration"], BATCH_SIZE))
        balanced.install_temperatures(agent, temperatures)
        metrics = {name: training.evaluate_logits(training.raw_logits(agent, items[name], BATCH_SIZE), temperatures)[0]
                   for name in ("validation", "test")}
        gates = {name: balanced.metric_gate(values, MIN_RECALL, MIN_MACRO_F1) for name, values in metrics.items()}
        if stage == "action":
            for name in ("validation", "test"):
                gates[name + "_pipeline"] = runtime_gate(agent, command_splits[name])
        training.save_checkpoint(agent, directory, model_name, temperatures)
        # Record the actual balanced method, not the legacy training description.
        config_path = directory / "rl_agent_config.json"
        cfg = json.loads(config_path.read_text())
        cfg["training"].update({"sampling": info["sampling"], "separate_task": stage,
            "last_encoder_layers": LAST_ENCODER_LAYERS, "checkpoint_selection": "min_class_recall_then_macro_f1_then_nll"})
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        probes = records["test"][:3]
        before = [{k: v["choice"] for k, v in agent.predict(r["state"], r["questions"])["answers"].items()} for r in probes]
        balanced.release(agent); del agent; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        reloaded = laya.load(str(directory), device=DEVICE)
        after = [{k: v["choice"] for k, v in reloaded.predict(r["state"], r["questions"])["answers"].items()} for r in probes]
        assert before == after, "Reload changed choices; refuse export"
        balanced.release(reloaded); del reloaded; gc.collect()
        qualified = not SMOKE_TEST and all(g["passed"] for g in gates.values())
        stage_report = {"stage": stage, "smoke_only": SMOKE_TEST, "qualified_on_synthetic_holdouts": qualified,
            "planner_enabled_in_bot": False, "data_manifest": DATA_MANIFEST, "source_sha256": SNAPSHOT["source_sha256"],
            "seed": SEED, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "epochs": EPOCHS, "draws_per_epoch": draws, "batch_size": BATCH_SIZE, "accumulation": ACCUMULATION,
            "head_lr": HEAD_LR, "encoder_lr": ENCODER_LR, "min_recall": MIN_RECALL, "min_macro_f1": MIN_MACRO_F1,
            "training": info, "temperatures": temperatures, "metrics": metrics, "gates": gates,
            "checkpoint_roundtrip": "passed", "split_event_ids": ids,
            "environment": {k: importlib.metadata.version(k) for k in ("torch", "laya", "transformers", "huggingface-hub")},
            "limitation": "Synthetic/rule-labelled development holdouts; not independently reviewed real-user accuracy"}
        (directory / "training_report.json").write_text(json.dumps(stage_report, ensure_ascii=False, indent=2), encoding="utf-8")
        hashes = {str(p.relative_to(directory)): training.sha256_file(p) for p in directory.rglob("*") if p.is_file() and p.name != "sha256.json"}
        (directory / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        if qualified:
            archive = shutil.make_archive(str(WORKDIR / model_name), "zip", root_dir=directory.parent, base_dir=directory.name)
            print("QUALIFIED ON SYNTHETIC HOLDOUTS:", archive)
        else:
            print("NO USABLE MODEL ZIP:", stage, [g["failures"] for g in gates.values() if not g["passed"]],
                  "SMOKE ONLY" if SMOKE_TEST else "Inspect diagnostics; do not deploy candidate")
        return stage_report
    ''', "stage-functions")
    md("## 7. Action: восемь классов текущего worker")
    code('''
    reports = {}
    reports["action"] = run_stage("action", action_records, ACTION_DRAWS)
    ''', "train-action")
    md("## 8. Planner: отдельный экспериментальный checkpoint\n\nРезультат action не перезаписывается. Новые веса planner требуют отдельного подключения; доступ нельзя доверять модели.")
    code('''
    if TRAIN_PLANNER:
        reports["planner"] = run_stage("planner", planner_records, PLANNER_DRAWS)
    ''', "train-planner")
    md("## 9. Проверить каждый класс\n\nСмотрите особенно respond, player_control, pause, resume, skip, stop, volume_up/down. Общая accuracy не заменяет эти показатели.")
    code('''
    for stage, report in reports.items():
        history = report["training"]["history"]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot([r["epoch"] for r in history], [r["validation_score"][0] for r in history], "o-", label="Минимальная recall")
        ax.plot([r["epoch"] for r in history], [r["validation_score"][1] for r in history], "s-", label="Macro-F1")
        ax.axhline(MIN_RECALL, ls="--", color="#555555", label="Порог")
        ax.set(xlabel="Эпоха", ylabel="Доля", ylim=(0, 1.04), title=stage + ": выбор checkpoint по validation")
        ax.set_xticks([r["epoch"] for r in history]); ax.legend(); fig.tight_layout(); plt.show()
        for task, values in report["metrics"]["test"].items():
            labels = list(values["per_class"])
            recalls = [values["per_class"][label]["recall"] for label in labels]
            fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.3)))
            bars = ax.barh(labels, recalls, color=["#245b91" if r >= MIN_RECALL else "#b24d32" for r in recalls])
            ax.axvline(MIN_RECALL, ls="--", color="#555555")
            for bar, label in zip(bars, labels):
                ax.text(min(0.90, bar.get_width() + 0.02), bar.get_y() + bar.get_height() / 2,
                        f'{bar.get_width():.0%}; n={values["per_class"][label]["support"]}', va="center")
            ax.set(xlabel="Recall на синтетическом test", xlim=(0, 1.15), title=task + f'; macro-F1={values["macro_f1"]:.3f}')
            fig.tight_layout(); plt.show()
        print(stage, "EXPORT QUALIFIED:", report["qualified_on_synthetic_holdouts"])
    ''', "plots")
    md('''
    ## 10. Файлы результата

    Прошедшие проверки ZIP находятся в `/kaggle/working/lia-v2/`, скачайте их из Output после Save Version / Run All.
    `laya-muz-bot-controls-v2.zip` — action для текущего бота. `laya-muz-bot-planner-v2.zip` — отдельная будущая модель.
    Если ZIP отсутствует, gates не пройдены; кандидат в `candidates/` не следует подключать. Диагностика доступна всегда.
    После завершения проверяйте реальные новые записи до использования на сервере. Бот и его `.env` автоматически не меняются.
    ''')
    code((ROOT / "scripts/kaggle_finish_diagnostics.py").read_text(encoding="utf-8"), "diagnostics")
    md('''
    ## 11. Подключение и границы проверки

    Только прошедший проверки **action** распакуйте локально в `.runtime/voice/models/laya-muz-bot-controls-v2/`,
    установите `LAYA_MODEL_PATH` на эту директорию и перезапустите бота. Planner нельзя подставлять вместо action.
    Не называйте checkpoint `laya-muz-bot-ds-v3`: это другой старый профиль. Изменять slash-команды не требуется.

    Независимо от качества весов есть ограничения текущих TypeScript-правил: например, некоторые фразы
    «поставь музыку на паузу» интерпретируются как музыкальный поиск. Датасет не выдаёт такие несовместимые просьбы
    за корректные примеры текущего pipeline; кандидаты сохранены для проверки. Это требует отдельного исправления парсера.
    Реальная точность Whisper/аудио и Discord, latency и доступность YouTube этим ноутбуком не измеряются.

    Реализация использует [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    и [Transformers gradient checkpointing](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable).
    Примеры и функции Laya проверены по установленной версии 0.3.22; обновление версии требует повторной проверки.
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}}, "cells": cells}
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
    print("cells", len(cells), "bytes", OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
