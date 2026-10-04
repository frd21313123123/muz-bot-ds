"""Self-contained, Action-only training notebook with literal authored data."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
from build_laya_notebook import snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/datasets/lia-action-v4"
OUTPUT = DATA.parent / "lia_action_v4_embedded.ipynb"


def main():
    snap = snapshot()
    for key, filename in (("previous_helper", "laya_training_v2.py"), ("v3_helper", "laya_training_v3.py"), ("v4_helper", "laya_training_v4.py")):
        snap[key] = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        snap["source_sha256"]["scripts/" + filename] = hashlib.sha256(snap[key].encode()).hexdigest()
    for filename in ("kaggle_v4_workflow.py", "kaggle_v4_finish.py", "build_lia_kaggle_v4.py", "build_lia_dataset_v4.py"):
        snap["source_sha256"]["scripts/" + filename] = hashlib.sha256((ROOT / "scripts" / filename).read_bytes()).hexdigest()
    payload = DATA.with_suffix(".zip").read_bytes()
    encoded = base64.b64encode(payload).decode()
    chunks = "\n".join(repr(encoded[i:i + 120]) for i in range(0, len(encoded), 120))
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "id": f"v4-{len(cells)}", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})
    def code(text, tag):
        source = textwrap.dedent(text).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "id": f"v4-{len(cells)}", "metadata": {"tags": [tag]},
                      "execution_count": None, "outputs": [], "source": source})
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    md(f'''
    # LIA Action V4: авторская синтетика и совместное обучение двух вопросов

    **Этот ноутбук обучает только Action для текущего worker.** Все данные и исходники встроены.
    {manifest['new_literal_declarations']} фраз явно составлены Codex; {manifest['authored_accepted']} прошли проверку парсера и удаление дубликатов.
    {manifest['rows']} записей обучения/оценки с историческим replay. Логи: 44 исходные, 31 автоматически размеченная только в train,
    13 неоднозначных вне обучения. Это приватный ноутбук с речью участников; выбирайте приватный запуск.

    V3 путала pause/stop и resume/skip; в коротком вопросе pause имела recall 0%. V4 добавляет контрастные фразы,
    учит оба реальных вопроса на каждой команде вместе и выбирает checkpoint с учётом худших классов.
    **Повышение качества не обещано:** оно измеряется после полного запуска. Новые тестовые примеры — авторские
    парафразы разработки, не независимые пользователи или аудио. Размеры классов показаны рядом с метриками.

    Kaggle: импортировать этот `.ipynb`, включить GPU и Internet, выполнить Run All / Save & Run All, сохранить Output.
    Colab: открыть `.ipynb`, выбрать GPU и выполнить все ячейки. Отдельный Dataset добавлять не нужно.
    GPU используется один; в обычном запуске одновременно в памяти только одна модель.
    ''')
    md('''
    ## 1. Параметры и продолжение

    Новый запуск начинает обучение из закреплённой базовой Laya, без неудачных весов V3.
    Максимум 12 эпох; остановка после четырёх эпох без улучшения, начиная с четвёртой эпохи.
    `ACTION_DRAWS=4096` — число входов с вопросом на эпоху: 2048 выбранных команд × два вопроса.
    Batch 2 означает одну такую пару; накопление 8 даёт 256 optimizer updates за полную эпоху.
    Обучаются голова и четыре последних encoder-слоя; act/escalate не обучается.

    `SMOKE_TEST=True` проверяет вычислительный путь, не качество и не экспортирует рабочие веса.
    `RESUME=True` восстанавливает optimizer/scaler/RNG после последней завершённой эпохи.
    `FINISH_ONLY=True` повторяет только финальные проверки завершённого обучения, без новых эпох.
    Для новой среды задайте `RESUME_FROM` сохранённой папкой с `candidates/` и `resume/`; настройки,
    точность, данные и версия ноутбука должны совпадать. Если Output удалён, веса не восстановить.
    ''')
    code('''
    from pathlib import Path
    import os, sys, json, hashlib, shutil, zipfile
    output_root = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content") if Path("/content").exists() else Path.cwd() / ".runtime/training"
    WORKDIR = output_root / "lia-action-v4"
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HOME", str(Path("/kaggle/temp/lia-v4-hf") if Path("/kaggle").exists() else WORKDIR.parent / "lia-v4-hf"))
    BASE_MODEL = "convaiinnovations/laya-multilingual"
    PINNED_BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    BASE_REVISION = PINNED_BASE_REVISION
    SEED = 20261003
    EPOCHS = 12
    ACTION_DRAWS = 4096
    BATCH_SIZE = 2
    ACCUMULATION = 8
    HEAD_LR = 2e-5
    ENCODER_LR = 1e-5
    LAST_ENCODER_LAYERS = 4
    CONSISTENCY_WEIGHT = .15
    REPLAY_FRACTION = .20
    PATIENCE = 4
    AMP_PRECISION = "auto"  # BF16 when supported, otherwise FP16; FP32 model weights
    MIN_RECALL = .80
    MIN_MACRO_F1 = .80
    SMOKE_TEST = False
    RESUME = True
    FINISH_ONLY = False
    RESUME_FROM = ""
    if RESUME_FROM:
        previous_output = Path(RESUME_FROM).resolve()
        if not all((previous_output / name).is_dir() for name in ("candidates", "resume")):
            raise FileNotFoundError("RESUME_FROM needs candidates/ AND resume/")
        if any((WORKDIR / name).exists() for name in ("candidates", "resume")):
            raise FileExistsError("Do not overwrite local state; clear RESUME_FROM to use existing WORKDIR")
        for name in ("candidates", "resume"): shutil.copytree(previous_output / name, WORKDIR / name)
    ''', "parameters")
    md('''
    ## 2. Встроенные данные и их происхождение

    Каждая новая фраза и её метка записаны в `lia_action_v4_authored.txt`; это декларации ассистента,
    без массового перемножения шаблонов и без предсказаний модели в качестве gold.
    SHA-256 проверяет архив и все файлы. Validation, calibration и test содержат только новые авторские семьи.
    Исторические семьи с пересечением тестовых входов исключены целиком; остальные — только train replay.
    Сценарии доступа, имена и выбор функции остаются правилами бота, не метками Action.
    ''')
    code(f'''import base64, io
EMBEDDED_DATASET_BASE64 = (
{chunks}
)
blob = base64.b64decode(EMBEDDED_DATASET_BASE64, validate=True)
assert hashlib.sha256(blob).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
DATA_DIR = WORKDIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    assert archive.testzip() is None
    for member in archive.infolist(): (DATA_DIR / member.filename).resolve().relative_to(DATA_DIR.resolve())
    archive.extractall(DATA_DIR)
DATA_MANIFEST = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
for name, digest in DATA_MANIFEST["file_sha256"].items():
    assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest, name
print({{k: DATA_MANIFEST[k] for k in ("new_literal_declarations", "authored_accepted", "rows", "all_bot_log_events", "parser_exclusions")}})
''', "embedded-data")
    cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
    md("## 3. Среда\n\nInternet нужен для pip и первого скачивания базы. CUDA PyTorch среды сохраняется. Веса остаются FP32; AMP меняет вычисления.")
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
    if DEVICE != "cuda" and not SMOKE_TEST: raise RuntimeError("Enable GPU before full training")
    if AMP_PRECISION not in ("auto", "fp16", "bf16"): raise ValueError("AMP_PRECISION: auto/fp16/bf16")
    use_bf16 = DEVICE == "cuda" and (AMP_PRECISION == "bf16" or AMP_PRECISION == "auto" and torch.cuda.is_bf16_supported())
    if use_bf16 and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported; use fp16")
    AMP_DTYPE = torch.bfloat16 if use_bf16 else torch.float16 if DEVICE == "cuda" else torch.float32
    assert importlib.metadata.version("laya") == "0.3.22"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    print({"device": DEVICE, "GPU": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "AMP": str(AMP_DTYPE), "torch": torch.__version__, "python": platform.python_version()})
    ''', "environment")
    md('''
    ## 4. Точный контракт модели

    Восемь классов и два вопроса взяты из реального `voice_worker.py`. Текст проходит тот же TypeScript-парсер
    и нормализацию worker. Никаких полей player/phase/gold внутри входа модели: только каноническая команда.
    Поиск песен и режимы имеют класс unknown для Action; их обрабатывает код отдельно.
    ''')
    code("SNAPSHOT = json.loads(" + repr(json.dumps(snap, ensure_ascii=False)) + ")\n" + textwrap.dedent('''
    contract_dir = WORKDIR / "contract/scripts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    for key, name in (("helper", "laya_training.py"), ("previous_helper", "laya_training_v2.py"),
                      ("v3_helper", "laya_training_v3.py"), ("v4_helper", "laya_training_v4.py"),
                      ("worker", "voice_worker.py"), ("bridge", "bot_parser.mjs")):
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
    v4 = load_module("laya_training_v4", contract_dir / "laya_training_v4.py")
    worker = load_module("lia_worker_v4", contract_dir / "voice_worker.py")
    bridge_path = contract_dir / "bot_parser.mjs"
    ''').lstrip(), "contract")
    md('''
    ## 5. Проверка выборок

    Общие контрастные семьи, исходные тексты и входы после обоих нормализаторов не пересекаются между выборками.
    Размеры классов — уникальные входы, а не размноженные варианты пунктуации. Семьи V4 — соседние формулировки,
    а не новые люди; test не доказывает качество на реальной речи. Исторические метки не меняются по ответам модели.
    ''')
    code('''
    commands = v4.load_commands(DATA_DIR / "commands.jsonl", bridge_path)
    by_id = {r["event_id"]: r for r in commands}
    ids = json.loads((DATA_DIR / "split-manifest.json").read_text(encoding="utf-8"))
    assert set(by_id) == {i for part in ids.values() for i in part}
    assert sum(map(len, ids.values())) == len(by_id)
    action_splits = {name: [by_id[i] for i in ids[name]] for name in training.SPLITS}
    locations, targets, split_summary = {}, {}, {}
    for name, rows in action_splits.items():
        for row in rows:
            assert row["declared_split"] == name
            for key in [("family", row["group_id"]), ("raw", training.normalized(row["message"])),
                        ("input", v3.runtime_text(row["canonical"]))]:
                assert key not in locations or locations[key] == name, "Cross-fold leakage"
                locations[key] = name
            text = v3.runtime_text(row["canonical"])
            assert text not in targets or targets[text] == row["intent"], "Contradictory model inputs"
            targets[text] = row["intent"]
            if name != "train": assert row["source_role"] == "authored" and row["origin"] == "synthetic"
        assert {r["intent"] for r in rows} == set(worker.QUESTIONS["action"]["criteria"])
        split_summary[name] = {"rows": len(rows), "unique_inputs_per_class": {
            label: len({v3.runtime_text(r["canonical"]) for r in rows if r["intent"] == label}) for label in sorted(v4.LABELS)}}
    assert sum(r["origin"] == "bot_log" for r in action_splits["train"]) == 31
    v3.atomic_json(WORKDIR / "diagnostics/split-summary.json", split_summary)
    print(json.dumps(split_summary, indent=2))
    if SMOKE_TEST:
        EPOCHS = 2; ACTION_DRAWS = 32
        for name, rows in action_splits.items():
            selected = {}
            for row in rows:
                if row["intent"] not in selected and row["source_role"] == "authored": selected[row["intent"]] = row
            action_splits[name] = list(selected.values())
    action_records = {name: training.runtime_records(rows, worker) for name, rows in action_splits.items()}
    ''', "data-loading")
    md('''
    ## 6. Обучение, допуск и сохранение

    Cross entropy учит правильный класс на обоих вопросах; дополнительный Jensen–Shannon loss с весом 0,15
    уменьшает расхождение их распределений. Это проверяемая гипотеза, не гарантия улучшения.
    Классы балансируются, авторские примеры получают 80% выборки, исторический replay — максимум 20%.
    Внутри unknown балансируются запреты, вопросы, несколько действий, числа, песни, режимы и прочие случаи.

    Эпоха выбирается только по validation: доля пройденных проверок recall и macro-F1 ≥ 0,80, худшая recall, худший macro-F1 вопроса,
    средний macro-F1 и NLL. Это не позволяет хорошей средней метрике скрыть провал pause.
    Температура калибруется на отдельной calibration, общей для двух вопросов согласно контракту worker.
    Test не используется для выбора эпохи, температуры, loss или порогов.

    Допуск: macro-F1 каждого вопроса ≥ 0,80, recall каждого класса ≥ 0,80 на validation и test;
    фактический worker + парсер: model-required accuracy ≥ 0,85, recall управлений ≥ 0,80,
    false control rate ≤ 0,02. Реальные шаги optimizer и пропуски AMP записываются отдельно.
    Проверяется равенство решений пакетной оценки и настоящего worker, затем повторная загрузка checkpoint.
    ZIP получает CANDIDATE при любом провале; smoke не экспортирует веса для подключения.

    Состояние optimizer/scaler/RNG сохраняется после каждой завершённой эпохи, лучший checkpoint — отдельно.
    Веса и входы находятся на одном устройстве; перед повторной загрузкой видеопамять освобождается.
    Validation, calibration и test рассчитываются по одному вопросу за вызов, как в production;
    confidence округляется до четырёх знаков перед порогами worker. Это особенно важно для BF16.
    Подход к AMP и восстановлению сверяется с [PyTorch AMP](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
    и [сохранением состояния](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
    ''')
    code((ROOT / "scripts/kaggle_v4_workflow.py").read_text(encoding="utf-8"), "stage-functions")
    code('report = run_action()', "train-action")
    md("## 7. Графики\n\nСравнивайте оба вопроса и каждый класс. Графики полного обучения отличаются от технического smoke; подписи показывают режим и размеры test.")
    code('''
    history = report["training"]["history"]
    label_prefix = "SMOKE: " if report["smoke_only"] else ""
    def display_figure(fig, filename):
        plot_dir = WORKDIR / "diagnostics/plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / filename, dpi=140, bbox_inches="tight")
        plt.show(); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_cross_entropy"] for r in history], "s-", color="#245b91", label="Cross entropy")
    axes[0].plot(epochs, [r["prompt_js_divergence"] for r in history], "o-", color="#b24d32", label="Расхождение вопросов JS")
    axes[0].set(xlabel="Эпоха", ylabel="Loss", title=label_prefix + "Обучение"); axes[0].legend()
    axes[1].plot(epochs, [r["validation_score"][2] for r in history], "s-", color="#245b91", label="Худший macro-F1")
    axes[1].plot(epochs, [r["validation_score"][1] for r in history], "o-", color="#b24d32", label="Худшая recall")
    axes[1].axhline(MIN_RECALL, ls="--", color="#555555", label="Порог")
    axes[1].set(xlabel="Эпоха", ylabel="Доля", ylim=(0, 1.04), title=label_prefix + "Validation"); axes[1].legend()
    for ax in axes: ax.set_xticks(epochs)
    fig.tight_layout(); display_figure(fig, "action-history.png")
    for task, values in report["metrics"]["test"].items():
        labels = list(values["per_class"])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        bars = ax.barh(labels, [values["per_class"][label]["recall"] for label in labels],
            color=["#245b91" if values["per_class"][label]["recall"] >= MIN_RECALL else "#b24d32" for label in labels])
        ax.axvline(MIN_RECALL, ls="--", color="#555555")
        for bar, label in zip(bars, labels):
            ax.text(bar.get_width() + .02, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.0%}; n={values["per_class"][label]["support"]}', va="center")
        ax.set(xlabel="Recall на авторском синтетическом test", xlim=(0, 1.2),
            title=label_prefix + task + f'; macro-F1={values["macro_f1"]:.3f}')
        fig.tight_layout(); display_figure(fig, task.replace("/", "-") + "-recall.png")
    predicted = json.loads((WORKDIR / "diagnostics/test-predictions.json").read_text(encoding="utf-8"))
    labels = sorted(v4.LABELS)
    matrix_vmax = max(1, max(c["support"] for m in report["metrics"]["test"].values() for c in m["per_class"].values()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ("action_primary", "action_short")):
        matrix = np.zeros((8, 8), dtype=int)
        for p in predicted:
            if p["task"] == task: matrix[labels.index(p["gold"]), labels.index(p["prediction"])] += 1
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix_vmax)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > matrix_vmax/2 else "#222222")
        ax.set_xticks(range(8), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(8), labels=labels)
        ax.set(xlabel="Предсказано", ylabel="Правильная метка", title=label_prefix + task + ": число примеров")
    fig.tight_layout(); display_figure(fig, "action-confusion.png")
    print("Selected epoch:", report["training"]["selected_epoch"], "updates:", report["training"]["optimizer_updates"],
          "skipped:", report["training"]["skipped_updates"])
    print("Gates:", {k: v["passed"] for k, v in report["gates"].items()})
    ''', "plots")
    md('''
    ## 8. Результаты

    Последняя ячейка работает и после потери переменных, если файлы сохранились.
    Скачайте модель ZIP и диагностику из Output Kaggle; для продолжения сохраните весь Output, включая `resume/`.
    В Colab можно скачать конкретный ZIP отдельной командой `from google.colab import files; files.download(str(path))`
    с существующим путём, показанным ниже. `CANDIDATE` не подключайте к боту.
    ''')
    code((ROOT / "scripts/kaggle_v4_finish.py").read_text(encoding="utf-8"), "diagnostics")
    md('''
    ## 9. Ограничения перед подключением

    Даже прошедший синтетические gates checkpoint требует проверки новых реальных расшифровок с ручными метками.
    Файл `parser-and-duplicate-exclusions.jsonl` показывает корректные просьбы, которые текущие правила не поддерживают:
    например, ряд фраз с «песня»/«трек» для pause/resume. Они исключены из обучения этого runtime-профиля и из его метрик.
    Не расширяйте правила подключения на основании этих метрик: синтетический допуск относится к поддерживаемому синтаксису.
    Слишком малая validation/test не гарантирует перенос на голос; размер каждого класса виден на графиках.
    Ноутбук не обучает Semantic/Whisper/TTS, не меняет сервер или `.env` и не подключает checkpoint автоматически.
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "language": "python", "sourceType": "notebook"}}, "cells": cells}
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
    print("cells", len(cells), "bytes", OUTPUT.stat().st_size)


if __name__ == "__main__": main()
