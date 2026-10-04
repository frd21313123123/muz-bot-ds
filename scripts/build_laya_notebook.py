"""Build a single-file notebook using snapshots of the actual bot contracts.

Run npm run build first. No notebook authoring packages are required.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "laya_training_colab_kaggle.ipynb"


def snapshot():
    # Only pure parsing functions; the resolver's Discord/YouTube imports are absent.
    intents = (ROOT / "dist/src/voice/intents.js").read_text(encoding="utf-8")
    music = (ROOT / "dist/src/voice/music.js").read_text(encoding="utf-8").split("export async function resolveMusicRequest")[0]
    parser = intents + "\n" + re.sub(r"^import .*;\n", "", music, flags=re.M)
    parser = re.sub(r"^export ", "", parser, flags=re.M)
    bridge = parser + r'''
import { readFileSync } from 'node:fs';
const rows = JSON.parse(readFileSync(0, 'utf8'));
const output = rows.map(row => {
  const canonical = contextualCommand(row.message, Boolean(row.player?.paused));
  const direct = modeCommand(canonical);
  const explicit = extractMusicRequest(canonical);
  const music = explicit ?? extractMusicRequest(canonical, true);
  const unsupported = !music && unsupportedSpeech(canonical);
  const model_required = !unsupported && !explicit && !direct;
  let final_action = direct?.action ?? 'unknown';
  if (unsupported) final_action = 'unknown';
  else if (!direct && !explicit && row.decision) {
    final_action = playerControlRequest(canonical) ? validateIntent(canonical, row.decision).action : 'unknown';
  }
  if (!unsupported && final_action === 'unknown' && music) final_action = 'play';
  return {canonical, direct_action: direct?.action ?? null, music_kind: music?.kind ?? null,
          unsupported, model_required, final_action,
          wake_match: typeof row.wake_name === 'string' ? isWakePhrase(row.message, row.wake_name) : null};
});
process.stdout.write(JSON.stringify(output));
'''
    training = (ROOT / "dist/src/voice/training.js").read_text(encoding="utf-8")
    questions_code = training.split("export const NLI_QUESTIONS = ")[1].split("\n// Separate")[0]
    questions_code = questions_code[:questions_code.rfind(";")]
    result = subprocess.run(["node", "--input-type=module", "-e", "process.stdout.write(JSON.stringify(" + questions_code + "))"],
                            text=True, encoding="utf-8", capture_output=True, check=True)
    paths = ["src/voice/intents.ts", "src/voice/music.ts", "src/voice/session.ts", "src/voice/GuildVoice.ts",
             "src/voice/training.ts", "src/voice/export-training.ts", "scripts/voice_worker.py",
             "scripts/laya_music_v3.json", "scripts/laya_training.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    return {"helper": (ROOT / "scripts/laya_training.py").read_text(encoding="utf-8"),
            "worker": (ROOT / "scripts/voice_worker.py").read_text(encoding="utf-8"),
            "music_schema": json.loads((ROOT / "scripts/laya_music_v3.json").read_text(encoding="utf-8")),
            "intent_questions": json.loads(result.stdout), "bridge": bridge, "source_sha256": hashes}


def build():
    snap = snapshot()
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}", "metadata": {},
                      "source": textwrap.dedent(source).strip() + "\n"})
    def code(source, tag=None):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "id": f"cell-{len(cells):02d}",
                      "metadata": {"tags": [tag]} if tag else {}, "execution_count": None, "outputs": [], "source": source})

    md('''
    # LIA / Laya: обучение для muz-bot-ds в Google Colab и Kaggle

    **Цель:** дообучить `convaiinnovations/laya-multilingual` на вручную размеченных
    русских командах, проверить ошибки и сохранить checkpoint, который загружается через `LAYA_MODEL_PATH`.
    Это классификатор вариантов ответа, а не генеративная модель. Whisper и TTS здесь не обучаются.
    Результатов обучения пока нет: все выходы появятся после запуска на ваших данных.

    Ноутбук самодостаточен: копии функций обработки текста, вопросов, Python-обработчика и
    обучающих утилит встроены ниже. Клонирование репозитория и токен Hugging Face не нужны.
    Контракт зафиксирован по исходникам; их SHA-256 сохраняются в отчёт.

    **Запуск:** Colab → загрузить этот `.ipynb` → выбрать GPU → загрузить JSONL;
    Kaggle → Import Notebook → включить GPU и Internet → добавить приватный Dataset с JSONL.
    Один GPU достаточен. Второй T4 автоматически не используется. Начните с замороженного encoder.
    При нехватке памяти уменьшите `BATCH_SIZE` до 1. Интернет нужен для pip и базового checkpoint.
    ''')
    md('''
    ## 1. Что реально делает бот

    | Решение | Текущая реализация | Что обучается |
    |---|---|---|
    | Реагировать на обращение | `VoiceSession`, имя целиком, тот же канал, владелец окна, таймеры и отмена | Правила остаются; отдельная экспериментальная схема ниже |
    | Команда управления | Laya `action`: skip, pause, resume, stop, volume_set/up/down, unknown | Основная задача этого ноутбука |
    | Автоплей, повтор, очередь, голос | `modeCommand` в TypeScript | Не добавляем неподдерживаемые классы в runtime `action` |
    | Название, исполнитель, ссылка | `extractMusicRequest` → YouTube → первый корректный результат | Не требуется знать все названия песен |
    | Громкость | Число разбирается кодом, 1–150%; относительное изменение ±10 | Laya выбирает действие, не извлекает число |
    | Голосовой ответ | Готовые подтверждения `confirmations.json`, с учётом успешности действия | Модель не сочиняет текст ответа |

    `music_route`, `music_policy` и `music_rerank` в `voice_worker.py` и схема
    `laya-muz-bot-ds-v3` — сохранённый старый контракт. Сейчас `resolveMusicRequest` их не вызывает.
    `/play`, slash-команды и кнопки не должны превращаться в классы голосовой модели.
    Обучение новых голов само по себе не меняет поведение бота.
    ''')
    md('''
    ## 2. Данные и ручная разметка

    На компьютере с ботом включите `VOICE_NLI_LOG=1`, соберите команды **после имени и сигнала**.
    Журнал `.runtime/nli/commands-*.jsonl` содержит подсказки модели, но `gold=null`.
    Он не содержит фоновых разговоров и обращений до сигнала: по нему нельзя обучить решение «реагировать».
    Не считайте `suggested_intent`, `model_decision`, `outcome=changed` правильной разметкой.

    Создайте `reviewed.jsonl`, по одной проверенной записи:
    ```json
    {"event_id":"ID из журнала","intent":"pause"}
    ```
    Экспортируйте **в новый файл**, затем загрузите его в Colab/Kaggle:
    ```powershell
    npm run export:nli -- reviewed.jsonl reviewed-nli.jsonl
    ```
    Либо загрузите исходные журналы и `reviewed.jsonl` и укажите `REVIEWS_PATH`.
    Есть и прямой формат `commands-reviewed.jsonl`:
    ```json
    {"event_id":"e1","message":"поставь на паузу","intent":"pause","reviewed":true,"group_id":"pause-family-01","player":{"connected":true,"playing":true,"paused":false,"autoplay":false,"queue_length":1}}
    ```
    `group_id` объединяет варианты одной реплики, запись одного сеанса/говорящего или шаблон генерации.
    Ставьте общий ID для родственных примеров. Точные дубликаты исходного и приведённого текста
    объединяются автоматически. Если экспорт не содержит группы, можно добавить её вручную.
    Без такой группировки метрики могут быть завышены.

    Для стартового корпуса `lia-starter-v1` задайте `ALLOW_AUTOMATIC_LABELS=True`.
    При этом дополнительно загружаются `commands-synthetic.jsonl` и `bot-commands-labeled.jsonl`.
    Их метки помечены `reviewed=false`, `rule_verified` и происхождением, а не выдаются за ручной gold.
    Непроверенные исходные логи и очередь ручной проверки автоматически в обучение не попадают.
    Синтетический test и автоматическая разметка не измеряют качество на реальной речи пользователей.

    Размечайте **запрос**, а не успешность операции: pause остаётся pause даже если уже пауза;
    no_op, сбой сети и отмена не означают unknown. Испорченный STT нельзя исправлять подсказкой в state:
    неоднозначную расшифровку исключите или разметьте unknown. Сохраняйте опечатки STT, если смысл понятен.
    «Громче на 50», отрицания, несколько команд, чужие устройства — важные отрицательные примеры.
    `play` — поиск названия/исполнителя; на уровне runtime action он становится unknown.
    Режимы autoplay/loop/voice и queue_clear обрабатываются кодом; в runtime action это тоже unknown.
    Для contextual «включи музыку» нужен реальный `player.paused`; без контекста такие строки исключите.

    Нужны разнообразные независимые семейства каждого класса, включая unknown и музыкальные запросы.
    Минимум 4 семьи на размеченный класс — лишь технический минимум для четырёх выборок,
    а не достаточный объём для качественного обучения. Начальный ориентир: сотни проверенных
    примеров на основной класс, много разных негативов и отдельные реальные записи для test.
    Журналы содержат речь участников: загружайте только согласованные, очищенные данные в приватный Dataset.
    ''')
    md('''
    ## 3. Параметры

    `DATA_PATHS` — список файлов. При пустом списке ищутся только `reviewed-nli.jsonl` и
    `commands-reviewed.jsonl` в `/content/data`, `/content`, `/kaggle/input`.
    Для Colab можно выполнить отдельную ячейку `from google.colab import files; files.upload()`.
    `SMOKE_TEST=True` только проверяет код на крошечном искусственном наборе и помечает экспорт **не для бота**.
    Используйте обычный режим с вашими метками для настоящего обучения.
    ''')
    code('''
    from pathlib import Path
    import os
    DATA_PATHS = []
    REVIEWS_PATH = None
    PLANNER_PATHS = []  # Опциональные вручную размеченные контекстные сценарии, раздел 7.
    ALLOW_AUTOMATIC_LABELS = False  # Явное разрешение rule_verified: синтетика / автоматическая разметка логов.
    AUTO_LOAD_PLANNER = False  # Искать planner-synthetic.jsonl рядом с выбранными файлами команд.
    BASE_MODEL = "convaiinnovations/laya-multilingual"
    BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"  # Подготовленная версия этого бота.
    SEED = 42
    EPOCHS = 4
    BATCH_SIZE = 2
    ACCUMULATION = 8
    HEAD_LR = 2e-5
    ENCODER_LR = 2e-6
    TRAIN_ENCODER = False
    SMOKE_TEST = False
    WORKDIR = Path("/kaggle/working/lia-training" if Path("/kaggle/working").exists() else "/content/lia-training")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HOME", str(WORKDIR / "hf-cache"))
    ''', "parameters")
    md('''
    ## 4. Зависимости

    Версия Laya соответствует `scripts/voice-requirements.txt`. PyTorch платформы сохраняется:
    не устанавливаем CPU wheel поверх CUDA. После установки, если среда просит, перезапустите kernel.
    Внутренние функции Laya используются с фиксированной версией; при обновлении нужна повторная проверка.
    Метод обучения: supervised cross entropy по логитам вариантов. Это осознанно более простой
    метод для ручных одношаговых меток, **не RLCD/GRPO**. Температуры подбираются отдельно.
    ''')
    code('''
    import subprocess, sys, shutil
    subprocess.run([sys.executable, "-m", "pip", "install", "laya==0.3.22", "transformers==5.17.0",
                    "huggingface-hub==1.33.0", "matplotlib>=3.9,<4"], check=True)
    if shutil.which("node") is None:
        if not shutil.which("apt-get"):
            raise RuntimeError("Install Node.js to run the exact TypeScript parser")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "nodejs"], check=True)
    ''', "install")
    code('''
    import importlib.util, importlib.metadata, json, random, platform, shutil
    import numpy as np
    import torch
    import laya
    assert importlib.metadata.version("laya") == "0.3.22"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu" and not SMOKE_TEST:
        raise RuntimeError("Select GPU in Colab/Kaggle before real training")
    print({"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "torch": torch.__version__, "python": platform.python_version()})
    ''')
    md('''
    ## 5. Встроенный контракт бота

    Следующая ячейка разворачивает снимок исходников. Расшифровки проходят **те же** TypeScript
    `contextualCommand`, `modeCommand`, `extractMusicRequest`, `unsupportedSpeech`, `validateIntent`.
    Мы не обучаем модель на JSON с полем intent, когда в production ей приходит строка и вопрос action.
    `canonical_message` из журнала игнорируется и вычисляется заново из исходной речи и player.
    ''')
    code("SNAPSHOT = json.loads(" + repr(json.dumps(snap, ensure_ascii=False)) + ")\n" + textwrap.dedent('''
    scripts_dir = WORKDIR / "contract" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    helper_path = scripts_dir / "laya_training.py"
    worker_path = scripts_dir / "voice_worker.py"
    bridge_path = scripts_dir / "bot_parser.mjs"
    helper_path.write_text(SNAPSHOT["helper"], encoding="utf-8")
    worker_path.write_text(SNAPSHOT["worker"], encoding="utf-8")
    bridge_path.write_text(SNAPSHOT["bridge"], encoding="utf-8")
    (scripts_dir / "laya_music_v3.json").write_text(json.dumps(SNAPSHOT["music_schema"], ensure_ascii=False), encoding="utf-8")
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    training = load_module("lia_training", helper_path)
    worker = load_module("lia_worker_contract", worker_path)  # Не загружает Whisper и TTS.
    INTENT_QUESTIONS = SNAPSHOT["intent_questions"]
    PLANNER_QUESTIONS = training.planner_questions(INTENT_QUESTIONS)
    print("Runtime action:", list(worker.QUESTIONS["action"]["criteria"]))
    print("Reviewed intent:", list(INTENT_QUESTIONS["intent"]["criteria"]))
    '''))
    md('''
    ## 6. Загрузка и разделение без утечки

    Четыре независимые части: train 60%, validation 15% для выбора эпохи,
    calibration 10% для температуры и test 15% для окончательной оценки.
    Доли приблизительные: группы неделимы. Варианты одной фразы и оба prompt остаются вместе.
    Корпуса runtime и planner группируются совместно, включая одинаковые исходные и приведённые тексты.
    Если классов/семейств недостаточно, выполнение остановится с объяснением.
    Test не участвует в выборе эпохи, температуры или порогов.
    ''')
    code('''
    if SMOKE_TEST:
        # Искусственные данные только для технической проверки, никаких заявлений о качестве.
        action_texts = {"skip": "пропусти", "pause": "поставь на паузу", "resume": "продолжи музыку",
                        "stop": "останови музыку", "volume_set": "громкость 70",
                        "volume_up": "сделай громче", "volume_down": "сделай тише",
                        "unknown": "не выключай музыку", "play": "включи песню Numb"}
        smoke_rows = []
        for label, phrase in action_texts.items():
            for family in range(12):
                smoke_rows.append({"event_id": f"smoke-{label}-{family}", "message": f"{phrase} номер {family}",
                                   "intent": label, "reviewed": True, "group_id": f"{label}-{family}"})
        smoke_path = WORKDIR / "commands-reviewed.jsonl"
        smoke_path.write_text("\\n".join(json.dumps(x, ensure_ascii=False) for x in smoke_rows) + "\\n", encoding="utf-8")
        DATA_PATHS = [str(smoke_path)]
        EPOCHS, BATCH_SIZE, ACCUMULATION = 1, 2, 2
    if not DATA_PATHS:
        found = set()
        for root in (Path("/content/data"), Path("/content"), Path("/kaggle/input")):
            if root.exists():
                names = ["reviewed-nli.jsonl", "commands-reviewed.jsonl"]
                if ALLOW_AUTOMATIC_LABELS:
                    names += ["commands-synthetic.jsonl", "bot-commands-labeled.jsonl"]
                for name in names:
                    found.update(root.rglob(name))
        DATA_PATHS = sorted(str(p) for p in found if WORKDIR not in p.parents)
    if not DATA_PATHS:
        raise FileNotFoundError("Upload reviewed-nli.jsonl/commands-reviewed.jsonl, or fill DATA_PATHS")
    commands = training.load_commands(DATA_PATHS, REVIEWS_PATH, INTENT_QUESTIONS["intent"]["criteria"], ALLOW_AUTOMATIC_LABELS)
    commands = training.prepare_commands(commands, bridge_path)
    runtime_labels = set(worker.QUESTIONS["action"]["criteria"])
    present_labels = {r["intent"] if r["intent"] in runtime_labels else "unknown" for r in commands}
    if runtime_labels - present_labels:
        raise ValueError(f"Missing runtime action classes: {sorted(runtime_labels - present_labels)}; add reviewed examples")
    if AUTO_LOAD_PLANNER and not PLANNER_PATHS:
        PLANNER_PATHS = sorted({str(Path(path).parent / "planner-synthetic.jsonl") for path in DATA_PATHS
                                if (Path(path).parent / "planner-synthetic.jsonl").is_file()})
        if not PLANNER_PATHS:
            raise FileNotFoundError("AUTO_LOAD_PLANNER=True but no planner-synthetic.jsonl beside command files")
    planner_rows = training.load_planner(PLANNER_PATHS, PLANNER_QUESTIONS, bridge_path, ALLOW_AUTOMATIC_LABELS) if PLANNER_PATHS else []
    if planner_rows:
        processed = training.node_bridge([{"message": r["message"], "player": r["state"]["player"]} for r in planner_rows], bridge_path)
        for row, parsed in zip(planner_rows, processed, strict=True):
            row["canonical"] = parsed["canonical"]
    all_rows = commands + planner_rows
    if len({r["event_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate event_id across runtime/planner corpora")
    # Joint grouping prevents leakage when a phrase appears in both tasks or several contexts.
    all_splits = training.grouped_split(all_rows, SEED, fractions=(0.4, 0.2, 0.2, 0.2) if SMOKE_TEST else (0.6, 0.15, 0.1, 0.15))
    command_splits = {name: [r for r in rows if r.get("task") != "planner"] for name, rows in all_splits.items()}
    planner_splits = {name: [r for r in rows if r.get("task") == "planner"] for name, rows in all_splits.items()} if planner_rows else None
    record_splits = {name: training.runtime_records(rows, worker) for name, rows in command_splits.items()}
    # Сохраняем ID/группы, не тексты участников, в отчёт экспорта.
    split_manifest = {name: [{"event_id": r["event_id"], "group_id": r["group_id"], "intent": r["intent"]} for r in rows]
                      for name, rows in command_splits.items()}
    for name, rows in command_splits.items():
        print(name, len(rows), dict(training.Counter(r["intent"] for r in rows)),
              "model_required:", sum(r["model_required"] for r in rows))
    print("Data origins:", dict(training.Counter(r["origin"] for r in all_rows)))
    if ALLOW_AUTOMATIC_LABELS:
        print("Automatic labels enabled: synthetic/rule-labelled data is not human-reviewed ground truth")
    ''')
    md('''
    ## 7. Дополнительно: реагировать и выбирать функцию

    `PLANNER_PATHS=[]` по умолчанию. Эти вопросы **не вызываются текущим ботом**.
    Они позволяют обучать тот же checkpoint для будущего расширения без изменения runtime action.
    Нужен отдельный вручную размеченный корпус, включая фон, неверное имя, чужого говорящего,
    выключенный голос, неподдерживаемую просьбу после активации и нормальные команды.
    Экспорт журнала команд недостаточен: в нём нет idle/owner/voice_enabled.

    Пример записи (не выдаётся за обучающий корпус):
    ```json
    {"event_id":"planner-1","group_id":"pause-context-family","reviewed":true,"state":{"message":"поставь на паузу","phase":"awaiting","voice_enabled":true,"in_bot_channel":true,"is_owner":true,"wake_name":"Муза","player":{"connected":true,"playing":true,"paused":false,"autoplay":false,"queue_length":1}},"gold":{"should_respond":"respond","next_tool":"player_control","intent":"pause"}}
    ```
    `should_respond`: respond/ignore. `next_tool`: wake_ack/youtube_music_search/direct_youtube_video/player_control/no_tool.
    `intent`: те же 16 меток ручной разметки. В idle допустимо отдельное имя → respond/wake_ack/unknown;
    фон, чужой канал, disabled и busy → ignore/no_tool/unknown; неподдерживаемая реплика владельца
    в awaiting → respond/no_tool/unknown (текущее голосовое подтверждение unknown).
    В awaiting запрос песни → respond/search или direct_video/play; режим → respond/player_control/его intent.
    Плейлист как отдельный инструмент не поддерживается голосовым путём; selected/current_track не используются.

    `group_id` объединяет контекстные варианты одной реплики. Для каждой комбинации gold нужен набор
    независимых семей; иначе проверка покрытия остановит обучение. Корпус planner разделяется до обучения.
    Даже после подключения модели проверки канала, доступа, владельца, таймеров и отмены остаются в коде.
    ''')
    code('''
    if PLANNER_PATHS:
        for name, rows in planner_splits.items():
            record_splits[name].extend(rows)
        split_manifest["planner"] = {name: [{"event_id": r["event_id"], "group_id": r["group_id"]} for r in rows]
                                     for name, rows in planner_splits.items()}
    print("Planner training:", bool(PLANNER_PATHS))
    ''')
    md('''
    ## 8. Модель и токенизация

    Загружается существующая архитектура Laya со scorer по маркерам вариантов, не
    `AutoModelForSequenceClassification` с несовместимыми весами. Act/escalate-head отдельно не обучается:
    это не поле `action` музыкального бота. Сокращённые и основные вопросы берутся из обработчика.
    Температуры базы обнуляются до 1 для честного сравнения некалиброванных логитов.
    ''')
    code('''
    agent = laya.load(BASE_MODEL, device=DEVICE, revision=BASE_REVISION)
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    # T4 не поддерживает BF16; RTX/A100 можно использовать BF16.
    agent.dtype = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    agent.cfg["amp_dtype"] = "bf16" if agent.dtype == torch.bfloat16 else "fp16"
    agent.cfg["temperature"] = [1.0, 1.0, 1.0]
    agent.cfg["temperature_by_options"] = {}
    training.batches.pad_id = agent.tok.pad_token_id
    encoded = {name: training.encode_records(agent, records) for name, records in record_splits.items()}
    print({name: {"questions": len(items), "max_tokens": max(len(x["ids"]) for x in items)} for name, items in encoded.items()})
    ''')
    md('''
    ## 9. Baseline и обучение

    Baseline считается на validation. Лучший checkpoint выбирается по validation NLL;
    test остаётся нетронутым до финальной оценки. Encoder по умолчанию заморожен, его dropout выключен.
    Если реальных данных достаточно и validation перестал улучшаться, отдельным экспериментом
    включите TRAIN_ENCODER и сравните на validation. Не выбирайте конфигурацию по test.
    ''')
    code('''
    baseline_validation, _ = training.evaluate_logits(training.raw_logits(agent, encoded["validation"], BATCH_SIZE))
    print("Baseline validation:", json.dumps(baseline_validation, ensure_ascii=False, indent=2))
    history = training.train(agent, encoded["train"], encoded["validation"], WORKDIR / "best",
                             epochs=EPOCHS, batch_size=BATCH_SIZE, accumulation=ACCUMULATION,
                             head_lr=HEAD_LR, encoder_lr=ENCODER_LR, train_encoder=TRAIN_ENCODER, seed=SEED)
    ''')
    md('''
    ## 10. Калибровка уверенности

    Используется только calibration. Для каждого события fallback-вопрос не удваивает объём.
    Laya 0.3.22 требует минимум 10 независимых записей для общей температуры типа и 2000 для
    отдельного bucket. При меньшем числе сохраняется 1; это не доказательство калиброванности.
    Confidence в боте — `answer_confidence=max(probabilities)`, не entropy-confidence и не act_probability.
    Фиксированные runtime пороги сохраняются: 0.6, для слабого fallback 0.75.
    Порог нельзя подбирать на test; исследовать его следует в отдельном эксперименте на validation.
    ''')
    code('''
    calibration_rows = training.raw_logits(agent, encoded["calibration"], BATCH_SIZE)
    temperatures = training.calibrate(calibration_rows)
    print("Independent calibration events:", len({x["item"]["event_id"] for x in calibration_rows}))
    print("Temperatures:", temperatures)
    agent.temperature = temperatures["temperature"]
    agent.temperature_by_options = temperatures["temperature_by_options"]
    test_logits = training.raw_logits(agent, encoded["test"], BATCH_SIZE)
    uncalibrated_test, _ = training.evaluate_logits(test_logits)
    calibrated_test, test_predictions = training.evaluate_logits(test_logits, temperatures)
    pipeline_metrics, pipeline_rows = training.runtime_evaluation(agent, command_splits["test"], worker, bridge_path)
    print("Test:", json.dumps(calibrated_test, ensure_ascii=False, indent=2))
    print("Bot text pipeline:", pipeline_metrics)
    ''')
    md('''
    ## 11. Проверка качества

    Macro-F1, precision/recall по классам, NLL, Brier и ECE относятся к выбору вариантов.
    Отдельно смотрите `model_required_accuracy`: общий pipeline включает команды, выполненные правилами,
    и может скрывать слабую модель. False-control-rate — доля play/unknown, ставших управляющей командой.
    Не путайте отказ классификатора с необходимостью молчать: после активации unknown может озвучиваться.
    Маленький test не подтверждает надёжность; просмотрите ошибки и проверьте реальные записи отдельно.
    ''')
    code('''
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (10, 5), "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots()
    ax.plot([x["epoch"] for x in history], [x["train_nll"] for x in history], "o-", label="Train NLL", color="#245b91")
    ax.plot([x["epoch"] for x in history], [x["validation_nll"] for x in history], "s--", label="Validation NLL", color="#b47b24")
    ax.set(xlabel="Эпоха", ylabel="NLL (меньше лучше)", title="Обучение: train и независимая validation")
    ax.set_xticks([x["epoch"] for x in history])
    ax.legend(); fig.tight_layout(); plt.show()
    primary = [r for r in test_predictions if r["task"] == "action_primary"]
    labels = list(worker.QUESTIONS["action"]["criteria"])
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in primary:
        matrix[labels.index(row["gold"]), labels.index(row["prediction"])] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Предсказание", ylabel="Ручная метка", title=f"Primary action на test: {len(primary)} команд")
    fig.colorbar(plot, ax=ax, label="Количество команд"); fig.tight_layout(); plt.show()
    fig, ax = plt.subplots()
    means, accuracies, sizes = [], [], []
    for lo in np.arange(0, 1, 0.1):
        selected = [r for r in primary if lo <= r["confidence"] < lo + 0.1 or lo >= 0.9 and r["confidence"] == 1]
        if selected:
            means.append(np.mean([r["confidence"] for r in selected])); accuracies.append(np.mean([r["correct"] for r in selected])); sizes.append(len(selected))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="Идеальная калибровка")
    ax.plot(means, accuracies, "o-", color="#245b91", label="Наблюдаемая точность")
    for x, y, n in zip(means, accuracies, sizes): ax.annotate(f"n={n}", (x, y), xytext=(4, 7), textcoords="offset points")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.08), xlabel="Средняя уверенность", ylabel="Доля верных ответов",
           title="Калибровка primary action на независимом test (bin=0.1)")
    ax.legend(); fig.tight_layout(); plt.show()
    print("Ошибки pipeline (ID без текста):", [r for r in pipeline_rows if r["gold"] != r["prediction"]][:30])
    ''', "plots")
    md('''
    ## 12. Экспорт и повторная загрузка

    Экспорт содержит только веса, конфигурации, tokenizer и отчёт с ID/группами/метриками.
    Исходные реплики участников не включаются. Проверяем совпадение ответов после загрузки с диска.
    Архив не публикуется на Hugging Face автоматически.
    ''')
    code('''
    # Ячейку можно повторить после частичного экспорта: agent мог остаться на CPU.
    if "reloaded" in globals():
        del reloaded
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    agent.device = torch.device(DEVICE)
    agent.model.to(device=agent.device, dtype=agent.dtype).eval()
    model_name = "laya-muz-bot-controls-v1" + ("-planner" if PLANNER_PATHS else "")
    export_dir = WORKDIR / (model_name + ("-SMOKE-ONLY" if SMOKE_TEST else ""))
    training.save_checkpoint(agent, export_dir, model_name, temperatures)
    probe_texts = [r["canonical"] for r in command_splits["test"][:5]]
    before = [worker.classify(agent, text) for text in probe_texts]
    # Освобождаем видеопамять до второй загрузки (важно для T4/8 GB).
    agent.model.cpu()
    agent.device = torch.device("cpu")
    reloaded = laya.load(str(export_dir), device=DEVICE)
    after = [worker.classify(reloaded, text) for text in probe_texts]
    # GPU/FP16 может немного менять округлённую уверенность после загрузки.
    assert all(a["action"] == b["action"] and abs(a["confidence"] - b["confidence"]) <= 0.005
               for a, b in zip(before, after, strict=True)), "Checkpoint round-trip changed runtime decisions"
    if PLANNER_PATHS:
        example = planner_splits["test"][0]
        planner_answers = reloaded.predict(example["state"], PLANNER_QUESTIONS)["answers"]
        print("Experimental planner probe:", {k: v["choice"] for k, v in planner_answers.items()})
    input_hashes = {Path(p).name + f"#{i}": training.sha256_file(p) for i, p in enumerate(DATA_PATHS + PLANNER_PATHS)}
    if REVIEWS_PATH: input_hashes["reviews"] = training.sha256_file(REVIEWS_PATH)
    report = {"smoke_only": SMOKE_TEST, "runtime_contract": "action + primary/fallback + TS guards",
              "planner_enabled_in_bot": False, "planner_trained": bool(PLANNER_PATHS),
              "automatic_labels_allowed": ALLOW_AUTOMATIC_LABELS,
              "evaluation_warning": "Synthetic/rule-labelled holdouts are not independently annotated real-user accuracy" if ALLOW_AUTOMATIC_LABELS else None,
              "data_origins": dict(training.Counter(r["origin"] for r in all_rows)),
              "source_sha256": SNAPSHOT["source_sha256"], "input_sha256": input_hashes,
              "split_manifest": split_manifest, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "accumulation": ACCUMULATION,
              "head_lr": HEAD_LR, "encoder_lr": ENCODER_LR, "train_encoder": TRAIN_ENCODER,
              "history": history, "baseline_validation": baseline_validation,
              "uncalibrated_test": uncalibrated_test, "calibrated_test": calibrated_test,
              "pipeline_metrics": pipeline_metrics, "temperatures": temperatures,
              "environment": {k: importlib.metadata.version(k) for k in ("torch", "laya", "transformers", "huggingface-hub", "numpy")},
              "checkpoint_roundtrip": "passed"}
    (export_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_hashes = {str(p.relative_to(export_dir)): training.sha256_file(p) for p in export_dir.rglob("*") if p.is_file()}
    (export_dir / "sha256.json").write_text(json.dumps(checkpoint_hashes, indent=2), encoding="utf-8")
    archive_path = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir.parent, base_dir=export_dir.name)
    print("Archive:", archive_path)
    print("SMOKE ONLY — не подключать к боту" if SMOKE_TEST else "Проверьте метрики и ошибки перед подключением к боту")
    ''')
    md('''
    ## 13. Подключение

    В Colab скачайте архив через `from google.colab import files; files.download(archive_path)`.
    В Kaggle он находится в Output (`/kaggle/working/lia-training/`). Распакуйте обычный,
    проверенный checkpoint в `.runtime/voice/models/laya-muz-bot-controls-v1/`.
    В `.env` задайте:
    ```dotenv
    LAYA_MODEL_PATH=.runtime/voice/models/laya-muz-bot-controls-v1
    ```
    Перезапустите бота; проверьте имя модели в логе. Если voice runtime ещё не подготовлен,
    выполните `npm run setup:voice`. Для смены весов `npm run deploy` не требуется.
    Не переименовывайте `model_name` в `laya-muz-bot-ds-v3`: это активирует другой старый адаптер.
    Для отката уберите `LAYA_MODEL_PATH` и перезапустите бот с базовой моделью из подготовленного кэша.

    Перед использованием проверьте реальные команды, ложные обращения, уход говорящего, таймаут,
    паузу/продолжение, числа 1/150/0/151, отрицания, новые названия песен и неизвестные просьбы.
    Для новой planner-схемы нужна отдельная интеграция в VoiceSession/worker: сейчас она не вызывается.
    Модель не заменяет проверки канала, владельца и AbortSignal; уверенность не даёт права выполнить команду.

    Источники метода и API: [Laya upstream](https://github.com/NandhaKishorM/laya),
    [официальный fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
    Автор этого ноутбука использует supervised CE и отдельную калибровку; это не копия upstream RLCD.
    Схема и состояния проверены по локальному `laya==0.3.22` и исходникам muz-bot-ds.
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": OUTPUT.name}, "accelerator": "GPU"},
                "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT.name}: {len(cells)} cells")


if __name__ == "__main__":
    build()
