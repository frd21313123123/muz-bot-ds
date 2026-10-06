"""Build a self-contained Kaggle notebook; embedded modules need no repository checkout."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks' / 'wakeword_bot_kaggle.ipynb'


def build(output=OUTPUT):
    cells = []

    def cell(kind, source, tag=None, hidden=False):
        source = textwrap.dedent(source).strip() + '\n'
        if kind == 'code': ast.parse(source)
        item = {'cell_type': kind, 'id': f'wake-{len(cells):02d}', 'metadata': {}, 'source': source}
        if tag: item['metadata']['tags'] = [tag]
        if hidden: item['metadata']['jupyter'] = {'source_hidden': True}
        if kind == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''
    # «Бот» V4: несколько локальных TTS и многоязычный акустический encoder, две GPU

    **Результат:** компактный бинарный детектор, который получает звук и выдаёт
    `probability` / `wake`. Он не распознаёт команды и не возвращает текст.
    По умолчанию положительное обращение — отдельно произнесённое **«бот» / “bot”**.
    Для нового слова измените параметры ниже и обучите новую модель.

    В ноутбуке уже есть тексты, генератор нейросетевой речи, аугментации, модели,
    обучение, калибровка и экспорт. Внешний Dataset, исходники бота и API-ключи не нужны.
    Датасет создаётся **при выполнении**, готовые WAV-файлы не спрятаны внутри файла.

    В Kaggle выберите **Accelerator → GPU T4 ×2**, включите **Internet**, импортируйте
    этот `.ipynb` и запускайте ячейки сверху вниз / Save & Run All.
    Генерация запускает два независимых процесса TTS, обучение — два процесса DDP.
    Программа проверяет число GPU; одна карта не будет молча выдана за две.

    V4 использует Silero v4/v5.5, Piper, MMS, Kokoro, реальные отрицательные слова,
    многоязычный encoder Whisper Tiny без decoder, баланс русского языка, hard-example mining, EMA,
    два запуска с разными seed и выбор ансамбля только по validation.
    Максимальная точность не гарантируется количеством эпох или синтетических файлов.
    Test оценивает перенос на другие TTS-голоса,
    но не доказывает качество на реальных микрофонах, акцентах и музыке Discord.
    Модель пока не подключается к боту автоматически: будущая схема — локальный
    wake-detector → сигнал → Groq STT только для команды.
    ''')
    cell('markdown', '''
    ## 1. Параметры

    `STANDALONE_ONLY=True` соответствует текущему сценарию: имя → сигнал → команда.
    Упоминания вроде «этот бот» — отрицательные примеры. При `False` положительные
    примеры включают короткие фразы с обращением; это другая задача, нужен новый запуск.
    Один акустический детектор не определяет намерение говорящего по идентичному звуку.

    `USE_ENGLISH=True` добавляет английское произношение “bot” и голоса всех движков.
    При смене русского слова обновите английский вариант либо отключите английскую часть.
    Созвучные слова дополнительно настройте в `wake_data.py`, если выбрано другое имя.
    `SMOKE=True` — проверка конвейера на небольшом наборе; экспорт помечается SMOKE-ONLY.
    ''')
    cell('code', '''
    from pathlib import Path
    import json, os, sys, subprocess, hashlib, platform

    WAKE_WORD_RU = "бот"
    WAKE_WORD_EN = "bot"
    USE_ENGLISH = True
    STANDALONE_ONLY = True
    REQUIRE_TWO_GPUS = True
    SMOKE = False
    SEED = 20261005
    POSITIVE_PER_VOICE = 300
    NEGATIVE_PER_VOICE = 600
    EVAL_POSITIVE_PER_VOICE = 200
    EVAL_NEGATIVE_PER_VOICE = 800
    RUSSIAN_MULTIPLIER = 2
    TTS_ENGINES = ['silero', 'piper', 'mms', 'kokoro']
    SILERO_RU_VERSIONS = ['v5_5_ru', 'v4_ru']
    GENDER_BALANCED_RU = True
    POSITIVE_TTS_VARIANTS = 6
    REAL_SPEECH_SOURCE = 'full'  # 'mini' is smaller; full download is about 2.4 GB
    REAL_SPEECH_WINDOWS = 12000  # 0 disables real negative speech
    REAL_WAKE_MANIFEST = None  # Optional reviewed real recordings; schema is described below
    INDEPENDENT_PITCH = True
    NOISE_WINDOWS = 2400
    EPOCHS = 100  # maximum, early stopping may finish sooner
    PATIENCE = 18
    MIN_EPOCHS = 25
    ENSEMBLE_MEMBERS = 2
    ARCHITECTURE = 'whisper_encoder_v4'  # alternatives: 'temporal_v2', 'speech_embedding_v3', 'legacy'; new WORK_DIR required
    ENCODER_TRAIN_LAYERS = 2  # Whisper encoder only: 0..4, last layers, no speech decoder
    ENCODER_LEARNING_RATE = 0.00001
    BATCH_PER_GPU = 64
    LEARNING_RATE = 0.0007
    TARGET_FPR_PER_WINDOW = 0.005
    RESUME_CHECKPOINT = None  # ensemble: path to previous checkpoints/ containing member-0/ and member-1/
    INCLUDE_DATASET_IN_ZIP = False  # WAV + features remain in Kaggle Output either way

    # Use a new directory for a different word, split or synthesis recipe.
    WORK_DIR = Path('/kaggle/working/wakeword-bot-v4')
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not WAKE_WORD_RU.strip() or len(WAKE_WORD_RU.split()) != 1:
        raise ValueError('WAKE_WORD_RU must be one word')
    if USE_ENGLISH and (not WAKE_WORD_EN.strip() or len(WAKE_WORD_EN.split()) != 1):
        raise ValueError('Set one English word, or USE_ENGLISH=False')
    if not 0 <= TARGET_FPR_PER_WINDOW < 1:
        raise ValueError('TARGET_FPR_PER_WINDOW must be in [0, 1)')
    CONFIG = dict(work_dir=str(WORK_DIR), wake_word_ru=WAKE_WORD_RU, wake_word_en=WAKE_WORD_EN,
                  use_english=USE_ENGLISH, standalone_only=STANDALONE_ONLY, seed=SEED,
                  positive_per_voice=8 if SMOKE else POSITIVE_PER_VOICE,
                  negative_per_voice=12 if SMOKE else NEGATIVE_PER_VOICE,
                  noise_windows=16 if SMOKE else NOISE_WINDOWS, epochs=2 if SMOKE else EPOCHS,
                  batch_per_gpu=8 if SMOKE else BATCH_PER_GPU, learning_rate=LEARNING_RATE,
                  target_fpr=TARGET_FPR_PER_WINDOW, resume_checkpoint=RESUME_CHECKPOINT,
                  loader_workers=2, cpu_threads=2, mixed_precision=True, smoke=SMOKE)
    CONFIG.update(recipe_version=4, tts_engines=TTS_ENGINES, silero_ru_versions=SILERO_RU_VERSIONS,
                  gender_balanced_ru=GENDER_BALANCED_RU,
                  eval_positive_per_voice=EVAL_POSITIVE_PER_VOICE, eval_negative_per_voice=EVAL_NEGATIVE_PER_VOICE,
                  russian_multiplier=RUSSIAN_MULTIPLIER, positive_tts_variants=3 if SMOKE else POSITIVE_TTS_VARIANTS,
                  real_speech_source='mini' if SMOKE else REAL_SPEECH_SOURCE,
                  real_speech_windows=min(160, REAL_SPEECH_WINDOWS) if SMOKE else REAL_SPEECH_WINDOWS,
                  real_wake_manifest=REAL_WAKE_MANIFEST,
                  independent_pitch=INDEPENDENT_PITCH, architecture=ARCHITECTURE, balanced_sampling=True,
                  encoder_train_layers=ENCODER_TRAIN_LAYERS, encoder_learning_rate=ENCODER_LEARNING_RATE,
                  ema=True, preload_features=True, mining_interval=10, patience=PATIENCE, min_epochs=MIN_EPOCHS,
                  ensemble_members=1 if SMOKE else ENSEMBLE_MEMBERS)
    immutable = {key: CONFIG[key] for key in ['wake_word_ru', 'wake_word_en', 'use_english', 'standalone_only',
                 'seed', 'positive_per_voice', 'negative_per_voice', 'noise_windows', 'smoke', 'recipe_version',
                 'tts_engines', 'silero_ru_versions', 'gender_balanced_ru', 'eval_positive_per_voice', 'eval_negative_per_voice',
                 'russian_multiplier', 'positive_tts_variants', 'real_speech_source', 'real_speech_windows', 'real_wake_manifest', 'independent_pitch', 'architecture']}
    recipe = WORK_DIR / 'dataset_recipe.json'
    if recipe.exists() and json.loads(recipe.read_text(encoding='utf-8')) != immutable:
        raise ValueError('WORK_DIR already contains a different dataset recipe; choose a new directory')
    recipe.write_text(json.dumps(immutable, ensure_ascii=False, indent=2), encoding='utf-8')
    CONFIG_FILE = WORK_DIR / 'run_config.json'
    CONFIG_FILE.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
    os.environ['OPENBLAS_NUM_THREADS'] = str(CONFIG['cpu_threads'])
    os.environ['OMP_NUM_THREADS'] = str(CONFIG['cpu_threads'])
    CODE_DIR = WORK_DIR / 'code'
    CODE_DIR.mkdir(exist_ok=True)
    print(f'Word: {WAKE_WORD_RU}; English: {WAKE_WORD_EN if USE_ENGLISH else "disabled"}; smoke: {SMOKE}')
    ''', 'parameters')
    cell('markdown', '''
    ## 2. Зависимости и обе GPU

    Сохраняем установленную в Kaggle сборку PyTorch/CUDA: переустановка `torch`
    и несовместимых `torchaudio` не нужна. Устанавливаются только вспомогательные пакеты.
    FP16 включается на GPU, веса и экспорт остаются FP32.
    В файле `environment.json` фиксируются фактические версии пакетов и названия GPU.
    ''')
    cell('code', '''
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                    'numpy>=2.0.2,<3', 'scipy>=1.12,<2', 'matplotlib>=3.8,<4', 'onnx>=1.17,<2', 'onnxruntime>=1.20,<2',
                    'piper-tts==1.8.0', 'kokoro-onnx==0.4.9', 'transformers==4.57.6', 'uroman==1.3.1.1',
                    'librosa==0.11.0', 'huggingface_hub>=0.34,<1'], check=True)
    import torch
    GPU_COUNT = min(2, torch.cuda.device_count())
    if not torch.cuda.is_available() or GPU_COUNT == 0:
        raise RuntimeError('Enable GPU T4 ×2 in Kaggle Settings')
    if REQUIRE_TWO_GPUS and GPU_COUNT != 2:
        raise RuntimeError(f'Found {GPU_COUNT} GPU; select T4 ×2, or explicitly disable REQUIRE_TWO_GPUS')
    import importlib.metadata
    environment = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                       gpus=[torch.cuda.get_device_name(i) for i in range(GPU_COUNT)],
                       packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'matplotlib', 'onnx', 'onnxruntime',
                                                                                   'piper-tts', 'kokoro-onnx', 'transformers', 'librosa']})
    (WORK_DIR / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment, indent=2))
    ''', 'dependencies')
    cell('markdown', '''
    ## 3. Встроенный код

    Следующие восемь ячеек записывают полностью включённые в ноутбук модули в Output.
    Для обычного запуска их редактировать не нужно; содержимое можно раскрыть.
    Одна копия кода используется для обучения и экспорта frontend, чтобы избежать
    несовпадения признаков. Вход: mono 16 кГц, окно 2 секунды. `temporal_v2`
    использует log-mel 40 × 201, `speech_embedding_v3` — готовые акустические
    признаки 96 × 16 из закреплённого encoder. Его две ONNX-модели скачиваются
    автоматически и включаются в ZIP; ASR для детектора не нужен.
    `whisper_encoder_v4` использует 80 × 200 log-mel и многоязычный encoder
    Whisper Tiny с окном, сокращённым до двух секунд. Дообучаются последние
    два encoder-слоя и бинарная голова, с разными learning rate. Decoder отсутствует.
    ''')
    modules = ['wake_audio.py', 'wake_model.py', 'wake_data.py', 'wake_tts.py', 'wake_data_v2.py', 'wake_metrics.py', 'wake_train.py', 'wake_export.py']
    hashes = {}
    for filename in modules:
        source = (ROOT / 'scripts/wakeword' / filename).read_text(encoding='utf-8')
        ast.parse(source)
        hashes[filename] = hashlib.sha256(source.encode()).hexdigest()
        # Existing notebook scaffold uses readable embedded source rather than magic-cell syntax.
        literal = "r'''" + source + "'''" if "'''" not in source else repr(source)
        cell('code', f"(CODE_DIR / {filename!r}).write_text({literal}, encoding='utf-8', newline='\\n')\n", f'embedded-{filename}', hidden=True)
    cell('code', f'''
    SOURCE_SHA256 = {hashes!r}
    for name, expected in SOURCE_SHA256.items():
        assert hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest() == expected
    (WORK_DIR / 'source_sha256.json').write_text(json.dumps(SOURCE_SHA256, indent=2))
    sys.path.insert(0, str(CODE_DIR))
    import wake_data_v2 as wake_data, wake_audio
    print('Embedded modules ready; no Git clone required.')
    ''', 'source-check')
    cell('markdown', '''
    ## 4. Нейросетевая синтетика на двух GPU

    **Silero v4/v5.5/v3_en + Piper + MMS + Kokoro**: 66 наборов TTS/голос,
    61 группа голоса. Разные версии Silero с одним голосом остаются в одной части;
    версии и изменение pitch не выдаются за новых людей. Русский train включает
    2 голоса Silero, 3 Piper и MMS вместо прежних двух голосов одного синтезатора.
    Женский голос Piper находится в train: validation выявил плохой перенос,
    когда в train этого движка были только мужские голоса. Baya остаётся в validation.
    Все голоса назначаются в train/validation/calibration/test до аугментаций.
    Вспомогательный Speech Commands даёт реальные негативные слова и фоновые записи.
    По умолчанию полный архив занимает около 2.4 GB; для малого запуска есть `mini`.
    Дикторы реальных негативов тоже не пересекаются между частями.

    Положительные примеры меняют интонацию и темп TTS. Отрицательные включают
    «вот», «кот», «год», «борт», «порт», “boat”, “boot”, “but”, обычные фразы,
    команды и контекстные упоминания. Шум, реверберация, уровень сигнала,
    изменение темпа и независимый pitch shift применяются к обоим классам.
    Среди речевых негативов выделены 50% коротких слов, 25% упоминаний и 25% фраз.
    Есть чистые примеры, умеренный SNR, saturation/quantization и полоса 8 кГц.
    Реальные негативные слова не заменяют реальные положительные обращения и музыку.

    Два процесса делят голоса: Silero/MMS используют соответствующую GPU,
    Piper/Kokoro — CPU ONNX с ограничением потоков. Генерация не заявляет ускорение
    каждой модели на CUDA. При отсутствии выбранного TTS запуск завершается ошибкой.
    Скачивание весов производится заранее один раз. Промежуточные WAV сохранены,
    поэтому повторный запуск использует уже озвученные исходники.
    Если генерация прервалась, повторите эту ячейку: шард публикуется только целиком.

    Для дальнейшего улучшения есть `REAL_WAKE_MANIFEST`: JSONL с действительно
    проверенными записями. Строка: `{"wav":"call.wav","label":1,"language":"ru",
    "speaker":"person-1","split":"train","text":"бот","reviewed":true}`.
    Отрицательные примеры имеют label=0. WAV — mono PCM16, 16 кГц, полное высказывание
    короче двух секунд; относительные пути считаются от JSONL. Один человек должен
    принадлежать одной части. Не используйте текст, выданный ASR, как истинную метку.
    ''')
    cell('code', '''
    wake_data.download_models(WORK_DIR, CONFIG)
    wake_data.prepare_real(CONFIG)
    wake_data.prepare_human(CONFIG)
    import time
    processes = []
    try:
        for rank in range(GPU_COUNT):
            log_path = WORK_DIR / f'tts-gpu-{rank}.log'
            handle = log_path.open('w')
            process = subprocess.Popen([sys.executable, '-u', str(CODE_DIR / 'wake_data_v2.py'),
                                        '--config', str(CONFIG_FILE), '--rank', str(rank), '--world', str(GPU_COUNT)],
                                       stdout=handle, stderr=subprocess.STDOUT)
            processes.append((rank, process, handle, log_path))
        while any(process.poll() is None for _, process, _, _ in processes):
            for rank, process, _, log_path in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'TTS rank {rank} failed; inspect {log_path}')
            statuses = []
            for rank, process, _, log_path in processes:
                lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                statuses.append(f'GPU {rank}: {lines[-1] if lines else "loading TTS..."}')
            print(' | '.join(statuses), flush=True)
            time.sleep(30)
        for rank, process, _, log_path in processes:
            if process.returncode:
                raise RuntimeError(f'TTS rank {rank} failed; inspect {log_path}')
    finally:
        for _, process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            handle.close()
    rows, dataset_summary = wake_data.merge_manifest(CONFIG, GPU_COUNT)
    print(json.dumps(dataset_summary, indent=2))
    print('Total windows:', len(rows), '| Source groups:', len({row['source_id'] for row in rows}))
    ''', 'generate-data')
    cell('markdown', '''
    ## 5. Проверить примеры и разделение

    Все аугментации одного исходного аудио остаются в одной части. Совпадающее
    ключевое слово и созвучные негативы намеренно встречаются в разных частях,
    но их голоса и исходные записи различаются. Это speaker-held-out split.
    Аугментации не считаются независимыми новыми дикторами.
    Прослушайте несколько примеров перед большим обучением.
    ''')
    cell('code', '''
    from IPython.display import Audio, display
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11})
    for label in [1, 0]:
        candidates = [row for row in rows if row['split'] == 'train' and row['label'] == label and row['generator'] not in ['real_speech', 'procedural_noise']]
        samples = [next(row for row in candidates if row['generator'] == engine) for engine in TTS_ENGINES
                   if any(row['generator'] == engine for row in candidates)]
        for row in samples:
            print(f'label={label}; voice={row["speaker"]}; text={row["text"]}')
            display(Audio(filename=str(WORK_DIR / 'dataset' / row['wav'])))
    fig, ax = plt.subplots()
    names = list(dataset_summary)
    ax.bar(names, [dataset_summary[name]['positive'] for name in names], label='Wake', color='#E69F00', hatch='/')
    ax.bar(names, [dataset_summary[name]['negative'] for name in names],
           bottom=[dataset_summary[name]['positive'] for name in names], label='Non-wake', color='#0072B2')
    ax.set(ylabel='Synthetic windows (2 seconds)', title='Dataset size by speaker-disjoint split')
    ax.legend(); fig.tight_layout(); plt.show()
    ''', 'dataset-preview')
    cell('markdown', '''
    ## 6. Обучение DDP: обе GPU

    `torch.distributed.run --standalone --nproc_per_node=2` создаёт два процесса,
    `DistributedSampler` даёт каждому свою часть train, DDP синхронизирует градиенты.
    Эффективный batch по умолчанию 128 = 64 × 2. Это реальное обучение обеими GPU,
    но удвоенная скорость не гарантируется: загрузка данных и синхронизация тоже занимают время.

    По умолчанию дообучаются последние два слоя акустического Whisper Tiny encoder
    и бинарная голова. Encoder использует learning rate 1e-5, голова — 7e-4.
    `temporal_v2` позволяет отдельно сравнить обучение CNN с нуля. Weighted sampler
    балансирует классы, русский/английский и TTS-движки; каждые 10 эпох разбирает
    трудные примеры **только train**. EMA сглаживает веса. Лучшая эпоха выбирается
    по macro recall и худшему языку при заданном validation FPR; loss служит
    дополнительным критерием. Максимум 100 эпох, early stopping после 18 без улучшения.
    Два seed обучаются последовательно, каждый использует обе GPU через DDP.
    Затем validation выбирает одну модель или среднее logits двух. Test не участвует.
    После каждой эпохи сохраняются model, optimizer, scheduler, scaler и RNG каждого rank.
    Для продолжения загрузите прошлый Output как Dataset, задайте `RESUME_CHECKPOINT`,
    сохраните те же данные/seed/число GPU и общее `EPOCHS` исходного запуска.
    Для ансамбля задайте путь к `checkpoints/`; в каждом `member-i/` рядом с `latest.pt`
    нужны `latest-rng-0.pt` и `latest-rng-1.pt`. Для одной модели — путь к latest.pt.
    Смена датасета или количества GPU при resume отклоняется.
    ''')
    cell('code', '''
    def train_member(member):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={GPU_COUNT}', str(CODE_DIR / 'wake_train.py'), '--config', str(CONFIG_FILE), '--member', str(member)]
        print(' '.join(command))
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as training_process:
            try:
                for line in training_process.stdout: print(line.rstrip(), flush=True)
                if training_process.wait(): raise subprocess.CalledProcessError(training_process.returncode, command)
            except BaseException:
                training_process.terminate()
                try: training_process.wait(timeout=15)
                except subprocess.TimeoutExpired: training_process.kill(); training_process.wait()
                raise
    for member in range(CONFIG['ensemble_members']): train_member(member)
    training_run = json.loads((WORK_DIR / 'training_run.json').read_text(encoding='utf-8'))
    assert training_run['world_size'] == GPU_COUNT
    print(training_run)
    ''', 'train-ddp')
    cell('markdown', '''
    ## 7. Порог, независимый test и экспорт ONNX

    Порог выбирается на **calibration**, чтобы доля false positive среди негативных
    окон была не выше заданной цели — и в целом, и отдельно в русском и английском.
    После фиксации порога оценивается test. Дополнительные цели FPR 0.2/0.5/1/2%
    заданы заранее и показывают компромисс, но основной порог остаётся `TARGET_FPR_PER_WINDOW`.
    Число ложных срабатываний, пропуски, recall/precision и группы голосов записываются
    в `training_report.json`. FPR на синтетических окнах не является FAR/час реального
    непрерывного разговора. При провале recall высокий порог не делает модель хорошей.

    Экспорт включает CPU ONNX, веса PyTorch, frontend, конфигурацию, отчёты и источники.
    В V4 encoder входит в ONNX; после обучения для инференса нужны только NumPy и ONNX Runtime.
    ONNX проверяется сравнением с PyTorch на WAV из test; новый независимый инференс
    использует тот же frontend. При `speech_embedding_v3` это отдельные фиксированные
    mel/embedding ONNX и обученная бинарная голова; вся цепочка входит в ZIP.
    ''')
    cell('code', '''
    subprocess.run([sys.executable, str(CODE_DIR / 'wake_export.py'), '--config', str(CONFIG_FILE)], check=True)
    MODEL_DIR = WORK_DIR / ('wake-model-SMOKE-ONLY' if SMOKE else 'wake-model')
    report = json.loads((MODEL_DIR / 'training_report.json').read_text(encoding='utf-8'))
    print('Calibration:', report['calibration'])
    print('Untouched test:', report['test'])
    if report['calibration']['recall'] < 0.8 or report['test']['recall'] < 0.8:
        print('LOW RECALL: inspect errors and improve data/model before deploying this candidate.')
    if report['test']['false_positive_rate_per_window'] > TARGET_FPR_PER_WINDOW:
        print('Test FPR exceeds the calibration target. Do not retune on test; add new training/calibration data.')
    ''', 'export')
    cell('markdown', '''
    ## 8. Результаты запуска

    Графики показывают фактически выполненное обучение и распределение оценок
    независимого синтетического test. Пересечение распределений показывает ошибки;
    высокая accuracy на фоне большого числа негативов может скрывать пропуски.
    Смотрите одновременно recall и false-positive rate, включая отдельные голоса.
    ''')
    cell('code', '''
    history = json.loads((WORK_DIR / 'training_history.json').read_text(encoding='utf-8'))
    predictions = json.loads((WORK_DIR / 'test_predictions.json').read_text(encoding='utf-8'))
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for member in sorted({row.get('member', 0) for row in history}):
        subset = [row for row in history if row.get('member', 0) == member]
        axes[0].plot([row['epoch'] for row in subset], [row['train_loss'] for row in subset], label=f'Train seed {member}', linestyle='--')
        axes[0].plot([row['epoch'] for row in subset], [row['validation_loss'] for row in subset], label=f'Validation seed {member}')
        points = [row for row in subset if 'validation_operating_point' in row]
        if points:
            axes[1].plot([row['epoch'] for row in points], [row['validation_operating_point']['macro_recall'] for row in points], label=f'Macro seed {member}')
            axes[1].plot([row['epoch'] for row in points], [row['validation_operating_point']['worst_language_recall'] for row in points], label=f'Worst language seed {member}', linestyle='--')
    axes[0].set(title='Loss by epoch (different class weighting)', xlabel='Epoch', ylabel='Binary cross-entropy')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()
    axes[1].set(title=f'Validation recall at FPR ≤ {TARGET_FPR_PER_WINDOW:g}', xlabel='Epoch', ylabel='Recall', ylim=(0, 1))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[1].legend()
    for label, title in [(0, 'Non-wake'), (1, 'Wake')]:
        values = [row['probability'] for row in predictions if row['label'] == label]
        axes[2].hist(values, bins=np.linspace(0, 1, 31), histtype='step', linewidth=2, density=True,
                     color='#0072B2' if label == 0 else '#E69F00', label=f'{title} (n={len(values)})')
    axes[2].axvline(report['calibration_threshold'], color='black', linestyle='--', label='Calibration threshold')
    axes[2].set(title='Held-out test scores', xlabel='Wake probability', ylabel='Density', xlim=(0, 1))
    axes[2].legend(); fig.tight_layout()
    fig.savefig(MODEL_DIR / 'training_and_scores.png', dpi=150); plt.show()
    print('False positives:', report['test']['fp'], '/', report['test']['negative_windows'])
    print('Missed wakes:', report['test']['fn'], '/', report['test']['positive_windows'])
    print('Language metrics:', json.dumps(report['test_groups']['language'], indent=2))
    print('Predeclared FPR profiles:', json.dumps(report.get('threshold_profiles', []), indent=2))
    errors = [row for row in predictions if row['predicted_wake'] != bool(row['label'])]
    print('First test errors (at most 10):', json.dumps(errors[:10], ensure_ascii=False, indent=2))
    ''', 'results')
    cell('markdown', '''
    ## 9. Скачать и проверить в отдельном процессе

    В Kaggle Output найдёте ZIP с моделью и `dataset/` с WAV, признаками и manifest.
    `INCLUDE_DATASET_IN_ZIP=True` дополнительно упаковывает датасет в отдельный ZIP.
    Скачайте `checkpoints/`, если планируете продолжать обучение.
    Локальный пример: `python wake_audio.py <model_directory> <mono_16k_pcm16.wav>`.
    Для короткой записи frontend дополняет её тишиной слева до двух секунд.
    Аудио должно иметь частоту 16 кГц; frontend не угадывает частоту записи.

    Обязательно проверьте реальные отдельные обращения, похожие слова и фоновые
    разговоры через ваш микрофон. Для оценки непрерывного детектора отдельно нужны
    правила окон, debounce/cooldown и длительные реальные негативные записи.
    В текущем боте можно будет оставить распознавание команд в Groq и заменить
    облачную проверку имени этим локальным детектором после такой проверки.
    ''')
    cell('code', '''
    import shutil
    from IPython.display import FileLink
    # A fresh process reloads ONNX and frontend, independent of the training objects.
    example = next(row for row in rows if row['split'] == 'test' and row['label'] == 1)
    subprocess.run([sys.executable, str(MODEL_DIR / 'wake_audio.py'), str(MODEL_DIR),
                    str(WORK_DIR / 'dataset' / example['wav'])], check=True)
    for name in ['environment.json', 'source_sha256.json']:
        shutil.copy2(WORK_DIR / name, MODEL_DIR / name)
    model_zip = shutil.make_archive(str(MODEL_DIR), 'zip', root_dir=MODEL_DIR)
    display(FileLink(model_zip))
    if INCLUDE_DATASET_IN_ZIP:
        dataset_zip = shutil.make_archive(str(WORK_DIR / 'synthetic-wake-dataset'), 'zip', root_dir=WORK_DIR / 'dataset')
        display(FileLink(dataset_zip))
    print('Model:', model_zip)
    print('Dataset:', WORK_DIR / 'dataset' / 'manifest.jsonl')
    print('Quality on real users has not been validated; synthetic metrics are in training_report.json.')
    ''', 'download')
    cell('markdown', '''
    ## Источники и границы проверки

    - [Silero TTS: модели, голоса, standalone API и лицензии](https://github.com/snakers4/silero-models).
      Используются официальные `v5_5_ru` и `v3_en`; SHA-256 загруженных файлов
      записываются в `tts_provenance.json`. Тексты негативов составлены для этого ноутбука.
    - [Piper](https://github.com/OHF-Voice/piper1-gpl) и [голоса](https://huggingface.co/rhasspy/piper-voices).
    - [MMS/VITS](https://huggingface.co/docs/transformers/model_doc/vits).
    - [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) и [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
    - [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands): реальные негативные слова,
      исходные ссылки, SHA и лицензионные ссылки сохраняются в `real_provenance.json`.
    - [openWakeWord: Google speech embeddings](https://github.com/dscripka/openWakeWord#model-architecture):
      закреплённые преобразования v0.5.1, SHA-256 проверяются до вычисления признаков.
      Код upstream имеет Apache-2.0, для распространяемых готовых моделей upstream
      также указывает CC BY-NC-SA; лицензии и ссылки сохранены в provenance.
    - [Whisper Tiny](https://huggingface.co/openai/whisper-tiny): закреплена ревизия
      `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; оригинальный encoder адаптируется
      к окну 2 секунды и обучается для бинарной задачи, а не генерации транскрипции.
    - [PyTorch torchrun](https://pytorch.org/docs/stable/elastic/run.html) и
      [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
    - [Kaggle: GPU T4 ×2](https://www.kaggle.com/product-feedback/361104).

    Encoder загружается из готовой многоязычной модели, бинарная голова обучается здесь.
    Метрики вашего запуска появятся только после выполнения ячеек.
    SMOKE-ONLY проверяет код, а не точность; его нельзя считать готовым детектором.
    Синтетический датасет воспроизводим по recipe/seed, исходникам и сохранённым
    TTS-весам; после обновления пакетов/весов полное совпадение не гарантируется.
    Голос aidar и en_20–23 уже встречались в отчёте V1: эта часть test — повторный
    benchmark, не полностью новый слепой тест. Новые движки/голоса и реальные
    негативы расширяют проверку. Для окончательной оценки нужен новый реальный корпус.
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.11'},
                             'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True},
                             'wakeword': {'source_sha256': hashes, 'target': 'бот / bot', 'requires_gpus': 2}}, 'cells': cells}
    if output is not None:
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Created {output} ({len(cells)} cells; {output.stat().st_size:,} bytes)')
    return notebook


if __name__ == '__main__':
    build()
