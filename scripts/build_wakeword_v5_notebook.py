"""Self-contained Kaggle V5: Large V3 teacher, Tiny student, small published output."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
from build_wakeword_notebook import build, ROOT

OUTPUT = ROOT / 'notebooks/wakeword_bot_kaggle_v5.ipynb'


def make_cell(kind, source, tag=None, hidden=False):
    source = textwrap.dedent(source).strip() + '\n'
    if kind == 'code': ast.parse(source)
    cell = dict(cell_type=kind, metadata={}, source=source)
    if tag: cell['metadata']['tags'] = [tag]
    if hidden: cell['metadata']['jupyter'] = {'source_hidden': True}
    if kind == 'code': cell.update(execution_count=None, outputs=[])
    return cell


def build_v5(output=OUTPUT):
    notebook = build(output=None)
    cells = notebook['cells']
    cells[0] = make_cell('markdown', '''
    # «Бот» V5: Whisper Large V3 → компактный детектор

    Цель: **recall ≥90%** обращений «бот» / “bot” при **FPR ≤0.5%** отрицательных
    двухсекундных окон, отдельно для русского и английского. Это проверяемая цель,
    а не обещание результата. Общая accuracy не заменяет recall.
    Предыдущая V4 пропускала 40% русских обращений; увеличение эпох само по себе
    не решает перенос на другой голос. Новый запуск усиливает hard negatives,
    сравнивает обычное обучение с дистилляцией и проверяет сжатие.

    Учитель — локальный многоязычный **Whisper Large V3**: его encoder дообучается
    на бинарную задачу «отдельное обращение / прочий звук». Учитель передаёт ученику
    мягкие оценки только train-примеров. Decoder и транскрипция для этой задачи
    не требуются. Ученик — encoder Whisper Tiny с бинарной головой, около 31 МБ FP32.
    Дополнительно проверяется INT8; точный размер и качество появятся после запуска.

    В Kaggle: **GPU T4 ×2**, **Internet ON**, затем **Run All**. API-ключи не нужны.
    Генерация, обучение учителя, подсказки и обучение учеников используют обе GPU.
    Четыре семейства TTS: Silero, Piper, MMS, Kokoro; голоса разделены до аугментаций.

    **Output содержит только ZIP модели и небольшие отчёты.** Датасет, TTS,
    учитель и checkpoints создаются в `/kaggle/temp`, вне `/kaggle/working`.
    Для финальной проверки реальных людей подключите проверенный `REAL_WAKE_MANIFEST`.
    Без него метрики относятся к синтетическому benchmark и реальным негативным словам.
    ''')
    by_tag = {tag: cell for cell in cells for tag in cell['metadata'].get('tags', [])}
    parameter = by_tag['parameters']
    source = parameter['source']
    replacements = {
        'SEED = 20261005': 'SEED = 20261015',
        'POSITIVE_PER_VOICE = 300': 'POSITIVE_PER_VOICE = 400',
        'NEGATIVE_PER_VOICE = 600': 'NEGATIVE_PER_VOICE = 700',
        'EVAL_POSITIVE_PER_VOICE = 200': 'EVAL_POSITIVE_PER_VOICE = 300',
        'EVAL_NEGATIVE_PER_VOICE = 800': 'EVAL_NEGATIVE_PER_VOICE = 1200',
        'RUSSIAN_MULTIPLIER = 2': 'RUSSIAN_MULTIPLIER = 3',
        'POSITIVE_TTS_VARIANTS = 6': 'POSITIVE_TTS_VARIANTS = 12',
        'EPOCHS = 100': 'EPOCHS = 70', 'PATIENCE = 18': 'PATIENCE = 14', 'MIN_EPOCHS = 25': 'MIN_EPOCHS = 20',
        'ENCODER_TRAIN_LAYERS = 2': 'ENCODER_TRAIN_LAYERS = 4',
        'ENCODER_LEARNING_RATE = 0.00001': 'ENCODER_LEARNING_RATE = 0.000005',
        "ARCHITECTURE = 'whisper_encoder_v4'  # alternatives: 'temporal_v2', 'speech_embedding_v3', 'legacy'; new WORK_DIR required":
        "ARCHITECTURE = 'whisper_encoder_v4'  # V5 student uses Tiny; teacher has a separate encoder",
        "INCLUDE_DATASET_IN_ZIP = False  # WAV + features remain in Kaggle Output either way":
        "TARGET_RECALL = 0.90\nTEACHER_REPOSITORY = 'openai/whisper-large-v3'\nTEACHER_REVISION = None  # resolved once to an immutable commit; recorded in teacher_provenance.json\nTEACHER_EPOCHS = 18\nTEACHER_BATCH_PER_GPU = 16\nTEACHER_TRAIN_LAYERS = 2\nDISTILL_WEIGHT = 0.4\nDISTILL_TEMPERATURE = 2.0\nNEGATIVE_TTS_VARIANTS = 3\nQUANTIZATION_RECALL_TOLERANCE = 0.01",
        "WORK_DIR = Path('/kaggle/working/wakeword-bot-v4')":
        "WORK_DIR = Path('/kaggle/temp/wakeword-bot-v5')\nOUTPUT_DIR = Path('/kaggle/working/wakeword-bot-v5')",
        'recipe_version=4': 'recipe_version=5',
        'immutable = {key: CONFIG[key]': '''CONFIG.update(output_dir=str(OUTPUT_DIR), target_recall=TARGET_RECALL,
                  teacher_repository=TEACHER_REPOSITORY, teacher_revision=TEACHER_REVISION,
                  teacher_epochs=TEACHER_EPOCHS, teacher_batch_per_gpu=TEACHER_BATCH_PER_GPU,
                  teacher_train_layers=TEACHER_TRAIN_LAYERS, distill_temperature=DISTILL_TEMPERATURE,
                  negative_tts_variants=1 if SMOKE else NEGATIVE_TTS_VARIANTS,
                  negative_kind_weights=dict(hard_word=.65, context=.20, ordinary=.15),
                  teacher_targets=str(WORK_DIR / 'teacher-targets.npz'),
                  member_overrides={'0': dict(distill_weight=0, encoder_train_layers=2, encoder_learning_rate=1e-5),
                                    '1': dict(distill_weight=DISTILL_WEIGHT)},
                  quantization_recall_tolerance=QUANTIZATION_RECALL_TOLERANCE,
                  allow_ensemble=False, export_pytorch_weights=False, ddp_timeout_seconds=1800)
    if ARCHITECTURE != 'whisper_encoder_v4':
        raise ValueError('V5 requires the Whisper Tiny student; use the V4 notebook for other architectures')
    if SMOKE:
        CONFIG['member_overrides'] = {'0': dict(distill_weight=DISTILL_WEIGHT)}
    for key in ['HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME']:
        os.environ[key] = str(WORK_DIR / 'cache' / key.lower())
    (WORK_DIR / 'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(WORK_DIR / 'tmp')
    immutable = {key: CONFIG[key]''',
        "'real_wake_manifest', 'independent_pitch', 'architecture']}":
        "'real_wake_manifest', 'independent_pitch', 'architecture', 'negative_tts_variants', 'negative_kind_weights']}",
    }
    for old, new in replacements.items():
        if old not in source: raise ValueError(f'Missing V4 parameter scaffold: {old}')
        # The inserted CONFIG block must align with top-level code after dedent.
        source = source.replace(old, textwrap.dedent(new) if old.startswith('immutable') else new)
    source = source.replace('\n    if ARCHITECTURE', '\nif ARCHITECTURE').replace("\n        raise ValueError('V5", "\n    raise ValueError('V5")
    source = source.replace('\n    if SMOKE:\n', '\nif SMOKE:\n').replace("\n        CONFIG['member_overrides']", "\n    CONFIG['member_overrides']")
    # Restore alignment for the inserted top-level block, which sits inside an already dedented cell.
    source = source.replace("\n    for key in ['HF_HOME'", "\nfor key in ['HF_HOME'").replace('\n        os.environ[key]', '\n    os.environ[key]')
    source = source.replace("\n    (WORK_DIR / 'tmp')", "\n(WORK_DIR / 'tmp')").replace("\n    os.environ['TMPDIR']", "\nos.environ['TMPDIR']").replace('\n    immutable =', '\nimmutable =')
    parameter['source'] = source
    dependencies = by_tag['dependencies']
    dependencies['source'] += '''
import shutil
free_gib = shutil.disk_usage(WORK_DIR).free / 2**30
if free_gib < (12 if SMOKE else 40):
    raise RuntimeError(f'Transient disk has only {free_gib:.1f} GiB free; Large V3 + two feature sets require more space')
print(f'Transient workspace has {free_gib:.1f} GiB free; datasets will not enter Output.')
'''
    modules = ['wake_audio.py', 'wake_model.py', 'wake_data.py', 'wake_tts.py', 'wake_data_v2.py',
               'wake_metrics.py', 'wake_train.py', 'wake_export.py', 'wake_distill.py', 'wake_package_v5.py']
    hashes = {}
    embedded = []
    for name in modules:
        content = (ROOT / 'scripts/wakeword' / name).read_text(encoding='utf-8')
        ast.parse(content)
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
        embedded.append(make_cell('code', f'(CODE_DIR / {name!r}).write_text({content!r}, encoding="utf-8", newline="\\n")', f'embedded-{name}', True))
    start = next(i for i, cell in enumerate(cells) if 'embedded-wake_audio.py' in cell['metadata'].get('tags', []))
    end = next(i for i, cell in enumerate(cells) if 'source-check' in cell['metadata'].get('tags', []))
    cells[start:end] = embedded
    by_tag['source-check']['source'] = by_tag['source-check']['source'].replace(
        by_tag['source-check']['source'].splitlines()[0], f'SOURCE_SHA256 = {hashes!r}')
    train_index = cells.index(by_tag['train-ddp'])
    cells[train_index - 1] = make_cell('markdown', '''
    ## 6. Large V3: обучение учителя и подсказки на двух GPU

    Large V3 использует 128 mel-каналов, Tiny — 80: признаки учителя рассчитываются
    из тех же WAV отдельно. Учитель видит только train и validation, калибровочные
    и тестовые примеры ему не передаются. Дообучаются два последних encoder-слоя
    и бинарная голова; decoder не загружается в память.
    Каждая стадия обучения запускает DDP на двух T4; batch учителя — 16 на карту.
    Validation и train-only mining тоже делятся между обеими GPU, с FP16-инференсом.
    Результаты собираются в исходном порядке без дублирования окон. Тайм-аут
    группы — 30 минут, чтобы запись больших checkpoints не срывала синхронизацию.
    Максимум 18 эпох, early stopping; веса и датасет учителя остаются временными.

    Мягкие ответы кешируются двумя GPU только для train. Cache привязан к ID,
    контрольным суммам признаков и checkpoint учителя. При противоречии с разметкой
    подсказка игнорируется; правильная метка всегда участвует в supervised loss.
    ''')
    cells[train_index:train_index + 1] = [make_cell('code', '''
    def run_live(command):
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout: print(line.rstrip(), flush=True)
                if process.wait(): raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise

    def train_ddp(config_file, member=0):
        run_live([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                  f'--nproc_per_node={GPU_COUNT}', str(CODE_DIR / 'wake_train.py'),
                  '--config', str(config_file), '--member', str(member)])

    run_live([sys.executable, str(CODE_DIR / 'wake_distill.py'), '--config', str(CONFIG_FILE), '--action', 'prepare'])
    train_ddp(WORK_DIR / 'teacher/run_config.json')
    cache_processes = []
    try:
        for rank in range(GPU_COUNT):
            handle = (WORK_DIR / f'teacher-cache-{rank}.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', str(CODE_DIR / 'wake_distill.py'),
                       '--config', str(CONFIG_FILE), '--action', 'cache', '--rank', str(rank), '--world', str(GPU_COUNT)],
                       stdout=handle, stderr=subprocess.STDOUT)
            cache_processes.append((process, handle))
        while any(process.poll() is None for process, _ in cache_processes):
            if any(process.poll() not in (None, 0) for process, _ in cache_processes):
                raise RuntimeError('Teacher cache failed; inspect teacher-cache-*.log')
            time.sleep(2)
        if any(process.returncode for process, _ in cache_processes): raise RuntimeError('Teacher cache failed')
    finally:
        for process, handle in cache_processes:
            if process.poll() is None: process.terminate(); process.wait(timeout=15)
            handle.close()
    run_live([sys.executable, str(CODE_DIR / 'wake_distill.py'), '--config', str(CONFIG_FILE),
              '--action', 'merge', '--world', str(GPU_COUNT)])
    ''', 'teacher-ddp'), make_cell('markdown', '''
    ### Ученики: baseline и дистилляция

    В полном запуске member 0 обучается без подсказок, member 1 — с KL-loss учителя
    и исходными метками. У distilled-ученика дообучаются все четыре Tiny encoder-слоя
    с меньшим learning rate. EMA, баланс языков/движков и mining используют только train.
    Дополнительный небольшой SpecAugment применяется к тем же размеченным окнам.
    Validation выбирает лучшую одиночную модель: большой ансамбль не экспортируется.
    Дистилляция может проиграть baseline; ноутбук сохраняет такой результат честно.
    '''), make_cell('code', '''
    for member in range(CONFIG['ensemble_members']): train_ddp(CONFIG_FILE, member)
    training_run = json.loads((WORK_DIR / 'training_run.json').read_text(encoding='utf-8'))
    assert training_run['world_size'] == GPU_COUNT
    print(training_run)
    ''', 'train-ddp')]
    export_index = cells.index(by_tag['export'])
    cells[export_index - 1] = make_cell('markdown', '''
    ## 7. Экспорт, INT8 и контроль качества

    Эпоха и ученик выбираются на validation. Затем ONNX проходит проверку с PyTorch.
    INT8 сжимает постоянные матрицы attention/MLP; frontend и свёртки остаются FP32.
    INT8 принимается, если на validation macro/worst-language recall теряет не более
    1 процентного пункта. Выбор фиксируется до просмотра calibration/test.
    Оценки INT8 вычисляются по одному окну: динамические масштабы активаций
    зависят от batch, поэтому пакетная оценка может не совпасть с рабочим детектором.
    Для FP32 и INT8 отдельно рассчитываются пороги только на calibration.
    Test оценивается с этими фиксированными порогами.
    Если квантизация недоступна в текущем окружении, сохраняется FP32 и ошибка
    фиксируется в `compression_selection.json`; успешное обучение не теряется.

    `quality_gate.passed` требует recall ≥90% и FPR ≤0.5% в целом и отдельно RU/EN.
    Если цель не достигнута, статус будет failed, без изменения порога по test.
    Это проверка наблюдаемых оценок на benchmark, не гарантия на реальных пользователях.
    ''')
    by_tag['export']['source'] = '''
run_live([sys.executable, str(CODE_DIR / 'wake_package_v5.py'), '--config', str(CONFIG_FILE), '--action', 'export'])
MODEL_DIR = WORK_DIR / ('wake-model-SMOKE-ONLY' if SMOKE else 'wake-model')
run_live([sys.executable, str(CODE_DIR / 'wake_package_v5.py'), '--config', str(CONFIG_FILE)])
report = json.loads((OUTPUT_DIR / 'training_report.json').read_text(encoding='utf-8'))
print('Fixed-threshold test:', report['test'])
print('Quality gate:', report['quality_gate'])
'''
    result_index = cells.index(by_tag['results'])
    cells[result_index - 1] = make_cell('markdown', '''
    ## 8. Фактические результаты

    Сравните FP32 и INT8 на графике; основная модель уже выбрана по validation.
    Не выбирайте другой порог или модель по test. Если quality gate не пройден,
    нужны новые train-данные и новые реальные независимые записи, а не обещание 90%.
    ''')
    by_tag['results']['source'] = '''
history = json.loads((WORK_DIR / 'training_history.json').read_text(encoding='utf-8'))
reports = {name: json.loads((OUTPUT_DIR / f'training_report_{name}.json').read_text(encoding='utf-8')) for name in ['fp32', 'int8'] if (OUTPUT_DIR / f'training_report_{name}.json').is_file()}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for member in sorted({r.get('member', 0) for r in history}):
    subset = [r for r in history if r.get('member', 0) == member]
    axes[0].plot([r['epoch'] for r in subset], [r['validation_operating_point']['worst_language_recall'] for r in subset], label=f'Student {member}')
axes[0].axhline(TARGET_RECALL, color='black', linestyle='--', label='90% target')
axes[0].set(title='Validation: worst-language recall', xlabel='Epoch', ylabel='Recall', ylim=(0, 1)); axes[0].legend()
x = np.arange(3)
for i, (name, item) in enumerate(reports.items()):
    recalls = [item['test']['recall']] + [item['test_groups']['language'].get(lang, {}).get('recall', np.nan) for lang in ['ru', 'en']]
    axes[1].bar(x + (i - .5) * .35, recalls, width=.35, label=name, hatch='/' if i else '')
axes[1].axhline(TARGET_RECALL, color='black', linestyle='--'); axes[1].set(xticks=x, xticklabels=['All', 'RU', 'EN'], ylabel='Recall', ylim=(0, 1), title='Test: detected wakes'); axes[1].legend()
for i, (name, item) in enumerate(reports.items()):
    rates = [item['test']['false_positive_rate_per_window']] + [item['test_groups']['language'].get(lang, {}).get('false_positive_rate_per_window', np.nan) for lang in ['ru', 'en']]
    axes[2].bar(x + (i - .5) * .35, np.array(rates) * 100, width=.35, label=name, hatch='/' if i else '')
axes[2].axhline(TARGET_FPR_PER_WINDOW * 100, color='black', linestyle='--'); axes[2].set(xticks=x, xticklabels=['All', 'RU', 'EN'], ylabel='False positive windows (%)', title='Test: false positives'); axes[2].legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'quality.png', dpi=150); plt.show()
for name, item in reports.items():
    print(name, 'ONNX MB:', round(item['model_bytes'] / 1e6, 2), 'test:', item['test'], 'gate:', item['quality_gate'])
'''
    download_index = cells.index(by_tag['download'])
    cells[download_index - 1] = make_cell('markdown', '''
    ## 9. Скачать модель

    Скачайте **wake-model.zip**: внутри только ONNX, frontend, конфигурация и пример
    локального запуска. Альтернативный FP32/INT8 ZIP оставлен для сравнения.
    `output_summary.json` сообщает выбранный формат, размер и прохождение цели.
    Датасета, TTS, учителя, PyTorch-весов и checkpoints в Output нет.
    Исходники встроены в этот ноутбук, recipe и SHA остаются в отчётах.
    Прерванный интерактивный запуск можно продолжать с временного scratch-каталога;
    после завершения Kaggle-сессии датасет/checkpoints не сохраняются.

    Локальный запуск после распаковки: `pip install numpy onnxruntime`, затем
    `python wake_audio.py . call.wav` (mono PCM16 WAV, 16 кГц).
    ''')
    by_tag['download']['source'] = '''
from IPython.display import FileLink
summary = json.loads((OUTPUT_DIR / 'output_summary.json').read_text(encoding='utf-8'))
for file in sorted(OUTPUT_DIR.glob('*.zip')): display(FileLink(str(file)))
print(json.dumps(summary, ensure_ascii=False, indent=2))
assert not any(OUTPUT_DIR.glob('*dataset*.zip'))
assert not any(file.is_dir() for file in OUTPUT_DIR.iterdir())
print('Download wake-model.zip. Dataset and Large V3 teacher are transient and are not in Output.')
'''
    # Replace obsolete V4 prose while retaining its useful parameters/data preview sections.
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            cell['source'] = cell['source'].replace('Следующие восемь ячеек записывают полностью включённые в ноутбук модули в Output.',
                'Следующие десять ячеек записывают встроенные модули во временную папку.')
            cell['source'] = cell['source'].replace('Среди речевых негативов выделены 50% коротких слов, 25% упоминаний и 25% фраз.',
                'В train доли негативов: 65% коротких слов, 20% упоминаний и 15% фраз; каждый имеет три варианта темпа TTS. В оценочных частях сохраняются прежние доли 50/25/25.')
            cell['source'] = cell['source'].replace('Дообучаются последние\nдва encoder-слоя и бинарная голова, с разными learning rate.',
                'Число дообучаемых encoder-слоёв и learning rate задаёт рецепт каждого ученика.')
    cells[-1] = make_cell('markdown', '''
    ## Источники и границы результата

    - [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3): многоязычный STT encoder учителя, 128 mel-каналов. Commit и SHA записываются при подготовке.
    - [Whisper Tiny](https://huggingface.co/openai/whisper-tiny): ученик с 80 mel-каналами; закреплённый commit `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
    - [Knowledge distillation](https://arxiv.org/abs/1503.02531): мягкие ответы учителя дополняют исходную разметку.
    - [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): динамическая INT8-квантизация и необходимость проверки качества.
    - [Silero](https://github.com/snakers4/silero-models), [Piper](https://huggingface.co/rhasspy/piper-voices), [MMS](https://huggingface.co/docs/transformers/model_doc/vits), [Kokoro](https://github.com/thewh1teagle/kokoro-onnx): источники и лицензии в provenance.
    - [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands): реальные отрицательные слова; реальные положительные обращения этот корпус не даёт.
    - [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): каждый этап обучения использует обе GPU.

    Голоса из V4 test уже анализировались: это регрессионный benchmark, а не новый
    слепой тест. Они остаются вне train, validation и calibration. Новый synthesis seed
    не превращает их в новых дикторов. Для подтверждения 90% на людях нужен новый
    независимо размеченный корпус с разделением по людям и условиям записи.
    Перекрёстная проверка настоящих длительных разговоров нужна для FAR/час.
    Ни число эпох, ни Large V3, ни дистилляция не гарантируют результат.
    ''')
    notebook['metadata']['wakeword'].update(source_sha256=hashes, version=5, teacher='openai/whisper-large-v3', dataset_in_output=False)
    for i, cell in enumerate(cells):
        cell['id'] = f'wake-v5-{i:02d}'
        if cell['cell_type'] == 'code': ast.parse(cell['source'])
    if output is not None:
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Created {output}: {len(cells)} cells, {output.stat().st_size:,} bytes')
    return notebook


if __name__ == '__main__': build_v5()
