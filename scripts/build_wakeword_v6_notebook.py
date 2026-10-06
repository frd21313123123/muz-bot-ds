"""V6 addresses the observed V5 data coverage and acoustic-label problems."""
import ast
import json
from build_wakeword_v5_notebook import build_v5, make_cell, ROOT

OUTPUT = ROOT / 'notebooks/wakeword_bot_kaggle_v6.ipynb'


def build_v6(output=OUTPUT):
    notebook = build_v5(output=None)
    cells = notebook['cells']
    by_tag = {tag: cell for cell in cells for tag in cell['metadata'].get('tags', [])}
    cells[0] = make_cell('markdown', '''
    # «Бот» V6: исправление данных и контроль переноса между TTS

    V5 завершилась, но дала test recall **66.59%**, RU **55.67%**, EN **75.95%**.
    Piper Denis дал 75 из 96 ложных срабатываний; русские Piper отсутствовали
    в validation/calibration. Для Silero Aidar результат v4 был 39.11%, v5.5 — 93.56%.
    Поэтому следующий эксперимент меняет данные и отбор модели, а не число эпох.

    Цель остаётся **recall ≥90% и FPR ≤0.5%**, в целом и отдельно RU/EN.
    Результат V6 ещё не измерен. Увеличение модели, дистилляция и синтетика не
    гарантируют достижения цели или переноса на реальные микрофоны.

    Исправления: русские Piper присутствуют во всех четырёх частях с разными
    голосами; потенциально неоднозначное «бод» исключено из отрицательных текстов;
    добавлено нейтральное «бот» / bot без пунктуации. Конечное оглушение /д/
    делает «бод» рискованной отрицательной меткой для акустического детектора.
    Одинаковость конкретных TTS-записей не установлена: исходное аудио V5 отсутствует.

    Порог и отбор учитывают TTS/язык и версии Silero отдельно. Оба ученика обучают
    только последний Tiny encoder-слой, с одинаковой инициализацией и learning rate;
    один использует подсказки Large V3, второй — исходные метки. Это эксперимент
    с меньшим изменением предобученного encoder, а не обещанное улучшение.

    **Kaggle: T4 ×2, Internet ON, Run All.** Все исходники встроены. Учитель —
    Whisper **Large V3**; рабочая модель — одиночный компактный Tiny encoder.
    Output содержит модель и небольшие отчёты. Датасет, TTS, Large V3 и checkpoints
    находятся в `/kaggle/temp`; архив датасета не создаётся.

    V5 test уже использован для разработки V6. Этот test — регрессионная проверка,
    а не новый слепой эксперимент; рецепт оценки изменён. Нужны новые независимо
    размеченные реальные люди для подтверждения качества. Их можно добавить через
    `REAL_WAKE_MANIFEST` с разделением по дикторам, без автоматических ASR-меток.
    ''')
    parameter = by_tag['parameters']
    source = parameter['source'].replace('SEED = 20261015', 'SEED = 202610061')
    source = source.replace('ENCODER_TRAIN_LAYERS = 4', 'ENCODER_TRAIN_LAYERS = 1')
    source = source.replace('ENCODER_LEARNING_RATE = 0.000005', 'ENCODER_LEARNING_RATE = 0.000003')
    source = source.replace('/kaggle/temp/wakeword-bot-v5', '/kaggle/temp/wakeword-bot-v6')
    source = source.replace('/kaggle/working/wakeword-bot-v5', '/kaggle/working/wakeword-bot-v6')
    source = source.replace('# V5 student uses Tiny;', '# V6 student uses Tiny;')
    source = source.replace("'V5 requires the Whisper Tiny student", "'V6 requires the Whisper Tiny student")
    source = source.replace('TEACHER_REVISION = None  # resolved once to an immutable commit; recorded in teacher_provenance.json',
                            "TEACHER_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'  # Large V3 revision used in the completed V5 run")
    marker = 'immutable = {key: CONFIG[key]'
    assert marker in source
    block = '''CONFIG.update(recipe_version=6,
              piper_ru_splits={'train': ['ru_RU-irina-medium'],
                               'validation': ['ru_RU-ruslan-medium'],
                               'calibration': ['ru_RU-dmitri-medium'],
                               'test': ['ru_RU-denis-medium']},
              negative_exclusions={'ru': ['бод']} if WAKE_WORD_RU.strip().casefold() == 'бот' else {},
              positive_bare_word=True, source_groups=True, export_sample_metadata=True,
              member_seed_stride=0,
              member_overrides={'0': dict(distill_weight=0, encoder_train_layers=1, encoder_learning_rate=3e-6),
                                '1': dict(distill_weight=DISTILL_WEIGHT, encoder_train_layers=1, encoder_learning_rate=3e-6)})
if SMOKE:
    CONFIG['member_overrides'] = {'0': dict(distill_weight=DISTILL_WEIGHT, encoder_train_layers=1, encoder_learning_rate=3e-6)}
'''
    source = source.replace(marker, block + marker)
    old = "'negative_tts_variants', 'negative_kind_weights']}"
    assert old in source
    source = source.replace(old, "'negative_tts_variants', 'negative_kind_weights', 'piper_ru_splits', 'negative_exclusions', 'positive_bare_word']}")
    parameter['source'] = source
    for cell in cells:
        if cell['cell_type'] == 'markdown' and 'Whisper Tiny с окном' in cell['source']:
            cell['source'] = make_cell('markdown', '''
            ## 3. Акустическая модель и встроенные исходники

            Рабочая модель получает 80 × 200 Whisper log-mel для двух секунд аудио.
            Tiny encoder сохраняет все четыре слоя; обучается только последний слой
            и бинарная голова, с разными learning rate. Decoder отсутствует.
            Замороженные слои остаются в рабочей модели, а не удаляются при экспорте.
            Учитель Large V3 получает отдельные 128 mel-каналов из тех же train/validation WAV.
            ''')['source']
    source_check_index = cells.index(by_tag['source-check'])
    cells[source_check_index + 1:source_check_index + 1] = [make_cell('markdown', '''
    ## Проверка разбиения до скачивания больших весов

    Из четырёх доступных русских Piper-голосов один остаётся в train (Irina),
    один в validation (Ruslan), один в calibration (Dmitri), один в test (Denis).
    Это уменьшает разнообразие Piper train относительно V5, зато оценка не
    маскирует перенос между движками. Silero train содержит мужской и женский
    голоса, MMS тоже остаётся в train. Denis не добавляется в обучение.
    ''', 'coverage-notes'), make_cell('code', '''
    from wake_tts import voice_jobs
    jobs = voice_jobs(CONFIG)
    identity_splits = {}
    for job in jobs:
        assert identity_splits.setdefault(job['speaker_group'], job['split']) == job['split']
    for split in ['train', 'validation', 'calibration', 'test']:
        selected = [job for job in jobs if job['engine'] == 'piper' and job['language'] == 'ru' and job['split'] == split]
        assert selected, f'Missing Russian Piper coverage: {split}'
        print(split, [job['voice'] for job in selected])
    print('Excluded acoustic-label candidates:', CONFIG['negative_exclusions'])
    ''', 'coverage-check')]
    generation_index = next(i for i, cell in enumerate(cells) if cell['cell_type'] == 'code' and
                            'wake_data.download_models(WORK_DIR, CONFIG)' in cell['source'])
    cells[generation_index - 1] = make_cell('markdown', '''
    ## 4. Данные на двух GPU

    Используются четыре локальных семейства TTS: Silero v4/v5.5/v3_en, Piper,
    MMS, Kokoro. Изменения pitch/tempo — вариации голоса, не новые дикторы.
    Разбиение Piper RU показано выше; голоса и исходные записи не пересекаются.
    Положительные тексты: слово без пунктуации, с точкой, вопросом, восклицанием.
    Полное слово сохраняется при аугментациях. Контекстные упоминания остаются
    отрицательными в `STANDALONE_ONLY=True`; задача не меняется на распознавание команд.

    «Бод» исключено из всех частей нового рецепта, а не только из test. «Боту»,
    «бок», «борт», «вот», «кот» и прочие различимые похожие слова остаются.
    V5 не пересчитывается после удаления сложных примеров. Метрики V6 нельзя
    сравнивать с V5 как на одном неизменном наборе.

    Шум, реверберация, независимый pitch, темп и громкость применяются к обоим
    классам. Speech Commands даёт реальные негативы. Для реальных положительных
    записей укажите `REAL_WAKE_MANIFEST`: JSONL с wav, label, language, speaker,
    split, text и reviewed=true; один диктор принадлежит одной части.
    WAV — mono PCM16 16 kHz, полное высказывание короче двух секунд.
    ''')
    training_index = cells.index(by_tag['train-ddp'])
    cells[training_index - 1] = make_cell('markdown', '''
    ## 6b. Tiny: осторожное дообучение и сравнение подсказок

    Обе модели имеют одинаковый seed, последний trainable encoder-слой и его
    learning rate 3e-6. Остальные Tiny encoder-слои заморожены. Ветка 0 использует
    только исходные метки, ветка 1 добавляет KL-подсказки Large V3 только для train.
    Неверные подсказки не подменяют метки. EMA и mining сохраняются.
    Потери student train/validation и метрики по группам записываются в историю.

    Один общий порог удовлетворяет FPR на validation отдельно RU/EN и каждой
    группе TTS/язык; версии Silero проверяются отдельно. При выборе эпохи и ученика
    также учитывается худший recall группы. Такой контроль может снизить общий
    recall; он не является способом искусственно получить 90%.
    ''')
    export_index = cells.index(by_tag['export'])
    cells[export_index - 1]['source'] += '''
В V6 критерий INT8 дополнительно сохраняет worst-source recall. Порог каждой
версии фиксируется на calibration с ограничением FPR по тем же группам источников.
Группы calibration и их знаменатели сохраняются в calibration_source_groups.
Test открывается после всех решений. Старый опубликованный порог V5 не изменяется.
'''
    cells[-1] = make_cell('markdown', '''
    ## Источники и границы эксперимента

    V6 разработана по 21 060 предсказаниям V5; исходное аудио этих ошибок не
    предоставлено. Нельзя установить качество произношения, шум или обрезание
    конкретных TTS-примеров только по JSON. Новые данные и отбор — гипотезы для
    следующего эксперимента, их преимущество ещё не измерено.

    - [Whisper](https://github.com/openai/whisper): многоязычный encoder, веса Tiny и Large V3.
    - [openWakeWord](https://github.com/dscripka/openWakeWord): синтетическая речь и frozen-feature подходы.
    - [Russian final devoicing](https://www.sciencedirect.com/science/article/abs/pii/S0095447014000175):
      оглушение создаёт риск неоднозначных меток; исследование также описывает
      неполную нейтрализацию, поэтому конкретные аудио не объявляются идентичными.
    - [Silero](https://github.com/snakers4/silero-models), [Piper](https://huggingface.co/rhasspy/piper-voices),
      [MMS](https://huggingface.co/docs/transformers/model_doc/vits), [Kokoro](https://github.com/thewh1teagle/kokoro-onnx):
      ссылки и лицензии сохраняются в provenance.

    Голоса test остаются вне train/validation/calibration нового запуска. Поскольку
    ошибки этого benchmark уже изучены и рецепт изменён, результат не считается
    новым независимым подтверждением 90%. Для него нужен новый human holdout.
    ''')
    notebook['metadata']['wakeword'].update(version=6, group_calibration=True,
                                            benchmark='V5-informed regression; changed label/synthesis recipe')
    for i, cell in enumerate(cells):
        cell['id'] = f'wake-v6-{i:02d}'
        if cell['cell_type'] == 'code': ast.parse(cell['source'])
    if output is not None:
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Created {output}: {len(cells)} cells, {output.stat().st_size:,} bytes')
    return notebook


if __name__ == '__main__': build_v6()
