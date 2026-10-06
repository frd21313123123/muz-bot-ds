"""V7: separate language FPR constraints from source monitoring; compare Tiny adaptation."""
import ast
import json
from build_wakeword_v6_notebook import build_v6, ROOT
from build_wakeword_v5_notebook import make_cell

OUTPUT = ROOT / 'notebooks/wakeword_bot_kaggle_v7.ipynb'


def build_v7(output=OUTPUT):
    notebook = build_v6(output=None)
    cells = notebook['cells']
    by_tag = {tag: cell for cell in cells for tag in cell['metadata'].get('tags', [])}
    cells[0] = make_cell('markdown', '''
    # «Бот» V7: перенос между голосами и проверка дообучения Tiny

    V6 завершилась, но выбранная модель дала **44.46% recall** при **0.128% FPR**.
    RU recall — 48.72%, EN — 40.81%. Piper RU: validation 8.22%, calibration 0%,
    test 21.44%. Один порог не может довести текущие тестовые оценки до 90% при
    FPR ≤0.5%: даже лучший диагностический RU-порог дал бы только 57% recall.
    Это вывод о прежних оценках, не новый порог для использования.

    V7 проверяет обучение всех четырёх Tiny-слоёв и убирает дополнительное
    ограничение FPR по каждому TTS из решения о пороге. **FPR ≤0.5% остаётся
    обязательным в целом и отдельно RU/EN.** Источники и версии Silero остаются
    в отчётах и в критерии выбора через worst-source recall. Данные слабых
    голосов не удаляются из оценки. Исходные goal/quality gate не смягчаются.

    Три ученика: контроль с одним trainable слоем; четыре trainable слоя без
    подсказок; четыре слоя с подсказками **Whisper Large V3**. Все имеют одинаковые
    seed, learning rate, размер окна, sampler и предел эпох. Первый сравнивается
    со вторым для проверки адаптации, второй с третьим — для проверки дистилляции.
    Validation выбирает один компактный Tiny; ансамбль и учитель не публикуются.

    **Kaggle: T4 ×2, Internet ON, Run All.** Каждая ветка учится на обеих GPU,
    последовательно. API-ключи не нужны. Четыре семейства локальных TTS и разделение
    голосов сохранены из V6; seed синтеза сохранён для сопоставимого рецепта данных.
    Для нового рабочего каталога потребуется повторная генерация. Совпадение
    аудио зависит также от версий библиотек/весов; provenance сохраняется.

    Это эксперимент, **90% ещё не достигнуты**. Benchmark уже использован
    при разработке, нужен новый independently reviewed human holdout.
    Output содержит только модель и отчёты, включая оценки calibration;
    датасет, TTS/STT-веса и checkpoints остаются в `/kaggle/temp`.
    ''')
    parameter = by_tag['parameters']
    source = parameter['source'].replace('ENSEMBLE_MEMBERS = 2', 'ENSEMBLE_MEMBERS = 3')
    source = source.replace('wakeword-bot-v6', 'wakeword-bot-v7').replace('recipe_version=6', 'recipe_version=7')
    source = source.replace('# V6 student', '# V7 student').replace("'V6 requires", "'V7 requires")
    marker = 'immutable = {key: CONFIG[key]'
    assert marker in source
    block = '''# Keep the synthesis seed and all V6 voice splits unchanged.
CONFIG.update(source_group_fpr_constraint=False, threshold_diagnostics=True,
              export_calibration_predictions=True,
              member_overrides={
                  '0': dict(distill_weight=0, encoder_train_layers=1, encoder_learning_rate=3e-6),
                  '1': dict(distill_weight=0, encoder_train_layers=4, encoder_learning_rate=3e-6),
                  '2': dict(distill_weight=DISTILL_WEIGHT, encoder_train_layers=4, encoder_learning_rate=3e-6)})
if SMOKE:
    CONFIG['member_overrides'] = {'0': dict(distill_weight=DISTILL_WEIGHT, encoder_train_layers=4, encoder_learning_rate=3e-6)}
'''
    parameter['source'] = source.replace(marker, block + marker)
    for cell in cells:
        if cell['cell_type'] == 'markdown' and 'Tiny encoder сохраняет все четыре слоя' in cell['source']:
            cell['source'] = make_cell('markdown', '''
            ## 3. Акустическая модель и встроенные исходники

            Все ученики получают 80 × 200 Whisper log-mel для двух секунд аудио.
            Tiny encoder сохраняет четыре слоя: контроль обучает последний,
            две другие ветки обучают все четыре и бинарную голову. Decoder отсутствует.
            Large V3 получает отдельные 128 mel-каналов из тех же train/validation WAV.
            ''')['source']
    training_index = cells.index(by_tag['train-ddp'])
    cells[training_index - 1] = make_cell('markdown', '''
    ## 6b. Контроль, адаптация и подсказки

    | Member | Trainable Tiny-слои | Encoder LR | Distillation weight |
    |---|---:|---:|---:|
    | 0: контроль | 1 из 4 | 3e-6 | 0 |
    | 1: адаптация | 4 из 4 | 3e-6 | 0 |
    | 2: адаптация + Large V3 | 4 из 4 | 3e-6 | 0.4 |

    В каждой ветке также обучается бинарная голова. Все ветки сохраняют четыре
    слоя при экспорте, поэтому изменение числа обучаемых слоёв не увеличивает
    рабочую модель. Общее обучение — максимум 70 эпох, minimum 20, patience 14;
    EMA, SpecAugment и train-only mining сохранены. Seed одинаковый, не три
    независимых случайных повтора. После изменения слоёв траектории расходятся.

    Два сравнения помогают проверить гипотезы в одном наборе данных. Одного
    запуска недостаточно для причинного/статистического подтверждения улучшения.
    Подсказки Large V3 доступны только train. Исходные метки сохраняются,
    противоречивые подсказки игнорируются. Учитель видит только train/validation.

    На validation ограничивается FPR в целом и отдельно RU/EN. Для каждого
    источника сохраняются метрики, отрицательные хвосты и границы порога.
    Дополнительные source-boundaries явно помечены `applied=false` и не
    поднимают порог. Worst-source recall остаётся в критерии выбора ученика.
    ''')
    export_index = cells.index(by_tag['export'])
    cells[export_index - 1] = make_cell('markdown', '''
    ## 7. Экспорт, INT8, calibration и качество

    Эпоха и ученик выбираются на validation. ONNX сверяется с PyTorch. INT8
    принимается на validation при потере macro/worst-language/worst-source
    recall не более 1 п.п., до открытия calibration/test. Runtime INT8
    оценивается по одному окну, как рабочий детектор. Ошибка INT8 даёт FP32 fallback.

    Затем порог выбранного runtime определяется только по calibration,
    с ограничением FPR ≤0.5% в целом и отдельно RU/EN. Исходники по TTS
    остаются в `calibration_source_groups`; их FPR не гарантируется ≤0.5%.
    `calibration_threshold_audit` показывает, какая применяемая граница
    определила порог, число негативов и допустимых FP. Сохраняются также
    `calibration_predictions_fp32.json` / `calibration_predictions_int8.json`:
    ID, текст, голос, label, probability, без аудио/признаков/архива датасета.

    Test читается после фиксации выбора и порога. `quality_gate` по-прежнему
    требует recall ≥90% и FPR ≤0.5% в целом и отдельно RU/EN. Passed относится
    только к наблюдаемым оценкам этого benchmark, не к новым людям/микрофонам.
    Скачивать для работы нужно `wake-model.zip`; Large V3 для inference не нужен.
    ''')
    cells[-1] = make_cell('markdown', '''
    ## Источники и ограничения

    Рецепт V7 основан на V6 reports/21 060 test predictions. В V6 calibration
    Kokoro достигал лимита FP 12/2400, а Piper RU не распознал 900 положительных
    окон. По агрегатам нельзя строго восстановить конкретную максимальную
    границу: raw calibration scores не были предоставлены. V7 сохраняет их
    вместе с threshold audit, чтобы следующий разбор не зависел от догадки.

    Голоса Piper RU остаются: Irina train, Ruslan validation, Dmitri calibration,
    Denis test; последнего нельзя добавлять в train после изучения его ошибок.
    Проблему малого разнообразия русского Piper train V7 пока не решает.
    Добавление новых дикторов/движков требует независимого разбиения и проверки
    произношения; изменение pitch не считается новым диктором. Нужны проверенные
    реальные positive/negative записи через `REAL_WAKE_MANIFEST` и новые
    отдельно отложенные люди для подтверждения 90%.

    - [Whisper](https://github.com/openai/whisper): encoder Tiny и Large V3.
    - [Silero](https://github.com/snakers4/silero-models),
      [Piper](https://huggingface.co/rhasspy/piper-voices),
      [MMS](https://huggingface.co/docs/transformers/model_doc/vits),
      [Kokoro](https://github.com/thewh1teagle/kokoro-onnx): TTS/provenance.

    Увеличение эпох не подменяет независимую оценку; ранняя остановка сохраняется.
    V6 и его опубликованный порог не меняются. Новый threshold policy действует
    только в V7 с новым WORK_DIR. Это проверка новых рецептов, не обещание 90%.
    ''')
    notebook['metadata']['wakeword'].update(version=7, group_calibration=False, source_monitoring=True,
        benchmark='V6-informed regression; V6 synthesis recipe/seed retained; no fresh blind human holdout')
    for i, cell in enumerate(cells):
        cell['id'] = f'wake-v7-{i:02d}'
        if cell['cell_type'] == 'code': ast.parse(cell['source'])
    if output is not None:
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Created {output}: {len(cells)} cells, {output.stat().st_size:,} bytes')
    return notebook


if __name__ == '__main__': build_v7()
