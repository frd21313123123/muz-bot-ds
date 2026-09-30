# Laya dataset for muz-bot-ds

Стартовый датасет для дообучения Laya Multilingual под этот Discord-музыкальный бот.

## Цель

Нейросеть отвечает только за семантические решения:

- `intent`: play, skip, stop, pause, resume, clear, volume, autoplay, queue, nowplaying, player, watch, unknown;
- `autoplay_mode`: enable, disable, toggle, unchanged;
- `is_music_control`: является ли текст командой музыкальному боту;
- `best_track`: выбор лучшего кандидата из результатов YouTube-поиска, включая `none`;
- `has_good_match`: есть ли среди кандидатов подходящий результат;
- `next_tool`: `youtube_music_search`, `direct_youtube_video`, `player_control` или `no_tool`;
- `search_result_policy`: `play_first_result`, `rerank_results` или `no_result`.

Проверка Discord permissions, нахождение пользователя в voice channel, диапазон громкости, управление соединением и очередью должны оставаться детерминированным TypeScript-кодом.

## Поток с YouTube Music

Датасет теперь отдельно обучает выбор инструмента и поведение после ответа инструмента.

Обычный запрос:

```text
"включи Numb"
    ↓
next_tool = youtube_music_search
    ↓
youtube_music_search("Numb")
    ↓
[0] Linkin Park - Numb
[1] Linkin Park - Numb (Lyrics)
[2] Numb Encore
    ↓
search_result_policy = play_first_result
    ↓
воспроизвести results[0]
```

Запрос со специальной версией:

```text
"включи Numb live"
    ↓
next_tool = youtube_music_search
    ↓
результаты поиска
    ↓
search_result_policy = rerank_results
    ↓
best_track = подходящая live-версия
```

Если результатов нет, модель выбирает `no_result`.

Laya не генерирует произвольный текст аргумента инструмента. Поэтому окружающий код передаёт ей подготовленный `search_query` в `state.parsed_request`. Модель решает, нужно ли вызывать `youtube_music_search`, а затем как обработать полученные результаты.

## Формат

Каждая строка JSONL содержит `state`, `questions`, `gold`, `language` и `tags`.

Поля `state`, `questions` и `gold` специально хранятся как JSON-строки. Это совпадает с формой данных, которую официальный fine-tuning notebook Laya разбирает через `json.loads(...)`.

```json
{
  "state": "{\"message\":\"включи Numb от Linkin Park\"}",
  "questions": "{\"intent\":{...}}",
  "gold": "{\"intent\":{\"probabilities\":{...}}}",
  "language": "ru",
  "tags": ["music-bot", "intent", "play"]
}
```

Gold содержит распределение вероятностей. Для однозначных синтетических примеров используется one-hot target.

## Сплиты

- `data/train-*.jsonl` для обучения.
- `data/calibration.jsonl` только для post-hoc temperature calibration.
- `data/eval.jsonl` только для финальной проверки.

Не смешивайте calibration/eval с train.

## Использование в notebook

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "train": "laya_dataset/data/train-*.jsonl",
        "calibration": "laya_dataset/data/calibration.jsonl",
        "test": "laya_dataset/data/eval.jsonl",
    },
)

ds_train = ds["train"]
```

Вместо английского checkpoint используйте multilingual:

```python
MODEL_ID = "convaiinnovations/laya-multilingual"
```

Дальше `build_training_item` из официального notebook может читать `state`, `questions` и `gold` без изменения схемы.

## Почему есть rerank

Сейчас `YtdlpClient.search()` использует `ytsearch1:`, то есть берёт первый результат. Датасет `search-rerank` готовит Laya к более полезной схеме: запросить несколько результатов, передать их как criteria и выбрать самый подходящий с учётом original/live/remix/cover/instrumental.

## Проверка

```bash
node laya_dataset/validate.mjs
```

Валидатор проверяет JSONL, вложенный JSON, соответствие `questions` и `gold`, сумму вероятностей и отсутствие точных дубликатов state между train/calibration/eval.

## Ограничение

Это стартовый синтетический набор. После интеграции Laya стоит добавлять анонимизированные реальные запросы и ошибки модели как hard cases/on-policy corrections.
