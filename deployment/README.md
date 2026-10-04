# Linux / systemd

Бот установлен в `/srv/muz-bot-ds` под отдельным пользователем `muzbot`.
Node.js находится в `.runtime/node`; системный Node.js других приложений не меняется.
Токен хранится только в `.env` с правами `600`.

На сервере с 1 CPU / 1 ГиБ RAM используется профиль `light`, Whisper `small`,
`VOICE_CPU_THREADS=1` и готовые ответы `VOICE_TTS_ENGINE=bundled`.
Модель загружается при подключении к голосовому каналу и выгружается через
60 секунд после отключения прослушивания. Для коротких команд лёгкий профиль
использует окно от 15 секунд с увеличением для длинной записи; тяжёлый сохраняет полное окно. Прогрев использует такое же окно.
На этом VPS также используются заранее преобразованные INT8-веса из `.runtime/whisper-small-int8`,
заданные через `WHISPER_LIGHT_MODEL_PATH`. Подключение производится к каналу
пользователя, вызвавшего команду; при старте службы бот в канал не входит.

`muz-bot.service` предполагает указанное расположение проекта и Node.js.
После подготовки зависимостей и сборки:

```bash
sudo install -m 644 deployment/muz-bot.service /etc/systemd/system/muz-bot.service
sudo systemd-analyze verify /etc/systemd/system/muz-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now muz-bot
sudo systemctl status muz-bot
sudo journalctl -u muz-bot -n 40 --no-pager
```

Обновление: остановить службу, обновить исходники, выполнить `npm ci` и
`npm run build` от пользователя `muzbot`, затем запустить службу.
После изменения slash-команд также выполнить `npm run deploy`.
Для смены модели изменить `WHISPER_LIGHT_MODEL` в `.env`, остановить службу,
выполнить `npm run setup:voice` и запустить службу. Профиль `heavy` требует
отдельной установки окружения и значительно больше памяти и места.

Чтобы подготовить компактные веса на другом компьютере с уже установленными Torch,
Transformers и CTranslate2, выполните из корня проекта:

```bash
python scripts/convert-whisper-int8.py --model small --output .runtime/whisper-small-int8
```

Перенесите полученный каталог на сервер, задайте
`WHISPER_LIGHT_MODEL_PATH=.runtime/whisper-small-int8` в `.env` и выполните
`npm run setup:voice`. Конвертация использует официальный `openai/whisper-small`
и [API CTranslate2](https://github.com/SYSTRAN/faster-whisper#model-conversion).
Для работы готовой модели серверу Torch и Transformers не нужны.
При переходе на `base` или `tiny` удалите эту настройку либо подготовьте
соответствующие веса и укажите их каталог. Подготовленный путь сохраняется в
манифесте; после изменения `.env` требуется повторить `setup:voice`.

Комплектный Linux `ffmpeg-static` 7.0.2 аварийно завершался при открытии HTTPS.
Для этого сервера установлен отдельный [FFmpeg BtbN](https://github.com/BtbN/FFmpeg-Builds/releases)
8.1 в `.runtime/ffmpeg-linux`, архив проверен по SHA256.
`FFMPEG_PATH` указывает на `.runtime/ffmpeg-linux/bin/ffmpeg`.
Проверка сертификатов остаётся включённой.

Служба автоматически перезапускается при сбое. Её файловая система доступна
для записи только в `.runtime`, а дочерние процессы завершаются вместе со службой.
