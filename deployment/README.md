# Linux / systemd

Бот установлен в `/srv/muz-bot-ds` под отдельным пользователем `muzbot`.
Node.js находится в `.runtime/node`; системный Node.js других приложений не меняется.
Токен хранится только в `.env` с правами `600`.

Для сервера с 1 CPU / 1 ГиБ RAM используйте `VOICE_PROFILE=light`,
`STT_PROVIDER=groq`, `GROQ_API_KEY` и готовые ответы `VOICE_TTS_ENGINE=bundled`.
Groq `whisper-large-v3-turbo` распознаёт имя и команды без загрузки локальных
весов и без процесса Python STT. Нужен исходящий HTTPS к `api.groq.com`;
ключ хранится только в `.env` с правами `600`. Короткие записи речи участников
при включённом прослушивании отправляются в Groq. `npm run test:groq` проверяет
API на двух комплектных синтетических фразах. Подключение производится к каналу
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
Для перехода существующей установки на облачную STT задайте `STT_PROVIDER=groq`
и `GROQ_API_KEY` в `.env`, обновите исходники, выполните сборку и перезапустите
службу. Для профиля `light` подготовка Python STT не нужна. Профиль `heavy`
продолжает использовать локальную Laya и требует отдельной подготовки окружения.

Следующие инструкции нужны только для `STT_PROVIDER=local`:

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
