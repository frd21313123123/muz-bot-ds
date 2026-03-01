# 🎵 muz-bot-ds — Discord Music Bot

Быстрый музыкальный бот для Discord с поддержкой YouTube и YouTube Music.

## Возможности

| Команда | Описание |
|---|---|
| `/play <запрос>` | Воспроизвести трек / плейлист. Принимает ссылки YouTube, YouTube Music, текстовый поиск |
| `/skip` | Пропустить текущий трек |
| `/stop` | Остановить воспроизведение и покинуть канал |
| `/pause` | Пауза |
| `/resume` | Продолжить |
| `/queue [page]` | Показать очередь |
| `/nowplaying` | Показать текущий трек |
| `/volume <1-150>` | Установить громкость |
| `/autoplay` | Вкл/выкл авто-рекомендации YouTube |
| `/clear` | Очистить очередь |

## Поддерживаемые форматы ссылок

```
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://music.youtube.com/watch?v=VIDEO_ID
https://music.youtube.com/watch?v=VIDEO_ID&list=RDAMVM...
https://www.youtube.com/playlist?list=PLAYLIST_ID
```

## Установка

### 1. Требования

- **Node.js ≥ 18** — https://nodejs.org/
- **FFmpeg** — устанавливается автоматически через пакет `ffmpeg-static`

### 2. Клонировать / скопировать проект

```bash
cd muz-bot-ds
npm install
```

> **Windows:** если `@discordjs/opus` не устанавливается, добавьте в package.json и выполните:
> ```bash
> npm install opusscript
> ```

### 3. Настройить .env

```bash
cp .env.example .env
```

Откройте `.env` и вставьте:

```
DISCORD_TOKEN=токен_бота
CLIENT_ID=id_приложения
# GUILD_ID=id_сервера  # только для быстрой отладки
```

Получить токен: https://discord.com/developers/applications → ваше приложение → **Bot → Reset Token**

### 4. Зарегистрировать команды

```bash
npm run deploy
```

> С `GUILD_ID` команды появятся мгновенно.
> Без `GUILD_ID` — глобально, до 1 часа.

### 5. Запустить бота

```bash
npm start
# или для разработки с авто-перезапуском:
npm run dev
```

## Разрешения бота в Discord

Необходимые привилегии OAuth2:
- `bot` + `applications.commands`

Bot permissions:
- Connect
- Speak
- Send Messages
- Embed Links

Gateway Intents (в панели разработчика → Bot):
- `GUILDS`
- `GUILD_VOICE_STATES`

## Автовоспроизведение

После включения `/autoplay` бот, когда очередь закончится, автоматически подбирает следующий трек из рекомендаций YouTube (на основе текущего трека).

## Архитектура

```
src/
├── index.js               — точка входа, загрузка команд/событий
├── deploy-commands.js     — регистрация слэш-команд в Discord API
├── commands/              — по одному файлу на каждую команду
├── events/
│   ├── ready.js
│   └── interactionCreate.js
└── utils/
    ├── GuildQueue.js      — управление очередью на сервер
    ├── embeds.js          — билдеры Embed
    └── resolve.js         — разрешение URL / поискового запроса в треки
```
