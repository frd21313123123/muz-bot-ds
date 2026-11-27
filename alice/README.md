# Мост Alice ↔ muz-bot

Готовый MCP-инструмент `discord_music` для `quasar-bridge` на Яндекс Станции.
Он позволяет не активировать навык Яндекс.Диалогов: станция передаёт распознанную
фразу в пульт бота, а бот включает трек в Discord.

Схема:

```text
Алиса на станции
  → quasar-bridge (LLM + MCP)
  → muz-mcp.sh
  → POST http://<ПК>:8787/api/alice/command
  → Discord-бот (очередь, голосовой канал)
```

Стандартный путь через навык Яндекс.Диалогов описан в основном
[README](../README.md#навык-яндексдиалогов). Этот мост нужен, если хочется
говорить «Алиса, включи Кино в дискорде» без «запусти навык».

## 1. Подготовьте пульт

На ПК с ботом в `.env`:

```dotenv
WEB_ENABLED=1
WEB_HOST=0.0.0.0
WEB_PORT=8787
WEB_ADMIN_PASSWORD=<длинный пароль>
```

`0.0.0.0` нужен, чтобы станция (`192.168.1.172`) видела пульт. Не публикуйте
порт 8787 в интернет; доступ по локальной сети/через VPN. На Windows разрешите
входящий TCP 8787 для частной сети (брандмауэр) или используйте туннель.

Запустите бота, откройте пульт и создайте код для нужного сервера и канала.
Затем на ПК выполните:

```bash
npm run alice:bridge -- pair 123456 "station-mini"
```

Или сохраните токен вручную:

```powershell
$pair = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8787/api/alice/pair' `
  -ContentType 'application/json' -Body '{"code":"123456","name":"station-mini"}'
$pair.token
```

Токен показывается один раз. Он привязан к каналу; при удалении колонки на
сайте перестаёт работать.

## 2. Установите скрипт на станцию

Файл [`muz-mcp.sh`](muz-mcp.sh) — BusyBox-совместимый stdio MCP-сервер. Он
использует только `sh`, `sed`, `awk` и `wget`, как `search-mcp` из проекта
Alice.

```bash
scp -l 4000 alice/muz-mcp.sh station-mini:/data/opt/quasar-bridge/mcp/muz-mcp.sh
ssh station-mini 'chmod 755 /data/opt/quasar-bridge/mcp/muz-mcp.sh; sed -i "s/\r$//" /data/opt/quasar-bridge/mcp/muz-mcp.sh; md5sum /data/opt/quasar-bridge/mcp/muz-mcp.sh'
```

`scp` на этой станции иногда возвращает код 1 — сверяйте размер и `md5sum`.

Создайте файл конфигурации (токен передаётся через stdin, а не аргументом
командной строки):

```bash
ssh station-mini 'umask 077; cat > /data/opt/quasar-bridge/mcp/muz-bot.env' <<'EOF'
MUZ_BOT_URL=http://192.168.1.160:8787
MUZ_BOT_TOKEN=<токен из шага 1>
EOF
```

Проверка без голоса (команда должна вернуть JSON-RPC с `isError:false`):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"discord_music","arguments":{"command":"что играет"}}}' \
| ssh station-mini 'sh /data/opt/quasar-bridge/mcp/muz-mcp.sh'
```

## 3. Подключите MCP в quasar-bridge

В `/data/opt/quasar-bridge/config.json` добавьте сервер в
`assistant_tools.mcp_servers` (сохраните бэкап):

```json
{
  "name": "muz",
  "transport": "stdio",
  "command": "/bin/sh",
  "args": ["/data/opt/quasar-bridge/mcp/muz-mcp.sh"],
  "timeout_ms": 25000
}
```

Проверка конфигурации и инструментов:

```bash
ssh station-mini 'cd /data/opt/quasar-bridge && sh app/start.sh -check-config && sh app/start.sh -check-mcp'
```

В списке должен появиться `discord_music`.

## 4. Маршрутизация фраз

`quasar-bridge` должен отправлять фразы про Discord в LLM, а не в Яндекс.
В `routing.rules` добавьте правило **перед** `yandex_basic` (синтаксис зависит
от версии bridge; ориентируйтесь на существующие правила):
```json
{
  "name": "music_to_bot",
  "route": "llm",
  "match": {
    "contains_any": ["в дискорде", "в боте", "дискорд", "через бота"]
  }
}
```

Нужен режим bridge, в котором включены `assistant_tools` и LLM; в `observe`
инструменты не вызываются. Проверка конфигурации: `sh app/start.sh -check-config`.

Затем перезапустите bridge или примените конфиг способом из
`F:\programing 2\Alice\ALICE_INFO.md`. Проверка маршрута:

```bash
ssh station-mini 'cd /data/opt/quasar-bridge && sh app/start.sh -route "включи Кино в дискорде"'
```

Должно быть `llm`. После этого скажите станции:

> «Алиса, включи Кино в дискорде»

LLM вызывает `discord_music`, скрипт отправляет текст в пульт, и трек
включается в привязанном канале. Ответ инструмента — короткая фраза из пульта
(«Включаю: Кино», «Следующий трек» и т.п.); LLM пересказывает её.

Если хочется обойтись без LLM, можно вызвать API из собственного скрипта
станции/`alice-station`; формат описан в основном README, раздел
[«Локальный мост»](../README.md#локальный-мост-для-проекта-alice).

## Безопасность и откат

- `muz-bot.env` — только `chmod 600`; токен не печатается в логах пульта.
- Станция должна иметь доступ только к `/api/alice/*`; админ-панель на
  `0.0.0.0` защищена паролем, но лучше ограничить её firewall'ом.
- Удалите колонку на сайте — токен сразу перестанет приниматься.
- Откат: уберите сервер `muz` из `mcp_servers` и правило `music_to_bot`,
  перезапустите bridge. Файлы на станции можно оставить.
- Проект Alice сейчас может быть в stock mode: сначала верните кастомные
  сервисы по `F:\programing 2\Alice\stock-mode\README.md`, как и для любого
  другого патча. Бот и пульт при этом менять не нужно.
