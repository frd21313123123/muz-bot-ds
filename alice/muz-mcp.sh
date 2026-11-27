#!/bin/sh
# Minimal stdio MCP server for Yandex Station Mini (BusyBox sh + wget).
# Tool: discord_music — sends a spoken command to the muz-bot control panel.
#
# Install on the station:
#   /data/opt/quasar-bridge/mcp/muz-mcp.sh   (chmod 755, LF line endings)
#   /data/opt/quasar-bridge/mcp/muz-bot.env  (chmod 600)
# with:
#   MUZ_BOT_URL=http://192.168.1.160:8787
#   MUZ_BOT_TOKEN=<token from POST /api/alice/pair>
#
# The script follows the same newline-delimited JSON-RPC convention as
# search-mcp/ddg-search.sh in the Alice project.

CONF="${MUZ_BRIDGE_ENV:-/data/opt/quasar-bridge/mcp/muz-bot.env}"
[ -f "$CONF" ] && . "$CONF"
MUZ_BOT_URL=$(printf '%s' "${MUZ_BOT_URL:-http://192.168.1.160:8787}" | tr -d '\r')
MUZ_BOT_URL=$(printf '%s' "$MUZ_BOT_URL" | sed 's:/*$::')
MUZ_BOT_TOKEN=$(printf '%s' "${MUZ_BOT_TOKEN:-}" | tr -d '\r')
TMP="${TMPDIR:-/tmp}/muz-mcp.$$"
trap 'rm -f "$TMP"' EXIT

send() { printf '%s\n' "$1"; }

# JSON field extractors for compact newline-delimited requests.
jstr() { printf '%s' "$2" | sed -nE "s/.*\"$1\" *: *\"((\\\\.|[^\"\\\\])*)\".*/\\1/p"; }
jid()  { printf '%s' "$1" | sed -nE 's/.*"id" *: *("([^"\\]|\\.)*"|-?[0-9]+).*/\1/p'; }

junescape() {
  printf '%s' "$1" | sed -e 's/\\u0026/\&/g' -e 's/\\u003c/</g' -e 's/\\u003e/>/g' \
    -e 's/\\"/"/g' -e 's/\\\//\//g' -e 's/\\[nrt]/ /g' -e 's/\\\\/\\/g'
}

jescape() {
  awk 'BEGIN { ORS = "" }
    { gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); gsub(/\t/, " "); gsub(/\r/, "")
      if (NR > 1) print "\\n"; print }'
}

TOOL='{"name":"discord_music","description":"Управляет музыкой в Discord через пульт бота: включи трек или ссылку, включи радио, пауза, продолжай, следующий трек, стоп, громкость, что играет. Передавай текст команды так, как его сказал пользователь, без обращения «Алиса» и без слов «в дискорде» / «в боте».","inputSchema":{"type":"object","properties":{"command":{"type":"string","description":"Команда для пульта: например «включи Кино», «следующий трек», «громкость 40», «что играет»"}},"required":["command"]},"annotations":{"readOnlyHint":false,"openWorldHint":false}}'

call_bot() {
  cmd=$1
  [ -n "$MUZ_BOT_TOKEN" ] || { printf '%s' 'Мост не настроен: в muz-bot.env нет MUZ_BOT_TOKEN.'; return 1; }
  [ -n "$MUZ_BOT_URL" ] || { printf '%s' 'Мост не настроен: в muz-bot.env нет MUZ_BOT_URL.'; return 1; }
  body=$(printf '%s' "{\"text\":\"$(printf '%s' "$cmd" | jescape)\"}")
  rm -f "$TMP"
  wget -q -T 20 \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $MUZ_BOT_TOKEN" \
    --post-data "$body" -O "$TMP" "$MUZ_BOT_URL/api/alice/command" 2>/dev/null || true
  [ -s "$TMP" ] || { printf '%s' 'Пульт не ответил. Проверьте адрес и что бот запущен.'; return 1; }
  reply=$(junescape "$(jstr text "$(cat "$TMP")")")
  if grep -q '"ok":true' "$TMP"; then
    [ -n "$reply" ] && printf '%s' "$reply" || printf '%s' 'Готово.'
    return 0
  fi
  [ -n "$reply" ] && printf '%s' "$reply" || printf '%s' 'Не удалось выполнить команду.'
  return 1
}

while IFS= read -r line; do
  [ -n "$line" ] || continue
  method=$(jstr method "$line")
  id=$(jid "$line")
  case "$method" in
    initialize)
      ver=$(jstr protocolVersion "$line"); [ -n "$ver" ] || ver=2025-06-18
      send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"$ver\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"muz-discord\",\"version\":\"1.0.0\"}}}"
      ;;
    tools/list)
      send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[$TOOL]}}"
      ;;
    tools/call)
      name=$(jstr name "$line")
      if [ "$name" != discord_music ]; then
        send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"error\":{\"code\":-32602,\"message\":\"unknown tool\"}}"
        continue
      fi
      cmd=$(junescape "$(jstr command "$line")")
      if [ -z "$cmd" ]; then
        text='Пустая команда.'
        err=true
      elif text=$(call_bot "$cmd"); then
        err=false
      else
        err=true
      fi
      body=$(printf '%s' "$text" | jescape)
      send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"$body\"}],\"isError\":$err}}"
      ;;
    ping)
      send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{}}"
      ;;
    *)
      [ -n "$id" ] && send "{\"jsonrpc\":\"2.0\",\"id\":$id,\"error\":{\"code\":-32601,\"message\":\"method not found\"}}"
      ;;
  esac
done
