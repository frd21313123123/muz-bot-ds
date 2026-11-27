#!/usr/bin/env node
// Small local bridge client for the control panel. It pairs a device with a
// code from the website and sends spoken commands to the bot. The Alice
// project can call the same HTTP endpoints directly; this script is handy for
// testing and for shell/MCP wrappers.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const base = (process.env.ALICE_BRIDGE_URL ?? 'http://127.0.0.1:8787').replace(/\/+$/u, '');
const tokenFile = path.resolve(process.env.ALICE_BRIDGE_TOKEN_FILE ?? '.runtime/alice-bridge.json');
const [command, ...args] = process.argv.slice(2);

async function readToken() {
  try {
    const data = JSON.parse(await readFile(tokenFile, 'utf8'));
    if (typeof data?.token === 'string' && data.token) return data.token;
  } catch {}
  throw new Error(`Нет токена. Сначала выполните: npm run alice:bridge -- pair <код> "Название"`);
}

async function request(pathname, options = {}) {
  const token = options.token;
  const response = await fetch(`${base}${pathname}`, {
    method: options.method ?? 'GET',
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  let data = {};
  try { data = await response.json(); } catch {}
  if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
  return data;
}

function usage() {
  console.log(`Использование:
  npm run alice:bridge -- pair <код> ["Название"]
  npm run alice:bridge -- say "<команда>"
  npm run alice:bridge -- state
  npm run alice:bridge -- heartbeat

Адрес пульта: ALICE_BRIDGE_URL (сейчас ${base})
Токен: ${tokenFile}`);
}

async function main() {
  if (!command || command === 'help' || command === '--help') { usage(); return; }
  if (command === 'pair') {
    const code = args[0];
    if (!code) throw new Error('Укажите код с сайта.');
    const name = args.slice(1).join(' ') || 'Колонка';
    const data = await request('/api/alice/pair', { method: 'POST', body: { code, name } });
    await mkdir(path.dirname(tokenFile), { recursive: true });
    await writeFile(tokenFile, `${JSON.stringify({ url: base, token: data.token, speaker: data.speaker }, null, 2)}\n`, {
      encoding: 'utf8', mode: 0o600,
    });
    console.log(`Подключено: ${data.speaker?.name ?? name}. Токен сохранён в ${tokenFile}.`);
    return;
  }
  const token = await readToken();
  if (command === 'say' || command === 'command') {
    const text = args.join(' ').trim();
    if (!text) throw new Error('Укажите текст команды.');
    const data = await request('/api/alice/command', { method: 'POST', token, body: { text } });
    console.log(data.text ?? 'Готово.');
    return;
  }
  if (command === 'state') {
    const data = await request('/api/alice/state', { token });
    console.log(JSON.stringify(data, null, 2));
    return;
  }
  if (command === 'heartbeat') {
    await request('/api/alice/heartbeat', { method: 'POST', token });
    console.log('ok');
    return;
  }
  usage();
  process.exitCode = 1;
}

main().catch((error) => {
  console.error(`[alice-bridge] ${error instanceof Error ? error.message : error}`);
  process.exitCode = 1;
});
