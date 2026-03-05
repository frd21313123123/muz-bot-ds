'use strict';

// Установить путь к ffmpeg-static до всех импортов
const ffmpegPath = require('ffmpeg-static');
process.env.FFMPEG_PATH = ffmpegPath;
const dns = require('dns');

// On some hosts IPv6 is present in DNS answers but unreachable.
// Force IPv4-first resolution so Discord voice endpoints connect reliably.
try {
  dns.setDefaultResultOrder('ipv4first');
} catch (_) {}

require('dotenv').config();

const { Client, GatewayIntentBits, Collection } = require('discord.js');
const fs = require('fs');
const path = require('path');

// ─── Discord Client ─────────────────────────────────────────────────────────
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
  ],
});

/** @type {Collection<string, import('./types').Command>} */
client.commands = new Collection();

/** @type {Map<string, import('./utils/GuildQueue')>} */
client.queues = new Map();

// ─── Загрузка команд ────────────────────────────────────────────────────────
const commandsDir = path.join(__dirname, 'commands');
for (const file of fs.readdirSync(commandsDir).filter(f => f.endsWith('.js'))) {
  const command = require(path.join(commandsDir, file));
  if (command.data && command.execute) {
    client.commands.set(command.data.name, command);
  }
}

// ─── Загрузка событий ───────────────────────────────────────────────────────
const eventsDir = path.join(__dirname, 'events');
for (const file of fs.readdirSync(eventsDir).filter(f => f.endsWith('.js'))) {
  const event = require(path.join(eventsDir, file));
  const fn = (...args) => event.execute(...args, client);
  event.once ? client.once(event.name, fn) : client.on(event.name, fn);
}

// ─── Глобальная обработка ошибок ────────────────────────────────────────────
process.on('unhandledRejection', (error) => {
  console.error('[UnhandledRejection]', error);
});

process.on('uncaughtException', (error) => {
  console.error('[UncaughtException]', error);
});

// ─── Запуск ─────────────────────────────────────────────────────────────────
client.login(process.env.DISCORD_TOKEN).catch((err) => {
  console.error('Не удалось войти в Discord:', err.message);
  process.exit(1);
});
