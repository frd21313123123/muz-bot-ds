import 'dotenv/config';
import { Client, Events, GatewayIntentBits } from 'discord.js';
import type { MusicClient } from './types.js';
import { onInteraction } from './events/interactionCreate.js';
import { prepareYtdlp } from './utils/ytdlp.js';
import { requireFfmpeg } from './utils/stream.js';

async function main(): Promise<void> {
  if (!process.env.DISCORD_TOKEN) throw new Error('Укажите DISCORD_TOKEN в .env');
  requireFfmpeg();
  const ytdlp = await prepareYtdlp();
  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
  }) as MusicClient;
  client.queues = new Map();
  client.ytdlp = ytdlp;
  client.once(Events.ClientReady, () => {
    console.log(`✅ Бот запущен как ${client.user?.tag}`);
    client.user?.setActivity('/play');
  });
  client.on(Events.InteractionCreate, (interaction) => void onInteraction(interaction, client));
  const shutdown = async (): Promise<void> => {
    await Promise.allSettled([...client.queues.values()].map((queue) => queue.stop()));
    client.destroy();
  };
  process.once('SIGINT', () => void shutdown().then(() => { process.exitCode = 0; }));
  process.once('SIGTERM', () => void shutdown().then(() => { process.exitCode = 0; }));
  process.on('unhandledRejection', (error) => console.error('[UnhandledRejection]', error));
  await client.login(process.env.DISCORD_TOKEN);
}

main().catch((error) => {
  console.error('Не удалось запустить бота:', error);
  process.exitCode = 1;
});
