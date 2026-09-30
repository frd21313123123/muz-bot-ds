import 'dotenv/config';
import { Client, Events, GatewayIntentBits } from 'discord.js';
import type { MusicClient } from './types.js';
import { onInteraction } from './events/interactionCreate.js';
import { prepareYtdlp } from './utils/ytdlp.js';
import { requireFfmpeg } from './utils/stream.js';
import { VoiceRuntime } from './voice/runtime.js';
import { VoiceSettings } from './voice/settings.js';

async function main(): Promise<void> {
  if (!process.env.DISCORD_TOKEN) throw new Error('Укажите DISCORD_TOKEN в .env');
  requireFfmpeg();
  const ytdlp = await prepareYtdlp();
  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
  }) as MusicClient;
  client.queues = new Map();
  client.ytdlp = ytdlp;
  client.voiceSettings = new VoiceSettings();
  await client.voiceSettings.load().catch(() => console.error('[Voice] Не удалось прочитать настройки имени.'));
  client.voiceRuntime = new VoiceRuntime();
  const voiceReady = await client.voiceRuntime.start().catch(() => false);
  console.log(voiceReady ? '[Voice] Whisper + Laya готовы.' : '[Voice] Недоступно; выполните npm run setup:voice. Музыка работает без голосовых команд.');
  client.voiceRuntime.on('unavailable', () => console.error('[Voice] Обработчик остановлен; повторное включение через /voice on.'));
  client.once(Events.ClientReady, () => {
    console.log(`✅ Бот запущен как ${client.user?.tag}`);
    client.user?.setActivity('/play');
  });
  client.on(Events.InteractionCreate, (interaction) => void onInteraction(interaction, client));
  client.on(Events.VoiceStateUpdate, (oldState, newState) => {
    if (oldState.channelId !== newState.channelId) {
      const voice = client.queues.get(oldState.guild.id)?.voice;
      if (oldState.id === client.user?.id) voice?.reset(); else voice?.cancelUser(oldState.id);
    }
  });
  client.on(Events.GuildMemberUpdate, (oldMember, newMember) => {
    if (newMember.id === client.user?.id && oldMember.displayName !== newMember.displayName) {
      client.queues.get(newMember.guild.id)?.voice?.reset();
    }
  });
  const shutdown = async (): Promise<void> => {
    await Promise.allSettled([...client.queues.values()].map((queue) => queue.stop()));
    client.voiceRuntime?.close();
    client.destroy();
  };
  process.once('SIGINT', () => void shutdown().then(() => { process.exitCode = 0; }));
  process.once('SIGTERM', () => void shutdown().then(() => { process.exitCode = 0; }));
  process.on('unhandledRejection', (error) => console.error('[UnhandledRejection]', error));
  try { await client.login(process.env.DISCORD_TOKEN); }
  catch (error) { await shutdown(); throw error; }
}

main().catch((error) => {
  console.error('Не удалось запустить бота:', error);
  process.exitCode = 1;
});
