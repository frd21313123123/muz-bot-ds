import 'dotenv/config';
import { Client, Events, GatewayIntentBits } from 'discord.js';
import type { MusicClient } from './types.js';
import { onInteraction } from './events/interactionCreate.js';
import { prepareYtdlp } from './utils/ytdlp.js';
import { requireFfmpeg } from './utils/stream.js';
import { VoiceRuntime } from './voice/runtime.js';
import { VoiceSettings } from './voice/settings.js';
import { LocalTts } from './voice/tts.js';
import { VoiceTrainingLog } from './voice/training.js';
import { VoiceDriveArchive } from './voice/drive.js';

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
  client.voiceRuntime = new VoiceRuntime({ autoRestart: true, startupTimeoutMs: 180_000 });
  client.voiceRuntime.on('failure', (event) => console.error('[Voice] Сбой обработчика:', JSON.stringify({ at: new Date().toISOString(), ...event })));
  client.voiceRuntime.on('recovering', (event) => console.error('[Voice] Перезапуск обработчика:', JSON.stringify(event)));
  client.voiceTrainingLog = new VoiceTrainingLog();
  client.voiceDriveArchive = new VoiceDriveArchive();
  client.voiceDriveArchive.start();
  if (client.voiceDriveArchive.enabled) console.log(client.voiceDriveArchive.configured
    ? '[Drive] Сохранение голосовых команд в WAV и загрузка в Google Drive включены.'
    : '[Drive] WAV сохраняются локально; для загрузки настройте GOOGLE_DRIVE_CLIENT_ID, GOOGLE_DRIVE_CLIENT_SECRET и GOOGLE_DRIVE_REFRESH_TOKEN.');
  if (client.voiceTrainingLog.enabled) console.log('[NLI] Учебный журнал команд включён: .runtime/nli (текст, без аудио).');
  const tts = new LocalTts();
  client.voiceTts = tts;
  console.log(await tts.load() ? `[TTS] ${tts.voice} готов.` : '[TTS] Озвучивание выключено; выполните npm run setup:tts.');
  console.log(`[Voice] Профиль ${client.voiceRuntime.profile}; STT ${client.voiceRuntime.sttProvider}; обработчик запустится при подключении к голосовому каналу.`);
  client.voiceRuntime.on('available', () => console.log(`[Voice] Whisper ${client.voiceRuntime!.sttModelName ?? '?'} (wake: ${client.voiceRuntime!.wakeModelName ?? '?'}, device: ${client.voiceRuntime!.inferenceDevice ?? '?'}) + ${client.voiceRuntime!.modelName ?? '?'} готовы.`));
  client.voiceRuntime.on('unavailable', () => console.error('[Voice] Обработчик остановлен; прослушивание временно выключено. Музыка продолжает работать.'));
  client.once(Events.ClientReady, () => {
    console.log('[Bot] Ready');
    console.log(`✅ Бот запущен как ${client.user?.tag}`);
    client.user?.setActivity('/play');
  });
  client.on(Events.InteractionCreate, (interaction) => void onInteraction(interaction, client));
  client.on(Events.VoiceStateUpdate, (oldState, newState) => {
    if (oldState.channelId !== newState.channelId) {
      client.queues.get(oldState.guild.id)?.cancelRadioRequest(oldState.id === client.user?.id ? undefined : oldState.id);
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
    await client.voiceTrainingLog?.flush();
    await client.voiceDriveArchive?.close();
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
