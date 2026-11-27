import 'dotenv/config';
import { Client, Events, GatewayIntentBits, Options } from 'discord.js';
import type { MusicClient } from './types.js';
import { onInteraction } from './events/interactionCreate.js';
import { prepareYtdlp } from './utils/ytdlp.js';
import { requireFfmpeg } from './utils/stream.js';
import { VoiceRuntime } from './voice/runtime.js';
import { VoiceSettings } from './voice/settings.js';
import { LocalTts } from './voice/tts.js';
import { VoiceTrainingLog } from './voice/training.js';
import { VoiceDriveArchive } from './voice/drive.js';
import { startPerformanceMonitor } from './utils/performance.js';
import { startControlServer, type ControlServer } from './web/server.js';

async function main(): Promise<void> {
  const stopPerformanceMonitor = startPerformanceMonitor();
  if (!process.env.DISCORD_TOKEN) throw new Error('Укажите DISCORD_TOKEN в .env');
  requireFfmpeg();
  const ytdlp = await prepareYtdlp();
  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
    // The Guilds intent delivers emojis, stickers, events and similar data the
    // bot never reads. Skip caching it; channels, roles, members and voice
    // states (used for permissions and voice control) keep default caching.
    makeCache: Options.cacheWithLimits({
      ...Options.DefaultMakeCacheSettings,
      ApplicationEmojiManager: 0, AutoModerationRuleManager: 0, BaseGuildEmojiManager: 0,
      EntitlementManager: 0, GuildBanManager: 0, GuildEmojiManager: 0, GuildInviteManager: 0,
      GuildScheduledEventManager: 0, GuildStickerManager: 0, PresenceManager: 0,
      ReactionManager: 0, ReactionUserManager: 0, StageInstanceManager: 0, ThreadMemberManager: 0,
    }),
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
    ? '[Drive] Голосовые команды сохраняются в WAV и отправляются в Google Drive через rclone.'
    : '[Drive] WAV сохраняются локально. Для загрузки укажите VOICE_AUDIO_RCLONE_REMOTE.');
  if (client.voiceTrainingLog.enabled) console.log('[NLI] Учебный журнал команд включён: .runtime/nli (текст, без аудио).');
  const tts = new LocalTts();
  client.voiceTts = tts;
  console.log(await tts.load() ? `[TTS] ${tts.voice} готов.` : '[TTS] Озвучивание выключено; выполните npm run setup:tts.');
  const prewarm = process.env.VOICE_PREWARM === '1' && client.voiceRuntime.profile === 'light'
    && client.voiceRuntime.sttProvider === 'groq';
  console.log(`[Voice] Профиль ${client.voiceRuntime.profile}; STT ${client.voiceRuntime.sttProvider}.`);
  client.voiceRuntime.on('available', () => console.log(`[Voice] Whisper ${client.voiceRuntime!.sttModelName ?? '?'} (wake: ${client.voiceRuntime!.wakeModelName ?? '?'}, device: ${client.voiceRuntime!.inferenceDevice ?? '?'}) + ${client.voiceRuntime!.modelName ?? '?'} готовы.`));
  client.voiceRuntime.on('unavailable', () => console.error('[Voice] Обработчик остановлен; прослушивание временно выключено. Музыка продолжает работать.'));
  let controlServer: ControlServer | null = null;
  try { controlServer = await startControlServer(client); }
  catch (error) { console.error('[Web] Не удалось запустить пульт:', error); }
  client.once(Events.ClientReady, () => {
    console.log('[Bot] Ready');
    console.log(`✅ Бот запущен как ${client.user?.tag}`);
    client.user?.setActivity('/play');
    if (prewarm && client.voiceRuntime?.prepared) {
      void client.voiceRuntime.acquire(client).catch(error => console.error('[Voice] Не удался прогрев:', error));
    }
  });
  client.on(Events.InteractionCreate, (interaction) => void onInteraction(interaction, client));
  client.on(Events.VoiceStateUpdate, (oldState, newState) => {
    client.queues.get(oldState.guild.id)?.onVoiceChannelChange(oldState.id, oldState.channelId, newState.channelId);
  });
  client.on(Events.GuildMemberUpdate, (oldMember, newMember) => {
    if (newMember.id === client.user?.id && oldMember.displayName !== newMember.displayName) {
      client.queues.get(newMember.guild.id)?.voice?.reset();
    }
  });
  const shutdown = async (): Promise<void> => {
    await controlServer?.close();
    await Promise.allSettled([...client.queues.values()].map((queue) => queue.stop()));
    stopPerformanceMonitor();
    await client.voiceRuntime?.close();
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
