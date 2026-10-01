import type { ChatInputCommandInteraction, Client, SlashCommandBuilder, SlashCommandOptionsOnlyBuilder, SlashCommandSubcommandsOnlyBuilder } from 'discord.js';
import type { GuildQueue } from './utils/GuildQueue.js';
import type { YtdlpClient } from './utils/ytdlp.js';
import type { VoiceRuntime } from './voice/runtime.js';
import type { VoiceSettings } from './voice/settings.js';
import type { VoiceTts } from './voice/tts.js';
import type { VoiceTrainingLog } from './voice/training.js';

export interface Track {
  url: string;
  videoId: string;
  title: string;
  duration: string;
  thumbnail: string | null;
  requestedBy: string;
  isAutoplay?: boolean;
}

export interface MusicClient extends Client {
  queues: Map<string, GuildQueue>;
  ytdlp: YtdlpClient;
  voiceRuntime?: VoiceRuntime;
  voiceSettings?: VoiceSettings;
  voiceTts?: VoiceTts;
  voiceTrainingLog?: VoiceTrainingLog;
}

export interface Command {
  data: SlashCommandBuilder | SlashCommandOptionsOnlyBuilder | SlashCommandSubcommandsOnlyBuilder;
  execute(interaction: ChatInputCommandInteraction, client: MusicClient): Promise<unknown>;
}
