import type { ChatInputCommandInteraction, Client, SlashCommandBuilder, SlashCommandOptionsOnlyBuilder } from 'discord.js';
import type { GuildQueue } from './utils/GuildQueue.js';
import type { YtdlpClient } from './utils/ytdlp.js';

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
}

export interface Command {
  data: SlashCommandBuilder | SlashCommandOptionsOnlyBuilder;
  execute(interaction: ChatInputCommandInteraction, client: MusicClient): Promise<unknown>;
}
