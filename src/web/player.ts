import {
  ChannelType, PermissionsBitField,
  type Guild, type GuildMember, type VoiceChannel,
} from 'discord.js';
import type { MusicClient, Track } from '../types.js';
import { GuildQueue } from '../utils/GuildQueue.js';
import { findRadioStation, radioStations, type RadioStation } from '../utils/radio.js';
import { resolveQuery, type SearchSource } from '../utils/resolve.js';
import type { BotView, GuildView, QueueView, TrackView, WebTarget } from './types.js';

export type PlayerAction = 'pause' | 'resume' | 'skip' | 'stop' | 'volume' | 'autoplay' | 'loop' | 'clear';

export interface PlayResult {
  track: TrackView;
  started: boolean;
  playlist: boolean;
}

export interface ActResult {
  changed: boolean;
  level?: number;
  count?: number;
}

export interface PlayerSnapshot {
  bot: BotView;
  guilds: GuildView[];
  radioStations: { id: string; name: string }[];
  sampledAt: number;
}

export interface PlayerApi {
  play(target: WebTarget, query: string, requestedBy: string,
    options?: { source?: SearchSource; move?: boolean }): Promise<PlayResult>;
  radio(target: WebTarget, station: RadioStation, requestedBy: string,
    options?: { move?: boolean }): Promise<{ station: string }>;
  act(target: WebTarget, action: PlayerAction, value?: number | boolean): Promise<ActResult>;
  snapshot(): PlayerSnapshot;
}

function trackView(track: Track): TrackView {
  return {
    title: track.title,
    url: track.url,
    thumbnail: track.thumbnail,
    duration: track.duration,
    requestedBy: track.requestedBy,
    isAutoplay: Boolean(track.isAutoplay),
    isLive: Boolean(track.isLive),
    source: track.source ?? 'youtube',
  };
}

function queueView(queue: GuildQueue): QueueView {
  return {
    guildId: queue.guildId,
    channelId: queue.voiceChannel?.id ?? null,
    channelName: queue.voiceChannel?.name ?? null,
    playing: Boolean(queue.currentTrack) && !queue.isPaused,
    paused: queue.isPaused,
    radio: queue.isRadio,
    volume: Math.round(queue.volume * 100),
    speed: queue.speed,
    autoplay: queue.autoplay,
    loop: queue.loopCurrent,
    elapsed: Math.max(0, queue.getElapsedSeconds()),
    current: queue.currentTrack ? trackView(queue.currentTrack) : null,
    upcoming: queue.tracks.slice(0, 25).map(trackView),
    upcomingCount: queue.tracks.length,
    sampledAt: Date.now(),
  };
}

export class PlayerService implements PlayerApi {
  constructor(readonly client: MusicClient) {}

  async play(target: WebTarget, query: string, requestedBy: string,
    options: { source?: SearchSource; move?: boolean } = {}): Promise<PlayResult> {
    const trimmed = query.trim();
    if (!trimmed) throw new Error('Укажите название трека или ссылку.');
    if (trimmed.length > 500) throw new Error('Слишком длинный запрос.');
    const { channel } = await this.resolveTarget(target);
    let queue = this.client.queues.get(target.guildId);
    if (queue?.closed) queue = undefined;
    if (queue?.voiceChannel && queue.voiceChannel.id !== channel.id && !options.move) {
      throw new Error('Бот уже работает в другом голосовом канале.');
    }
    const created = !queue;
    if (!queue) {
      queue = new GuildQueue(target.guildId, this.client);
      // Playback from the panel starts music without Discord voice listening. Existing
      // queues keep whatever /join or /voice configured.
      await queue.setVoiceEnabled(false);
      this.client.queues.set(target.guildId, queue);
    }
    const request = queue.beginMetadataRequest('web');
    try {
      const result = await Promise.all([
        queue.join(channel),
        resolveQuery(trimmed, requestedBy, this.client.ytdlp, 1, {
          signal: request.signal, source: options.source ?? 'youtube',
        }),
      ]).then(([, resolved]) => resolved);
      if (queue.closed) throw new Error('Очередь остановлена.');
      if (!result) throw new Error('Ничего не найдено.');
      const wasPlaying = Boolean(queue.currentTrack) && !queue.isRadio;
      if (result.type === 'playlist') {
        const tracks = result.tracks.slice(0, 1);
        const first = tracks[0];
        if (!first) throw new Error('Ничего не найдено.');
        await queue.addTracks(tracks);
        return { track: trackView(first), started: !wasPlaying, playlist: result.tracks.length > 1 };
      }
      await queue.addTrack(result.track);
      return { track: trackView(result.track), started: !wasPlaying, playlist: false };
    } catch (error) {
      request.finish();
      if (created && !queue.closed && !queue.currentTrack && !queue.tracks.length && !queue.hasMetadataRequests) {
        await queue.stop().catch(() => {});
      }
      throw error;
    } finally {
      request.finish();
    }
  }

  async radio(target: WebTarget, station: RadioStation, requestedBy: string,
    options: { move?: boolean } = {}): Promise<{ station: string }> {
    const { channel } = await this.resolveTarget(target);
    let queue = this.client.queues.get(target.guildId);
    if (queue?.closed) queue = undefined;
    if (queue?.voiceChannel && queue.voiceChannel.id !== channel.id && !options.move) {
      throw new Error('Бот уже работает в другом голосовом канале.');
    }
    const created = !queue;
    if (!queue) {
      queue = new GuildQueue(target.guildId, this.client);
      await queue.setVoiceEnabled(false);
      this.client.queues.set(target.guildId, queue);
    }
    try {
      await queue.join(channel);
      const changed = await queue.playRadio(station, requestedBy, { valid: () => !queue.closed });
      if (!changed) throw new Error('Запрос радио отменён.');
      return { station: station.name };
    } catch (error) {
      if (created && !queue.closed && !queue.currentTrack && !queue.tracks.length) await queue.stop().catch(() => {});
      throw error;
    }
  }

  async act(target: WebTarget, action: PlayerAction, value?: number | boolean): Promise<ActResult> {
    const queue = this.client.queues.get(target.guildId);
    if (action === 'stop') {
      if (!queue) return { changed: false };
      await queue.stop();
      return { changed: true };
    }
    if (!queue || queue.closed) return { changed: false };
    switch (action) {
      case 'pause': return { changed: queue.pause() };
      case 'resume': return { changed: queue.resume() };
      case 'skip': return { changed: queue.skip() };
      case 'volume': {
        const level = typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : NaN;
        if (!Number.isFinite(level) || level < 1 || level > 150) throw new Error('Громкость должна быть от 1 до 150%.');
        queue.setVolume(level);
        return { changed: true, level: Math.round(queue.volume * 100) };
      }
      case 'autoplay': {
        const before = queue.autoplay;
        const enabled = queue.setAutoplay(typeof value === 'boolean' ? value : !before);
        return { changed: enabled !== before, level: enabled ? 1 : 0 };
      }
      case 'loop': {
        const before = queue.loopCurrent;
        const enabled = queue.setLoop(typeof value === 'boolean' ? value : !before);
        return { changed: enabled !== before, level: enabled ? 1 : 0 };
      }
      case 'clear': {
        const count = queue.clearQueue();
        return { changed: count > 0, count };
      }
      default:
        throw new Error('Неизвестное действие.');
    }
  }

  snapshot(): PlayerSnapshot {
    const guilds: GuildView[] = [...this.client.guilds.cache.values()]
      .sort((a, b) => a.name.localeCompare(b.name, 'ru'))
      .map((guild) => this.guildView(guild));
    const user = this.client.user;
    return {
      bot: {
        id: user?.id ?? null,
        tag: user?.tag ?? 'Бот запускается',
        avatar: user?.displayAvatarURL({ size: 64 }) ?? null,
        ready: Boolean(user),
        uptimeMs: Math.max(0, Math.round(process.uptime() * 1000)),
      },
      guilds,
      radioStations: radioStations.map(station => ({ id: station.id, name: station.name })),
      sampledAt: Date.now(),
    };
  }

  private guildView(guild: Guild): GuildView {
    const me = guild.members.me;
    const channels = [...guild.channels.cache.values()]
      .filter((channel): channel is VoiceChannel => channel.type === ChannelType.GuildVoice)
      .map((channel) => ({
        id: channel.id,
        name: channel.name,
        canJoin: Boolean(me && channel.permissionsFor(me)?.has([
          PermissionsBitField.Flags.Connect, PermissionsBitField.Flags.Speak,
        ])),
      }))
      .sort((a, b) => a.name.localeCompare(b.name, 'ru'));
    const queue = this.client.queues.get(guild.id);
    return {
      id: guild.id,
      name: guild.name,
      icon: guild.iconURL({ size: 64 }) ?? null,
      channels,
      queue: queue && !queue.closed ? queueView(queue) : null,
    };
  }

  private async resolveTarget(target: WebTarget): Promise<{ guild: Guild; channel: VoiceChannel; me: GuildMember }> {
    const guild = this.client.guilds.cache.get(target.guildId)
      ?? await this.client.guilds.fetch(target.guildId).catch(() => null);
    if (!guild) throw new Error('Сервер Discord не найден.');
    const channel = guild.channels.cache.get(target.channelId)
      ?? await guild.channels.fetch(target.channelId).catch(() => null);
    if (!channel || !channel.isVoiceBased() || channel.type !== ChannelType.GuildVoice) {
      throw new Error('Голосовой канал не найден.');
    }
    const me = guild.members.me ?? await guild.members.fetchMe().catch(() => null);
    if (!me) throw new Error('Не удалось проверить права бота на сервере.');
    const permissions = channel.permissionsFor(me);
    if (!permissions?.has(PermissionsBitField.Flags.Connect) || !permissions.has(PermissionsBitField.Flags.Speak)) {
      throw new Error('Нет прав на подключение и речь в этом канале.');
    }
    return { guild, channel, me };
  }
}

export function radioStationById(id: string): RadioStation | null {
  return radioStations.find(station => station.id === id) ?? findRadioStation(id);
}
