import {
  AudioPlayerStatus, VoiceConnectionStatus, NoSubscriberBehavior,
  createAudioPlayer, createAudioResource, entersState, getVoiceConnection, joinVoiceChannel,
  type AudioPlayer, type VoiceConnection,
} from '@discordjs/voice';
import type { Message, VoiceBasedChannel } from 'discord.js';
import type { MusicClient, Track } from '../types.js';
import { createYtdlpStream, type ManagedAudioStream } from './stream.js';
import { TrackQueue } from './TrackQueue.js';
import { PlayerMessage } from './PlayerMessage.js';

const IDLE_MS = 5 * 60_000;

export class GuildQueue {
  readonly player: AudioPlayer = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Pause } });
  readonly pending = new TrackQueue();
  readonly playerMessage: PlayerMessage;
  voiceChannel: VoiceBasedChannel | null = null;
  connection: VoiceConnection | null = null;
  currentTrack: Track | null = null;
  autoplay = false;
  loopCurrent = false;
  volume = 0.5;
  closed = false;

  private activeStream: ManagedAudioStream | null = null;
  private advancing = false;
  private joining: Promise<void> | null = null;
  private stopping: Promise<void> | null = null;
  private idleTimer: NodeJS.Timeout | null = null;
  private fadeTimer: NodeJS.Timeout | null = null;
  private trackStartedAt = 0;
  private pausedAt = 0;
  private pausedTotal = 0;
  private activeResource: ReturnType<typeof createAudioResource> | null = null;
  private advancePending = false;
  private readonly mediaFactory: (url: string) => ManagedAudioStream;

  constructor(readonly guildId: string, readonly client: MusicClient,
    mediaFactory?: (url: string) => ManagedAudioStream) {
    this.mediaFactory = mediaFactory ?? ((url) => createYtdlpStream(client.ytdlp, url));
    this.playerMessage = new PlayerMessage(client, this);
    this.player.on(AudioPlayerStatus.Idle, (oldState) => {
      if (oldState.status !== AudioPlayerStatus.Idle && oldState.resource === this.activeResource) {
        this.requestAdvance();
      }
    });
    this.player.on('error', (error) => {
      console.error(`[Queue:${guildId}] audio: ${error.message}`);
      if (error.resource === this.activeResource) this.requestAdvance();
    });
  }

  get tracks(): Track[] { return this.pending.items; }

  private clearIdleTimer(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  private startIdleTimer(): void {
    this.clearIdleTimer();
    this.idleTimer = setTimeout(() => { if (!this.currentTrack) void this.stop(); }, IDLE_MS);
  }

  private clearFade(): void {
    if (this.fadeTimer) clearInterval(this.fadeTimer);
    this.fadeTimer = null;
  }

  private destroyStream(): void {
    this.activeStream?.destroy();
    this.activeStream = null;
  }

  private playTrack(track: Track): void {
    const media = this.mediaFactory(track.url);
    let resource;
    try {
      resource = createAudioResource(media.stream, { inputType: media.type, inlineVolume: true });
    } catch (error) {
      media.destroy();
      throw error;
    }
    this.activeStream = media;
    this.activeResource = resource;
    resource.volume?.setVolume(this.volume);
    this.currentTrack = track;
    this.trackStartedAt = Date.now();
    this.pausedAt = 0;
    this.pausedTotal = 0;
    this.clearIdleTimer();
    this.player.play(resource);
    void this.playerMessage.update();
    console.log(`[Queue:${this.guildId}] ▶ ${track.title}`);
  }

  private async advance(): Promise<void> {
    if (this.closed || this.advancing) return;
    this.advancing = true;
    this.advancePending = false;
    this.activeResource = null;
    this.clearFade();
    this.destroyStream();
    try {
      let next = this.loopCurrent ? this.currentTrack : this.pending.shift();
      if (!next && this.autoplay && this.currentTrack) {
        try {
          const related = await this.client.ytdlp.related(this.currentTrack.videoId);
          if (!this.closed && this.autoplay) this.pending.addMany(related);
        } catch (error) {
          console.error(`[Queue:${this.guildId}] recommendations:`, error);
        }
        next = this.pending.shift();
      }
      while (next && !this.closed) {
        try { this.playTrack(next); return; }
        catch (error) {
          console.error(`[Queue:${this.guildId}] track setup:`, error);
          next = this.pending.shift();
        }
      }
      if (!this.closed) {
        this.currentTrack = null;
        void this.playerMessage.update();
        this.startIdleTimer();
      }
    } finally {
      this.advancing = false;
      if (!this.closed && (this.advancePending || (!this.currentTrack && this.pending.items.length))) {
        this.advancePending = false;
        queueMicrotask(() => void this.advance());
      }
    }
  }

  private requestAdvance(): void {
    if (this.closed) return;
    if (this.advancing) this.advancePending = true;
    else void this.advance();
  }

  async join(channel: VoiceBasedChannel): Promise<void> {
    if (this.closed) throw new Error('Очередь уже остановлена.');
    if (this.connection && this.voiceChannel?.id === channel.id
      && this.connection.state.status === VoiceConnectionStatus.Ready) return;
    if (this.joining) return this.joining;
    this.joining = this.connect(channel);
    try { await this.joining; } finally { this.joining = null; }
  }

  private async connect(channel: VoiceBasedChannel): Promise<void> {
    this.connection?.destroy();
    this.connection = null;
    this.voiceChannel = channel;
    for (let attempt = 1; attempt <= 3; attempt++) {
      if (this.closed) throw new Error('Очередь остановлена.');
      const connection = joinVoiceChannel({
        channelId: channel.id, guildId: channel.guild.id,
        adapterCreator: channel.guild.voiceAdapterCreator, selfDeaf: true,
      });
      try {
        await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
        if (this.closed) { connection.destroy(); throw new Error('Очередь остановлена.'); }
        this.connection = connection;
        connection.subscribe(this.player);
        connection.on(VoiceConnectionStatus.Disconnected, () => void this.reconnect(connection));
        return;
      } catch (error) {
        if (connection.state.status !== VoiceConnectionStatus.Destroyed) connection.destroy();
        if (attempt === 3) throw new Error(`Не удалось подключиться к голосовому каналу: ${String(error)}`);
        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }
    }
  }

  private async reconnect(connection: VoiceConnection): Promise<void> {
    if (this.closed || this.connection !== connection) return;
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
        entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
      ]);
      await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
    } catch {
      if (this.closed || this.connection !== connection) return;
      try {
        connection.rejoin();
        await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
      } catch {
        await this.stop();
      }
    }
  }

  async addTrack(track: Track): Promise<void> {
    this.pending.add(track);
    if (!this.currentTrack) await this.advance();
    void this.playerMessage.update();
  }

  async addTracks(tracks: Track[]): Promise<void> {
    this.pending.addMany(tracks);
    if (!this.currentTrack) await this.advance();
    void this.playerMessage.update();
  }

  skip(): boolean {
    if (!this.currentTrack) return false;
    this.loopCurrent = false;
    const fading = Boolean(this.fadeTimer);
    this.clearFade();
    const state = this.player.state;
    const resource = state.status === AudioPlayerStatus.Idle ? null : state.resource;
    if (!fading && this.player.state.status === AudioPlayerStatus.Playing && resource?.volume) {
      let step = 10;
      const initial = resource.volume.volume;
      this.fadeTimer = setInterval(() => {
        step--;
        resource.volume?.setVolume(Math.max(0, initial * step / 10));
        if (step <= 0) {
          this.clearFade();
          this.destroyStream();
          this.player.stop(true);
        }
      }, 150);
    } else {
      this.destroyStream();
      this.player.stop(true);
    }
    return true;
  }

  pause(): boolean {
    if (this.player.state.status !== AudioPlayerStatus.Playing) return false;
    this.pausedAt = Date.now();
    this.player.pause();
    void this.playerMessage.update();
    return true;
  }

  resume(): boolean {
    if (this.player.state.status !== AudioPlayerStatus.Paused) return false;
    if (this.pausedAt) this.pausedTotal += Date.now() - this.pausedAt;
    this.pausedAt = 0;
    this.player.unpause();
    void this.playerMessage.update();
    return true;
  }

  getElapsedSeconds(): number {
    if (!this.trackStartedAt) return 0;
    return Math.max(0, Math.floor(((this.pausedAt || Date.now()) - this.trackStartedAt - this.pausedTotal) / 1000));
  }

  setVolume(percent: number): void {
    this.volume = Math.max(0.01, Math.min(1.5, percent / 100));
    const state = this.player.state;
    if (state.status !== AudioPlayerStatus.Idle) state.resource.volume?.setVolume(this.volume);
    void this.playerMessage.update();
  }

  setAutoplay(value: boolean): boolean {
    this.autoplay = value;
    void this.playerMessage.update();
    return value;
  }
  toggleAutoplay(): boolean { return this.setAutoplay(!this.autoplay); }
  toggleLoop(): boolean {
    this.loopCurrent = !this.loopCurrent;
    void this.playerMessage.update();
    return this.loopCurrent;
  }
  clearQueue(): number { const count = this.pending.clear(); void this.playerMessage.update(); return count; }
  setPlayerChannel(channelId: string): void { this.playerMessage.setChannel(channelId); }
  async showPlayer(message: Message, channelId: string): Promise<void> {
    await this.playerMessage.replace(message, channelId);
  }

  async stop(): Promise<void> {
    if (this.stopping) return this.stopping;
    this.closed = true;
    this.stopping = (async () => {
      this.clearIdleTimer();
      this.clearFade();
      this.destroyStream();
      this.pending.clear();
      this.currentTrack = null;
      this.activeResource = null;
      this.player.stop(true);
      this.player.removeAllListeners();
      if (this.connection?.state.status !== VoiceConnectionStatus.Destroyed) this.connection?.destroy();
      this.connection = null;
      const orphaned = getVoiceConnection(this.guildId);
      if (orphaned?.state.status !== VoiceConnectionStatus.Destroyed) orphaned?.destroy();
      this.voiceChannel = null;
      if (this.client.queues.get(this.guildId) === this) this.client.queues.delete(this.guildId);
      await Promise.race([
        this.playerMessage.destroy(),
        new Promise<void>((resolve) => setTimeout(resolve, 700)),
      ]);
    })();
    return this.stopping;
  }
}
