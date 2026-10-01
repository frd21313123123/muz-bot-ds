import {
  AudioPlayerStatus, VoiceConnectionStatus, NoSubscriberBehavior,
  createAudioPlayer, createAudioResource, entersState, getVoiceConnection, joinVoiceChannel, StreamType,
  type AudioPlayer, type VoiceConnection,
} from '@discordjs/voice';
import type { Message, VoiceBasedChannel } from 'discord.js';
import type { MusicClient, Track } from '../types.js';
import { createYtdlpStream, type ManagedAudioStream } from './stream.js';
import { TrackQueue } from './TrackQueue.js';
import { PlayerMessage } from './PlayerMessage.js';
import { Readable } from 'node:stream';
import { GuildVoice } from '../voice/GuildVoice.js';
import { configureMusicEncoder } from './audio.js';

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
  voice: GuildVoice | null = null;

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
  private voiceDucking = false;
  private cueState: { resumeAfter: boolean; finish(): void } | null = null;

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
  get wakeName(): string {
    return this.client.voiceSettings?.get(this.guildId)
      ?? this.client.guilds.cache.get(this.guildId)?.members.me?.displayName
      ?? this.client.user?.username ?? '';
  }

  async setVoiceEnabled(enabled: boolean): Promise<boolean> {
    if (!enabled) { this.voice?.setEnabled(false); return true; }
    const runtime = this.client.voiceRuntime;
    if (!runtime || !this.connection || !await runtime.start()) return false;
    if (this.closed || !this.connection) return false;
    if (!this.voice) this.voice = new GuildVoice(this, runtime);
    this.voice.setEnabled(true);
    return true;
  }

  setVoiceDucking(enabled: boolean): void {
    this.voiceDucking = enabled;
    const state = this.player.state;
    if (state.status !== AudioPlayerStatus.Idle && !this.fadeTimer) {
      state.resource.volume?.setVolume(this.volume * (enabled ? 0.2 : 1));
    }
  }

  async playVoiceCue(signal: AbortSignal): Promise<void> {
    // 220 ms sine tone with a short envelope, raw 48 kHz stereo PCM.
    const samples = 10_560;
    const pcm = Buffer.alloc(samples * 4);
    for (let i = 0; i < samples; i++) {
      const envelope = Math.min(1, i / 480, (samples - i) / 480);
      const value = Math.round(Math.sin(2 * Math.PI * 880 * i / 48_000) * 6000 * envelope);
      pcm.writeInt16LE(value, i * 4); pcm.writeInt16LE(value, i * 4 + 2);
    }
    await this.playVoiceAudio(pcm, signal, 2000);
  }

  async playVoiceAudio(pcm: Buffer, signal: AbortSignal, timeoutMs = 15000): Promise<void> {
    const connection = this.connection;
    if (this.closed || !connection || connection.state.status !== VoiceConnectionStatus.Ready || signal.aborted) throw new Error('Cue cancelled');
    if (!pcm.length || pcm.length % 4 || pcm.length > 48000 * 4 * 12) throw new Error('Invalid voice audio');
    this.cueState?.finish();
    const player = createAudioPlayer();
    const resource = createAudioResource(Readable.from([Buffer.from(pcm)]), { inputType: StreamType.Raw, inlineVolume: true });
    resource.volume?.setVolume(this.volume);
    configureMusicEncoder(resource);
    const wasPlaying = this.player.state.status === AudioPlayerStatus.Playing;
    if (wasPlaying) { this.pausedAt = Date.now(); this.player.pause(); }
    await new Promise<void>((resolve, reject) => {
      let finished = false;
      const abort = (): void => finish(new Error('Cue cancelled'));
      const finish = (error?: Error): void => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        signal.removeEventListener('abort', abort);
        const resumeAfter = this.cueState?.resumeAfter ?? false;
        this.cueState = null;
        player.stop(true);
        player.removeAllListeners();
        if (!this.closed && this.connection === connection && connection.state.status !== VoiceConnectionStatus.Destroyed) {
          try {
            connection.subscribe(this.player);
            if (resumeAfter) this.resume();
          } catch { error ??= new Error('Cue restoration failed'); }
        }
        if (error || signal.aborted) reject(error ?? new Error('Cue cancelled')); else resolve();
      };
      const timer = setTimeout(() => finish(new Error('Cue timeout')), timeoutMs);
      this.cueState = { resumeAfter: wasPlaying, finish };
      signal.addEventListener('abort', abort, { once: true });
      player.once(AudioPlayerStatus.Idle, () => finish());
      player.once('error', () => finish(new Error('Cue failed')));
      try { connection.subscribe(player); player.play(resource); }
      catch { finish(new Error('Cue failed')); }
    });
  }

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
    this.cueState?.finish();
    const media = this.mediaFactory(track.url);
    let resource;
    try {
      resource = createAudioResource(media.stream, { inputType: media.type, inlineVolume: true });
      configureMusicEncoder(resource);
    } catch (error) {
      media.destroy();
      throw error;
    }
    this.activeStream = media;
    this.activeResource = resource;
    resource.volume?.setVolume(this.volume * (this.voiceDucking ? 0.2 : 1));
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
      let next = this.loopCurrent && this.currentTrack ? this.currentTrack : this.pending.shift();
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
    try {
      await this.joining;
      if (!this.closed && !this.currentTrack && !this.tracks.length) this.startIdleTimer();
    } finally { this.joining = null; }
  }

  private async connect(channel: VoiceBasedChannel): Promise<void> {
    this.voice?.suspend();
    this.connection?.destroy();
    this.connection = null;
    this.voiceChannel = channel;
    for (let attempt = 1; attempt <= 3; attempt++) {
      if (this.closed) throw new Error('Очередь остановлена.');
      const connection = joinVoiceChannel({
        channelId: channel.id, guildId: channel.guild.id,
        adapterCreator: channel.guild.voiceAdapterCreator,
        selfDeaf: !(this.voice ? this.voice.enabled : this.client.voiceRuntime?.ready),
      });
      try {
        await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
        if (this.closed) { connection.destroy(); throw new Error('Очередь остановлена.'); }
        this.connection = connection;
        connection.subscribe(this.player);
        if (!this.voice && this.client.voiceRuntime?.ready) this.voice = new GuildVoice(this, this.client.voiceRuntime);
        this.voice?.attach(connection);
        connection.on('stateChange', (oldState, newState) => {
          if (this.closed || this.connection !== connection) return;
          if (newState.status === VoiceConnectionStatus.Ready && oldState.status !== VoiceConnectionStatus.Ready) this.voice?.attach(connection);
          else if (oldState.status === VoiceConnectionStatus.Ready && newState.status !== VoiceConnectionStatus.Ready) this.voice?.suspend();
        });
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
    this.voice?.suspend();
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
    this.cueState?.finish();
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
    if (this.cueState) { const wasPlaying = this.cueState.resumeAfter; this.cueState.resumeAfter = false; return wasPlaying; }
    if (this.player.state.status !== AudioPlayerStatus.Playing) return false;
    this.pausedAt = Date.now();
    this.player.pause();
    void this.playerMessage.update();
    return true;
  }

  resume(): boolean {
    if (this.cueState) { const wasPaused = !this.cueState.resumeAfter; this.cueState.resumeAfter = true; return wasPaused; }
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
    if (state.status !== AudioPlayerStatus.Idle && !this.fadeTimer) state.resource.volume?.setVolume(this.volume * (this.voiceDucking ? 0.2 : 1));
    void this.playerMessage.update();
  }

  setAutoplay(value: boolean): boolean {
    this.autoplay = value;
    if (value) this.loopCurrent = false;
    void this.playerMessage.update();
    return value;
  }
  toggleAutoplay(): boolean { return this.setAutoplay(!this.autoplay); }
  toggleLoop(): boolean {
    return this.setLoop(!this.loopCurrent);
  }
  setLoop(value: boolean): boolean {
    this.loopCurrent = value;
    if (value) this.autoplay = false;
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
      this.voice?.destroy();
      this.voice = null;
      this.cueState?.finish();
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
