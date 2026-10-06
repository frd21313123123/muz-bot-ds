import {
  AudioPlayerStatus, VoiceConnectionStatus, NoSubscriberBehavior,
  createAudioPlayer, createAudioResource, entersState, getVoiceConnection, joinVoiceChannel, StreamType,
  type AudioPlayer, type VoiceConnection,
} from '@discordjs/voice';
import type { Message, VoiceBasedChannel } from 'discord.js';
import type { MusicClient, Track, RadioTrack } from '../types.js';
import { createRadioStream, createYtdlpStream, waitForAudio, type ManagedAudioStream } from './stream.js';
import { radioStations, radioTrack, type RadioStation } from './radio.js';
import { setTimeout as delay } from 'node:timers/promises';
import { TrackQueue } from './TrackQueue.js';
import { PlayerMessage } from './PlayerMessage.js';
import { Readable } from 'node:stream';
import { GuildVoice } from '../voice/GuildVoice.js';
import { configureMusicEncoder, validateSpeed, type MusicPlaybackOptions } from './audio.js';
import { formatDuration, parseDuration, videoDuration } from './duration.js';
import { isLiveVideo } from './ytdlp.js';

const IDLE_MS = 5 * 60_000;
const AUTOPLAY_HISTORY_LIMIT = 500;
const AUTOPLAY_SEED_LIMIT = 3;

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
  speed = 1;
  closed = false;
  voice: GuildVoice | null = null;

  private activeStream: ManagedAudioStream | null = null;
  private advancing = false;
  private joining: Promise<void> | null = null;
  private stopping: Promise<void> | null = null;
  private idleTimer: NodeJS.Timeout | null = null;
  private fadeTimer: NodeJS.Timeout | null = null;
  private trackOffset = 0;
  private trackSpeed = 1;
  private pauseRequested = false;
  private activeResource = null as ReturnType<typeof createAudioResource> | null;
  private advancePending = false;
  private readonly mediaFactory: (url: string, options: MusicPlaybackOptions) => ManagedAudioStream;
  private voiceDucking = false;
  private cueState: { resumeAfter: boolean; finish(): void } | null = null;
  private radioRequest: { controller: AbortController; userId?: string } | null = null;
  private radioRecovery: AbortController | null = null;
  private playbackEpoch = 0;
  private autoplayEpoch = 0;
  private readonly recentTrackIds: string[] = [];
  private readonly recentTrackIdSet = new Set<string>();
  private readonly radioFactory: (url: string) => ManagedAudioStream;
  private voiceWanted = true;
  private prepared: { track: Track; media: ManagedAudioStream; epoch: number } | null = null;
  private prefetchTimer: NodeJS.Timeout | null = null;
  private prefetchPromise: Promise<void> | null = null;

  constructor(readonly guildId: string, readonly client: MusicClient,
    mediaFactory?: (url: string, options: MusicPlaybackOptions) => ManagedAudioStream,
    private readonly radioOptions: { factory?: (url: string) => ManagedAudioStream;
      startupTimeoutMs?: number; retryDelays?: readonly number[] } = {}) {
    this.mediaFactory = mediaFactory ?? ((url, options) => createYtdlpStream(client.ytdlp, url, options));
    this.radioFactory = radioOptions.factory ?? createRadioStream;
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
    this.player.on(AudioPlayerStatus.Playing, () => {
      if (this.pauseRequested) this.player.pause(false);
    });
  }

  get tracks(): Track[] { return this.pending.items; }
  get isPaused(): boolean { return this.pauseRequested || this.player.state.status === AudioPlayerStatus.Paused; }
  get isRadio(): boolean { return this.currentTrack?.source === 'radio'; }
  get wakeName(): string {
    return this.client.voiceSettings?.get(this.guildId)
      ?? this.client.guilds.cache.get(this.guildId)?.members.me?.displayName
      ?? this.client.user?.username ?? '';
  }

  async setVoiceEnabled(enabled: boolean): Promise<boolean> {
    this.voiceWanted = enabled;
    if (!enabled) { this.voice?.setEnabled(false); this.client.voiceRuntime?.release?.(this); return true; }
    const runtime = this.client.voiceRuntime;
    if (!runtime || !this.connection) return false;
    if (!await (runtime.acquire?.(this) ?? runtime.start())) return false;
    if (this.closed || !this.connection || !this.voiceWanted) { runtime.release?.(this); return false; }
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
    // Feed 20 ms frames instead of encoding the entire reply in one synchronous
    // Opus transform. Copy once: inline volume may mutate the source bytes.
    const copy = Buffer.from(pcm);
    function* frames(): Generator<Buffer> {
      for (let offset = 0; offset < copy.length; offset += 3840) yield copy.subarray(offset, offset + 3840);
    }
    const resource = createAudioResource(Readable.from(frames(), { objectMode: false, highWaterMark: 3840 }),
      { inputType: StreamType.Raw, inlineVolume: true });
    resource.volume?.setVolume(this.volume);
    configureMusicEncoder(resource);
    const wasPlaying = this.player.state.status === AudioPlayerStatus.Playing;
    if (wasPlaying) this.player.pause();
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

  private clearPrepared(): void {
    if (this.prefetchTimer) {
      clearTimeout(this.prefetchTimer);
      this.prefetchTimer = null;
    }
    if (this.prepared) {
      this.prepared.media.destroy();
      this.prepared = null;
    }
  }

  private schedulePrefetch(epoch: number): void {
    if (this.prefetchTimer) {
      clearTimeout(this.prefetchTimer);
      this.prefetchTimer = null;
    }
    if (this.closed || epoch !== this.playbackEpoch || !this.currentTrack || this.isRadio) {
      return;
    }
    const nextTrack = this.loopCurrent ? this.currentTrack : this.pending.items[0];
    if (!nextTrack && !this.autoplay) {
      return;
    }

    const durationSec = parseDuration(this.currentTrack.duration);
    if (durationSec === null || durationSec <= 0) {
      return;
    }

    const elapsedSec = this.sourcePosition();
    const remainingSec = durationSec - elapsedSec;
    const delayMs = Math.max(0, Math.round((remainingSec - 20) * 1000));

    this.prefetchTimer = setTimeout(() => {
      this.prefetchTimer = null;
      void this.triggerPrefetch(epoch);
    }, delayMs);
  }

  private async triggerPrefetch(epoch: number): Promise<void> {
    if (this.closed || epoch !== this.playbackEpoch || !this.currentTrack || this.isRadio) {
      return;
    }
    if (this.prepared && this.prepared.epoch === epoch) {
      return;
    }

    let nextTrack = this.loopCurrent ? this.currentTrack : this.pending.items[0];
    if (!nextTrack && this.autoplay && this.currentTrack.source !== 'radio') {
      try {
        const targetId = this.currentTrack.videoId;
        const autoEpoch = this.autoplayEpoch;
        const related = await this.findAutoplayTracks(targetId, epoch);
        if (!this.closed && this.autoplay && epoch === this.playbackEpoch && autoEpoch === this.autoplayEpoch && related.length) {
          this.pending.addMany(related);
          void this.playerMessage.update();
          nextTrack = this.pending.items[0];
        }
      } catch {
        // Ignored in prefetch
      }
    }

    if (!nextTrack || this.closed || epoch !== this.playbackEpoch) {
      return;
    }

    const currentPromise = (async () => {
      try {
        const media = nextTrack.source === 'radio'
          ? this.radioFactory(nextTrack.streamUrl)
          : this.mediaFactory(nextTrack.url, { speed: this.speed, startSeconds: 0 });

        const onStreamError = (): void => {
          if (this.prepared?.media === media) {
            this.clearPrepared();
          }
        };
        media.stream.once('error', onStreamError);

        if (this.closed || epoch !== this.playbackEpoch) {
          media.destroy();
          return;
        }

        if (this.prepared) {
          this.prepared.media.destroy();
        }
        this.prepared = {
          track: nextTrack,
          media,
          epoch,
        };
      } catch {
        // Ignored in prefetch; advance() will retry
      }
    })();

    this.prefetchPromise = currentPromise;
    try {
      await currentPromise;
    } finally {
      if (this.prefetchPromise === currentPromise) {
        this.prefetchPromise = null;
      }
    }
  }

  private async hydrateTrackMetadata(track: Track, epoch: number): Promise<void> {
    if (track.source === 'radio') return;
    if (typeof this.client.ytdlp?.video !== 'function') return;
    const videoId = track.videoId;
    try {
      const info = await this.client.ytdlp.video(`https://www.youtube.com/watch?v=${videoId}`);
      if (this.closed || epoch !== this.playbackEpoch) return;
      if (info.id === videoId) {
        let changed = false;
        if ((track.title === 'Загрузка…' || !track.title.trim()) && info.title?.trim()) {
          track.title = info.title.trim();
          changed = true;
        }
        const seconds = videoDuration(info);
        if (seconds !== null) {
          const formatted = formatDuration(seconds);
          if (track.duration !== formatted) {
            track.duration = formatted;
            changed = true;
          }
        }
        if (info.thumbnail && track.thumbnail !== info.thumbnail) {
          track.thumbnail = info.thumbnail;
          changed = true;
        }
        const live = isLiveVideo(info);
        if (track.isLive !== live) {
          track.isLive = live;
          changed = true;
        }
        if (changed) {
          void this.playerMessage.update();
          if (this.currentTrack === track) {
            this.schedulePrefetch(epoch);
          }
        }
      }
    } catch {
      // Non-fatal background error
    }
  }

  private playTrack(track: Track, startSeconds = 0, paused = false, speed = this.speed,
    preparedMedia?: ManagedAudioStream): void {
    this.cueState?.finish();
    const media = preparedMedia ?? (track.source === 'radio' ? this.radioFactory(track.streamUrl)
      : this.mediaFactory(track.url, { speed, startSeconds }));
    let resource;
    try {
      resource = createAudioResource(media.stream, { inputType: media.type, inlineVolume: true });
      configureMusicEncoder(resource);
    } catch (error) {
      media.destroy();
      throw error;
    }
    // Install the replacement before retiring the old stream, so late events
    // from the old resource cannot advance the queue during a speed change.
    const oldStream = this.activeStream;
    this.activeStream = media;
    this.activeResource = resource;
    resource.volume?.setVolume(this.volume * (this.voiceDucking ? 0.2 : 1));
    this.currentTrack = track;
    if (track.source !== 'radio') this.rememberTrack(track.videoId);
    if (!track.isAutoplay && track.source !== 'radio') {
      this.pending.clearAutoplay();
      this.autoplayEpoch++;
    }
    this.trackOffset = startSeconds;
    this.trackSpeed = speed;
    this.pauseRequested = paused;
    this.clearIdleTimer();
    this.player.play(resource);
    oldStream?.destroy();
    void this.playerMessage.update();
    console.log(`[Queue:${this.guildId}] ▶ ${track.title} [${track.source === 'radio' ? track.stationId : track.videoId}] • ${track.isLive ? 'live' : track.duration}`);
    this.schedulePrefetch(this.playbackEpoch);
    if (track.source !== 'radio' && (track.duration === '?' || track.title === 'Загрузка…')) {
      void this.hydrateTrackMetadata(track, this.playbackEpoch);
    }
  }

  private rememberTrack(videoId: string): void {
    const previousIndex = this.recentTrackIds.indexOf(videoId);
    if (previousIndex >= 0) this.recentTrackIds.splice(previousIndex, 1);
    else this.recentTrackIdSet.add(videoId);
    this.recentTrackIds.push(videoId);
    if (this.recentTrackIds.length > AUTOPLAY_HISTORY_LIMIT) {
      const expired = this.recentTrackIds.shift();
      if (expired) this.recentTrackIdSet.delete(expired);
    }
  }

  private async findAutoplayTracks(videoId: string, epoch: number): Promise<Track[]> {
    const seeds = [videoId, ...this.recentTrackIds.slice().reverse().filter((id) => id !== videoId)]
      .slice(0, AUTOPLAY_SEED_LIMIT);
    for (const seed of seeds) {
      try {
        const related = await this.client.ytdlp.related(seed);
        if (this.closed || !this.autoplay || epoch !== this.playbackEpoch) return [];
        const excluded = new Set(this.recentTrackIdSet);
        for (const track of this.pending.items) {
          if (track.source !== 'radio') excluded.add(track.videoId);
        }
        const fresh: Track[] = [];
        for (const track of related) {
          if (track.source === 'radio' || excluded.has(track.videoId)) continue;
          excluded.add(track.videoId);
          fresh.push(track);
        }
        if (fresh.length) return fresh;
      } catch (error) {
        console.error(`[Queue:${this.guildId}] recommendations from ${seed}:`, error);
        if (this.closed || !this.autoplay || epoch !== this.playbackEpoch) return [];
      }
    }
    return [];
  }

  cancelRadioRequest(userId?: string): void {
    if (!userId || this.radioRequest?.userId === userId) this.radioRequest?.controller.abort();
  }

  private cancelRadioRecovery(): void {
    this.radioRecovery?.abort(); this.radioRecovery = null;
  }

  private async openRadio(station: RadioStation, requestedBy: string, signal: AbortSignal): Promise<{ track: RadioTrack; media: ManagedAudioStream }> {
    for (const url of station.streams) {
      signal.throwIfAborted();
      const media = this.radioFactory(url);
      try {
        await waitForAudio(media, signal, this.radioOptions.startupTimeoutMs);
        return { track: radioTrack(station, requestedBy, url), media };
      } catch {
        media.destroy(); signal.throwIfAborted();
      }
    }
    throw new Error('Эфир радиостанции недоступен. Попробуйте другую станцию.');
  }

  async playRadio(station: RadioStation, requestedBy: string,
    options: { signal?: AbortSignal; valid?: () => boolean; userId?: string } = {}): Promise<boolean> {
    this.cancelRadioRequest();
    const controller = new AbortController();
    const request = { controller, userId: options.userId };
    this.radioRequest = request;
    const abort = (): void => controller.abort();
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) controller.abort();
    let prepared: { track: RadioTrack; media: ManagedAudioStream } | null = null;
    try {
      if (this.closed || options.valid?.() === false) return false;
      prepared = await this.openRadio(station, requestedBy, controller.signal);
      if (this.closed || controller.signal.aborted || options.valid?.() === false) return false;
      // Only commit after audio exists. Installation retires the old resource
      // atomically; failed station requests leave playback and the queue intact.
      this.playTrack(prepared.track, 0, false, 1, prepared.media);
      prepared = null;
      this.playbackEpoch++;
      this.advancePending = false;
      this.cancelRadioRecovery();
      this.clearPrepared();
      this.clearFade(); this.pending.clear();
      this.autoplay = false; this.loopCurrent = false; this.speed = 1;
      void this.playerMessage.update();
      return true;
    } catch (error) {
      if (controller.signal.aborted || this.closed) return false;
      throw error;
    } finally {
      prepared?.media.destroy();
      options.signal?.removeEventListener('abort', abort);
      if (this.radioRequest === request) this.radioRequest = null;
    }
  }

  private async recoverRadio(track: RadioTrack, controller: AbortController, immediate = false): Promise<boolean> {
    const station = radioStations.find(item => item.id === track.stationId);
    if (!station) return false;
    const waits = immediate ? [0, ...(this.radioOptions.retryDelays ?? [1000, 2000, 4000])]
      : this.radioOptions.retryDelays ?? [1000, 2000, 4000];
    for (const ms of waits) {
      try {
        await delay(ms, undefined, { signal: controller.signal });
        const prepared = await this.openRadio(station, track.requestedBy, controller.signal);
        if (controller.signal.aborted || this.closed || this.currentTrack !== track) { prepared.media.destroy(); return false; }
        this.playTrack(prepared.track, 0, false, 1, prepared.media);
        return true;
      } catch {
        if (controller.signal.aborted || this.closed) return false;
      }
    }
    console.error(`[Queue:${this.guildId}] Эфир ${track.stationId} недоступен после повторных подключений.`);
    return false;
  }

  private async advance(): Promise<void> {
    if (this.closed || this.advancing) return;
    this.advancing = true;
    this.advancePending = false;
    const epoch = this.playbackEpoch;
    this.activeResource = null;
    this.clearFade();
    this.destroyStream();
    if (this.prefetchTimer) {
      clearTimeout(this.prefetchTimer);
      this.prefetchTimer = null;
    }
    try {
      if (this.currentTrack?.source === 'radio') {
        if (this.pauseRequested) return;
        const controller = new AbortController(); this.radioRecovery = controller;
        try { if (await this.recoverRadio(this.currentTrack, controller)) return; }
        finally { if (this.radioRecovery === controller) this.radioRecovery = null; }
        if (controller.signal.aborted || epoch !== this.playbackEpoch || this.closed) return;
      }
      if (this.prefetchPromise) {
        await Promise.race([this.prefetchPromise, delay(500)]).catch(() => {});
      }
      let next = this.loopCurrent && this.currentTrack ? this.currentTrack : this.pending.shift();
      if (!next && this.autoplay && this.currentTrack && this.currentTrack.source !== 'radio') {
        const targetId = this.currentTrack.videoId;
        const autoEpoch = this.autoplayEpoch;
        const related = await this.findAutoplayTracks(targetId, epoch);
        if (!this.closed && this.autoplay && epoch === this.playbackEpoch && autoEpoch === this.autoplayEpoch) {
          this.pending.addMany(related);
        }
        if (epoch !== this.playbackEpoch) return;
        next = this.pending.shift();
      }
      while (next && !this.closed && epoch === this.playbackEpoch) {
        try {
          let preparedMedia: ManagedAudioStream | undefined;
          if (this.prepared && this.prepared.track === next && this.prepared.epoch === epoch) {
            preparedMedia = this.prepared.media;
            this.prepared = null;
          } else {
            this.clearPrepared();
          }
          this.playTrack(next, 0, false, this.speed, preparedMedia);
          return;
        }
        catch (error) {
          console.error(`[Queue:${this.guildId}] track setup:`, error);
          this.clearPrepared();
          next = this.pending.shift();
        }
      }
      this.clearPrepared();
      if (!this.closed) {
        this.currentTrack = null;
        this.pauseRequested = false;
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
        if (this.voiceWanted && this.client.voiceRuntime) void this.setVoiceEnabled(true).catch(() => {});
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
    this.cancelRadioRequest();
    if (!track.isAutoplay) {
      this.pending.clearAutoplay();
      this.autoplayEpoch++;
    }
    this.pending.add(track);
    if (this.isRadio) this.leaveRadio();
    if (!this.currentTrack) await this.advance();
    else {
      this.schedulePrefetch(this.playbackEpoch);
      if (track.source !== 'radio' && (track.duration === '?' || track.title === 'Загрузка…')) {
        void this.hydrateTrackMetadata(track, this.playbackEpoch);
      }
    }
    void this.playerMessage.update();
  }

  async addTracks(tracks: Track[]): Promise<void> {
    this.cancelRadioRequest();
    if (tracks.some((track) => !track.isAutoplay)) {
      this.pending.clearAutoplay();
      this.autoplayEpoch++;
    }
    this.pending.addMany(tracks);
    if (this.isRadio) this.leaveRadio();
    if (!this.currentTrack) await this.advance();
    else {
      this.schedulePrefetch(this.playbackEpoch);
      for (const track of tracks) {
        if (track.source !== 'radio' && (track.duration === '?' || track.title === 'Загрузка…')) {
          void this.hydrateTrackMetadata(track, this.playbackEpoch);
        }
      }
    }
    void this.playerMessage.update();
  }

  private leaveRadio(): void {
    this.playbackEpoch++;
    this.cancelRadioRecovery();
    this.clearPrepared();
    this.activeResource = null;
    this.clearFade(); this.destroyStream();
    this.currentTrack = null; this.pauseRequested = false;
    this.player.stop(true);
    void this.playerMessage.update();
  }

  skip(): boolean {
    this.cancelRadioRequest();
    this.cueState?.finish();
    if (!this.currentTrack) return false;
    if (this.isRadio) { this.leaveRadio(); this.requestAdvance(); return true; }
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
    if (this.isRadio && !this.pauseRequested) {
      this.cueState?.finish();
      this.pauseRequested = true;
      this.cancelRadioRecovery();
      this.activeResource = null; this.destroyStream(); this.player.stop(true);
      void this.playerMessage.update(); return true;
    }
    if (this.cueState) { const wasPlaying = this.cueState.resumeAfter; this.cueState.resumeAfter = false; return wasPlaying; }
    if ((this.player.state.status !== AudioPlayerStatus.Playing && this.player.state.status !== AudioPlayerStatus.Buffering)
      || this.pauseRequested) return false;
    this.pauseRequested = true;
    if (this.player.state.status === AudioPlayerStatus.Playing) this.player.pause();
    void this.playerMessage.update();
    return true;
  }

  resume(): boolean {
    if (this.currentTrack?.source === 'radio' && this.pauseRequested) {
      this.cueState?.finish();
      const track = this.currentTrack;
      this.pauseRequested = false;
      const controller = new AbortController(); this.radioRecovery = controller;
      void this.recoverRadio(track, controller, true).then(ok => {
        if (!ok && !controller.signal.aborted && this.currentTrack === track && !this.closed) {
          this.leaveRadio(); this.requestAdvance();
        }
      }).finally(() => { if (this.radioRecovery === controller) this.radioRecovery = null; });
      void this.playerMessage.update(); return true;
    }
    if (this.cueState) { const wasPaused = !this.cueState.resumeAfter; this.cueState.resumeAfter = true; return wasPaused; }
    if (this.player.state.status !== AudioPlayerStatus.Paused
      && !(this.player.state.status === AudioPlayerStatus.Buffering && this.pauseRequested)) return false;
    this.pauseRequested = false;
    if (this.player.state.status === AudioPlayerStatus.Paused) this.player.unpause();
    void this.playerMessage.update();
    return true;
  }

  getElapsedSeconds(): number {
    return Math.floor(this.sourcePosition());
  }

  private sourcePosition(): number {
    // Count audio actually consumed, excluding startup, pauses and voice cues.
    return this.currentTrack ? this.trackOffset + (this.activeResource?.playbackDuration ?? 0) / 1000 * this.trackSpeed : 0;
  }

  setSpeed(speed: number): void {
    validateSpeed(speed);
    if (this.isRadio) throw new Error('Скорость прямого эфира всегда 1×.');
    if (this.closed) throw new Error('Очередь уже остановлена.');
    if (speed === this.speed) return;
    if (this.fadeTimer) throw new Error('Подождите завершения переключения трека.');
    this.cueState?.finish();
    this.clearPrepared();
    if (this.currentTrack && this.activeResource) {
      const paused = this.pauseRequested || this.player.state.status === AudioPlayerStatus.Paused;
      this.playTrack(this.currentTrack, this.sourcePosition(), paused, speed);
    }
    this.speed = speed;
    void this.playerMessage.update();
  }

  setVolume(percent: number): void {
    this.volume = Math.max(0.01, Math.min(1.5, percent / 100));
    const state = this.player.state;
    if (state.status !== AudioPlayerStatus.Idle && !this.fadeTimer) state.resource.volume?.setVolume(this.volume * (this.voiceDucking ? 0.2 : 1));
    void this.playerMessage.update();
  }

  setAutoplay(value: boolean): boolean {
    if (this.isRadio && value) throw new Error('Рекомендации недоступны во время радио.');
    this.autoplay = value;
    if (value) this.loopCurrent = false;
    this.pending.clearAutoplay();
    this.autoplayEpoch++;
    this.clearPrepared();
    this.schedulePrefetch(this.playbackEpoch);
    void this.playerMessage.update();
    return value;
  }
  toggleAutoplay(): boolean { return this.setAutoplay(!this.autoplay); }
  toggleLoop(): boolean {
    return this.setLoop(!this.loopCurrent);
  }
  setLoop(value: boolean): boolean {
    if (this.isRadio && value) throw new Error('Повтор недоступен во время радио.');
    this.loopCurrent = value;
    if (value) this.autoplay = false;
    this.clearPrepared();
    this.schedulePrefetch(this.playbackEpoch);
    void this.playerMessage.update();
    return this.loopCurrent;
  }
  clearQueue(): number {
    this.autoplayEpoch++;
    this.clearPrepared();
    const count = this.pending.clear();
    void this.playerMessage.update();
    return count;
  }
  setPlayerChannel(channelId: string): void { this.playerMessage.setChannel(channelId); }
  async showPlayer(message: Message, channelId: string): Promise<void> {
    await this.playerMessage.replace(message, channelId);
  }

  async stop(): Promise<void> {
    if (this.stopping) return this.stopping;
    this.closed = true;
    this.playbackEpoch++;
    this.cancelRadioRequest(); this.cancelRadioRecovery();
    this.clearPrepared();
    this.stopping = (async () => {
      this.voiceWanted = false; this.client.voiceRuntime?.release?.(this);
      this.voice?.destroy();
      this.voice = null;
      this.cueState?.finish();
      this.clearIdleTimer();
      this.clearFade();
      this.destroyStream();
      this.pending.clear();
      this.currentTrack = null;
      this.pauseRequested = false;
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
