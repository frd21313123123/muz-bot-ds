import { AudioPlayerStatus, EndBehaviorType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import { VoiceDecoder } from './decoder.js';
import type { GuildQueue } from '../utils/GuildQueue.js';
import { VoiceSession, type SpeechCapture, type VoiceDiagnostic } from './session.js';
import type { VoiceRuntime } from './runtime.js';
import { resolveMusicRequest } from './music.js';
import type { Confirmation } from './tts.js';
import { monitorEventLoopDelay, performance } from 'node:perf_hooks';

export class GuildVoice {
  readonly session: VoiceSession;
  enabled = true;
  private listeningRequested = true;
  private connection: VoiceConnection | null = null;
  private captures = new Map<string, () => void>();
  private readonly speaking = (userId: string): void => this.receive(userId);
  private readonly available = (): void => {
    if (!this.listeningRequested || this.enabled || this.queue.closed) return;
    try {
      this.setEnabled(true);
      console.log(`[Voice:${this.queue.guildId}] Прослушивание восстановлено после перезапуска обработчика.`);
    } catch { console.error(`[Voice:${this.queue.guildId}] Не удалось восстановить прослушивание; повторите /voice on.`); }
  };
  private readonly unavailable = (): void => {
    this.enabled = false;
    this.connection?.receiver.speaking.off('start', this.speaking);
    this.session.backendUnavailable();
    for (const release of [...this.captures.values()]) release();
    // Keep the connection and an already recognized request alive for fallback.
    const connection = this.connection;
    if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) {
      connection.rejoin({ ...connection.joinConfig, selfDeaf: true });
    }
  };

  constructor(private readonly queue: GuildQueue, private readonly runtime: VoiceRuntime) {
    this.session = this.newSession();
    runtime.on('unavailable', this.unavailable);
    runtime.on('available', this.available);
  }

  private newSession(): VoiceSession {
    const diagnostic = (event: VoiceDiagnostic): void => {
      if (process.env.VOICE_DEBUG === '1') console.log(`[Voice:${this.queue.guildId}] ${JSON.stringify({ ...event,
        player: this.queue.player.state.status, track: Boolean(this.queue.currentTrack) })}`);
    };
    return new VoiceSession(this.runtime, {
      wakeName: () => this.queue.wakeName,
      paused: () => this.queue.isPaused,
      present: (userId) => Boolean(!this.queue.closed && this.queue.voiceChannel
        && this.queue.client.guilds.cache.get(this.queue.guildId)?.voiceStates.cache.get(this.queue.client.user?.id ?? '')?.channelId === this.queue.voiceChannel.id
        && this.queue.client.guilds.cache.get(this.queue.guildId)?.voiceStates.cache.get(userId)?.channelId === this.queue.voiceChannel.id
        && !this.queue.client.users.cache.get(userId)?.bot),
      cue: (signal) => this.queue.playVoiceCue(signal),
      duck: (enabled) => this.queue.setVoiceDucking(enabled),
      diagnostic,
      playerState: () => ({ connected: Boolean(this.queue.connection),
        playing: this.queue.player.state.status === AudioPlayerStatus.Playing,
        paused: this.queue.isPaused,
        autoplay: this.queue.autoplay, queue_length: this.queue.tracks.length }),
      training: this.queue.client.voiceTrainingLog?.enabled ? (example) => this.queue.client.voiceTrainingLog?.append(example,
        { stt: this.runtime.sttModelName ?? null, nli: this.runtime.modelName ?? null }) : undefined,
      feedback: (key, signal) => this.confirm(key, signal),
      playMusic: async (message, userId, signal, valid, report = diagnostic) => {
        const member = this.queue.client.guilds.cache.get(this.queue.guildId)?.members.cache.get(userId);
        const requestedBy = member?.displayName ?? this.queue.client.users.cache.get(userId)?.username ?? 'Участник';
        const track = await resolveMusicRequest(message, requestedBy, this.runtime, this.queue.client.ytdlp, signal, report, {
          connected: Boolean(this.queue.connection), playing: this.queue.player.state.status === AudioPlayerStatus.Playing,
          paused: this.queue.isPaused, autoplay: this.queue.autoplay,
          queue_length: this.queue.tracks.length,
        });
        if (!valid() || this.queue.closed) return false;
        if (!track) { await this.confirm('not_found', signal); return false; }
        const queued = Boolean(this.queue.currentTrack || this.queue.tracks.length);
        await this.queue.addTrack(track);
        if (valid() && !this.queue.closed) await this.confirm(queued ? 'queued' : 'play', signal);
        return true;
      },
      execute: async (intent, signal) => {
        signal.throwIfAborted();
        let changed: boolean;
        switch (intent.action) {
          case 'autoplay_on': case 'autoplay_off': {
            const enabled = intent.action === 'autoplay_on';
            changed = this.queue.autoplay !== enabled || (enabled && this.queue.loopCurrent);
            this.queue.setAutoplay(enabled);
            await this.confirm(intent.action, signal);
            return changed;
          }
          case 'loop_on': case 'loop_off': {
            const enabled = intent.action === 'loop_on';
            changed = this.queue.loopCurrent !== enabled || (enabled && this.queue.autoplay);
            this.queue.setLoop(enabled);
            await this.confirm(intent.action, signal);
            return changed;
          }
          case 'queue_clear': changed = this.queue.clearQueue() > 0; break;
          case 'voice_on':
            // A spoken on-command implies listening is already active.
            await this.confirm('voice_on', signal); return false;
          case 'voice_off':
            await this.confirm('voice_off', signal);
            signal.throwIfAborted();
            return this.queue.setVoiceEnabled(false);
          case 'skip':
            if (!this.queue.currentTrack) return false;
            // Finish the confirmation before skip's fade/advance replaces the
            // music resource. Cancellation must not trigger a delayed skip.
            await this.confirm('skip', signal);
            signal.throwIfAborted();
            return this.queue.skip();
          case 'pause': changed = this.queue.pause(); break;
          case 'resume': changed = this.queue.resume(); break;
          case 'stop':
            await this.confirm('stop', signal);
            signal.throwIfAborted();
            await this.queue.stop(); return true;
          case 'volume_set': this.queue.setVolume(intent.level); changed = true; break;
          case 'volume_up': {
            const before = this.queue.volume;
            this.queue.setVolume(Math.round(before * 100) + 10);
            changed = this.queue.volume !== before; break;
          }
          case 'volume_down': {
            const before = this.queue.volume;
            this.queue.setVolume(Math.round(before * 100) - 10);
            changed = this.queue.volume !== before; break;
          }
          case 'unknown': return false;
        }
        if (changed) await this.confirm(intent.action, signal);
        return changed;
      },
    });
  }

  private async confirm(key: Confirmation, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    const pcm = this.queue.client.voiceTts?.audio(key);
    if (!pcm || this.queue.closed) return;
    const started = performance.now();
    const delay = process.env.VOICE_DEBUG === '1' ? monitorEventLoopDelay({ resolution: 10 }) : null;
    delay?.enable();
    let status = 'played';
    try { await this.queue.playVoiceAudio(pcm, signal); }
    catch {
      status = signal.aborted ? 'cancelled' : 'failed';
      // Playback/model failures must not undo a successful player action.
      // Explicit session cancellation must still prevent delayed skip/stop.
      signal.throwIfAborted();
      if (process.env.VOICE_DEBUG === '1') console.log(`[TTS:${this.queue.guildId}] playback failed`);
    } finally {
      delay?.disable();
      if (delay) console.log(`[TTS:${this.queue.guildId}] ${JSON.stringify({ phrase: key, status,
        audioMs: Math.round(pcm.length / 192), elapsedMs: Math.round(performance.now() - started),
        eventLoopMaxMs: Math.round(delay.max / 1e6) })}`);
    }
  }

  attach(connection: VoiceConnection): void {
    this.suspend();
    this.connection = connection;
    if (this.enabled && this.runtime.ready) connection.receiver.speaking.on('start', this.speaking);
  }

  suspend(): void {
    this.connection?.receiver.speaking.off('start', this.speaking);
    this.session.cancel();
    for (const release of [...this.captures.values()]) release();
    this.captures.clear();
    this.connection = null;
  }

  setEnabled(enabled: boolean): void {
    this.listeningRequested = enabled;
    this.enabled = enabled && this.runtime.ready;
    const connection = this.connection ?? this.queue.connection;
    this.suspend();
    if (this.enabled) this.session.enable(); else this.session.disable();
    if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) {
      connection.rejoin({ ...connection.joinConfig, selfDeaf: !this.enabled });
      this.attach(connection);
    }
  }

  cancelUser(userId: string): void { this.session.cancelUser(userId); this.captures.get(userId)?.(); }
  reset(): void { this.session.cancel(); for (const release of [...this.captures.values()]) release(); }

  destroy(): void {
    this.listeningRequested = false;
    this.suspend(); this.session.disable();
    this.runtime.off('unavailable', this.unavailable);
    this.runtime.off('available', this.available);
  }

  private receive(userId: string): void {
    if (!this.enabled || !this.runtime.ready || !this.connection
      || this.connection.state.status !== VoiceConnectionStatus.Ready || this.captures.has(userId)) return;
    const capture = this.session.begin(userId);
    if (!capture) return;
    try { this.capture(userId, capture); }
    catch { capture.cancel(); }
  }

  private capture(userId: string, capture: SpeechCapture): void {
    const decoder = new VoiceDecoder();
    let stream;
    try {
      stream = this.connection!.receiver.subscribe(userId, {
        end: { behavior: EndBehaviorType.AfterSilence, duration: capture.silenceMs },
      });
    } catch (error) { decoder.delete(); throw error; }
    const chunks: Buffer[] = [];
    let bytes = 0;
    let released = false;
    const release = (cancel = true): void => {
      if (released) return;
      released = true;
      clearTimeout(timer);
      capture.signal.removeEventListener('abort', abort);
      stream.removeListener('data', data);
      stream.destroy();
      decoder.delete();
      chunks.length = 0;
      this.captures.delete(userId);
      if (cancel) capture.cancel();
    };
    const abort = (): void => release();
    const timer = setTimeout(abort, capture.maxMs);
    const data = (packet: Buffer): void => {
      try {
        const pcm = decoder.decode(packet);
        if (!pcm) return;
        bytes += pcm.length;
        if (bytes > capture.maxMs * 32) { release(); return; }
        chunks.push(pcm);
      } catch { release(); }
    };
    this.captures.set(userId, abort);
    capture.signal.addEventListener('abort', abort, { once: true });
    stream.on('data', data);
    stream.on('error', abort);
    stream.once('end', () => {
      if (released) return;
      const pcm = Buffer.concat(chunks);
      release(false);
      void capture.complete(pcm);
    });
    stream.once('close', () => { if (!released) release(); });
  }
}
