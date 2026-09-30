import { AudioPlayerStatus, EndBehaviorType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import OpusScript from 'opusscript';
import type { GuildQueue } from '../utils/GuildQueue.js';
import { VoiceSession, type SpeechCapture } from './session.js';
import type { VoiceRuntime } from './runtime.js';

export class GuildVoice {
  readonly session: VoiceSession;
  enabled = true;
  private connection: VoiceConnection | null = null;
  private captures = new Map<string, () => void>();
  private readonly speaking = (userId: string): void => this.receive(userId);
  private readonly unavailable = (): void => this.setEnabled(false);

  constructor(private readonly queue: GuildQueue, private readonly runtime: VoiceRuntime) {
    this.session = this.newSession();
    runtime.on('unavailable', this.unavailable);
  }

  private newSession(): VoiceSession {
    return new VoiceSession(this.runtime, {
      wakeName: () => this.queue.wakeName,
      paused: () => this.queue.player.state.status === AudioPlayerStatus.Paused,
      present: (userId) => Boolean(!this.queue.closed && this.queue.voiceChannel
        && this.queue.client.guilds.cache.get(this.queue.guildId)?.voiceStates.cache.get(this.queue.client.user?.id ?? '')?.channelId === this.queue.voiceChannel.id
        && this.queue.client.guilds.cache.get(this.queue.guildId)?.voiceStates.cache.get(userId)?.channelId === this.queue.voiceChannel.id
        && !this.queue.client.users.cache.get(userId)?.bot),
      cue: (signal) => this.queue.playVoiceCue(signal),
      duck: (enabled) => this.queue.setVoiceDucking(enabled),
      execute: async (intent) => {
        switch (intent.action) {
          case 'skip': this.queue.skip(); break;
          case 'pause': this.queue.pause(); break;
          case 'resume': this.queue.resume(); break;
          case 'stop': await this.queue.stop(); break;
          case 'volume_set': this.queue.setVolume(intent.level); break;
          case 'volume_up': this.queue.setVolume(Math.round(this.queue.volume * 100) + 10); break;
          case 'volume_down': this.queue.setVolume(Math.round(this.queue.volume * 100) - 10); break;
          case 'unknown': break;
        }
      },
    });
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
    this.enabled = enabled && this.runtime.ready;
    const connection = this.connection ?? this.queue.connection;
    this.suspend();
    if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) {
      connection.rejoin({ ...connection.joinConfig, selfDeaf: !this.enabled });
      this.attach(connection);
    }
  }

  cancelUser(userId: string): void { this.session.cancelUser(userId); this.captures.get(userId)?.(); }
  reset(): void { this.session.cancel(); for (const release of [...this.captures.values()]) release(); }

  destroy(): void {
    this.suspend(); this.session.disable();
    this.runtime.off('unavailable', this.unavailable);
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
    const decoder = new OpusScript(16_000, 1, OpusScript.Application.VOIP);
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
        if (packet.length > OpusScript.MAX_PACKET_SIZE) { release(); return; }
        const pcm = decoder.decode(packet);
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
