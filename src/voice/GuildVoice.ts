import { AudioPlayerStatus, EndBehaviorType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import { VoiceDecoder } from './decoder.js';
import { PlaybackEchoFilter, PlaybackEchoReference, type EchoResult } from './echo.js';
import type { GuildQueue } from '../utils/GuildQueue.js';
import { VoiceSession, type SpeechCapture, type VoiceDiagnostic } from './session.js';
import type { VoiceRuntime } from './runtime.js';
import { resolveMusicRequest } from './music.js';
import type { Confirmation } from './tts.js';
import { monitorEventLoopDelay, performance } from 'node:perf_hooks';

const PCM_BYTES_PER_MS = 32; // mono, 16 kHz, signed 16-bit
const PREROLL_BYTES = 100 * PCM_BYTES_PER_MS;
// A codec can leave a short, ambiguous transient after matched playback.
// Require sustained voice in that case, retaining its first words in preroll.
const MIN_RELATED_VOICE_BYTES = 100 * PCM_BYTES_PER_MS;
const COMMAND_PREROLL_BYTES = 200 * PCM_BYTES_PER_MS;
const MAX_RECEIVERS = 8; // bounded decoders, including idle open microphones
const RECEIVE_RETRY_MS = [40, 120, 360] as const;
type ReceiveFailure = 'subscribe' | 'stream_error' | 'unexpected_close' | 'decode' | 'no_audio';

function frameRms(pcm: Buffer): number {
  let energy = 0;
  for (let i = 0; i < pcm.length; i += 2) energy += pcm.readInt16LE(i) ** 2;
  return Math.sqrt(energy / (pcm.length / 2));
}

function voicedBytes(pcm: Buffer): number {
  let bytes = 0;
  for (let offset = 0; offset < pcm.length; offset += 20 * PCM_BYTES_PER_MS) {
    const frame = pcm.subarray(offset, offset + 20 * PCM_BYTES_PER_MS);
    if (frameRms(frame) > 12) bytes += frame.length;
  }
  return bytes;
}

export class GuildVoice {
  readonly session: VoiceSession;
  enabled = true;
  private listeningRequested = true;
  private connection: VoiceConnection | null = null;
  private readonly playback = new PlaybackEchoReference();
  private captures = new Map<string, { release(): void; dormant(): boolean }>();
  private lastInput = new Map<string, { at: number; message: string }>();
  private speakerRefresh?: ReturnType<typeof setTimeout>;
  private readonly retries = new Map<string, { failures: number; timer?: ReturnType<typeof setTimeout> }>();
  private readonly endSpeaking = (userId: string): void => {
    clearTimeout(this.retries.get(userId)?.timer);
    this.retries.delete(userId);
  };
  private readonly speaking = (userId: string): void => {
    this.endSpeaking(userId);
    this.receive(userId);
  };
  private readonly available = (): void => {
    if (!this.listeningRequested || this.enabled || this.queue.closed) return;
    try {
      this.setEnabled(true);
      console.log(`[Voice:${this.queue.guildId}] Прослушивание восстановлено после перезапуска обработчика.`);
    } catch { console.error(`[Voice:${this.queue.guildId}] Не удалось восстановить прослушивание; повторите /voice on.`); }
  };
  private readonly unavailable = (): void => {
    this.enabled = false;
    this.playback.clear();
    this.detachSpeaking();
    this.session.backendUnavailable();
    for (const capture of [...this.captures.values()]) capture.release();
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

  /** Post-volume audio read by the player, never the buffered source stream. */
  observePlayback(pcm: Buffer): void {
    if (this.enabled && this.runtime.ready && this.connection?.state.status === VoiceConnectionStatus.Ready) {
      this.playback.append(pcm);
    }
  }

  private noteInput(userId: string, message: string): void {
    this.lastInput.set(userId, { at: performance.now(), message });
  }

  inputStatus(userId: string): string {
    if (!this.enabled || !this.present(userId)) return '';
    const last = this.lastInput.get(userId);
    const age = last ? Math.floor((performance.now() - last.at) / 1000) : Infinity;
    return last && age < 60 ? `Ваш микрофон, ${age} с назад — ${last.message}`
      : 'За последнюю минуту у бота нет сведений о речи с вашего микрофона.';
  }

  private newSession(): VoiceSession {
    const diagnostic = (event: VoiceDiagnostic, userId?: string): void => {
      if (userId && event.kind === 'wake') {
        if (event.stage === 'recognition' && event.words !== undefined) {
          this.noteInput(userId, event.matched ? 'имя распознано.' : event.words ? 'в расшифровке не найдено имя.' : 'распознаватель вернул пустой текст.');
        } else if (event.stage === 'rejected') {
          this.noteInput(userId, event.reason === 'empty' ? 'распознаватель вернул пустой текст.' : 'произнесённое слово не подтверждено как отдельное имя.');
        } else if (event.stage === 'recognition') {
          this.noteInput(userId, event.matched || event.standaloneCheck ? 'проверяю имя по расшифровке.' : 'звук получен, локальный детектор не узнал имя.');
        }
      } else if (userId && event.stage === 'error') {
        this.noteInput(userId, 'обработчик не смог завершить голосовой запрос.');
      }
      if (process.env.VOICE_DEBUG === '1') console.log(`[Voice:${this.queue.guildId}] ${JSON.stringify({ ...event,
        player: this.queue.player.state.status, track: Boolean(this.queue.currentTrack) })}`);
    };
    return new VoiceSession(this.runtime, {
      wakeName: () => this.queue.wakeName,
      paused: () => this.queue.isPaused,
      radio: () => this.queue.isRadio,
      present: (userId) => this.present(userId),
      idle: () => this.scheduleSpeakerRefresh(),
      cue: (signal) => this.queue.playVoiceCue(signal),
      duck: (enabled) => this.queue.setVoiceDucking(enabled),
      diagnostic,
      archiveAudio: (pcm) => this.queue.client.voiceDriveArchive?.append(pcm),
      playerState: () => ({ connected: Boolean(this.queue.connection),
        playing: this.queue.player.state.status === AudioPlayerStatus.Playing,
        paused: this.queue.isPaused,
        autoplay: this.queue.autoplay, queue_length: this.queue.tracks.length }),
      training: this.queue.client.voiceTrainingLog?.enabled ? (example) => this.queue.client.voiceTrainingLog?.append(example,
        { stt: this.runtime.sttModelName ?? null, nli: this.runtime.modelName ?? null }) : undefined,
      feedback: (key, signal) => this.confirm(key, signal),
      playRadio: async (station, userId, signal, valid) => {
        const member = this.queue.client.guilds.cache.get(this.queue.guildId)?.members?.cache?.get(userId);
        const requestedBy = member?.displayName ?? this.queue.client.users.cache.get(userId)?.username ?? 'Участник';
        try {
          const changed = await this.queue.playRadio(station, requestedBy, { signal, valid, userId });
          if (changed && valid()) await this.confirm('radio_play', signal);
          return changed;
        } catch {
          signal.throwIfAborted();
          if (valid()) await this.confirm('radio_not_found', signal);
          return false;
        }
      },
      playMusic: async (message, userId, signal, valid, report = diagnostic) => {
        const member = this.queue.client.guilds.cache.get(this.queue.guildId)?.members?.cache?.get(userId);
        const requestedBy = member?.displayName ?? this.queue.client.users.cache.get(userId)?.username ?? 'Участник';
        const track = await resolveMusicRequest(message, requestedBy, this.runtime, this.queue.client.ytdlp, signal, report, {
          connected: Boolean(this.queue.connection), playing: this.queue.player.state.status === AudioPlayerStatus.Playing,
          paused: this.queue.isPaused, autoplay: this.queue.autoplay,
          queue_length: this.queue.tracks.length,
        });
        if (!valid() || this.queue.closed) return false;
        if (!track) { await this.confirm('not_found', signal); return false; }
        const queued = !this.queue.isRadio && Boolean(this.queue.currentTrack || this.queue.tracks.length);
        await this.queue.addTrack(track);
        if (valid() && !this.queue.closed) await this.confirm(queued ? 'queued' : 'play', signal);
        return true;
      },
      execute: async (intent, signal, executed) => {
        signal.throwIfAborted();
        let changed: boolean;
        switch (intent.action) {
          case 'autoplay_on': case 'autoplay_off': {
            const enabled = intent.action === 'autoplay_on';
            if (this.queue.isRadio && enabled) { await this.confirm('unknown', signal); return false; }
            changed = this.queue.autoplay !== enabled || (enabled && this.queue.loopCurrent);
            this.queue.setAutoplay(enabled);
            if (changed) executed?.();
            await this.confirm(intent.action, signal);
            return changed;
          }
          case 'loop_on': case 'loop_off': {
            const enabled = intent.action === 'loop_on';
            if (this.queue.isRadio && enabled) { await this.confirm('unknown', signal); return false; }
            changed = this.queue.loopCurrent !== enabled || (enabled && this.queue.autoplay);
            this.queue.setLoop(enabled);
            if (changed) executed?.();
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
            signal.throwIfAborted();
            changed = this.queue.skip(); if (changed) executed?.(); return changed;
          case 'pause': changed = this.queue.pause(); break;
          case 'resume': changed = this.queue.resume(); break;
          case 'stop':
            signal.throwIfAborted();
            { const stopping = this.queue.stop(); executed?.(); await stopping; return true; }
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
        if (changed) { executed?.(); await this.confirm(intent.action, signal); }
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
    if (this.enabled && this.runtime.ready) {
      connection.receiver.speaking.on('start', this.speaking);
      connection.receiver.speaking.on('end', this.endSpeaking);
      // The receiver removes destroyed subscriptions on the asynchronous close event.
      this.scheduleSpeakerRefresh();
    }
  }

  private detachSpeaking(): void {
    this.connection?.receiver.speaking.off('start', this.speaking);
    this.connection?.receiver.speaking.off('end', this.endSpeaking);
    clearTimeout(this.speakerRefresh); this.speakerRefresh = undefined;
    for (const userId of this.retries.keys()) this.endSpeaking(userId);
  }

  suspend(): void {
    this.lastInput.clear();
    this.playback.clear();
    this.detachSpeaking();
    this.connection = null;
    this.session.cancel();
    for (const capture of [...this.captures.values()]) capture.release();
    this.captures.clear();
  }

  setEnabled(enabled: boolean): void {
    const connection = this.connection ?? this.queue.connection;
    if (enabled && this.enabled && this.runtime.ready && this.connection === connection
      && connection?.joinConfig.selfDeaf === false) return;
    this.listeningRequested = enabled;
    this.enabled = enabled && this.runtime.ready;
    this.suspend();
    if (this.enabled) this.session.enable(); else this.session.disable();
    if (connection && connection.state.status !== VoiceConnectionStatus.Destroyed) {
      if (connection.joinConfig.selfDeaf !== !this.enabled) {
        connection.rejoin({ ...connection.joinConfig, selfDeaf: !this.enabled });
      }
      this.attach(connection);
    }
  }

  cancelUser(userId: string): void {
    this.endSpeaking(userId); this.session.cancelUser(userId); this.captures.get(userId)?.release(); this.lastInput.delete(userId);
  }
  reset(): void {
    for (const userId of this.retries.keys()) this.endSpeaking(userId);
    this.session.cancel();
    for (const capture of [...this.captures.values()]) capture.release();
  }

  destroy(): void {
    this.listeningRequested = false;
    this.suspend(); this.session.disable();
    this.runtime.off('unavailable', this.unavailable);
    this.runtime.off('available', this.available);
  }

  private present(userId: string): boolean {
    const clientUserId = this.queue.client.user?.id;
    const channelId = this.queue.voiceChannel?.id;
    if (this.queue.closed || !channelId || !clientUserId || !userId || userId === clientUserId) return false;
    const guild = this.queue.client.guilds.cache.get(this.queue.guildId);
    const user = this.queue.client.users.cache.get(userId);
    const memberUser = guild?.members?.cache?.get(userId)?.user;
    return guild?.voiceStates.cache.get(clientUserId)?.channelId === channelId
      && guild.voiceStates.cache.get(userId)?.channelId === channelId
      && user?.bot !== true && memberUser?.bot !== true
      && (user?.bot === false || memberUser?.bot === false);
  }

  private scheduleSpeakerRefresh(delay = 0): void {
    if (this.speakerRefresh || !this.enabled || !this.connection?.receiver.speaking.users?.size) return;
    this.speakerRefresh = setTimeout(() => {
      this.speakerRefresh = undefined;
      this.refreshSpeakers();
    }, delay);
    this.speakerRefresh.unref();
  }

  private receiverFailure(userId: string, reason: ReceiveFailure, resumeWake: boolean, decoded: boolean): void {
    const connection = this.connection;
    const started = connection?.receiver.speaking.users?.get(userId);
    if (!this.enabled || !this.runtime.ready || !connection || started === undefined || !this.present(userId)) return;
    this.noteInput(userId, reason === 'no_audio' ? 'от Discord не пришёл звук фразы.' : 'приём звука от Discord прервался.');
    const retry = this.retries.get(userId) ?? { failures: 0 };
    if (retry.timer || retry.failures > RECEIVE_RETRY_MS.length) return;
    const delay = RECEIVE_RETRY_MS[retry.failures++];
    this.retries.set(userId, retry);
    if (process.env.VOICE_DEBUG === '1') console.log(`[Voice:${this.queue.guildId}] ${JSON.stringify({
      stage: 'receive_error', reason, decoded, failure: retry.failures, retryMs: delay ?? null,
    })}`);
    if (delay === undefined) return;
    retry.timer = setTimeout(() => {
      retry.timer = undefined;
      if (connection !== this.connection || connection.receiver.speaking.users.get(userId) !== started
        || !this.enabled || !this.runtime.ready || !this.present(userId) || this.captures.has(userId)) return;
      // Only an uninterrupted, newly started wake may resume mid-word. A
      // partial command is discarded; an older open mic waits for fresh speech.
      if (resumeWake && Date.now() - started < 400 && this.session.phase === 'idle' && !this.queue.isSpeakingVoiceAudio) {
        this.receive(userId);
      } else this.refreshSpeakers();
    }, delay);
    retry.timer.unref();
  }

  private refreshSpeakers(): void {
    const connection = this.connection;
    if (!this.enabled || !this.runtime.ready || this.session.phase !== 'idle'
      || connection?.state.status !== VoiceConnectionStatus.Ready || !connection.receiver.speaking.users) return;
    for (const userId of connection.receiver.speaking.users.keys()) {
      if (this.captures.size >= MAX_RECEIVERS) break;
      if (this.captures.has(userId) || !this.present(userId)) continue;
      const retry = this.retries.get(userId);
      if (retry && (retry.timer || retry.failures > RECEIVE_RETRY_MS.length)) continue;
      if (this.queue.isSpeakingVoiceAudio) { this.scheduleSpeakerRefresh(400); break; }
      // The start event was missed. Open at the live edge and wait for a fresh
      // speech onset instead of treating the tail of the old utterance as a wake.
      try { this.capture(userId, null); } catch { this.receiverFailure(userId, 'subscribe', false, false); }
    }
  }

  private receive(userId: string): void {
    // Ignore other voices while the bot's cue or TTS is audible to avoid
    // hearing its own output. The activated speaker is exempt: a command
    // begun right after the cue must not be lost, because Discord emits
    // "start" only once for an uninterrupted utterance.
    if (this.queue.isSpeakingVoiceAudio && !this.session.awaits(userId)) {
      if (this.session.phase === 'idle') this.scheduleSpeakerRefresh(400);
      return;
    }
    if (!this.enabled || !this.runtime.ready || !this.connection
      || this.connection.state.status !== VoiceConnectionStatus.Ready || this.captures.has(userId)) return;
    if (!this.present(userId)) return;
    // A cue reflected by the caller's microphone must not claim (and then
    // cancel) their one command. Wait for audio that survives echo removal.
    const waitingForCommand = this.session.awaits(userId);
    const capture = waitingForCommand ? null : this.session.begin(userId);
    if (!capture && !waitingForCommand) {
      if (this.present(userId) && this.session.phase !== 'idle') this.noteInput(userId, 'шло предыдущее обращение.');
      return;
    }
    this.noteInput(userId, 'Discord сообщил о начале речи.');
    if (this.captures.size >= MAX_RECEIVERS) {
      const idle = [...this.captures.entries()].find(([id, entry]) => entry.dormant() && !this.session.awaits(id));
      if (!idle) { capture?.cancel(); return; }
      idle[1].release();
    }
    try { this.capture(userId, capture, waitingForCommand); }
    catch {
      this.receiverFailure(userId, 'subscribe', this.session.phase === 'idle', false);
      capture?.cancel();
    }
  }

  private capture(userId: string, first: SpeechCapture | null, freshCommand = false): void {
    const decoder = new VoiceDecoder();
    const echo = new PlaybackEchoFilter(this.playback);
    const end = { behavior: EndBehaviorType.AfterSilence as const, duration: first?.silenceMs ?? 400 };
    let stream;
    try {
      stream = this.connection!.receiver.subscribe(userId, { end });
    } catch (error) { decoder.delete(); throw error; }

    // Discord can keep sending non-silence Opus packets for a whole conversation
    // or for music leaking through a microphone. Keep one decoder for that talkspurt.
    // A completed/expired utterance releases its session slot independently.
    let active: SpeechCapture | null = null;
    let chunks: Buffer[] = [];
    let bytes = 0;
    let quiet = 0, peak = 0, foundOnset = false, hasVoice = false;
    let voiceBytes = 0, candidateVoiceBytes = 0;
    let noise: number | undefined, lastRms = 0;
    let calibrateBytes = first || freshCommand ? 0 : PREROLL_BYTES;
    let lead: Buffer[] = [], leadBytes = 0;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let released = false;

    const detach = (): void => {
      clearTimeout(deadline);
      active?.signal.removeEventListener('abort', detach);
      active = null;
      chunks = []; bytes = 0; peak = 0; quiet = 0; foundOnset = false; hasVoice = false;
      voiceBytes = 0; candidateVoiceBytes = 0;
      lead = []; leadBytes = 0;
    };
    const finish = (complete: boolean): void => {
      const current = active;
      if (!current) return;
      const pcm = complete && bytes && hasVoice ? Buffer.concat(chunks, bytes) : null;
      detach();
      noise = lastRms;
      if (pcm) void current.complete(pcm); else current.cancel();
    };
    const release = (): void => {
      if (released) return;
      released = true;
      clearTimeout(deadline);
      finish(false);
      stream.removeListener('data', data);
      stream.destroy();
      decoder.delete();
      echo.reset();
      lead = [];
      this.captures.delete(userId);
    };
    const failed = (reason: ReceiveFailure): void => {
      if (released) return;
      const decoded = noise !== undefined;
      const resumeWake = reason !== 'no_audio' && !decoded && first !== null && active === first && this.session.phase === 'idle';
      this.receiverFailure(userId, reason, resumeWake, decoded);
      release();
    };
    const setDeadline = (next: SpeechCapture): void => {
      clearTimeout(deadline);
      deadline = setTimeout(() => { if (bytes) finish(false); else failed('no_audio'); }, next.maxMs);
    };
    const start = (next: SpeechCapture): void => {
      active = next;
      end.duration = next.silenceMs;
      next.signal.addEventListener('abort', detach, { once: true });
      setDeadline(next);
    };
    const accept = ({ pcm, echoOnly, echoRelated = false }: EchoResult): void => {
      if (!pcm.length) return;
      if (echoOnly) {
        this.noteInput(userId, 'подавлено эхо собственного воспроизведения.');
        if (active && !hasVoice) finish(false);
        candidateVoiceBytes = 0;
        if (!active) { lead = []; leadBytes = 0; }
      } else if (active && bytes === 0) this.noteInput(userId, 'бот получил звук.');
      const rms = lastRms = frameRms(pcm);
      const voiced = echoOnly ? 0 : voicedBytes(pcm);
      if (calibrateBytes > 0) {
        calibrateBytes -= pcm.length;
        noise = rms;
        return;
      }
      if (!active && !this.session.awaits(userId) && (this.session.phase !== 'idle' || this.queue.isSpeakingVoiceAudio)) {
        noise = rms; lead = []; leadBytes = 0;
        return;
      }
      const floor = noise ?? rms;
      const rise = rms > Math.max(12, floor * 2.5);
      // Track quieter frames quickly and a rising background slowly. Both
      // onset and endpoint are relative to this microphone's own levels.
      const frameMs = pcm.length / PCM_BYTES_PER_MS;
      noise = floor + (rms - floor) * (1 - Math.exp(-frameMs / (rms < floor ? 80 : 800)));
      const awaitingCommand = !active && this.session.awaits(userId);
      if (awaitingCommand) {
        if (!voiced) candidateVoiceBytes = 0;
        else if (candidateVoiceBytes || rise || freshCommand) candidateVoiceBytes += voiced;
      }
      const commandReady = awaitingCommand && candidateVoiceBytes > 0
        && (!echoRelated || candidateVoiceBytes >= MIN_RELATED_VOICE_BYTES);
      if (!active && !echoOnly && (awaitingCommand ? commandReady : rise)) {
        const next = this.session.begin(userId);
        if (next) {
          start(next);
          if (awaitingCommand) {
            chunks = [...lead]; bytes = leadBytes;
            voiceBytes = Math.max(0, candidateVoiceBytes - voiced);
            peak = lead.reduce((value, frame) => Math.max(value, frameRms(frame)), 0);
            foundOnset = true;
          }
          freshCommand = false; candidateVoiceBytes = 0;
        }
      }
      if (active) {
        if (rise && !foundOnset) {
          foundOnset = true;
          if (bytes <= PREROLL_BYTES || peak < Math.max(24, floor * 2)) {
            if (bytes > leadBytes) setDeadline(active);
            chunks = [...lead]; bytes = leadBytes;
          }
        }
        chunks.push(pcm); bytes += pcm.length;
        voiceBytes += voiced;
        hasVoice ||= voiced > 0 && (!echoRelated || voiceBytes >= MIN_RELATED_VOICE_BYTES);
        peak = Math.max(peak, rms);
        quiet = peak > 12 && rms < Math.max(6, peak * 0.3) ? quiet + pcm.length : 0;
        if (bytes > active.maxMs * PCM_BYTES_PER_MS) finish(false);
        else if (quiet >= active.silenceMs * PCM_BYTES_PER_MS) finish(true);
      }
      lead.push(pcm); leadBytes += pcm.length;
      const preroll = candidateVoiceBytes ? COMMAND_PREROLL_BYTES : PREROLL_BYTES;
      while (leadBytes > preroll) leadBytes -= lead.shift()!.length;
    };
    const data = (packet: Buffer): void => {
      try {
        const decoded = decoder.decode(packet);
        if (!decoded) return;
        if (!first && noise === undefined) clearTimeout(deadline);
        accept(echo.process(decoded));
      } catch { failed('decode'); }
    };
    if (first) start(first);
    // A subscription created after start may receive no more packets. The
    // Discord stream only starts its own end timer after the next packet.
    else deadline = setTimeout(() => failed('no_audio'), end.duration);
    this.captures.set(userId, { release, dormant: () => active === null });
    stream.on('data', data);
    stream.on('error', () => failed('stream_error'));
    stream.once('end', () => {
      if (released) return;
      try { accept(echo.flush()); } catch { failed('decode'); return; }
      finish(true);
      release();
    });
    stream.once('close', () => failed('unexpected_close'));
  }
}
