import { afterWakeName, contextualCommand, modeCommand, normalizeSpeech, playerControlRequest, wakeDistance, unsupportedSpeech, validateIntent, type VoiceAction, type VoiceDecision, type VoiceIntent } from './intents.js';
import { recordMetric } from '../utils/performance.js';
import { extractMusicRequest, type MusicBackend, type MusicDiagnostic, type MusicPlayerState } from './music.js';
import type { TrainingExample } from './training.js';
import { extractRadioRequest, type RadioStation } from '../utils/radio.js';

export interface VoiceDiagnostic {
  stage: 'recognition' | 'rejected' | 'decision' | 'execution' | 'error' | 'timeout' | MusicDiagnostic['stage'];
  kind?: 'wake' | 'command';
  ms?: number;
  words?: number;
  audioMs?: number;
  rms?: number;
  peak?: number;
  vadMs?: number;
  segments?: number;
  rejectedSegments?: number;
  wakeDistance?: number;
  matched?: boolean;
  standaloneCheck?: boolean;
  paused?: boolean;
  canonicalized?: boolean;
  requestKind?: 'search' | 'video' | 'radio' | 'control' | 'rejected';
  modelAction?: VoiceAction;
  confidence?: number;
  action?: VoiceAction | 'play' | 'radio_play';
  candidates?: number;
  bestTrack?: number;
  nextTool?: MusicDiagnostic['nextTool'];
  policy?: MusicDiagnostic['policy'];
  changed?: boolean;
  reason?: 'empty' | 'unsupported' | 'stale' | 'command' | 'wait' | 'pipeline' | MusicDiagnostic['reason'];
}

export interface VoiceBackend extends MusicBackend {
  transcribe(pcm: Buffer, signal: AbortSignal, command: boolean, wakeName?: string,
    diagnostic?: (metrics: SpeechMetrics) => void): Promise<string>;
  classify(text: string, signal: AbortSignal): Promise<VoiceDecision>;
  detectWake?(pcm: Buffer, signal: AbortSignal, threshold?: number): Promise<{ wake: boolean; probability: number }>;
  supportsWake?(name: string): boolean;
  readonly hasCommandDetector?: boolean;
  detectCommand?(pcm: Buffer, signal: AbortSignal, threshold?: number): Promise<{ class: string; action: string; confidence: number; matched: boolean }>;
}
export interface SpeechMetrics { vadMs: number; segments: number; rejectedSegments: number }
export interface VoiceHost {
  wakeName(): string;
  present(userId: string): boolean;
  paused?(): boolean;
  radio?(): boolean;
  playerState?(): MusicPlayerState;
  training?(example: TrainingExample): void;
  archiveAudio?(pcm: Buffer): void;
  cue(signal: AbortSignal): Promise<void>;
  duck(enabled: boolean): void;
  /** A completed turn frees the receiver to pick up other open microphones. */
  idle?(): void;
  execute(intent: VoiceIntent, signal: AbortSignal, executed?: () => void): Promise<void | boolean>;
  playMusic(message: string, userId: string, signal: AbortSignal, valid: () => boolean,
    diagnostic?: (event: VoiceDiagnostic) => void): Promise<boolean>;
  playRadio?(station: RadioStation, userId: string, signal: AbortSignal, valid: () => boolean): Promise<boolean>;
  feedback?(key: 'unknown' | 'not_found' | 'radio_unsupported', signal: AbortSignal): Promise<void>;
  diagnostic?(event: VoiceDiagnostic, userId?: string): void;
}
export interface SpeechCapture {
  signal: AbortSignal;
  maxMs: number;
  silenceMs: number;
  complete(pcm: Buffer): Promise<void>;
  cancel(): void;
}
type Phase = 'idle' | 'signalling' | 'awaiting' | 'capturing' | 'processing' | 'disabled';
const WAKE_WINDOW_BYTES = 2 * 16_000 * 2; // Two seconds of 16 kHz mono PCM16.
const STANDALONE_RECHECK_MS = 4_000;
// This lower candidate threshold only grants an exact-name transcription check.
const STANDALONE_CANDIDATE_SCORE = 0.25;
const MAX_ISOLATED_VOICE_MS = 1_200;

// Measure the voiced portion independently of the silence Discord appends.
function shortVoiceBurst(pcm: Buffer): { start: number; end: number } | null {
  if (pcm.length < 120 * 32) return null;
  const frameBytes = 20 * 32;
  const powers: number[] = [];
  let loudest = 0, quietest = Infinity, audibleFrames = 0;
  for (let offset = 0; offset + frameBytes <= pcm.length; offset += frameBytes) {
    let sum = 0, squares = 0;
    for (let i = offset; i < offset + frameBytes; i += 2) {
      const sample = pcm.readInt16LE(i);
      sum += sample;
      squares += sample * sample;
    }
    const samples = frameBytes / 2;
    const power = squares / samples - (sum / samples) ** 2;
    powers.push(power);
    loudest = Math.max(loudest, power);
    quietest = Math.min(quietest, power);
    if (power >= 32 ** 2) audibleFrames++;
  }
  if (audibleFrames < 4 || loudest < 4 * quietest) return null;
  const floor = Math.max(32 ** 2, loudest * 0.04);
  const start = powers.findIndex(value => value >= floor) * frameBytes;
  let last = powers.length - 1;
  while (powers[last]! < floor) last--;
  const end = (last + 1) * frameBytes;
  return end - start <= MAX_ISOLATED_VOICE_MS * 32 ? { start, end } : null;
}

function liftQuietWake(pcm: Buffer, rms: number, peak: number): Buffer {
  // Give a quiet microphone up to 18 dB of gain, aiming for -22 dBFS RMS.
  // Leave peak headroom and keep the original recording untouched.
  const gain = Math.min(8, 2600 / rms, 29_400 / peak);
  if (gain <= 1) return pcm;
  const adjusted = Buffer.allocUnsafe(pcm.length);
  for (let i = 0; i < pcm.length; i += 2) {
    adjusted.writeInt16LE(Math.round(pcm.readInt16LE(i) * gain), i);
  }
  return adjusted;
}

const acousticControls = ['pause', 'resume', 'skip', 'stop', 'volume_up', 'volume_down', 'loop_on', 'autoplay_on', 'queue_clear'] as const;
function isAcousticControl(action: string): action is (typeof acousticControls)[number] {
  return acousticControls.some(control => control === action);
}

export class VoiceSession {
  phase: Phase = 'idle';
  private epoch = 0;
  private owner: string | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private tasks = new Map<string, Set<AbortController>>();
  private capturing = new Set<string>();
  private cueReady: Promise<void> | null = null;
  private recognizedMusic = false;
  private standaloneChecks = new Map<string, number>();

  constructor(private readonly backend: VoiceBackend, private readonly host: VoiceHost,
    private readonly timings = { waitMs: 10_000, commandMs: 15_000, wakeMs: 4_000 }) {}

  private clearTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  cancel(): void {
    const wasActive = this.phase !== 'idle' && this.phase !== 'disabled';
    this.epoch++;
    this.clearTimer();
    this.cueReady = null;
    for (const tasks of this.tasks.values()) for (const task of tasks) task.abort();
    this.tasks.clear();
    this.capturing.clear();
    this.owner = null;
    this.recognizedMusic = false;
    this.host.duck(false);
    if (this.phase !== 'disabled') this.phase = 'idle';
    if (wasActive) this.host.idle?.();
  }

  disable(): void { this.cancel(); this.phase = 'disabled'; }
  enable(): void { this.cancel(); this.phase = 'idle'; }
  /** True while this speaker may answer the current wake activation. */
  awaits(userId: string): boolean {
    return this.owner === userId && (this.phase === 'signalling' || this.phase === 'awaiting');
  }
  // A worker failure cannot recognize new audio, but a decoded music request
  // may still search with its fallback. Explicit cancellation always aborts it.
  backendUnavailable(): void { if (!this.recognizedMusic) this.cancel(); }
  cancelUser(userId: string): void {
    this.standaloneChecks.delete(userId);
    if (this.owner === userId) this.cancel();
    else {
      for (const task of this.tasks.get(userId) ?? []) task.abort();
      this.tasks.delete(userId); this.capturing.delete(userId);
    }
  }

  private claim(userId: string, controller: AbortController, phase: 'signalling' | 'processing'): void {
    this.owner = userId;
    this.phase = phase;
    // Only the winning utterance may start a cue or dispatch a command.
    for (const [id, pending] of this.tasks) {
      for (const task of pending) if (task !== controller) { task.abort(); pending.delete(task); }
      if (!pending.size) this.tasks.delete(id);
    }
    this.capturing.clear();
    this.host.duck(true);
  }

  private async arm(controller: AbortController, valid: () => boolean): Promise<void> {
    const cue = this.host.cue(controller.signal);
    this.cueReady = cue;
    try { await cue; }
    finally { if (this.cueReady === cue) this.cueReady = null; }
    if (!valid()) return;
    if (this.phase === 'signalling') {
      this.phase = 'awaiting';
      this.timer = setTimeout(() => { this.host.diagnostic?.({ stage: 'timeout', reason: 'wait' }); this.cancel(); }, this.timings.waitMs);
    }
  }

  private archive(pcm: Buffer): void {
    try { this.host.archiveAudio?.(pcm); } catch { /* Archiving cannot interrupt recognition. */ }
  }

  private allowStandaloneCheck(userId: string): boolean {
    const now = performance.now();
    if ((this.standaloneChecks.get(userId) ?? 0) > now) return false;
    for (const [speaker, expires] of this.standaloneChecks) {
      if (expires <= now) this.standaloneChecks.delete(speaker);
    }
    this.standaloneChecks.set(userId, now + STANDALONE_RECHECK_MS);
    return true;
  }

  private makeRoomForWake(userId: string): boolean {
    let count = 0;
    let busiest: Set<AbortController> | undefined;
    for (const pending of this.tasks.values()) {
      count += pending.size;
      if (!busiest || pending.size > busiest.size) busiest = pending;
    }
    if (count < 4) return true;
    if (this.tasks.has(userId) || !busiest || busiest.size < 2) return false;
    // Each speaker can have only one open capture, always the newest entry.
    // Give a new speaker the oldest duplicate recognition slot.
    const oldest = busiest.values().next().value!;
    busiest.delete(oldest);
    oldest.abort();
    return true;
  }

  begin(userId: string): SpeechCapture | null {
    if (!this.host.present(userId) || this.capturing.has(userId)) return null;
    // The activated speaker may start the command while the cue is still
    // playing; waiting for the cue otherwise drops their first words because
    // Discord emits "start" only once per uninterrupted utterance.
    const command = this.awaits(userId);
    const pendingCue = this.cueReady;
    if (!command && this.phase !== 'idle') return null;
    if (!command && !this.makeRoomForWake(userId)) return null;
    const controller = new AbortController();
    const tasks = this.tasks.get(userId) ?? new Set<AbortController>();
    tasks.add(controller); this.tasks.set(userId, tasks); this.capturing.add(userId);
    const epoch = this.epoch;
    let finished = false;
    const forget = (): void => {
      tasks.delete(controller);
      if (!tasks.size && this.tasks.get(userId) === tasks) this.tasks.delete(userId);
    };
    const cancel = (): void => {
      if (controller.signal.aborted) return;
      controller.abort();
      if (!finished) this.capturing.delete(userId);
      forget();
      if (command && epoch === this.epoch) this.cancel();
    };
    if (command) {
      this.clearTimer();
      this.phase = 'capturing';
      this.timer = setTimeout(() => { this.host.diagnostic?.({ stage: 'timeout', reason: 'command' }); cancel(); }, this.timings.commandMs);
    }
    return {
      signal: controller.signal,
      maxMs: command ? this.timings.commandMs : this.timings.wakeMs,
      silenceMs: command ? 800 : 400,
      cancel,
      complete: async (pcm) => {
        const started = performance.now();
        if (finished || controller.signal.aborted) return;
        finished = true;
        this.capturing.delete(userId);
        const valid = (): boolean => !controller.signal.aborted && epoch === this.epoch && this.host.present(userId);
        let example: TrainingExample | null = null;
        let closeTurn = command;
        let executionMeasured = false;
        const executed = (): void => {
          if (executionMeasured) return;
          executionMeasured = true;
          recordMetric('voice.action', performance.now() - started);
          recordMetric('voice.total', performance.now() - started + (command ? 800 : 400));
        };
        const report = (event: VoiceDiagnostic): void => {
          if (event.stage === 'execution' && event.changed) executed();
          if (example && example.diagnostics.length < 32) example.diagnostics.push({ ...event });
          this.host.diagnostic?.(event, userId);
        };
        try {
          if (command) {
            this.clearTimer(); this.phase = 'processing';
            // Record from the first tone, but only act once the cue has played successfully.
            if (pendingCue) await pendingCue;
          }
          if (!pcm.length || pcm.length % 2 || !valid()) return;
          let energy = 0, peak = 0;
          for (let i = 0; i + 1 < pcm.length; i += 2) {
            const sample = pcm.readInt16LE(i); energy += sample * sample; peak = Math.max(peak, Math.abs(sample));
          }

          const wakeName = this.host.wakeName();
          const rms = Math.sqrt(energy / (pcm.length / 2));
          const shortWake = command ? null : shortVoiceBurst(pcm);
          const speechPcm = shortWake ? liftQuietWake(pcm, rms, peak) : pcm;
          let standaloneCheck = false;
          if (!command && this.backend.detectWake && (this.backend.supportsWake?.(wakeName) ?? true)) {
            const wakeStart = performance.now();
            const prefix = speechPcm.subarray(0, WAKE_WINDOW_BYTES);
            const candidate = await this.backend.detectWake(prefix, controller.signal);
            if (!valid()) return;
            let confidence = candidate.probability;
            if (!candidate.wake && shortWake && performance.now() - started <= STANDALONE_RECHECK_MS
              && (this.standaloneChecks.get(userId) ?? 0) <= performance.now()) {
              const start = Math.max(0, shortWake.start - 100 * 32);
              const end = Math.min(speechPcm.length, shortWake.end + 80 * 32);
              // The detector right-aligns its input. Try again without the long
              // quiet lead/tail before letting STT verify this isolated burst.
              if (confidence < STANDALONE_CANDIDATE_SCORE && (start >= 100 * 32 || end + 100 * 32 <= prefix.length
                || end > prefix.length)) {
                confidence = Math.max(confidence, (await this.backend.detectWake(speechPcm.subarray(start, end), controller.signal)).probability);
                if (!valid() || performance.now() - started > STANDALONE_RECHECK_MS) return;
              }
              // A different voice/mic can score near zero on the first try.
              // Only an exact isolated name can open the speaker-bound turn.
              standaloneCheck = this.allowStandaloneCheck(userId);
            }
            report({ stage: 'recognition', kind: 'wake', audioMs: Math.round(prefix.length / 32),
              ms: Math.round(performance.now() - wakeStart), confidence,
              matched: candidate.wake, standaloneCheck });
            if (!candidate.wake && !standaloneCheck) return;
          }

          if (command) this.archive(pcm);
          if (command && this.backend.detectCommand && this.backend.hasCommandDetector !== false) {
            const cmdStart = performance.now();
            // A detector failure may fall back to STT. A playback failure must never retry the action.
            const result = await this.backend.detectCommand(pcm, controller.signal).catch(() => null);
            if (!valid()) return;
            if (result?.matched && isAcousticControl(result.action)) {
              report({ stage: 'recognition', kind: 'command', audioMs: Math.round(pcm.length / 32),
                ms: Math.round(performance.now() - cmdStart), matched: true, confidence: result.confidence,
                action: result.action });
              const changed = await this.host.execute({ action: result.action }, controller.signal, executed);
              report({ stage: 'execution', action: result.action, changed: changed !== false });
              return;
            }
          }

          const asrStart = performance.now();
          let metrics: SpeechMetrics | undefined;
          const transcript = await this.backend.transcribe(speechPcm, controller.signal, command, command ? undefined : wakeName,
            value => { metrics = value; });
          recordMetric('speech.recognize', performance.now() - asrStart);
          if (!valid() || (!command && this.phase !== 'idle')) return;
          if (standaloneCheck && performance.now() - started > STANDALONE_RECHECK_MS) return;

          const remainder = afterWakeName(transcript, this.host.wakeName());
          // A rescued isolated name may only open the usual speaker-bound window.
          if (standaloneCheck && remainder !== '') {
            report({ stage: 'rejected', kind: 'wake', standaloneCheck, matched: false,
              reason: transcript.trim() ? 'unsupported' : 'empty', ...metrics });
            return;
          }
          const paused = this.host.paused?.() ?? false;
          const text = contextualCommand(remainder ?? transcript, paused, this.host.radio?.() ?? false);
          const radio = extractRadioRequest(text, true);
          const music = radio ? null : extractMusicRequest(text);
          const control = playerControlRequest(text) && !unsupportedSpeech(text);
          const route = radio?.kind === 'station' ? 'radio' : music?.kind ?? (control ? 'control' : 'rejected');
          if (command && this.host.training) example = {
            state: { phase: 'request', message: transcript, canonical_message: text,
              selected_track: null, player: this.host.playerState?.() ?? null },
            route, model_decision: null, validated_intent: { action: 'unknown' },
            query: music?.query ?? null, outcome: 'cancelled', diagnostics: [],
          };
          report({ stage: 'recognition', kind: command ? 'command' : 'wake',
            audioMs: Math.round(pcm.length / 32), rms: Math.round(rms), peak, ...metrics,
            wakeDistance: command ? undefined : wakeDistance(transcript, wakeName),
            ms: Math.round(performance.now() - asrStart), words: normalizeSpeech(transcript).split(' ').filter(Boolean).length,
            paused, canonicalized: text !== transcript, matched: !command && remainder !== null,
            requestKind: command ? route : undefined });

          if (!command && remainder === null) return;
          if (remainder === '') {
            this.standaloneChecks.delete(userId);
            this.claim(userId, controller, 'signalling');
            closeTurn = true;
            await this.arm(controller, valid);
            closeTurn = !valid();
            return;
          }
          // A recognizable incomplete order can be acknowledged once. A fragment of
          // conversation or an empty transcription simply ends this turn.
          const incompleteOrder = !text.includes('?') && /^(?:включи|поставь) (?:эту(?: песню)?|этот(?: трек)?|песню|трек)$/u.test(normalizeSpeech(text));
          if (!radio && !music && !control && !incompleteOrder) {
            if (example) example.outcome = 'rejected';
            report({ stage: 'rejected', reason: text.trim() ? 'unsupported' : 'empty' });
            return;
          }
          if (!command) {
            this.claim(userId, controller, 'processing');
            closeTurn = true;
            this.archive(speechPcm);
          }
          this.recognizedMusic = Boolean(music || radio?.kind === 'station');

          if (radio) {
            if (radio.kind === 'station' && this.host.playRadio) {
              if (example) { example.validated_intent = { action: 'radio_play' }; example.query = radio.station.id; }
              const changed = await this.host.playRadio(radio.station, userId, controller.signal, valid);
              if (example) example.outcome = changed ? 'changed' : (valid() ? 'no_op' : 'cancelled');
              report({ stage: 'execution', action: 'radio_play', changed });
            } else {
              if (example) { example.route = 'rejected'; example.outcome = 'rejected'; }
              if (radio.kind === 'unsupported') await this.host.feedback?.('radio_unsupported', controller.signal);
            }
            return;
          }
          if (music) {
            if (example) example.validated_intent = { action: 'play' };
            const changed = await this.host.playMusic(text, userId, controller.signal, valid, report);
            if (example) example.outcome = changed ? 'changed' : (valid() ? 'no_op' : 'cancelled');
            report({ stage: 'execution', action: 'play', changed });
            return;
          }
          if (incompleteOrder) {
            if (example) example.outcome = 'rejected';
            await this.host.feedback?.('unknown', controller.signal);
            return;
          }

          let intent = modeCommand(text);
          if (!intent) {
            const decisionStart = performance.now();
            const decision = await this.backend.classify(text, controller.signal);
            if (!valid()) { report({ stage: 'rejected', reason: 'stale' }); return; }
            intent = validateIntent(text, decision);
            const safeAction = ['skip', 'pause', 'resume', 'stop', 'volume_set', 'volume_up', 'volume_down', 'unknown'].includes(decision.action) ? decision.action : 'unknown';
            const confidence = Number.isFinite(decision.confidence) ? decision.confidence : 0;
            if (example) example.model_decision = { action: safeAction, confidence };
            report({ stage: 'decision', modelAction: safeAction, confidence,
              action: intent.action, ms: Math.round(performance.now() - decisionStart) });
          }
          if (intent.action === 'unknown') {
            if (example) { example.route = 'rejected'; example.outcome = 'rejected'; }
            return;
          }
          if (example) example.validated_intent = intent;
          const changed = await this.host.execute(intent, controller.signal, executed);
          if (example) example.outcome = changed === false ? (valid() ? 'no_op' : 'cancelled') : 'changed';
          report({ stage: 'execution', action: intent.action, changed: changed !== false });
        } catch {
          if (example) example.outcome = controller.signal.aborted ? 'cancelled' : 'failed';
          // Audio/text and model error payloads must never reach ordinary logs.
          if (!controller.signal.aborted) {
            report({ stage: 'error', reason: 'pipeline' });
            if (epoch === this.epoch && (command || this.owner === userId)) this.cancel();
          }
        } finally {
          if (example) {
            try { this.host.training?.(example); } catch { /* Corpus errors cannot change playback. */ }
          }
          forget();
          if (closeTurn && epoch === this.epoch) this.cancel();
        }
      },
    };
  }
}
