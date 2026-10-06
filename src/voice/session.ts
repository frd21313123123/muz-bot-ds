import { contextualCommand, isWakePhrase, modeCommand, normalizeSpeech, playerControlRequest, wakeDistance, unsupportedSpeech, validateIntent, type VoiceAction, type VoiceDecision, type VoiceIntent } from './intents.js';
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
}
export interface SpeechMetrics { vadMs: number; segments: number; rejectedSegments: number }
export interface VoiceHost {
  wakeName(): string;
  present(userId: string): boolean;
  paused?(): boolean;
  radio?(): boolean;
  playerState?(): MusicPlayerState;
  training?(example: TrainingExample): void;
  cue(signal: AbortSignal): Promise<void>;
  duck(enabled: boolean): void;
  execute(intent: VoiceIntent, signal: AbortSignal): Promise<void | boolean>;
  playMusic(message: string, userId: string, signal: AbortSignal, valid: () => boolean,
    diagnostic?: (event: VoiceDiagnostic) => void): Promise<boolean>;
  playRadio?(station: RadioStation, userId: string, signal: AbortSignal, valid: () => boolean): Promise<boolean>;
  feedback?(key: 'unknown' | 'not_found' | 'radio_unsupported', signal: AbortSignal): Promise<void>;
  diagnostic?(event: VoiceDiagnostic): void;
}
export interface SpeechCapture {
  signal: AbortSignal;
  maxMs: number;
  silenceMs: number;
  complete(pcm: Buffer): Promise<void>;
  cancel(): void;
}
type Phase = 'idle' | 'signalling' | 'awaiting' | 'capturing' | 'processing' | 'disabled';

export class VoiceSession {
  phase: Phase = 'idle';
  private epoch = 0;
  private owner: string | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private tasks = new Map<string, Set<AbortController>>();
  private capturing = new Set<string>();
  private activation: AbortController | null = null;
  private recognizedMusic = false;

  constructor(private readonly backend: VoiceBackend, private readonly host: VoiceHost,
    private readonly timings = { waitMs: 10_000, commandMs: 15_000, wakeMs: 4_000 }) {}

  private clearTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  cancel(): void {
    this.epoch++;
    this.clearTimer();
    this.activation?.abort();
    this.activation = null;
    for (const tasks of this.tasks.values()) for (const task of tasks) task.abort();
    this.tasks.clear();
    this.capturing.clear();
    this.owner = null;
    this.recognizedMusic = false;
    this.host.duck(false);
    if (this.phase !== 'disabled') this.phase = 'idle';
  }

  disable(): void { this.cancel(); this.phase = 'disabled'; }
  enable(): void { this.cancel(); this.phase = 'idle'; }
  // A worker failure cannot recognize new audio, but a decoded music request
  // may still search with its fallback. Explicit cancellation always aborts it.
  backendUnavailable(): void { if (!this.recognizedMusic) this.cancel(); }
  cancelUser(userId: string): void {
    if (this.owner === userId) this.cancel();
    else {
      for (const task of this.tasks.get(userId) ?? []) task.abort();
      this.tasks.delete(userId); this.capturing.delete(userId);
    }
  }

  begin(userId: string): SpeechCapture | null {
    if (!this.host.present(userId) || this.capturing.has(userId)) return null;
    const command = this.phase === 'awaiting' && this.owner === userId;
    if (!command && this.phase !== 'idle') return null;
    if ([...this.tasks.values()].reduce((sum, tasks) => sum + tasks.size, 0) >= 4) return null;
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
        if (finished || controller.signal.aborted) return;
        finished = true;
        this.capturing.delete(userId);
        const valid = (): boolean => !controller.signal.aborted && epoch === this.epoch && this.host.present(userId);
        let example: TrainingExample | null = null;
        const report = (event: VoiceDiagnostic): void => {
          if (example && example.diagnostics.length < 32) example.diagnostics.push({ ...event });
          this.host.diagnostic?.(event);
        };
        try {
          if (command) { this.clearTimer(); this.phase = 'processing'; }
          if (!pcm.length || !valid()) return;
          let energy = 0, peak = 0;
          for (let i = 0; i + 1 < pcm.length; i += 2) {
            const sample = pcm.readInt16LE(i); energy += sample * sample; peak = Math.max(peak, Math.abs(sample));
          }

          let localWakeDetected = false;
          if (!command && this.backend.detectWake) {
            const wakeStart = performance.now();
            let wakeResult: { wake: boolean; probability: number };
            try {
              wakeResult = await this.backend.detectWake(pcm, controller.signal);
            } catch {
              if (!controller.signal.aborted) report({ stage: 'error', reason: 'pipeline' });
              return;
            }
            if (!valid()) return;
            const wakeElapsedMs = Math.round(performance.now() - wakeStart);
            const audioMs = Math.round(pcm.length / 32);
            report({
              stage: 'recognition', kind: 'wake', audioMs,
              rms: Math.round(Math.sqrt(energy / (pcm.length / 2))), peak,
              ms: wakeElapsedMs, matched: wakeResult.wake, confidence: wakeResult.probability,
            });
            if (!wakeResult.wake) return;
            localWakeDetected = true;

            // Standalone wake word: short utterance (<= 1.3s)
            if (audioMs <= 1300) {
              this.owner = userId;
              this.phase = 'signalling';
              const activation = new AbortController();
              this.activation = activation;
              for (const [id, pending] of this.tasks) {
                for (const task of pending) if (task !== controller) { task.abort(); pending.delete(task); }
                if (!pending.size) { this.tasks.delete(id); this.capturing.delete(id); }
              }
              await this.host.cue(activation.signal);
              if (!valid() || activation.signal.aborted) { if (epoch === this.epoch) this.cancel(); return; }
              this.host.duck(true);
              this.phase = 'awaiting';
              this.timer = setTimeout(() => { this.host.diagnostic?.({ stage: 'timeout', reason: 'wait' }); this.cancel(); }, this.timings.waitMs);
              return;
            }
          }

          let metrics: SpeechMetrics | undefined;
          const asrStart = performance.now();
          const transcript = await this.backend.transcribe(pcm, controller.signal, command, command ? undefined : this.host.wakeName(),
            (value) => { metrics = value; });
          const paused = this.host.paused?.() ?? false;
          const text = command ? contextualCommand(transcript, paused, this.host.radio?.() ?? false) : contextualCommand(transcript, paused, this.host.radio?.() ?? false);
          const direct = modeCommand(text);
          const radio = extractRadioRequest(text, true);
          const explicitMusic = !radio ? extractMusicRequest(text) : null;
          const music = explicitMusic ?? (!radio ? extractMusicRequest(text, true) : null);
          const unsupported = !music && unsupportedSpeech(text);
          const wakeMatched = localWakeDetected || isWakePhrase(transcript, this.host.wakeName()) || isWakePhrase(text, this.host.wakeName());

          if (command && this.host.training) example = {
            state: { phase: 'request', message: transcript, canonical_message: text,
              selected_track: null, player: this.host.playerState?.() ?? null },
            route: radio?.kind === 'station' ? 'radio' : music?.kind ?? (unsupported ? 'rejected' : 'control'), model_decision: null,
            validated_intent: { action: 'unknown' }, query: music?.query ?? null,
            outcome: 'cancelled', diagnostics: [],
          };
          if (!valid()) return;
          report({ stage: 'recognition', kind: command ? 'command' : 'wake',
            audioMs: Math.round(pcm.length / 32), rms: Math.round(Math.sqrt(energy / (pcm.length / 2))), peak, ...metrics,
            wakeDistance: command ? undefined : wakeDistance(text, this.host.wakeName()),
            ms: Math.round(performance.now() - asrStart), words: normalizeSpeech(transcript).split(' ').filter(Boolean).length,
            paused, canonicalized: text !== transcript, matched: !command && wakeMatched,
            requestKind: command ? (radio?.kind === 'station' ? 'radio' : music?.kind ?? (unsupported ? 'rejected' : 'control')) : undefined });

          const shouldExecuteCommand = command || (localWakeDetected && (direct || radio || music || playerControlRequest(text)));
          if (shouldExecuteCommand) {
            // Bare titles first pass through control classification. Explicit
            // play requests remain independent of model routing/artist labels.
            this.recognizedMusic = Boolean(music || radio?.kind === 'station');
            if (radio) {
              if (radio.kind === 'station' && this.host.playRadio && valid()) {
                if (example) { example.validated_intent = { action: 'radio_play' }; example.query = radio.station.id; }
                const changed = await this.host.playRadio(radio.station, userId, controller.signal, valid);
                if (example) example.outcome = changed ? 'changed' : (valid() ? 'no_op' : 'cancelled');
                report({ stage: 'execution', action: 'radio_play', changed });
              } else {
                if (example) { example.route = 'rejected'; example.outcome = 'rejected'; }
                if (valid()) await this.host.feedback?.(radio.kind === 'unsupported' ? 'radio_unsupported' : 'unknown', controller.signal);
              }
              return;
            }
            if (unsupported) {
              if (example) example.outcome = 'rejected';
              report({ stage: 'rejected', reason: text.trim() ? 'unsupported' : 'empty' });
              if (valid()) await this.host.feedback?.('unknown', controller.signal);
              return;
            }
            let intent: VoiceIntent = direct ?? { action: 'unknown' };
            if (!explicitMusic && !direct) {
              const decisionStart = performance.now();
              try {
                const decision = await this.backend.classify(text, controller.signal);
                if (!valid()) { report({ stage: 'rejected', reason: 'stale' }); return; }
                // A hallucinated high-confidence control cannot turn an artist
                // name into a skip. Control words must support the decision.
                intent = playerControlRequest(text) ? validateIntent(text, decision) : { action: 'unknown' };
                const safeAction = ['skip', 'pause', 'resume', 'stop', 'volume_set', 'volume_up', 'volume_down', 'unknown'].includes(decision.action) ? decision.action : 'unknown';
                const confidence = Number.isFinite(decision.confidence) ? decision.confidence : 0;
                if (example) example.model_decision = { action: safeAction, confidence };
                report({ stage: 'decision', modelAction: safeAction, confidence,
                  action: intent.action, ms: Math.round(performance.now() - decisionStart) });
              } catch {
                controller.signal.throwIfAborted();
                if (!music) throw new Error('Control classification failed');
                report({ stage: 'fallback', reason: 'model' });
              }
            }
            if (intent.action !== 'unknown') {
              if (example) { example.route = 'control'; example.validated_intent = intent; }
              const changed = await this.host.execute(intent, controller.signal);
              if (example) example.outcome = changed === false ? (valid() ? 'no_op' : 'cancelled') : 'changed';
              report({ stage: 'execution', action: intent.action, changed: changed !== false });
            } else if (music && valid()) {
              if (example) { example.route = music.kind; example.validated_intent = { action: 'play' }; }
              const changed = await this.host.playMusic(text, userId, controller.signal, valid, report);
              if (example) example.outcome = changed ? 'changed' : (valid() ? 'no_op' : 'cancelled');
              report({ stage: 'execution', action: 'play', changed });
            } else if (valid()) {
              if (example) { example.route = 'rejected'; example.outcome = 'rejected'; }
              await this.host.feedback?.('unknown', controller.signal);
            }
          } else if (this.phase === 'idle' && wakeMatched) {
            this.owner = userId;
            this.phase = 'signalling';
            const activation = new AbortController();
            this.activation = activation;
            // Cancel other wake recognitions, but keep this activation alive.
            for (const [id, pending] of this.tasks) {
              for (const task of pending) if (task !== controller) { task.abort(); pending.delete(task); }
              if (!pending.size) { this.tasks.delete(id); this.capturing.delete(id); }
            }
            await this.host.cue(activation.signal);
            if (!valid() || activation.signal.aborted) { if (epoch === this.epoch) this.cancel(); return; }
            this.host.duck(true);
            this.phase = 'awaiting';
            this.timer = setTimeout(() => { this.host.diagnostic?.({ stage: 'timeout', reason: 'wait' }); this.cancel(); }, this.timings.waitMs);
          }
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
          if (command && epoch === this.epoch) this.cancel();
        }
      },
    };
  }
}
