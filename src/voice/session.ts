import { contextualCommand, isWakePhrase, normalizeSpeech, wakeDistance, unsupportedSpeech, validateIntent, type VoiceAction, type VoiceDecision, type VoiceIntent } from './intents.js';
import { extractMusicRequest, type MusicBackend, type MusicDiagnostic } from './music.js';

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
  modelAction?: VoiceAction;
  confidence?: number;
  action?: VoiceAction | 'play';
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
}
export interface SpeechMetrics { vadMs: number; segments: number; rejectedSegments: number }
export interface VoiceHost {
  wakeName(): string;
  present(userId: string): boolean;
  paused?(): boolean;
  cue(signal: AbortSignal): Promise<void>;
  duck(enabled: boolean): void;
  execute(intent: VoiceIntent): Promise<void | boolean>;
  playMusic(message: string, userId: string, signal: AbortSignal, valid: () => boolean): Promise<boolean>;
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
        try {
          if (command) { this.clearTimer(); this.phase = 'processing'; }
          if (!pcm.length || !valid()) return;
          let energy = 0, peak = 0;
          for (let i = 0; i + 1 < pcm.length; i += 2) {
            const sample = pcm.readInt16LE(i); energy += sample * sample; peak = Math.max(peak, Math.abs(sample));
          }
          let metrics: SpeechMetrics | undefined;
          const asrStart = performance.now();
          const transcript = await this.backend.transcribe(pcm, controller.signal, command, command ? undefined : this.host.wakeName(),
            (value) => { metrics = value; });
          const paused = this.host.paused?.() ?? false;
          const text = command ? contextualCommand(transcript, paused) : transcript;
          if (!valid()) return;
          this.host.diagnostic?.({ stage: 'recognition', kind: command ? 'command' : 'wake',
            audioMs: Math.round(pcm.length / 32), rms: Math.round(Math.sqrt(energy / (pcm.length / 2))), peak, ...metrics,
            wakeDistance: command ? undefined : wakeDistance(text, this.host.wakeName()),
            ms: Math.round(performance.now() - asrStart), words: normalizeSpeech(transcript).split(' ').filter(Boolean).length,
            paused, canonicalized: text !== transcript, matched: !command && isWakePhrase(text, this.host.wakeName()) });
          if (command) {
            if (extractMusicRequest(text)) {
              this.recognizedMusic = true;
              const changed = await this.host.playMusic(text, userId, controller.signal, valid);
              this.host.diagnostic?.({ stage: 'execution', action: 'play', changed });
              return;
            }
            if (unsupportedSpeech(text)) { this.host.diagnostic?.({ stage: 'rejected', reason: text.trim() ? 'unsupported' : 'empty' }); return; }
            const decisionStart = performance.now();
            const decision = await this.backend.classify(text, controller.signal);
            if (!valid()) { this.host.diagnostic?.({ stage: 'rejected', reason: 'stale' }); return; }
            const intent = validateIntent(text, decision);
            const safeAction = ['skip', 'pause', 'resume', 'stop', 'volume_set', 'volume_up', 'volume_down', 'unknown'].includes(decision.action) ? decision.action : 'unknown';
            this.host.diagnostic?.({ stage: 'decision', modelAction: safeAction, confidence: Number.isFinite(decision.confidence) ? decision.confidence : 0,
              action: intent.action, ms: Math.round(performance.now() - decisionStart) });
            if (intent.action !== 'unknown') {
              const changed = await this.host.execute(intent);
              this.host.diagnostic?.({ stage: 'execution', action: intent.action, changed: changed !== false });
            }
          } else if (this.phase === 'idle' && isWakePhrase(text, this.host.wakeName())) {
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
          // Audio/text and model error payloads must never reach ordinary logs.
          if (!controller.signal.aborted) {
            this.host.diagnostic?.({ stage: 'error', reason: 'pipeline' });
            if (epoch === this.epoch && (command || this.owner === userId)) this.cancel();
          }
        } finally {
          forget();
          if (command && epoch === this.epoch) this.cancel();
        }
      },
    };
  }
}
