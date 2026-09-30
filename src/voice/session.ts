import { contextualCommand, isWakePhrase, normalizeSpeech, unsupportedSpeech, validateIntent, type VoiceAction, type VoiceDecision, type VoiceIntent } from './intents.js';

export interface VoiceDiagnostic {
  stage: 'recognition' | 'rejected' | 'decision' | 'execution' | 'error' | 'timeout';
  kind?: 'wake' | 'command';
  ms?: number;
  words?: number;
  matched?: boolean;
  paused?: boolean;
  canonicalized?: boolean;
  modelAction?: VoiceAction;
  confidence?: number;
  action?: VoiceAction;
  changed?: boolean;
  reason?: 'empty' | 'unsupported' | 'stale' | 'command' | 'wait' | 'pipeline';
}

export interface VoiceBackend {
  transcribe(pcm: Buffer, signal: AbortSignal, command: boolean, wakeName?: string): Promise<string>;
  classify(text: string, signal: AbortSignal): Promise<VoiceDecision>;
}
export interface VoiceHost {
  wakeName(): string;
  present(userId: string): boolean;
  paused?(): boolean;
  cue(signal: AbortSignal): Promise<void>;
  duck(enabled: boolean): void;
  execute(intent: VoiceIntent): Promise<void | boolean>;
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
  private tasks = new Map<string, AbortController>();
  private activation: AbortController | null = null;

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
    for (const task of this.tasks.values()) task.abort();
    this.tasks.clear();
    this.owner = null;
    this.host.duck(false);
    if (this.phase !== 'disabled') this.phase = 'idle';
  }

  disable(): void { this.cancel(); this.phase = 'disabled'; }
  cancelUser(userId: string): void {
    if (this.owner === userId) this.cancel();
    else { this.tasks.get(userId)?.abort(); this.tasks.delete(userId); }
  }

  begin(userId: string): SpeechCapture | null {
    if (!this.host.present(userId) || this.tasks.has(userId)) return null;
    const command = this.phase === 'awaiting' && this.owner === userId;
    if (!command && this.phase !== 'idle') return null;
    if (this.tasks.size >= 4) return null;
    const controller = new AbortController();
    this.tasks.set(userId, controller);
    const epoch = this.epoch;
    let finished = false;
    const cancel = (): void => {
      controller.abort();
      if (this.tasks.get(userId) === controller) this.tasks.delete(userId);
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
        const valid = (): boolean => !controller.signal.aborted && epoch === this.epoch && this.host.present(userId);
        try {
          if (command) { this.clearTimer(); this.phase = 'processing'; }
          if (!pcm.length || !valid()) return;
          const asrStart = performance.now();
          const transcript = await this.backend.transcribe(pcm, controller.signal, command, command ? undefined : this.host.wakeName());
          const paused = this.host.paused?.() ?? false;
          const text = command ? contextualCommand(transcript, paused) : transcript;
          if (!valid()) return;
          this.host.diagnostic?.({ stage: 'recognition', kind: command ? 'command' : 'wake',
            ms: Math.round(performance.now() - asrStart), words: normalizeSpeech(transcript).split(' ').filter(Boolean).length,
            paused, canonicalized: text !== transcript, matched: !command && isWakePhrase(text, this.host.wakeName()) });
          if (command) {
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
            for (const [id, task] of this.tasks) if (id !== userId) { task.abort(); this.tasks.delete(id); }
            await this.host.cue(activation.signal);
            if (!valid() || activation.signal.aborted) { if (epoch === this.epoch) this.cancel(); return; }
            this.host.duck(true);
            this.phase = 'awaiting';
            this.timer = setTimeout(() => { this.host.diagnostic?.({ stage: 'timeout', reason: 'wait' }); this.cancel(); }, this.timings.waitMs);
          }
        } catch {
          // Audio/text and model error payloads must never reach ordinary logs.
          if (!controller.signal.aborted) this.host.diagnostic?.({ stage: 'error', reason: 'pipeline' });
          if (epoch === this.epoch && (command || this.owner === userId)) this.cancel();
        } finally {
          if (this.tasks.get(userId) === controller) this.tasks.delete(userId);
          if (command && epoch === this.epoch) this.cancel();
        }
      },
    };
  }
}
