import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { VoiceBackend, SpeechMetrics } from './session.js';
import type { VoiceDecision } from './intents.js';
import type { MusicCandidate, MusicDecision, MusicSelection, MusicState, MusicResultPolicy } from './music.js';

export const VOICE_DIR = path.resolve('.runtime/voice');
export const voicePython = (): string => path.join(VOICE_DIR, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

interface Request {
  id: number;
  payload: Record<string, unknown>;
  priority: boolean;
  signal: AbortSignal;
  resolve(value: unknown): void;
  reject(error: Error): void;
  cleanup(): void;
}

export interface VoiceFailure {
  reason: 'startup_timeout' | 'request_timeout' | 'invalid_reply' | 'worker_error' | 'worker_exit' | 'stdin_error';
  operation?: string;
  elapsedMs?: number;
  exitCode?: number | null;
}

export class VoiceRuntime extends EventEmitter implements VoiceBackend {
  ready = false;
  modelName: string | null = null;
  sttModelName: string | null = null;
  wakeModelName: string | null = null;
  inferenceDevice: 'cpu' | 'cuda' | null = null;
  private child: ChildProcessWithoutNullStreams | null = null;
  private waiting: Request[] = [];
  private active: Request | null = null;
  private sequence = 0;
  private timeout: NodeJS.Timeout | null = null;
  private starting: Promise<boolean> | null = null;
  private activeSince = 0;
  private stopped = false;
  private recoveryTimer: NodeJS.Timeout | null = null;
  private recoveryAttempts = 0;

  constructor(private readonly options: { command?: string; args?: string[]; prepared?: boolean;
    requestTimeoutMs?: number; autoRestart?: boolean; recoveryDelayMs?: number } = {}) { super(); }

  async start(): Promise<boolean> {
    this.stopped = false;
    if (this.recoveryTimer) clearTimeout(this.recoveryTimer);
    this.recoveryTimer = null;
    if (this.ready) return true;
    if (this.starting) return this.starting;
    this.starting = this.launch();
    try { return await this.starting; } finally { this.starting = null; }
  }

  private async launch(): Promise<boolean> {
    const prepared = this.options.prepared ?? (existsSync(voicePython()) && existsSync(path.join(VOICE_DIR, 'ready.json')));
    if (!prepared) return false;
    const child = spawn(this.options.command ?? voicePython(), this.options.args ?? ['-u', path.resolve('scripts/voice_worker.py')], {
      windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', HF_HOME: path.join(VOICE_DIR, 'cache'),
        HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', USE_TF: '0', TOKENIZERS_PARALLELISM: 'false' },
    });
    this.child = child;
    // Drain stderr without logging model input or error payloads.
    child.stderr.resume();
    return new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (ok: boolean): void => { if (!settled) { settled = true; clearTimeout(timer); resolve(ok); } };
      const timer = setTimeout(() => { if (this.child === child) this.fail('startup_timeout'); finish(false); }, 90_000);
      const lines = createInterface({ input: child.stdout });
      lines.on('line', (line) => {
        if (this.child !== child) return;
        try {
          if (line.length > 64_000) throw new Error('Invalid worker reply');
          const value: unknown = JSON.parse(line);
          if (!value || typeof value !== 'object') throw new Error('Invalid worker reply');
          const message = value as Record<string, unknown>;
          if (message.ready === true) {
            const safeName = (value: unknown): string | null => typeof value === 'string'
              && /^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,95}$/.test(value) ? value : null;
            this.modelName = typeof message.modelName === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,95}$/.test(message.modelName)
              ? message.modelName : null;
            this.sttModelName = safeName(message.sttModelName);
            this.wakeModelName = safeName(message.wakeModelName);
            this.inferenceDevice = message.inferenceDevice === 'cpu' || message.inferenceDevice === 'cuda'
              ? message.inferenceDevice : null;
            this.ready = true; finish(true); this.emit('available'); return;
          }
          const request = this.active;
          if (!request || message.id !== request.id) return;
          if (this.timeout) clearTimeout(this.timeout);
          this.timeout = null;
          this.active = null;
          this.activeSince = 0;
          request.cleanup();
          if (message.error) request.reject(new Error('Локальная модель не обработала запрос.'));
          else { this.recoveryAttempts = 0; request.resolve(message.result); }
          this.pump();
        } catch { this.fail('invalid_reply'); finish(false); }
      });
      child.once('error', () => { if (this.child === child) this.fail('worker_error'); finish(false); });
      child.once('exit', (code) => { if (this.child === child) this.fail('worker_exit', code); finish(false); lines.close(); });
      child.stdin.on('error', () => { if (this.child === child) this.fail('stdin_error'); finish(false); });
    });
  }

  private fail(reason: VoiceFailure['reason'] | 'closed', exitCode?: number | null): void {
    const wasReady = this.ready;
    const operation = this.active?.payload.op;
    const failure: VoiceFailure | null = reason === 'closed' ? null : { reason,
      ...(typeof operation === 'string' && ['transcribe', 'classify', 'music_route', 'music_policy', 'music_rerank'].includes(operation) ? { operation } : {}),
      ...(this.activeSince ? { elapsedMs: Date.now() - this.activeSince } : {}),
      ...(exitCode !== undefined ? { exitCode } : {}) };
    this.ready = false;
    this.modelName = null;
    this.sttModelName = null;
    this.wakeModelName = null;
    this.inferenceDevice = null;
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
    const child = this.child;
    this.child = null;
    child?.kill();
    for (const request of [this.active, ...this.waiting]) {
      if (request) { request.cleanup(); request.reject(new Error('Голосовой обработчик недоступен.')); }
    }
    this.active = null;
    this.activeSince = 0;
    this.waiting = [];
    if (wasReady) this.emit('unavailable');
    if (failure) { this.emit('failure', failure); this.scheduleRecovery(); }
  }

  private scheduleRecovery(): void {
    if (!this.options.autoRestart || this.stopped || this.recoveryTimer || this.ready || this.recoveryAttempts >= 3) return;
    const delayMs = (this.options.recoveryDelayMs ?? 5_000) * 2 ** this.recoveryAttempts;
    this.emit('recovering', { attempt: this.recoveryAttempts + 1, delayMs });
    this.recoveryTimer = setTimeout(() => {
      this.recoveryTimer = null;
      if (this.stopped) return;
      this.recoveryAttempts++;
      void this.start().then((ok) => { if (!ok) this.scheduleRecovery(); })
        .catch(() => this.scheduleRecovery());
    }, delayMs);
  }

  close(): void {
    this.stopped = true;
    if (this.recoveryTimer) clearTimeout(this.recoveryTimer);
    this.recoveryTimer = null;
    this.fail('closed');
  }

  private pump(): void {
    if (!this.ready || !this.child || this.active) return;
    const request = this.waiting.shift();
    if (!request) return;
    this.active = request;
    this.activeSince = Date.now();
    this.timeout = setTimeout(() => this.fail('request_timeout'), this.options.requestTimeoutMs ?? 30_000);
    this.child.stdin.write(`${JSON.stringify({ id: request.id, ...request.payload })}\n`);
  }

  private request(payload: Record<string, unknown>, signal: AbortSignal, priority: boolean): Promise<unknown> {
    if (!this.ready || signal.aborted) return Promise.reject(new Error('Голосовой запрос отменён.'));
    return new Promise((resolve, reject) => {
      const abort = (): void => {
        const index = this.waiting.indexOf(request);
        if (index >= 0) this.waiting.splice(index, 1);
        // Keep the active slot until Python responds; it cannot interrupt an inference.
        request.cleanup();
        reject(new Error('Голосовой запрос отменён.'));
      };
      const request: Request = { id: ++this.sequence, payload, priority, signal, resolve, reject,
        cleanup: () => signal.removeEventListener('abort', abort) };
      if (this.waiting.length >= 12) {
        let index = -1;
        if (priority) for (let i = this.waiting.length - 1; i >= 0; i--) {
          if (!this.waiting[i]!.priority) { index = i; break; }
        }
        if (index < 0) { reject(new Error('Голосовой обработчик занят.')); return; }
        const removed = this.waiting.splice(index, 1)[0]!;
        removed.cleanup(); removed.reject(new Error('Голосовой обработчик занят.'));
      }
      signal.addEventListener('abort', abort, { once: true });
      const index = priority ? this.waiting.findIndex((item) => !item.priority) : -1;
      if (index >= 0) this.waiting.splice(index, 0, request); else this.waiting.push(request);
      this.pump();
    });
  }

  async transcribe(pcm: Buffer, signal: AbortSignal, command: boolean, wakeName?: string,
    diagnostic?: (metrics: SpeechMetrics) => void): Promise<string> {
    if (pcm.length > 16_000 * 2 * 15 || pcm.length % 2) throw new Error('Invalid audio length');
    let result = await this.request({ op: 'transcribe', pcm: pcm.toString('base64'), wakeName: wakeName?.slice(0, 64) }, signal, command);
    if (result && typeof result === 'object') {
      const reply = result as { text?: unknown; metrics?: SpeechMetrics };
      const metrics = reply.metrics;
      if (metrics && [metrics.vadMs, metrics.segments, metrics.rejectedSegments].every((value) =>
        Number.isInteger(value) && value >= 0 && value <= 15_000)) {
        diagnostic?.({ vadMs: metrics.vadMs, segments: metrics.segments, rejectedSegments: metrics.rejectedSegments });
      }
      result = reply.text;
    }
    if (typeof result !== 'string' || result.length > 1000) throw new Error('Invalid transcription');
    return result;
  }

  async classify(text: string, signal: AbortSignal): Promise<VoiceDecision> {
    const result = await this.request({ op: 'classify', text: text.slice(0, 1000) }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid decision');
    const decision = result as VoiceDecision;
    if (typeof decision.action !== 'string' || typeof decision.confidence !== 'number') throw new Error('Invalid decision');
    return decision;
  }

  async decideMusic(state: MusicState, signal: AbortSignal): Promise<MusicDecision> {
    if (state.message.length > 1000 || state.selected_track !== null) throw new Error('Invalid music state');
    const result = await this.request({ op: 'music_route', state }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid music decision');
    const decision = result as MusicDecision;
    if (!['youtube_music_search', 'direct_youtube_video', 'player_control', 'unknown'].includes(decision.next_tool)
      || decision.query_source !== 'message'
      || !['play_first_result', 'rerank_results'].includes(decision.search_result_policy)
      || (decision.defer_result_policy !== undefined && typeof decision.defer_result_policy !== 'boolean')
      || typeof decision.confidence !== 'number' || !Number.isFinite(decision.confidence)
      || decision.confidence < 0 || decision.confidence > 1) throw new Error('Invalid music decision');
    return decision;
  }

  async rerankMusic(query: string, candidates: MusicCandidate[], signal: AbortSignal): Promise<MusicSelection> {
    this.validateCandidates(query, candidates);
    const result = await this.request({ op: 'music_rerank', query, candidates }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid music selection');
    const selection = result as MusicSelection;
    if (!Number.isInteger(selection.best_track)
      || (selection.no_match === true ? selection.best_track !== -1 : !candidates[selection.best_track])
      || (selection.no_match !== undefined && typeof selection.no_match !== 'boolean')
      || typeof selection.confidence !== 'number' || !Number.isFinite(selection.confidence)
      || selection.confidence < 0 || selection.confidence > 1) throw new Error('Invalid music selection');
    return selection;
  }

  private validateCandidates(query: string, candidates: MusicCandidate[]): void {
    if (!query || query.length > 1000 || candidates.length < 1 || candidates.length > 5
      || candidates.some((item, index) => item.index !== index || item.title.length > 240
        || item.artist.length > 120 || item.duration.length > 32)) throw new Error('Invalid music candidates');
  }

  async decideMusicResults(query: string, candidates: MusicCandidate[], signal: AbortSignal): Promise<MusicResultPolicy> {
    this.validateCandidates(query, candidates);
    const result = await this.request({ op: 'music_policy', query, candidates }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid music policy');
    const policy = result as MusicResultPolicy;
    if (!['play_first_result', 'rerank_results', 'no_result'].includes(policy.search_result_policy)
      || typeof policy.confidence !== 'number' || !Number.isFinite(policy.confidence)
      || policy.confidence < 0 || policy.confidence > 1) throw new Error('Invalid music policy');
    return policy;
  }
}
