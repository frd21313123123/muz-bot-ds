import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { VoiceBackend, SpeechMetrics } from './session.js';
import type { VoiceDecision } from './intents.js';
import type { MusicCandidate, MusicDecision, MusicSelection, MusicState, MusicResultPolicy } from './music.js';
import { extractMusicRequest } from './music.js';
import { playerControlRequest } from './intents.js';
import { ruleIntent } from './intents.js';
import { voiceDirectory, voiceProfile, type VoiceProfile } from './profile.js';
import { GroqSpeech, GROQ_STT_MODEL, sttProvider, type SttProvider } from './groq.js';
import { terminateProcess } from '../utils/processes.js';
import { recordMetric, trackProcess } from '../utils/performance.js';

export const VOICE_DIR = path.resolve('.runtime/voice');
export const voicePython = (profile = voiceProfile()): string => path.join(voiceDirectory(profile), process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

interface Request {
  queuedAt: number;
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
  readonly profile: VoiceProfile;
  readonly sttProvider: SttProvider;
  private readonly groq: GroqSpeech;
  ready = false;
  modelName: string | null = null;
  sttModelName: string | null = null;
  wakeModelName: string | null = null;
  commandModelName: string | null = null;
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
  private owners = new Set<object>();
  private unloadTimer: NodeJS.Timeout | null = null;
  private supportedWakeNames: string[] = [];
  private readonly disposing = new Set<Promise<void>>();

  constructor(private readonly options: { command?: string; args?: string[]; prepared?: boolean;
    requestTimeoutMs?: number; startupTimeoutMs?: number; autoRestart?: boolean; recoveryDelayMs?: number;
    profile?: VoiceProfile; idleTimeoutMs?: number; sttProvider?: SttProvider;
    groq?: ConstructorParameters<typeof GroqSpeech>[0] } = {}) {
    super(); this.profile = options.profile ?? voiceProfile();
    this.sttProvider = options.sttProvider ?? sttProvider(); this.groq = new GroqSpeech(options.groq);
  }

  get prepared(): boolean {
    if (this.sttProvider === 'groq' && !this.groq.configured) return false;
    if (this.sttProvider === 'groq' && this.profile === 'light') return true;
    return this.options.prepared ?? (existsSync(voicePython(this.profile)) && existsSync(path.join(voiceDirectory(this.profile), 'ready.json')));
  }

  async acquire(owner: object): Promise<boolean> {
    this.owners.add(owner); this.clearUnload();
    const ready = await this.start();
    if (!ready) this.release(owner);
    return ready;
  }
  release(owner: object): void {
    this.owners.delete(owner);
    if (!this.owners.size && this.recoveryTimer && !this.ready) {
      clearTimeout(this.recoveryTimer); this.recoveryTimer = null; this.stopped = true;
    }
    this.scheduleUnload();
  }
  private clearUnload(): void { if (this.unloadTimer) clearTimeout(this.unloadTimer); this.unloadTimer = null; }
  private scheduleUnload(): void {
    this.clearUnload();
    const timeout = this.options.idleTimeoutMs ?? 60_000;
    // Let a pending load settle first; start() arms this again on completion.
    if (this.owners.size || this.stopped || !timeout || !this.ready) return;
    this.unloadTimer = setTimeout(() => {
      this.unloadTimer = null;
      if (this.owners.size) return;
      if (this.active || this.waiting.length || this.groq.busy) this.scheduleUnload(); else this.close();
    }, timeout);
    this.unloadTimer.unref();
  }

  async start(): Promise<boolean> {
    this.stopped = false;
    if (this.recoveryTimer) clearTimeout(this.recoveryTimer);
    this.recoveryTimer = null;
    if (this.ready) { this.scheduleUnload(); return true; }
    if (this.starting) return this.starting;
    this.starting = this.launch();
    try {
      const started = await this.starting;
      if (!started && this.prepared && this.sttProvider === 'groq' && this.profile === 'light' && !this.options.command && !this.stopped) {
        if (this.recoveryTimer) clearTimeout(this.recoveryTimer);
        this.recoveryTimer = null;
        this.modelName = 'rules'; this.sttModelName = GROQ_STT_MODEL; this.wakeModelName = GROQ_STT_MODEL;
        this.ready = true; this.emit('available'); return true;
      }
      return started;
    } finally { this.starting = null; this.scheduleUnload(); }
  }

  private async launch(): Promise<boolean> {
    const directory = voiceDirectory(this.profile);
    if (!this.prepared) return false;
    if (this.sttProvider === 'groq' && this.profile === 'light') {
      const commandIsWorker = Boolean(this.options.command && existsSync(this.options.command));
      const pythonPath = voicePython(this.profile);
      const pythonCmd = existsSync(pythonPath) ? pythonPath
        : (existsSync(voicePython('heavy')) ? voicePython('heavy') : null);
      const hasWakeModel = existsSync(path.resolve('models/wake-model/wake_model.onnx'))
        || existsSync(path.resolve('word_training/runs/bot-v5-acc90/wake-model/wake_model.onnx'))
        || Boolean(process.env.WAKE_MODEL_PATH);
      const hasCommandModel = existsSync(path.resolve('models/command-model/command_model.onnx'))
        || Boolean(process.env.COMMAND_MODEL_PATH);
      if (!commandIsWorker && (!pythonCmd || (!hasWakeModel && !hasCommandModel) || this.options.command)) {
        this.modelName = 'rules'; this.sttModelName = GROQ_STT_MODEL; this.wakeModelName = GROQ_STT_MODEL;
        this.ready = true; this.emit('available'); return true;
      }
    }
    const pythonCmd = this.options.command
      ?? (existsSync(voicePython(this.profile)) ? voicePython(this.profile)
        : (existsSync(voicePython('heavy')) ? voicePython('heavy') : voicePython(this.profile)));
    const child = spawn(pythonCmd, this.options.args ?? ['-u', path.resolve('scripts/voice_worker.py')], {
      windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, VOICE_PROFILE: this.profile, STT_PROVIDER: this.sttProvider, PYTHONIOENCODING: 'utf-8', HF_HOME: path.join(directory, 'cache'),
        HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', USE_TF: '0', TOKENIZERS_PARALLELISM: 'false' },
    });
    this.child = child;
    trackProcess(child);
    // Drain stderr without logging model input or error payloads.
    child.stderr.resume();
    return new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (ok: boolean): void => { if (!settled) { settled = true; clearTimeout(timer); resolve(ok); } };
      const startupMs = this.sttProvider === 'groq' && this.profile === 'light'
        ? Math.min(this.options.startupTimeoutMs ?? 5000, 5000) : this.options.startupTimeoutMs ?? 90_000;
      const timer = setTimeout(() => { if (this.child === child) this.fail('startup_timeout'); finish(false); }, startupMs);
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
            this.sttModelName = this.sttProvider === 'groq' ? GROQ_STT_MODEL : safeName(message.sttModelName);
            this.wakeModelName = safeName(message.wakeModelName) ?? (this.sttProvider === 'groq' ? GROQ_STT_MODEL : safeName(message.wakeModelName));
            this.commandModelName = safeName(message.commandModelName);
            this.supportedWakeNames = Array.isArray(message.wakeNames)
              ? message.wakeNames.filter((name): name is string => typeof name === 'string' && name.length <= 64).map(name => name.trim().toLowerCase())
              : [];
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
      ...(typeof operation === 'string' && ['transcribe', 'classify', 'music_route', 'music_policy', 'music_rerank', 'wake_detect', 'command_detect'].includes(operation) ? { operation } : {}),
      ...(this.activeSince ? { elapsedMs: Date.now() - this.activeSince } : {}),
      ...(exitCode !== undefined ? { exitCode } : {}) };
    this.ready = false;
    this.groq.close();
    this.modelName = null;
    this.sttModelName = null;
    this.wakeModelName = null;
    this.commandModelName = null;
    this.supportedWakeNames = [];
    this.inferenceDevice = null;
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
    const child = this.child;
    this.child = null;
    if (child) {
      const termination = terminateProcess(child);
      this.disposing.add(termination);
      void termination.finally(() => this.disposing.delete(termination));
    }
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

  close(): Promise<void> {
    this.stopped = true;
    this.owners.clear(); this.clearUnload();
    if (this.recoveryTimer) clearTimeout(this.recoveryTimer);
    this.recoveryTimer = null;
    this.fail('closed');
    return Promise.all([...this.disposing]).then(() => {});
  }

  private pump(): void {
    if (!this.ready || !this.child || this.active) return;
    const request = this.waiting.shift();
    if (!request) return;
    this.active = request;
    recordMetric('speech.wait', performance.now() - request.queuedAt);
    this.activeSince = Date.now();
    this.timeout = setTimeout(() => this.fail('request_timeout'), this.options.requestTimeoutMs ?? 30_000);
    this.child.stdin.write(`${JSON.stringify({ id: request.id, ...request.payload })}\n`);
  }

  private request(payload: Record<string, unknown>, signal: AbortSignal, priority: boolean): Promise<unknown> {
    if (!this.ready || signal.aborted) return Promise.reject(new Error('Голосовой запрос отменён.'));
    this.scheduleUnload();
    return new Promise((resolve, reject) => {
      const abort = (): void => {
        const index = this.waiting.indexOf(request);
        if (index >= 0) this.waiting.splice(index, 1);
        // Keep the active slot until Python responds; it cannot interrupt an inference.
        request.cleanup();
        reject(new Error('Голосовой запрос отменён.'));
      };
      const request: Request = { id: ++this.sequence, payload, priority, signal, resolve, reject, queuedAt: performance.now(),
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
    if (!pcm.length || pcm.length > 16_000 * 2 * 15 || pcm.length % 2) throw new Error('Invalid audio length');
    if (this.sttProvider === 'groq') {
      if (!this.ready || signal.aborted) throw new Error('Голосовой запрос отменён.');
      this.scheduleUnload();
      try { return await this.groq.transcribe(pcm, signal, command, wakeName, diagnostic); }
      finally { this.scheduleUnload(); }
    }
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

  get hasWakeDetector(): boolean {
    return Boolean(this.ready && this.child && this.wakeModelName && this.wakeModelName !== GROQ_STT_MODEL);
  }

  supportsWake(name: string): boolean {
    return this.hasWakeDetector && this.supportedWakeNames.includes(name.trim().toLowerCase());
  }

  async detectWake(pcm: Buffer, signal: AbortSignal, threshold?: number): Promise<{ wake: boolean; probability: number }> {
    if (!pcm.length || pcm.length > 16_000 * 2 * 15 || pcm.length % 2) throw new Error('Invalid audio length');
    if (!this.ready || signal.aborted) throw new Error('Голосовой запрос отменён.');
    this.scheduleUnload();
    const payload: Record<string, unknown> = { op: 'wake_detect', pcm: pcm.toString('base64') };
    if (threshold !== undefined && Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) {
      payload.threshold = threshold;
    }
    const result = await this.request(payload, signal, false);
    if (!result || typeof result !== 'object') throw new Error('Invalid wake detection result');
    const res = result as { wake?: unknown; probability?: unknown };
    if (typeof res.wake !== 'boolean' || typeof res.probability !== 'number') throw new Error('Invalid wake detection result');
    return { wake: res.wake, probability: res.probability };
  }

  get hasCommandDetector(): boolean {
    return Boolean(this.ready && this.child && this.commandModelName);
  }

  async detectCommand(pcm: Buffer, signal: AbortSignal, threshold?: number): Promise<{ class: string; action: string; confidence: number; matched: boolean }> {
    if (!pcm.length || pcm.length > 16_000 * 2 * 15 || pcm.length % 2) throw new Error('Invalid audio length');
    if (!this.ready || signal.aborted) throw new Error('Голосовой запрос отменён.');
    this.scheduleUnload();
    const payload: Record<string, unknown> = { op: 'command_detect', pcm: pcm.toString('base64') };
    if (threshold !== undefined && Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) {
      payload.threshold = threshold;
    }
    const result = await this.request(payload, signal, false);
    if (!result || typeof result !== 'object') throw new Error('Invalid command detection result');
    const res = result as { class?: unknown; action?: unknown; confidence?: unknown; matched?: unknown };
    if (typeof res.class !== 'string' || typeof res.action !== 'string' || typeof res.confidence !== 'number' || typeof res.matched !== 'boolean') {
      throw new Error('Invalid command detection result');
    }
    return { class: res.class, action: res.action, confidence: res.confidence, matched: res.matched };
  }

  async classify(text: string, signal: AbortSignal): Promise<VoiceDecision> {
    signal.throwIfAborted();
    if (this.profile === 'light') return { action: ruleIntent(text).action, confidence: 1 };
    const result = await this.request({ op: 'classify', text: text.slice(0, 1000) }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid decision');
    const decision = result as VoiceDecision;
    if (typeof decision.action !== 'string' || typeof decision.confidence !== 'number') throw new Error('Invalid decision');
    return decision;
  }

  async decideMusic(state: MusicState, signal: AbortSignal): Promise<MusicDecision> {
    if (state.message.length > 1000 || state.selected_track !== null) throw new Error('Invalid music state');
    signal.throwIfAborted();
    if (this.profile === 'light') {
      const parsed = extractMusicRequest(state.message, true);
      return { next_tool: parsed?.kind === 'video' ? 'direct_youtube_video' : parsed ? 'youtube_music_search'
        : playerControlRequest(state.message) ? 'player_control' : 'unknown',
      query_source: 'message', search_result_policy: 'play_first_result', confidence: 1 };
    }
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
    signal.throwIfAborted();
    if (this.profile === 'light') return { best_track: 0, confidence: 1 };
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
    signal.throwIfAborted();
    if (this.profile === 'light') return { search_result_policy: 'play_first_result', confidence: 1 };
    const result = await this.request({ op: 'music_policy', query, candidates }, signal, true);
    if (!result || typeof result !== 'object') throw new Error('Invalid music policy');
    const policy = result as MusicResultPolicy;
    if (!['play_first_result', 'rerank_results', 'no_result'].includes(policy.search_result_policy)
      || typeof policy.confidence !== 'number' || !Number.isFinite(policy.confidence)
      || policy.confidence < 0 || policy.confidence > 1) throw new Error('Invalid music policy');
    return policy;
  }
}
