import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { VoiceBackend } from './session.js';
import type { VoiceDecision } from './intents.js';

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

export class VoiceRuntime extends EventEmitter implements VoiceBackend {
  ready = false;
  private child: ChildProcessWithoutNullStreams | null = null;
  private waiting: Request[] = [];
  private active: Request | null = null;
  private sequence = 0;
  private timeout: NodeJS.Timeout | null = null;
  private starting: Promise<boolean> | null = null;

  constructor(private readonly options: { command?: string; args?: string[]; prepared?: boolean;
    requestTimeoutMs?: number } = {}) { super(); }

  async start(): Promise<boolean> {
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
      const timer = setTimeout(() => { if (this.child === child) this.fail(); finish(false); }, 90_000);
      const lines = createInterface({ input: child.stdout });
      lines.on('line', (line) => {
        if (this.child !== child) return;
        try {
          if (line.length > 64_000) throw new Error('Invalid worker reply');
          const value: unknown = JSON.parse(line);
          if (!value || typeof value !== 'object') throw new Error('Invalid worker reply');
          const message = value as Record<string, unknown>;
          if (message.ready === true) { this.ready = true; finish(true); return; }
          const request = this.active;
          if (!request || message.id !== request.id) return;
          if (this.timeout) clearTimeout(this.timeout);
          this.timeout = null;
          this.active = null;
          request.cleanup();
          if (message.error) request.reject(new Error('Локальная модель не обработала запрос.'));
          else request.resolve(message.result);
          this.pump();
        } catch { this.fail(); finish(false); }
      });
      child.once('error', () => { if (this.child === child) this.fail(); finish(false); });
      child.once('exit', () => { if (this.child === child) this.fail(); finish(false); lines.close(); });
      child.stdin.on('error', () => { if (this.child === child) this.fail(); finish(false); });
    });
  }

  private fail(): void {
    const wasReady = this.ready;
    this.ready = false;
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
    const child = this.child;
    this.child = null;
    child?.kill();
    for (const request of [this.active, ...this.waiting]) {
      if (request) { request.cleanup(); request.reject(new Error('Голосовой обработчик недоступен.')); }
    }
    this.active = null;
    this.waiting = [];
    if (wasReady) this.emit('unavailable');
  }

  close(): void { this.fail(); }

  private pump(): void {
    if (!this.ready || !this.child || this.active) return;
    const request = this.waiting.shift();
    if (!request) return;
    this.active = request;
    this.timeout = setTimeout(() => this.fail(), this.options.requestTimeoutMs ?? 30_000);
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

  async transcribe(pcm: Buffer, signal: AbortSignal, command: boolean, wakeName?: string): Promise<string> {
    if (pcm.length > 16_000 * 2 * 15 || pcm.length % 2) throw new Error('Invalid audio length');
    const result = await this.request({ op: 'transcribe', pcm: pcm.toString('base64'), wakeName: wakeName?.slice(0, 64) }, signal, command);
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
}
