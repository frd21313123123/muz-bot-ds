import type { SpeechMetrics } from './session.js';

export const GROQ_STT_MODEL = 'whisper-large-v3-turbo';
export type SttProvider = 'groq' | 'local';
export function sttProvider(value = process.env.STT_PROVIDER): SttProvider {
  const provider = value?.trim().toLowerCase() || 'groq';
  if (provider !== 'groq' && provider !== 'local') throw new Error('STT_PROVIDER: groq или local.');
  return provider;
}

interface Request {
  pcm: Buffer;
  wakeName?: string;
  priority: boolean;
  signal: AbortSignal;
  diagnostic?: (metrics: SpeechMetrics) => void;
  resolve(text: string): void;
  reject(error: Error): void;
  cleanup(): void;
  controller: AbortController;
}

// Discord's decoder produces mono 16 kHz signed PCM; wrap it without disk I/O.
export function speechWav(pcm: Buffer): Buffer {
  if (!pcm.length || pcm.length > 480_000 || pcm.length % 2) throw new Error('Invalid audio length');
  const header = Buffer.alloc(44);
  header.write('RIFF'); header.writeUInt32LE(pcm.length + 36, 4); header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(16_000, 24); header.writeUInt32LE(32_000, 28);
  header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write('data', 36); header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

export class GroqSpeech {
  private waiting: Request[] = [];
  private active: Request | null = null;
  private cooldownUntil = 0;
  constructor(private readonly options: { apiKey?: string; fetch?: typeof fetch; timeoutMs?: number } = {}) {}
  get configured(): boolean { return Boolean((this.options.apiKey ?? process.env.GROQ_API_KEY)?.trim()); }
  get busy(): boolean { return this.active !== null || this.waiting.length > 0; }

  transcribe(pcm: Buffer, signal: AbortSignal, priority: boolean, wakeName?: string,
    diagnostic?: (metrics: SpeechMetrics) => void): Promise<string> {
    signal.throwIfAborted();
    // Validate before queueing; digital silence never needs a paid request.
    speechWav(pcm);
    if (!this.configured) return Promise.reject(new Error('Укажите GROQ_API_KEY в .env.'));
    if (Date.now() < this.cooldownUntil) return Promise.reject(new Error('Лимит Groq STT; повторите позже.'));
    if (pcm.every(value => value === 0)) {
      diagnostic?.({ vadMs: 0, segments: 0, rejectedSegments: 0 });
      return Promise.resolve('');
    }
    return new Promise((resolve, reject) => {
      const controller = new AbortController();
      const abort = (): void => {
        const index = this.waiting.indexOf(request);
        if (index >= 0) this.waiting.splice(index, 1);
        controller.abort(); request.cleanup(); reject(new Error('Голосовой запрос отменён.'));
      };
      const request: Request = { pcm, signal, priority, wakeName, diagnostic, resolve, reject, controller,
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
      const index = priority ? this.waiting.findIndex(item => !item.priority) : -1;
      if (index >= 0) this.waiting.splice(index, 0, request); else this.waiting.push(request);
      this.pump();
    });
  }

  close(): void {
    const error = new Error('Голосовой обработчик остановлен.');
    for (const request of [this.active, ...this.waiting]) {
      if (request) { request.controller.abort(); request.cleanup(); request.reject(error); }
    }
    this.waiting = [];
  }

  private pump(): void {
    if (this.active) return;
    const request = this.waiting.shift();
    if (!request) return;
    this.active = request;
    void this.send(request).then(request.resolve, request.reject).finally(() => {
      request.cleanup(); this.active = null; this.pump();
    });
  }

  private async send(request: Request): Promise<string> {
    try {
      request.controller.signal.throwIfAborted();
      if (Date.now() < this.cooldownUntil) throw new Error('Лимит Groq STT; повторите позже.');
      const form = new FormData();
      form.set('file', new Blob([new Uint8Array(speechWav(request.pcm))], { type: 'audio/wav' }), 'speech.wav');
      form.set('model', GROQ_STT_MODEL); form.set('response_format', 'verbose_json');
      form.set('language', 'ru'); form.set('temperature', '0');
      // Avoid a single-name prompt that makes Whisper hallucinate a wake word.
      if (request.wakeName?.trim().toLowerCase() === 'бот') form.set('prompt', 'Бот, вот, кот, год, рот, борт, порт.');
      const signal = AbortSignal.any([request.controller.signal, AbortSignal.timeout(this.options.timeoutMs ?? 15_000)]);
      const response = await (this.options.fetch ?? fetch)('https://api.groq.com/openai/v1/audio/transcriptions', {
        method: 'POST', headers: { Authorization: `Bearer ${(this.options.apiKey ?? process.env.GROQ_API_KEY)!.trim()}` },
        body: form, signal,
      });
      if (!response.ok) {
        if (response.status === 429) {
          const raw = response.headers.get('retry-after');
          const seconds = raw && /^\d+(\.\d+)?$/.test(raw) ? Number(raw) : 0;
          const delay = seconds ? seconds * 1000 : raw ? Date.parse(raw) - Date.now() : 30_000;
          this.cooldownUntil = Date.now() + Math.min(300_000, Math.max(1000, Number.isFinite(delay) ? delay : 30_000));
        }
        // Never include response bodies, transcripts or credentials in errors.
        await response.body?.cancel();
        throw new Error(`Groq STT: HTTP ${response.status}.`);
      }
      const value: unknown = await response.json();
      signal.throwIfAborted();
      if (!value || typeof value !== 'object') throw new Error('Invalid Groq transcription');
      const reply = value as { text?: unknown; segments?: unknown };
      if (typeof reply.text !== 'string' || reply.text.length > 1000 || !Array.isArray(reply.segments)
        || reply.segments.length > 100) throw new Error('Invalid Groq transcription');
      let speechMs = 0, rejectedSegments = 0;
      const text: string[] = [];
      for (const segment of reply.segments) {
        if (!segment || typeof segment !== 'object' || typeof segment.text !== 'string'
          || ![segment.start, segment.end, segment.no_speech_prob, segment.avg_logprob, segment.compression_ratio]
            .every(value => typeof value === 'number' && Number.isFinite(value))
          || segment.start < 0 || segment.end < segment.start || segment.no_speech_prob < 0 || segment.no_speech_prob > 1) {
          throw new Error('Invalid Groq segment');
        }
        if (segment.no_speech_prob >= 0.6 || segment.avg_logprob < -1 || segment.compression_ratio > 2.4) {
          rejectedSegments++; continue;
        }
        text.push(segment.text.trim()); speechMs += (segment.end - segment.start) * 1000;
      }
      const transcript = text.join(' ').trim();
      if (transcript.length > 1000) throw new Error('Invalid Groq transcription');
      request.diagnostic?.({ vadMs: Math.min(Math.round(request.pcm.length / 32), Math.round(speechMs)),
        segments: reply.segments.length, rejectedSegments });
      return transcript;
    } catch (error) {
      if (request.controller.signal.aborted) throw new Error('Голосовой запрос отменён.');
      if (error instanceof Error && /^(Groq STT: HTTP \d{3}\.|Invalid Groq (transcription|segment)|Лимит Groq STT; повторите позже\.)$/.test(error.message)) throw error;
      throw new Error('Groq STT недоступен или превышено время ожидания.');
    }
  }
}
