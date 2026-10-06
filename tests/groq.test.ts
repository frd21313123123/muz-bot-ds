import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { once } from 'node:events';
import { GroqSpeech, GROQ_STT_MODEL, speechWav, sttProvider } from '../src/voice/groq.js';
import { VoiceRuntime } from '../src/voice/runtime.js';

const pcm = Buffer.alloc(32_000, 1);
const segment = (text = 'Следующий трек', extra = {}) => ({ text, start: 0, end: 0.8,
  no_speech_prob: 0.01, avg_logprob: -0.2, compression_ratio: 1, ...extra });
const reply = (segments = [segment()]) => Response.json({ text: segments.map(item => item.text).join(' '), segments });
const idle = () => new Promise(resolve => setTimeout(resolve, 0));
const signal = () => new AbortController().signal;

test('Groq defaults and WAV preserve mono 16 kHz PCM with bounded input', () => {
  assert.equal(sttProvider(''), 'groq'); assert.equal(sttProvider(' LOCAL '), 'local');
  assert.throws(() => sttProvider('unknown'));
  const wav = speechWav(pcm);
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF'); assert.equal(wav.readUInt32LE(4), wav.length - 8);
  assert.equal(wav.readUInt16LE(22), 1); assert.equal(wav.readUInt32LE(24), 16_000);
  assert.equal(wav.readUInt16LE(34), 16); assert.equal(wav.readUInt32LE(40), pcm.length);
  assert.deepEqual(wav.subarray(44), pcm);
  for (const invalid of [Buffer.alloc(0), Buffer.alloc(3), Buffer.alloc(480_002)]) assert.throws(() => speechWav(invalid));
});

test('Groq sends authenticated multipart audio and filters unreliable speech before wake matching', async () => {
  let calls = 0, metrics: unknown;
  const speech = new GroqSpeech({ apiKey: 'test-only-key', fetch: async (url, init) => {
    calls++; assert.equal(url, 'https://api.groq.com/openai/v1/audio/transcriptions');
    assert.equal(init?.method, 'POST'); assert.deepEqual(init?.headers, { Authorization: 'Bearer test-only-key' });
    const form = init?.body as FormData;
    assert.equal(form.get('model'), GROQ_STT_MODEL); assert.equal(form.get('response_format'), 'verbose_json');
    assert.equal(form.get('language'), 'ru'); assert.equal(form.get('temperature'), '0');
    assert.equal(form.get('prompt'), 'Бот, вот, кот, год, рот, борт, порт.');
    const file = form.get('file') as File;
    assert.equal(file.name, 'speech.wav'); assert.equal(file.type, 'audio/wav');
    assert.deepEqual(Buffer.from(await file.arrayBuffer()), speechWav(pcm));
    return reply([segment('Бот'), segment('шум', { no_speech_prob: 0.7 }),
      segment('повтор', { compression_ratio: 3 }), segment('ошибка', { avg_logprob: -2 })]);
  } });
  assert.equal(await speech.transcribe(pcm, signal(), false, 'Бот', value => { metrics = value; }), 'Бот');
  assert.deepEqual(metrics, { vadMs: 800, segments: 4, rejectedSegments: 3 });
  assert.equal(await speech.transcribe(Buffer.alloc(3200), signal(), false), ''); assert.equal(calls, 1);
});

test('Groq errors omit provider payloads and apply Retry-After without retrying audio', async () => {
  let calls = 0;
  const speech = new GroqSpeech({ apiKey: 'test-only-key', fetch: async () => {
    calls++; return new Response('private transcript and test-only-key', { status: 429, headers: { 'Retry-After': '60' } });
  } });
  await assert.rejects(speech.transcribe(pcm, signal(), true), /^Error: Groq STT: HTTP 429\.$/);
  await assert.rejects(speech.transcribe(pcm, signal(), true), /Лимит Groq/); assert.equal(calls, 1);
  for (const response of [Response.json({ text: 'unexpected' }), reply([segment('x', { start: -1 })]),
    reply([segment('x'.repeat(1001))]), new Response('secret', { status: 401 })]) {
    const client = new GroqSpeech({ apiKey: 'test-only-key', fetch: async () => response });
    await assert.rejects(client.transcribe(pcm, signal(), true), error => error instanceof Error
      && !error.message.includes('secret') && !error.message.includes('test-only-key'));
  }
});

test('Groq prioritizes commands, removes cancelled queued audio and aborts active HTTP requests', async () => {
  const pending: { resolve(response: Response): void; signal: AbortSignal }[] = [];
  const speech = new GroqSpeech({ apiKey: 'test-only-key', fetch: async (_url, init) => new Promise<Response>((resolve, reject) => {
    const current = { resolve, signal: init!.signal! }; pending.push(current);
    current.signal.addEventListener('abort', () => reject(new Error('private network details')), { once: true });
  }) });
  const completed: string[] = [];
  const first = speech.transcribe(pcm, signal(), false).then(text => completed.push(text));
  const background = speech.transcribe(pcm, signal(), false).then(text => completed.push(text));
  const cancelled = new AbortController();
  const discarded = assert.rejects(speech.transcribe(pcm, cancelled.signal, false)); cancelled.abort();
  const command = speech.transcribe(pcm, signal(), true).then(text => completed.push(text));
  pending[0]!.resolve(reply([segment('first')])); await idle();
  pending[1]!.resolve(reply([segment('command')])); await idle();
  pending[2]!.resolve(reply([segment('background')]));
  await Promise.all([first, background, command, discarded]);
  assert.deepEqual(completed, ['first', 'command', 'background']); assert.equal(pending.length, 3);
  await idle();
  const controller = new AbortController();
  const aborted = assert.rejects(speech.transcribe(pcm, controller.signal, true), /отменён/);
  controller.abort(); await aborted; assert.equal(pending[3]!.signal.aborted, true);
  await idle(); assert.equal(speech.busy, false);
});

test('Groq times out HTTP calls, bounds the queue and close cancels active and pending speech', async () => {
  const hangingFetch: typeof fetch = async (_url, init) => new Promise<Response>((_resolve, reject) => {
    init!.signal!.addEventListener('abort', () => reject(new Error('private network details')), { once: true });
  });
  const timeout = new GroqSpeech({ apiKey: 'test-only-key', timeoutMs: 15, fetch: hangingFetch });
  // AbortSignal.timeout uses an unref'ed timer, as does Node's native fetch.
  const keepAlive = setTimeout(() => {}, 1000);
  try { await assert.rejects(timeout.transcribe(pcm, signal(), true), /время ожидания/); }
  finally { clearTimeout(keepAlive); }
  const speech = new GroqSpeech({ apiKey: 'test-only-key', fetch: hangingFetch });
  const pending = Array.from({ length: 13 }, () => assert.rejects(speech.transcribe(pcm, signal(), false)));
  await assert.rejects(speech.transcribe(pcm, signal(), false), /занят/);
  pending.push(assert.rejects(speech.transcribe(pcm, signal(), true)));
  speech.close(); await Promise.all(pending); await idle(); assert.equal(speech.busy, false);
});

test('cloud light starts without Python, survives API failures and closes in-flight STT', async () => {
  let calls = 0;
  const runtime = new VoiceRuntime({ profile: 'light', sttProvider: 'groq', command: 'nonexistent-python',
    groq: { apiKey: 'test-only-key', fetch: async () => ++calls === 1 ? new Response('secret', { status: 503 }) : reply() } });
  try {
    assert.equal(runtime.prepared, true); assert.equal(await runtime.start(), true);
    assert.equal(runtime.sttModelName, GROQ_STT_MODEL); assert.equal(runtime.wakeModelName, GROQ_STT_MODEL);
    assert.equal(runtime.modelName, 'rules'); assert.equal(runtime.inferenceDevice, null);
    await assert.rejects(runtime.transcribe(pcm, signal(), true), /HTTP 503/); assert.equal(runtime.ready, true);
    assert.equal(await runtime.transcribe(pcm, signal(), true), 'Следующий трек');
  } finally { runtime.close(); }
  assert.equal(runtime.ready, false); assert.equal(runtime.sttModelName, null);
  const unconfigured = new VoiceRuntime({ profile: 'light', sttProvider: 'groq', groq: { apiKey: '' } });
  assert.equal(unconfigured.prepared, false); assert.equal(await unconfigured.start(), false); unconfigured.close();
});

test('cloud lifecycle waits for in-flight STT when idle and aborts it on explicit close', async () => {
  let finish: ((response: Response) => void) | undefined;
  let requestSignal: AbortSignal | undefined;
  const runtime = new VoiceRuntime({ profile: 'light', sttProvider: 'groq', idleTimeoutMs: 20,
    groq: { apiKey: 'test-only-key', fetch: async (_url, init) => new Promise<Response>((resolve, reject) => {
      finish = resolve; requestSignal = init!.signal!;
      requestSignal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    }) } });
  try {
    assert.equal(await runtime.start(), true);
    const transcription = runtime.transcribe(pcm, signal(), true);
    await new Promise(resolve => setTimeout(resolve, 50)); assert.equal(runtime.ready, true);
    const unloaded = once(runtime, 'unavailable');
    finish!(reply()); assert.equal(await transcription, 'Следующий трек');
    await Promise.all([unloaded, new Promise(resolve => setTimeout(resolve, 50))]); assert.equal(runtime.ready, false);
    assert.equal(await runtime.start(), true);
    const cancelled = assert.rejects(runtime.transcribe(pcm, signal(), true));
    runtime.close(); await cancelled; assert.equal(requestSignal!.aborted, true);
  } finally { runtime.close(); }
});

test('heavy cloud uses Groq for speech while keeping the local worker for decisions', async () => {
  const runtime = new VoiceRuntime({ profile: 'heavy', sttProvider: 'groq', prepared: true,
    command: process.execPath, args: [path.resolve('tests/fixtures/voice-worker.mjs')],
    groq: { apiKey: 'test-only-key', fetch: async () => reply() } });
  try {
    assert.equal(await runtime.start(), true);
    assert.equal(runtime.sttModelName, GROQ_STT_MODEL);
    assert.equal(await runtime.transcribe(pcm, signal(), true), 'Следующий трек');
    assert.equal((await runtime.classify('Следующий трек', signal())).action, 'skip');
  } finally { runtime.close(); }
});
