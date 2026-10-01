import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import os from 'node:os';
import { once } from 'node:events';
import { mkdtemp, rm, rmdir } from 'node:fs/promises';
import { VoiceRuntime } from '../src/voice/runtime.js';

const fake = (extra: string[] = [], options: { autoRestart?: boolean; recoveryDelayMs?: number; requestTimeoutMs?: number } = {}) => new VoiceRuntime({ command: process.execPath,
  args: [path.resolve('tests/fixtures/voice-worker.mjs'), ...extra], prepared: true, requestTimeoutMs: 300, ...options });

test('ASR diagnostics expose only numeric speech metrics', async () => {
  const runtime = fake(['--metrics']);
  try {
    assert.equal(await runtime.start(), true);
    let metrics: unknown;
    const text = await runtime.transcribe(Buffer.from([0, 0]), new AbortController().signal, false, 'Бот',
      (value) => { metrics = value; });
    assert.equal(text, 'Бот');
    assert.deepEqual(metrics, { vadMs: 320, segments: 1, rejectedSegments: 0 });
  } finally { runtime.close(); }
});

test('music worker operations validate routes and selected indices', async () => {
  const candidates = [0, 1].map((index) => ({ index, title: 'Numb', artist: 'Linkin Park', duration: '3:00' }));
  for (const args of [[], ['--bad-index']]) {
    const runtime = fake(args);
    try {
      assert.equal(await runtime.start(), true);
      const signal = new AbortController().signal;
      assert.equal((await runtime.decideMusic({ message: 'включи Numb live', selected_track: null }, signal)).search_result_policy, 'rerank_results');
      if (args.length) await assert.rejects(runtime.rerankMusic('Numb live', candidates, signal));
      else assert.equal((await runtime.rerankMusic('Numb live', candidates, signal)).best_track, 1);
    } finally { runtime.close(); }
  }
});

test('v3 worker exposes its model identity, deferred policy and explicit no-match', async () => {
  const runtime = fake(['--v3', '--none']);
  try {
    assert.equal(await runtime.start(), true); assert.equal(runtime.modelName, 'laya-muz-bot-ds-v3');
    assert.equal(runtime.sttModelName, 'large-v3-turbo'); assert.equal(runtime.wakeModelName, 'small');
    assert.equal(runtime.inferenceDevice, 'cuda');
    const signal = new AbortController().signal;
    const candidates = [{ index: 0, title: 'Numb', artist: 'Linkin Park', duration: '3:00' }];
    assert.equal((await runtime.decideMusic({ message: 'включи Numb live', selected_track: null }, signal)).defer_result_policy, true);
    assert.equal((await runtime.decideMusicResults('Numb live', candidates, signal)).search_result_policy, 'rerank_results');
    assert.deepEqual(await runtime.rerankMusic('Numb live', candidates, signal), { best_track: -1, no_match: true, confidence: 0.99 });
  } finally { runtime.close(); }
  assert.equal(runtime.modelName, null);
  assert.equal(runtime.sttModelName, null); assert.equal(runtime.wakeModelName, null);
  assert.equal(runtime.inferenceDevice, null);
});

test('persistent worker starts once, prioritizes commands and drops cancelled queued audio', async () => {
  const runtime = fake();
  try {
    assert.deepEqual(await Promise.all([runtime.start(), runtime.start()]), [true, true]);
    const results: string[] = [];
    const signal = new AbortController();
    const first = runtime.transcribe(Buffer.from([1, 0]), signal.signal, false).then((value) => results.push(value));
    const background = runtime.transcribe(Buffer.from([2, 0]), signal.signal, false).then((value) => results.push(value));
    const cancelled = new AbortController();
    const discarded = runtime.transcribe(Buffer.from([3, 0]), cancelled.signal, false);
    const rejected = assert.rejects(discarded);
    cancelled.abort();
    const command = runtime.transcribe(Buffer.from([4, 0]), signal.signal, true).then((value) => results.push(value));
    await Promise.all([first, background, command, rejected]);
    assert.deepEqual(results, ['1', '4', '2']);
  } finally { runtime.close(); }
  assert.equal(runtime.ready, false);
});

test('worker timeout rejects pending requests and reports unavailability', async () => {
  const runtime = fake(['--hang']);
  let unavailable = 0;
  runtime.on('unavailable', () => { unavailable++; });
  const failure = once(runtime, 'failure');
  try {
    assert.equal(await runtime.start(), true);
    const signal = new AbortController().signal;
    await Promise.all([
      assert.rejects(runtime.transcribe(Buffer.from([1, 0]), signal, false)),
      assert.rejects(runtime.classify('Следующий', signal)),
    ]);
    assert.equal(unavailable, 1); assert.equal(runtime.ready, false);
    const [event] = await failure;
    assert.equal(event.reason, 'request_timeout'); assert.equal(event.operation, 'transcribe');
    assert.ok(event.elapsedMs >= 250); assert.equal(JSON.stringify(event).includes('Следующий'), false);
  } finally { runtime.close(); }
});

test('a failed worker restarts once and subsequent commands use the replacement', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'muz-voice-recovery-'));
  const marker = path.join(directory, 'first-launch');
  const runtime = fake(['--crash-once', marker], { autoRestart: true, recoveryDelayMs: 20 });
  try {
    assert.equal(await runtime.start(), true);
    const failure = once(runtime, 'failure'); const recovered = once(runtime, 'available');
    await assert.rejects(runtime.transcribe(Buffer.from([1, 0]), new AbortController().signal, false));
    const [event] = await failure;
    assert.equal(event.reason, 'worker_exit'); assert.equal(event.exitCode, 17);
    await recovered; assert.equal(runtime.ready, true);
    assert.equal((await runtime.classify('Следующий', new AbortController().signal)).action, 'skip');
  } finally { runtime.close(); await rm(marker, { force: true }); await rmdir(directory); }
});

test('explicit close cancels scheduled worker recovery', async () => {
  const runtime = fake(['--hang'], { autoRestart: true, recoveryDelayMs: 80, requestTimeoutMs: 30 });
  let launches = 0; runtime.on('available', () => { launches++; });
  try {
    assert.equal(await runtime.start(), true);
    await assert.rejects(runtime.classify('Следующий', new AbortController().signal));
    runtime.close(); await new Promise(resolve => setTimeout(resolve, 150));
    assert.equal(launches, 1); assert.equal(runtime.ready, false);
  } finally { runtime.close(); }
});

test('recovery stops after three attempts when every replacement hangs', async () => {
  const runtime = fake(['--hang'], { autoRestart: true, recoveryDelayMs: 5, requestTimeoutMs: 30 });
  let launches = 0; const attempts: number[] = [];
  runtime.on('available', () => { launches++; void runtime.classify('Следующий', new AbortController().signal).catch(() => {}); });
  runtime.on('recovering', (event) => attempts.push(event.attempt));
  try {
    assert.equal(await runtime.start(), true);
    const deadline = Date.now() + 2500;
    while ((launches < 4 || runtime.ready) && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
    await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal(launches, 4); assert.deepEqual(attempts, [1, 2, 3]); assert.equal(runtime.ready, false);
  } finally { runtime.close(); }
});
