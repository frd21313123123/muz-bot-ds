import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { VoiceRuntime } from '../src/voice/runtime.js';

const fake = (extra: string[] = []) => new VoiceRuntime({ command: process.execPath,
  args: [path.resolve('tests/fixtures/voice-worker.mjs'), ...extra], prepared: true, requestTimeoutMs: 300 });

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
  try {
    assert.equal(await runtime.start(), true);
    const signal = new AbortController().signal;
    await Promise.all([
      assert.rejects(runtime.transcribe(Buffer.from([1, 0]), signal, false)),
      assert.rejects(runtime.classify('Следующий', signal)),
    ]);
    assert.equal(unavailable, 1); assert.equal(runtime.ready, false);
  } finally { runtime.close(); }
});
