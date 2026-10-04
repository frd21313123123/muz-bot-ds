import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import phrases from '../src/voice/confirmations.json' with { type: 'json' };
import { LocalTts, type Confirmation } from '../src/voice/tts.js';
import { gzipSync } from 'node:zlib';

test('bundled short replies load from gzip without Python and corrupted gzip disables them', async () => {
  const bundled = new LocalTts();
  assert.equal(await bundled.load(path.resolve('assets/tts')), true);
  for (const key of Object.keys(phrases) as Confirmation[]) {
    const audio = bundled.audio(key)!;
    assert.ok(audio.length > 48000 * 4 * 0.3 && audio.length < 48000 * 4 * 2, key);
  }
  const directory = await mkdtemp(path.join(os.tmpdir(), 'muz-bundled-tts-'));
  try {
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify({ engine: 'bundled', voice: 'test', sampleRate: 48000, channels: 2, phrases }));
    for (const key of Object.keys(phrases)) await writeFile(path.join(directory, `${key}.pcm.gz`), gzipSync(Buffer.alloc(4800)));
    assert.equal(await bundled.load(directory), true);
    await writeFile(path.join(directory, 'pause.pcm.gz'), Buffer.from('broken gzip'));
    assert.equal(await bundled.load(directory), false); assert.equal(bundled.audio('play'), null);
    await writeFile(path.join(directory, 'pause.pcm.gz'), gzipSync(Buffer.alloc(48000 * 4 * 13)));
    assert.equal(await bundled.load(directory), false, 'oversized decompressed audio is rejected');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('local confirmations load without a worker; missing, stale or invalid caches disable TTS safely', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'muz-tts-'));
  const tts = new LocalTts();
  const manifest = { engine: 'piper', voice: 'ru_RU-irina-medium', sampleRate: 48000, channels: 2, phrases };
  try {
    assert.equal(await tts.load(directory), false);
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify(manifest));
    for (const key of Object.keys(phrases)) await writeFile(path.join(directory, `${key}.pcm`), Buffer.alloc(4800));
    assert.equal(await tts.load(directory), true);
    assert.equal(tts.ready, true);
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify({ ...manifest, engine: 'silero', voice: 'v5_5_ru/xenia' }));
    assert.equal(await tts.load(directory), true);
    assert.equal(tts.voice, 'v5_5_ru/xenia');
    const legacyPhrases = Object.fromEntries(Object.entries(phrases).filter(([key]) => !key.startsWith('radio_')));
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify({ ...manifest, phrases: legacyPhrases }));
    assert.equal(await tts.load(directory), true, 'new radio replies must not disable a valid old music cache');
    assert.ok(tts.audio('play')); assert.equal(tts.audio('radio_play'), null);
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify(manifest));
    assert.equal(await tts.load(directory), true);
    for (const key of Object.keys(phrases) as Confirmation[]) assert.equal(tts.audio(key)?.length, 4800);
    await writeFile(path.join(directory, 'pause.pcm'), Buffer.alloc(3));
    assert.equal(await tts.load(directory), false);
    assert.equal(tts.audio('play'), null, 'no partial cache may survive a failed reload');
    await writeFile(path.join(directory, 'pause.pcm'), Buffer.alloc(4800));
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify({ ...manifest, phrases: { play: 'Stale reply' } }));
    assert.equal(await tts.load(directory), false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
