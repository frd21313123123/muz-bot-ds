import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import phrases from '../src/voice/confirmations.json' with { type: 'json' };
import { LocalTts, type Confirmation } from '../src/voice/tts.js';

test('local confirmations load without a worker; missing, stale or invalid caches disable TTS safely', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'muz-tts-'));
  const tts = new LocalTts();
  const manifest = { engine: 'piper', sampleRate: 48000, channels: 2, phrases };
  try {
    assert.equal(await tts.load(directory), false);
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify(manifest));
    for (const key of Object.keys(phrases)) await writeFile(path.join(directory, `${key}.pcm`), Buffer.alloc(4800));
    assert.equal(await tts.load(directory), true);
    assert.equal(tts.ready, true);
    for (const key of Object.keys(phrases) as Confirmation[]) assert.equal(tts.audio(key)?.length, 4800);
    await writeFile(path.join(directory, 'pause.pcm'), Buffer.alloc(3));
    assert.equal(await tts.load(directory), false);
    assert.equal(tts.audio('play'), null, 'no partial cache may survive a failed reload');
    await writeFile(path.join(directory, 'pause.pcm'), Buffer.alloc(4800));
    await writeFile(path.join(directory, 'ready.json'), JSON.stringify({ ...manifest, phrases: { play: 'Stale reply' } }));
    assert.equal(await tts.load(directory), false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
