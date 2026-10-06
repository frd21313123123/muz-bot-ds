import 'dotenv/config';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import { VoiceRuntime } from '../src/voice/runtime.js';
import { normalizeSpeech } from '../src/voice/intents.js';
import { requireFfmpeg } from '../src/utils/stream.js';

// Only bundled synthetic speech is uploaded. No Discord connection or user audio.
const runtime = new VoiceRuntime({ profile: 'light', sttProvider: 'groq' });
try {
  assert.equal(await runtime.start(), true, 'Configure GROQ_API_KEY in .env');
  for (const [key, expected, wakeName] of [['pause', 'пауза', 'Пауза'], ['skip', 'следующий', undefined]] as const) {
    const stereo = gunzipSync(await readFile(path.resolve(`assets/tts/${key}.pcm.gz`)));
    const decoded = spawnSync(requireFfmpeg(), ['-hide_banner', '-loglevel', 'error', '-f', 's16le',
      '-ar', '48000', '-ac', '2', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'],
    { input: stereo, windowsHide: true, timeout: 10_000, maxBuffer: 1_000_000 });
    assert.equal(decoded.status, 0, 'Cannot resample bundled speech');
    const started = performance.now();
    const transcript = await runtime.transcribe(decoded.stdout, new AbortController().signal, !wakeName, wakeName);
    assert.ok(normalizeSpeech(transcript) === expected, `Groq bundled ${key} recognition failed`);
    console.log(`Groq ${runtime.sttModelName}: ${wakeName ? 'wake' : 'command'} PASS (${Math.round(performance.now() - started)} ms)`);
  }
} finally { runtime.close(); }
