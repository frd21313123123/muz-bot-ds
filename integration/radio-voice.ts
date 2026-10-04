import 'dotenv/config';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { VoiceRuntime } from '../src/voice/runtime.js';
import { requireFfmpeg } from '../src/utils/stream.js';
import { extractRadioRequest, findRadioStation } from '../src/utils/radio.js';

// Synthetic speech can check ASR plumbing; real Discord voices still need an
// end-to-end test. Fixtures are opt-in and transcripts are never printed.
const fixturePath = process.env.RADIO_VOICE_TEST_FIXTURES ?? process.argv[2];
if (!fixturePath) throw new Error('Укажите RADIO_VOICE_TEST_FIXTURES или путь к JSON: [{file, stationId}].');
const fixtures: unknown = JSON.parse(await readFile(fixturePath, 'utf8'));
assert.ok(Array.isArray(fixtures) && fixtures.length > 0 && fixtures.length <= 50, 'Некорректный список WAV-примеров.');
const runtime = new VoiceRuntime({ startupTimeoutMs: 180_000 });
try {
  assert.equal(await runtime.start(), true, 'Локальные модели недоступны; выполните setup:voice.');
  console.log(`Whisper ${runtime.sttModelName}; Laya ${runtime.modelName}; ${runtime.inferenceDevice}`);
  let passed = 0;
  for (const [index, fixture] of fixtures.entries()) {
    assert.ok(fixture && typeof fixture.file === 'string' && typeof fixture.stationId === 'string'
      && findRadioStation(fixture.stationId), 'Некорректный WAV-пример.');
    const decoder = spawn(requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error',
      '-i', path.resolve(path.dirname(fixturePath), fixture.file), '-t', '15', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'],
    { windowsHide: true });
    const chunks: Buffer[] = []; decoder.stdout.on('data', (chunk: Buffer) => chunks.push(chunk)); decoder.stderr.resume();
    await new Promise<void>((resolve, reject) => {
      decoder.once('error', reject); decoder.once('close', code => code === 0 ? resolve() : reject(new Error('Не удалось декодировать WAV.')));
    });
    const text = await runtime.transcribe(Buffer.concat(chunks), AbortSignal.timeout(30_000), true);
    const request = extractRadioRequest(text, true);
    const matched = request?.kind === 'station' && request.station.id === fixture.stationId;
    if (matched) passed++;
    console.log(`Пример ${index + 1}: ${matched ? 'PASS' : 'FAIL'} (${fixture.stationId})`);
    // Opt-in diagnostics for local test recordings only; production speech
    // and the default smoke-test output never include transcripts.
    if (!matched && process.env.RADIO_VOICE_TEST_DEBUG === '1') console.log(JSON.stringify({ transcript: text }));
  }
  console.log(`Распознавание радио: ${passed}/${fixtures.length}.`);
  assert.equal(passed, fixtures.length, 'Некоторые названия станций распознаны неверно.');
} finally { runtime.close(); }
