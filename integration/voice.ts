import 'dotenv/config';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { VoiceRuntime } from '../src/voice/runtime.js';
import { isWakePhrase, validateIntent, type VoiceAction } from '../src/voice/intents.js';
import { requireFfmpeg } from '../src/utils/stream.js';

const cases: [string, VoiceAction][] = [
  ['Следующий трек', 'skip'], ['Пропусти текущую музыку', 'skip'],
  ['Поставь на паузу', 'pause'], ['Приостанови музыку', 'pause'],
  ['Продолжи музыку', 'resume'], ['Сними с паузы', 'resume'],
  ['Останови музыку', 'stop'], ['Отключись от канала', 'stop'],
  ['Громкость пятьдесят процентов', 'volume_set'], ['Установи громкость на 150 процентов', 'volume_set'],
  ['Сделай громче', 'volume_up'], ['Сделай тише', 'volume_down'],
  ['Включи песню', 'unknown'], ['Добавь трек в очередь', 'unknown'],
  ['Не останавливай музыку', 'unknown'], ['Следующий трек и пауза', 'unknown'],
  ['Как погода сегодня', 'unknown'], ['Расскажи анекдот', 'unknown'],
  ['Мне нравится эта музыка', 'unknown'], ['Музыка на паузе', 'unknown'],
  ['Как сделать музыку громче?', 'unknown'], ['Песня называется Стоп', 'unknown'],
  ['Привет', 'unknown'], ['Спасибо', 'unknown'], ['Выключи свет', 'unknown'],
  ['Сделай паузу и продолжи', 'unknown'], ['Громкость 900', 'unknown'],
  ['Покажи очередь', 'unknown'], ['Расскажи что ты умеешь', 'unknown'],
  ['Поставь Metallica', 'unknown'],
];

async function decode(file: string): Promise<Buffer> {
  const ffmpeg = spawn(requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error',
    '-i', file, '-t', '15', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], { windowsHide: true });
  const chunks: Buffer[] = [];
  ffmpeg.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
  ffmpeg.stderr.resume();
  await new Promise<void>((resolve, reject) => {
    ffmpeg.once('error', reject);
    ffmpeg.once('close', (code) => code === 0 ? resolve() : reject(new Error('Cannot decode fixture')));
  });
  return Buffer.concat(chunks);
}

const runtime = new VoiceRuntime();
try {
  assert.equal(await runtime.start(), true, 'Run npm run setup:voice first');
  const signal = new AbortController().signal;
  let correct = 0;
  let wrongActions = 0;
  const timings: number[] = [];
  for (const [text, expected] of cases) {
    const start = performance.now();
    const decision = await runtime.classify(text, signal);
    const actual = validateIntent(text, decision).action;
    timings.push(performance.now() - start);
    if (actual === expected) correct++;
    else if (actual !== 'unknown') wrongActions++;
    console.log(`${expected}: ${actual} [model=${decision.action}, confidence=${decision.confidence.toFixed(3)}] (${Math.round(timings.at(-1)!)} ms)`);
  }
  timings.sort((a, b) => a - b);
  console.log(`Laya: ${correct}/${cases.length} correct; ${wrongActions} incorrect actions; median ${Math.round(timings[Math.floor(timings.length / 2)]!)} ms`);
  assert.equal(wrongActions, 0, 'Wrong playback actions need investigation before use');
  assert.ok(correct >= cases.length - 2, 'Too many supported commands rejected');
  // Optional local WAV fixtures: [{file, action}] or [{file, wakeName}]. Never print the transcript.
  const manifest = process.env.VOICE_TEST_FIXTURES;
  if (manifest) {
    const fixtures = JSON.parse(await readFile(manifest, 'utf8')) as { file: string; action?: VoiceAction; wakeName?: string; shouldWake?: boolean }[];
    for (const [i, fixture] of fixtures.entries()) {
      const start = performance.now();
      const text = await runtime.transcribe(await decode(path.resolve(path.dirname(manifest), fixture.file)), signal, !fixture.wakeName, fixture.wakeName);
      const passed = fixture.wakeName ? isWakePhrase(text, fixture.wakeName) === (fixture.shouldWake ?? true)
        : validateIntent(text, await runtime.classify(text, signal)).action === fixture.action;
      console.log(`Audio fixture ${i + 1}: ${passed ? 'PASS' : 'FAIL'} (${Math.round(performance.now() - start)} ms)`);
      assert.equal(passed, true, `Audio fixture ${i + 1} failed`);
    }
  } else console.log('Whisper audio check skipped: set VOICE_TEST_FIXTURES to a local WAV manifest.');
} finally { runtime.close(); }
