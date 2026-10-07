import 'dotenv/config';
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { requireFfmpeg } from '../src/utils/stream.js';
import { contextualCommand, validateIntent, isWakePhrase, normalizeSpeech } from '../src/voice/intents.js';
import { extractMusicRequest } from '../src/voice/music.js';

const option = (name: string, fallback: string) => process.argv.find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const manifest = option('fixtures', process.env.VOICE_TEST_FIXTURES ?? '');
assert.ok(manifest, 'Provide --fixtures=path to an authorized WAV manifest');
const bytes = await readFile(manifest);
const fixtures = JSON.parse(bytes.toString()) as { file: string; action?: string; level?: number; paused?: boolean;
  wakeName?: string; shouldWake?: boolean; musicQuery?: string }[];
assert.ok(Array.isArray(fixtures) && fixtures.length > 0 && fixtures.length <= 50);
const implementations = option('implementations', 'dist').split(',').map(value => path.resolve(value));
const reportPath = path.resolve(option('output', '.runtime/performance/voice-regression.json'));
const report: { datasetSha256: string; versions: { index: number; wakeBot: boolean | null; results: { index: number; correct: boolean; ms: number; error?: true }[] }[] } = {
  datasetSha256: createHash('sha256').update(bytes).digest('hex'), versions: [],
};
let previousRequest = 0;
for (const [index, implementation] of implementations.entries()) {
  const { VoiceRuntime } = await import(pathToFileURL(path.join(implementation, 'src/voice/runtime.js')).href) as typeof import('../src/voice/runtime.js');
  const runtime = new VoiceRuntime({ profile: 'light', sttProvider: 'groq' });
  const version: typeof report.versions[number] = { index, wakeBot: null, results: [] };
  report.versions.push(version);
  try {
    assert.equal(await runtime.start(), true);
    version.wakeBot = runtime.supportsWake?.('бот') ?? null;
    for (const [fixtureIndex, fixture] of fixtures.entries()) {
      assert.equal(typeof fixture.file, 'string');
      const decoded = spawnSync(requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error', '-i',
        path.resolve(path.dirname(manifest), fixture.file), '-t', '15', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'],
      { windowsHide: true, timeout: 10_000, maxBuffer: 1_000_000 });
      assert.equal(decoded.status, 0, 'Fixture decode failed');
      await delay(Math.max(0, 4000 - (performance.now() - previousRequest)));
      previousRequest = performance.now();
      const started = performance.now();
      try {
        const signal = AbortSignal.timeout(20_000);
        const transcript = await runtime.transcribe(decoded.stdout, signal, !fixture.wakeName, fixture.wakeName);
        const text = contextualCommand(transcript, fixture.paused ?? false);
        let correct: boolean;
        if (fixture.wakeName) correct = isWakePhrase(text, fixture.wakeName) === (fixture.shouldWake ?? true);
        else if (fixture.musicQuery) {
          const request = extractMusicRequest(text, true);
          correct = request !== null && normalizeSpeech(request.query) === normalizeSpeech(fixture.musicQuery);
        } else {
          const intent = validateIntent(text, await runtime.classify(text, signal));
          correct = intent.action === fixture.action && (fixture.level === undefined || (intent.action === 'volume_set' && intent.level === fixture.level));
        }
        version.results.push({ index: fixtureIndex, correct, ms: performance.now() - started });
      } catch { version.results.push({ index: fixtureIndex, correct: false, ms: performance.now() - started, error: true }); }
    }
  } finally { await runtime.close(); }
  console.log(JSON.stringify({ version: index, correct: version.results.filter(result => result.correct).length, total: fixtures.length, wakeBot: version.wakeBot }));
}
await mkdir(path.dirname(reportPath), { recursive: true });
await writeFile(reportPath, JSON.stringify(report, null, 2));
if (report.versions.some(version => version.results.some(result => !result.correct))) process.exitCode = 1;
