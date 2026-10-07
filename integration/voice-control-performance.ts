import 'dotenv/config';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { gunzipSync } from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { AudioPlayerStatus, VoiceConnectionStatus, StreamType, entersState, type AudioPlayer, type VoiceConnection } from '@discordjs/voice';
import { requireFfmpeg } from '../src/utils/stream.js';
import { LocalTts } from '../src/voice/tts.js';
import { validateIntent, normalizeSpeech } from '../src/voice/intents.js';
import type { VoiceHost } from '../src/voice/session.js';
import type { MusicClient } from '../src/types.js';

// Native player/Opus/TTS with a packet sink, without connecting to Discord.
// Synthetic PCM isolates command timing from YouTube extraction failures.
const option = (name: string, fallback: string) => process.argv.find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const implementations = option('implementations', 'dist').split(',').map(value => path.resolve(value));
const rounds = Number(option('rounds', '30'));
assert.ok(Number.isInteger(rounds) && rounds >= 1 && rounds <= 100);
const output = path.resolve(option('output', '.runtime/performance/voice-control.json'));
const tts = new LocalTts(); assert.equal(await tts.load(path.resolve('assets/tts')), true);
const source = gunzipSync(await readFile('assets/tts/skip.pcm.gz'));
const decoded = spawnSync(requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 's16le', '-ar', '48000', '-ac', '2',
  '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], { input: source, windowsHide: true, timeout: 10_000 });
assert.equal(decoded.status, 0);
const reports: { index: number; samplesMs: number[]; failures: number; packets: number }[] = [];
let previousRequest = 0;
for (const [index, implementation] of implementations.entries()) {
  const load = <T>(name: string): Promise<T> => import(pathToFileURL(path.join(implementation, name)).href) as Promise<T>;
  const { VoiceRuntime } = await load<typeof import('../src/voice/runtime.js')>('src/voice/runtime.js');
  const { GuildQueue } = await load<typeof import('../src/utils/GuildQueue.js')>('src/utils/GuildQueue.js');
  const { GuildVoice } = await load<typeof import('../src/voice/GuildVoice.js')>('src/voice/GuildVoice.js');
  const runtime = new VoiceRuntime({ profile: 'light', sttProvider: 'groq' });
  const report = { index, samplesMs: [] as number[], failures: 0, packets: 0 }; reports.push(report);
  try {
    assert.equal(await runtime.start(), true);
    for (let iteration = 0; iteration < rounds; iteration++) {
      const queue = new GuildQueue(`voice-benchmark-${index}-${iteration}`, { queues: new Map(), voiceTts: tts } as unknown as MusicClient, () => {
        function* frames(): Generator<Buffer> { for (let i = 0; i < 3000; i++) yield Buffer.alloc(3840); }
        const stream = Readable.from(frames(), { objectMode: false, highWaterMark: 3840 });
        return { stream, type: StreamType.Raw, destroy: () => { stream.destroy(); } };
      });
      let subscription: { unsubscribe(): void } | undefined;
      const connection = {
        state: { status: VoiceConnectionStatus.Ready },
        onSubscriptionRemoved: () => {}, setSpeaking: () => {},
        prepareAudioPacket: () => { report.packets++; }, dispatchAudio: () => true,
        subscribe: (player: AudioPlayer) => {
          subscription?.unsubscribe();
          subscription = Reflect.apply(Reflect.get(player, 'subscribe'), player, [connection]); return subscription;
        },
        destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
      } as unknown as VoiceConnection;
      queue.connection = connection; connection.subscribe(queue.player);
      const control = new GuildVoice(queue, runtime);
      try {
        const track = { url: 'https://www.youtube.com/watch?v=aaaaaaaaaaa', videoId: 'aaaaaaaaaaa', title: 'Synthetic benchmark',
          duration: '0:10', thumbnail: null, requestedBy: 'Benchmark' };
        await queue.addTracks([track, { ...track, videoId: 'bbbbbbbbbbb', url: 'https://www.youtube.com/watch?v=bbbbbbbbbbb' }]);
        await entersState(queue.player, AudioPlayerStatus.Playing, 3000);
        const preparationDeadline = performance.now() + 3000;
        while (!Reflect.get(queue, 'prepared') && performance.now() < preparationDeadline) await delay(10);
        assert.ok(Reflect.get(queue, 'prepared'));
        await delay(Math.max(0, 4000 - (performance.now() - previousRequest)));
        previousRequest = performance.now();
        const started = performance.now();
        const signal = AbortSignal.timeout(20_000);
        const text = await runtime.transcribe(decoded.stdout, signal, true);
        assert.equal(normalizeSpeech(text), 'следующий');
        const intent = validateIntent(text, await runtime.classify(text, signal)); assert.equal(intent.action, 'skip');
        const oldResource = Reflect.get(queue, 'activeResource');
        await (Reflect.get(control.session, 'host') as VoiceHost).execute(intent, signal);
        const deadline = performance.now() + 5000;
        while (Reflect.get(queue, 'activeResource') === oldResource && performance.now() < deadline) await delay(10);
        assert.notEqual(Reflect.get(queue, 'activeResource'), oldResource);
        await entersState(queue.player, AudioPlayerStatus.Playing, 3000);
        report.samplesMs.push(performance.now() - started);
      } catch { report.failures++; }
      finally { control.destroy(); await queue.stop(); }
    }
  } finally { await runtime.close(); }
  const sorted = [...report.samplesMs].sort((a, b) => a - b);
  console.log(JSON.stringify({ version: index, count: sorted.length, medianMs: sorted[Math.ceil(sorted.length * .5) - 1],
    p95Ms: sorted[Math.ceil(sorted.length * .95) - 1], failures: report.failures, packets: report.packets }));
}
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, JSON.stringify({ rounds, reports }, null, 2));
if (reports.some(report => report.failures)) process.exitCode = 1;
