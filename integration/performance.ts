import 'dotenv/config';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { monitorEventLoopDelay } from 'node:perf_hooks';
import { AudioPlayerStatus, NoSubscriberBehavior, createAudioPlayer, createAudioResource, entersState } from '@discordjs/voice';
import { configureMusicEncoder } from '../src/utils/audio.js';
import { processUsage, metricSummary } from '../src/utils/performance.js';
import { normalizeSpeech, contextualCommand, validateIntent } from '../src/voice/intents.js';
import { radioStations } from '../src/utils/radio.js';
import type { ManagedAudioStream } from '../src/utils/stream.js';
import type { MusicClient } from '../src/types.js';

const option = (name: string, fallback: string) => process.argv.find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
process.env.BOT_PERF = '1';
const rounds = Number(option('rounds', '30'));
const soakSeconds = Number(option('soak-seconds', '0'));
const implementation = path.resolve(option('implementation', 'dist'));
const reportPath = path.resolve(option('output', '.runtime/performance/report.json'));
assert.ok(Number.isInteger(rounds) && rounds >= 1 && rounds <= 100);
assert.ok(Number.isInteger(soakSeconds) && soakSeconds >= 0 && soakSeconds <= 7200);
const load = <T>(relative: string): Promise<T> => import(pathToFileURL(path.join(implementation, relative)).href) as Promise<T>;
const ytdlpModule = await load<typeof import('../src/utils/ytdlp.js')>('src/utils/ytdlp.js');
const streamModule = await load<typeof import('../src/utils/stream.js')>('src/utils/stream.js');
const voiceModule = await load<typeof import('../src/voice/runtime.js')>('src/voice/runtime.js');
const queueModule = await load<typeof import('../src/utils/GuildQueue.js')>('src/utils/GuildQueue.js');
const ytdlp = await ytdlpModule.prepareYtdlp();
const samples: Record<string, number[]> = {};
const usage: { atMs: number; processes: number; rssBytes: number; cpuSeconds: number; eventLoopP95Ms: number }[] = [];
const loop = monitorEventLoopDelay({ resolution: 20 }); loop.enable();
let sampling = false;
const sampler = setInterval(() => {
  if (sampling) return;
  sampling = true;
  void processUsage().then(sample => { if (sample) usage.push({ ...sample, atMs: performance.now(), eventLoopP95Ms: loop.percentile(95) / 1e6 }); loop.reset(); })
    .catch(() => {}).finally(() => { sampling = false; });
}, 1000);
sampler.unref();
const failures: Record<string, number> = {};
const recognition: { fixture: string; correct: boolean }[] = [];
let soakElapsedMs = 0, soakAudioMs = 0;
const summary = () => Object.fromEntries(Object.entries(samples).map(([name, values]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return [name, { count: sorted.length, medianMs: sorted[Math.ceil(sorted.length * .5) - 1], p95Ms: sorted[Math.ceil(sorted.length * .95) - 1] }];
}));
const checkpoint = async (stage: string) => {
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, JSON.stringify({ stage, rounds, soakSeconds, soakElapsedMs, soakAudioMs, summary: summary(), internal: metricSummary(), failures, recognition, usage }, null, 2));
  console.log(JSON.stringify({ stage, soakElapsedMs, soakAudioMs, summary: summary(), failures }));
};
const measure = async (name: string, operation: () => Promise<void>) => {
  const started = performance.now();
  try { await operation(); (samples[name] ??= []).push(performance.now() - started); }
  catch { failures[name] = (failures[name] ?? 0) + 1; }
};
const mediaPlayer = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Play, maxMissedFrames: 1500 } });
let media: ManagedAudioStream | null = null;
const closeMedia = () => { media?.destroy(); media = null; mediaPlayer.stop(true); };
const install = async (next: ManagedAudioStream) => {
  try {
    await streamModule.waitForAudio(next, AbortSignal.timeout(20_000));
    const resource = createAudioResource(next.stream, { inputType: next.type, inlineVolume: true });
    configureMusicEncoder(resource); resource.volume?.setVolume(.5);
    mediaPlayer.play(resource);
    const previous = media; media = next; previous?.destroy();
    await entersState(mediaPlayer, AudioPlayerStatus.Playing, 20_000);
  } catch { next.destroy(); throw new Error('Audio check failed'); }
};
mediaPlayer.on('error', () => { failures['audio.stream'] = (failures['audio.stream'] ?? 0) + 1; });
const voice = new voiceModule.VoiceRuntime({ profile: 'light', sttProvider: 'groq' });
const pcm: { key: string; expected: string; audio: Buffer }[] = [];
for (const [key, expected] of [['pause', 'пауза'], ['skip', 'следующий']] as const) {
  const stereo = gunzipSync(await readFile(path.resolve(`assets/tts/${key}.pcm.gz`)));
  const result = spawnSync(streamModule.requireFfmpeg(), ['-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 's16le', '-ar', '48000', '-ac', '2',
    '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], { input: stereo, windowsHide: true, timeout: 10_000 });
  assert.equal(result.status, 0); pcm.push({ key, expected, audio: result.stdout });
}
// Exercise an actual pause action after STT, then restore continuous playback.
// Skip is measured separately with the real GuildQueue transition below.
const voiceCommand = async (fixture: typeof pcm[number]): Promise<void> => {
  const started = performance.now();
  const text = await voice.transcribe(fixture.audio, AbortSignal.timeout(15_000), true);
  const correct = normalizeSpeech(text) === fixture.expected;
  recognition.push({ fixture: fixture.key, correct }); assert.ok(correct);
  const command = contextualCommand(text, mediaPlayer.state.status === AudioPlayerStatus.Paused);
  const intent = validateIntent(command, await voice.classify(command, AbortSignal.timeout(5000)));
  assert.equal(intent.action, fixture.key);
  if (intent.action === 'pause') {
    assert.equal(mediaPlayer.pause(false), true);
    assert.equal(mediaPlayer.state.status, AudioPlayerStatus.Paused);
    (samples['voice.executed-pause'] ??= []).push(performance.now() - started);
    assert.equal(mediaPlayer.unpause(), true);
  }
};
let cancelled = false;
const stop = () => { cancelled = true; closeMedia(); void voice.close(); };
process.once('SIGINT', stop); process.once('SIGTERM', stop);
try {
  for (const state of ['idle', 'music']) {
    if (state === 'music') await install(streamModule.createRadioStream(radioStations[0]!.streams[0]!));
    for (let i = 0; i < rounds && !cancelled; i++) {
      const query = ['Linkin Park Numb', 'Rick Astley Never Gonna Give You Up', 'Queen Radio Ga Ga'][i % 3]!;
      ytdlp.clearCache();
      await measure(`search.cold.${state}`, async () => { assert.ok(await ytdlp.search(query)); });
      await measure(`search.warm.${state}`, async () => { assert.ok(await ytdlp.search(query)); });
      // Identical simultaneous requests expose subprocess duplication under contention.
      ytdlp.clearCache();
      await measure(`search.shared.${state}`, async () => { const results = await Promise.all([ytdlp.search(query), ytdlp.search(query), ytdlp.search(query)]); assert.ok(results.every(Boolean)); });
      if (i % 5 === 4) await checkpoint(`search.${state}.${i + 1}`);
    }
  }
  await measure('voice.cold-start', async () => { assert.equal(await voice.start(), true); });
  for (let i = 0; i < rounds && !cancelled; i++) {
    if (i) await delay(4000); // Reserve headroom within the free request quota.
    const fixture = pcm[i % pcm.length]!;
    await measure('voice.with-music', async () => voiceCommand(fixture));
  }
  closeMedia();
  // Real yt-dlp/FFmpeg/Opus resources, consumed without joining Discord.
  for (let i = 0; i < rounds && !cancelled; i++) {
    // Pin the test duration so metadata hydration cannot move the artificial
    // preparation deadline to the end of the full three-minute source track.
    const metadata = { video: async () => ({ id: 'dQw4w9WgXcQ', duration: 15 }) };
    const queue = new queueModule.GuildQueue(`benchmark-${i}`, { queues: new Map(), ytdlp: metadata } as unknown as MusicClient,
      (url, options) => streamModule.createYtdlpStream(ytdlp, url, options));
    Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
    const track = { url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', videoId: 'dQw4w9WgXcQ', title: 'Benchmark',
      duration: '0:15', thumbnail: null, requestedBy: 'Benchmark', isLive: false };
    try {
      await measure('audio.start', async () => { await queue.addTracks([track, { ...track }]); await entersState(queue.player, AudioPlayerStatus.Playing, 25_000); });
      const deadline = Date.now() + 20_000;
      while (!Reflect.get(queue, 'prepared') && Date.now() < deadline && !cancelled) await delay(50);
      await measure('audio.prepared-transition', async () => {
        assert.ok(Reflect.get(queue, 'prepared'));
        const oldResource = Reflect.get(queue, 'activeResource');
        queue.skip();
        const deadline = Date.now() + 25_000;
        while (Reflect.get(queue, 'activeResource') === oldResource && Date.now() < deadline) await delay(10);
        assert.notEqual(Reflect.get(queue, 'activeResource'), oldResource);
        await entersState(queue.player, AudioPlayerStatus.Playing, 25_000);
      });
    } finally { await queue.stop(); }
    if (i % 5 === 4) await checkpoint(`audio.${i + 1}`);
  }
  if (soakSeconds && !cancelled) {
    await install(streamModule.createRadioStream(radioStations[0]!.streams[0]!));
    const soakStarted = performance.now(), until = soakStarted + soakSeconds * 1000;
    let observed = mediaPlayer.state.status === AudioPlayerStatus.Idle ? null : mediaPlayer.state.resource;
    let consumed = observed?.playbackDuration ?? 0;
    let iteration = 0;
    while (performance.now() < until && !cancelled) {
      await delay(Math.min(10_000, Math.max(1, until - performance.now())));
      const resource = mediaPlayer.state.status === AudioPlayerStatus.Idle ? null : mediaPlayer.state.resource;
      if (resource !== observed) {
        if (observed) soakAudioMs += Math.max(0, observed.playbackDuration - consumed);
        observed = resource; consumed = 0;
      }
      if (observed) {
        soakAudioMs += Math.max(0, observed.playbackDuration - consumed);
        consumed = observed.playbackDuration;
      }
      soakElapsedMs = performance.now() - soakStarted;
      if (mediaPlayer.state.status === AudioPlayerStatus.Idle) {
        failures['soak.reconnect'] = (failures['soak.reconnect'] ?? 0) + 1;
        await measure('soak.recovery', async () => install(streamModule.createRadioStream(radioStations[0]!.streams[0]!)));
      }
      if (++iteration % 6 === 0) {
        const fixture = pcm[iteration % pcm.length]!;
        await measure('voice.soak', async () => {
          assert.equal(await voice.start(), true);
          await voiceCommand(fixture);
        });
        await checkpoint('soak');
      }
    }
    soakElapsedMs = performance.now() - soakStarted;
  }
  await checkpoint(cancelled ? 'cancelled' : 'complete');
  if (Object.values(failures).some(Boolean)) process.exitCode = 1;
} finally {
  clearInterval(sampler); loop.disable();
  stop(); await voice.close(); mediaPlayer.removeAllListeners();
  process.removeListener('SIGINT', stop); process.removeListener('SIGTERM', stop);
}
