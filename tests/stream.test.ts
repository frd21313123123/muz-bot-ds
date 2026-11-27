import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { PassThrough, Readable } from 'node:stream';
import { buffer } from 'node:stream/consumers';
import { setTimeout as wait } from 'node:timers/promises';
import test from 'node:test';
import { AudioPlayerStatus, StreamType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import type { MusicClient } from '../src/types.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import { bufferMusicPcm, createProcessStream, waitForAudio } from '../src/utils/stream.js';
import { terminateProcess } from '../src/utils/processes.js';

test('PCM startup releases immediately for a fast stream or short EOF, and bounds the wait after the first byte', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const input = new PassThrough();
  const output = bufferMusicPcm(input);
  const first = Buffer.alloc(3840, 11);
  context.mock.timers.tick(1000);
  input.write(first);
  await new Promise<void>(resolve => setImmediate(resolve));
  context.mock.timers.tick(399);
  assert.equal(output.read(), null, 'an empty input starts no deadline');
  context.mock.timers.tick(1);
  assert.deepEqual(output.read(), first);
  input.end(first);
  assert.deepEqual(await buffer(output), first);

  const fast = new PassThrough();
  const ready = bufferMusicPcm(fast);
  const rest = Buffer.alloc(3840 * 14, 22);
  fast.write(first); fast.write(rest);
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.deepEqual(ready.read(), Buffer.concat([first, rest]), 'a full buffer needs no wall-clock delay');
  fast.end();

  const short = new PassThrough();
  const tail = bufferMusicPcm(short);
  short.end(first);
  assert.deepEqual(await buffer(tail), first);
});

test('PCM startup propagates a failed source and discards partial audio on cancellation', async () => {
  const input = new PassThrough();
  const output = bufferMusicPcm(input);
  const failed = assert.rejects(buffer(output), /input failed/);
  input.write(Buffer.alloc(10));
  input.destroy(new Error('input failed'));
  await failed;

  const slowInput = new PassThrough();
  const slow = bufferMusicPcm(slowInput);
  const abort = new AbortController();
  const waiting = assert.rejects(waitForAudio({ type: StreamType.Raw, stream: slow, destroy: () => slow.destroy() },
    abort.signal, 1000), /отменён/);
  slowInput.write(Buffer.alloc(3840));
  abort.abort();
  const closed = new Promise<void>(resolve => slowInput.once('close', resolve));
  slow.destroy();
  await Promise.all([waiting, closed]);
  assert.equal(slow.read(), null);
});

test('a prefetched PCM stream stops reading the producer until a player consumes it', async () => {
  const block = Buffer.alloc(16384, 7);
  let produced = 0;
  function* blocks() {
    for (let i = 0; i < 100; i++) { produced++; yield block; }
  }
  const output = bufferMusicPcm(Readable.from(blocks(), { objectMode: false }));
  await once(output, 'readable');
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.ok(produced < 100, 'prefetch must not drain a whole song into memory');
  assert.equal((await buffer(output)).length, 100 * block.length);
});

test('a short startup gap in music is buffered before the real player emits Opus packets', async () => {
  const input = new PassThrough();
  const queue = new GuildQueue('startup-gap', { queues: new Map() } as unknown as MusicClient, () => {
    const stream = bufferMusicPcm(input);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  const packets: Buffer[] = [];
  let subscription: { unsubscribe(): void } | undefined;
  const connection = {
    state: { status: VoiceConnectionStatus.Ready },
    onSubscriptionRemoved: () => {}, setSpeaking: () => {},
    prepareAudioPacket: (packet: Buffer) => { packets.push(packet); }, dispatchAudio: () => true,
    destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  queue.connection = connection;
  subscription = Reflect.apply(Reflect.get(queue.player, 'subscribe'), queue.player, [connection]);
  try {
    const frame = Buffer.alloc(3840);
    for (let i = 0; i < 960; i++) {
      const sample = Math.round(6000 * Math.sin(2 * Math.PI * 400 * i / 48000));
      frame.writeInt16LE(sample, 4 * i); frame.writeInt16LE(sample, 4 * i + 2);
    }
    await queue.addTrack({ videoId: 'aaaaaaaaaaa', title: 'Fixture', url: 'fixture', duration: '2:00', thumbnail: null, requestedBy: 'Test' });
    input.write(frame);
    await wait(100);
    assert.equal(queue.player.state.status, AudioPlayerStatus.Buffering);
    assert.equal(packets.length, 0);
    input.write(Buffer.concat(Array<Buffer>(50).fill(frame)));
    const end = Date.now() + 2000;
    while (packets.length < 8 && Date.now() < end) await wait(10);
    assert.ok(packets.length >= 8);
    assert.ok(packets.every(packet => !packet.equals(Buffer.from([0xf8, 0xff, 0xfe]))), 'no inserted Discord silence');
  } finally { await queue.stop(); }
});

test('Windows teardown kills descendants that inherit stdout', { skip: process.platform !== 'win32' }, async () => {
  const source = spawn(process.execPath, ['-e', `
    const { spawn } = require('node:child_process');
    const descendant = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'inherit', windowsHide: true });
    process.stdout.write(String(descendant.pid) + '\\n');
    setInterval(() => {}, 1000);
  `], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let pid: number | undefined;
  try {
    const data = await new Promise<Buffer>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('descendant did not start')), 5000);
      source.stdout.once('data', value => { clearTimeout(timer); resolve(value); });
      source.once('error', error => { clearTimeout(timer); reject(error); });
    });
    pid = Number(data.toString().trim()); assert.ok(Number.isInteger(pid) && pid > 0);
    await terminateProcess(source);
    assert.equal(source.stdout.destroyed, true);
    assert.throws(() => process.kill(pid!, 0), { code: 'ESRCH' });
  } finally {
    await terminateProcess(source);
    if (pid) try { process.kill(pid, 'SIGKILL'); } catch { /* Already stopped. */ }
  }
});

test('destroy ends both child processes', async () => {
  const source = spawn(process.execPath, ['-e', 'setInterval(() => process.stdout.write("x"), 10)'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  const sink = spawn(process.execPath, ['-e', 'process.stdin.pipe(process.stdout); setInterval(() => {}, 1000)'],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  const media = createProcessStream(source, sink);
  const sourceClosed = once(source, 'close');
  const sinkClosed = once(sink, 'close');
  media.destroy();
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([
    Promise.all([sourceClosed, sinkClosed]),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('child processes survived destroy')), 5_000); }),
  ]).finally(() => { if (timer) clearTimeout(timer); });
  assert.equal(media.stream.destroyed, true);
});

test('nonzero source exit reports an error and closes the sink', async () => {
  const source = spawn(process.execPath, ['-e', 'process.stderr.write("test failure");process.exit(7)'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  const sink = spawn(process.execPath, ['-e', 'process.stdin.pipe(process.stdout);setInterval(() => {}, 1000)'],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  const media = createProcessStream(source, sink);
  const sinkClosed = once(sink, 'close');
  const error = await new Promise<Error>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('no stream error')), 5_000);
    media.stream.once('error', (cause: Error) => { clearTimeout(timeout); resolve(cause); });
  });
  assert.match(error.message, /yt-dlp.*7.*test failure/);
  await sinkClosed;
});
