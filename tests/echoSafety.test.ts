import assert from 'node:assert/strict';
import test from 'node:test';
import { PlaybackEchoFilter, PlaybackEchoReference, type EchoResult } from '../src/voice/echo.js';

// Different seeds make the speech prefix independent of the playback reference.
function signal(samples: number, seed: number): Buffer {
  const pcm = Buffer.alloc(samples * 2);
  let smooth = 0;
  for (let i = 0; i < samples; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    smooth = 0.72 * smooth + 0.28 * ((seed >>> 16) - 32768);
    pcm.writeInt16LE(Math.round(smooth * 0.6), i * 2);
  }
  return pcm;
}

function append(reference: PlaybackEchoReference, pcm: Buffer, startMs = 0): void {
  for (let i = 0; i < pcm.length; i += 640) {
    const frame = pcm.subarray(i, i + 640);
    reference.append(frame, startMs + (i + frame.length) / 32);
  }
}

function attenuate(pcm: Buffer): Buffer {
  const result = Buffer.alloc(pcm.length);
  for (let i = 0; i < pcm.length; i += 2) result.writeInt16LE(Math.round(pcm.readInt16LE(i) * 0.6), i);
  return result;
}

function output(results: EchoResult[]): Buffer { return Buffer.concat(results.map(result => result.pcm)); }

test('two 60 ms packets preserve a human prefix before a longer matching playback tail', () => {
  const source = signal(16_000, 11), human = signal(640, 83);
  const reference = new PlaybackEchoReference(); append(reference, source);
  const filter = new PlaybackEchoFilter(reference);
  const input = Buffer.concat([human, attenuate(source.subarray(640 * 2, 1920 * 2))]);
  const cleaned = output([
    filter.process(input.subarray(0, 960 * 2), 1260),
    filter.process(input.subarray(960 * 2), 1320),
    filter.flush(1320),
  ]);
  assert.equal(cleaned.length, input.length, 'acquisition and normal end preserve every sample');
  assert.ok(cleaned.subarray(0, human.length).equals(human),
    'a match in the last 80 ms cannot justify subtracting playback from an earlier independent voice');
});

test('a receive gap does not align pending human speech with playback from the next packet', () => {
  const source = signal(16_000, 11), human = signal(640, 83);
  const reference = new PlaybackEchoReference(); append(reference, source);
  const filter = new PlaybackEchoFilter(reference);
  const beforeGap = filter.process(human, 1240);
  const echo = attenuate(source.subarray(640 * 2, 1920 * 2));
  const cleaned = output([beforeGap, filter.process(echo, 1700), filter.flush(1700)]);
  assert.equal(cleaned.length, human.length + echo.length);
  assert.ok(cleaned.subarray(0, human.length).equals(human),
    'buffered speech before a 460 ms gap keeps its own position and waveform');
});

test('clearing the playback reference cannot subtract new audio from pending human speech', () => {
  const reference = new PlaybackEchoReference(); append(reference, signal(16_000, 11));
  const filter = new PlaybackEchoFilter(reference), human = signal(640, 83);
  const beforeClear = filter.process(human, 1240);
  reference.clear();
  const nextSource = signal(8000, 19); append(reference, nextSource, 1250);
  const echo = attenuate(nextSource.subarray(640 * 2, 1920 * 2));
  const cleaned = output([beforeClear, filter.process(echo, 1800), filter.flush(1800)]);
  assert.equal(cleaned.length, human.length + echo.length);
  assert.ok(cleaned.subarray(0, human.length).equals(human),
    'a new reference generation must not reinterpret an old pending microphone prefix');
});

test('echo beginning after silence does not leak an unfiltered acquisition prefix', () => {
  const source = signal(16_000, 37);
  const reference = new PlaybackEchoReference(); append(reference, source);
  const filter = new PlaybackEchoFilter(reference);
  const silence = Buffer.alloc(80 * 32);
  const echo = attenuate(source.subarray(133 * 2, (133 + 2560) * 2));
  const input = Buffer.concat([silence, echo]);
  const results: EchoResult[] = [];
  for (let i = 0; i < input.length; i += 640) {
    results.push(filter.process(input.subarray(i, i + 640), 1200 + (i + 640) / 32));
  }
  results.push(filter.flush(1440));
  const cleaned = output(results);
  assert.equal(cleaned.length, input.length);
  assert.ok(cleaned.every(byte => byte === 0),
    'silent frames must not disable acquisition buffering for the first real playback echo');
});
