import assert from 'node:assert/strict';
import test from 'node:test';
import { PlaybackEchoFilter, PlaybackEchoReference, type EchoResult } from '../src/voice/echo.js';

function signal(samples: number, seed = 1): Buffer {
  const pcm = Buffer.alloc(samples * 2);
  let smooth = 0;
  for (let i = 0; i < samples; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    smooth = 0.72 * smooth + 0.28 * ((seed >>> 16) - 32768);
    pcm.writeInt16LE(Math.round(smooth * 0.6 + 1100 * Math.sin(i * 0.071)), i * 2);
  }
  return pcm;
}

function reference(pcm: Buffer): PlaybackEchoReference {
  const result = new PlaybackEchoReference();
  for (let i = 0; i < pcm.length; i += 640) result.append(pcm.subarray(i, i + 640), i / 32 + 20);
  return result;
}

function scaled(pcm: Buffer, gain: number, extra?: Buffer): Buffer {
  const output = Buffer.alloc(pcm.length);
  for (let i = 0; i < pcm.length; i += 2) output.writeInt16LE(Math.round(pcm.readInt16LE(i) * gain + (extra?.readInt16LE(i) ?? 0)), i);
  return output;
}

function rms(pcm: Buffer): number {
  let energy = 0;
  for (let i = 0; i < pcm.length; i += 2) energy += pcm.readInt16LE(i) ** 2;
  return Math.sqrt(energy / (pcm.length / 2));
}

function receive(filter: PlaybackEchoFilter, pcm: Buffer, start: number, sizes = [320]): EchoResult[] {
  const output: EchoResult[] = [];
  let index = 0, offset = 0;
  while (offset < pcm.length / 2) {
    const count = Math.min(sizes[index++ % sizes.length]!, pcm.length / 2 - offset);
    offset += count;
    const result = filter.process(pcm.subarray((offset - count) * 2, offset * 2), start + offset / 16);
    if (result.pcm.length) output.push(result);
  }
  const tail = filter.flush(start + offset / 16);
  if (tail.pcm.length) output.push(tail);
  return output;
}

test('playback echo cancels sample-offset delays, attenuation and phase inversion including its buffered prefix', () => {
  const source = signal(16_000);
  for (const gain of [0.35, -0.65]) {
    const input = scaled(source.subarray(133 * 2, (133 + 6400) * 2), gain);
    const output = receive(new PlaybackEchoFilter(reference(source)), input, 1200);
    assert.equal(Buffer.concat(output.map(value => value.pcm)).length, input.length);
    assert.ok(output.every(value => value.echoOnly), 'all emitted frames, including acquisition, are classified as playback');
    assert.ok(output.every(value => rms(value.pcm) === 0));
  }
});

test('simultaneous independent human speech survives reference subtraction', () => {
  const source = signal(16_000);
  const playback = source.subarray(221 * 2, (221 + 6400) * 2);
  // Different independent waveforms exercise the statistical double-talk gate;
  // one fortunate seed must not hide a false-positive first-word suppression.
  for (const seed of [13, 42, 77, 99, 123, 257, 793, 1024]) {
    const human = scaled(signal(6400, seed), 0.4);
    const input = scaled(playback, 0.65, human);
    const output = receive(new PlaybackEchoFilter(reference(source)), input, 1250);
    const cleaned = Buffer.concat(output.map(value => value.pcm));
    assert.equal(cleaned.length, human.length);
    assert.ok(output.every(value => !value.echoOnly), `mixed voice ${seed} is never muted as echo-only`);
    const error = scaled(cleaned, 1, scaled(human, -1));
    assert.ok(rms(error) < rms(human) * 0.25, `echo residue ${rms(error)} exceeds preserved voice ${rms(human)}`);
    assert.ok(rms(cleaned) > rms(human) * 0.8);
  }
});

test('unrelated speech, silence, missing and stale references remain byte-for-byte unchanged', () => {
  const source = signal(16_000);
  const unrelated = signal(3200, 83);
  for (const [ref, at] of [[reference(source), 1200], [reference(source), 3300],
    [new PlaybackEchoReference(), 1200], [reference(Buffer.alloc(32_000)), 1200]] as const) {
    for (const input of [unrelated, Buffer.alloc(640)]) {
      const output = receive(new PlaybackEchoFilter(ref), input, at);
      assert.deepEqual(Buffer.concat(output.map(value => value.pcm)), input);
      assert.ok(output.every(value => !value.echoOnly));
    }
  }
});

test('variable packet sizes and a short normal end drain preserve sample count', () => {
  const source = signal(16_000);
  for (const length of [320, 640, 960, 3200]) {
    const input = scaled(source.subarray(317 * 2, (317 + length) * 2), 0.45);
    const output = receive(new PlaybackEchoFilter(reference(source)), input, 1200, [160, 640, 960, 320]);
    assert.equal(Buffer.concat(output.map(value => value.pcm)).length, input.length);
    assert.ok(output.every(value => value.echoOnly), `echo ending after ${length / 16} ms should be filtered on flush`);
  }
});

test('reference changes and cancelled acquisition cannot leak audio into the next capture', () => {
  const source = signal(16_000);
  const ref = reference(source);
  const filter = new PlaybackEchoFilter(ref);
  assert.equal(filter.process(source.subarray(0, 640), 1200).pcm.length, 0);
  filter.reset();
  assert.equal(filter.flush(1200).pcm.length, 0);
  ref.clear();
  const human = signal(640, 87);
  assert.deepEqual(filter.process(human, 1220), { pcm: human, echoOnly: false });
  assert.equal(filter.flush(1220).pcm.length, 0);
});

test('a nearby tone frequency is not mistaken for a cue', () => {
  const tone = (frequency: number, samples: number): Buffer => {
    const pcm = Buffer.alloc(samples * 2);
    for (let i = 0; i < samples; i++) pcm.writeInt16LE(Math.round(3000 * Math.sin(i * Math.PI * 2 * frequency / 16000)), i * 2);
    return pcm;
  };
  const input = tone(662, 3200);
  const output = receive(new PlaybackEchoFilter(reference(tone(660, 16_000))), input, 1200);
  assert.deepEqual(Buffer.concat(output.map(value => value.pcm)), input);
  assert.ok(output.every(value => !value.echoOnly));
});

test('the reference expires after its two-second echo tail and remains bounded during long playback', () => {
  const ref = new PlaybackEchoReference();
  const pcm = signal(320);
  for (let i = 0; i < 10_000; i++) ref.append(pcm, i * 20);
  const snapshot = ref.snapshot(200_000)!;
  assert.ok(snapshot.raw.length <= 2440 * 16);
  assert.ok(snapshot.spans.length <= 123);
  assert.ok(ref.snapshot(201_980));
  assert.equal(ref.snapshot(202_181), null);
});
