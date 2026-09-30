import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import { mkdtemp, readFile, rm, rmdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import OpusScript from 'opusscript';
import { createAudioResource, StreamType, TransformerType } from '@discordjs/voice';
import { configureMusicEncoder, loudnessFilter, musicPcmArguments, parseLoudnessMeasurement } from '../src/utils/audio.js';
import { normalizeAudioFile } from '../src/utils/normalizeAudio.js';
import { requireFfmpeg } from '../src/utils/stream.js';

function tone(amplitude: number, seconds = 12): Buffer {
  const pcm = Buffer.alloc(48_000 * seconds * 4);
  for (let i = 0; i < pcm.length / 4; i++) {
    pcm.writeInt16LE(Math.round(32767 * amplitude * Math.sin(2 * Math.PI * 440 * i / 48000)), i * 4);
    pcm.writeInt16LE(Math.round(32767 * amplitude * Math.sin(2 * Math.PI * 12000 * i / 48000)), i * 4 + 2);
  }
  return pcm;
}

const rawInput = ['-f', 's16le', '-ar', '48000', '-ac', '2', '-i', 'pipe:0'];
async function ffmpeg(input: Buffer | null, args: string[]): Promise<{ pcm: Buffer; stderr: string }> {
  const child = spawn(requireFfmpeg(), ['-nostdin', '-hide_banner', '-nostats', ...args], { windowsHide: true });
  const chunks: Buffer[] = [];
  let stderr = '';
  child.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
  child.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString(); });
  child.stdin.on('error', () => {});
  const done = new Promise<void>((resolve, reject) => {
    child.once('error', reject);
    child.once('close', code => code === 0 ? resolve() : reject(new Error(stderr)));
  });
  child.stdin.end(input ?? undefined);
  await done;
  return { pcm: Buffer.concat(chunks), stderr };
}

async function measure(pcm: Buffer) {
  const { stderr } = await ffmpeg(pcm, [...rawInput, '-af', loudnessFilter(null, true), '-f', 'null', '-']);
  return parseLoudnessMeasurement(stderr);
}

test('real FFmpeg brings tracks 20 dB apart to the same loudness and keeps volume headroom', async () => {
  const results = [];
  for (const amplitude of [0.02, 0.2]) {
    const { pcm } = await ffmpeg(tone(amplitude), [...rawInput, ...musicPcmArguments()]);
    assert.equal(pcm.length, 48000 * 12 * 4, 'duration and stereo sample rate must be preserved');
    const stats = await measure(pcm);
    assert.ok(stats);
    assert.ok(Math.abs(stats.input_i + 16) < 0.5, `unexpected loudness ${stats.input_i}`);
    assert.ok(stats.input_tp <= -3.8, `unexpected true peak ${stats.input_tp}`);
    for (let offset = 0; offset < pcm.length; offset += 2) {
      assert.ok(Math.abs(pcm.readInt16LE(offset)) * 1.5 < 32767, '150% volume must not clip');
    }
    results.push(stats.input_i);
  }
  assert.ok(Math.abs(results[0]! - results[1]!) < 0.2, 'normalized tracks must agree');
});

test('normalization preserves silence and short tracks instead of inventing sound or truncating', async () => {
  for (const input of [Buffer.alloc(48000 * 4), tone(0.1, 0.1)]) {
    const { pcm } = await ffmpeg(input, [...rawInput, ...musicPcmArguments()]);
    assert.equal(pcm.length, input.length);
    if (input.every(value => value === 0)) {
      for (let i = 0; i < pcm.length; i += 2) assert.ok(Math.abs(pcm.readInt16LE(i)) <= 1);
    }
  }
});

test('normalization handles silent introductions and limits transient peaks', async () => {
  const input = Buffer.concat([Buffer.alloc(48000 * 4 * 3), tone(0.02, 6), Buffer.alloc(48000 * 4)]);
  for (const position of [4, 6, 8]) {
    for (let i = 0; i < 48; i++) {
      const value = Math.round(32700 * Math.sin(2 * Math.PI * i / 48));
      input.writeInt16LE(value, (position * 48000 + i) * 4);
      input.writeInt16LE(value, (position * 48000 + i) * 4 + 2);
    }
  }
  const { pcm } = await ffmpeg(input, [...rawInput, ...musicPcmArguments()]);
  assert.equal(pcm.length, input.length);
  const stats = await measure(pcm);
  assert.ok(stats && stats.input_tp <= -3.8, `transient peak ${stats?.input_tp}`);
  for (let offset = 0; offset < 48000 * 4 * 2; offset += 2) {
    assert.ok(Math.abs(pcm.readInt16LE(offset)) <= 1, 'silent introduction must stay silent');
  }
});

test('music is encoded once; real Opus keeps stereo separation and a 12 kHz tone across concurrent resources', async () => {
  const resources = [0, 1, 2].map(() => {
    const resource = createAudioResource(Readable.from([tone(0.1, 1)]), { inputType: StreamType.Raw, inlineVolume: true });
    configureMusicEncoder(resource);
    resource.volume!.setVolume(1.5);
    assert.deepEqual(resource.edges.map(edge => edge.type), [TransformerType.InlineVolume, TransformerType.OpusEncoder]);
    const memory = resource.encoder!.encoder as { inPCM: Uint16Array; inPCMPointer: number; inPCMLength: number };
    assert.equal(memory.inPCM.byteOffset, memory.inPCMPointer);
    assert.equal(memory.inPCM.byteLength, memory.inPCMLength);
    return resource;
  });
  for (const resource of resources) {
    const decoder = new OpusScript(48000, 2, OpusScript.Application.AUDIO, { wasm: false });
    const memory = decoder as unknown as { outPCM: Uint16Array; outPCMPointer: number; outPCMLength: number };
    memory.outPCM = new Uint16Array(memory.outPCM.buffer, memory.outPCMPointer, memory.outPCMLength / 2);
    try {
      const chunks: Buffer[] = [];
      for await (const packet of resource.playStream) chunks.push(decoder.decode(packet as Buffer));
      const pcm = Buffer.concat(chunks);
      assert.equal(pcm.length, 48000 * 4);
      for (const [channel, frequency] of [[0, 440], [1, 12000]] as const) {
        let energy = 0, real = 0, imaginary = 0;
        const start = 4800, end = 43200;
        for (let i = start; i < end; i++) {
          const sample = pcm.readInt16LE(i * 4 + channel * 2);
          energy += sample * sample;
          real += sample * Math.cos(2 * Math.PI * frequency * i / 48000);
          imaginary += sample * Math.sin(2 * Math.PI * frequency * i / 48000);
        }
        assert.ok(Math.sqrt(energy / (end - start)) > 3000, 'volume must remain audible');
        assert.ok(2 * (real * real + imaginary * imaginary) / ((end - start) * energy) > 0.95,
          `channel ${channel} lost its ${frequency} Hz tone`);
      }
    } finally { decoder.delete(); resource.playStream.destroy(); }
  }
});

test('two-pass file normalization writes a lossless copy and refuses to overwrite source or output', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'muz-audio-'));
  const input = path.join(directory, 'quiet source.wav');
  const output = path.join(directory, 'normalized.flac');
  try {
    await ffmpeg(tone(0.02), [...rawInput, '-c:a', 'pcm_s16le', input]);
    const original = await readFile(input);
    await normalizeAudioFile(input, output);
    assert.deepEqual(await readFile(input), original);
    const normalized = await ffmpeg(null, ['-i', output, '-f', 's16le', '-c:a', 'pcm_s16le', '-']);
    const stats = await measure(normalized.pcm);
    assert.ok(stats && Math.abs(stats.input_i + 16) < 0.5);
    await assert.rejects(normalizeAudioFile(input, output), /уже существует/);
    await assert.rejects(normalizeAudioFile(input, input), /уже существует/);
    await assert.rejects(normalizeAudioFile(input, path.join(directory, 'lossy.mp3')), /flac/);
  } finally {
    await rm(output, { force: true }); await rm(input, { force: true }); await rmdir(directory);
  }
});
