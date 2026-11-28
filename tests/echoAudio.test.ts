import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { gunzipSync } from 'node:zlib';
import OpusScript from 'opusscript';
import { VoiceDecoder } from '../src/voice/decoder.js';
import { PlaybackEchoFilter, PlaybackEchoReference } from '../src/voice/echo.js';
import { speechWav } from '../src/voice/groq.js';

const rate = 16_000;
const frame = 320;
const samples = 4 * rate;
const speechStart = 27_840; // 1,740 ms, intentionally between filter block boundaries.
const wavDirectory = process.env.ECHO_AUDIO_WAV_DIR;

function fixture(name: string): Buffer {
  return gunzipSync(readFileSync(join('tests', 'fixtures', 'echo-audio', name + '.pcm.gz')));
}

function transcode(pcm: Buffer, sampleRate: 16_000 | 48_000, channels: number, bitrate: number): Buffer {
  const application = channels === 2 ? OpusScript.Application.AUDIO : OpusScript.Application.VOIP;
  const encoder = new OpusScript(sampleRate, channels, application);
  const decoder = new VoiceDecoder();
  const packetSamples = sampleRate / 50;
  const packetBytes = packetSamples * channels * 2;
  const chunks: Buffer[] = [];
  try {
    encoder.setBitrate(bitrate);
    assert.equal(pcm.length % packetBytes, 0);
    for (let first = 0; first < pcm.length; first += packetBytes) {
      const packet = Buffer.from(encoder.encode(pcm.subarray(first, first + packetBytes), packetSamples));
      const decoded = decoder.decode(packet);
      assert.equal(decoded?.length, frame * 2);
      chunks.push(decoded);
    }
    return Buffer.concat(chunks);
  } finally {
    decoder.delete();
    encoder.delete();
  }
}

function pcm(signal: Float64Array): Buffer {
  const bytes = Buffer.alloc(signal.length * 2);
  for (let i = 0; i < signal.length; i++) {
    assert.ok(Math.abs(signal[i]!) < 32767, 'test mixture must not clip');
    bytes.writeInt16LE(Math.round(signal[i]!), i * 2);
  }
  return bytes;
}

function rms(signal: Float64Array): number {
  return Math.sqrt(signal.reduce((energy, value) => energy + value * value, 0) / signal.length);
}

function energy(bytes: Buffer): number {
  let result = 0;
  for (let i = 0; i < bytes.length; i += 2) result += bytes.readInt16LE(i) ** 2;
  return result;
}

function powerDb(numerator: number, denominator: number): number {
  return 10 * Math.log10(Math.max(numerator, 1) / Math.max(denominator, 1));
}

// Project onto the clean, level-matched microphone decode to estimate speech
// retention. RMS alone could report leaked music as surviving speech.
function speechGainDb(output: Buffer, clean: Buffer, first: number, last: number): number {
  let dot = 0;
  for (let i = first * 2; i < last * 2; i += 2) dot += output.readInt16LE(i) * clean.readInt16LE(i);
  return 20 * Math.log10(Math.max(1e-6, Math.abs(dot) / energy(clean.subarray(first * 2, last * 2))));
}

function filterAudio(playback: Buffer, microphone: Buffer, first: number, last: number) {
  const reference = new PlaybackEchoReference();
  const filter = new PlaybackEchoFilter(reference);
  const chunks: Buffer[] = [];
  let written = 0;
  let suppressed = 0;
  const accept = ({ pcm: output, echoOnly }: ReturnType<PlaybackEchoFilter['process']>) => {
    const end = written + output.length / 2;
    if (echoOnly) suppressed += Math.max(0, Math.min(last, end) - Math.max(first, written));
    chunks.push(output);
    written = end;
  };
  for (let offset = 0; offset < samples; offset += frame) {
    const now = (offset + frame) / (rate / 1000);
    reference.append(playback.subarray(offset * 2, (offset + frame) * 2), now);
    accept(filter.process(microphone.subarray(offset * 2, (offset + frame) * 2), now));
  }
  accept(filter.flush(samples / (rate / 1000)));
  assert.equal(written, samples);
  return { output: Buffer.concat(chunks), suppressedMs: suppressed / (rate / 1000) };
}

function saveWav(name: string, audio: Buffer): void {
  if (!wavDirectory) return;
  mkdirSync(wavDirectory, { recursive: true });
  writeFileSync(join(wavDirectory, name + '.wav'), speechWav(audio));
}

const rooms = [
  { name: 'direct', delay: 71, reflections: [[0, 1]], bitrate: 0 },
  { name: 'opus', delay: 137, reflections: [[0, 1]], bitrate: 24_000 },
  { name: 'room', delay: 480, reflections: [[0, 1], [19, 0.38], [43, -0.26], [87, 0.19], [129, -0.12]], bitrate: 24_000 },
] as const;

const commands = [
  { name: 'stop', fromMs: 0, toMs: 530 },
  { name: 'pause', fromMs: 320, toMs: 950 },
] as const;

test('real Russian speech and recorded music through the playback echo filter', context => {
  for (const track of ['piano', 'ensemble']) {
    const playback = transcode(fixture(track), 48_000, 2, 96_000);
    for (const room of rooms) {
      const reflected = Float64Array.from({ length: samples }, (_, i) =>
        room.reflections.reduce((value, [offset, gain]) => {
          const source = i - (room.delay + offset) * (rate / 1000);
          return value + (source >= 0 ? playback.readInt16LE(source * 2) * gain : 0);
        }, 0));
      const receive = (signal: Float64Array) => {
        const bytes = pcm(signal);
        return room.bitrate ? transcode(bytes, rate, 1, room.bitrate) : bytes;
      };
      for (const command of commands) {
        const recording = fixture(command.name);
        const activeStart = speechStart + command.fromMs * (rate / 1000);
        const activeEnd = speechStart + command.toMs * (rate / 1000);
        const speech = Float64Array.from({ length: samples }, (_, i) => {
          const byte = (i - speechStart) * 2;
          return byte >= 0 && byte + 1 < recording.length ? recording.readInt16LE(byte) : 0;
        });
        const speechRms = rms(speech.subarray(activeStart, activeEnd));
        const echoRms = rms(reflected.subarray(activeStart, activeEnd));
        assert.ok(speechRms > 100 && echoRms > 100, 'reference intervals must contain audio');
        const echoLevel = 32768 * 10 ** (-24 / 20);
        const echo = reflected.map(value => value * echoLevel / echoRms);
        const measureEnd = activeEnd + (room.bitrate ? frame : 0);
        const prefix = [track, room.name, command.name].join('-');
        const echoIn = receive(echo);
        const echoOut = filterAudio(playback, echoIn, activeStart, measureEnd).output;
        const activeEnergy = (audio: Buffer) => energy(audio.subarray(activeStart * 2, measureEnd * 2));
        const reduction = powerDb(activeEnergy(echoIn), activeEnergy(echoOut));
        context.diagnostic(`${prefix} echo only: ${reduction.toFixed(1)} dB reduction`);
        saveWav(prefix + '-echo', echoOut);
        if (!room.bitrate) assert.ok(reduction > 25, prefix);

        for (const relativeDb of [0, -10, -20, -30]) {
          const voice = speech.map(value => value * echoLevel / speechRms * 10 ** (relativeDb / 20));
          const clean = receive(voice);
          const input = receive(echo.map((value, i) => value + voice[i]!));
          const { output, suppressedMs } = filterAudio(playback, input, activeStart, measureEnd);
          const gain = speechGainDb(output, clean, activeStart, measureEnd);
          const ratio = 100 * suppressedMs / ((measureEnd - activeStart) / (rate / 1000));
          context.diagnostic(`${prefix} ${relativeDb} dB: speech gain ${gain.toFixed(1)} dB, echo-only ${ratio.toFixed(0)}%`);
          if (room.name === 'direct' && relativeDb === 0) assert.ok(gain > -6, prefix);
          saveWav(`${prefix}-${relativeDb}-clean`, clean);
          saveWav(`${prefix}-${relativeDb}-mixed`, input);
          saveWav(`${prefix}-${relativeDb}-filtered`, output);
        }
      }
    }
  }
});
