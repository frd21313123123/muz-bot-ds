import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { VoiceDecoder } from '../src/voice/decoder.js';

// Generated with FFmpeg/libopus: 48 kHz stereo, 20 ms packets, 440 Hz sine.
const packets = (JSON.parse(readFileSync('tests/fixtures/voice-tone-opus.json', 'utf8')) as string[])
  .map((hex) => Buffer.from(hex, 'hex'));

test('voice decoder preserves a real stereo Opus waveform across repeated lifecycles', () => {
  for (let cycle = 0; cycle < 40; cycle++) {
    const decoders = [new VoiceDecoder(), new VoiceDecoder()];
    try {
      const outputs = decoders.map((decoder) => Buffer.concat(packets.map((packet) => {
        const pcm = decoder.decode(packet);
        assert.equal(pcm?.length, 640);
        return pcm!;
      })));
      assert.deepEqual(outputs[0], outputs[1]);
      const pcm = outputs[0]!;
      let energy = 0, real = 0, imaginary = 0;
      // Exclude encoder delay and the end padding.
      const start = 640, end = 2240;
      for (let i = start; i < end; i++) {
        const sample = pcm.readInt16LE(i * 2);
        const phase = 2 * Math.PI * 440 * i / 16_000;
        energy += sample * sample;
        real += sample * Math.cos(phase);
        imaginary += sample * Math.sin(phase);
      }
      assert.ok(Math.sqrt(energy / (end - start)) > 1_000, 'decoded tone must be audible');
      assert.ok(2 * (real * real + imaginary * imaginary) / ((end - start) * energy) > 0.98,
        'decoded signal must retain its 440 Hz frequency');
    } finally {
      for (const decoder of decoders) { decoder.delete(); decoder.delete(); }
    }
    assert.equal(decoders[0]!.decode(packets[0]!), null);
  }
});

test('voice decoder rejects oversized duration and malformed packets before decoding', () => {
  const decoder = new VoiceDecoder();
  try {
    // TOC 0x1b: SILK 60 ms, code 3, two frames = 120 ms.
    for (const packet of [Buffer.alloc(0), Buffer.from([0x1b, 2]), Buffer.from([0x03]),
      Buffer.alloc(4_000)]) assert.equal(decoder.decode(packet), null);
    assert.equal(decoder.decode(packets[0]!)?.length, 640);
  } finally { decoder.delete(); }
});
