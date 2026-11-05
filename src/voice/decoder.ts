import { createRequire } from 'node:module';
import OpusScript from 'opusscript';

interface OpusHandler { _decode(input: number, length: number, output: number): number }
interface OpusNative {
  HEAPU8: Uint8Array;
  OpusScriptHandler: (new (rate: number, channels: number, application: number) => OpusHandler)
    & { destroy_handler(handler: OpusHandler): void };
  _malloc(bytes: number): number;
  _free(pointer: number): void;
}

const SAMPLE_RATE = 16_000;
const MAX_SAMPLES = 960;
// opusscript 0.1.1's native helper stores each decoded PCM byte in a 16-bit
// slot at the byte address it receives: 60 ms of mono PCM needs 3840 bytes.
const OUTPUT_BYTES = SAMPLE_RATE * 60 / 1000 * 2 * 2;

// Incoming speech uses a private WebAssembly instance (about 3x faster than the
// asm.js build). Its heap is separate from the shared opusscript heap used by
// outgoing music encoders, so decoder allocations can grow memory without
// detaching the encoders' fixed PCM views. Views here are taken per call.
let native: OpusNative | null = null;
function decoderModule(): OpusNative {
  if (!native) {
    const require = createRequire(import.meta.url);
    const factory = require('opusscript/build/opusscript_native_wasm.js') as () => OpusNative;
    native = factory();
  }
  return native;
}

// Opus TOC frame sizes at 16 kHz. Reject packets longer than 60 ms before
// decoding: the PCM output allocation cannot hold a 120 ms output.
export function opusSamples(packet: Buffer): number | null {
  if (!packet.length || packet.length > OpusScript.MAX_PACKET_SIZE) return null;
  const toc = packet[0]!;
  const config = toc >> 3;
  const frame = config >= 16 ? 40 << (config & 3)
    : config >= 12 ? 160 << (config & 1) : [160, 320, 640, 960][config & 3]!;
  const code = toc & 3;
  const count = code === 0 ? 1 : code < 3 ? 2 : packet.length > 1 ? packet[1]! & 63 : 0;
  const samples = frame * count;
  return count > 0 && samples <= MAX_SAMPLES ? samples : null;
}

export class VoiceDecoder {
  private readonly native = decoderModule();
  private readonly handler: OpusHandler;
  private readonly input: number;
  private readonly output: number;
  private deleted = false;

  constructor() {
    this.handler = new this.native.OpusScriptHandler(SAMPLE_RATE, 1, OpusScript.Application.VOIP);
    this.input = this.native._malloc(OpusScript.MAX_PACKET_SIZE);
    this.output = this.native._malloc(OUTPUT_BYTES);
  }

  decode(packet: Buffer): Buffer | null {
    if (this.deleted) return null;
    const samples = opusSamples(packet);
    if (samples === null) return null;
    this.native.HEAPU8.set(packet, this.input);
    const length = this.handler._decode(this.input, packet.length, this.output);
    // Negative lengths are libopus errors; the caller releases the capture.
    if (length !== samples) throw new Error('Invalid decoded audio size');
    // Narrow each 16-bit slot back to its PCM byte.
    return Buffer.from(new Uint16Array(this.native.HEAPU8.buffer, this.output, samples * 2));
  }

  delete(): void {
    if (this.deleted) return;
    this.deleted = true;
    this.native.OpusScriptHandler.destroy_handler(this.handler);
    this.native._free(this.input);
    this.native._free(this.output);
  }
}
