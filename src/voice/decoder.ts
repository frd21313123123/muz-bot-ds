import OpusScript from 'opusscript';

interface DecoderMemory { outPCM: Uint16Array; outPCMPointer: number; outPCMLength: number }

// Opus TOC frame sizes at 16 kHz. Reject packets longer than 60 ms before
// decoding: opusscript's PCM output allocation cannot hold its 120 ms output.
export function opusSamples(packet: Buffer): number | null {
  if (!packet.length || packet.length > OpusScript.MAX_PACKET_SIZE) return null;
  const toc = packet[0]!;
  const config = toc >> 3;
  const frame = config >= 16 ? 40 << (config & 3)
    : config >= 12 ? 160 << (config & 1) : [160, 320, 640, 960][config & 3]!;
  const code = toc & 3;
  const count = code === 0 ? 1 : code < 3 ? 2 : packet.length > 1 ? packet[1]! & 63 : 0;
  const samples = frame * count;
  return count > 0 && samples <= 960 ? samples : null;
}

export class VoiceDecoder {
  private readonly codec: OpusScript;
  private deleted = false;

  constructor() {
    // Use a separate heap from the WASM encoders used for outgoing playback.
    this.codec = new OpusScript(16_000, 1, OpusScript.Application.VOIP, { wasm: false });
    const memory = this.codec as unknown as DecoderMemory;
    // opusscript 0.1.1 treats a byte pointer as a Uint16Array element offset,
    // doubling the address. Its native helper outputs each PCM byte as a short;
    // the view must start at the allocated byte address and stay inside it.
    memory.outPCM = new Uint16Array(memory.outPCM.buffer, memory.outPCMPointer, memory.outPCMLength / 2);
  }

  decode(packet: Buffer): Buffer | null {
    if (this.deleted) return null;
    const samples = opusSamples(packet);
    if (samples === null) return null;
    const pcm = this.codec.decode(packet);
    if (pcm.length !== samples * 2) throw new Error('Invalid decoded audio size');
    return pcm;
  }

  delete(): void {
    if (this.deleted) return;
    this.deleted = true;
    this.codec.delete();
  }
}
