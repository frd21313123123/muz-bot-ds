import type { AudioResource } from '@discordjs/voice';
import OpusScript from 'opusscript';

export const MUSIC_BITRATE = 128_000;
export const LOUDNESS_TARGET = -16;
export const TRUE_PEAK_TARGET = -4; // Headroom for /volume 150 (+3.52 dB).
export const LOUDNESS_RANGE = 11;
const target = `I=${LOUDNESS_TARGET}:TP=${TRUE_PEAK_TARGET}:LRA=${LOUDNESS_RANGE}`;
const stereo = 'aformat=channel_layouts=stereo';
// Some FFmpeg builds emit NaN for all-silent loudnorm input. Converting NaN
// straight to int16 produces full-scale DC; restore silence before conversion.
const finite = "aeval=exprs='if(isnan(val(0)),0,val(0))|if(isnan(val(1)),0,val(1))'";
const resample = 'aresample=48000:osf=s16:dither_method=triangular';

export interface LoudnessMeasurement {
  input_i: number; input_tp: number; input_lra: number; input_thresh: number; target_offset: number;
}

export function parseLoudnessMeasurement(stderr: string): LoudnessMeasurement | null {
  const block = stderr.match(/\{\s*"input_i"[\s\S]*?\}/)?.[0];
  if (!block) throw new Error('FFmpeg не вернул измерения громкости.');
  const stats = JSON.parse(block) as Record<string, unknown>;
  const result = Object.fromEntries(['input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset']
    .map(key => [key, Number(stats[key])])) as unknown as LoudnessMeasurement;
  // Silence has -inf measurements and must use the streaming filter unchanged.
  if (!Object.values(result).every(Number.isFinite)) return null;
  if (result.input_i < -99 || result.input_i > 0 || result.input_tp < -99 || result.input_tp > 99
    || result.input_lra < 0 || result.input_lra > 99 || result.input_thresh < -99 || result.input_thresh > 0
    || Math.abs(result.target_offset) > 99) return null;
  return result;
}

export function loudnessFilter(measured?: LoudnessMeasurement | null, report = false): string {
  const mode = measured
    ? `linear=true:measured_I=${measured.input_i}:measured_TP=${measured.input_tp}:measured_LRA=${measured.input_lra}:measured_thresh=${measured.input_thresh}:offset=${measured.target_offset}`
    : 'linear=false';
  return `${stereo},loudnorm=${target}:${mode}${report ? ':print_format=json' : ''},${finite},${resample}`;
}

export function musicPcmArguments(): string[] {
  return ['-map', '0:a:0', '-vn', '-sn', '-dn', '-af', loudnessFilter(),
    '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'];
}

interface EncoderMemory { inPCM: Uint16Array; inPCMPointer: number; inPCMLength: number }

export function configureMusicEncoder(resource: AudioResource): void {
  const encoder = resource.encoder;
  if (!encoder) return;
  const codec: unknown = encoder.encoder;
  if (codec instanceof OpusScript) {
    const memory = codec as unknown as EncoderMemory;
    // opusscript 0.1.1 uses a byte pointer as a Uint16Array element offset.
    // Its helper consumes each PCM byte as a short; a 20 ms stereo frame fits
    // in this allocation when the view starts at the correct byte address.
    memory.inPCM = new Uint16Array(memory.inPCM.buffer, memory.inPCMPointer, memory.inPCMLength / 2);
  }
  encoder.setBitrate(MUSIC_BITRATE);
}
