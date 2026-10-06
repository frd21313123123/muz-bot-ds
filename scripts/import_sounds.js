import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ffmpeg = require('ffmpeg-static');

const root = path.resolve('.');
const soundsDir = path.join(root, 'sounds');
const assetsDir = path.join(root, 'assets', 'tts');
const runtimeDir = path.join(root, '.runtime', 'tts');

const map = {
  play: 'ElevenLabs_2026-10-06T19_00_23_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  queued: 'ElevenLabs_2026-10-06T19_00_35_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  pause: 'ElevenLabs_2026-10-06T19_00_58_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  resume: 'ElevenLabs_2026-10-06T19_01_10_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  skip: 'ElevenLabs_2026-10-06T19_01_20_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  stop: 'ElevenLabs_2026-10-06T19_01_35_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  not_found: 'ElevenLabs_2026-10-06T19_02_21_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  volume_up: 'ElevenLabs_2026-10-06T19_02_29_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  volume_down: 'ElevenLabs_2026-10-06T19_03_01_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  volume_set: 'ElevenLabs_2026-10-06T19_03_09_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  autoplay_on: 'ElevenLabs_2026-10-06T19_03_18_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  autoplay_off: 'ElevenLabs_2026-10-06T19_03_27_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  loop_on: 'ElevenLabs_2026-10-06T19_03_37_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  loop_off: 'ElevenLabs_2026-10-06T19_03_46_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  queue_clear: 'ElevenLabs_2026-10-06T19_03_54_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  radio_play: 'ElevenLabs_2026-10-06T19_04_04_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  radio_not_found: 'ElevenLabs_2026-10-06T19_04_13_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  radio_unsupported: 'ElevenLabs_2026-10-06T19_04_19_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  voice_on: 'ElevenLabs_2026-10-06T19_04_28_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  voice_off: 'ElevenLabs_2026-10-06T19_04_36_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
  unknown: 'ElevenLabs_2026-10-06T19_04_44_Artspace - Engaging, Warm and Pleasant_pvc_sp100_s45_sb34_v4.mp3',
};

const phrases = JSON.parse(fs.readFileSync(path.join(root, 'src', 'voice', 'confirmations.json'), 'utf8'));

fs.mkdirSync(assetsDir, { recursive: true });
fs.mkdirSync(runtimeDir, { recursive: true });

for (const [key, filename] of Object.entries(map)) {
  const inputPath = path.join(soundsDir, filename);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`File not found: ${inputPath}`);
  }

  // Trim silence at both ends, slight tempo for longest phrase to guarantee < 2s duration,
  // EBU R128 loudnorm (-18 LUFS, -6 dBTP), resample to 48kHz stereo 16-bit LE PCM
  const afFilter = key === 'radio_unsupported'
    ? 'silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,atempo=1.04,loudnorm=I=-18:TP=-6:LRA=7,aresample=48000'
    : 'silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,loudnorm=I=-18:TP=-6:LRA=7,aresample=48000';

  const res = spawnSync(ffmpeg, [
    '-hide_banner', '-loglevel', 'error',
    '-y', '-i', inputPath,
    '-af', afFilter,
    '-ar', '48000', '-ac', '2', '-f', 's16le', 'pipe:1'
  ]);

  if (res.status !== 0 || !res.stdout || !res.stdout.length) {
    throw new Error(`FFmpeg error converting ${key}: ${res.stderr?.toString()}`);
  }

  const pcm = res.stdout;
  if (pcm.length % 4 !== 0) {
    throw new Error(`PCM length not divisible by 4 for ${key}`);
  }

  const durationSec = pcm.length / 192000;
  console.log(`Processed ${key.padEnd(20)} duration: ${durationSec.toFixed(3)}s (${pcm.length} bytes)`);

  const gz = zlib.gzipSync(pcm, { level: 9 });
  fs.writeFileSync(path.join(assetsDir, `${key}.pcm.gz`), gz);
  fs.writeFileSync(path.join(runtimeDir, `${key}.pcm.gz`), gz);
  // Also write uncompressed pcm in runtimeDir for compatibility
  fs.writeFileSync(path.join(runtimeDir, `${key}.pcm`), pcm);
}

const manifest = {
  engine: 'bundled',
  voice: 'elevenlabs/artspace',
  sampleRate: 48000,
  channels: 2,
  phrases,
};

const manifestJson = JSON.stringify(manifest, null, 2);
fs.writeFileSync(path.join(assetsDir, 'ready.json'), manifestJson, 'utf8');
fs.writeFileSync(path.join(runtimeDir, 'ready.json'), manifestJson, 'utf8');

console.log('Successfully updated assets/tts and .runtime/tts with ElevenLabs audio clips!');
