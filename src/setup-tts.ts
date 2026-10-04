import 'dotenv/config';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { requireFfmpeg } from './utils/stream.js';
import { voicePython } from './voice/runtime.js';
import { mkdir, readFile, copyFile, rename } from 'node:fs/promises';
import phrases from './voice/confirmations.json' with { type: 'json' };

const engine = process.env.VOICE_TTS_ENGINE ?? 'bundled';
if (engine === 'bundled') {
  const source = path.resolve('assets/tts'); const target = path.resolve('.runtime/tts');
  const ready = await readFile(path.join(source, 'ready.json'));
  if (JSON.stringify(JSON.parse(ready.toString()).phrases) !== JSON.stringify(phrases)) throw new Error('Готовые записи TTS устарели.');
  await mkdir(target, { recursive: true });
  for (const key of Object.keys(phrases)) await copyFile(path.join(source, `${key}.pcm.gz`), path.join(target, `${key}.pcm.gz`));
  await copyFile(path.join(source, 'ready.json'), path.join(target, 'ready.json.tmp'));
  await rename(path.join(target, 'ready.json.tmp'), path.join(target, 'ready.json'));
  console.log('Короткие готовые подтверждения установлены; модель TTS не нужна.');
} else {

const candidates = process.platform === 'win32'
  ? [[voicePython('heavy')], [voicePython()], ['py', '-3'], ['python'], ['python3']]
  : [[voicePython('heavy')], [voicePython()], ['python3'], ['python']];
const python = candidates.find(([command, ...prefix]) => spawnSync(command!, [...prefix, '-c',
  'import sys; assert sys.version_info >= (3, 11)'], { windowsHide: true, timeout: 10_000 }).status === 0);
if (!python) throw new Error('Нужен Python 3.11+ для setup:tts.');
const [command, ...prefix] = python;
if (engine !== 'silero' && engine !== 'piper') throw new Error('VOICE_TTS_ENGINE: bundled, silero или piper.');
const script = engine === 'silero' ? 'setup_silero_tts.py' : 'setup_tts.py';
const child = spawn(command!, [...prefix, path.resolve('scripts', script), requireFfmpeg()], {
  windowsHide: true, stdio: 'inherit',
});
child.once('error', () => { console.error('Не удалось запустить setup:tts.'); process.exitCode = 1; });
child.once('exit', (code) => { process.exitCode = code ?? 1; });
}
