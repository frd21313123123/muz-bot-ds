import 'dotenv/config';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { requireFfmpeg } from './utils/stream.js';
import { voicePython } from './voice/runtime.js';

const candidates = process.platform === 'win32'
  ? [[voicePython()], ['py', '-3'], ['python'], ['python3']]
  : [[voicePython()], ['python3'], ['python']];
const python = candidates.find(([command, ...prefix]) => spawnSync(command!, [...prefix, '-c',
  'import sys; assert sys.version_info >= (3, 11)'], { windowsHide: true, timeout: 10_000 }).status === 0);
if (!python) throw new Error('Нужен Python 3.11+ для setup:tts.');
const [command, ...prefix] = python;
const child = spawn(command!, [...prefix, path.resolve('scripts/setup_tts.py'), requireFfmpeg()], {
  windowsHide: true, stdio: 'inherit',
});
child.once('error', () => { console.error('Не удалось запустить setup:tts.'); process.exitCode = 1; });
child.once('exit', (code) => { process.exitCode = code ?? 1; });
