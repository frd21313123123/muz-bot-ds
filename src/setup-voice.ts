import 'dotenv/config';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { sttProvider } from './voice/groq.js';
import { voiceProfile } from './voice/profile.js';

if (sttProvider() === 'groq') {
  if (!process.env.GROQ_API_KEY?.trim()) throw new Error('Укажите GROQ_API_KEY в .env.');
  if (voiceProfile() === 'light') {
    console.log('Groq whisper-large-v3-turbo + правила готовы; локальные модели STT не нужны.');
    process.exit(0);
  }
}
const candidates = process.platform === 'win32'
  ? [['py', '-3'], ['python'], ['python3']] : [['python3'], ['python']];
const python = candidates.find(([command, ...prefix]) => spawnSync(command!, [...prefix, '-c',
  'import sys; assert sys.version_info >= (3, 11)'], { windowsHide: true, timeout: 10_000 }).status === 0);
if (!python) throw new Error('Нужен Python 3.11+ для setup:voice.');
const [command, ...prefix] = python;
const child = spawn(command!, [...prefix, path.resolve('scripts/setup_voice.py'), ...process.argv.slice(2)], { windowsHide: true, stdio: 'inherit' });
child.once('error', () => { console.error('Не удалось запустить setup:voice.'); process.exitCode = 1; });
child.once('exit', (code) => { process.exitCode = code ?? 1; });
