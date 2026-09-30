import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { access, realpath } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { loudnessFilter, parseLoudnessMeasurement } from './audio.js';
import { requireFfmpeg } from './stream.js';

const exec = promisify(execFile);

export async function normalizeAudioFile(input: string, output: string): Promise<void> {
  const source = await realpath(input);
  const destination = path.resolve(output);
  const extension = path.extname(destination).toLowerCase();
  if (!['.flac', '.wav'].includes(extension)) throw new Error('Выходной файл должен иметь расширение .flac или .wav.');
  await access(source, constants.R_OK);
  const exists = await access(destination).then(() => true, (error: NodeJS.ErrnoException) => {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    return false;
  });
  if (exists) throw new Error('Выходной файл уже существует; укажите новое имя.');
  const ffmpeg = requireFfmpeg();
  const common = ['-nostdin', '-hide_banner', '-nostats'];
  const { stderr } = await exec(ffmpeg, [...common, '-i', source, '-map', '0:a:0',
    '-af', loudnessFilter(null, true), '-f', 'null', '-'],
  { windowsHide: true, maxBuffer: 2 * 1024 * 1024 });
  const measured = parseLoudnessMeasurement(stderr);
  await exec(ffmpeg, [...common, '-loglevel', 'error', '-n', '-i', source, '-map', '0:a:0',
    '-vn', '-sn', '-dn', '-af', loudnessFilter(measured), '-ar', '48000', '-ac', '2',
    '-c:a', extension === '.flac' ? 'flac' : 'pcm_s16le', destination],
  { windowsHide: true, maxBuffer: 2 * 1024 * 1024 });
}
