import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import type { Readable } from 'node:stream';
import { createRequire } from 'node:module';
import { StreamType } from '@discordjs/voice';
import type { YtdlpClient } from './ytdlp.js';

let ffmpegCommand: string | null = null;
const require = createRequire(import.meta.url);
const bundledFfmpeg = require('ffmpeg-static') as string | null;

export function requireFfmpeg(): string {
  if (ffmpegCommand) return ffmpegCommand;
  for (const candidate of [process.env.FFMPEG_PATH, bundledFfmpeg, 'ffmpeg']) {
    if (!candidate) continue;
    const check = spawnSync(candidate, ['-hide_banner', '-encoders'], {
      encoding: 'utf8', timeout: 10_000, windowsHide: true,
    });
    if (check.status === 0 && check.stdout.includes('libopus')) {
      ffmpegCommand = candidate;
      return candidate;
    }
  }
  throw new Error('Не найден FFmpeg с кодеком libopus. Установите ffmpeg-static или задайте FFMPEG_PATH.');
}

export interface ManagedAudioStream {
  stream: Readable;
  type: StreamType;
  destroy(): void;
}

export function createYtdlpStream(ytdlp: YtdlpClient, url: string): ManagedAudioStream {
  const downloader = ytdlp.spawn([
    '-f', 'bestaudio[ext=webm]/bestaudio', '-o', '-', '--no-playlist', url,
  ]);
  const transcoder = spawn(requireFfmpeg(), [
    '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
    '-vn', '-c:a', 'libopus', '-ar', '48000', '-ac', '2', '-b:a', '128k',
    '-f', 'ogg', 'pipe:1',
  ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });

  return createProcessStream(downloader, transcoder);
}

export function createProcessStream(downloader: ChildProcess, transcoder: ChildProcess): ManagedAudioStream {
  const output = transcoder.stdout;
  if (!downloader.stdout || !downloader.stderr || !transcoder.stdin || !output || !transcoder.stderr) {
    downloader.kill();
    transcoder.kill();
    throw new Error('Не удалось создать аудиопоток.');
  }
  downloader.stdout.pipe(transcoder.stdin);
  transcoder.stdin.on('error', () => {});

  let closed = false;
  let ytError = '';
  let ffError = '';
  const capture = (child: ChildProcess, append: (chunk: string) => void): void => {
    child.stderr?.on('data', (chunk: Buffer) => append(chunk.toString().slice(0, 500)));
  };
  capture(downloader, (chunk) => { ytError = (ytError + chunk).slice(-1000); });
  capture(transcoder, (chunk) => { ffError = (ffError + chunk).slice(-1000); });

  const destroy = (): void => {
    if (closed) return;
    closed = true;
    downloader.stdout?.unpipe();
    downloader.kill();
    transcoder.stdin?.destroy();
    transcoder.kill();
    output.destroy();
  };
  const fail = (message: string): void => {
    if (closed) return;
    closed = true;
    downloader.stdout?.unpipe();
    downloader.kill();
    transcoder.stdin?.destroy();
    transcoder.kill();
    output.destroy(new Error(message));
  };
  downloader.on('error', (error) => fail(`yt-dlp: ${error.message}`));
  transcoder.on('error', (error) => fail(`FFmpeg: ${error.message}`));
  downloader.on('close', (code) => {
    if (code !== 0) fail(`yt-dlp завершился с кодом ${code}: ${ytError.trim()}`);
  });
  transcoder.on('close', (code) => {
    if (code !== 0) fail(`FFmpeg завершился с кодом ${code}: ${ffError.trim()}`);
  });

  return { stream: output, type: StreamType.OggOpus, destroy };
}
