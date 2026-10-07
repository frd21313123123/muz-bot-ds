import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import type { Readable } from 'node:stream';
import { createRequire } from 'node:module';
import { StreamType } from '@discordjs/voice';
import type { YtdlpClient } from './ytdlp.js';
import { musicPcmArguments, type MusicPlaybackOptions } from './audio.js';
import { performanceEnabled, recordMetric, trackProcess } from './performance.js';
import { terminateProcess } from './processes.js';

function observeFirstAudio(stream: Readable, started: number): void {
  if (!performanceEnabled()) return;
  const first = (): void => {
    if (stream.readableLength > 0) { recordMetric('audio.first', performance.now() - started); cleanup(); }
  };
  const cleanup = (): void => { stream.off('readable', first); stream.off('close', cleanup); };
  stream.on('readable', first); stream.once('close', cleanup);
}

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
      const filters = spawnSync(candidate, ['-hide_banner', '-filters'], {
        encoding: 'utf8', timeout: 10_000, windowsHide: true,
      });
      if (filters.status !== 0 || !['loudnorm', 'aresample', 'aeval', 'asetrate', 'atrim', 'asetpts'].every(filter => filters.stdout.includes(filter))) continue;
      ffmpegCommand = candidate;
      return candidate;
    }
  }
  throw new Error('Не найден FFmpeg с libopus и фильтрами loudnorm, aresample, aeval, asetrate, atrim, asetpts. Установите ffmpeg-static или задайте FFMPEG_PATH.');
}

export interface ManagedAudioStream {
  stream: Readable;
  type: StreamType;
  destroy(): void;
}

export function createRadioStream(url: string): ManagedAudioStream {
  const started = performance.now();
  const parsed = new URL(url);
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Некорректный адрес радио.');
  const child = spawn(requireFfmpeg(), [
    '-nostdin', '-hide_banner', '-loglevel', 'error',
    '-tls_verify', '1', '-rw_timeout', '10000000', '-i', url,
    ...musicPcmArguments(),
  ], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  const output = child.stdout;
  trackProcess(child); observeFirstAudio(output, started);
  let closed = false;
  const destroy = (): void => {
    if (closed) return;
    closed = true; void terminateProcess(child); output.destroy();
  };
  const fail = (): void => {
    if (closed) return;
    closed = true; output.destroy(new Error('Не удалось открыть эфир радиостанции.')); void terminateProcess(child);
  };
  // Never expose stream URLs, tokens or server error bodies in ordinary logs.
  child.stderr.resume();
  output.on('error', () => {});
  child.once('error', fail);
  child.once('close', code => { if (code !== 0) fail(); });
  return { stream: output, type: StreamType.Raw, destroy };
}

// Leave the first PCM bytes buffered for createAudioResource. A successful
// process spawn or HTTP response alone is not a playable broadcast.
export async function waitForAudio(media: ManagedAudioStream, signal: AbortSignal, timeoutMs = 15_000): Promise<void> {
  signal.throwIfAborted();
  const stream = media.stream;
  await new Promise<void>((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timer); signal.removeEventListener('abort', abort);
      stream.off('readable', ready); stream.off('error', fail); stream.off('end', fail); stream.off('close', fail);
    };
    const fail = (): void => { cleanup(); reject(new Error('Эфир радиостанции недоступен.')); };
    const abort = (): void => { cleanup(); reject(new Error('Запрос радио отменён.')); };
    const ready = (): void => { if (stream.readableLength > 0) { cleanup(); resolve(); } };
    const timer = setTimeout(fail, timeoutMs);
    signal.addEventListener('abort', abort, { once: true });
    stream.on('readable', ready); stream.once('error', fail); stream.once('end', fail); stream.once('close', fail);
    if (stream.destroyed || stream.readableEnded) fail(); else ready();
  });
  signal.throwIfAborted();
}

export function createYtdlpStream(ytdlp: YtdlpClient, url: string, options: MusicPlaybackOptions = {}): ManagedAudioStream {
  const started = performance.now();
  const audioArgs = musicPcmArguments(options);
  const ffmpeg = requireFfmpeg();
  const downloader = ytdlp.spawn([
    '-f', 'bestaudio/best', '-o', '-', '--no-playlist', url,
  ]);
  const transcoder = spawn(ffmpeg, [
    '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
    ...audioArgs,
  ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });

  trackProcess(transcoder);
  const media = createProcessStream(downloader, transcoder, StreamType.Raw);
  observeFirstAudio(media.stream, started);
  return media;
}

export function createProcessStream(downloader: ChildProcess, transcoder: ChildProcess, type = StreamType.OggOpus): ManagedAudioStream {
  const output = transcoder.stdout;
  if (!downloader.stdout || !downloader.stderr || !transcoder.stdin || !output || !transcoder.stderr) {
    void terminateProcess(downloader);
    void terminateProcess(transcoder);
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
    void terminateProcess(downloader);
    transcoder.stdin?.destroy();
    void terminateProcess(transcoder);
    output.destroy();
  };
  const fail = (message: string): void => {
    if (closed) return;
    closed = true;
    output.destroy(new Error(message));
    downloader.stdout?.unpipe();
    void terminateProcess(downloader);
    transcoder.stdin?.destroy();
    void terminateProcess(transcoder);
  };
  downloader.on('error', (error) => fail(`yt-dlp: ${error.message}`));
  transcoder.on('error', (error) => fail(`FFmpeg: ${error.message}`));
  downloader.on('close', (code) => {
    if (code !== 0) fail(`yt-dlp завершился с кодом ${code}: ${ytError.trim()}`);
  });
  transcoder.on('close', (code) => {
    if (code !== 0) fail(`FFmpeg завершился с кодом ${code}: ${ffError.trim()}`);
  });

  return { stream: output, type, destroy };
}
