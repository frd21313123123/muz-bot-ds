import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { PassThrough, pipeline, Transform, Readable } from 'node:stream';
import { setImmediate as nextTurn } from 'node:timers/promises';
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

// 300 ms of 48 kHz stereo s16le. Once playback starts the transform passes
// chunks straight through with ordinary stream backpressure.
const STARTUP_PCM_BYTES = 48_000 * 2 * 2 * 0.3;
const STARTUP_WAIT_MS = 400;

export function bufferMusicPcm(input: Readable): Readable {
  let pending: Buffer[] = [];
  let bytes = 0;
  let started = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const release = (): void => {
    clearTimeout(timer);
    started = true;
    if (bytes) output.push(Buffer.concat(pending, bytes));
    pending = []; bytes = 0;
  };
  const output = new Transform({
    readableHighWaterMark: 2 * 48_000 * 2 * 2, // two seconds of PCM once the producer gets ahead
    transform(chunk: Buffer, _encoding, done) {
      if (!chunk.length) { done(); return; }
      if (started) { done(null, chunk); return; }
      if (bytes + chunk.length >= STARTUP_PCM_BYTES) {
        if (bytes) { pending.push(chunk); bytes += chunk.length; release(); done(); }
        else { started = true; done(null, chunk); }
        return;
      }
      if (!timer) timer = setTimeout(release, STARTUP_WAIT_MS);
      pending.push(chunk); bytes += chunk.length;
      done();
    },
    flush(done) { release(); done(); },
    destroy(error, done) { clearTimeout(timer); pending = []; bytes = 0; done(error); },
  });
  // pipeline forwards source errors and tears down the upstream pipe if the
  // consumer cancels before the startup buffer has been released.
  pipeline(input, output, () => {});
  return output;
}

// prism-media encodes an entire input chunk synchronously. FFmpeg and the
// startup buffer can deliver many frames at once. Keep the larger upstream
// reservoir, but let Discord's 20 ms dispatch timer run between small batches.
export function pcmForEncoding(input: Readable): Readable {
  const batchBytes = 2 * 960 * 2 * 2; // two 20 ms stereo s16 frames
  const output = new PassThrough({ highWaterMark: batchBytes });
  pipeline(input, async function* (source: AsyncIterable<Buffer>) {
    for await (const chunk of source) {
      for (let offset = 0; offset < chunk.length; offset += batchBytes) {
        yield chunk.subarray(offset, offset + batchBytes);
        await nextTurn();
      }
    }
  }, output, () => {});
  return output;
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
  const output = bufferMusicPcm(child.stdout);
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

// Keep complete public YouTube Opus downloads briefly for replays and speed changes.
// Up to three 4 MiB entries and one in-flight capture per bot, all compressed.
class TrackAudioCache {
  private readonly tracks = new Map<string, { chunks: Buffer[]; until: number }>();
  private capturing = false;

  read(url: string): Readable | undefined {
    const entry = this.tracks.get(url);
    if (!entry) return;
    this.tracks.delete(url);
    if (entry.until < Date.now()) return;
    this.tracks.set(url, entry);
    return Readable.from(entry.chunks, { objectMode: false });
  }

  capture(url: string, child: ChildProcess): void {
    if (this.capturing || !child.stdout) return;
    this.capturing = true;
    const stdout = child.stdout;
    let bytes = 0;
    const chunks: Buffer[] = [];
    const collect = (chunk: Buffer): void => {
      bytes += chunk.length;
      if (bytes <= 4 * 1024 * 1024) chunks.push(chunk);
      else { chunks.length = 0; stdout.off('data', collect); }
    };
    stdout.on('data', collect);
    child.once('close', code => {
      this.capturing = false;
      stdout.off('data', collect);
      if (code !== 0 || !stdout.readableEnded || !chunks.length) return;
      if (this.tracks.size >= 3) this.tracks.delete(this.tracks.keys().next().value!);
      this.tracks.set(url, { chunks, until: Date.now() + 15 * 60_000 });
    });
  }
}
const audioCaches = new WeakMap<YtdlpClient, TrackAudioCache>();

export function createYtdlpStream(ytdlp: YtdlpClient, url: string, options: MusicPlaybackOptions = {}, isLive = false): ManagedAudioStream {
  const started = performance.now();
  const audioArgs = musicPcmArguments(options);
  const ffmpeg = requireFfmpeg();
  const outputArgs = ['-o', '-', '--no-playlist', url];
  const download = () => ytdlp.spawn(['-f', 'bestaudio/best', ...outputArgs]);
  // Standard YouTube songs usually have a direct Opus stream (itag 251).
  // Skip enumerating HLS/DASH manifests when we can play that stream.
  const direct = !isLive && /^https:\/\/www\.youtube\.com\/watch\?v=[\w-]{11}$/.test(url);
  let source: ChildProcess | Readable;
  if (direct) {
    const cache = audioCaches.get(ytdlp) ?? new TrackAudioCache();
    audioCaches.set(ytdlp, cache);
    const hit = cache.read(url);
    if (hit) source = hit;
    else {
      // This client provides the direct HTTPS Opus format. Enumerating the
      // other clients here can spend seconds on formats we will never select.
      source = ytdlp.spawn(['--extractor-args', 'youtube:player_client=visionos;skip=hls,dash',
        '-f', '251[protocol=https]', ...outputArgs]);
      cache.capture(url, source);
    }
  } else source = download();
  const transcoder = spawn(ffmpeg, [
    '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
    ...audioArgs,
  ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });

  trackProcess(transcoder);
  const media = createProcessStream(source, transcoder, StreamType.Raw, direct ? download : undefined);
  observeFirstAudio(media.stream, started);
  return media;
}

export function createProcessStream(downloader: ChildProcess | Readable, transcoder: ChildProcess, type = StreamType.OggOpus,
  fallback?: () => ChildProcess): ManagedAudioStream {
  const stopDownload = (): void => {
    if (downloader instanceof Readable) downloader.destroy();
    else { downloader.stdout?.unpipe(); void terminateProcess(downloader); }
  };
  if ((!(downloader instanceof Readable) && (!downloader.stdout || !downloader.stderr))
    || !transcoder.stdin || !transcoder.stdout || !transcoder.stderr) {
    stopDownload();
    void terminateProcess(transcoder);
    throw new Error('Не удалось создать аудиопоток.');
  }
  const output = type === StreamType.Raw ? bufferMusicPcm(transcoder.stdout) : transcoder.stdout;
  // Download compressed bytes ahead of playback. The first chunk passes immediately.
  // Let yt-dlp finish a typical YouTube song and release its Python heap
  // while FFmpeg consumes the compressed bytes at playback speed.
  const readAhead = fallback && !(downloader instanceof Readable) ? 4 : 1;
  const input = new PassThrough({ highWaterMark: readAhead * 1024 * 1024 });
  input.pipe(transcoder.stdin);
  transcoder.stdin.on('error', () => {});

  let closed = false;
  let ytError = '';
  let ffError = '';
  const capture = (child: ChildProcess, append: (chunk: string) => void): void => {
    child.stderr?.on('data', (chunk: Buffer) => append(chunk.toString().slice(0, 500)));
  };
  capture(transcoder, (chunk) => { ffError = (ffError + chunk).slice(-1000); });

  const destroy = (): void => {
    if (closed) return;
    closed = true;
    stopDownload(); input.destroy();
    transcoder.stdin?.destroy();
    void terminateProcess(transcoder);
    output.destroy();
  };
  const fail = (message: string): void => {
    if (closed) return;
    closed = true;
    output.destroy(new Error(message));
    stopDownload(); input.destroy();
    transcoder.stdin?.destroy();
    void terminateProcess(transcoder);
  };
  transcoder.on('error', (error) => fail(`FFmpeg: ${error.message}`));
  const pipeDownload = (child: ChildProcess): void => {
    let received = false;
    child.once('error', error => fail(`yt-dlp: ${error.message}`));
    capture(child, chunk => { ytError = (ytError + chunk).slice(-1000); });
    if (!child.stdout) { fail('Не удалось открыть выход yt-dlp.'); return; }
    child.stdout.once('data', () => { received = true; });
    // Keep FFmpeg's input open until we know whether another format is needed.
    child.stdout.pipe(input, { end: false });
    child.once('close', code => {
      if (closed) return;
      if (!received && fallback) {
        const retry = fallback;
        fallback = undefined;
        ytError = '';
        try { downloader = retry(); pipeDownload(downloader); }
        catch (error) { fail(`yt-dlp: ${String(error)}`); }
        return;
      }
      if (code === 0) { input.end(); return; }
      fail(`yt-dlp завершился с кодом ${code}: ${ytError.trim()}`);
    });
  };
  if (downloader instanceof Readable) downloader.pipe(input);
  else pipeDownload(downloader);
  transcoder.on('close', (code) => {
    if (code !== 0) fail(`FFmpeg завершился с кодом ${code}: ${ffError.trim()}`);
  });

  return { stream: output, type, destroy };
}
