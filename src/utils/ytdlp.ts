import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { promisify } from 'node:util';
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import type { Track, YoutubeTrack } from '../types.js';
import { formatDuration, videoDuration } from './duration.js';
import { MetadataRequests, metadataOptions, type MetadataInput, type MetadataOptions } from './MetadataRequests.js';
import { countMetric, recordMetric, trackProcess } from './performance.js';
import { terminateProcess } from './processes.js';

const execFileAsync = promisify(execFile);
const VENV_DIR = path.resolve('.runtime', 'yt-dlp');
const COMMON_ARGS = ['--ignore-config', '--no-warnings', '--js-runtimes', `node:${process.execPath}`];

interface Runtime { command: string; prefix: string[] }

// YouTube's EJS subprocess can otherwise grow to hundreds of MiB. This affects
// only yt-dlp's Node runtime, not the bot or the audio encoder.
export function ytdlpEnvironment(source: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env = { ...source };
  if (!/(?:^|\s)--max[-_]old[-_]space[-_]size(?:=|\s)/.test(env.NODE_OPTIONS ?? '')) {
    const mb = Number(env.YT_DLP_NODE_HEAP_MB ?? 128);
    if (Number.isInteger(mb) && mb >= 96 && mb <= 1024) {
      env.NODE_OPTIONS = `${env.NODE_OPTIONS ?? ''} --max-old-space-size=${mb}`.trim();
    }
  }
  return env;
}

export interface VideoInfo {
  id?: string;
  title?: string;
  duration?: number;
  duration_string?: string;
  thumbnail?: string;
  webpage_url?: string;
  artist?: string;
  artists?: string[];
  track?: string;
  categories?: string[];
  is_live?: boolean;
  live_status?: string;
  channel?: string;
  uploader?: string;
  entries?: VideoInfo[];
}

// yt-dlp's full output contains large format/subtitle collections we never use.
export function compactVideoInfo(info: VideoInfo): VideoInfo {
  const { id, title, duration, duration_string, thumbnail, webpage_url, artist, track,
    is_live, live_status, channel, uploader } = info;
  return { id, title, duration, duration_string, thumbnail, webpage_url, artist, track,
    is_live, live_status, channel, uploader, artists: info.artists?.slice(), categories: info.categories?.slice(),
    ...(info.entries ? { entries: info.entries.map(compactVideoInfo) } : {}) };
}

async function probe(command: string, args: string[]): Promise<boolean> {
  try {
    await execFileAsync(command, args, { timeout: 10_000, windowsHide: true });
    return true;
  } catch {
    return false;
  }
}

async function findPython(): Promise<Runtime> {
  const candidates: Runtime[] = process.platform === 'win32'
    ? [{ command: 'py', prefix: ['-3'] }, { command: 'python', prefix: [] }, { command: 'python3', prefix: [] }]
    : [{ command: 'python3', prefix: [] }, { command: 'python', prefix: [] }];
  for (const candidate of candidates) {
    if (await probe(candidate.command, [...candidate.prefix, '-c', 'import sys; assert sys.version_info >= (3, 11)'])) {
      return candidate;
    }
  }
  throw new Error('Нужен Python 3.11+ с модулем venv. Установите Python и перезапустите бота.');
}

function venvPython(): string {
  return path.join(VENV_DIR, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
}

export async function prepareYtdlp(update = false): Promise<YtdlpClient> {
  const override = process.env.YT_DLP_PATH?.trim();
  if (override) {
    if (!await probe(override, ['--version'])) {
      throw new Error(`YT_DLP_PATH не запускается: ${override}`);
    }
    if (update) throw new Error('Обновите внешнюю установку yt-dlp самостоятельно или уберите YT_DLP_PATH.');
    return new YtdlpClient({ command: override, prefix: [] });
  }

  const python = venvPython();
  if (!existsSync(python)) {
    const host = await findPython();
    await mkdir(path.dirname(VENV_DIR), { recursive: true });
    try {
      await execFileAsync(host.command, [...host.prefix, '-m', 'venv', VENV_DIR], {
        timeout: 120_000, windowsHide: true,
      });
    } catch (error) {
      throw new Error(`Не удалось создать окружение yt-dlp: ${String(error)}`);
    }
  }

  const installed = await probe(python, ['-m', 'yt_dlp', '--version']);
  const ejsInstalled = await probe(python, ['-m', 'pip', 'show', 'yt-dlp-ejs']);
  if (!installed || !ejsInstalled || update) {
    console.log('Устанавливаю yt-dlp[default] в .runtime/yt-dlp…');
    try {
      await execFileAsync(python, ['-m', 'pip', 'install', '--upgrade', 'yt-dlp[default]'], {
        timeout: 180_000, maxBuffer: 4 * 1024 * 1024, windowsHide: true,
      });
    } catch (error) {
      throw new Error(`Не удалось установить yt-dlp[default]. Проверьте pip и доступ к PyPI: ${String(error)}`);
    }
  }
  if (!await probe(python, ['-m', 'yt_dlp', '--version'])) {
    throw new Error('yt-dlp не запускается после установки. Проверьте Python и pip.');
  }
  return new YtdlpClient({ command: python, prefix: ['-m', 'yt_dlp'] });
}

export class YtdlpClient {
  private readonly requests = new MetadataRequests<VideoInfo>();
  private readonly searchCache = new Map<string, { entries: VideoInfo[]; timestamp: number }>();
  private readonly videoCache = new Map<string, { info: VideoInfo; timestamp: number }>();

  constructor(private readonly runtime: Runtime) {}

  clearCache(): void {
    this.searchCache.clear();
    this.videoCache.clear();
  }

  spawn(args: string[]): ChildProcess {
    const child = spawn(this.runtime.command, [...this.runtime.prefix, ...COMMON_ARGS, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true, env: ytdlpEnvironment(),
    });
    trackProcess(child);
    return child;
  }

  async json(args: string[], timeout = 20_000, signal?: AbortSignal): Promise<VideoInfo> {
    signal?.throwIfAborted();
    const stdout = await new Promise<string>((resolve, reject) => {
      let settled = false;
      let timer: NodeJS.Timeout | undefined;
      const cleanup = (): void => { if (timer) clearTimeout(timer); signal?.removeEventListener('abort', abort); };
      const fail = (): void => {
        if (settled) return;
        settled = true; cleanup();
        void terminateProcess(child).then(() => reject(new Error(signal?.aborted ? 'Metadata request cancelled' : 'yt-dlp metadata unavailable')));
      };
      const abort = (): void => fail();
      const child = execFile(this.runtime.command, [...this.runtime.prefix, ...COMMON_ARGS, '-J', ...args],
        { maxBuffer: 16 * 1024 * 1024, windowsHide: true, env: ytdlpEnvironment() }, (error, output) => {
          if (error) { fail(); return; }
          if (!settled) { settled = true; cleanup(); resolve(output); }
        });
      trackProcess(child);
      timer = setTimeout(fail, timeout);
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) fail();
    });
    return compactVideoInfo(JSON.parse(stdout) as VideoInfo);
  }

  async getStreamUrl(url: string, signal?: AbortSignal): Promise<string> {
    const { stdout } = await execFileAsync(this.runtime.command,
      [...this.runtime.prefix, ...COMMON_ARGS, '-g', '-f', 'bestaudio/best', '--no-playlist', url],
      { timeout: 20_000, maxBuffer: 1024 * 1024, windowsHide: true, signal, env: ytdlpEnvironment() },
    );
    return stdout.trim().split(/\r?\n/)[0] ?? '';
  }

  async video(url: string, input?: MetadataInput): Promise<VideoInfo> {
    const options = metadataOptions(input);
    options.signal?.throwIfAborted();
    this.pruneCaches();
    const cached = this.videoCache.get(url);
    if (cached && Date.now() - cached.timestamp < 30 * 60_000) {
      countMetric('cache.hit'); return compactVideoInfo(cached.info);
    }
    countMetric('cache.miss');
    const info = compactVideoInfo(await this.requests.run(`video:${url}`,
      signal => this.json(['--no-playlist', url], 20_000, signal), options));
    options.signal?.throwIfAborted();
    if (info.id) {
      this.videoCache.set(url, { info: compactVideoInfo(info), timestamp: Date.now() });
      if (this.videoCache.size > 500) {
        const first = this.videoCache.keys().next().value;
        if (first) this.videoCache.delete(first);
      }
    }
    return info;
  }

  async playlist(url: string, limit: number, input?: MetadataInput): Promise<VideoInfo> {
    return this.requests.run(`playlist:${url}:${limit}`, signal =>
      this.json(['--flat-playlist', '--playlist-items', `1:${limit}`, url], 30_000, signal), metadataOptions(input));
  }

  async search(query: string, input?: MetadataInput): Promise<VideoInfo | null> {
    return (await this.searchCandidates(query, 1, input))[0] ?? null;
  }

  async searchCandidates(query: string, limit = 5, input?: MetadataInput): Promise<VideoInfo[]> {
    const options = metadataOptions(input);
    const signal = options.signal;
    signal?.throwIfAborted();
    if (!query.trim()) return [];
    if (!Number.isInteger(limit) || limit < 1 || limit > 10) throw new Error('Invalid search limit');

    const cacheKey = `${query.trim().toLowerCase()}:${limit}`;
    this.pruneCaches();
    const cached = this.searchCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < 15 * 60_000) {
      countMetric('cache.hit'); return cached.entries.map(compactVideoInfo);
    }

    // The songs section excludes podcasts and other ordinary YouTube videos.
    // Do not fall back to unrestricted video search when no song is found.
    const url = `https://music.youtube.com/search?q=${encodeURIComponent(query.trim())}#songs`;
    countMetric('cache.miss');
    const started = performance.now();
    const result = await this.requests.run(`search:${cacheKey}`, sharedSignal =>
      this.json(['--flat-playlist', '--playlist-end', String(limit), url], 20_000, sharedSignal), options);
    recordMetric('search', performance.now() - started);
    signal?.throwIfAborted();
    const seen = new Set<string>();
    const candidates = (result.entries ?? (result.id ? [result] : [])).filter((entry) => {
      if (!entry?.id || !/^[\w-]{11}$/.test(entry.id) || !entry.title?.trim() || seen.has(entry.id)) return false;
      seen.add(entry.id);
      return true;
    }).slice(0, limit).map(compactVideoInfo);

    if (candidates.length) {
      this.searchCache.set(cacheKey, { entries: candidates.map(compactVideoInfo), timestamp: Date.now() });
      if (this.searchCache.size > 200) {
        const first = this.searchCache.keys().next().value;
        if (first) this.searchCache.delete(first);
      }
    }
    return candidates;
  }

  private pruneCaches(): void {
    const now = Date.now();
    for (const [key, item] of this.searchCache) if (now - item.timestamp >= 15 * 60_000) this.searchCache.delete(key);
    for (const [key, item] of this.videoCache) if (now - item.timestamp >= 30 * 60_000) this.videoCache.delete(key);
  }

  async related(videoId: string, limit = 25, options: MetadataOptions & { excludeIds?: ReadonlySet<string> } = {}): Promise<Track[]> {
    if (!/^[\w-]{11}$/.test(videoId)) return [];
    if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error('Invalid recommendation limit');
    const signal = AbortSignal.any([AbortSignal.timeout(25_000), ...(options.signal ? [options.signal] : [])]);
    const requestOptions: MetadataOptions = { priority: options.priority ?? 'background', signal };
    const candidateLimit = options.excludeIds ? 51 : Math.min(50, limit) + 1;
    const result = await this.requests.run(`related:${videoId}:${candidateLimit}`, sharedSignal => this.json([
      '--flat-playlist', '--playlist-items', `2:${candidateLimit}`,
      `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`,
    ], 25_000, sharedSignal), requestOptions);
    const seen = new Set([videoId]);
    const candidates = (result.entries ?? []).filter((entry) => {
      if (!entry?.id || !/^[\w-]{11}$/.test(entry.id) || !entry.title?.trim() || seen.has(entry.id) || options.excludeIds?.has(entry.id)) return false;
      seen.add(entry.id);
      return !isNonSong(entry);
    }).slice(0, 50);
    // Flat radio entries lack categories. Verify them before queueing, with
    // bounded concurrency and a shared deadline so a radio cannot stall playback.
    const songs: VideoInfo[] = [];
    for (let offset = 0; offset < candidates.length && songs.length < limit && !signal.aborted; offset++) {
      const entry = candidates[offset]!;
      try {
        const info = await this.video(`https://www.youtube.com/watch?v=${entry.id}`, requestOptions);
        if (info.id === entry.id && isSong(info)) songs.push(info);
      } catch { if (signal.aborted) break; }
    }
    return songs.slice(0, limit).map((info) => toTrack(info, '🤖 Бесконечное', true));
  }
}

export function isNonSong(info: VideoInfo): boolean {
  if (info.is_live || ['is_live', 'is_upcoming'].includes(info.live_status ?? '')) return true;
  const title = (info.title ?? '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ');
  return /(?:^| )(?:podcasts?|подкаст\p{L}*|interviews?|интервью|аудиокниг\p{L}*|audiobooks?|лекци\p{L}*|lectures?)(?: |$)/u.test(title)
    || /(?:^| )(?:full album|full concert|dj set|полный альбом|полный концерт|сборник)(?: |$)/u.test(title);
}

export function isSong(info: VideoInfo): boolean {
  if (!info.id || !/^[\w-]{11}$/.test(info.id) || !info.title?.trim() || isNonSong(info)) return false;
  // Artist channels also upload interviews: a channel name alone is not proof.
  return info.categories?.some((category) => category.toLowerCase() === 'music') === true
    || Boolean(info.track?.trim() && (info.artist?.trim() || info.artists?.some((artist) => artist.trim())));
}

export function toTrack(info: VideoInfo, requestedBy: string, isAutoplay = false): YoutubeTrack {
  if (!info.id || !/^[\w-]{11}$/.test(info.id)) {
    throw new Error('YouTube не вернул корректный идентификатор видео.');
  }
  const seconds = videoDuration(info);
  const duration = seconds === null ? '?' : formatDuration(seconds);
  return {
    url: `https://www.youtube.com/watch?v=${info.id}`,
    videoId: info.id,
    title: info.title?.trim() || 'Без названия',
    duration,
    thumbnail: info.thumbnail || `https://img.youtube.com/vi/${info.id}/hqdefault.jpg`,
    requestedBy,
    isAutoplay,
    isLive: isLiveVideo(info),
  };
}

export function isLiveVideo(info: VideoInfo): boolean {
  return info.live_status === 'is_live' || (info.live_status === undefined && info.is_live === true);
}
