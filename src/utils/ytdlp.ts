import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { promisify } from 'node:util';
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import type { Track, YoutubeTrack } from '../types.js';
import { formatDuration, videoDuration } from './duration.js';

const execFileAsync = promisify(execFile);
const VENV_DIR = path.resolve('.runtime', 'yt-dlp');
const COMMON_ARGS = ['--ignore-config', '--no-warnings', '--js-runtimes', `node:${process.execPath}`];

interface Runtime { command: string; prefix: string[] }

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
  private readonly searchCache = new Map<string, { entries: VideoInfo[]; timestamp: number }>();
  private readonly videoCache = new Map<string, { info: VideoInfo; timestamp: number }>();

  constructor(private readonly runtime: Runtime) {}

  clearCache(): void {
    this.searchCache.clear();
    this.videoCache.clear();
  }

  spawn(args: string[]): ChildProcess {
    return spawn(this.runtime.command, [...this.runtime.prefix, ...COMMON_ARGS, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    });
  }

  async json(args: string[], timeout = 20_000, signal?: AbortSignal): Promise<VideoInfo> {
    const { stdout } = await execFileAsync(this.runtime.command,
      [...this.runtime.prefix, ...COMMON_ARGS, '-J', ...args],
      { timeout, maxBuffer: 16 * 1024 * 1024, windowsHide: true, signal },
    );
    return JSON.parse(stdout) as VideoInfo;
  }

  async getStreamUrl(url: string, signal?: AbortSignal): Promise<string> {
    const { stdout } = await execFileAsync(this.runtime.command,
      [...this.runtime.prefix, ...COMMON_ARGS, '-g', '-f', 'bestaudio/best', '--no-playlist', url],
      { timeout: 20_000, maxBuffer: 1024 * 1024, windowsHide: true, signal },
    );
    return stdout.trim().split(/\r?\n/)[0] ?? '';
  }

  async video(url: string, signal?: AbortSignal): Promise<VideoInfo> {
    const cached = this.videoCache.get(url);
    if (cached && Date.now() - cached.timestamp < 30 * 60_000) {
      return cached.info;
    }
    const info = await this.json(['--no-playlist', url], 20_000, signal);
    if (info.id) {
      this.videoCache.set(url, { info, timestamp: Date.now() });
      if (this.videoCache.size > 500) {
        const first = this.videoCache.keys().next().value;
        if (first) this.videoCache.delete(first);
      }
    }
    return info;
  }

  async playlist(url: string, limit: number): Promise<VideoInfo> {
    return this.json(['--flat-playlist', '--playlist-items', `1:${limit}`, url], 30_000);
  }

  async search(query: string): Promise<VideoInfo | null> {
    return (await this.searchCandidates(query, 1))[0] ?? null;
  }

  async searchCandidates(query: string, limit = 5, signal?: AbortSignal): Promise<VideoInfo[]> {
    signal?.throwIfAborted();
    if (!query.trim()) return [];
    if (!Number.isInteger(limit) || limit < 1 || limit > 10) throw new Error('Invalid search limit');

    const cacheKey = `${query.trim().toLowerCase()}:${limit}`;
    const cached = this.searchCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < 15 * 60_000) {
      return cached.entries;
    }

    // The songs section excludes podcasts and other ordinary YouTube videos.
    // Do not fall back to unrestricted video search when no song is found.
    const url = `https://music.youtube.com/search?q=${encodeURIComponent(query.trim())}#songs`;
    const result = await this.json(['--flat-playlist', '--playlist-end', String(limit), url], 20_000, signal);
    signal?.throwIfAborted();
    const seen = new Set<string>();
    const candidates = (result.entries ?? (result.id ? [result] : [])).filter((entry) => {
      if (!entry?.id || !/^[\w-]{11}$/.test(entry.id) || !entry.title?.trim() || seen.has(entry.id)) return false;
      seen.add(entry.id);
      return true;
    }).slice(0, limit);

    if (candidates.length) {
      this.searchCache.set(cacheKey, { entries: candidates, timestamp: Date.now() });
      if (this.searchCache.size > 200) {
        const first = this.searchCache.keys().next().value;
        if (first) this.searchCache.delete(first);
      }
    }
    return candidates;
  }

  async related(videoId: string, limit = 25): Promise<Track[]> {
    if (!/^[\w-]{11}$/.test(videoId)) return [];
    if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error('Invalid recommendation limit');
    const result = await this.json([
      '--flat-playlist', '--playlist-items', `2:${Math.min(50, limit) + 1}`,
      `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`,
    ], 25_000);
    const seen = new Set([videoId]);
    const candidates = (result.entries ?? []).filter((entry) => {
      if (!entry?.id || !/^[\w-]{11}$/.test(entry.id) || !entry.title?.trim() || seen.has(entry.id)) return false;
      seen.add(entry.id);
      return !isNonSong(entry);
    }).slice(0, 50);
    // Flat radio entries lack categories. Verify them before queueing, with
    // bounded concurrency and a shared deadline so a radio cannot stall playback.
    const signal = AbortSignal.timeout(25_000);
    const songs: VideoInfo[] = [];
    for (let offset = 0; offset < candidates.length && songs.length < limit && !signal.aborted; offset += 4) {
      const batch = await Promise.all(candidates.slice(offset, offset + 4).map(async (entry) => {
        try {
          const info = await this.video(`https://www.youtube.com/watch?v=${entry.id}`, signal);
          return info.id === entry.id && isSong(info) ? info : null;
        } catch { return null; }
      }));
      songs.push(...batch.filter((info): info is VideoInfo => info !== null));
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
