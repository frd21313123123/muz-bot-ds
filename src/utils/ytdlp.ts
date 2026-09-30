import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { promisify } from 'node:util';
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import type { Track } from '../types.js';

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
  constructor(private readonly runtime: Runtime) {}

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

  async video(url: string, signal?: AbortSignal): Promise<VideoInfo> {
    return this.json(['--no-playlist', url], 20_000, signal);
  }

  async playlist(url: string, limit: number): Promise<VideoInfo> {
    return this.json(['--flat-playlist', '--playlist-items', `1:${limit}`, url], 30_000);
  }

  async search(query: string): Promise<VideoInfo | null> {
    const result = await this.json(['--flat-playlist', `ytsearch1:${query}`]);
    return result.entries?.[0] ?? (result.id ? result : null);
  }

  async searchCandidates(query: string, limit = 5, signal?: AbortSignal): Promise<VideoInfo[]> {
    signal?.throwIfAborted();
    if (!query.trim()) return [];
    if (!Number.isInteger(limit) || limit < 1 || limit > 10) throw new Error('Invalid search limit');
    const result = await this.json(['--flat-playlist', `ytsearch${limit}:${query}`], 20_000, signal);
    signal?.throwIfAborted();
    const seen = new Set<string>();
    return (result.entries ?? (result.id ? [result] : [])).filter((entry) => {
      if (!entry?.id || !/^[\w-]{11}$/.test(entry.id) || !entry.title?.trim() || seen.has(entry.id)) return false;
      seen.add(entry.id);
      return true;
    }).slice(0, limit);
  }

  async related(videoId: string, limit = 25): Promise<Track[]> {
    if (!/^[\w-]{11}$/.test(videoId)) return [];
    const result = await this.json([
      '--flat-playlist', '--playlist-items', `2:${Math.min(50, limit) + 1}`,
      `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`,
    ], 25_000);
    const seen = new Set([videoId]);
    return (result.entries ?? []).flatMap((entry) => {
      if (!entry.id || !/^[\w-]{11}$/.test(entry.id) || seen.has(entry.id)) return [];
      seen.add(entry.id);
      return [toTrack(entry, '🤖 Бесконечное', true)];
    });
  }
}

export function toTrack(info: VideoInfo, requestedBy: string, isAutoplay = false): Track {
  if (!info.id || !/^[\w-]{11}$/.test(info.id)) {
    throw new Error('YouTube не вернул корректный идентификатор видео.');
  }
  const seconds = Number.isFinite(info.duration) ? Math.floor(info.duration ?? 0) : 0;
  const duration = info.duration_string || (seconds > 0
    ? `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}` : '?');
  return {
    url: `https://www.youtube.com/watch?v=${info.id}`,
    videoId: info.id,
    title: info.title?.trim() || 'Без названия',
    duration,
    thumbnail: info.thumbnail || `https://img.youtube.com/vi/${info.id}/hqdefault.jpg`,
    requestedBy,
    isAutoplay,
  };
}
