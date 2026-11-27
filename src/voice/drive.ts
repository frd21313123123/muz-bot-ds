import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { speechWav } from './groq.js';

export const VOICE_DRIVE_FOLDER = '1uPQIM_CO3MTFmvSFV-Y93stxlNuLs3JP';
const execFileAsync = promisify(execFile);

interface DriveOptions {
  enabled?: boolean;
  folderId?: string;
  remote?: string;
  rclonePath?: string;
  directory?: string;
  runRclone?: (executable: string, args: string[]) => Promise<void>;
  retryMs?: number;
}

async function runRclone(executable: string, args: string[]): Promise<void> {
  await execFileAsync(executable, args, { timeout: 20_000, windowsHide: true });
}

// Persist first; network failures and bot restarts must not discard recordings.
export class VoiceDriveArchive {
  readonly enabled: boolean;
  readonly configured: boolean;
  private readonly folderId: string;
  private readonly directory: string;
  private readonly remote: string;
  private readonly rclonePath: string;
  private writes = Promise.resolve();
  private pendingWrites = 0;
  private upload: Promise<void> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;

  constructor(private readonly options: DriveOptions = {}) {
    this.enabled = options.enabled ?? process.env.VOICE_AUDIO_DRIVE === '1';
    this.folderId = options.folderId ?? process.env.GOOGLE_DRIVE_FOLDER_ID?.trim() ?? VOICE_DRIVE_FOLDER;
    this.directory = options.directory ?? path.resolve('.runtime/voice-drive', this.folderId);
    this.remote = options.remote ?? process.env.VOICE_AUDIO_RCLONE_REMOTE?.trim() ?? '';
    this.rclonePath = options.rclonePath ?? (process.env.RCLONE_PATH?.trim() || 'rclone');
    this.configured = this.remote !== '';
    if (!/^[\w-]+$/.test(this.folderId)) throw new Error('Invalid GOOGLE_DRIVE_FOLDER_ID');
    if (this.configured && !/^[\w][\w-]*$/.test(this.remote)) {
      throw new Error('Invalid VOICE_AUDIO_RCLONE_REMOTE: expected a remote name without a colon');
    }
  }

  start(): void { if (this.enabled && !this.stopped) this.pump(); }

  append(pcm: Buffer): void {
    if (!this.enabled || this.stopped) return;
    if (this.pendingWrites >= 32) { console.error('[Drive] Очередь сохранения аудио заполнена.'); return; }
    let wav: Buffer;
    try { wav = speechWav(pcm); }
    catch { console.error('[Drive] Некорректный размер аудио.'); return; }
    const name = `command-${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID()}.wav`;
    this.pendingWrites++;
    this.writes = this.writes.then(async () => {
      await mkdir(this.directory, { recursive: true });
      const file = path.join(this.directory, name);
      await writeFile(`${file}.tmp`, wav, { mode: 0o600 });
      await rename(`${file}.tmp`, file);
    }).catch(() => console.error('[Drive] Не удалось сохранить аудиофайл локально.'))
      .finally(() => { this.pendingWrites--; if (!this.stopped) this.pump(); });
  }

  async flush(): Promise<void> { await this.writes; }

  async close(): Promise<void> {
    this.stopped = true;
    if (this.retry) clearTimeout(this.retry);
    this.retry = null;
    await this.flush();
    await this.upload;
  }

  private pump(): void {
    if (this.stopped || this.upload || this.retry || !this.configured) return;
    this.upload = this.drain().catch((error: NodeJS.ErrnoException) => {
      const reason = error.code === 'ENOENT' ? ' Исполняемый файл rclone не найден.' : '';
      console.error(`[Drive] Загрузка через rclone не удалась; WAV сохранён локально, повтор через минуту.${reason}`);
      if (!this.stopped) {
        this.retry = setTimeout(() => { this.retry = null; this.pump(); }, this.options.retryMs ?? 60_000);
        this.retry.unref();
      }
    }).finally(() => {
      this.upload = null;
      // Poll also covers a write that completed while the directory was listed.
      if (!this.stopped && !this.retry) {
        this.retry = setTimeout(() => { this.retry = null; this.pump(); }, 1000);
        this.retry.unref();
      }
    });
  }

  private async drain(): Promise<void> {
    const names = await readdir(this.directory).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return [];
      throw error;
    });
    for (const name of names.sort()) {
      if (this.stopped) return;
      if (!/^command-[\w-]+\.wav$/.test(name)) continue;
      const file = path.join(this.directory, name);
      await (this.options.runRclone ?? runRclone)(this.rclonePath, [
        'copyto', path.resolve(file), `${this.remote}:${name}`,
        '--drive-root-folder-id', this.folderId, '--checksum', '--retries', '1', '--dry-run=false',
      ]);
      await unlink(file);
    }
  }
}
