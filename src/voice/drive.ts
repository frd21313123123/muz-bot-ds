import { randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { speechWav } from './groq.js';

export const VOICE_DRIVE_FOLDER = '1uPQIM_CO3MTFmvSFV-Y93stxlNuLs3JP';
interface DriveOptions {
  enabled?: boolean;
  folderId?: string;
  clientId?: string;
  clientSecret?: string;
  refreshToken?: string;
  directory?: string;
  fetch?: typeof fetch;
  retryMs?: number;
}

// Persist first; network failures and bot restarts must not discard recordings.
export class VoiceDriveArchive {
  readonly enabled: boolean;
  readonly configured: boolean;
  private readonly folderId: string;
  private readonly directory: string;
  private readonly clientId: string;
  private readonly clientSecret: string;
  private readonly refreshToken: string;
  private writes = Promise.resolve();
  private pendingWrites = 0;
  private upload: Promise<void> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private token = '';
  private expiresAt = 0;

  constructor(private readonly options: DriveOptions = {}) {
    this.enabled = options.enabled ?? process.env.VOICE_AUDIO_DRIVE === '1';
    this.folderId = options.folderId ?? process.env.GOOGLE_DRIVE_FOLDER_ID?.trim() ?? VOICE_DRIVE_FOLDER;
    this.directory = options.directory ?? path.resolve('.runtime/voice-drive', this.folderId);
    this.clientId = options.clientId ?? process.env.GOOGLE_DRIVE_CLIENT_ID?.trim() ?? '';
    this.clientSecret = options.clientSecret ?? process.env.GOOGLE_DRIVE_CLIENT_SECRET?.trim() ?? '';
    this.refreshToken = options.refreshToken ?? process.env.GOOGLE_DRIVE_REFRESH_TOKEN?.trim() ?? '';
    this.configured = Boolean(this.clientId && this.clientSecret && this.refreshToken);
    if (!/^[\w-]+$/.test(this.folderId)) throw new Error('Invalid GOOGLE_DRIVE_FOLDER_ID');
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
    this.upload = this.drain().catch(() => {
      console.error('[Drive] Загрузка не удалась; WAV сохранён локально, повтор через минуту.');
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

  private async accessToken(): Promise<string> {
    if (this.token && Date.now() < this.expiresAt) return this.token;
    const response = await (this.options.fetch ?? fetch)('https://oauth2.googleapis.com/token', {
      method: 'POST', body: new URLSearchParams({ grant_type: 'refresh_token',
        client_id: this.clientId, client_secret: this.clientSecret, refresh_token: this.refreshToken }),
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) { await response.body?.cancel(); throw new Error('Drive authorization failed'); }
    const value = await response.json() as { access_token?: unknown; expires_in?: unknown };
    if (typeof value.access_token !== 'string' || !value.access_token
      || typeof value.expires_in !== 'number' || !Number.isFinite(value.expires_in) || value.expires_in <= 0) {
      throw new Error('Invalid Drive token');
    }
    this.token = value.access_token;
    this.expiresAt = Date.now() + Math.max(0, value.expires_in - 60) * 1000;
    return this.token;
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
      const wav = await readFile(file);
      const token = await this.accessToken();
      const boundary = `voice_${randomUUID()}`;
      const metadata = JSON.stringify({ name, mimeType: 'audio/wav', parents: [this.folderId] });
      const body = Buffer.concat([
        Buffer.from(`--${boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n${metadata}\r\n`),
        Buffer.from(`--${boundary}\r\nContent-Type: audio/wav\r\n\r\n`), wav,
        Buffer.from(`\r\n--${boundary}--\r\n`),
      ]);
      const response = await (this.options.fetch ?? fetch)(
        'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&supportsAllDrives=true&fields=id', {
          method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': `multipart/related; boundary=${boundary}` },
          body: new Uint8Array(body), signal: AbortSignal.timeout(20_000),
        });
      if (!response.ok) {
        if (response.status === 401) { this.token = ''; this.expiresAt = 0; }
        await response.body?.cancel();
        throw new Error('Drive upload failed');
      }
      const result = await response.json() as { id?: unknown };
      if (typeof result.id !== 'string' || !result.id) throw new Error('Invalid Drive upload response');
      await unlink(file);
    }
  }
}
