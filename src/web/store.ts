import { createHash, randomInt, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { WebTarget } from './types.js';

export type SpeakerKind = 'alice' | 'bridge';

export interface Speaker {
  id: string;
  name: string;
  kind: SpeakerKind;
  guildId: string;
  channelId: string;
  deviceId?: string;
  tokenHash?: string;
  createdAt: number;
  lastSeenAt: number | null;
}

export interface PairingCode {
  code: string;
  name: string;
  guildId: string;
  channelId: string;
  expiresAt: number;
}

interface PersistedState {
  version: 1;
  speakers: Speaker[];
  lastTarget: WebTarget | null;
}

const CODE_TTL_MS = 10 * 60_000;
const MAX_CODES = 50;
const MAX_SPEAKERS = 50;
const ALICE_ONLINE_MS = 5 * 60_000;
const BRIDGE_ONLINE_MS = 2 * 60_000;

const isSnowflake = (value: unknown): value is string =>
  typeof value === 'string' && /^\d{5,25}$/u.test(value);
const isObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

export function validateSpeakerName(value: unknown): string {
  if (typeof value !== 'string') throw new Error('Название колонки обязательно.');
  const name = value.replace(/[\u0000-\u001f\u007f]/gu, ' ').trim().replace(/\s+/gu, ' ');
  if (name.length < 1 || name.length > 64) throw new Error('Название колонки должно быть от 1 до 64 символов.');
  return name;
}

export function hashToken(token: string): string {
  return createHash('sha256').update(token, 'utf8').digest('hex');
}

function isSpeaker(value: unknown): value is Speaker {
  if (!isObject(value)) return false;
  const kind = value.kind;
  if (kind !== 'alice' && kind !== 'bridge') return false;
  if (typeof value.id !== 'string' || !/^[0-9a-f-]{36}$/u.test(value.id)) return false;
  if (typeof value.name !== 'string' || !value.name.trim() || value.name.length > 64) return false;
  if (!isSnowflake(value.guildId) || !isSnowflake(value.channelId)) return false;
  if (typeof value.createdAt !== 'number' || !Number.isFinite(value.createdAt)) return false;
  if (value.deviceId !== undefined && (typeof value.deviceId !== 'string' || value.deviceId.length > 200)) return false;
  if (value.tokenHash !== undefined && (typeof value.tokenHash !== 'string' || !/^[0-9a-f]{64}$/u.test(value.tokenHash))) return false;
  if (value.lastSeenAt !== null && (typeof value.lastSeenAt !== 'number' || !Number.isFinite(value.lastSeenAt))) return false;
  const speaker = value as Partial<Speaker>;
  return kind === 'alice' ? typeof speaker.deviceId === 'string' && !!speaker.deviceId : typeof speaker.tokenHash === 'string';
}

export class SpeakerStore {
  private speakers: Speaker[] = [];
  private lastTarget: WebTarget | null = null;
  private codes = new Map<string, PairingCode>();
  private writes: Promise<void> = Promise.resolve();

  constructor(private readonly file = path.resolve('.runtime/web/state.json')) {}

  async load(): Promise<void> {
    try {
      const data: unknown = JSON.parse(await readFile(this.file, 'utf8'));
      if (!isObject(data) || data.version !== 1) throw new Error('Invalid web state');
      const speakers = Array.isArray(data.speakers) ? data.speakers.filter(isSpeaker) : [];
      this.speakers = speakers.slice(0, MAX_SPEAKERS);
      const target = data.lastTarget;
      this.lastTarget = isObject(target) && isSnowflake(target.guildId) && isSnowflake(target.channelId)
        ? { guildId: target.guildId, channelId: target.channelId } : null;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  list(): readonly Speaker[] { return this.speakers; }

  get(id: string): Speaker | undefined {
    return this.speakers.find(speaker => speaker.id === id);
  }

  findAliveAlice(deviceId: string): Speaker | undefined {
    return this.speakers.find(speaker => speaker.kind === 'alice' && speaker.deviceId === deviceId);
  }

  findByToken(token: string): Speaker | undefined {
    const hash = hashToken(token);
    return this.speakers.find(speaker => speaker.kind === 'bridge' && speaker.tokenHash === hash);
  }

  last(): WebTarget | null { return this.lastTarget; }

  async setLastTarget(target: WebTarget): Promise<void> {
    if (!isSnowflake(target.guildId) || !isSnowflake(target.channelId)) throw new Error('Некорректный голосовой канал.');
    this.lastTarget = { guildId: target.guildId, channelId: target.channelId };
    await this.save();
  }

  createCode(target: WebTarget, name: string, now = Date.now(), ttlMs = CODE_TTL_MS): PairingCode {
    if (!isSnowflake(target.guildId) || !isSnowflake(target.channelId)) throw new Error('Некорректный голосовой канал.');
    const speakerName = validateSpeakerName(name);
    for (const [code, item] of this.codes) if (item.expiresAt <= now) this.codes.delete(code);
    while (this.codes.size >= MAX_CODES) {
      const oldest = [...this.codes.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt)[0];
      if (!oldest) break;
      this.codes.delete(oldest[0]);
    }
    let code = '';
    do { code = String(randomInt(100_000, 1_000_000)); } while (this.codes.has(code));
    const item: PairingCode = { code, name: speakerName, guildId: target.guildId, channelId: target.channelId, expiresAt: now + ttlMs };
    this.codes.set(code, item);
    return item;
  }

  takeCode(code: string, now = Date.now()): PairingCode | null {
    const item = this.codes.get(code);
    if (!item) return null;
    this.codes.delete(code);
    return item.expiresAt > now ? item : null;
  }

  async pairAlice(deviceId: string, code: PairingCode, name?: string): Promise<Speaker> {
    if (!deviceId || deviceId.length > 200) throw new Error('Не удалось определить устройство Алисы.');
    const speakerName = validateSpeakerName(name ?? code.name);
    const existing = this.findAliveAlice(deviceId);
    if (existing) {
      existing.name = speakerName;
      existing.guildId = code.guildId;
      existing.channelId = code.channelId;
      existing.lastSeenAt = Date.now();
      await this.save();
      return existing;
    }
    const speaker: Speaker = {
      id: randomUUID(), name: speakerName, kind: 'alice', guildId: code.guildId,
      channelId: code.channelId, deviceId, createdAt: Date.now(), lastSeenAt: Date.now(),
    };
    this.speakers.push(speaker);
    this.trimSpeakers();
    await this.save();
    return speaker;
  }

  async pairBridge(token: string, code: PairingCode, name: string): Promise<Speaker> {
    const speaker: Speaker = {
      id: randomUUID(), name: validateSpeakerName(name), kind: 'bridge', guildId: code.guildId,
      channelId: code.channelId, tokenHash: hashToken(token), createdAt: Date.now(), lastSeenAt: Date.now(),
    };
    this.speakers.push(speaker);
    this.trimSpeakers();
    await this.save();
    return speaker;
  }

  touch(id: string): void {
    const speaker = this.get(id);
    if (speaker) speaker.lastSeenAt = Date.now();
  }

  async rename(id: string, name: string): Promise<Speaker> {
    const speaker = this.get(id);
    if (!speaker) throw new Error('Колонка не найдена.');
    speaker.name = validateSpeakerName(name);
    await this.save();
    return speaker;
  }

  async remove(id: string): Promise<boolean> {
    const index = this.speakers.findIndex(speaker => speaker.id === id);
    if (index < 0) return false;
    this.speakers.splice(index, 1);
    await this.save();
    return true;
  }

  online(speaker: Speaker, now = Date.now()): boolean {
    if (!speaker.lastSeenAt) return false;
    return now - speaker.lastSeenAt <= (speaker.kind === 'alice' ? ALICE_ONLINE_MS : BRIDGE_ONLINE_MS);
  }

  private trimSpeakers(): void {
    if (this.speakers.length > MAX_SPEAKERS) this.speakers.splice(0, this.speakers.length - MAX_SPEAKERS);
  }

  private save(): Promise<void> {
    const data: PersistedState = {
      version: 1,
      speakers: this.speakers,
      lastTarget: this.lastTarget,
    };
    const payload = `${JSON.stringify(data, null, 2)}\n`;
    const operation = this.writes.then(async () => {
      await mkdir(path.dirname(this.file), { recursive: true });
      const temporary = `${this.file}.tmp`;
      await writeFile(temporary, payload, { encoding: 'utf8', mode: 0o600 });
      await rename(temporary, this.file);
    });
    this.writes = operation.catch(() => {});
    return operation;
  }
}
