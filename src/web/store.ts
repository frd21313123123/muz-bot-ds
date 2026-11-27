import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { WebTarget } from './types.js';

interface PersistedState {
  version: 2;
  lastTarget: WebTarget | null;
}

const isSnowflake = (value: unknown): value is string =>
  typeof value === 'string' && /^\d{5,25}$/u.test(value);
const isObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

/** Stores the panel's last playback destination across restarts. */
export class WebStateStore {
  private lastTarget: WebTarget | null = null;
  private writes: Promise<void> = Promise.resolve();

  constructor(private readonly file = path.resolve('.runtime/web/state.json')) {}

  async load(): Promise<void> {
    let data: unknown;
    try {
      data = JSON.parse(await readFile(this.file, 'utf8'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return;
      throw error;
    }
    if (!isObject(data) || (data.version !== 1 && data.version !== 2)) throw new Error('Invalid web state');
    const target = data.lastTarget;
    this.lastTarget = isObject(target) && isSnowflake(target.guildId) && isSnowflake(target.channelId)
      ? { guildId: target.guildId, channelId: target.channelId } : null;
    // Upgrade the old format while discarding fields no longer used by the panel.
    if (data.version === 1) await this.save();
  }

  last(): WebTarget | null { return this.lastTarget; }

  async setLastTarget(target: WebTarget): Promise<void> {
    if (!isSnowflake(target.guildId) || !isSnowflake(target.channelId)) throw new Error('Некорректный голосовой канал.');
    this.lastTarget = { guildId: target.guildId, channelId: target.channelId };
    await this.save();
  }

  private save(): Promise<void> {
    const data: PersistedState = { version: 2, lastTarget: this.lastTarget };
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
