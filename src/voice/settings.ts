import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { validateWakeName } from './intents.js';

export class VoiceSettings {
  private names = new Map<string, string>();
  private writes: Promise<void> = Promise.resolve();
  constructor(private readonly file = path.resolve('.runtime/voice-settings.json')) {}

  async load(): Promise<void> {
    try {
      const data: unknown = JSON.parse(await readFile(this.file, 'utf8'));
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid voice settings');
      const names = new Map<string, string>();
      for (const [guild, name] of Object.entries(data)) {
        if (typeof name !== 'string') throw new Error('Invalid voice name');
        names.set(guild, validateWakeName(name));
      }
      this.names = names;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  get(guildId: string): string | undefined { return this.names.get(guildId); }

  async set(guildId: string, name: string | null): Promise<void> {
    const validated = name === null ? null : validateWakeName(name);
    const operation = this.writes.then(async () => {
      const next = new Map(this.names);
      if (validated === null) next.delete(guildId); else next.set(guildId, validated);
      await mkdir(path.dirname(this.file), { recursive: true });
      const temporary = `${this.file}.tmp`;
      await writeFile(temporary, JSON.stringify(Object.fromEntries(next), null, 2), 'utf8');
      await rename(temporary, this.file);
      this.names = next;
    });
    this.writes = operation.catch(() => {});
    return operation;
  }
}
