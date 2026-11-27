import type { LogEntryView } from './types.js';

const LIMIT = 60;

export class EventLog {
  private entries: LogEntryView[] = [];

  add(kind: string, text: string, ok = true): void {
    const clean = text.replace(/\s+/gu, ' ').trim().slice(0, 300);
    if (!clean) return;
    this.entries.unshift({ at: Date.now(), kind, text: clean, ok });
    if (this.entries.length > LIMIT) this.entries.length = LIMIT;
  }

  list(): LogEntryView[] {
    return this.entries.map(entry => ({ ...entry }));
  }
}
