import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import type { VoiceDecision, VoiceIntent } from './intents.js';
import type { MusicPlayerState } from './music.js';
import type { VoiceDiagnostic } from './session.js';

export type TrainingIntent = VoiceIntent['action'] | 'play';
export const NLI_QUESTIONS = { intent: { type: 'choice',
  instructions: 'Определи одно действие музыкального бота. Непонятная, отрицательная или составная просьба: unknown.',
  criteria: { play: 'Найти и включить песню или исполнителя', skip: 'Пропустить трек, включить следующий',
    pause: 'Поставить на паузу', resume: 'Продолжить после паузы', stop: 'Остановить и отключиться',
    volume_set: 'Установить громкость в процентах', volume_up: 'Сделать громче',
    volume_down: 'Сделать тише', unknown: 'Другая, отрицательная или составная просьба' } } };

export interface TrainingExample {
  state: { phase: 'request'; message: string; canonical_message: string; selected_track: null; player: MusicPlayerState | null };
  route: 'control' | 'search' | 'video' | 'rejected';
  model_decision: VoiceDecision | null;
  validated_intent: VoiceIntent | { action: 'play' };
  query: string | null;
  outcome: 'changed' | 'no_op' | 'rejected' | 'cancelled' | 'failed';
  diagnostics: VoiceDiagnostic[];
}
export interface TrainingModels { stt: string | null; nli: string | null }

// Separate, explicitly enabled corpus. Predictions are never treated as gold.
export class VoiceTrainingLog {
  readonly enabled: boolean;
  private pending = Promise.resolve();
  private pendingCount = 0;
  private part = 0;
  private day = '';
  constructor(private readonly directory = path.resolve('.runtime/nli'),
    enabled = process.env.VOICE_NLI_LOG === '1', private readonly maxBytes = 10 * 1024 * 1024) {
    this.enabled = enabled;
  }
  append(example: TrainingExample, models: TrainingModels): void {
    if (!this.enabled) return;
    if (this.pendingCount >= 128) { console.error('[NLI] Training log queue is full.'); return; }
    const timestamp = new Date().toISOString();
    // Serialize now so later cancellation or caller mutation cannot alter a row.
    const line = JSON.stringify({ schema_version: 1, event_id: randomUUID(), recorded_at: timestamp,
      ...example, models, questions: NLI_QUESTIONS, gold: null,
      suggested_intent: example.validated_intent.action, label_status: 'unreviewed' }) + '\n';
    const size = Buffer.byteLength(line);
    if (size > this.maxBytes) { console.error('[NLI] Training example exceeds the file limit.'); return; }
    this.pendingCount++;
    this.pending = this.pending.then(async () => {
      await mkdir(this.directory, { recursive: true });
      const day = timestamp.slice(0, 10);
      if (this.day !== day) { this.day = day; this.part = 0; }
      let file: string;
      for (;;) {
        file = path.join(this.directory, `commands-${day}-${this.part}.jsonl`);
        const current = await stat(file).catch((error: NodeJS.ErrnoException) => {
          if (error.code === 'ENOENT') return null;
          throw error;
        });
        if ((current?.size ?? 0) + size <= this.maxBytes) break;
        this.part++;
      }
      await appendFile(file, line, { encoding: 'utf8', mode: 0o600 });
    }).catch(() => { console.error('[NLI] Could not write a training example.'); })
      .finally(() => { this.pendingCount--; });
  }
  async flush(): Promise<void> { await this.pending; }
}
