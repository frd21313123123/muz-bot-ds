import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { VoiceTrainingLog, type TrainingExample } from '../src/voice/training.js';
import { exportTraining } from '../src/voice/export-training.js';

const example = (): TrainingExample => ({ state: { phase: 'request', message: 'Next\nпожалуйста',
  canonical_message: 'Следующий трек', selected_track: null, player: null }, route: 'control',
  model_decision: { action: 'skip', confidence: 0.99 }, validated_intent: { action: 'skip' },
  query: null, outcome: 'changed', diagnostics: [] });
const models = { stt: 'large-v3-turbo', nli: 'laya-muz-bot-ds-v3' };

test('training logs are opt-in, serialize concurrent commands safely and rotate without deleting data', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'muz-nli-'));
  try {
    const disabled = new VoiceTrainingLog(path.join(root, 'off'), false);
    disabled.append(example(), models); await disabled.flush();
    await assert.rejects(stat(path.join(root, 'off')));
    const log = new VoiceTrainingLog(root, true, 3500);
    const row = example();
    for (let i = 0; i < 6; i++) log.append(row, models);
    row.state.message = 'mutated'; await log.flush();
    const files = await readdir(root); assert.ok(files.length > 1);
    const stored = [];
    for (const file of files) {
      assert.ok((await stat(path.join(root, file))).size <= 3500);
      stored.push(...(await readFile(path.join(root, file), 'utf8')).trim().split('\n').map(line => JSON.parse(line)));
    }
    assert.equal(stored.length, 6); assert.equal(new Set(stored.map(row => row.event_id)).size, 6);
    for (const saved of stored) {
      assert.equal(saved.state.message, 'Next\nпожалуйста'); assert.equal(saved.gold, null);
      assert.equal(saved.label_status, 'unreviewed'); assert.equal(saved.suggested_intent, 'skip');
      assert.equal('user_id' in saved, false); assert.equal('audio' in saved, false);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('export requires human labels, applies corrections and never overwrites existing files', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'muz-nli-export-'));
  try {
    const log = new VoiceTrainingLog(root, true);
    log.append(example(), models); log.append(example(), models); await log.flush();
    const file = (await readdir(root))[0]!;
    const source = (await readFile(path.join(root, file), 'utf8')).trim().split('\n').map(line => JSON.parse(line));
    const labels = path.join(root, 'reviewed.jsonl'); const output = path.join(root, 'dataset.jsonl');
    await writeFile(labels, JSON.stringify({ event_id: source[0].event_id, intent: 'pause' }) + '\n');
    assert.equal(await exportTraining(root, labels, output), 1);
    const row = JSON.parse((await readFile(output, 'utf8')).trim());
    assert.deepEqual(JSON.parse(row.gold), { intent: 'pause' });
    assert.equal(JSON.parse(row.state).message, 'Next\nпожалуйста');
    assert.equal('canonical_message' in JSON.parse(row.state), false);
    assert.equal(row.source_event_id, source[0].event_id);
    await assert.rejects(exportTraining(root, labels, output));
    await writeFile(labels, JSON.stringify({ event_id: 'missing', intent: 'skip' }));
    await assert.rejects(exportTraining(root, labels, path.join(root, 'missing.jsonl')));
    await writeFile(labels, JSON.stringify({ event_id: source[0].event_id, intent: 'invalid' }));
    await assert.rejects(exportTraining(root, labels, path.join(root, 'invalid.jsonl')));
  } finally { await rm(root, { recursive: true, force: true }); }
});
