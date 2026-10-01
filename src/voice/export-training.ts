import { readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { NLI_QUESTIONS } from './training.js';

const intents = new Set(Object.keys(NLI_QUESTIONS.intent.criteria));
function rows(text: string): Record<string, unknown>[] {
  return text.split(/\r?\n/u).filter(line => line.trim()).map(line => {
    const row: unknown = JSON.parse(line);
    if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error('Invalid JSONL row');
    return row as Record<string, unknown>;
  });
}

// Only manually reviewed labels enter an export; predictions/outcomes cannot
// accidentally become training truth, including on cancelled requests.
export async function exportTraining(directory: string, labelsFile: string, outputFile: string): Promise<number> {
  const labels = new Map<string, string>();
  for (const row of rows(await readFile(labelsFile, 'utf8'))) {
    if (typeof row.event_id !== 'string' || typeof row.intent !== 'string'
      || !intents.has(row.intent) || labels.has(row.event_id)) throw new Error('Invalid or duplicate review');
    labels.set(row.event_id, row.intent);
  }
  if (!labels.size) throw new Error('No reviewed labels');
  const exported: string[] = [];
  const found = new Set<string>();
  for (const name of (await readdir(directory)).sort()) {
    if (!/^commands-.*\.jsonl$/u.test(name)) continue;
    for (const row of rows(await readFile(path.join(directory, name), 'utf8'))) {
      if (typeof row.event_id !== 'string' || !labels.has(row.event_id)) continue;
      if (row.schema_version !== 1 || !row.state || typeof row.state !== 'object' || found.has(row.event_id)) {
        throw new Error('Invalid or duplicate source example');
      }
      found.add(row.event_id);
      // Keep preprocessing in the audit log, but do not give the new model a
      // corrected command as an answer hint when it should learn the raw text.
      const { canonical_message: _canonical, ...state } = row.state as Record<string, unknown>;
      exported.push(JSON.stringify({ state: JSON.stringify(state), questions: JSON.stringify(NLI_QUESTIONS),
        gold: JSON.stringify({ intent: labels.get(row.event_id) }), tags: ['voice', 'human-reviewed'],
        source_event_id: row.event_id }));
    }
  }
  if (found.size !== labels.size) throw new Error('Some reviewed events were not found');
  await writeFile(outputFile, exported.join('\n') + '\n', { encoding: 'utf8', flag: 'wx', mode: 0o600 });
  return exported.length;
}
