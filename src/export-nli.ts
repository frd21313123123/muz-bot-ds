import path from 'node:path';
import { exportTraining } from './voice/export-training.js';

const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error('Usage: npm run export:nli -- reviewed.jsonl output.jsonl');
  process.exitCode = 1;
} else {
  exportTraining(path.resolve('.runtime/nli'), path.resolve(args[0]!), path.resolve(args[1]!))
    .then(count => console.log(`Exported ${count} manually reviewed NLI examples.`))
    .catch(() => { console.error('NLI export failed: check reviews, source logs and a new output path.'); process.exitCode = 1; });
}
