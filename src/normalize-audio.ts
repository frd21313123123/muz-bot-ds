import 'dotenv/config';
import { normalizeAudioFile } from './utils/normalizeAudio.js';
import { LOUDNESS_TARGET, TRUE_PEAK_TARGET } from './utils/audio.js';

const [input, output, ...extra] = process.argv.slice(2);
if (!input || !output || extra.length) {
  console.error('Использование: npm run normalize:audio -- "input.mp3" "output.flac"');
  process.exitCode = 1;
} else {
  normalizeAudioFile(input, output).then(() => {
    console.log(`✅ Нормализация завершена: ${LOUDNESS_TARGET} LUFS, максимум ${TRUE_PEAK_TARGET} dBTP. ${output}`);
  }).catch((error: unknown) => {
    console.error('Не удалось нормализовать файл:', error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
