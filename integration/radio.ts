import 'dotenv/config';
import { radioStations } from '../src/utils/radio.js';
import { createRadioStream, waitForAudio } from '../src/utils/stream.js';

// Check actual normalized audio, not only HTTP status or an HLS playlist.
let failures = 0;
for (const station of radioStations) {
  let playable = false;
  for (const url of station.streams) {
    const signal = AbortSignal.timeout(25_000);
    const media = createRadioStream(url);
    const abort = (): void => media.destroy();
    signal.addEventListener('abort', abort, { once: true });
    try {
      await waitForAudio(media, signal, 20_000);
      let bytes = 0;
      for await (const chunk of media.stream) {
        signal.throwIfAborted();
        bytes += (chunk as Buffer).length;
        if (bytes >= 48000 * 2 * 2 * 3) break;
      }
      if (bytes < 48000 * 2 * 2 * 3) throw new Error('Недостаточно аудиоданных.');
      console.log(`PASS ${station.name}: 3 секунды PCM, TLS проверен.`);
      playable = true; break;
    } catch { /* Try the next configured broadcaster endpoint. */ }
    finally { signal.removeEventListener('abort', abort); media.destroy(); }
  }
  if (!playable) { console.error(`FAIL ${station.name}: эфир недоступен.`); failures++; }
}
console.log(`Радио: ${radioStations.length - failures}/${radioStations.length} станций доступны.`);
if (failures) process.exitCode = 1;
