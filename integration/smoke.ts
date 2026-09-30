import 'dotenv/config';
import { prepareYtdlp } from '../src/utils/ytdlp.js';
import { resolveQuery } from '../src/utils/resolve.js';
import { createYtdlpStream, requireFfmpeg } from '../src/utils/stream.js';
import { createAudioResource, StreamType, TransformerType } from '@discordjs/voice';
import { configureMusicEncoder } from '../src/utils/audio.js';

async function main(): Promise<void> {
  requireFfmpeg();
  const ytdlp = await prepareYtdlp();
  const candidates = await ytdlp.searchCandidates('Linkin Park Numb', 5);
  if (candidates.length < 2 || candidates.length > 5) throw new Error('Поиск не вернул несколько кандидатов.');
  if (new Set(candidates.map((candidate) => candidate.id)).size !== candidates.length) throw new Error('Поиск вернул дубликаты.');
  console.log(`Получено ${candidates.length} кандидатов поиска.`);
  const searched = await resolveQuery('Rick Astley Never Gonna Give You Up', 'Smoke', ytdlp);
  if (searched?.type !== 'single') throw new Error('Текстовый поиск не вернул видео.');
  console.log(`Поиск: ${searched.track.title}`);
  const result = await resolveQuery(process.env.SMOKE_VIDEO_URL || 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'Smoke', ytdlp);
  if (result?.type !== 'single') throw new Error('Не удалось получить метаданные видео.');
  console.log(`Видео: ${result.track.title}`);
  const media = createYtdlpStream(ytdlp, result.track.url);
  if (media.type !== StreamType.Raw) { media.destroy(); throw new Error('Ожидался нормализованный PCM.'); }
  const resource = createAudioResource(media.stream, { inputType: media.type, inlineVolume: true });
  configureMusicEncoder(resource);
  resource.volume!.setVolume(0.5);
  if (resource.edges.some(edge => edge.type === TransformerType.OpusDecoder)) {
    media.destroy(); resource.playStream.destroy(); throw new Error('Обнаружена лишняя перекодировка Opus.');
  }
  let packets = 0;
  const bytes = await new Promise<number>((resolve, reject) => {
    let total = 0;
    const timer = setTimeout(() => { media.destroy(); resource.playStream.destroy(); resolve(total); }, 10_000);
    media.stream.on('data', (chunk: Buffer) => { total += chunk.length; });
    const fail = (error: Error): void => { clearTimeout(timer); media.destroy(); resource.playStream.destroy(); reject(error); };
    media.stream.on('error', fail);
    resource.playStream.on('data', () => { packets++; });
    resource.playStream.on('error', fail);
    resource.playStream.on('end', () => { clearTimeout(timer); media.destroy(); resolve(total); });
  });
  if (bytes === 0 || packets === 0) throw new Error('Аудиопоток пуст.');
  console.log(`Получено ${bytes} байт нормализованного PCM и ${packets} пакетов Opus после регулировки громкости.`);
  const related = await ytdlp.related(result.track.videoId, 5);
  if (!related.length) throw new Error('Не получены рекомендации.');
  console.log(`Получено ${related.length} рекомендаций.`);
}

main().catch((error) => {
  console.error('Интеграционная проверка не пройдена. Проверьте сеть, YouTube, Python, yt-dlp и FFmpeg.', error);
  process.exitCode = 1;
});
