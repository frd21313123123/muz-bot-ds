'use strict';

const play = require('play-dl');
const { createYtdlpStream, destroyStream } = require('./src/utils/stream');
const { resolveQuery } = require('./src/utils/resolve');

const TEST_URL = 'https://music.youtube.com/watch?v=p90KmzZCAEs&list=RDAMVMU3MN_no4yh4';
const SINGLE_URL = 'https://music.youtube.com/watch?v=p90KmzZCAEs';

let passed = 0;
let failed = 0;

function ok(name) { passed++; console.log(`  ✅ ${name}`); }
function fail(name, err) { failed++; console.log(`  ❌ ${name}: ${err}`); }

async function test(name, fn) {
  try {
    await fn();
    ok(name);
  } catch (err) {
    fail(name, err.message ?? err);
  }
}

(async () => {
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 1. URL нормализация ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('music.youtube.com → www.youtube.com', async () => {
    const normalized = TEST_URL.replace('music.youtube.com', 'www.youtube.com');
    if (!normalized.includes('www.youtube.com')) throw 'Нормализация не сработала';
    if (normalized.includes('music.youtube.com')) throw 'music.youtube.com осталось';
  });

  await test('yt_validate распознаёт нормализованный URL', async () => {
    const normalized = TEST_URL.replace('music.youtube.com', 'www.youtube.com');
    const type = play.yt_validate(normalized);
    if (type !== 'video' && type !== 'playlist') {
      throw `Ожидал video или playlist, получил: ${type}`;
    }
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 2. resolveQuery — одиночное видео ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('resolveQuery одиночного YouTube Music URL', async () => {
    const result = await resolveQuery(SINGLE_URL, 'TestUser');
    if (!result) throw 'result === null';
    if (result.type !== 'single') throw `Ожидал type=single, получил: ${result.type}`;
    const t = result.track;
    if (!t.url) throw 'track.url пустой';
    if (!t.url.startsWith('https://www.youtube.com')) throw `URL не нормализован: ${t.url}`;
    if (!t.videoId) throw 'track.videoId пустой';
    if (!t.title) throw 'track.title пустой';
    if (t.requestedBy !== 'TestUser') throw `requestedBy: ${t.requestedBy}`;
    console.log(`         title: "${t.title}", duration: ${t.duration}`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 3. resolveQuery — плейлист (YouTube Music mix) ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('resolveQuery плейлиста / микса', async () => {
    const result = await resolveQuery(TEST_URL, 'TestUser');
    if (!result) throw 'result === null';

    if (result.type === 'playlist') {
      if (!Array.isArray(result.tracks)) throw 'tracks не массив';
      if (result.tracks.length === 0) throw 'Пустой плейлист';
      console.log(`         Плейлист: "${result.name}", треков: ${result.tracks.length}`);

      // Проверяем первые 3 трека
      for (const t of result.tracks.slice(0, 3)) {
        if (!t.url) throw `Трек "${t.title}" — пустой url`;
        if (!t.url.startsWith('https://www.youtube.com')) {
          throw `Трек "${t.title}" — URL не нормализован: ${t.url}`;
        }
        if (!t.videoId) throw `Трек "${t.title}" — пустой videoId`;
      }
    } else if (result.type === 'single') {
      // RDAMVM миксы могут fallback в single — это ок
      console.log(`         Fallback в single: "${result.track.title}"`);
      if (!result.track.url.startsWith('https://www.youtube.com')) {
        throw `URL не нормализован: ${result.track.url}`;
      }
    } else {
      throw `Неожиданный type: ${result.type}`;
    }
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 4. resolveQuery — текстовый поиск ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('resolveQuery текстового запроса', async () => {
    const result = await resolveQuery('Interstellar Hans Zimmer', 'TestUser');
    if (!result) throw 'result === null';
    if (result.type !== 'single') throw `Ожидал single, получил: ${result.type}`;
    if (!result.track.url.startsWith('https://')) throw `Плохой URL: ${result.track.url}`;
    console.log(`         Найдено: "${result.track.title}"`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 5. yt-dlp стриминг ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('yt-dlp стриминг одиночного видео (5 сек чтение)', async () => {
    const url = 'https://www.youtube.com/watch?v=p90KmzZCAEs';
    const { stream, type } = createYtdlpStream(url);

    // Читаем 5 секунд и проверяем что данные приходят
    const bytes = await new Promise((resolve, reject) => {
      let total = 0;
      const timeout = setTimeout(() => {
        destroyStream(stream);
        resolve(total);
      }, 5000);

      stream.on('data', (chunk) => { total += chunk.length; });
      stream.on('error', (err) => { clearTimeout(timeout); reject(err); });
      stream.on('end', () => { clearTimeout(timeout); resolve(total); });
    });

    if (bytes === 0) throw 'Получено 0 байт — стрим пустой';
    console.log(`         Получено ${(bytes / 1024).toFixed(1)} KB за 5 сек, type: ${type}`);
  });

  await test('yt-dlp стриминг первого трека из плейлиста', async () => {
    // Сначала резолвим плейлист
    const result = await resolveQuery(TEST_URL, 'TestUser');
    if (!result) throw 'resolveQuery вернул null';

    const trackUrl = result.type === 'playlist'
      ? result.tracks[0].url
      : result.track.url;
    const trackTitle = result.type === 'playlist'
      ? result.tracks[0].title
      : result.track.title;

    console.log(`         Стримим: "${trackTitle}" → ${trackUrl}`);

    const { stream } = createYtdlpStream(trackUrl);

    const bytes = await new Promise((resolve, reject) => {
      let total = 0;
      const timeout = setTimeout(() => {
        destroyStream(stream);
        resolve(total);
      }, 5000);
      stream.on('data', (chunk) => { total += chunk.length; });
      stream.on('error', (err) => { clearTimeout(timeout); reject(err); });
      stream.on('end', () => { clearTimeout(timeout); resolve(total); });
    });

    if (bytes === 0) throw 'Получено 0 байт';
    console.log(`         Получено ${(bytes / 1024).toFixed(1)} KB за 5 сек`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 6. createAudioResource совместимость ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('createAudioResource принимает yt-dlp поток', async () => {
    const { createAudioResource } = require('@discordjs/voice');
    const url = 'https://www.youtube.com/watch?v=p90KmzZCAEs';
    const { stream, type } = createYtdlpStream(url);

    const resource = createAudioResource(stream, {
      inputType: type,
      inlineVolume: true,
    });

    if (!resource) throw 'resource === null';
    if (!resource.volume) throw 'inlineVolume не работает';
    resource.volume.setVolume(0.5);

    // Дать 2 сек чтобы данные потекли
    await new Promise(r => setTimeout(r, 2000));
    destroyStream(stream);
    console.log(`         resource.playbackDuration: ${resource.playbackDuration}ms`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n═══ 7. Автовоспроизведение — yt-dlp Radio Mix ═══');
  // ─────────────────────────────────────────────────────────────────────────

  await test('yt-dlp Radio Mix возвращает рекомендации', async () => {
    const { execFile } = require('child_process');
    const videoId = 'p90KmzZCAEs';
    const mixUrl = `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`;

    const stdout = await new Promise((resolve, reject) => {
      execFile('yt-dlp', [
        '--flat-playlist',
        '--print', '%(id)s\t%(title)s\t%(duration_string)s',
        '--playlist-items', '2:6',
        '--quiet', '--no-warnings',
        mixUrl,
      ], { timeout: 15_000, windowsHide: true }, (err, out) => {
        if (err) return reject(err);
        resolve(out);
      });
    });

    const lines = stdout.trim().split('\n').filter(l => l.trim());
    if (lines.length === 0) throw 'yt-dlp не вернул рекомендации';

    console.log(`         Найдено ${lines.length} рекомендаций:`);
    for (const line of lines) {
      const [id, title, dur] = line.split('\t');
      console.log(`           - [${id}] ${title} (${dur})`);
    }
  });

  await test('Рекомендация из yt-dlp стримится', async () => {
    const { execFile } = require('child_process');
    const videoId = 'p90KmzZCAEs';
    const mixUrl = `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`;

    const stdout = await new Promise((resolve, reject) => {
      execFile('yt-dlp', [
        '--flat-playlist', '--print', '%(id)s',
        '--playlist-items', '2:2',
        '--quiet', '--no-warnings',
        mixUrl,
      ], { timeout: 15_000, windowsHide: true }, (err, out) => {
        if (err) return reject(err);
        resolve(out.trim());
      });
    });

    if (!stdout) throw 'Не получен ID рекомендации';

    const relatedUrl = `https://www.youtube.com/watch?v=${stdout}`;
    console.log(`         Стримим рекомендацию: ${relatedUrl}`);

    const { stream } = createYtdlpStream(relatedUrl);
    const bytes = await new Promise((resolve, reject) => {
      let total = 0;
      const timeout = setTimeout(() => { destroyStream(stream); resolve(total); }, 5000);
      stream.on('data', (chunk) => { total += chunk.length; });
      stream.on('error', (err) => { clearTimeout(timeout); reject(err); });
      stream.on('end', () => { clearTimeout(timeout); resolve(total); });
    });

    if (bytes === 0) throw 'Получено 0 байт';
    console.log(`         Получено ${(bytes / 1024).toFixed(1)} KB за 5 сек`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Итоги
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n════════════════════════════════════════');
  console.log(`  ИТОГО: ${passed} passed, ${failed} failed`);
  console.log('════════════════════════════════════════\n');

  process.exit(failed > 0 ? 1 : 0);
})();
