'use strict';

const play = require('play-dl');

/**
 * Нормализует URL YouTube Music → обычный YouTube.
 * @param {string} url
 */
function normalizeUrl(url) {
  return url.replace('music.youtube.com', 'www.youtube.com');
}

/**
 * Конвертирует объект YouTubeVideo в Track.
 * @param {object} v — video_details из play-dl
 * @param {string} requestedBy
 * @returns {import('./GuildQueue').Track}
 */
function videoToTrack(v, requestedBy) {
  // v.url может быть пустым (incomplete playlist) или music.youtube.com — строим надёжный URL
  const rawUrl = (v.url && v.url.length > 0)
    ? v.url
    : `https://www.youtube.com/watch?v=${v.id}`;
  const url = rawUrl.replace('music.youtube.com', 'www.youtube.com');

  return {
    url,
    videoId: v.id,
    title: v.title ?? 'Без названия',
    duration: v.durationRaw ?? '?',
    thumbnail: v.thumbnails?.[v.thumbnails.length - 1]?.url ?? null,
    requestedBy,
  };
}

/**
 * Разрешает пользовательский запрос (URL или поисковая строка) в треки.
 *
 * @param {string} query
 * @param {string} requestedBy
 * @returns {Promise<
 *   { type: 'single'; track: Track } |
 *   { type: 'playlist'; name: string; tracks: Track[] } |
 *   null
 * >}
 */
async function resolveQuery(query, requestedBy) {
  const normalized = normalizeUrl(query.trim());
  const validated  = play.yt_validate(normalized);
  let urlObj = null;
  try { urlObj = new URL(normalized); } catch (_) {}
  const listId = urlObj?.searchParams.get('list') || '';
  const videoIdFromUrl = urlObj?.searchParams.get('v') || '';

  // ── Одно видео ────────────────────────────────────────────────────────────
  if (validated === 'video') {
    const { video_details } = await play.video_basic_info(normalized);
    return { type: 'single', track: videoToTrack(video_details, requestedBy) };
  }

  // ── Плейлист или ссылка с параметром list= ───────────────────────────────
  if (validated === 'playlist') {
    // YouTube Music auto-mix (RD/RDAMVM) надёжнее воспроизводить как одиночное видео по ?v=
    if (listId.startsWith('RD') && videoIdFromUrl) {
      const { video_details } = await play.video_basic_info(
        `https://www.youtube.com/watch?v=${videoIdFromUrl}`,
      );
      return { type: 'single', track: videoToTrack(video_details, requestedBy) };
    }

    // Попытка загрузить плейлист
    try {
      const playlist = await play.playlist_info(normalized, { incomplete: true });
      const videos   = await playlist.all_videos();

      if (videos.length > 0) {
        return {
          type: 'playlist',
          name: playlist.title ?? 'Плейлист',
          tracks: videos.map(v => videoToTrack(v, requestedBy)),
        };
      }
    } catch (_) {
      // Плейлист недоступен (например, авто-микс RDAMVM), пробуем одиночное видео
    }

    // Попытка вытащить одиночное видео (параметр v=)
    try {
      const videoId = videoIdFromUrl;
      if (videoId) {
        const { video_details } = await play.video_basic_info(
          `https://www.youtube.com/watch?v=${videoId}`,
        );
        return { type: 'single', track: videoToTrack(video_details, requestedBy) };
      }
    } catch (_) {}

    return null;
  }

  // ── Текстовый поиск ───────────────────────────────────────────────────────
  const results = await play.search(query, { limit: 1, source: { youtube: 'video' } });
  if (!results.length) return null;
  return { type: 'single', track: videoToTrack(results[0], requestedBy) };
}

module.exports = { resolveQuery };
