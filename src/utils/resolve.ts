import type { YoutubeTrack } from '../types.js';
import { toSoundCloudTrack, toTrack, type VideoInfo } from './ytdlp.js';
import type { MetadataInput } from './MetadataRequests.js';

export interface MetadataClient {
  video(url: string, options?: MetadataInput): Promise<VideoInfo>;
  playlist(url: string, limit: number, options?: MetadataInput): Promise<VideoInfo>;
  search(query: string, options?: MetadataInput): Promise<VideoInfo | null>;
  searchSoundCloud?(query: string, options?: MetadataInput): Promise<VideoInfo | null>;
}

export type SearchSource = 'youtube' | 'soundcloud';

export type ResolveResult = { type: 'single'; track: YoutubeTrack }
  | { type: 'playlist'; name: string; tracks: YoutubeTrack[] } | null;

const HOSTS = new Set(['youtube.com', 'www.youtube.com', 'music.youtube.com', 'm.youtube.com', 'youtu.be']);
const SOUNDCLOUD_HOSTS = new Set(['soundcloud.com', 'www.soundcloud.com', 'm.soundcloud.com']);
const SOUNDCLOUD_SHORT_HOST = 'on.soundcloud.com';
// Second path segments that name a list of tracks rather than a single track.
const SOUNDCLOUD_LISTS = new Set(['tracks', 'sets', 'albums', 'popular-tracks', 'likes', 'reposts']);

export type QueryTarget = { kind: 'video' | 'playlist' | 'search'; value: string; source?: 'soundcloud' };

function classifySoundCloud(url: URL): QueryTarget {
  if (url.hostname.toLowerCase() === SOUNDCLOUD_SHORT_HOST) {
    // A short link may lead to a track or a playlist; the resolver handles both.
    if (!/^\/[\w-]+\/?$/.test(url.pathname)) throw new Error('Некорректная ссылка SoundCloud.');
    return { kind: 'playlist', value: `https://${SOUNDCLOUD_SHORT_HOST}${url.pathname}`, source: 'soundcloud' };
  }
  const parts = url.pathname.split('/').filter(Boolean);
  if (!parts.length || parts.length > 4 || parts.some(part => !/^[^\s/]+$/.test(part))) {
    throw new Error('Некорректная ссылка SoundCloud.');
  }
  // The query string only carries tracking data; private links keep their secret in the path.
  const value = `https://soundcloud.com/${parts.join('/')}`;
  const list = parts.length === 1 || (parts.length === 2 && SOUNDCLOUD_LISTS.has(parts[1]!.toLowerCase()))
    || (parts.length >= 3 && parts[1]!.toLowerCase() === 'sets');
  return { kind: list ? 'playlist' : 'video', value, source: 'soundcloud' };
}

export function classifyQuery(query: string): QueryTarget {
  const trimmed = query.trim();
  let url: URL;
  try { url = new URL(trimmed); } catch { return { kind: 'search', value: trimmed }; }
  const host = url.hostname.toLowerCase();
  if (['http:', 'https:'].includes(url.protocol) && (SOUNDCLOUD_HOSTS.has(host) || host === SOUNDCLOUD_SHORT_HOST)) {
    return classifySoundCloud(url);
  }
  if (!['http:', 'https:'].includes(url.protocol) || !HOSTS.has(host)) {
    throw new Error('Поддерживаются только ссылки YouTube, YouTube Music и SoundCloud.');
  }
  const id = url.hostname.toLowerCase() === 'youtu.be'
    ? url.pathname.slice(1).split('/')[0]
    : url.searchParams.get('v');
  const list = url.searchParams.get('list');
  if (list === 'LM') {
    if (!id) {
      throw new Error('«Понравившаяся музыка» не содержит конкретного трека. Отправьте ссылку на песню с параметром v= или её название.');
    }
    if (!/^[\w-]{11}$/.test(id)) throw new Error('Некорректный ID видео.');
    return { kind: 'video', value: `https://www.youtube.com/watch?v=${id}` };
  }
  if (list?.startsWith('RD') && id) {
    if (!/^[\w-]{11}$/.test(id)) throw new Error('Некорректный ID видео.');
    return { kind: 'video', value: `https://www.youtube.com/watch?v=${id}` };
  }
  if (list) return { kind: 'playlist', value: `https://www.youtube.com/playlist?list=${encodeURIComponent(list)}` };
  if (!id || !/^[\w-]{11}$/.test(id)) throw new Error('Некорректная ссылка YouTube.');
  return { kind: 'video', value: `https://www.youtube.com/watch?v=${id}` };
}

export async function resolveQuery(query: string, requestedBy: string, metadata: MetadataClient,
  playlistLimit = 1, options: { fast?: boolean; signal?: AbortSignal; source?: SearchSource } = {}): Promise<ResolveResult> {
  options.signal?.throwIfAborted();
  const target = classifyQuery(query);
  const soundcloud = target.source === 'soundcloud';
  if (target.kind === 'video') {
    if (soundcloud) {
      return { type: 'single', track: toSoundCloudTrack(await metadata.video(target.value, options.signal), requestedBy) };
    }
    if (options.fast) {
      const id = new URL(target.value).searchParams.get('v');
      if (id && /^[\w-]{11}$/.test(id)) {
        return {
          type: 'single',
          track: {
            url: target.value,
            videoId: id,
            title: 'Загрузка…',
            duration: '?',
            thumbnail: `https://img.youtube.com/vi/${id}/hqdefault.jpg`,
            requestedBy,
            isAutoplay: false,
            isLive: false,
          },
        };
      }
    }
    return { type: 'single', track: toTrack(await metadata.video(target.value, options.signal), requestedBy) };
  }
  if (target.kind === 'playlist') {
    const limit = Math.max(1, Math.min(25, Math.floor(playlistLimit)));
    const info = await metadata.playlist(target.value, limit, options.signal);
    const convert = soundcloud ? toSoundCloudTrack : toTrack;
    // A short link can point at one track: yt-dlp then returns it without entries.
    if (soundcloud && !info.entries) return { type: 'single', track: toSoundCloudTrack(info, requestedBy) };
    const tracks = (info.entries ?? []).flatMap((entry) => {
      try { return [convert(entry, requestedBy)]; } catch { return []; }
    }).slice(0, limit);
    return tracks.length ? { type: 'playlist', name: info.title || 'Плейлист', tracks } : null;
  }
  if (!target.value) return null;
  if (options.source === 'soundcloud') {
    if (!metadata.searchSoundCloud) throw new Error('Поиск SoundCloud недоступен.');
    const found = await metadata.searchSoundCloud(target.value, options.signal);
    return found ? { type: 'single', track: toSoundCloudTrack(found, requestedBy) } : null;
  }
  const result = await metadata.search(target.value, options.signal);
  return result ? { type: 'single', track: toTrack(result, requestedBy) } : null;
}
