import type { YoutubeTrack } from '../types.js';
import { toTrack, type VideoInfo } from './ytdlp.js';

export interface MetadataClient {
  video(url: string): Promise<VideoInfo>;
  playlist(url: string, limit: number): Promise<VideoInfo>;
  search(query: string): Promise<VideoInfo | null>;
}

export type ResolveResult = { type: 'single'; track: YoutubeTrack }
  | { type: 'playlist'; name: string; tracks: YoutubeTrack[] } | null;

const HOSTS = new Set(['youtube.com', 'www.youtube.com', 'music.youtube.com', 'm.youtube.com', 'youtu.be']);

export function classifyQuery(query: string): { kind: 'video' | 'playlist' | 'search'; value: string } {
  const trimmed = query.trim();
  let url: URL;
  try { url = new URL(trimmed); } catch { return { kind: 'search', value: trimmed }; }
  if (!['http:', 'https:'].includes(url.protocol) || !HOSTS.has(url.hostname.toLowerCase())) {
    throw new Error('Поддерживаются только ссылки YouTube и YouTube Music.');
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
  playlistLimit = 1, options: { fast?: boolean } = {}): Promise<ResolveResult> {
  const target = classifyQuery(query);
  if (target.kind === 'video') {
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
    return { type: 'single', track: toTrack(await metadata.video(target.value), requestedBy) };
  }
  if (target.kind === 'playlist') {
    const limit = Math.max(1, Math.min(25, Math.floor(playlistLimit)));
    const info = await metadata.playlist(target.value, limit);
    const tracks = (info.entries ?? []).flatMap((entry) => {
      try { return [toTrack(entry, requestedBy)]; } catch { return []; }
    }).slice(0, limit);
    return tracks.length ? { type: 'playlist', name: info.title || 'Плейлист', tracks } : null;
  }
  if (!target.value) return null;
  const result = await metadata.search(target.value);
  return result ? { type: 'single', track: toTrack(result, requestedBy) } : null;
}
