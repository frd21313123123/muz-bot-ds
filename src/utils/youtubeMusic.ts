import type { VideoInfo } from './ytdlp.js';
import { parseDuration } from './duration.js';

// Same WEB_REMIX endpoints used by the YouTube Music web player and ytmusicapi.
// Protocol references: https://github.com/sigma67/ytmusicapi/tree/main/ytmusicapi/mixins
const ORIGIN = 'https://music.youtube.com';
export type MusicEndpoint = 'search' | 'next';
type JsonObject = Record<string, unknown>;

function object(value: unknown): JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {};
}
function at(value: unknown, ...keys: (string | number)[]): unknown {
  for (const key of keys) value = typeof key === 'number'
    ? (Array.isArray(value) ? value[key] : undefined) : object(value)[key];
  return value;
}
function array(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function text(value: unknown): string {
  const simple = object(value).simpleText;
  return (typeof simple === 'string' ? simple : array(object(value).runs)
    .map(run => object(run).text).filter((part): part is string => typeof part === 'string').join('')).trim();
}
function renderers(value: unknown, name: string): JsonObject[] {
  if (Array.isArray(value)) return value.flatMap(item => renderers(item, name));
  const record = object(value);
  if (record[name]) return [object(record[name])];
  return Object.values(record).flatMap(item => renderers(item, name));
}
function artists(runs: unknown): string[] {
  return array(runs).flatMap(run => {
    const endpoint = at(run, 'navigationEndpoint', 'browseEndpoint');
    const pageType = at(endpoint, 'browseEndpointContextSupportedConfigs', 'browseEndpointContextMusicConfig', 'pageType');
    const name = object(run).text;
    return pageType === 'MUSIC_PAGE_TYPE_ARTIST' && typeof name === 'string' && name.trim() ? [name.trim()] : [];
  });
}
function trackInfo(id: unknown, title: string, duration: string, names: string[], thumbnails: unknown): VideoInfo | null {
  if (typeof id !== 'string' || !/^[\w-]{11}$/.test(id) || !title) return null;
  const images = array(thumbnails);
  const thumbnail = object(images.at(-1)).url;
  return { id, title, track: title, artists: names, artist: names.join(', '), categories: ['Music'],
    duration: parseDuration(duration) ?? undefined,
    thumbnail: typeof thumbnail === 'string' ? thumbnail : undefined,
    webpage_url: `${ORIGIN}/watch?v=${id}` };
}

function searchTrack(watch: unknown, title: string, runs: unknown[], thumbnails: unknown): VideoInfo[] {
  const type = at(watch, 'watchEndpointMusicSupportedConfigs', 'watchEndpointMusicConfig', 'musicVideoType');
  if (!['MUSIC_VIDEO_TYPE_ATV', 'MUSIC_VIDEO_TYPE_OMV', 'MUSIC_VIDEO_TYPE_UGC'].includes(String(type))) return [];
  const duration = runs.map(run => object(run).text)
    .find(part => typeof part === 'string' && /^\d+(?::\d{2}){1,2}$/.test(part.trim()));
  const info = trackInfo(object(watch).videoId, title, typeof duration === 'string' ? duration : '', artists(runs), thumbnails);
  return info ? [info] : [];
}

function searchContents(value: unknown): VideoInfo[] {
  if (Array.isArray(value)) return value.flatMap(searchContents);
  const item = object(value);
  if (item.musicCardShelfRenderer) {
    const card = object(item.musicCardShelfRenderer);
    const watch = at(card, 'onTap', 'watchEndpoint') ?? at(card, 'title', 'runs', 0, 'navigationEndpoint', 'watchEndpoint');
    return [...searchTrack(watch, text(card.title), array(at(card, 'subtitle', 'runs')),
      at(card, 'thumbnail', 'musicThumbnailRenderer', 'thumbnail', 'thumbnails')), ...searchContents(card.contents)];
  }
  if (item.musicShelfRenderer) return searchContents(object(item.musicShelfRenderer).contents);
  if (item.itemSectionRenderer) return searchContents(object(item.itemSectionRenderer).contents);
  if (item.musicResponsiveListItemRenderer) {
    const row = object(item.musicResponsiveListItemRenderer);
    if (row.musicItemRendererDisplayPolicy === 'MUSIC_ITEM_RENDERER_DISPLAY_POLICY_GREY_OUT' || row.isPlayable === false) return [];
    const columns = array(row.flexColumns).map(column => at(column, 'musicResponsiveListItemFlexColumnRenderer', 'text'));
    const watch = at(row, 'overlay', 'musicItemThumbnailOverlayRenderer', 'content', 'musicPlayButtonRenderer',
      'playNavigationEndpoint', 'watchEndpoint') ?? at(columns[0], 'runs', 0, 'navigationEndpoint', 'watchEndpoint')
      ?? at(row, 'navigationEndpoint', 'watchEndpoint');
    return searchTrack(watch, text(columns[0]), columns.flatMap(column => array(object(column).runs)),
      at(row, 'thumbnail', 'musicThumbnailRenderer', 'thumbnail', 'thumbnails'));
  }
  return Object.values(item).flatMap(searchContents);
}

export function parseMusicSearch(response: unknown): VideoInfo[] {
  // The main Music search can place an official music video in its top-result
  // card even when the song is absent from the audio-only songs filter. Keep
  // that card and subsequent musical rows in the exact order of the service.
  return searchContents(at(response, 'contents'));
}

export function parseMusicRecommendations(response: unknown): VideoInfo[] {
  // A wrapper also contains a video counterpart. Keep only its primary song so
  // a second version of the same recording cannot become the next suggestion.
  const panels = renderers(at(response, 'contents'), 'playlistPanelRenderer');
  return panels.flatMap(panel => array(panel.contents).flatMap(item => {
    const primary = at(item, 'playlistPanelVideoWrapperRenderer', 'primaryRenderer') ?? item;
    const row = object(at(primary, 'playlistPanelVideoRenderer'));
    if (row.unplayableText || row.isPlayable === false) return [];
    const type = at(row, 'navigationEndpoint', 'watchEndpoint', 'watchEndpointMusicSupportedConfigs',
      'watchEndpointMusicConfig', 'musicVideoType');
    if (!['MUSIC_VIDEO_TYPE_ATV', 'MUSIC_VIDEO_TYPE_OMV', 'MUSIC_VIDEO_TYPE_UGC'].includes(String(type))) return [];
    const info = trackInfo(row.videoId, text(row.title), text(row.lengthText), artists(at(row, 'longBylineText', 'runs')),
      at(row, 'thumbnail', 'thumbnails'));
    return info ? [info] : [];
  }));
}

export async function youtubeMusicJson(endpoint: MusicEndpoint, payload: JsonObject, signal?: AbortSignal): Promise<VideoInfo> {
  const requestSignal = AbortSignal.any([AbortSignal.timeout(20_000), ...(signal ? [signal] : [])]);
  requestSignal.throwIfAborted();
  // ytmusicapi also derives this public client version from the current UTC day.
  const version = `1.${new Date().toISOString().slice(0, 10).replaceAll('-', '')}.01.00`;
  const response = await fetch(`${ORIGIN}/youtubei/v1/${endpoint}?prettyPrint=false`, {
    method: 'POST', signal: requestSignal,
    headers: { 'Content-Type': 'application/json', Origin: ORIGIN,
      'X-Youtube-Client-Name': '67', 'X-Youtube-Client-Version': version },
    body: JSON.stringify({ ...payload, context: { client: { clientName: 'WEB_REMIX', clientVersion: version, hl: 'ru', gl: 'US' } } }),
  });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`YouTube Music: HTTP ${response.status}`);
  }
  const data: unknown = await response.json();
  requestSignal.throwIfAborted();
  if (object(data).error) throw new Error('YouTube Music не вернул музыкальную выдачу.');
  return { entries: endpoint === 'search' ? parseMusicSearch(data) : parseMusicRecommendations(data) };
}
