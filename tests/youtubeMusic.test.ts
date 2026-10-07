import assert from 'node:assert/strict';
import test from 'node:test';
import { parseMusicSearch, parseMusicRecommendations, youtubeMusicJson } from '../src/utils/youtubeMusic.js';

const runs = (value: string) => ({ runs: [{ text: value }] });
const artist = { text: 'Artist', navigationEndpoint: { browseEndpoint: {
  browseId: 'UCartist', browseEndpointContextSupportedConfigs: { browseEndpointContextMusicConfig: { pageType: 'MUSIC_PAGE_TYPE_ARTIST' } },
} } };
const watch = (videoId: string, type = 'MUSIC_VIDEO_TYPE_ATV') => ({ videoId,
  watchEndpointMusicSupportedConfigs: { watchEndpointMusicConfig: { musicVideoType: type } },
});
function searchRow(id: string, title: string, type?: string) {
  return { musicResponsiveListItemRenderer: {
    playlistItemData: { videoId: id },
    flexColumns: [
      { musicResponsiveListItemFlexColumnRenderer: { text: { runs: [{ text: title, navigationEndpoint: { watchEndpoint: watch(id, type) } }] } } },
      { musicResponsiveListItemFlexColumnRenderer: { text: { runs: [artist, { text: ' • ' }, { text: 'Album' }, { text: ' • ' }, { text: '3:08' }] } } },
    ],
    thumbnail: { musicThumbnailRenderer: { thumbnail: { thumbnails: [{ url: 'small' }, { url: 'large' }] } } },
  } };
}
function panelRow(id: string, title: string, type?: string) {
  return { playlistPanelVideoRenderer: { videoId: id, title: runs(title), lengthText: runs('3:37'),
    longBylineText: { runs: [artist, { text: ' • ' }, { text: 'Album' }] },
    navigationEndpoint: { watchEndpoint: watch(id, type) }, thumbnail: { thumbnails: [{ url: 'large' }] },
  } };
}
const searchResponse = (contents: unknown[]) => ({ contents: { tabbedSearchResultsRenderer: { tabs: [
  { tabRenderer: { content: { sectionListRenderer: { contents: [{ musicShelfRenderer: { contents } }] } } } },
] } } });
const nextResponse = (contents: unknown[]) => ({ contents: { singleColumnMusicWatchNextResultsRenderer: { tabbedRenderer: {
  watchNextTabbedResultsRenderer: { tabs: [{ tabRenderer: { content: { musicQueueRenderer: { content: { playlistPanelRenderer: { contents } } } } } }] },
} } } });

test('Music search parses songs in service order with artist, duration and artwork, excluding episodes and malformed rows', () => {
  const songs = parseMusicSearch(searchResponse([
    searchRow('aaaaaaaaaaa', 'Interview'), searchRow('bbbbbbbbbbb', 'Second'),
    searchRow('ccccccccccc', 'Episode', 'MUSIC_VIDEO_TYPE_PODCAST_EPISODE'),
    searchRow('bad', 'Invalid'), searchRow('ddddddddddd', ' '), { musicTwoRowItemRenderer: { title: runs('Album') } },
  ]));
  assert.deepEqual(songs.map(song => song.id), ['aaaaaaaaaaa', 'bbbbbbbbbbb']);
  assert.equal(songs[0]?.title, 'Interview', 'song titles must not be interpreted as spoken-content labels');
  assert.equal(songs[0]?.duration, 188);
  assert.deepEqual(songs[0]?.artists, ['Artist']);
  assert.equal(songs[0]?.thumbnail, 'large');
  assert.equal(songs[0]?.webpage_url, 'https://music.youtube.com/watch?v=aaaaaaaaaaa');
  assert.deepEqual(parseMusicSearch({}), []);
});

test('Music watch queue uses primary songs, skips counterpart videos, unavailable entries and podcasts', () => {
  const unavailable = panelRow('eeeeeeeeeee', 'Unavailable');
  Object.assign(unavailable.playlistPanelVideoRenderer, { unplayableText: runs('Unavailable') });
  const songs = parseMusicRecommendations(nextResponse([
    panelRow('aaaaaaaaaaa', 'Seed'),
    { playlistPanelVideoWrapperRenderer: {
      primaryRenderer: panelRow('bbbbbbbbbbb', 'First recommendation'),
      counterpart: [{ counterpartRenderer: panelRow('ccccccccccc', 'Video counterpart', 'MUSIC_VIDEO_TYPE_OMV') }],
    } },
    panelRow('ddddddddddd', 'Second recommendation', 'MUSIC_VIDEO_TYPE_OMV'), unavailable,
    panelRow('fffffffffff', 'Podcast', 'MUSIC_VIDEO_TYPE_PODCAST_EPISODE'),
    panelRow('ggggggggggg', 'Unknown', 'UNKNOWN'), panelRow('bad', 'Invalid'),
  ]));
  assert.deepEqual(songs.map(song => song.id), ['aaaaaaaaaaa', 'bbbbbbbbbbb', 'ddddddddddd']);
  assert.equal(songs[1]?.duration, 217);
  assert.deepEqual(parseMusicRecommendations({}), []);
});

test('Music transport sends an unfiltered query to WEB_REMIX and propagates cancellation', async context => {
  const controller = new AbortController();
  let called = 0;
  context.mock.method(globalThis, 'fetch', async (url: string, options: RequestInit) => {
    called++;
    assert.equal(url, 'https://music.youtube.com/youtubei/v1/search?prettyPrint=false');
    assert.equal(options.method, 'POST');
    assert.ok(options.signal);
    controller.abort();
    assert.equal(options.signal.aborted, true);
    const body = JSON.parse(String(options.body));
    assert.equal(body.query, 'Кино & live #1');
    assert.equal('params' in body, false);
    assert.equal(body.context.client.clientName, 'WEB_REMIX');
    assert.match(body.context.client.clientVersion, /^1\.\d{8}\.01\.00$/);
    return Response.json(searchResponse([searchRow('aaaaaaaaaaa', 'Song')]));
  });
  await assert.rejects(youtubeMusicJson('search', { query: 'Кино & live #1' }, controller.signal));
  await assert.rejects(youtubeMusicJson('search', { query: 'Cancelled' }, controller.signal));
  assert.equal(called, 1);
});

test('Music transport parses successful responses and rejects HTTP and API errors', async context => {
  let response = Response.json(searchResponse([searchRow('aaaaaaaaaaa', 'Song')]));
  context.mock.method(globalThis, 'fetch', async () => response);
  assert.equal((await youtubeMusicJson('search', { query: 'Song' })).entries?.[0]?.title, 'Song');
  response = Response.json(nextResponse([panelRow('bbbbbbbbbbb', 'Next')]));
  assert.equal((await youtubeMusicJson('next', { videoId: 'aaaaaaaaaaa' })).entries?.[0]?.id, 'bbbbbbbbbbb');
  response = new Response('Unavailable', { status: 503 });
  await assert.rejects(youtubeMusicJson('search', { query: 'Song' }), /HTTP 503/);
  response = Response.json({ error: { code: 400 } });
  await assert.rejects(youtubeMusicJson('search', { query: 'Song' }), /не вернул/);
});

test('Monetochka Monopoly top-result card precedes similarly named songs by other artists', () => {
  const originalArtist = { ...artist, text: 'Монеточка' };
  const response = { contents: { tabbedSearchResultsRenderer: { tabs: [{ tabRenderer: { content: {
    sectionListRenderer: { contents: [
      { musicCardShelfRenderer: {
        title: runs('Монополия'), subtitle: { runs: [{ text: 'Видео' }, originalArtist, { text: '3:54' }] },
        onTap: { watchEndpoint: watch('vwsSW-xXikw', 'MUSIC_VIDEO_TYPE_OMV') },
        thumbnail: { musicThumbnailRenderer: { thumbnail: { thumbnails: [{ url: 'original-artwork' }] } } },
      } },
      { itemSectionRenderer: { contents: [searchRow('2cVsLKX-GbQ', 'Монополія')] } },
      { musicShelfRenderer: { contents: [searchRow('7o9v5FRME6M', 'Монополия')] } },
    ] },
  } } }] } } };
  const songs = parseMusicSearch(response);
  assert.deepEqual(songs.map(song => song.id), ['vwsSW-xXikw', '2cVsLKX-GbQ', '7o9v5FRME6M']);
  assert.equal(songs[0]?.artist, 'Монеточка');
  assert.equal(songs[0]?.title, 'Монополия');
  assert.equal(songs[0]?.duration, 234);
});

test('artist top cards contribute their song rows but never become a playable track themselves', () => {
  const response = { contents: { musicCardShelfRenderer: {
    title: runs('Artist'), onTap: { browseEndpoint: { browseId: 'UCartist' } },
    contents: [searchRow('aaaaaaaaaaa', 'First song'), searchRow('bbbbbbbbbbb', 'Second song')],
  } } };
  assert.deepEqual(parseMusicSearch(response).map(song => song.id), ['aaaaaaaaaaa', 'bbbbbbbbbbb']);
});
