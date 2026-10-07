import assert from 'node:assert/strict';
import test from 'node:test';
import { isSong, YtdlpClient, ytdlpEnvironment, type VideoInfo } from '../src/utils/ytdlp.js';
import type { MusicEndpoint } from '../src/utils/youtubeMusic.js';

class MusicStub extends YtdlpClient {
  calls: { endpoint: MusicEndpoint; payload: Record<string, unknown> }[] = [];
  entries: VideoInfo[] = [];
  constructor() { super({ command: 'unused', prefix: [] }); }
  override async musicJson(endpoint: MusicEndpoint, payload: Record<string, unknown>): Promise<VideoInfo> {
    this.calls.push({ endpoint, payload });
    return { entries: this.entries };
  }
  override async json(): Promise<VideoInfo> { assert.fail('Music must never use ordinary YouTube extraction'); }
}

test('YouTube JS heap is bounded without overwriting explicit Node options', () => {
  assert.equal(ytdlpEnvironment({ NODE_OPTIONS: '--no-warnings' }).NODE_OPTIONS, '--no-warnings --max-old-space-size=128');
  assert.equal(ytdlpEnvironment({ NODE_OPTIONS: '--max-old-space-size=256' }).NODE_OPTIONS, '--max-old-space-size=256');
  assert.equal(ytdlpEnvironment({ NODE_OPTIONS: '', YT_DLP_NODE_HEAP_MB: '0' }).NODE_OPTIONS, '');
});

test('ordinary video song verification still requires music metadata and excludes spoken content and broadcasts', () => {
  const song: VideoInfo = { id: 'aaaaaaaaaaa', title: 'Artist — Song', categories: ['Music'] };
  assert.equal(isSong(song), true);
  assert.equal(isSong({ ...song, title: 'Song (Live at Wembley)', live_status: 'was_live' }), true);
  assert.equal(isSong({ ...song, categories: undefined, track: 'Song', artists: ['Artist'] }), true);
  assert.equal(isSong({ ...song, categories: undefined, track: 'Song', artist: 'Artist' }), true);
  for (const info of [
    { ...song, title: 'Artist Podcast #42' }, { ...song, title: 'Подкаст о музыке' },
    { ...song, title: 'Интервью с исполнителем' }, { ...song, title: 'Artist interview' },
    { ...song, title: 'Лекция о музыке' }, { ...song, title: 'Аудиокнига — глава 1' },
    { ...song, title: 'Artist — Full Album' }, { ...song, title: 'Artist DJ set' },
    { ...song, is_live: true }, { ...song, live_status: 'is_upcoming' },
    { ...song, categories: ['People & Blogs'] }, { ...song, categories: undefined, channel: 'Artist' },
    { ...song, categories: undefined, artist: 'Artist' }, { ...song, id: 'bad' }, { ...song, title: ' ' },
  ]) assert.equal(isSong(info), false, JSON.stringify(info));
});

test('main Music search preserves service order, removes invalid entries and returns the first musical result', async () => {
  const client = new MusicStub();
  client.entries = [{ id: 'bad', title: 'Bad' }, { id: 'aaaaaaaaaaa', title: 'First', duration: 180 },
    { id: 'aaaaaaaaaaa', title: 'Duplicate' }, { id: 'bbbbbbbbbbb', title: 'Second' }, { id: 'ccccccccccc', title: ' ' }];
  assert.deepEqual((await client.searchCandidates('Numb')).map(entry => entry.title), ['First', 'Second']);
  assert.deepEqual(client.calls, [{ endpoint: 'search', payload: { query: 'Numb' } }]);
  assert.equal((await client.search('Numb'))?.id, 'aaaaaaaaaaa');
  assert.equal((await client.searchCandidates('Numb', 1)).length, 1);
  assert.deepEqual(await client.searchCandidates(''), []);
  await assert.rejects(client.searchCandidates('Numb', 11));
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.searchCandidates('Numb', 5, controller.signal));
  assert.equal(client.calls.length, 2);
  await client.searchCandidates('  Кино & live #1  ', 3);
  assert.equal(client.calls[2]?.payload.query, 'Кино & live #1');
});

test('song search never blocks on ordinary YouTube video metadata', async () => {
  class Stub extends MusicStub {
    override async video(): Promise<VideoInfo> { assert.fail('search must not request ordinary video metadata'); }
  }
  const client = new Stub();
  client.entries = [{ id: '5FHjH3NBFDI', title: 'Не дано', duration: 216 }];
  assert.equal((await client.search('Hi-Fi Не дано'))?.id, '5FHjH3NBFDI');
});

test('Music search cache is isolated from callers and clearCache resets it', async () => {
  const client = new MusicStub();
  client.entries = [{ id: 'aaaaaaaaaaa', title: 'Cached Song', artists: ['Artist'] }];
  const first = await client.searchCandidates('Cached Song', 5);
  first[0]!.title = 'Changed'; first[0]!.artists!.push('Changed');
  const second = await client.searchCandidates('cached song', 5);
  assert.equal(client.calls.length, 1);
  assert.equal(second[0]?.title, 'Cached Song');
  assert.deepEqual(second[0]?.artists, ['Artist']);
  client.clearCache();
  await client.searchCandidates('Cached Song', 5);
  assert.equal(client.calls.length, 2);
});

test('ordinary video metadata retains its independent cache', async () => {
  let calls = 0;
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> { calls++; return { id: 'aaaaaaaaaaa', title: 'Video' }; }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  await client.video('https://www.youtube.com/watch?v=aaaaaaaaaaa');
  await client.video('https://www.youtube.com/watch?v=aaaaaaaaaaa');
  assert.equal(calls, 1);
  client.clearCache();
  await client.video('https://www.youtube.com/watch?v=aaaaaaaaaaa');
  assert.equal(calls, 2);
});

test('recommendations use the current song Music mix, preserving order and excluding history and duplicates', async () => {
  const client = new MusicStub();
  client.entries = [
    { id: 'aaaaaaaaaaa', title: 'Current' }, { id: 'bbbbbbbbbbb', title: 'Already played' },
    { id: 'ccccccccccc', title: 'First fresh', duration: 200 }, { id: 'ccccccccccc', title: 'Duplicate' },
    { id: 'invalid', title: 'Invalid' }, { id: 'ddddddddddd', title: 'Second fresh' },
  ];
  const songs = await client.related('aaaaaaaaaaa', 1, { excludeIds: new Set(['bbbbbbbbbbb']) });
  assert.deepEqual(songs.map(track => track.videoId), ['ccccccccccc']);
  assert.equal(songs[0]?.duration, '3:20');
  assert.ok(songs.every(track => track.isAutoplay && track.requestedBy === '🤖 Бесконечное'));
  assert.deepEqual(client.calls, [{ endpoint: 'next', payload: {
    videoId: 'aaaaaaaaaaa', playlistId: 'RDAMVMaaaaaaaaaaa', params: 'wAEB', isAudioOnly: true,
    enablePersistentPlaylistPanel: true, tunerSettingValue: 'AUTOMIX_SETTING_NORMAL',
  } }]);
  assert.deepEqual((await client.related('aaaaaaaaaaa', 5)).map(track => track.videoId), ['bbbbbbbbbbb', 'ccccccccccc', 'ddddddddddd']);
  assert.deepEqual(await client.related('bad'), []);
  await assert.rejects(client.related('aaaaaaaaaaa', 0));
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.related('aaaaaaaaaaa', 1, { signal: controller.signal }));
  assert.equal(client.calls.length, 2);
});

test('empty Music search and recommendations do not fall back to ordinary YouTube', async () => {
  const client = new MusicStub();
  assert.equal(await client.search('Numb'), null);
  assert.equal(await client.search(' '), null);
  assert.deepEqual(await client.related('aaaaaaaaaaa'), []);
  assert.equal(client.calls.length, 2);
});

test('cancellation during Music response handling never returns a search or recommendation', async () => {
  for (const operation of ['search', 'related']) {
    const controller = new AbortController();
    class Aborted extends MusicStub {
      override async musicJson(): Promise<VideoInfo> {
        controller.abort(); return { entries: [{ id: 'bbbbbbbbbbb', title: 'Song' }] };
      }
    }
    const client = new Aborted();
    await assert.rejects(operation === 'search' ? client.search('Numb', controller.signal)
      : client.related('aaaaaaaaaaa', 1, { signal: controller.signal }));
  }
});
