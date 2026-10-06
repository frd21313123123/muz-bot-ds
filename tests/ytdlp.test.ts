import assert from 'node:assert/strict';
import test from 'node:test';
import { isSong, YtdlpClient, type VideoInfo } from '../src/utils/ytdlp.js';

test('search uses only the YouTube Music songs section, keeps order and removes invalid entries and duplicates', async () => {
  const calls: string[][] = [];
  class Stub extends YtdlpClient {
    override async json(args: string[]): Promise<VideoInfo> {
      calls.push(args);
      return { entries: [{ id: 'bad', title: 'Bad' }, { id: 'aaaaaaaaaaa', title: 'First', duration: 180 },
        { id: 'aaaaaaaaaaa', title: 'Duplicate' }, { id: 'bbbbbbbbbbb', title: 'Second' },
        { id: 'ccccccccccc', title: ' ' }] };
    }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  assert.deepEqual((await client.searchCandidates('Numb')).map((entry) => entry.title), ['First', 'Second']);
  assert.deepEqual(calls, [['--flat-playlist', '--playlist-end', '5', 'https://music.youtube.com/search?q=Numb#songs']]);
  assert.deepEqual(await client.searchCandidates(''), []);
  await assert.rejects(client.searchCandidates('Numb', 11));
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.searchCandidates('Numb', 5, controller.signal));
  assert.equal(calls.length, 1);
  await client.search('Numb');
  assert.deepEqual(calls[1], ['--flat-playlist', '--playlist-end', '1', 'https://music.youtube.com/search?q=Numb#songs']);
  await client.searchCandidates('  Кино & live #1  ', 3);
  assert.equal(calls[2]?.[3], `https://music.youtube.com/search?q=${encodeURIComponent('Кино & live #1')}#songs`);
});

test('song search returns candidates immediately without blocking on video metadata', async () => {
  let videoCalled = false;
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> {
      return { entries: [{ id: '5FHjH3NBFDI', title: 'Не дано' }, { id: 'bbbbbbbbbbb', title: 'Another' }] };
    }
    override async video(url: string): Promise<VideoInfo> {
      videoCalled = true;
      return { id: '5FHjH3NBFDI', title: 'Не дано', duration: 216 };
    }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  const candidates = await client.searchCandidates('Hi-Fi Не дано');
  assert.equal(videoCalled, false, 'searchCandidates must not block on video metadata');
  assert.equal(candidates[0]?.id, '5FHjH3NBFDI');
  assert.equal(candidates[0]?.title, 'Не дано');
  assert.equal(candidates[1]?.id, 'bbbbbbbbbbb');
  assert.equal((await client.search('Hi-Fi Не дано'))?.id, '5FHjH3NBFDI');
});

test('search candidates and videos are cached, and clearCache resets them', async () => {
  let jsonCalls = 0;
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> {
      jsonCalls++;
      return { id: 'aaaaaaaaaaa', entries: [{ id: 'aaaaaaaaaaa', title: 'Cached Song' }] };
    }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  const first = await client.searchCandidates('Cached Song', 5);
  const second = await client.searchCandidates('Cached Song', 5);
  assert.equal(jsonCalls, 1);
  assert.deepEqual(first, second);

  // Video caching
  await client.video('https://www.youtube.com/watch?v=aaaaaaaaaaa');
  await client.video('https://www.youtube.com/watch?v=aaaaaaaaaaa');
  assert.equal(jsonCalls, 2);

  // Clear cache resets them
  client.clearCache();
  await client.searchCandidates('Cached Song', 5);
  assert.equal(jsonCalls, 3);
});

test('radio recommendations skip the current video and duplicate entries', async () => {
  const calls: string[][] = [];
  class StubYtdlp extends YtdlpClient {
    override async video(url: string): Promise<VideoInfo> {
      return { id: new URL(url).searchParams.get('v')!, title: 'Song', categories: ['Music'] };
    }
    override async json(args: string[]): Promise<VideoInfo> {
      calls.push(args);
      return { entries: [
        { id: 'aaaaaaaaaaa', title: 'Current' },
        { id: 'bbbbbbbbbbb', title: 'Related' },
        { id: 'bbbbbbbbbbb', title: 'Duplicate' },
        { id: 'invalid', title: 'Invalid' },
        { id: 'ccccccccccc', title: 'Another' },
      ] };
    }
  }
  const ytdlp = new StubYtdlp({ command: 'unused', prefix: [] });
  const related = await ytdlp.related('aaaaaaaaaaa', 5);
  assert.deepEqual(related.map((item) => item.videoId), ['bbbbbbbbbbb', 'ccccccccccc']);
  assert.ok(related.every((item) => item.isAutoplay && item.requestedBy === '🤖 Бесконечное'));
  assert.deepEqual(calls[0], [
    '--flat-playlist', '--playlist-items', '2:6',
    'https://www.youtube.com/watch?v=aaaaaaaaaaa&list=RDaaaaaaaaaaa',
  ]);
  assert.deepEqual(await ytdlp.related('bad'), []);
  assert.equal(calls.length, 1);
});

test('song verification requires music metadata and excludes spoken content and broadcasts', () => {
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

test('radio verifies full metadata, skips unavailable and non-music videos and caps songs in radio order', async () => {
  const checked: string[] = [];
  const entries: VideoInfo[] = [
    { id: 'bbbbbbbbbbb', title: 'Artist podcast' },
    { id: 'ccccccccccc', title: 'Looks like a song' },
    { id: 'ddddddddddd', title: 'Unknown video' },
    { id: 'eeeeeeeeeee', title: 'Unavailable song' },
    { id: 'fffffffffff', title: 'First song' },
    { id: 'ggggggggggg', title: 'Second song' },
    { id: 'hhhhhhhhhhh', title: 'Third song' },
  ];
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> { return { entries }; }
    override async video(url: string, signal?: AbortSignal): Promise<VideoInfo> {
      assert.ok(signal);
      const id = new URL(url).searchParams.get('v')!;
      checked.push(id);
      if (id === 'eeeeeeeeeee') throw new Error('Unavailable');
      if (id === 'ccccccccccc') return { id, title: 'Conversation', categories: ['People & Blogs'] };
      if (id === 'ddddddddddd') return { id, title: 'Unknown video' };
      return { id, title: 'Confirmed song', categories: ['Music'], duration: 200 };
    }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  const songs = await client.related('aaaaaaaaaaa', 2);
  assert.deepEqual(songs.map((track) => track.videoId), ['fffffffffff', 'ggggggggggg']);
  assert.equal(checked.includes('bbbbbbbbbbb'), false, 'obvious podcasts must not trigger full extraction');
  assert.ok(songs.every((track) => track.duration === '3:20'));
  await assert.rejects(client.related('aaaaaaaaaaa', 0));
});

test('radio with no confirmed songs returns no tracks rather than unverified videos', async () => {
  class Stub extends YtdlpClient {
    override async json(): Promise<VideoInfo> {
      return { entries: [{ id: 'bbbbbbbbbbb', title: 'Video' }, { id: 'ccccccccccc', title: 'Video' }] };
    }
    override async video(url: string): Promise<VideoInfo> {
      return { id: new URL(url).searchParams.get('v')!, title: 'Video' };
    }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  assert.deepEqual(await client.related('aaaaaaaaaaa'), []);
});

test('empty song search does not fall back to ordinary videos and cancellation remains effective', async () => {
  const calls: string[][] = [];
  const controller = new AbortController();
  class Stub extends YtdlpClient {
    override async json(args: string[]): Promise<VideoInfo> { calls.push(args); return {}; }
  }
  const client = new Stub({ command: 'unused', prefix: [] });
  assert.equal(await client.search('Numb'), null);
  assert.equal(await client.search(' '), null);
  assert.equal(calls.length, 1);
  class Aborted extends Stub {
    override async json(): Promise<VideoInfo> { controller.abort(); return { entries: [{ id: 'aaaaaaaaaaa', title: 'Song' }] }; }
  }
  await assert.rejects(new Aborted({ command: 'unused', prefix: [] }).searchCandidates('Numb', 5, controller.signal));
});
