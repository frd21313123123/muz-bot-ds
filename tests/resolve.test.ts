import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyQuery, resolveQuery, type MetadataClient } from '../src/utils/resolve.js';
import type { VideoInfo } from '../src/utils/ytdlp.js';

const entry = (id: string): VideoInfo => ({ id, title: `Title ${id}`, duration: 90 });
const seen: string[] = [];
const metadata: MetadataClient = {
  async video(url) { seen.push(`video:${url}`); return entry('abcdefghijk'); },
  async playlist(url, limit) {
    seen.push(`playlist:${limit}:${url}`);
    return { title: 'Test list', entries: [entry('abcdefghijk'), entry('lmnopqrstuv')] };
  },
  async search(query) { seen.push(`search:${query}`); return entry('abcdefghijk'); },
};

test('YouTube Music and radio mix resolve to canonical video', async () => {
  assert.deepEqual(classifyQuery('https://music.youtube.com/watch?v=abcdefghijk&list=RDAMVMfoo'), {
    kind: 'video', value: 'https://www.youtube.com/watch?v=abcdefghijk',
  });
  const result = await resolveQuery('https://music.youtube.com/watch?v=abcdefghijk&list=RDAMVMfoo', 'Alice', metadata);
  assert.equal(result?.type, 'single');
  if (result?.type === 'single') assert.equal(result.track.url, 'https://www.youtube.com/watch?v=abcdefghijk');
});

test('liked music track link resolves as an ordinary video', async () => {
  seen.length = 0;
  assert.deepEqual(classifyQuery('https://music.youtube.com/watch?v=abcdefghijk&list=LM'), {
    kind: 'video', value: 'https://www.youtube.com/watch?v=abcdefghijk',
  });
  const result = await resolveQuery('https://music.youtube.com/watch?v=abcdefghijk&list=LM', 'Alice', metadata);
  assert.equal(result?.type, 'single');
  assert.deepEqual(seen, ['video:https://www.youtube.com/watch?v=abcdefghijk']);
});

test('liked music playlist without a video gives a useful error', () => {
  assert.throws(() => classifyQuery('https://www.youtube.com/playlist?list=LM'),
    /не содержит конкретного трека.*v=/);
});

test('playlist limit is passed to yt-dlp and result is capped', async () => {
  seen.length = 0;
  const result = await resolveQuery('https://www.youtube.com/playlist?list=PLxyz', 'Alice', metadata, 1);
  assert.equal(result?.type, 'playlist');
  if (result?.type === 'playlist') assert.equal(result.tracks.length, 1);
  assert.match(seen[0] ?? '', /^playlist:1:/);
});

test('text search returns one track', async () => {
  const result = await resolveQuery('Test Song', 'Alice', metadata);
  assert.equal(result?.type, 'single');
});

test('unsupported host and malformed video id fail before invoking yt-dlp', () => {
  assert.throws(() => classifyQuery('https://example.com/watch?v=abcdefghijk'));
  assert.throws(() => classifyQuery('https://www.youtube.com/watch?v=bad'));
});

test('fast resolve on video url creates track immediately without calling metadata.video', async () => {
  seen.length = 0;
  const result = await resolveQuery('https://www.youtube.com/watch?v=abcdefghijk', 'Alice', metadata, 1, { fast: true });
  assert.equal(result?.type, 'single');
  if (result?.type === 'single') {
    assert.equal(result.track.videoId, 'abcdefghijk');
    assert.equal(result.track.title, 'Загрузка…');
    assert.equal(result.track.duration, '?');
    assert.equal(result.track.url, 'https://www.youtube.com/watch?v=abcdefghijk');
  }
  assert.equal(seen.length, 0, 'metadata.video must not be called in fast mode');
});

const soundcloudEntry: VideoInfo = {
  id: '2408461857', title: 'Dâu Hào Môn Lofi', duration: 244.1, thumbnail: 'https://i1.sndcdn.com/a-t500x500.jpg',
  webpage_url: 'https://soundcloud.com/some-artist/some-track',
};

test('SoundCloud links are classified without tracking parameters', () => {
  assert.deepEqual(classifyQuery('https://soundcloud.com/some-artist/some-track?si=abc&utm_source=clipboard'), {
    kind: 'video', value: 'https://soundcloud.com/some-artist/some-track', source: 'soundcloud',
  });
  assert.deepEqual(classifyQuery('https://m.soundcloud.com/some-artist/sets/mix'), {
    kind: 'playlist', value: 'https://soundcloud.com/some-artist/sets/mix', source: 'soundcloud',
  });
  assert.equal(classifyQuery('https://soundcloud.com/some-artist').kind, 'playlist');
  assert.equal(classifyQuery('https://soundcloud.com/some-artist/tracks').kind, 'playlist');
  assert.equal(classifyQuery('https://soundcloud.com/some-artist/some-track/s-Secret1').kind, 'video');
  assert.equal(classifyQuery('https://on.soundcloud.com/AbC123').source, 'soundcloud');
  assert.throws(() => classifyQuery('https://soundcloud.com/'), /SoundCloud/);
  assert.throws(() => classifyQuery('https://soundcloud.com.evil.example/a/b'));
});

test('SoundCloud track resolves with a namespaced id and its own url', async () => {
  const result = await resolveQuery('https://soundcloud.com/some-artist/some-track', 'Alice',
    { ...metadata, async video() { return soundcloudEntry; } }, 1, { fast: true });
  assert.equal(result?.type, 'single');
  if (result?.type === 'single') {
    assert.equal(result.track.source, 'soundcloud');
    assert.equal(result.track.videoId, 'sc:2408461857');
    assert.equal(result.track.url, 'https://soundcloud.com/some-artist/some-track');
    assert.equal(result.track.duration, '4:04');
  }
});

test('SoundCloud set uses entry urls and a short link to one track becomes a single track', async () => {
  const set = { title: 'Mix', entries: [{ id: '1', title: 'One', url: 'https://soundcloud.com/a/one' },
    { id: '2', title: 'Two', url: 'https://soundcloud.com/a/two' }, { id: 'bad', title: 'Bad' }] };
  const result = await resolveQuery('https://soundcloud.com/a/sets/mix', 'Alice',
    { ...metadata, async playlist() { return set; } }, 25);
  assert.equal(result?.type, 'playlist');
  if (result?.type === 'playlist') assert.deepEqual(result.tracks.map(track => track.videoId), ['sc:1', 'sc:2']);
  const single = await resolveQuery('https://on.soundcloud.com/AbC123', 'Alice',
    { ...metadata, async playlist() { return soundcloudEntry; } });
  assert.equal(single?.type, 'single');
});

test('SoundCloud search is used only when requested', async () => {
  const queries: string[] = [];
  const client: MetadataClient = { ...metadata, async searchSoundCloud(query) { queries.push(query); return soundcloudEntry; } };
  seen.length = 0;
  const result = await resolveQuery('lofi', 'Alice', client, 1, { source: 'soundcloud' });
  assert.equal(result?.type === 'single' && result.track.source, 'soundcloud');
  assert.deepEqual(queries, ['lofi']);
  assert.equal(seen.length, 0, 'YouTube search must not run');
  await assert.rejects(resolveQuery('lofi', 'Alice', metadata, 1, { source: 'soundcloud' }), /недоступен/);
});
