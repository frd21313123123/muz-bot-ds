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
