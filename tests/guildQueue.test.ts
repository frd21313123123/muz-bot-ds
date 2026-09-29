import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { StreamType } from '@discordjs/voice';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { MusicClient, Track } from '../src/types.js';
import type { ManagedAudioStream } from '../src/utils/stream.js';

const track = (id: string): Track => ({
  videoId: id, url: `https://www.youtube.com/watch?v=${id}`, title: id,
  duration: '1:00', thumbnail: null, requestedBy: 'Tester',
});

const waitFor = async (predicate: () => boolean): Promise<void> => {
  const end = Date.now() + 3_000;
  while (!predicate()) {
    if (Date.now() > end) throw new Error('queue did not advance');
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
};

test('skip advances to the next manual track and stops the old stream', async () => {
  const streams: PassThrough[] = [];
  const factory = (): ManagedAudioStream => {
    const stream = new PassThrough();
    streams.push(stream);
    return { stream, type: StreamType.OggOpus, destroy: () => stream.destroy() };
  };
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild', client, factory);
  client.queues.set('guild', queue);
  try {
    await queue.addTrack(track('aaaaaaaaaaa'));
    await queue.addTrack(track('bbbbbbbbbbb'));
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(queue.tracks.length, 1);
    assert.equal(queue.skip(), true);
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.equal(streams[0]?.destroyed, true);
    assert.equal(queue.tracks.length, 0);
  } finally {
    await queue.stop();
  }
});

test('manual track added while recommendations load plays before them', async () => {
  const streams: PassThrough[] = [];
  let resolveRelated: ((tracks: Track[]) => void) | undefined;
  const related = new Promise<Track[]>((resolve) => { resolveRelated = resolve; });
  let requested = false;
  const client = {
    queues: new Map(),
    ytdlp: { related: async () => { requested = true; return related; } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-recommendations', client, () => {
    const stream = new PassThrough();
    streams.push(stream);
    return { stream, type: StreamType.OggOpus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    queue.player.stop(true);
    await waitFor(() => requested);
    await queue.addTrack(track('bbbbbbbbbbb'));
    resolveRelated?.([{ ...track('ccccccccccc'), isAutoplay: true }]);
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.deepEqual(queue.tracks.map((item) => item.videoId), ['ccccccccccc']);
    assert.equal(streams[0]?.destroyed, true);
  } finally {
    resolveRelated?.([]);
    await queue.stop();
  }
});

test('stop during recommendation lookup prevents playback and removes queue', async () => {
  let resolveRelated: ((tracks: Track[]) => void) | undefined;
  const related = new Promise<Track[]>((resolve) => { resolveRelated = resolve; });
  let requested = false;
  const streams: PassThrough[] = [];
  const client = {
    queues: new Map(),
    ytdlp: { related: async () => { requested = true; return related; } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-stop', client, () => {
    const stream = new PassThrough();
    streams.push(stream);
    return { stream, type: StreamType.OggOpus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    queue.player.stop(true);
    await waitFor(() => requested);
    await queue.stop();
    resolveRelated?.([{ ...track('bbbbbbbbbbb'), isAutoplay: true }]);
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(queue.closed, true);
    assert.equal(queue.currentTrack, null);
    assert.equal(queue.tracks.length, 0);
    assert.equal(streams.length, 1);
    assert.equal(streams[0]?.destroyed, true);
    assert.equal(client.queues.has(queue.guildId), false);
  } finally {
    resolveRelated?.([]);
    await queue.stop();
  }
});

test('a track setup failure skips to the next queued track', async () => {
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-failure', client, (url) => {
    if (url.includes('aaaaaaaaaaa')) throw new Error('test setup failure');
    const stream = new PassThrough();
    return { stream, type: StreamType.OggOpus, destroy: () => stream.destroy() };
  });
  try {
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    assert.equal(queue.currentTrack?.videoId, 'bbbbbbbbbbb');
    assert.equal(queue.tracks.length, 0);
  } finally {
    await queue.stop();
  }
});
