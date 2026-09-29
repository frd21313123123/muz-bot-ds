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
