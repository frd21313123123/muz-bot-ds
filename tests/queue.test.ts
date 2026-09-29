import assert from 'node:assert/strict';
import test from 'node:test';
import { TrackQueue } from '../src/utils/TrackQueue.js';
import type { Track } from '../src/types.js';

const track = (videoId: string, isAutoplay = false): Track => ({
  videoId, url: `https://www.youtube.com/watch?v=${videoId}`, title: videoId,
  duration: '1:00', thumbnail: null, requestedBy: 'Test', isAutoplay,
});

test('manual tracks stay before recommendations', () => {
  const queue = new TrackQueue();
  queue.addMany([track('aaaaaaaaaaa'), track('bbbbbbbbbbb', true), track('ccccccccccc', true)]);
  queue.add(track('ddddddddddd'));
  assert.deepEqual(queue.items.map((item) => item.videoId),
    ['aaaaaaaaaaa', 'ddddddddddd', 'bbbbbbbbbbb', 'ccccccccccc']);
  assert.equal(queue.shift()?.videoId, 'aaaaaaaaaaa');
  assert.equal(queue.clear(), 3);
  assert.equal(queue.items.length, 0);
});
