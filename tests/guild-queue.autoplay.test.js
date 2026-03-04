'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const GuildQueue = require('../src/utils/GuildQueue');

test('_playAutoplay uses the first recommendation as next track', async () => {
  const played = [];
  const queue = {
    guildId: 'g-1',
    currentTrack: { videoId: 'current-video', title: 'Current' },
    _fetchRelatedTracks: async () => ([
      { videoId: 'rec-1', title: 'First recommendation' },
      { videoId: 'rec-2', title: 'Second recommendation' },
    ]),
    _playTrack: async (track) => {
      played.push(track);
    },
    _startIdleTimer: () => {
      throw new Error('idle timer should not start when recommendations exist');
    },
  };

  await GuildQueue.prototype._playAutoplay.call(queue);

  assert.equal(played.length, 1);
  assert.equal(played[0].videoId, 'rec-1');
});

test('_playAutoplay clears current track and starts idle timer when no recommendations', async () => {
  let idleStarted = false;
  const queue = {
    guildId: 'g-2',
    currentTrack: { videoId: 'current-video', title: 'Current' },
    _fetchRelatedTracks: async () => [],
    _playTrack: async () => {
      throw new Error('play should not be called when recommendations are empty');
    },
    _startIdleTimer: () => {
      idleStarted = true;
    },
  };

  await GuildQueue.prototype._playAutoplay.call(queue);

  assert.equal(queue.currentTrack, null);
  assert.equal(idleStarted, true);
});
