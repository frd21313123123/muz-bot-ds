import assert from 'node:assert/strict';
import test from 'node:test';
import { formatDuration, parseDuration } from '../src/utils/duration.js';
import { toTrack } from '../src/utils/ytdlp.js';
import { nowPlayingEmbed, playerEmbed } from '../src/utils/embeds.js';
import type { GuildQueue } from '../src/utils/GuildQueue.js';
import type { Track } from '../src/types.js';

test('track duration prefers numeric metadata and parses valid minute/hour strings as fallback', () => {
  const video = { id: '5FHjH3NBFDI', title: 'Не дано' };
  assert.equal(toTrack({ ...video, duration: 216, duration_string: 'LIVE' }, 'Test').duration, '3:36');
  assert.equal(toTrack({ ...video, duration_string: ' 03:36 ' }, 'Test').duration, '3:36');
  assert.equal(toTrack({ ...video, duration: 3661.9 }, 'Test').duration, '1:01:01');
  assert.equal(toTrack({ ...video, duration: NaN, duration_string: '01:02:03' }, 'Test').duration, '1:02:03');
  for (const duration_string of ['LIVE', '?', '', '3:99', '1:60:00', '-3:00']) {
    assert.equal(toTrack({ ...video, duration_string }, 'Test').duration, '?');
    assert.equal(parseDuration(duration_string), null);
  }
  assert.equal(parseDuration('90:00'), 5400);
  assert.equal(formatDuration(5400), '1:30:00');
});

test('only current live broadcasts are marked live; past broadcasts and missing metadata are not', () => {
  const video = { id: 'aaaaaaaaaaa', title: 'Song' };
  assert.equal(toTrack({ ...video, live_status: 'is_live' }, 'Test').isLive, true);
  assert.equal(toTrack({ ...video, is_live: true }, 'Test').isLive, true);
  assert.equal(toTrack({ ...video, is_live: true, live_status: 'was_live', duration: 200 }, 'Test').isLive, false);
  assert.equal(toTrack({ ...video, live_status: 'is_upcoming' }, 'Test').isLive, false);
  assert.equal(toTrack(video, 'Test').isLive, false);
});

test('player displays actual elapsed/total time and never labels missing duration as a live stream', () => {
  const base = toTrack({ id: '5FHjH3NBFDI', title: 'Не дано', duration: 216 }, 'Test');
  const queue = (track: Track, elapsed = 100) => ({ currentTrack: track, isPaused: false, speed: 1,
    volume: 0.5, autoplay: false, tracks: [], getElapsedSeconds: () => elapsed }) as unknown as GuildQueue;
  const known = playerEmbed(queue(base)).toJSON();
  assert.match(known.description!, /1:40 \/ 3:36/);
  assert.equal(known.description!.includes('Прямой эфир'), false);
  assert.match(playerEmbed(queue({ ...base, duration: '?' })).toJSON().description!, /1:40 \/ длительность неизвестна/);
  assert.equal(playerEmbed(queue({ ...base, duration: '?' })).toJSON().description!.includes('Прямой эфир'), false);
  assert.match(playerEmbed(queue({ ...base, isLive: true })).toJSON().description!, /🔴 Прямой эфир/);
  assert.match(playerEmbed(queue(base, 1000)).toJSON().description!, /3:36 \/ 3:36/);
  assert.equal(nowPlayingEmbed({ ...base, duration: '?' }).toJSON().fields?.[0]?.value, 'Длительность неизвестна');
});
