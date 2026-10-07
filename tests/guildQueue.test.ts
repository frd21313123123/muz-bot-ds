import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { StreamType, NoSubscriberBehavior } from '@discordjs/voice';
import { AudioPlayerStatus, VoiceConnectionStatus, type AudioPlayer, type VoiceConnection } from '@discordjs/voice';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { MusicClient, Track } from '../src/types.js';
import type { ManagedAudioStream } from '../src/utils/stream.js';
import type { MusicPlaybackOptions } from '../src/utils/audio.js';

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

test('a temporary audio stall resumes the same resource and EOF still advances promptly', async () => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-stall', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  let subscription: { unsubscribe(): void } | undefined;
  let packets = 0;
  const connection = {
    state: { status: VoiceConnectionStatus.Ready },
    onSubscriptionRemoved: () => {}, setSpeaking: () => {},
    prepareAudioPacket: () => { packets++; }, dispatchAudio: () => true,
    destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  queue.connection = connection;
  subscription = Reflect.apply(Reflect.get(queue.player, 'subscribe'), queue.player, [connection]);
  try {
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    streams[0]!.write(Buffer.alloc(3840));
    await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
    const original = queue.player.state;
    assert.ok(original.status !== AudioPlayerStatus.Idle);
    await new Promise(resolve => setTimeout(resolve, 400));
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa', 'a download gap must not skip the track');
    assert.equal(streams[0]!.destroyed, false);
    assert.equal(queue.player.state.status, AudioPlayerStatus.Playing);
    assert.equal(queue.player.state.resource, original.resource);
    const elapsed = original.resource.playbackDuration;
    const previousPackets = packets;
    streams[0]!.write(Buffer.alloc(3840 * 10));
    await waitFor(() => original.resource.playbackDuration > elapsed && packets > previousPackets);
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    streams[0]!.end();
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.equal(streams.length, 2);
    assert.equal(queue.tracks.length, 0);
  } finally { await queue.stop(); }
});

test('stream error followed by Idle advances only once while recommendations load', async () => {
  let resolveRelated: ((tracks: Track[]) => void) | undefined;
  let requested = false;
  const related = new Promise<Track[]>(resolve => { resolveRelated = resolve; });
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => {
    requested = true; return related;
  } } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-error-idle', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    streams[0]!.destroy(new Error('temporary download failure'));
    await waitFor(() => requested);
    assert.equal(queue.player.state.status, AudioPlayerStatus.Idle);
    await queue.addTracks([track('bbbbbbbbbbb'), track('ccccccccccc')]);
    resolveRelated?.([]);
    await waitFor(() => queue.currentTrack?.videoId !== 'aaaaaaaaaaa');
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(queue.currentTrack?.videoId, 'bbbbbbbbbbb');
    assert.deepEqual(queue.tracks.map(item => item.videoId), ['ccccccccccc']);
    assert.equal(streams.length, 2);
  } finally { resolveRelated?.([]); await queue.stop(); }
});

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

test('repeat can be armed on an empty queue, repeats the current track and releases the manual queue when disabled', async () => {
  let resources = 0;
  const client = { queues: new Map(), ytdlp: { related: async () => { assert.fail('Repeat must not fetch recommendations'); } } } as unknown as MusicClient;
  const queue = new GuildQueue('repeat', client, () => {
    resources++; const stream = new PassThrough();
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true); queue.setLoop(true);
    assert.equal(queue.autoplay, false);
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    queue.player.stop(true); await waitFor(() => resources >= 2);
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.deepEqual(queue.tracks.map(track => track.videoId), ['bbbbbbbbbbb']);
    queue.setLoop(false); queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    queue.setAutoplay(true); assert.equal(queue.loopCurrent, false);
  } finally { await queue.stop(); }
});

test('speed replaces the current stream at its source position, keeps the queue and applies to subsequent tracks', async () => {
  const calls: { url: string; options: MusicPlaybackOptions; stream: PassThrough }[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('speed', client, (url, options) => {
    const stream = new PassThrough(); calls.push({ url, options, stream });
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  const resource = () => {
    const state = queue.player.state;
    assert.notEqual(state.status, AudioPlayerStatus.Idle);
    if (state.status === AudioPlayerStatus.Idle) throw new Error('Missing resource');
    return state.resource;
  };
  try {
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    resource().playbackDuration = 3250;
    queue.setVolume(150); queue.setVoiceDucking(true); queue.setLoop(true);
    queue.setSpeed(0.5);
    assert.equal(calls[0]?.stream.destroyed, true);
    assert.deepEqual(calls[1]?.options, { speed: 0.5, startSeconds: 3.25 });
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.deepEqual(queue.tracks.map(track => track.videoId), ['bbbbbbbbbbb']);
    assert.equal(queue.loopCurrent, true);
    assert.ok(Math.abs(resource().volume!.volume - 0.3) < 0.00001);
    resource().playbackDuration = 2000;
    assert.equal(queue.getElapsedSeconds(), 4);
    queue.setSpeed(1.25);
    assert.deepEqual(calls[2]?.options, { speed: 1.25, startSeconds: 4.25 });
    resource().playbackDuration = 2000;
    assert.equal(queue.getElapsedSeconds(), 6);
    queue.setSpeed(1.25); assert.equal(calls.length, 3, 'same speed must not restart');
    queue.setLoop(false); queue.skip();
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.deepEqual(calls[3]?.options, { speed: 1.25, startSeconds: 0 });
    assert.equal(queue.getElapsedSeconds(), 0);
  } finally { await queue.stop(); }
});

test('changing speed preserves a paused track through buffering and resume can cancel a pending pause', async () => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('paused-speed', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  try {
    await queue.addTrack(track('aaaaaaaaaaa'));
    assert.equal(queue.pause(), true);
    queue.setSpeed(0.8);
    assert.equal(queue.isPaused, true);
    streams[1]!.write(Buffer.from([0xf8, 0xff, 0xfe]));
    await waitFor(() => queue.player.state.status === AudioPlayerStatus.Paused);
    assert.equal(queue.resume(), true);
    assert.equal(queue.isPaused, false);
    queue.pause(); queue.setSpeed(1.25);
    assert.equal(queue.player.state.status, AudioPlayerStatus.Buffering);
    assert.equal(queue.resume(), true);
    assert.equal(queue.isPaused, false);
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
  } finally { await queue.stop(); }
});

test('speed can be set before playback and failed replacements keep the original stream and settings', async () => {
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const stream = new PassThrough();
  let options: MusicPlaybackOptions | undefined;
  let fail = false;
  const queue = new GuildQueue('speed-failure', client, (_url, settings) => {
    if (fail) throw new Error('replacement failed');
    options = settings;
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  try {
    queue.setSpeed(0.8);
    await queue.addTrack(track('aaaaaaaaaaa'));
    assert.deepEqual(options, { speed: 0.8, startSeconds: 0 });
    const original = queue.player.state;
    fail = true;
    assert.throws(() => queue.setSpeed(1.25), /replacement failed/);
    assert.equal(queue.speed, 0.8); assert.equal(queue.player.state, original);
    assert.equal(stream.destroyed, false);
    for (const value of [NaN, Infinity, 0.4, 2.1]) assert.throws(() => queue.setSpeed(value));
  } finally { await queue.stop(); }
  assert.throws(() => queue.setSpeed(1), /остановлена/);
});

test('manual track added while recommendations load discards stale recommendations and plays the manual track', async () => {
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
    assert.deepEqual(queue.tracks.map((item) => item.videoId), []);
    assert.equal(streams[0]?.destroyed, true);
  } finally {
    resolveRelated?.([]);
    await queue.stop();
  }
});

test('autoplay clears recommendations from old track when a new track is queued or played, and picks from the new track', async () => {
  const calls: string[] = [];
  const suggestions: Record<string, Track[]> = {
    aaaaaaaaaaa: [{ ...track('rec_a1'), isAutoplay: true }, { ...track('rec_a2'), isAutoplay: true }],
    bbbbbbbbbbb: [{ ...track('rec_b1'), isAutoplay: true }, { ...track('rec_b2'), isAutoplay: true }],
  };
  const client = {
    queues: new Map(),
    ytdlp: { related: async (id: string) => { calls.push(id); return suggestions[id] ?? []; } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-autoplay-switch', client, () => {
    const stream = new PassThrough();
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    // aaaaaaaaaaa ends, recommendations for aaaaaaaaaaa load
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === 'rec_a1');
    assert.equal(calls[0], 'aaaaaaaaaaa');
    assert.deepEqual(queue.tracks.map((t) => t.videoId), ['rec_a2']);

    // User switches to new song bbbbbbbbbbb
    await queue.addTrack(track('bbbbbbbbbbb'));
    // Old recommendations must be cleared, leaving only bbbbbbbbbbb in queue
    assert.deepEqual(queue.tracks.map((t) => t.videoId), ['bbbbbbbbbbb']);

    // Skip current autoplay song rec_a1 to play bbbbbbbbbbb
    queue.skip();
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.equal(queue.tracks.length, 0);

    // bbbbbbbbbbb ends, bot must now pick from bbbbbbbbbbb, NOT aaaaaaaaaaa!
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === 'rec_b1');
    assert.equal(calls[1], 'bbbbbbbbbbb');
    assert.deepEqual(queue.tracks.map((t) => t.videoId), ['rec_b2']);
  } finally {
    await queue.stop();
  }
});

test('autoplay filters recently played recommendations and tries up to two recent seed tracks', async () => {
  const streams: PassThrough[] = [];
  const calls: string[] = [];
  const suggestions: Record<string, Track[]> = {
    ccccccccccc: [track('aaaaaaaaaaa'), track('bbbbbbbbbbb')],
    bbbbbbbbbbb: [track('aaaaaaaaaaa')],
    aaaaaaaaaaa: [{ ...track('ddddddddddd'), isAutoplay: true }, { ...track('ddddddddddd'), isAutoplay: true }],
  };
  const client = {
    queues: new Map(),
    ytdlp: { related: async (id: string) => { calls.push(id); return suggestions[id] ?? []; } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-autoplay-dedup', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb'), track('ccccccccccc')]);
    for (const id of ['bbbbbbbbbbb', 'ccccccccccc']) {
      queue.player.stop(true);
      await waitFor(() => queue.currentTrack?.videoId === id);
    }
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === 'ddddddddddd');
    assert.deepEqual(calls, ['ccccccccccc', 'bbbbbbbbbbb', 'aaaaaaaaaaa']);
    assert.equal(queue.tracks.length, 0);
    assert.equal(streams.length, 4);
  } finally { await queue.stop(); }
});

test('autoplay history keeps only the latest 500 unique tracks', async () => {
  const streams: PassThrough[] = [];
  const ids = Array.from({ length: 501 }, (_, index) => `video${String(index + 1).padStart(6, '0')}`);
  const calls: string[] = [];
  const client = {
    queues: new Map(),
    ytdlp: { related: async (id: string) => {
      calls.push(id);
      return [{ ...track(ids[0]!), isAutoplay: true }, { ...track(ids[1]!), isAutoplay: true }];
    } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-autoplay-history', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    await queue.addTracks(ids.map((id) => track(id)));
    for (const id of ids.slice(1)) {
      queue.player.stop(true);
      await waitFor(() => queue.currentTrack?.videoId === id);
    }
    queue.setAutoplay(true);
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === ids[0]);
    assert.deepEqual(calls, [ids.at(-1)]);
    assert.equal(streams.length, 502);
  } finally { await queue.stop(); }
});

test('autoplay leaves the queue idle when every recommendation is in recent history', async () => {
  const calls: string[] = [];
  const client = {
    queues: new Map(),
    ytdlp: { related: async (id: string) => { calls.push(id); return [track(id)]; } },
  } as unknown as MusicClient;
  const queue = new GuildQueue('guild-autoplay-exhausted', client, () => {
    const stream = new PassThrough();
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack === null);
    assert.deepEqual(calls, ['aaaaaaaaaaa']);
    assert.equal(queue.tracks.length, 0);
  } finally { await queue.stop(); }
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

test('ducking preserves user volume and applies to subsequent tracks after immediate skip', async () => {
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-duck', client, () => {
    const stream = new PassThrough();
    return { stream, type: StreamType.OggOpus, destroy: () => stream.destroy() };
  });
  const gain = () => queue.player.state.status === AudioPlayerStatus.Idle ? null : queue.player.state.resource.volume?.volume;
  try {
    await queue.addTrack(track('aaaaaaaaaaa'));
    queue.setVolume(100); queue.setVoiceDucking(true);
    assert.equal(queue.volume, 1); assert.equal(gain(), 0.2);
    queue.setVolume(150); assert.ok(Math.abs(gain()! - 0.3) < 0.00001);
    queue.setVoiceDucking(false); assert.equal(gain(), 1.5);
    queue.setVoiceDucking(true);
    await queue.addTrack(track('bbbbbbbbbbb')); queue.skip();
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.ok(Math.abs(gain()! - 0.3) < 0.00001);
    queue.setVoiceDucking(false); assert.equal(gain(), 1.5);
  } finally { await queue.stop(); }
});

test('voice cue keeps the current resource and queue; cancellation restores subscription', async () => {
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  let destroyed = 0;
  const queue = new GuildQueue('guild-cue', client, () => {
    const stream = new PassThrough();
    return { stream, type: StreamType.OggOpus, destroy: () => { destroyed++; stream.destroy(); } };
  });
  const outputs: AudioPlayer[] = [];
  const connection = { state: { status: VoiceConnectionStatus.Ready },
    subscribe: (player: AudioPlayer) => { outputs.push(player); },
    destroy: () => { connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  queue.connection = connection;
  try {
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    const original = queue.player.state;
    const signal = new AbortController();
    const cue = queue.playVoiceCue(signal.signal);
    const rejected = assert.rejects(cue);
    assert.notEqual(outputs[0], queue.player);
    signal.abort(); await rejected;
    assert.equal(outputs.at(-1), queue.player);
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(queue.tracks.length, 1); assert.equal(destroyed, 0);
    if (original.status !== AudioPlayerStatus.Idle && queue.player.state.status !== AudioPlayerStatus.Idle) {
      assert.equal(original.resource, queue.player.state.resource);
    }
  } finally { await queue.stop(); }
});

test('audible cue preserves playing and paused resources, which can subsequently resume', async () => {
  for (const mode of ['playing', 'pause-during-cue', 'already-paused']) {
    const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
    const raw = new PassThrough();
    const queue = new GuildQueue(`guild-cue-${mode}`, client, () => ({ stream: raw,
      type: StreamType.Raw, destroy: () => raw.destroy() }));
    // Simulate the network side, while using the real AudioPlayer and Opus encoder.
    let subscription: { unsubscribe(): void } | undefined;
    let packets = 0;
    const connection = {
      state: { status: VoiceConnectionStatus.Ready },
      subscribe: (player: AudioPlayer) => {
        subscription?.unsubscribe();
        subscription = Reflect.apply(Reflect.get(player, 'subscribe'), player, [connection]) as { unsubscribe(): void };
        return subscription;
      },
      onSubscriptionRemoved: () => {}, setSpeaking: () => {},
      prepareAudioPacket: () => { packets++; }, dispatchAudio: () => true,
      destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
    } as unknown as VoiceConnection;
    queue.connection = connection; connection.subscribe(queue.player);
    try {
      await queue.addTrack(track('aaaaaaaaaaa'));
      raw.write(Buffer.alloc(48_000 * 4 * 3));
      await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
      if (mode === 'already-paused') assert.equal(queue.pause(), true);
      const before = queue.player.state;
      assert.ok(before.status !== AudioPlayerStatus.Idle);
      const startPackets = packets;
      const cue = queue.playVoiceCue(new AbortController().signal);
      if (mode === 'pause-during-cue') assert.equal(queue.pause(), true);
      await cue;
      assert.ok(packets > startPackets, 'cue produced actual Opus packets');
      assert.equal(queue.player.state.status, mode !== 'playing' ? AudioPlayerStatus.Paused : AudioPlayerStatus.Playing);
      assert.equal(queue.player.state.resource, before.resource);
      assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
      if (mode !== 'playing') {
        const resumePackets = packets;
        assert.equal(queue.resume(), true);
        await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing && packets > resumePackets);
        assert.equal(queue.player.state.resource, before.resource);
      }
    } finally { await queue.stop(); }
  }
});

test('next track is prefetched when remaining duration is under 20s, and advance uses prepared media', async () => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-prefetch', client, () => {
    const stream = new PassThrough();
    streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
  client.queues.set(queue.guildId, queue);
  try {
    const track1: Track = { ...track('aaaaaaaaaaa'), duration: '0:15' };
    const track2: Track = { ...track('bbbbbbbbbbb'), duration: '1:00' };
    await queue.addTrack(track1);
    await queue.addTrack(track2);
    assert.equal(streams.length, 1, 'buffering must not arm preparation');
    streams[0]!.write(Buffer.alloc(3840 * 4));
    await waitFor(() => streams.length >= 2);
    assert.equal(Reflect.get(queue, 'prepared'), null, 'process creation is not audio readiness');
    streams[1]!.write(Buffer.alloc(3840 * 4));
    await waitFor(() => Reflect.get(queue, 'prepared') !== null);
    assert.equal(streams.length, 2, 'Track 2 should have been prefetched while Track 1 is playing');
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    queue.player.stop(true);
    await waitFor(() => queue.currentTrack?.videoId === 'bbbbbbbbbbb');
    assert.equal(streams.length, 2, 'Advance should have used the prefetched stream');
    assert.equal(streams[0]?.destroyed, true);
  } finally {
    await queue.stop();
  }
});

test('background metadata hydration updates track title, duration, and thumbnail without stalling playback', async () => {
  const client = {
    queues: new Map(),
    ytdlp: {
      related: async () => [],
      video: async () => ({
        id: 'fastvideo111',
        title: 'Hydrated Title',
        duration: 185,
        thumbnail: 'https://img.youtube.com/vi/fastvideo111/maxresdefault.jpg',
      }),
    },
  } as unknown as MusicClient;
  const raw = new PassThrough();
  const queue = new GuildQueue('guild-hydrate', client, () => ({
    stream: raw, type: StreamType.Raw, destroy: () => raw.destroy(),
  }));
  client.queues.set(queue.guildId, queue);
  try {
    const fastTrack: Track = {
      videoId: 'fastvideo111',
      url: 'https://www.youtube.com/watch?v=fastvideo111',
      title: 'Загрузка…',
      duration: '?',
      thumbnail: null,
      requestedBy: 'Tester',
    };
    await queue.addTrack(fastTrack);
    assert.equal(queue.currentTrack?.videoId, 'fastvideo111');
    await waitFor(() => queue.currentTrack?.title === 'Hydrated Title');
    assert.equal(queue.currentTrack?.duration, '3:05');
    assert.equal(queue.currentTrack?.thumbnail, 'https://img.youtube.com/vi/fastvideo111/maxresdefault.jpg');
  } finally {
    await queue.stop();
  }
});

for (const speed of [0.5, 2]) test(`prefetch uses real remaining time at ${speed}x and pause cancels pending media`, async context => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue(`prefetch-speed-${speed}`, client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
  try {
    queue.setSpeed(speed);
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    streams[0]!.write(Buffer.alloc(3840 * 5));
    await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
    queue.pause();
    const resource = Reflect.get(queue, 'activeResource'); resource.playbackDuration = 0;
    context.mock.timers.enable({ apis: ['setTimeout'] });
    queue.resume();
    const delay = (60 / speed - 20) * 1000;
    context.mock.timers.tick(delay - 1); assert.equal(streams.length, 1);
    resource.playbackDuration = delay;
    context.mock.timers.tick(1); assert.equal(streams.length, 2);
    assert.equal(Reflect.get(queue, 'prepared'), null, 'must wait for first PCM');
    queue.pause(); await Promise.resolve();
    assert.equal(streams[1]!.destroyed, true);
    assert.equal(Reflect.get(queue, 'prepared'), null);
  } finally { await queue.stop(); context.mock.timers.reset(); }
});

test('prefetch rechecks consumed audio when the scheduled deadline follows a stall', async context => {
  const streams: PassThrough[] = [];
  const queue = new GuildQueue('prepare-stall-time', { queues: new Map() } as unknown as MusicClient, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
  try {
    await queue.addTracks([track('aaaaaaaaaaa'), track('bbbbbbbbbbb')]);
    streams[0]!.write(Buffer.alloc(3840 * 5));
    await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
    queue.pause();
    const resource = Reflect.get(queue, 'activeResource'); resource.playbackDuration = 0;
    context.mock.timers.enable({ apis: ['setTimeout'] });
    queue.resume();
    context.mock.timers.tick(40_000);
    assert.equal(streams.length, 1, 'a wall-clock deadline alone cannot prepare early');
    resource.playbackDuration = 40_000;
    context.mock.timers.tick(40_000);
    assert.equal(streams.length, 2);
  } finally { await queue.stop(); context.mock.timers.reset(); }
});

test('stale and failed prepared streams are discarded and skip starts a fresh stream immediately', async () => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('prepare-stale', client, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
  try {
    await queue.addTracks([{ ...track('aaaaaaaaaaa'), duration: '0:15' }, track('bbbbbbbbbbb')]);
    streams[0]!.write(Buffer.alloc(3840 * 5));
    await waitFor(() => streams.length === 2);
    streams[1]!.destroy(new Error('prefetch failed'));
    await waitFor(() => Reflect.get(queue, 'prefetchPromise') === null);
    const started = performance.now(); queue.skip();
    assert.equal(queue.currentTrack?.videoId, 'bbbbbbbbbbb');
    assert.ok(performance.now() - started < 200);
    assert.equal(streams.length, 3);
    // Prepare another track then invalidate it while it is still buffering.
    await queue.addTrack(track('ccccccccccc'));
    const task = Reflect.apply(Reflect.get(queue, 'triggerPrefetch'), queue, [Reflect.get(queue, 'playbackEpoch')]);
    const stale = streams.at(-1)!;
    queue.clearQueue(); await task;
    assert.equal(stale.destroyed, true); assert.equal(Reflect.get(queue, 'prepared'), null);
  } finally { await queue.stop(); }
});

test('a prepared stream closed without an error falls back to the same next track', async () => {
  const streams: PassThrough[] = [];
  const queue = new GuildQueue('prepare-closed', { queues: new Map() } as unknown as MusicClient, () => {
    const stream = new PassThrough(); streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
  try {
    await queue.addTracks([{ ...track('aaaaaaaaaaa'), duration: '0:15' }, track('bbbbbbbbbbb')]);
    streams[0]!.write(Buffer.alloc(3840 * 5));
    await waitFor(() => streams.length === 2);
    streams[1]!.write(Buffer.alloc(3840));
    await waitFor(() => Reflect.get(queue, 'prepared') !== null);
    streams[1]!.destroy();
    queue.skip();
    assert.equal(queue.currentTrack?.videoId, 'bbbbbbbbbbb');
    assert.equal(streams.length, 3);
  } finally { await queue.stop(); }
});
