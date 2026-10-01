import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { StreamType } from '@discordjs/voice';
import { AudioPlayerStatus, VoiceConnectionStatus, type AudioPlayer, type VoiceConnection } from '@discordjs/voice';
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

test('ducking preserves user volume and applies to subsequent tracks, without interrupting a skip fade', async () => {
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
