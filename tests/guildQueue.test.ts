import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { StreamType, NoSubscriberBehavior } from '@discordjs/voice';
import { AudioPlayerStatus, VoiceConnectionStatus, entersState, type AudioPlayer, type VoiceConnection } from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { MusicClient, Track } from '../src/types.js';
import type { ManagedAudioStream } from '../src/utils/stream.js';
import type { MusicPlaybackOptions } from '../src/utils/audio.js';
import { VoiceDecoder } from '../src/voice/decoder.js';
import type { GuildVoice } from '../src/voice/GuildVoice.js';
import { radioStations } from '../src/utils/radio.js';

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

test('the expected initial bot join preserves a pending search, departure and moves cancel it', async () => {
  const queue = new GuildQueue('guild', { queues: new Map(), user: { id: 'bot' } } as unknown as MusicClient);
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  try {
    const first = queue.beginMetadataRequest('first'), second = queue.beginMetadataRequest('second');
    queue.onVoiceChannelChange('bot', null, 'voice');
    assert.equal(first.signal.aborted, false);
    queue.onVoiceChannelChange('first', 'voice', null);
    assert.equal(first.signal.aborted, true);
    assert.equal(second.signal.aborted, false);
    queue.onVoiceChannelChange('bot', 'voice', 'other');
    assert.equal(second.signal.aborted, true);
  } finally { await queue.stop(); }
});

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
    assert.equal(queue.tracks.length, 0);

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
    assert.equal(queue.tracks.length, 0);
  } finally {
    await queue.stop();
  }
});

test('autoplay follows the first recommendation from each newly played song', async () => {
  const calls: string[] = [];
  const suggestions: Record<string, string[]> = {
    aaaaaaaaaaa: ['bbbbbbbbbbb', 'ccccccccccc'],
    bbbbbbbbbbb: ['ddddddddddd', 'eeeeeeeeeee'],
    ddddddddddd: ['fffffffffff'],
  };
  const client = { queues: new Map(), ytdlp: { related: async (id: string, limit: number) => {
    calls.push(id);
    assert.equal(limit, 1);
    return (suggestions[id] ?? []).map(id => ({ ...track(id), isAutoplay: true }));
  } } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-autoplay-chain', client, () => {
    const stream = new PassThrough();
    return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  client.queues.set(queue.guildId, queue);
  try {
    queue.setAutoplay(true);
    await queue.addTrack(track('aaaaaaaaaaa'));
    for (const id of ['bbbbbbbbbbb', 'ddddddddddd', 'fffffffffff']) {
      queue.player.stop(true);
      await waitFor(() => queue.currentTrack?.videoId === id);
      assert.equal(queue.tracks.length, 0);
    }
    assert.deepEqual(calls, ['aaaaaaaaaaa', 'bbbbbbbbbbb', 'ddddddddddd']);
  } finally { await queue.stop(); }
});

test('autoplay filters recently played recommendations without switching to older seed tracks', async () => {
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
    await waitFor(() => queue.currentTrack === null);
    assert.deepEqual(calls, ['ccccccccccc']);
    assert.equal(queue.tracks.length, 0);
    assert.equal(streams.length, 3);
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

test('the wake cue plays two rising notes with a silent gap as one audio clip', async () => {
  const queue = new GuildQueue('guild-wake-notes', { queues: new Map() } as unknown as MusicClient);
  const clips: Buffer[] = [];
  queue.playVoiceAudio = async pcm => { clips.push(pcm); };
  try {
    await queue.playVoiceCue(new AbortController().signal);
    assert.equal(clips.length, 1);
    const pcm = clips[0]!;
    const cycles = (startMs: number, endMs: number): number => {
      let count = 0;
      for (let frame = startMs * 48; frame < endMs * 48; frame++) {
        const offset = frame * 4;
        if (pcm.readInt16LE(offset - 4) <= 0 && pcm.readInt16LE(offset) > 0) count++;
      }
      return count;
    };
    const first = cycles(20, 100), second = cycles(180, 260);
    assert.ok(first > 0 && second > first);
    assert.ok(pcm.subarray(120 * 48 * 4, 160 * 48 * 4).every(sample => sample === 0));
  } finally { await queue.stop(); }
});

test('voice cue preserves the track on cancellation and reports interruption on a manual skip', async () => {
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
    const interrupted = assert.rejects(queue.playVoiceCue(new AbortController().signal), /interrupted/);
    assert.equal(queue.skip(), true);
    await interrupted;
    assert.equal(outputs.at(-1), queue.player);
  } finally { await queue.stop(); }
});

// Simulate the network side, while using the real AudioPlayer and Opus encoder.
function attachPlayback(queue: GuildQueue, onPacket?: (source: 'music' | 'cue', packet: Buffer) => void) {
  let subscription: { unsubscribe(): void } | undefined;
  let output: AudioPlayer;
  const packets = { music: 0, cue: 0 };
  const connection = {
    state: { status: VoiceConnectionStatus.Ready },
    subscribe: (player: AudioPlayer) => {
      subscription?.unsubscribe();
      output = player;
      subscription = Reflect.apply(Reflect.get(player, 'subscribe'), player, [connection]) as { unsubscribe(): void };
      return subscription;
    },
    onSubscriptionRemoved: () => {}, setSpeaking: () => {},
    prepareAudioPacket: (packet: Buffer) => {
      const source = output === queue.player ? 'music' : 'cue';
      packets[source]++;
      onPacket?.(source, packet);
    }, dispatchAudio: () => true,
    destroy: () => { subscription?.unsubscribe(); connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  queue.connection = connection;
  connection.subscribe(queue.player);
  return packets;
}

function playbackTone(frames = 50): Buffer {
  const pcm = Buffer.alloc(3840 * frames);
  for (let sample = 0; sample < pcm.length / 4; sample++) {
    const value = Math.round(6000 * Math.sin(sample * 2 * Math.PI * 440 / 48_000));
    pcm.writeInt16LE(value, sample * 4);
    pcm.writeInt16LE(value, sample * 4 + 2);
  }
  return pcm;
}

test('echo references follow consumed music, cues, voice clips and radio, and release every decoder', async context => {
  const decoders = new Set<VoiceDecoder>(), deleted = new Set<VoiceDecoder>();
  const decodedPackets: Buffer[] = [], references: Buffer[] = [];
  const decode = VoiceDecoder.prototype.decode, dispose = VoiceDecoder.prototype.delete;
  context.mock.method(VoiceDecoder.prototype, 'decode', function(this: VoiceDecoder, packet: Buffer) {
    decoders.add(this); decodedPackets.push(Buffer.from(packet));
    return decode.call(this, packet);
  });
  context.mock.method(VoiceDecoder.prototype, 'delete', function(this: VoiceDecoder) {
    deleted.add(this); dispose.call(this);
  });
  const raw = new PassThrough();
  const queue = new GuildQueue('guild-playback-reference', { queues: new Map() } as unknown as MusicClient,
    () => ({ stream: raw, type: StreamType.Raw, destroy: () => raw.destroy() }), {
      factory: () => {
        const stream = new PassThrough(); stream.write(playbackTone());
        return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
      },
    });
  queue.volume = 0.25;
  const voice = { enabled: true, observePlayback: (pcm: Buffer) => { references.push(pcm); },
    destroy: () => { voice.enabled = false; } };
  queue.voice = voice as unknown as GuildVoice;
  const silencePadding = Buffer.from([0xf8, 0xff, 0xfe]);
  let matched = 0;
  const packets = attachPlayback(queue, (_source, packet) => {
    if (packet.equals(silencePadding)) return;
    assert.equal(references.length, ++matched, 'buffered source audio must not be observed before its packet is played');
    assert.deepEqual(decodedPackets.at(-1), packet, 'reference is decoded from the post-volume outgoing Opus packet');
    assert.equal(references.at(-1)?.length, 640, 'the playback reference uses 16 kHz mono PCM');
  });
  try {
    await queue.addTrack(track('aaaaaaaaaaa'));
    assert.equal(queue.player.state.status, AudioPlayerStatus.Buffering);
    assert.equal(references.length, 0);
    raw.write(playbackTone(150));
    await waitFor(() => references.length >= 5);
    const steady = references[4]!;
    let energy = 0;
    for (let i = 0; i < steady.length; i += 2) energy += steady.readInt16LE(i) ** 2;
    const rms = Math.sqrt(energy / (steady.length / 2));
    assert.ok(rms > 500 && rms < 1800, 'the reference includes the selected 25% playback volume');
    assert.equal(decoders.size, 1);

    assert.equal(queue.pause(), true);
    const pausedReferences = references.length;
    const original = queue.player.state;
    const pausedDuration = original.resource.playbackDuration;
    await new Promise(resolve => setTimeout(resolve, 80));
    assert.equal(references.length, pausedReferences, 'paused music and its queued PCM produce no reference');
    assert.equal(original.resource.playbackDuration, pausedDuration);

    await queue.playVoiceCue(new AbortController().signal);
    assert.ok(references.length > pausedReferences);
    assert.equal(decoders.size, 2, 'the cue has its own Opus decoder state');
    assert.equal(original.resource.playbackDuration, pausedDuration, 'music is not observed while the cue owns the connection');
    await queue.playVoiceAudio(playbackTone(8), new AbortController().signal);
    assert.equal(decoders.size, 3, 'arbitrary TTS/effect clips are observed too');
    assert.equal(queue.isPaused, true);

    const beforeResume = references.length;
    assert.equal(queue.resume(), true);
    await waitFor(() => references.length > beforeResume);
    assert.equal(decoders.size, 3, 'resuming the same track preserves its decoder');
    const beforeRadio = references.length;
    await queue.playRadio(radioStations[0]!, 'Tester');
    await waitFor(() => references.length >= beforeRadio + 3);
    assert.equal(queue.isRadio, true);
    assert.equal(decoders.size, 4, 'radio uses the same playback observer');
    assert.ok(packets.music > 0 && packets.cue > 0);
  } finally {
    await queue.stop();
    await new Promise<void>(resolve => setImmediate(resolve));
  }
  assert.deepEqual(deleted, decoders, 'finished, replaced and stopped resources release their private decoder');
});

test('playback observation can start mid-track and failures never interrupt audible playback', async () => {
  const raw = new PassThrough();
  const queue = new GuildQueue('guild-reference-toggle', { queues: new Map() } as unknown as MusicClient,
    () => ({ stream: raw, type: StreamType.Raw, destroy: () => raw.destroy() }));
  let observed = 0, fail = false;
  const voice = { enabled: false, observePlayback: () => { observed++; if (fail) throw new Error('fixture observer failure'); },
    destroy: () => { voice.enabled = false; } };
  queue.voice = voice as unknown as GuildVoice;
  const packets = attachPlayback(queue);
  try {
    await queue.addTrack(track('aaaaaaaaaaa')); raw.write(playbackTone(150));
    await waitFor(() => packets.music >= 3);
    assert.equal(observed, 0, 'voice off does not decode playback');
    voice.enabled = true;
    await waitFor(() => observed >= 3);
    voice.enabled = false;
    const beforeDisable = observed, disabledPackets = packets.music;
    await waitFor(() => packets.music >= disabledPackets + 3);
    assert.equal(observed, beforeDisable);
    voice.enabled = true; fail = true;
    const beforeFailure = packets.music;
    await waitFor(() => packets.music >= beforeFailure + 3);
    assert.ok(observed > beforeDisable, 're-enabling voice attaches to the live playback position');
    assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(queue.player.state.status, AudioPlayerStatus.Playing);
    fail = false;
    const beforeRecovery = observed;
    await waitFor(() => observed >= beforeRecovery + 3);
  } finally { await queue.stop(); }
});

test('music and wake cues honour the channel bitrate and stay at 128 kbps when the limit is higher or missing', async () => {
  // Independent, repeatable stereo noise keeps the VBR encoder busy at both bitrates.
  const pcm = Buffer.alloc(3840 * 50);
  let seed = 23;
  for (let offset = 0; offset < pcm.length; offset += 2) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    pcm.writeInt16LE(Math.round(((seed >>> 16) - 32768) * 0.4), offset);
  }
  const silencePadding = Buffer.from([0xf8, 0xff, 0xfe]);
  const reference = { music: 0, cue: 0 };
  for (const cap of [128_000, 64_000, 256_000, undefined]) {
    const raw = new PassThrough();
    const queue = new GuildQueue('guild-opus-bitrate', { queues: new Map() } as unknown as MusicClient,
      () => ({ stream: raw, type: StreamType.Raw, destroy: () => raw.destroy() }));
    queue.voiceChannel = { id: 'voice', bitrate: cap } as VoiceBasedChannel;
    const sizes: Record<'music' | 'cue', number[]> = { music: [], cue: [] };
    attachPlayback(queue, (source, packet) => {
      if (!packet.equals(silencePadding)) sizes[source].push(packet.length);
    });
    try {
      await queue.addTrack(track('aaaaaaaaaaa'));
      raw.write(pcm);
      await waitFor(() => sizes.music.length >= 20);
      await queue.playVoiceCue(new AbortController().signal);
      for (const source of ['music', 'cue'] as const) {
        const frames = sizes[source].slice(0, 20);
        assert.ok(frames.length >= 8, `expected audible ${source} packets`);
        const bitsPerSecond = frames.reduce((sum, bytes) => sum + bytes, 0) * 8 / (frames.length * 0.02);
        if (cap === 128_000) reference[source] = bitsPerSecond;
        const expected = cap === 64_000 ? reference[source] / 2 : reference[source];
        assert.ok(reference[source] > 95_000 && reference[source] < 160_000);
        assert.ok(Math.abs(bitsPerSecond - expected) < expected * 0.22, `${source}: ${bitsPerSecond} bps at limit ${cap}`);
      }
    } finally { await queue.stop(); }
  }
});

test('audible cue preserves playing and paused resources, which can subsequently resume', async () => {
  for (const mode of ['playing', 'pause-during-cue', 'already-paused']) {
    const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
    const raw = new PassThrough();
    const queue = new GuildQueue(`guild-cue-${mode}`, client, () => ({ stream: raw,
      type: StreamType.Raw, destroy: () => raw.destroy() }));
    const packets = attachPlayback(queue);
    try {
      await queue.addTrack(track('aaaaaaaaaaa'));
      raw.write(Buffer.alloc(48_000 * 4 * 3));
      await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing);
      if (mode === 'already-paused') assert.equal(queue.pause(), true);
      const before = queue.player.state;
      assert.ok(before.status !== AudioPlayerStatus.Idle);
      const cue = queue.playVoiceCue(new AbortController().signal);
      assert.equal(queue.isPaused, mode === 'already-paused', 'the cue must not appear as a user-requested pause');
      if (mode === 'pause-during-cue') assert.equal(queue.pause(), true);
      assert.equal(queue.isPaused, mode !== 'playing');
      await cue;
      assert.ok(packets.cue > 0, 'cue produced actual Opus packets');
      assert.equal(queue.player.state.status, mode !== 'playing' ? AudioPlayerStatus.Paused : AudioPlayerStatus.Playing);
      assert.equal(queue.player.state.resource, before.resource);
      assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
      if (mode !== 'playing') {
        const resumePackets = packets.music;
        assert.equal(queue.resume(), true);
        await waitFor(() => queue.player.state.status === AudioPlayerStatus.Playing && packets.music > resumePackets);
        assert.equal(queue.player.state.resource, before.resource);
      }
    } finally { await queue.stop(); }
  }
});

test('buffering music waits for the cue and honours pause or resume requested during loading', async () => {
  for (const command of ['keep-playing', 'pause', 'resume']) {
    const raw = new PassThrough();
    const queue = new GuildQueue(`guild-loading-cue-${command}`, { queues: new Map() } as unknown as MusicClient,
      () => ({ stream: raw, type: StreamType.Raw, destroy: () => raw.destroy() }));
    // Feed the stream from the cue's first packet. Wall-clock sleeps can miss
    // the entire cue when the test process is descheduled.
    const packets = attachPlayback(queue, source => {
      if (source === 'cue' && packets.cue === 1) raw.write(Buffer.alloc(3840 * 50));
    });
    try {
      await queue.addTrack(track('aaaaaaaaaaa'));
      assert.equal(queue.player.state.status, AudioPlayerStatus.Buffering);
      const held = entersState(queue.player, AudioPlayerStatus.Paused, 3000);
      const cue = queue.playVoiceCue(new AbortController().signal);
      if (command !== 'keep-playing') assert.equal(queue.pause(), true);
      if (command === 'resume') assert.equal(queue.resume(), true);
      await held;
      assert.equal(queue.isPaused, command === 'pause');
      assert.equal(packets.music, 0);
      await cue;
      assert.ok(packets.cue > 0);
      if (command === 'pause') {
        assert.equal(queue.player.state.status, AudioPlayerStatus.Paused);
        assert.equal(queue.resume(), true);
      }
      await waitFor(() => packets.music > 0);
      assert.equal(queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    } finally { await queue.stop(); }
  }
});

test('an automatic track replacement after EOF can finish loading while a wake cue plays', async () => {
  const streams: PassThrough[] = [];
  let requested = false;
  let resolveRelated!: (tracks: Track[]) => void;
  const related = new Promise<Track[]>(resolve => { resolveRelated = resolve; });
  const client = { queues: new Map(), ytdlp: { related: () => { requested = true; return related; } } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-eof-cue', client, () => {
    const stream = new PassThrough();
    streams.push(stream);
    if (streams.length === 2) stream.write(Buffer.alloc(3840 * 50));
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  const packets = attachPlayback(queue, source => {
    if (source === 'cue' && packets.cue === 1) {
      resolveRelated([{ ...track('bbbbbbbbbbb'), duration: '5:00', isAutoplay: true }]);
    }
  });
  try {
    queue.setAutoplay(true);
    await queue.addTrack({ ...track('aaaaaaaaaaa'), duration: '5:00' });
    streams[0]!.end(Buffer.alloc(3840 * 3));
    await waitFor(() => requested);
    assert.equal(queue.player.state.status, AudioPlayerStatus.Idle);
    const musicPackets = packets.music;
    const held = entersState(queue.player, AudioPlayerStatus.Paused, 3000);
    const cue = queue.playVoiceCue(new AbortController().signal).then(() => true, () => false);
    await held;
    assert.equal(packets.music, musicPackets, 'the next track waits until the voice output is released');
    assert.equal(await cue, true, 'an automatic transition must allow both notes to finish');
    assert.ok(packets.cue > 0);
    await waitFor(() => packets.music > musicPackets);
    assert.equal(queue.currentTrack?.videoId, 'bbbbbbbbbbb');
    assert.equal(queue.tracks.length, 0);
  } finally { resolveRelated([]); await queue.stop(); }
});

test('a wake cue preserves the next track prefetched in the last 20 seconds', async () => {
  const streams: PassThrough[] = [];
  const client = { queues: new Map(), ytdlp: { related: async () => [] } } as unknown as MusicClient;
  const queue = new GuildQueue('guild-prefetch', client, () => {
    const stream = new PassThrough();
    streams.push(stream);
    return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
  });
  attachPlayback(queue);
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
    await queue.playVoiceCue(new AbortController().signal);
    assert.equal(streams[1]!.destroyed, false);
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
    assert.equal(queue.currentTrack?.title, 'Загрузка…');
    Reflect.set(Reflect.get(queue.player, 'behaviors'), 'noSubscriber', NoSubscriberBehavior.Play);
    raw.write(Buffer.alloc(3840 * 5));
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
