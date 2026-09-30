import assert from 'node:assert/strict';
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { StreamType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';
import OpusScript from 'opusscript';
import { GuildVoice } from '../src/voice/GuildVoice.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { VoiceRuntime } from '../src/voice/runtime.js';
import type { MusicClient } from '../src/types.js';
import type { MusicDecision, MusicResultPolicy } from '../src/voice/music.js';
import { toTrack } from '../src/utils/ytdlp.js';

const waitFor = async (condition: () => boolean): Promise<void> => {
  const end = Date.now() + 2000;
  while (!condition()) {
    if (Date.now() > end) throw new Error('voice session did not advance');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
};

function musicHarness() {
  let text = 'Муза';
  const members = new Map([['alice', { displayName: 'Alice' }]]);
  const states = new Map([['bot', { channelId: 'voice' }], ['alice', { channelId: 'voice' }]]);
  const metadata = {
    related: async () => [],
    searchCandidates: async () => [{ id: 'aaaaaaaaaaa', title: 'Numb' }, { id: 'bbbbbbbbbbb', title: 'Numb live' }],
    video: async () => ({ id: 'aaaaaaaaaaa', title: 'Numb' }),
  };
  const client = { queues: new Map(), user: { id: 'bot', username: 'Муза' },
    guilds: { cache: new Map([['music', { members: { cache: members, me: { displayName: 'Муза' } }, voiceStates: { cache: states } }]]) },
    users: { cache: new Map([['alice', { bot: false, username: 'Alice' }]]) }, ytdlp: metadata,
  } as unknown as MusicClient;
  const runtime = Object.assign(new EventEmitter(), { ready: true,
    transcribe: async () => text,
    classify: async () => { throw new Error('Music must not reach player classifier'); },
    decideMusic: async (): Promise<MusicDecision> => ({ next_tool: 'youtube_music_search', query_source: 'message', search_result_policy: 'rerank_results', confidence: 1 }),
    rerankMusic: async () => ({ best_track: 1, confidence: 1 }),
    decideMusicResults: async (): Promise<MusicResultPolicy> => ({ search_result_policy: 'rerank_results', confidence: 1 }),
  });
  const connection = { state: { status: VoiceConnectionStatus.Ready }, joinConfig: {},
    receiver: { speaking: new EventEmitter() }, rejoin: () => true,
    destroy: () => { connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  const queue = new GuildQueue('music', client, () => {
    const stream = new PassThrough(); return { stream, type: StreamType.Opus, destroy: () => stream.destroy() };
  });
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  queue.connection = connection; queue.playVoiceCue = async () => {};
  client.queues.set('music', queue);
  const voice = new GuildVoice(queue, runtime as unknown as VoiceRuntime); queue.voice = voice;
  voice.attach(connection);
  return { queue, voice, runtime, metadata, states, async command(message = 'включи Numb live') {
    text = 'Муза'; await voice.session.begin('alice')!.complete(Buffer.from([0, 0]));
    text = message; const capture = voice.session.begin('alice')!;
    return { capture, processing: capture.complete(Buffer.from([0, 0])) };
  } };
}

test('voice music starts an empty queue, attributes the speaker, and preserves busy playback', async () => {
  for (const busy of [false, true]) {
    const h = musicHarness();
    try {
      if (busy) await h.queue.addTrack(toTrack({ id: 'ccccccccccc', title: 'Existing' }, 'Bob'));
      const { capture, processing } = await h.command();
      await processing; await capture.complete(Buffer.from([0, 0]));
      assert.equal(h.queue.currentTrack?.videoId, busy ? 'ccccccccccc' : 'bbbbbbbbbbb');
      const requested = busy ? h.queue.tracks[0] : h.queue.currentTrack;
      assert.equal(requested?.requestedBy, 'Alice');
      assert.equal(h.queue.tracks.length, busy ? 1 : 0);
      assert.equal(h.voice.session.phase, 'idle');
    } finally { await h.queue.stop(); }
  }
});

test('a plain Moscow request starts or queues once despite a model that would reject the search results', async () => {
  for (const busy of [false, true]) {
    const h = musicHarness();
    let policies = 0;
    h.runtime.decideMusic = async () => ({ next_tool: 'youtube_music_search', query_source: 'message',
      search_result_policy: 'play_first_result', defer_result_policy: true, confidence: 1 });
    h.runtime.decideMusicResults = async () => { policies++; return { search_result_policy: 'no_result', confidence: 0.9966 }; };
    h.metadata.searchCandidates = async () => [{ id: 'aaaaaaaaaaa', title: 'DJ SMASH — MOSCOW NEVER SLEEPS' }];
    try {
      if (busy) await h.queue.addTrack(toTrack({ id: 'ccccccccccc', title: 'Existing' }, 'Bob'));
      const { capture, processing } = await h.command('Включи Moscow never sleep');
      await processing; await capture.complete(Buffer.from([0, 0]));
      const requested = busy ? h.queue.tracks[0] : h.queue.currentTrack;
      assert.equal(requested?.title, 'DJ SMASH — MOSCOW NEVER SLEEPS');
      assert.equal(requested?.requestedBy, 'Alice');
      assert.equal(h.queue.tracks.length, busy ? 1 : 0);
      assert.equal(policies, 0);
      assert.equal(h.voice.session.phase, 'idle');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('worker failure after recognition keeps music fallback alive and detaches listening', async () => {
  const h = musicHarness();
  h.runtime.decideMusic = async () => {
    h.runtime.ready = false; h.runtime.emit('unavailable'); throw new Error('worker stopped');
  };
  try {
    await (await h.command()).processing;
    assert.equal(h.queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(h.voice.enabled, false);
    assert.equal(h.voice.session.phase, 'idle');
  } finally { await h.queue.stop(); }
});

test('leaving, voice off, stop and reset during search, policy or reranking prevent enqueue', async () => {
  for (const stage of ['search', 'policy', 'rerank']) for (const cancel of ['leave', 'off', 'stop', 'reset']) {
    const h = musicHarness();
    let release!: () => void;
    let entered!: () => void;
    const started = new Promise<void>((resolve) => { entered = resolve; });
    const deferred = async () => { entered(); await new Promise<void>((resolve) => { release = resolve; }); };
    if (stage === 'search') h.metadata.searchCandidates = async () => { await deferred(); return [{ id: 'aaaaaaaaaaa', title: 'Numb' }]; };
    else if (stage === 'policy') {
      h.runtime.decideMusic = async () => ({ next_tool: 'youtube_music_search', query_source: 'message',
        search_result_policy: 'play_first_result', defer_result_policy: true, confidence: 1 });
      h.runtime.decideMusicResults = async () => { await deferred(); return { search_result_policy: 'rerank_results', confidence: 1 }; };
    }
    else h.runtime.rerankMusic = async () => { await deferred(); return { best_track: 1, confidence: 1 }; };
    try {
      const { capture, processing } = await h.command(); await started;
      if (cancel === 'leave') { h.states.get('alice')!.channelId = 'other'; h.voice.cancelUser('alice'); }
      else if (cancel === 'off') h.voice.setEnabled(false);
      else if (cancel === 'stop') await h.queue.stop(); else h.voice.reset();
      assert.equal(capture.signal.aborted, true);
      release(); await processing;
      assert.equal(h.queue.currentTrack, null); assert.equal(h.queue.tracks.length, 0);
      assert.equal(h.voice.session.phase, cancel === 'stop' ? 'disabled' : 'idle');
    } finally { release?.(); await h.queue.stop(); }
  }
});

test('receiver decodes real Opus, excludes other channels/bots, and releases subscriptions on off', async () => {
  const streams = new Map<string, PassThrough>();
  const speaking = new EventEmitter();
  const deaf: boolean[] = [];
  const connection = {
    state: { status: VoiceConnectionStatus.Ready }, joinConfig: { channelId: 'voice', selfMute: false, selfDeaf: false },
    receiver: { speaking, subscribe: (id: string) => {
      const stream = new PassThrough({ objectMode: true }); streams.set(id, stream); return stream;
    } },
    rejoin: (config: { selfDeaf: boolean }) => { deaf.push(config.selfDeaf); return true; },
    destroy: () => { connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  const client = { queues: new Map(), user: { id: 'bot', username: 'Муза' },
    guilds: { cache: new Map([['guild-receiver', { members: { me: { displayName: 'Муза' } },
      voiceStates: { cache: new Map([['bot', { channelId: 'voice' }], ['alice', { channelId: 'voice' }],
        ['bob', { channelId: 'other' }], ['otherbot', { channelId: 'voice' }]]) } }]]) },
    users: { cache: new Map([['alice', { bot: false }], ['otherbot', { bot: true }]]) },
    ytdlp: { related: async () => [] },
  } as unknown as MusicClient;
  let transcriptions = 0;
  const runtime = Object.assign(new EventEmitter(), {
    ready: true,
    transcribe: async (pcm: Buffer, _signal: AbortSignal, command: boolean, name?: string) => {
      assert.equal(pcm.length, 640); transcriptions++;
      if (!command) assert.equal(name, 'Муза');
      return command ? 'Громкость шестьдесят процентов' : 'Муза';
    },
    classify: async () => ({ action: 'volume_set', confidence: 1 }),
  }) as unknown as VoiceRuntime;
  const queue = new GuildQueue('guild-receiver', client);
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  queue.connection = connection;
  queue.playVoiceCue = async () => {};
  const voice = new GuildVoice(queue, runtime); queue.voice = voice;
  const encoder = new OpusScript(16_000, 1, OpusScript.Application.VOIP);
  try {
    voice.attach(connection);
    speaking.emit('start', 'bob'); speaking.emit('start', 'otherbot');
    assert.equal(streams.size, 0);
    const packet = encoder.encode(Buffer.alloc(640), 320);
    speaking.emit('start', 'alice'); streams.get('alice')!.end(packet);
    await waitFor(() => voice.session.phase === 'awaiting');
    speaking.emit('start', 'alice'); streams.get('alice')!.end(packet);
    await waitFor(() => voice.session.phase === 'idle');
    assert.equal(queue.volume, 0.6); assert.equal(transcriptions, 2);
    speaking.emit('start', 'alice');
    const capture = streams.get('alice')!;
    voice.setEnabled(false);
    assert.equal(capture.destroyed, true); assert.equal(deaf.at(-1), true);
    assert.equal(speaking.listenerCount('start'), 0);
    assert.equal(runtime.listenerCount('unavailable'), 1);
  } finally { encoder.delete(); await queue.stop(); }
  assert.equal(runtime.listenerCount('unavailable'), 0);
});
