import assert from 'node:assert/strict';
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { AudioReceiveStream, StreamType, VoiceConnectionStatus, type VoiceConnection } from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';
import OpusScript from 'opusscript';
import { GuildVoice } from '../src/voice/GuildVoice.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { VoiceRuntime } from '../src/voice/runtime.js';
import type { MusicClient, Track } from '../src/types.js';
import type { MusicDecision, MusicResultPolicy } from '../src/voice/music.js';
import { toTrack } from '../src/utils/ytdlp.js';
import type { ManagedAudioStream } from '../src/utils/stream.js';
import type { VoiceDecision } from '../src/voice/intents.js';
import type { Confirmation } from '../src/voice/tts.js';

const waitFor = async (condition: () => boolean): Promise<void> => {
  const end = Date.now() + 2000;
  while (!condition()) {
    if (Date.now() > end) throw new Error('voice session did not advance');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
};

function musicHarness(radioFactory?: (url: string) => ManagedAudioStream) {
  let text = 'Муза';
  const members = new Map([['alice', { displayName: 'Alice' }]]);
  const users = new Map([['alice', { bot: false, username: 'Alice' }], ['bob', { bot: false, username: 'Bob' }]]);
  const states = new Map([['bot', { channelId: 'voice' }], ['alice', { channelId: 'voice' }]]);
  const metadata = {
    related: async (_videoId: string): Promise<Track[]> => [],
    searchCandidates: async () => [{ id: 'aaaaaaaaaaa', title: 'Numb' }, { id: 'bbbbbbbbbbb', title: 'Numb live' }],
    video: async () => ({ id: 'aaaaaaaaaaa', title: 'Numb' }),
  };
  const client = { queues: new Map(), user: { id: 'bot', username: 'Муза' },
    guilds: { cache: new Map([['music', { members: { cache: members, me: { displayName: 'Муза' } }, voiceStates: { cache: states } }]]) },
    users: { cache: users }, ytdlp: metadata,
  } as unknown as MusicClient;
  const runtime = Object.assign(new EventEmitter(), { ready: true,
    transcribe: async () => text,
    classify: async (_text: string): Promise<VoiceDecision> => ({ action: 'unknown', confidence: 1 }),
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
  }, { factory: radioFactory, startupTimeoutMs: 30, retryDelays: [0, 0, 0] });
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  queue.connection = connection; queue.playVoiceCue = async () => {};
  client.queues.set('music', queue);
  const voice = new GuildVoice(queue, runtime as unknown as VoiceRuntime); queue.voice = voice;
  voice.attach(connection);
  return { queue, voice, runtime, metadata, states, users, async command(message = 'включи Numb live') {
    text = 'Муза'; await voice.session.begin('alice')!.complete(Buffer.from([0, 0]));
    text = message; const capture = voice.session.begin('alice')!;
    return { capture, processing: capture.complete(Buffer.from([0, 0])) };
  } };
}

test('voice status reports recent recognition only for the requesting channel member', async context => {
  const h = musicHarness();
  let now = 1000;
  context.mock.method(performance, 'now', () => now);
  h.states.set('bob', { channelId: 'voice' });
  try {
    await h.voice.session.begin('alice')!.complete(Buffer.alloc(640));
    assert.match(h.voice.inputStatus('alice'), /имя распознано/);
    assert.match(h.voice.inputStatus('bob'), /нет сведений/);
    assert.equal(h.voice.inputStatus('absent'), '');
    h.voice.session.cancel();
    h.runtime.transcribe = async () => 'вот';
    await h.voice.session.begin('bob')!.complete(Buffer.alloc(640));
    assert.match(h.voice.inputStatus('bob'), /в расшифровке не найдено имя/);
    h.voice.cancelUser('bob');
    assert.match(h.voice.inputStatus('bob'), /нет сведений/);
    now += 60_000;
    assert.match(h.voice.inputStatus('alice'), /нет сведений/);
    h.voice.setEnabled(false);
    assert.equal(h.voice.inputStatus('alice'), '');
  } finally { await h.queue.stop(); }
});

test('voice off or stop during model loading cannot enable listening afterwards', async () => {
  for (const cancel of ['off', 'stop']) {
    const h = musicHarness(); let ready!: (value: boolean) => void;
    const owners = new Set<object>();
    h.queue.client.voiceRuntime = Object.assign(h.runtime, {
      acquire: async (owner: object) => { owners.add(owner); return new Promise<boolean>(resolve => { ready = resolve; }); },
      release: (owner: object) => { owners.delete(owner); },
    }) as unknown as VoiceRuntime;
    try {
      const loading = h.queue.setVoiceEnabled(true);
      if (cancel === 'off') await h.queue.setVoiceEnabled(false); else await h.queue.stop();
      ready(true);
      assert.equal(await loading, false); assert.equal(owners.size, 0);
      assert.equal(h.queue.voice?.enabled ?? false, false);
    } finally { await h.queue.stop(); }
  }
});

test('voice music starts an empty queue, attributes the speaker, and preserves busy playback', async () => {
  for (const busy of [false, true]) {
    const h = musicHarness();
    try {
      if (busy) await h.queue.addTrack(toTrack({ id: 'ccccccccccc', title: 'Existing' }, 'Bob'));
      const { capture, processing } = await h.command();
      await processing; await capture.complete(Buffer.from([0, 0]));
      assert.equal(h.queue.currentTrack?.videoId, busy ? 'ccccccccccc' : 'aaaaaaaaaaa');
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

test('a song-from request reaches search from conversational and ASR forms and restores volume', async () => {
  for (const message of ['включи песню из Лунтика', 'Вот включи песню из Лунтика',
    'Включи, песню из Лунтика', 'Включить песню из Лунтика']) {
    const h = musicHarness();
    let searches = 0;
    h.metadata.searchCandidates = async (query?: string) => {
      searches++; assert.equal(query, 'песня из Лунтика');
      return [{ id: 'aaaaaaaaaaa', title: 'Песня из Лунтика' }];
    };
    try {
      const { capture, processing } = await h.command(message);
      await processing; await capture.complete(Buffer.from([0, 0]));
      assert.equal(searches, 1, message);
      assert.equal(h.queue.currentTrack?.title, 'Песня из Лунтика');
      assert.equal(h.queue.currentTrack?.requestedBy, 'Alice');
      assert.equal(h.voice.session.phase, 'idle');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('worker failure after recognition keeps music fallback alive and detaches listening', async () => {
  const h = musicHarness();
  h.metadata.searchCandidates = async () => {
    h.runtime.ready = false; h.runtime.emit('unavailable');
    return [{ id: 'aaaaaaaaaaa', title: 'Numb' }];
  };
  try {
    await (await h.command()).processing;
    assert.equal(h.queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(h.voice.enabled, false);
    assert.equal(h.voice.session.phase, 'idle');
  } finally { await h.queue.stop(); }
});

test('leaving, voice off, stop and reset during recognition or search prevent enqueue', async () => {
  for (const stage of ['recognition', 'search']) for (const cancel of ['leave', 'off', 'stop', 'reset']) {
    const h = musicHarness();
    let release!: () => void;
    let entered!: () => void;
    const started = new Promise<void>((resolve) => { entered = resolve; });
    const deferred = async () => { entered(); await new Promise<void>((resolve) => { release = resolve; }); };
    if (stage === 'search') h.metadata.searchCandidates = async () => { await deferred(); return [{ id: 'aaaaaaaaaaa', title: 'Numb' }]; };
    else {
      const transcribe = h.runtime.transcribe;
      h.runtime.transcribe = async () => { const text = await transcribe(); if (text !== 'Муза') await deferred(); return text; };
    }
    try {
      const { capture, processing } = await h.command(); await started;
      if (cancel === 'leave') { h.states.get('alice')!.channelId = 'other'; h.voice.cancelUser('alice'); }
      else if (cancel === 'off') h.voice.setEnabled(false);
      else if (cancel === 'stop') await h.queue.stop(); else h.voice.reset();
      assert.equal(capture.signal.aborted, true);
      release(); await processing;
      assert.equal(h.queue.currentTrack, null); assert.equal(h.queue.tracks.length, 0);
      assert.equal(h.voice.session.phase, cancel === 'stop' || cancel === 'off' ? 'disabled' : 'idle');
    } finally { release?.(); await h.queue.stop(); }
  }
});

test('short next aliases skip a busy queue exactly once without searching or enqueueing', async () => {
  for (const message of ['Следующее', 'Далее', 'Next', 'Skip', 'Переключи']) {
    const h = musicHarness();
    let skips = 0;
    h.runtime.classify = async text => {
      assert.equal(text, 'Следующий трек');
      return { action: 'skip', confidence: 0.99 };
    };
    h.metadata.searchCandidates = async () => { assert.fail('Skip must not search'); };
    h.queue.skip = () => { skips++; return true; };
    try {
      await h.queue.addTrack(toTrack({ id: 'ccccccccccc', title: 'Existing' }, 'Bob'));
      await h.queue.addTrack(toTrack({ id: 'ddddddddddd', title: 'Already queued' }, 'Bob'));
      const { capture, processing } = await h.command(message);
      await processing; await capture.complete(Buffer.alloc(2));
      assert.equal(skips, 1);
      assert.equal(h.queue.tracks.length, 1);
      assert.equal(h.queue.tracks[0]?.title, 'Already queued');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('voice autoplay and repeat set explicit states, preserve the queue and stay mutually exclusive', async () => {
  const h = musicHarness();
  h.metadata.searchCandidates = async () => { assert.fail('Modes must never search'); };
  h.runtime.classify = async () => { assert.fail('Modes must bypass the old classifier'); };
  try {
    await h.queue.addTracks([toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob'),
      toTrack({ id: 'ddddddddddd', title: 'Manual' }, 'Bob')]);
    const resource = Reflect.get(h.queue, 'activeResource');
    for (const [phrase, autoplay, loop] of [
      ['Включи бесконечный режим', true, false], ['Включи бесконечный режим', true, false],
      ['Включи повтор трека', false, true], ['Включи повтор трека', false, true],
      ['Выключи повтор трека', false, false], ['Включи автоплей', true, false],
      ['Отключи бесконечный режим', false, false],
    ] as const) {
      await (await h.command(phrase)).processing;
      assert.equal(h.queue.autoplay, autoplay, phrase); assert.equal(h.queue.loopCurrent, loop, phrase);
      assert.equal(h.queue.currentTrack?.videoId, 'ccccccccccc');
      assert.deepEqual(h.queue.tracks.map(track => track.videoId), ['ddddddddddd']);
      assert.equal(Reflect.get(h.queue, 'activeResource'), resource);
    }
    await (await h.command('Очисти очередь')).processing;
    assert.equal(h.queue.tracks.length, 0); assert.equal(h.queue.currentTrack?.videoId, 'ccccccccccc');
    assert.equal(Reflect.get(h.queue, 'activeResource'), resource);
  } finally { await h.queue.stop(); }
});

test('spoken infinite mode fetches YouTube recommendations only after manual tracks finish', async () => {
  const h = musicHarness();
  const relatedCalls: string[] = [];
  h.metadata.related = async (...args: unknown[]) => {
    relatedCalls.push(String(args[0]));
    return [toTrack({ id: 'eeeeeeeeeee', title: 'YouTube recommendation' }, 'Auto', true)];
  };
  h.metadata.searchCandidates = async () => { assert.fail('Autoplay is not a search request'); };
  try {
    await h.queue.addTracks([toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob'),
      toTrack({ id: 'ddddddddddd', title: 'Manual' }, 'Bob')]);
    await (await h.command('Включи бесконечный режим')).processing;
    assert.deepEqual(relatedCalls, []);
    h.queue.player.stop(true); await waitFor(() => h.queue.currentTrack?.videoId === 'ddddddddddd');
    assert.deepEqual(relatedCalls, []);
    h.queue.player.stop(true); await waitFor(() => h.queue.currentTrack?.videoId === 'eeeeeeeeeee');
    assert.deepEqual(relatedCalls, ['ddddddddddd']);
  } finally { await h.queue.stop(); }
});

test('spoken voice off disables listening and never enqueues a track', async () => {
  const h = musicHarness();
  h.metadata.searchCandidates = async () => { assert.fail('Voice off must not search'); };
  try {
    await (await h.command('Выключи голосовое управление')).processing;
    assert.equal(h.voice.enabled, false); assert.equal(h.voice.session.phase, 'disabled');
    assert.equal(h.queue.currentTrack, null); assert.equal(h.queue.tracks.length, 0);
    assert.equal(h.voice.session.begin('alice'), null);
    h.voice.setEnabled(true);
    assert.equal(h.voice.enabled, true); assert.equal(h.voice.session.phase, 'idle');
    await (await h.command('Включи бесконечный режим')).processing;
    assert.equal(h.queue.autoplay, true);
  } finally { await h.queue.stop(); }
});

test('worker recovery restores requested listening without changing playback or overriding voice off', async () => {
  for (const requested of [true, false]) {
    const h = musicHarness(); const deaf: boolean[] = [];
    h.queue.connection!.rejoin = (config) => { deaf.push(Boolean(config?.selfDeaf)); return true; };
    try {
      await h.queue.addTrack(toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Alice'));
      await h.queue.addTrack(toTrack({ id: 'ddddddddddd', title: 'Pending' }, 'Alice'));
      const current = h.queue.currentTrack;
      h.voice.setEnabled(requested);
      h.runtime.ready = false; h.runtime.emit('unavailable');
      assert.equal(h.voice.enabled, false); assert.equal(deaf.at(-1), true);
      h.runtime.ready = true; h.runtime.emit('available');
      assert.equal(h.voice.enabled, requested); assert.equal(deaf.at(-1), !requested);
      assert.equal(h.queue.currentTrack, current); assert.equal(h.queue.tracks.length, 1);
      assert.equal(h.queue.closed, false);
      if (requested) {
        await (await h.command('Включи бесконечный режим')).processing;
        assert.equal(h.queue.autoplay, true);
      } else assert.equal(h.voice.session.phase, 'disabled');
    } finally { await h.queue.stop(); }
    assert.equal(h.runtime.listenerCount('available'), 0);
  }
});

test('bare titles and background speech after the cue quietly release the queue', async () => {
  const h = musicHarness();
  let searches = 0;
  const replies: Confirmation[] = [];
  h.metadata.searchCandidates = async () => { searches++; return []; };
  h.queue.client.voiceTts = { audio: key => { replies.push(key); return null; } };
  try {
    const current = toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob');
    await h.queue.addTrack(current);
    for (const phrase of ['Монеточка', 'Moscow Never Sleeps', 'Меня это тоже удивило', '']) {
      await (await h.command(phrase)).processing;
      assert.equal(h.voice.session.phase, 'idle', phrase);
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
      assert.equal(h.queue.currentTrack, current);
      assert.deepEqual(h.queue.tracks, []);
    }
    assert.deepEqual(replies, []);
    assert.equal(searches, 0);
    assert.equal(h.voice.enabled, true);
  } finally { await h.queue.stop(); }
});

test('explicit artist requests search once without music model approval in a voice session', async () => {
  for (const message of ['Включи Монеточку', 'Вот, включи Монеточку', 'Поставь Кино', 'Сыграй песни Монеточки']) {
    const h = musicHarness();
    let searches = 0;
    h.runtime.decideMusic = async () => { assert.fail('Must bypass model routing'); };
    h.metadata.searchCandidates = async () => { searches++; return [{ id: 'aaaaaaaaaaa', title: 'First artist result' }]; };
    try {
      const { capture, processing } = await h.command(message);
      await processing; await capture.complete(Buffer.alloc(2));
      assert.equal(searches, 1);
      assert.equal(h.queue.currentTrack?.title, 'First artist result');
      assert.equal(h.voice.session.phase, 'idle');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('failed search and incomplete music commands get feedback, negated requests end quietly', async () => {
  for (const [message, reply] of [['Включи Монеточку', 'not_found'], ['Включи эту', 'unknown'], ['Не включи Кино', null]] as const) {
    const h = musicHarness();
    const replies: Confirmation[] = [];
    h.metadata.searchCandidates = async () => [];
    h.runtime.classify = async () => ({ action: 'unknown', confidence: 1 });
    h.queue.client.voiceTts = { audio: key => { replies.push(key); return Buffer.alloc(4); } };
    h.queue.playVoiceAudio = async () => {};
    try {
      await (await h.command(message)).processing;
      assert.deepEqual(replies, reply ? [reply] : []);
      assert.equal(h.queue.currentTrack, null);
      assert.equal(h.voice.session.phase, 'idle');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('voice actions confirm once except immediate skip and stop, with a separate queued reply', async () => {
  const h = musicHarness();
  const replies: Confirmation[] = [];
  const pcm = Buffer.alloc(4800);
  h.queue.client.voiceTts = { audio: (key) => { replies.push(key); return pcm; } };
  h.queue.playVoiceAudio = async (audio, signal) => { assert.equal(audio, pcm); assert.equal(signal.aborted, false); };
  try {
    await (await h.command('включи Numb live')).processing;
    await (await h.command('включи песню из Лунтика')).processing;
    assert.deepEqual(replies, ['play', 'queued']);
    h.queue.pause = () => true; h.queue.resume = () => true; h.queue.skip = () => true;
    for (const [message, action] of [['Поставь на паузу', 'pause'], ['Продолжи музыку', 'resume'],
      ['Следующий трек', 'skip'], ['Громкость 70', 'volume_set'], ['Громче', 'volume_up'], ['Тише', 'volume_down']] as const) {
      h.runtime.classify = async () => ({ action, confidence: 1 });
      const before = replies.length;
      const { capture, processing } = await h.command(message);
      await processing; await capture.complete(Buffer.alloc(2));
      if (action === 'skip') assert.equal(replies.length, before);
      else assert.equal(replies.at(-1), action);
    }
    h.queue.pause = () => false;
    const count = replies.length;
    h.runtime.classify = async () => ({ action: 'pause', confidence: 1 });
    await (await h.command('Поставь на паузу')).processing;
    assert.equal(replies.length, count, 'no successful-action reply for a no-op');
    h.runtime.classify = async () => ({ action: 'stop', confidence: 1 });
    await (await h.command('Останови музыку')).processing;
    assert.equal(replies.length, count);
    assert.equal(h.queue.closed, true);
    assert.equal(h.voice.session.phase, 'disabled');
  } finally { await h.queue.stop(); }
});

test('TTS playback failure does not discard accepted music or prevent a stop', async () => {
  const h = musicHarness();
  h.queue.client.voiceTts = { audio: () => Buffer.alloc(4) };
  h.queue.playVoiceAudio = async () => { throw new Error('TTS unavailable'); };
  try {
    await (await h.command()).processing;
    assert.equal(h.queue.currentTrack?.videoId, 'aaaaaaaaaaa');
    assert.equal(h.voice.session.phase, 'idle');
    assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    h.runtime.classify = async () => ({ action: 'stop', confidence: 1 });
    await (await h.command('Останови музыку')).processing;
    assert.equal(h.queue.closed, true);
  } finally { await h.queue.stop(); }
});

test('leaving, voice off, stop and reset during spoken confirmation cancel audio and any delayed action', async () => {
  for (const action of ['play'] as const) for (const cancel of ['leave', 'off', 'stop', 'reset']) {
    const h = musicHarness();
    let entered!: () => void;
    const started = new Promise<void>((resolve) => { entered = resolve; });
    h.queue.client.voiceTts = { audio: () => Buffer.alloc(4) };
    h.queue.playVoiceAudio = async (_pcm, signal) => {
      entered();
      await new Promise<void>((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true }));
    };
    let skips = 0;
    h.queue.skip = () => { skips++; return true; };
    h.runtime.classify = async () => ({ action: 'unknown', confidence: 1 });
    try {
      const { capture, processing } = await h.command('включи Numb');
      await started;
      if (cancel === 'leave') { h.states.get('alice')!.channelId = 'other'; h.voice.cancelUser('alice'); }
      else if (cancel === 'off') h.voice.setEnabled(false);
      else if (cancel === 'stop') await h.queue.stop(); else h.voice.reset();
      await processing; await capture.complete(Buffer.alloc(2));
      assert.equal(capture.signal.aborted, true);
      assert.equal(skips, 0);
      assert.equal(h.queue.closed, cancel === 'stop', 'cancelled voice stop must not execute later');
      assert.equal(h.queue.tracks.length, 0, 'a cancelled confirmation must not enqueue again');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('receiver decodes real Opus, excludes self, bots, unknown users and other channels, and releases on off', async () => {
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
        ['bob', { channelId: 'other' }], ['otherbot', { channelId: 'voice' }], ['unknown', { channelId: 'voice' }]]) } }]]) },
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
    for (const id of ['bot', 'bob', 'otherbot', 'unknown']) speaking.emit('start', id);
    assert.equal(streams.size, 0);
    const pcm = Buffer.alloc(640);
    for (let i = 0; i < 320; i++) pcm.writeInt16LE(Math.round(3000 * Math.sin(i * 2 * Math.PI * 350 / 16_000)), i * 2);
    const packet = encoder.encode(pcm, 320);
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

function openMicHarness() {
  const h = musicHarness();
  const receiver = h.queue.connection!.receiver;
  Object.assign(receiver.speaking, { users: new Map<string, number>() });
  const streams = new Map<string, AudioReceiveStream>();
  const subscriptions = new Map<string, AudioReceiveStream>();
  receiver.subscribe = (id, options) => {
    const existing = subscriptions.get(id);
    if (existing) return existing;
    assert.ok(options?.end);
    const stream = new AudioReceiveStream({ end: options.end });
    subscriptions.set(id, stream);
    stream.once('close', () => subscriptions.delete(id));
    streams.set(id, stream);
    return stream;
  };
  const recognized: { command: boolean; ms: number }[] = [];
  const runtime = h.runtime as unknown as VoiceRuntime;
  runtime.transcribe = async (pcm, _signal, command) => {
    recognized.push({ command, ms: pcm.length / 32 });
    return command ? 'Громкость 70' : 'Муза';
  };
  runtime.classify = async () => ({ action: 'volume_set', confidence: 1 });
  const encoder = new OpusScript(16_000, 1, OpusScript.Application.VOIP);
  const packets = (amplitude: number, hz: number): Buffer[] => {
    const pcm = Buffer.alloc(640);
    return Array.from({ length: 10 }, (_, frame) => {
      for (let i = 0; i < 320; i++) {
        pcm.writeInt16LE(Math.round(amplitude * Math.sin((frame * 320 + i) * 2 * Math.PI * hz / 16_000)), i * 2);
      }
      return Buffer.from(encoder.encode(pcm, 320));
    });
  };
  // Each sender keeps its UDP talkspurt open, including while a soft background
  // tone is the only audio. Only emit the first Discord speaking start explicitly.
  const background = packets(80, 250), foreground = packets(5200, 700);
  const send = async (id: string, frames: Buffer[], count: number): Promise<void> => {
    const stream = streams.get(id)!;
    for (let i = 0; i < count; i++) stream.push(frames[i % frames.length]!);
    await new Promise<void>(resolve => setImmediate(resolve));
  };
  const startSpeaking = (id: string): void => {
    h.states.set(id, { channelId: 'voice' });
    if (!h.users.has(id)) h.users.set(id, { bot: false, username: id });
    receiver.speaking.users.set(id, Date.now());
    receiver.speaking.emit('start', id);
  };
  const open = (id: string): void => {
    startSpeaking(id);
    assert.ok(streams.get(id));
  };
  const phrase = async (id: string): Promise<void> => {
    await send(id, foreground, 18); await send(id, background, 50);
  };
  encoder.delete();
  return { ...h, streams, recognized, background, foreground, send, startSpeaking, open, phrase };
}

test('two open microphones each need the wake name before a command can change playback', async context => {
  const h = openMicHarness();
  const runtime = h.runtime as unknown as VoiceRuntime;
  const calls: boolean[] = [];
  const replies: Confirmation[] = [];
  let transcript = '', cues = 0;
  runtime.supportsWake = () => true;
  runtime.detectWake = async () => ({ wake: true, probability: 0.99 });
  runtime.transcribe = async (pcm, _signal, command) => {
    assert.ok(pcm.length > 320 * 32, 'the receiver must pass the decoded utterance to STT');
    calls.push(command);
    return transcript;
  };
  runtime.classify = async text => ({ action: text.includes('тише') ? 'volume_down' : 'pause', confidence: 1 });
  const speak = async (id: string, text: string) => { transcript = text; await h.phrase(id); };
  const pause = context.mock.method(h.queue, 'pause');
  const volume = context.mock.method(h.queue, 'setVolume');
  h.queue.playVoiceCue = async () => { cues++; };
  h.queue.client.voiceTts = { audio(key) { replies.push(key); return null; } };
  try {
    const track = toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Alice');
    await h.queue.addTrack(track);
    h.open('alice'); h.open('bob');
    await h.send('alice', h.background, 265); await h.send('bob', h.background, 265);
    assert.deepEqual(calls, []);
    await speak('alice', 'Сделай тише');
    await speak('bob', 'Поставь на паузу');
    assert.equal(cues, 0, 'even a positive acoustic hit must be confirmed by the name in the transcript');
    assert.deepEqual(replies, []);
    assert.deepEqual(calls, [false, false]);
    assert.equal(volume.mock.callCount(), 0); assert.equal(pause.mock.callCount(), 0);
    assert.equal(h.queue.volume, 0.5);
    assert.equal(h.queue.isPaused, false);
    await speak('alice', 'Муза');
    assert.equal(cues, 1);
    assert.equal(h.voice.session.awaits('alice'), true);
    await speak('bob', 'Поставь на паузу');
    assert.deepEqual(calls, [false, false, false], 'another speaker cannot answer in the active window');
    await speak('alice', 'Сделай тише');
    assert.equal(h.queue.volume, 0.4);
    assert.equal(h.queue.isPaused, false);
    assert.equal(h.voice.session.phase, 'idle');
    await speak('bob', 'Муза, поставь на паузу');
    assert.deepEqual(calls, [false, false, false, true, false]);
    assert.equal(volume.mock.callCount(), 1); assert.equal(pause.mock.callCount(), 1);
    assert.deepEqual(replies, ['volume_down', 'pause']);
    assert.equal(h.voice.session.phase, 'idle');
    assert.equal(h.queue.currentTrack, track);
    assert.equal(h.queue.isPaused, true);
  } finally { await h.queue.stop(); }
});

test('two Discord microphone streams can each wake even when one scores below the acoustic threshold', async () => {
  const h = openMicHarness();
  const runtime = h.runtime as unknown as VoiceRuntime;
  let confidence = 0.99, cues = 0, transcriptions = 0;
  runtime.supportsWake = () => true;
  runtime.detectWake = async () => ({ wake: confidence > 0.98, probability: confidence });
  runtime.transcribe = async (_pcm, _signal, command) => {
    assert.equal(command, false);
    transcriptions++;
    return 'Муза';
  };
  h.queue.playVoiceCue = async () => { cues++; };
  try {
    h.open('alice'); h.open('bob');
    await h.send('alice', h.background, 10); await h.send('bob', h.background, 10);
    await h.phrase('alice');
    assert.ok(h.voice.session.awaits('alice'));
    h.voice.session.cancel();
    confidence = 0.001;
    await h.phrase('bob');
    assert.ok(h.voice.session.awaits('bob'));
    assert.equal(h.voice.session.begin('alice'), null);
    assert.equal(cues, 2); assert.equal(transcriptions, 2);
  } finally { await h.queue.stop(); }
});

test('a receive failure on the first packet can recover the same wake without a second Discord start', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const h = openMicHarness();
  try {
    h.open('bob');
    const failed = h.streams.get('bob')!;
    failed.destroy(new Error('fixture receive failure'));
    await new Promise<void>(resolve => setImmediate(resolve));
    context.mock.timers.tick(39);
    assert.equal(h.streams.get('bob'), failed);
    context.mock.timers.tick(1);
    assert.notEqual(h.streams.get('bob'), failed);
    await h.phrase('bob');
    assert.equal(h.voice.session.awaits('bob'), true);
    assert.deepEqual(h.recognized.map(turn => turn.command), [false]);
  } finally { await h.queue.stop(); }
});

test('receive retries back off, stop after three attempts, and reset on a new Discord talkspurt', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const h = openMicHarness();
  try {
    h.open('bob');
    for (const backoff of [40, 120, 360]) {
      const failed = h.streams.get('bob')!;
      failed.destroy(new Error('fixture receive failure'));
      await new Promise<void>(resolve => setImmediate(resolve));
      context.mock.timers.tick(backoff - 1);
      assert.equal(h.streams.get('bob'), failed);
      context.mock.timers.tick(1);
      assert.notEqual(h.streams.get('bob'), failed);
    }
    const last = h.streams.get('bob')!;
    last.destroy();
    await new Promise<void>(resolve => setImmediate(resolve));
    context.mock.timers.tick(60_000);
    assert.equal(h.streams.get('bob'), last, 'an unexpected close cannot bypass the retry cap');
    const speaking = h.queue.connection!.receiver.speaking;
    speaking.users.delete('bob');
    speaking.emit('end', 'bob');
    h.startSpeaking('bob');
    assert.notEqual(h.streams.get('bob'), last);
    await h.phrase('bob');
    assert.equal(h.voice.session.awaits('bob'), true);
  } finally { await h.queue.stop(); }
});

test('a receiver that delivered no decryptable packets can recover on fresh speech in the same talkspurt', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const h = openMicHarness();
  try {
    h.open('bob');
    const silent = h.streams.get('bob')!;
    context.mock.timers.tick(4000);
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(silent.destroyed, true);
    assert.equal(h.recognized.length, 0);
    context.mock.timers.tick(40);
    assert.notEqual(h.streams.get('bob'), silent);
    await h.send('bob', h.background, 15); await h.phrase('bob');
    assert.equal(h.voice.session.awaits('bob'), true);
  } finally { await h.queue.stop(); }
});

test('a partial command is discarded on a receive error, and the remaining audio cannot execute', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  const h = openMicHarness();
  try {
    h.open('bob');
    await h.phrase('bob');
    await h.send('bob', h.foreground, 6);
    assert.equal(h.voice.session.phase, 'capturing');
    const failed = h.streams.get('bob')!;
    failed.destroy(new Error('fixture receive failure'));
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(h.voice.session.phase, 'idle');
    context.mock.timers.tick(40);
    assert.notEqual(h.streams.get('bob'), failed);
    await h.send('bob', h.foreground, 20); await h.send('bob', h.background, 50);
    assert.deepEqual(h.recognized.map(turn => turn.command), [false]);
    assert.equal(h.queue.volume, 0.5);
    await h.phrase('bob');
    assert.equal(h.voice.session.awaits('bob'), true);
  } finally { await h.queue.stop(); }
});

test('a pending receive retry is cancelled when speech ends, the user leaves, or listening stops', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  for (const cancel of ['end', 'leave', 'off', 'suspend']) {
    const h = openMicHarness();
    try {
      h.open('bob');
      const failed = h.streams.get('bob')!;
      failed.destroy();
      await new Promise<void>(resolve => setImmediate(resolve));
      if (cancel === 'end') {
        const speaking = h.queue.connection!.receiver.speaking;
        speaking.users.delete('bob'); speaking.emit('end', 'bob');
      }
      if (cancel === 'leave') { h.states.set('bob', { channelId: 'other' }); h.voice.cancelUser('bob'); }
      if (cancel === 'off') h.voice.setEnabled(false);
      if (cancel === 'suspend') h.voice.suspend();
      context.mock.timers.tick(1000);
      assert.equal(h.streams.get('bob'), failed, cancel);
    } finally { await h.queue.stop(); }
  }
});

test('a speaker who opened their mic during someone else’s turn can wake on fresh speech without restarting the mic', async () => {
  const h = openMicHarness();
  try {
    h.open('alice');
    await h.send('alice', h.background, 12); await h.phrase('alice');
    assert.equal(h.voice.session.phase, 'awaiting');
    h.startSpeaking('bob');
    assert.equal(h.streams.has('bob'), false);
    await h.phrase('alice');
    await waitFor(() => h.streams.has('bob'));
    const initialTurns = h.recognized.length;
    await h.send('bob', h.foreground, 32); await h.send('bob', h.background, 50);
    assert.equal(h.recognized.length, initialTurns, 'a resubscription must discard speech already underway');
    await h.phrase('bob');
    assert.equal(h.voice.session.phase, 'awaiting');
    assert.equal(h.voice.session.awaits('bob'), true);
    assert.deepEqual(h.recognized.map(turn => turn.command), [false, true, false]);
  } finally { await h.queue.stop(); }
});

test('attaching while a mic is open waits for fresh speech and ignores users outside the channel', async () => {
  const h = openMicHarness();
  const connection = h.queue.connection!;
  try {
    h.voice.suspend();
    h.startSpeaking('alice'); h.startSpeaking('bob');
    h.states.set('bob', { channelId: 'other' });
    h.voice.attach(connection);
    await waitFor(() => h.streams.has('alice'));
    assert.equal(h.streams.has('bob'), false);
    const previous = h.streams.get('alice')!;
    h.voice.attach(connection);
    await waitFor(() => h.streams.get('alice') !== previous);
    assert.equal(previous.destroyed, true, 'reattaching must replace the recently closed subscription');
    await h.send('alice', h.background, 20); await h.phrase('alice');
    assert.equal(h.voice.session.awaits('alice'), true);
    h.voice.setEnabled(false);
    assert.equal(h.streams.get('alice')!.destroyed, true);
  } finally { await h.queue.stop(); }
});

test('rechecking open microphones honours the TTS tail, departure, suspension and voice off', async context => {
  context.mock.timers.enable({ apis: ['setTimeout'] });
  for (const outcome of ['listen', 'left', 'suspended', 'off']) {
    const h = openMicHarness();
    let audible = true;
    Object.defineProperty(h.queue, 'isSpeakingVoiceAudio', { get: () => audible });
    try {
      h.startSpeaking('bob');
      context.mock.timers.tick(400);
      assert.equal(h.streams.size, 0);
      if (outcome === 'left') { h.states.set('bob', { channelId: 'other' }); h.voice.cancelUser('bob'); }
      if (outcome === 'suspended') h.voice.suspend();
      if (outcome === 'off') h.voice.setEnabled(false);
      audible = false;
      context.mock.timers.tick(400);
      assert.equal(h.streams.has('bob'), outcome === 'listen', outcome);
      assert.deepEqual(h.recognized, []);
    } finally { await h.queue.stop(); }
  }
});

test('soft background through open mics does not consume a wake or require a second speaking start', async () => {
  const h = openMicHarness();
  try {
    h.open('alice'); h.open('bob');
    await h.send('alice', h.background, 265); await h.send('bob', h.background, 80);
    assert.equal(h.recognized.length, 0, 'steady background alone never reaches STT at the wake deadline');
    await h.phrase('alice');
    assert.equal(h.voice.session.phase, 'awaiting');
    assert.ok(h.recognized[0]!.ms < 1200, 'the old background is trimmed before checking the wake');
    await h.phrase('alice');
    assert.equal(h.voice.session.phase, 'idle'); assert.equal(h.queue.volume, 0.7);
    await h.phrase('bob');
    assert.equal(h.voice.session.phase, 'awaiting');
    assert.deepEqual(h.recognized.map(turn => turn.command), [false, true, false]);
    h.voice.cancelUser('bob');
    assert.equal(h.streams.get('bob')!.destroyed, true);
    assert.equal(h.voice.session.phase, 'idle');
    h.voice.setEnabled(false); assert.equal(h.streams.get('alice')!.destroyed, true);
  } finally { await h.queue.stop(); }
});

test('an overlong open-mic command cannot dispatch truncated audio or disable the next wake', async () => {
  const h = openMicHarness();
  try {
    h.open('alice');
    await h.send('alice', h.background, 12); await h.phrase('alice');
    assert.equal(h.voice.session.phase, 'awaiting');
    await h.send('alice', h.foreground, 820);
    assert.equal(h.voice.session.phase, 'idle');
    assert.deepEqual(h.recognized.map(turn => turn.command), [false]);
    await h.send('alice', h.background, 65); await h.phrase('alice');
    assert.equal(h.voice.session.phase, 'awaiting');
    assert.deepEqual(h.recognized.map(turn => turn.command), [false, false]);
  } finally { await h.queue.stop(); }
});


test('only the configured wake name cues its speaker, who can answer while the cue plays', async () => {
  const streams = new Map<string, PassThrough>();
  const speaking = new EventEmitter();
  let cueResolve!: () => void;
  let cueStarted!: () => void;
  const cueStart = new Promise<void>((resolve) => { cueStarted = resolve; });
  const connection = {
    state: { status: VoiceConnectionStatus.Ready }, joinConfig: {},
    receiver: { speaking, subscribe: (id: string) => {
      const stream = new PassThrough({ objectMode: true }); streams.set(id, stream); return stream;
    } },
    rejoin: () => true, destroy: () => {},
  } as unknown as VoiceConnection;
  const client = { queues: new Map(), user: { id: 'bot', username: 'Муза' },
    voiceSettings: { get: () => 'Бот' },
    guilds: { cache: new Map([['guild-cue-capture', { members: { me: { displayName: 'Муза' } },
      voiceStates: { cache: new Map([['bot', { channelId: 'voice' }], ['alice', { channelId: 'voice' }],
        ['bob', { channelId: 'voice' }]]) } }]]) },
    users: { cache: new Map([['alice', { bot: false }], ['bob', { bot: false }]]) },
    ytdlp: { related: async () => [] },
  } as unknown as MusicClient;
  let wakeTranscript = 'Муза';
  const runtime = Object.assign(new EventEmitter(), {
    ready: true,
    detectWake: async () => ({ wake: true, probability: 0.99 }),
    supportsWake: () => true,
    transcribe: async (_pcm: Buffer, _signal: AbortSignal, command: boolean) => {
      return command ? 'Громкость шестьдесят процентов' : wakeTranscript;
    },
    classify: async () => ({ action: 'volume_set', level: 60, confidence: 1 }),
  }) as unknown as VoiceRuntime;
  const queue = new GuildQueue('guild-cue-capture', client);
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  queue.connection = connection;
  let cueCount = 0;
  queue.playVoiceCue = () => {
    cueCount++;
    if (wakeTranscript === 'Муза') return Promise.resolve();
    Reflect.set(queue, 'speakingVoiceAudioUntil', performance.now() + 1000);
    cueStarted();
    return new Promise<void>((resolve) => { cueResolve = resolve; });
  };
  const voice = new GuildVoice(queue, runtime); queue.voice = voice;
  const encoder = new OpusScript(16_000, 1, OpusScript.Application.VOIP);
  try {
    voice.attach(connection);
    await voice.session.begin('alice')!.complete(Buffer.alloc(640, 1));
    assert.equal(cueCount, 0, 'a detector hit on the Discord nickname must be rejected');
    assert.equal(voice.session.phase, 'idle');
    wakeTranscript = 'Бот';
    const packet = encoder.encode(Buffer.alloc(640, 1), 320);
    speaking.emit('start', 'alice');
    streams.get('alice')!.end(packet);
    await cueStart;
    assert.equal(voice.session.phase, 'signalling');
    speaking.emit('start', 'bob');
    assert.equal(streams.has('bob'), false, 'other speakers stay ignored while the signal is audible');
    speaking.emit('start', 'alice');
    assert.equal(voice.session.phase, 'signalling', 'the command waits for actual microphone audio');
    streams.get('alice')!.end(packet);
    await waitFor(() => voice.session.phase === 'processing');
    speaking.emit('start', 'bob');
    assert.equal(streams.has('bob'), false, 'no other speaker can answer during the second note');
    assert.equal(queue.volume, 0.5, 'the command waits for the complete signal');
    assert.equal(Reflect.get(queue, 'voiceDucking'), true);
    cueResolve();
    await waitFor(() => queue.volume === 0.6 && voice.session.phase === 'idle');
    assert.equal(Reflect.get(queue, 'voiceDucking'), false);
    assert.equal(cueCount, 1);
  } finally { encoder.delete(); await queue.stop(); }
});


test('voice radio requests bypass Laya and YouTube, switch stations and clear the music queue after the wake signal', async () => {
  for (const message of ['Включи Европа Плюс', 'Включи Retro FM', 'Поставь Ретро ФМ', 'Europa Plus']) {
    const h = musicHarness(() => {
      const stream = new PassThrough(); stream.write(Buffer.alloc(3840 * 3));
      return { stream, type: StreamType.Raw, destroy: () => stream.destroy() };
    });
    h.runtime.classify = async () => { assert.fail('Radio must not depend on model labels'); };
    h.metadata.searchCandidates = async () => { assert.fail('Radio must not search YouTube'); };
    try {
      await h.queue.addTracks([toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob'),
        toTrack({ id: 'ddddddddddd', title: 'Pending' }, 'Bob')]);
      const { processing } = await h.command(message); await processing;
      assert.equal(h.queue.currentTrack?.source, 'radio');
      assert.equal(h.queue.currentTrack?.requestedBy, 'Alice');
      assert.equal(h.queue.tracks.length, 0); assert.equal(h.voice.session.phase, 'idle');
      assert.equal(Reflect.get(h.queue, 'voiceDucking'), false);
    } finally { await h.queue.stop(); }
  }
});

test('unknown, negative and compound spoken radio requests never search or change playback', async () => {
  for (const message of ['Включи радио неизвестное', 'Не включай Европа Плюс',
    'Включи Retro FM?', 'Включи Европа Плюс и сделай громче']) {
    const h = musicHarness(() => { assert.fail('Unsupported station must not start a process'); });
    const replies: string[] = [];
    h.runtime.classify = async () => { assert.fail('Unsafe radio must not ask the model'); };
    h.metadata.searchCandidates = async () => { assert.fail('Unsupported radio must not become a song'); };
    h.queue.client.voiceTts = { audio: key => { replies.push(key); return null; } };
    try {
      const current = toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob');
      await h.queue.addTrack(current);
      const { processing } = await h.command(message); await processing;
      assert.equal(h.queue.currentTrack, current);
      assert.deepEqual(replies, message.includes('неизвестное') ? ['radio_unsupported'] : [], message);
    } finally { await h.queue.stop(); }
  }
});

test('leaving, voice off and reset while a radio stream is opening cannot clear music or execute late', async () => {
  for (const mode of ['leave', 'off', 'reset']) {
    const stream = new PassThrough();
    const h = musicHarness(() => ({ stream, type: StreamType.Raw, destroy: () => stream.destroy() }));
    try {
      const current = toTrack({ id: 'ccccccccccc', title: 'Current' }, 'Bob');
      await h.queue.addTrack(current);
      const { processing } = await h.command('Включи Европа Плюс');
      await waitFor(() => stream.listenerCount('readable') > 0);
      if (mode === 'leave') { h.states.set('alice', { channelId: 'other' }); h.voice.cancelUser('alice'); }
      if (mode === 'off') h.voice.setEnabled(false);
      if (mode === 'reset') h.voice.reset();
      await processing;
      assert.equal(h.queue.currentTrack, current); assert.equal(stream.destroyed, true);
    } finally { await h.queue.stop(); }
  }
});
