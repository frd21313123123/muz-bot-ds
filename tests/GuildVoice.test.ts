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

const waitFor = async (condition: () => boolean): Promise<void> => {
  const end = Date.now() + 2000;
  while (!condition()) {
    if (Date.now() > end) throw new Error('voice session did not advance');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
};

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
