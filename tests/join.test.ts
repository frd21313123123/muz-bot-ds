import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import test, { type TestContext } from 'node:test';
import { VoiceConnectionStatus, StreamType, type VoiceConnection } from '@discordjs/voice';
import { MessageFlags, PermissionsBitField, type ChatInputCommandInteraction, type VoiceBasedChannel } from 'discord.js';
import { commands } from '../src/commands.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { MusicClient, Track } from '../src/types.js';

const command = commands.find(command => command.data.name === 'join')!;
const fixture = (t: TestContext) => {
  const speaking = new EventEmitter();
  const deafened: boolean[] = [];
  const connection = {
    state: { status: VoiceConnectionStatus.Ready }, receiver: { speaking }, joinConfig: {},
    rejoin: ({ selfDeaf }: { selfDeaf: boolean }) => { deafened.push(selfDeaf); return true; },
    destroy: () => { connection.state.status = VoiceConnectionStatus.Destroyed; },
  } as unknown as VoiceConnection;
  const channel = { id: 'voice', permissionsFor: () => new PermissionsBitField([
    PermissionsBitField.Flags.Connect, PermissionsBitField.Flags.Speak,
  ]) } as unknown as VoiceBasedChannel;
  const voiceStates = new Map([['user', { channel }]]);
  const guild = { members: { me: {} }, voiceStates: { cache: voiceStates } };
  const runtime = Object.assign(new EventEmitter(), { ready: true, start: async () => true });
  const client = { queues: new Map(), guilds: { cache: new Map([['guild-join', guild]]) },
    voiceRuntime: runtime, voiceSettings: { get: () => 'Бот' }, ytdlp: {},
  } as unknown as MusicClient;
  const replies: unknown[] = [];
  const interaction = { guildId: 'guild-join', guild, user: { id: 'user' }, channelId: 'text',
    deferReply: async (options: unknown) => { replies.push(options); },
    editReply: async (content: unknown) => { replies.push(content); },
  } as unknown as ChatInputCommandInteraction;
  // Exercise the real queue/listener; replace only the network handshake.
  const connect = t.mock.method(GuildQueue.prototype as unknown as { connect(channel: VoiceBasedChannel): Promise<void> },
    'connect', async function (this: GuildQueue, target: VoiceBasedChannel) {
      this.voiceChannel = target; this.connection = connection;
    });
  const execute = () => command.execute(interaction, client);
  const cleanup = async () => { for (const queue of [...client.queues.values()]) await queue.stop(); };
  return { client, channel, runtime, voiceStates, interaction, replies, speaking, deafened, connect, execute, cleanup };
};

test('/join connects without music, enables one listener and replies privately', async t => {
  const f = fixture(t);
  try {
    await f.execute();
    const queue = f.client.queues.get('guild-join')!;
    assert.equal(queue.voiceChannel?.id, 'voice');
    assert.equal(queue.voice?.enabled, true);
    assert.equal(f.speaking.listenerCount('start'), 1);
    assert.equal(f.deafened.at(-1), false);
    assert.equal(queue.currentTrack, null);
    assert.equal(queue.tracks.length, 0);
    assert.deepEqual(f.replies[0], { flags: MessageFlags.Ephemeral });
    assert.match(JSON.stringify(f.replies.at(-1)), /Бот/);
    await f.execute();
    assert.equal(f.connect.mock.callCount(), 1);
    assert.equal(f.speaking.listenerCount('start'), 1);
  } finally { await f.cleanup(); }
});

test('/join reenables listening while preserving a current resource and pending tracks', async t => {
  const f = fixture(t);
  const stream = new PassThrough();
  const queue = new GuildQueue('guild-join', f.client, () => ({ stream, type: StreamType.OggOpus, destroy: () => stream.destroy() }));
  f.client.queues.set(queue.guildId, queue);
  const track: Track = { videoId: 'aaaaaaaaaaa', url: 'https://www.youtube.com/watch?v=aaaaaaaaaaa',
    title: 'Numb', duration: '3:00', requestedBy: 'User', thumbnail: null };
  try {
    await queue.join(f.channel);
    await queue.setVoiceEnabled(true);
    await queue.addTracks([track, { ...track, videoId: 'bbbbbbbbbbb' }]);
    await queue.setVoiceEnabled(false);
    const playerState = queue.player.state;
    await f.execute();
    assert.equal(queue.voice?.enabled, true);
    assert.equal(f.speaking.listenerCount('start'), 1);
    assert.equal(queue.player.state, playerState);
    assert.equal(queue.currentTrack, track);
    assert.equal(queue.tracks[0]?.videoId, 'bbbbbbbbbbb');
    assert.equal(stream.destroyed, false);
    assert.equal(f.connect.mock.callCount(), 1);
  } finally { await f.cleanup(); }
});

test('/join refuses missing guild/channel, missing permissions, unavailable models and a different bot channel', async t => {
  for (const reason of ['guild', 'channel', 'permissions', 'models', 'other-channel']) {
    const f = fixture(t);
    try {
      if (reason === 'guild') Object.assign(f.interaction, { guildId: null });
      if (reason === 'channel') f.voiceStates.clear();
      if (reason === 'permissions') Object.assign(f.channel, { permissionsFor: () => new PermissionsBitField() });
      if (reason === 'models') f.runtime.start = async () => false;
      if (reason === 'other-channel') {
        const queue = new GuildQueue('guild-join', f.client);
        queue.voiceChannel = { id: 'elsewhere' } as VoiceBasedChannel;
        f.client.queues.set(queue.guildId, queue);
      }
      await f.execute();
      assert.equal(f.connect.mock.callCount(), 0, reason);
      assert.equal(f.speaking.listenerCount('start'), 0, reason);
      assert.match(JSON.stringify(f.replies.at(-1)), /❌/, reason);
      if (reason !== 'other-channel') assert.equal(f.client.queues.size, 0, reason);
    } finally { await f.cleanup(); t.mock.restoreAll(); }
  }
});

test('/join rechecks presence after startup and connection, cleaning up an abandoned new queue', async t => {
  for (const stage of ['startup', 'connection']) {
    const f = fixture(t);
    try {
      if (stage === 'startup') f.runtime.start = async () => { f.voiceStates.clear(); return true; };
      else f.connect.mock.mockImplementation(async function (this: GuildQueue, channel: VoiceBasedChannel) {
        this.voiceChannel = channel;
        f.voiceStates.clear();
      });
      await f.execute();
      assert.equal(f.client.queues.size, 0);
      assert.equal(f.speaking.listenerCount('start'), 0);
      assert.match(JSON.stringify(f.replies.at(-1)), /вышли/);
    } finally { await f.cleanup(); t.mock.restoreAll(); }
  }
});

test('/join cleans up a failed connection and disconnects an empty queue after five minutes', async t => {
  const f = fixture(t);
  try {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    await f.execute();
    const queue = f.client.queues.get('guild-join')!;
    t.mock.timers.tick(5 * 60_000 - 1);
    assert.equal(queue.closed, false);
    t.mock.timers.tick(1);
    assert.equal(queue.closed, true);
    assert.equal(f.client.queues.size, 0);
    assert.equal(f.speaking.listenerCount('start'), 0);
    f.connect.mock.mockImplementation(async () => { throw new Error('simulated connection failure'); });
    await f.execute();
    assert.equal(f.client.queues.size, 0);
    assert.match(JSON.stringify(f.replies.at(-1)), /Не удалось/);
  } finally { await f.cleanup(); }
});
