import assert from 'node:assert/strict';
import test from 'node:test';
import { MessageFlags, type ChatInputCommandInteraction, type VoiceBasedChannel } from 'discord.js';
import { commands } from '../src/commands.js';
import { GuildQueue } from '../src/utils/GuildQueue.js';
import type { MusicClient } from '../src/types.js';

const command = commands.find(command => command.data.name === 'speed')!;

function fixture() {
  const client = { queues: new Map() } as unknown as MusicClient;
  const queue = new GuildQueue('guild-speed', client);
  queue.voiceChannel = { id: 'voice' } as VoiceBasedChannel;
  client.queues.set(queue.guildId, queue);
  let value = 0.8;
  const voiceStates = new Map([['user', { channel: { id: 'voice' } }]]);
  const replies: { content: string; flags: number }[] = [];
  const interaction = {
    guildId: queue.guildId, user: { id: 'user' }, guild: { voiceStates: { cache: voiceStates } },
    options: { getNumber: () => value },
    reply: async (payload: { content: string; flags: number }) => { replies.push(payload); },
  } as unknown as ChatInputCommandInteraction;
  return { client, queue, voiceStates, interaction, replies, setValue: (speed: number) => { value = speed; },
    execute: () => command.execute(interaction, client) };
}

test('/speed supports fractional factors, reset, private replies and a bounded Discord number option', async () => {
  assert.ok(command);
  const definition = command.data.toJSON();
  assert.equal(definition.options?.[0]?.type, 10);
  const f = fixture();
  try {
    for (const value of [0.5, 0.8, 1.25, 2, 1]) {
      f.setValue(value); await f.execute();
      assert.equal(f.queue.speed, value);
      assert.equal(f.replies.at(-1)?.flags, MessageFlags.Ephemeral);
      assert.ok(f.replies.at(-1)?.content.includes(`${value}×`));
    }
    assert.match(f.replies.at(-1)!.content, /Обычная тональность/);
  } finally { await f.queue.stop(); }
});

test('/speed rejects users outside the bot channel, missing queues and invalid values', async () => {
  const f = fixture();
  try {
    f.voiceStates.set('user', { channel: { id: 'other' } });
    await f.execute(); assert.equal(f.queue.speed, 1);
    assert.match(f.replies.at(-1)!.content, /тот же голосовой канал/);
    f.voiceStates.clear(); await f.execute(); assert.equal(f.queue.speed, 1);
    f.voiceStates.set('user', { channel: { id: 'voice' } });
    for (const value of [NaN, Infinity, 0, 0.4, 2.1]) {
      f.setValue(value); await f.execute(); assert.equal(f.queue.speed, 1);
      assert.match(f.replies.at(-1)!.content, /Скорость должна быть/);
    }
    f.client.queues.clear(); await f.execute();
    assert.match(f.replies.at(-1)!.content, /не в голосовом канале/);
    assert.ok(f.replies.every(reply => reply.flags === MessageFlags.Ephemeral));
  } finally { await f.queue.stop(); }
});
