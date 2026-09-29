import assert from 'node:assert/strict';
import test from 'node:test';
import type { ChatInputCommandInteraction } from 'discord.js';
import { canControl } from '../src/utils/voiceAccess.js';

const interaction = (channelId: string | null): ChatInputCommandInteraction => ({
  user: { id: 'user' },
  guild: { voiceStates: { cache: new Map([['user', { channel: channelId ? { id: channelId } : null }]]) } },
  member: null,
}) as unknown as ChatInputCommandInteraction;

test('control requires the same voice channel', () => {
  assert.equal(canControl(interaction('voice-a'), 'voice-a'), true);
  assert.equal(canControl(interaction('voice-b'), 'voice-a'), false);
  assert.equal(canControl(interaction(null), 'voice-a'), false);
  assert.equal(canControl(interaction('voice-a'), null), false);
});
