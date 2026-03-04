'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const playCommand = require('../src/commands/play');

test('/play handles null channel permissions without throwing', async () => {
  let editedReply = null;

  const interaction = {
    guildId: 'guild-1',
    client: { user: { id: 'bot-user' } },
    member: {
      displayName: 'Requester',
      voice: {
        channel: {
          permissionsFor: () => null,
        },
      },
    },
    options: {
      getString: () => 'unused',
    },
    deferReply: async () => {},
    editReply: async (message) => {
      editedReply = message;
    },
  };

  const client = {
    queues: new Map(),
  };

  await playCommand.execute(interaction, client);

  assert.equal(
    editedReply,
    '❌ У меня нет прав для подключения к вашему голосовому каналу!',
  );
});
