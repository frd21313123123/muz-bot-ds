'use strict';

const { ActivityType } = require('discord.js');

module.exports = {
  name: 'clientReady',
  once: true,
  execute(client) {
    console.log(`✅ Бот запущен как ${client.user.tag}`);
    client.user.setActivity('/play', { type: ActivityType.Listening });
  },
};
