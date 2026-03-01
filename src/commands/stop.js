'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('stop')
    .setDescription('⏹ Остановить воспроизведение, очистить очередь и покинуть канал'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: '❌ Бот не подключён к голосовому каналу.', ephemeral: true });
    }

    queue.stop();
    await interaction.reply({ content: '⏹ Воспроизведение остановлено, очередь очищена.', ephemeral: true });
  },
};
