'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('clear')
    .setDescription('🗑 Очистить очередь (текущий трек продолжит играть)'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: '❌ Бот не подключён.', ephemeral: true });
    }

    const count = queue.tracks.length;
    queue.clearQueue();

    await interaction.reply({
      content: count > 0
        ? `🗑 Очередь очищена (удалено **${count}** треков).`
        : '📋 Очередь уже была пустой.',
      ephemeral: true,
    });
  },
};
