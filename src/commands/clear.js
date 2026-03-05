'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('clear')
    .setDescription('🗑 Очистить очередь (текущий трек продолжит играть)'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: 'ℹ️ Очищать нечего: бот сейчас не в голосовом канале.', flags: MessageFlags.Ephemeral });
    }

    const count = queue.tracks.length;
    queue.clearQueue();

    await interaction.reply({
      content: count > 0
        ? `🗑 Очередь очищена (удалено **${count}** треков).`
        : '📋 Очередь уже была пустой.',
      flags: MessageFlags.Ephemeral,
    });
  },
};
