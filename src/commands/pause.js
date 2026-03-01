'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('pause')
    .setDescription('⏸ Поставить воспроизведение на паузу'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', ephemeral: true });
    }

    if (queue.pause()) {
      await interaction.reply({ content: '⏸ Пауза.', ephemeral: true });
    } else {
      await interaction.reply({ content: '❌ Уже на паузе.', ephemeral: true });
    }
  },
};
