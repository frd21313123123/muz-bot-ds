'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('resume')
    .setDescription('▶ Продолжить воспроизведение после паузы'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', ephemeral: true });
    }

    if (queue.resume()) {
      await interaction.reply({ content: '▶ Воспроизведение продолжено.', ephemeral: true });
    } else {
      await interaction.reply({ content: '❌ Воспроизведение не на паузе.', ephemeral: true });
    }
  },
};
