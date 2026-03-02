'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('resume')
    .setDescription('▶ Продолжить воспроизведение после паузы'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', flags: MessageFlags.Ephemeral });
    }

    if (queue.resume()) {
      await interaction.reply({ content: '▶ Воспроизведение продолжено.', flags: MessageFlags.Ephemeral });
    } else {
      await interaction.reply({ content: '❌ Воспроизведение не на паузе.', flags: MessageFlags.Ephemeral });
    }
  },
};
