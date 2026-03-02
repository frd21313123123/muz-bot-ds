'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('pause')
    .setDescription('⏸ Поставить воспроизведение на паузу'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', flags: MessageFlags.Ephemeral });
    }

    if (queue.pause()) {
      await interaction.reply({ content: '⏸ Пауза.', flags: MessageFlags.Ephemeral });
    } else {
      await interaction.reply({ content: '❌ Уже на паузе.', flags: MessageFlags.Ephemeral });
    }
  },
};
