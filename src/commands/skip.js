'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('skip')
    .setDescription('⏭ Пропустить текущий трек'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', flags: MessageFlags.Ephemeral });
    }

    const title = queue.currentTrack.title;
    queue.skip();
    await interaction.reply({ content: `⏭ Пропущено: **${title}**`, flags: MessageFlags.Ephemeral });
  },
};
