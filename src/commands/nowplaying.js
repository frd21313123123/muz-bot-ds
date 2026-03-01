'use strict';

const { SlashCommandBuilder } = require('discord.js');
const { nowPlayingEmbed } = require('../utils/embeds');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('nowplaying')
    .setDescription('🎵 Показать текущий трек'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', ephemeral: true });
    }

    await interaction.reply({ embeds: [nowPlayingEmbed(queue.currentTrack, queue.autoplay)], ephemeral: true });
  },
};
