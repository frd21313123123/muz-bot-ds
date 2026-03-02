'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');
const { nowPlayingEmbed } = require('../utils/embeds');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('nowplaying')
    .setDescription('🎵 Показать текущий трек'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.currentTrack) {
      return interaction.reply({ content: '❌ Сейчас ничего не играет.', flags: MessageFlags.Ephemeral });
    }

    await interaction.reply({ embeds: [nowPlayingEmbed(queue.currentTrack, queue.autoplay)], flags: MessageFlags.Ephemeral });
  },
};
