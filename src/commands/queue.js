'use strict';

const { SlashCommandBuilder } = require('discord.js');
const { queueEmbed } = require('../utils/embeds');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('queue')
    .setDescription('📋 Показать очередь воспроизведения')
    .addIntegerOption(opt =>
      opt.setName('page')
        .setDescription('Номер страницы (по 10 треков)')
        .setMinValue(1),
    ),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue || (!queue.currentTrack && queue.tracks.length === 0)) {
      return interaction.reply({ content: '📋 Очередь пуста.', ephemeral: true });
    }

    const page = interaction.options.getInteger('page') ?? 1;
    await interaction.reply({ embeds: [queueEmbed(queue, page)], ephemeral: true });
  },
};
