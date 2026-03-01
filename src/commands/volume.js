'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('volume')
    .setDescription('🔊 Установить громкость воспроизведения')
    .addIntegerOption(opt =>
      opt.setName('level')
        .setDescription('Уровень: 1–150 (по умолчанию 50)')
        .setRequired(true)
        .setMinValue(1)
        .setMaxValue(150),
    ),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: '❌ Бот не подключён.', ephemeral: true });
    }

    const level = interaction.options.getInteger('level', true);
    queue.setVolume(level);
    await interaction.reply({ content: `🔊 Громкость: **${level}%**`, ephemeral: true });
  },
};
