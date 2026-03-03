'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('autoplay')
    .setDescription('♾ Вкл/выкл бесконечное воспроизведение рекомендаций'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: '❌ Бот не подключён к голосовому каналу.', flags: MessageFlags.Ephemeral });
    }

    const enabled = queue.toggleAutoplay();
    await interaction.reply({
      content: enabled
        ? '♾ Бесконечное воспроизведение **включено** — после конца очереди бот добавляет ещё 25 рекомендаций.'
        : '♾ Бесконечное воспроизведение **выключено**.',
      flags: MessageFlags.Ephemeral,
    });
  },
};
