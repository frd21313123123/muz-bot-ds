'use strict';

const { SlashCommandBuilder } = require('discord.js');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('autoplay')
    .setDescription('🔁 Вкл/выкл автовоспроизведение по рекомендациям YouTube'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: '❌ Бот не подключён к голосовому каналу.', ephemeral: true });
    }

    const enabled = queue.toggleAutoplay();
    await interaction.reply({
      content: enabled
        ? '🔁 Автовоспроизведение **включено** — бот будет ставить похожие треки из рекомендаций YouTube.'
        : '🔁 Автовоспроизведение **выключено**.',
      ephemeral: true,
    });
  },
};
