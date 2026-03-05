'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');
const { playerEmbed, playerActionRow } = require('../utils/embeds');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('player')
    .setDescription('🎛 Показать интерактивный плеер с кнопками управления'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    if (!queue?.connection) {
      return interaction.reply({ content: 'ℹ️ Бот сейчас не в голосовом канале. Сначала запустите /play.', flags: MessageFlags.Ephemeral });
    }

    // Удалить старое сообщение плеера, если было
    await queue._destroyPlayerMessage();

    // Отправить НЕ-эфемерное сообщение
    const embed = playerEmbed(queue);
    const row = playerActionRow(queue);

    await interaction.deferReply(); // не ephemeral
    const msg = await interaction.editReply({ embeds: [embed], components: [row] });

    queue._playerMessage = msg;
    queue.setPlayerChannel(interaction.channelId);
    queue._startPlayerInterval();
  },
};
