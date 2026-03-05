'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');
const { getVoiceConnection } = require('@discordjs/voice');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('stop')
    .setDescription('⏹ Остановить воспроизведение, очистить очередь и покинуть канал'),

  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId);
    const orphanedConnection = getVoiceConnection(interaction.guildId);

    if (!queue && !orphanedConnection) {
      return interaction.reply({ content: '⏹ Бот уже остановлен.', flags: MessageFlags.Ephemeral });
    }

    // Пытаемся остановить через queue
    if (queue) {
      try {
        await queue.stop();
      } catch (err) {
        console.error(`[stop] queue.stop() error: ${err.message}`);
      }
      client.queues.delete(interaction.guildId);
    }

    // Fallback: уничтожить orphaned connection, если осталось
    const remainingConn = getVoiceConnection(interaction.guildId);
    if (remainingConn) {
      try { remainingConn.destroy(); } catch (_) {}
    }

    await interaction.reply({ content: '⏹ Воспроизведение остановлено, очередь очищена.', flags: MessageFlags.Ephemeral });
  },
};
