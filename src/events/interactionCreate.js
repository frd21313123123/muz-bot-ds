'use strict';

const { MessageFlags } = require('discord.js');
const { AudioPlayerStatus } = require('@discordjs/voice');

async function safeDeferUpdate(interaction) {
  if (interaction.deferred || interaction.replied) return;
  try {
    await interaction.deferUpdate();
  } catch (err) {
    // 40060: interaction already acknowledged, 10062: interaction expired.
    if (err?.code === 40060 || err?.code === 10062) return;
    throw err;
  }
}

module.exports = {
  name: 'interactionCreate',
  async execute(interaction, client) {
    // ── Slash commands ───────────────────────────────────────────────────
    if (interaction.isChatInputCommand()) {
      const command = client.commands.get(interaction.commandName);
      if (!command) return;

      try {
        await command.execute(interaction, client);
      } catch (error) {
        console.error(`[Command:${interaction.commandName}]`, error);

        const payload = { content: '❌ Произошла ошибка при выполнении команды.', flags: MessageFlags.Ephemeral };
        if (interaction.replied || interaction.deferred) {
          await interaction.followUp(payload).catch(() => {});
        } else {
          await interaction.reply(payload).catch(() => {});
        }
      }
      return;
    }

    // ── Button interactions (player) ─────────────────────────────────────
    if (interaction.isButton()) {
      const customId = interaction.customId;
      if (!customId.startsWith('player_')) return;

      const queue = client.queues.get(interaction.guildId);
      if (!queue) {
        return interaction.reply({ content: '⏹ Бот уже остановлен.', flags: MessageFlags.Ephemeral }).catch(() => {});
      }

      // Проверка: пользователь должен быть в том же голосовом канале
      const memberVoice = interaction.member?.voice?.channel;
      if (!memberVoice || memberVoice.id !== queue.voiceChannel?.id) {
        return interaction.reply({
          content: '❌ Вы должны быть в том же голосовом канале, что и бот.',
          flags: MessageFlags.Ephemeral,
        }).catch(() => {});
      }

      try {
        switch (customId) {
          case 'player_playpause': {
            const status = queue.player.state.status;
            if (status === AudioPlayerStatus.Playing) {
              queue.pause();
            } else if (status === AudioPlayerStatus.Paused) {
              queue.resume();
            }
            await safeDeferUpdate(interaction);
            break;
          }

          case 'player_skip': {
            if (queue.currentTrack) {
              queue.skip();
            }
            await safeDeferUpdate(interaction);
            break;
          }

          case 'player_stop': {
            await safeDeferUpdate(interaction);
            await queue.stop();
            break;
          }

          case 'player_autoplay': {
            const enabled = queue.toggleAutoplay();
            await interaction.reply({
              content: enabled
                ? '♾ Бесконечное воспроизведение **включено**'
                : '♾ Бесконечное воспроизведение **выключено**',
              flags: MessageFlags.Ephemeral,
            }).catch(() => {});
            break;
          }

          case 'player_loop': {
            const enabled = queue.toggleLoop();
            await interaction.reply({
              content: enabled
                ? '🔂 Повтор текущего трека **включён**'
                : '🔂 Повтор текущего трека **выключен**',
              flags: MessageFlags.Ephemeral,
            }).catch(() => {});
            break;
          }

          default:
            await safeDeferUpdate(interaction);
        }
      } catch (err) {
        // Ignore duplicate/expired interaction ack races to keep logs clean.
        if (err?.code === 40060 || err?.code === 10062) return;

        console.error(`[Button:${customId}]`, err);
        if (!interaction.replied && !interaction.deferred) {
          await interaction.reply({ content: '❌ Ошибка.', flags: MessageFlags.Ephemeral }).catch(() => {});
        }
      }
    }
  },
};
