'use strict';

const { MessageFlags } = require('discord.js');
const { AudioPlayerStatus } = require('@discordjs/voice');

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
        return interaction.reply({ content: '❌ Бот не подключён.', flags: MessageFlags.Ephemeral });
      }

      // Проверка: пользователь должен быть в том же голосовом канале
      const memberVoice = interaction.member?.voice?.channel;
      if (!memberVoice || memberVoice.id !== queue.voiceChannel?.id) {
        return interaction.reply({
          content: '❌ Вы должны быть в том же голосовом канале, что и бот.',
          flags: MessageFlags.Ephemeral,
        });
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
            await interaction.deferUpdate();
            break;
          }

          case 'player_skip': {
            if (queue.currentTrack) {
              queue.skip();
            }
            await interaction.deferUpdate();
            break;
          }

          case 'player_stop': {
            queue.stop();
            await interaction.deferUpdate();
            break;
          }

          case 'player_autoplay': {
            const enabled = queue.toggleAutoplay();
            await interaction.reply({
              content: enabled
                ? '🔁 Автовоспроизведение **включено**'
                : '🔁 Автовоспроизведение **выключено**',
              flags: MessageFlags.Ephemeral,
            });
            break;
          }

          default:
            await interaction.deferUpdate();
        }
      } catch (err) {
        console.error(`[Button:${customId}]`, err);
        if (!interaction.replied && !interaction.deferred) {
          await interaction.reply({ content: '❌ Ошибка.', flags: MessageFlags.Ephemeral }).catch(() => {});
        }
      }
    }
  },
};
