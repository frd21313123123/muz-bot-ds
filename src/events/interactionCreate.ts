import { MessageFlags, type Interaction } from 'discord.js';
import type { MusicClient } from '../types.js';
import { commands } from '../commands.js';
import { canControl } from '../utils/voiceAccess.js';
import { radioSuggestions } from '../utils/radio.js';
import { beginCommand, finishCommand } from '../utils/performance.js';

const byName = new Map(commands.map((command) => [command.data.name, command]));

export async function onInteraction(interaction: Interaction, client: MusicClient): Promise<void> {
  if (interaction.isAutocomplete() && interaction.commandName === 'radio') {
    await interaction.respond(radioSuggestions(interaction.options.getFocused())).catch(() => {});
    return;
  }
  if (interaction.isChatInputCommand()) {
    const command = byName.get(interaction.commandName);
    if (!command) return;
    beginCommand(interaction);
    try {
      await command.execute(interaction, client);
    } catch (error) {
      console.error(`[Command:${interaction.commandName}]`, error);
      const payload = { content: '❌ Ошибка выполнения команды.', flags: MessageFlags.Ephemeral as const };
      if (interaction.deferred || interaction.replied) await interaction.followUp(payload).catch(() => {});
      else await interaction.reply(payload).catch(() => {});
    } finally { finishCommand(interaction); }
    return;
  }

  if (!interaction.isButton() || !interaction.customId.startsWith('player_')) return;
  const queue = client.queues.get(interaction.guildId ?? '');
  if (!queue?.voiceChannel) {
    await interaction.reply({ content: '⏹ Бот уже остановлен.', flags: MessageFlags.Ephemeral }).catch(() => {});
    return;
  }
  if (!canControl(interaction, queue.voiceChannel.id)) {
    await interaction.reply({ content: '❌ Войдите в тот же голосовой канал, что и бот.', flags: MessageFlags.Ephemeral }).catch(() => {});
    return;
  }
  try {
    switch (interaction.customId) {
      case 'player_playpause':
        await interaction.deferUpdate();
        if (queue.isPaused) queue.resume();
        else queue.pause();
        break;
      case 'player_skip':
        await interaction.deferUpdate();
        queue.skip();
        break;
      case 'player_stop':
        await interaction.deferUpdate();
        await queue.stop();
        break;
      case 'player_autoplay':
        if (queue.isRadio) { await interaction.reply({ content: 'ℹ️ Рекомендации недоступны во время радио.', flags: MessageFlags.Ephemeral }); break; }
        await interaction.reply({ content: `♾ Бесконечное воспроизведение ${queue.toggleAutoplay() ? 'включено' : 'выключено'}.`, flags: MessageFlags.Ephemeral });
        break;
      case 'player_loop':
        if (queue.isRadio) { await interaction.reply({ content: 'ℹ️ Повтор недоступен во время радио.', flags: MessageFlags.Ephemeral }); break; }
        await interaction.reply({ content: `🔂 Повтор трека ${queue.toggleLoop() ? 'включён' : 'выключен'}.`, flags: MessageFlags.Ephemeral });
        break;
      default:
        await interaction.deferUpdate();
    }
  } catch (error) {
    console.error(`[Button:${interaction.customId}]`, error);
    if (!interaction.deferred && !interaction.replied) {
      await interaction.reply({ content: '❌ Ошибка управления плеером.', flags: MessageFlags.Ephemeral }).catch(() => {});
    }
  }
}
