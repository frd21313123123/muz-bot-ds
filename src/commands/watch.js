'use strict';

const { SlashCommandBuilder, MessageFlags } = require('discord.js');
const {
  canCreateWatchTogetherInvite,
  createWatchTogetherInvite,
} = require('../utils/watchTogether');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('watch')
    .setDescription('📺 Запустить совместный просмотр видео (Watch Together)'),

  /** @param {import('discord.js').ChatInputCommandInteraction} interaction */
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });

    const voiceChannel = interaction.member.voice.channel;
    if (!voiceChannel) {
      return interaction.editReply('❌ Вы должны находиться в голосовом канале!');
    }

    const me = interaction.guild.members.me ?? await interaction.guild.members.fetchMe().catch(() => null);
    if (!canCreateWatchTogetherInvite(voiceChannel, me)) {
      return interaction.editReply(
        '❌ Нужны права **Просмотр канала**, **Создать приглашение** и **Использовать активности** в этом голосовом канале.',
      );
    }

    let invite;
    try {
      invite = await createWatchTogetherInvite(voiceChannel, interaction.user.tag);
    } catch (err) {
      console.error('[watch] create invite:', err.message);
      return interaction.editReply('❌ Не удалось запустить Watch Together. Проверьте права и попробуйте снова.');
    }

    const queue = client.queues.get(interaction.guildId);
    const track = queue?.currentTrack && queue.voiceChannel?.id === voiceChannel.id
      ? queue.currentTrack
      : null;

    let text = `📺 Совместный просмотр запущен: ${invite.url}`;
    if (track?.url) {
      text += `\n🎵 Текущий трек: ${track.url}`;
    }
    text += '\nОткройте ссылку и выберите YouTube внутри активности.';

    return interaction.editReply(text);
  },
};
