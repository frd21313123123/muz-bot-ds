'use strict';

const { SlashCommandBuilder, PermissionsBitField, MessageFlags } = require('discord.js');
const GuildQueue = require('../utils/GuildQueue');
const { resolveQuery } = require('../utils/resolve');

module.exports = {
  data: new SlashCommandBuilder()
    .setName('play')
    .setDescription('▶ Воспроизвести трек или плейлист')
    .addStringOption(opt =>
      opt.setName('query')
        .setDescription('Ссылка YouTube / YouTube Music или название трека')
        .setRequired(true)
        .setMinLength(1)
        .setMaxLength(500),
    )
    .addBooleanOption(opt =>
      opt.setName('autoplay')
        .setDescription('Бесконечный подбор треков по рекомендациям'),
    ),

  /** @param {import('discord.js').ChatInputCommandInteraction} interaction */
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });

    // ── Проверка: пользователь в голосовом канале ─────────────────────────
    const voiceChannel = interaction.member.voice.channel;
    if (!voiceChannel) {
      return interaction.editReply('❌ Вы должны находиться в голосовом канале!');
    }

    // ── Проверка прав бота ────────────────────────────────────────────────
    const perms = voiceChannel.permissionsFor(interaction.client.user);
    if (
      !perms.has(PermissionsBitField.Flags.Connect) ||
      !perms.has(PermissionsBitField.Flags.Speak)
    ) {
      return interaction.editReply('❌ У меня нет прав для подключения к вашему голосовому каналу!');
    }

    // ── Разрешение запроса ─────────────────────────────────────────────────
    const query = interaction.options.getString('query', true);
    let result;
    try {
      result = await resolveQuery(query, interaction.member.displayName);
    } catch (err) {
      console.error('[play] resolveQuery:', err.message);
      return interaction.editReply(`❌ Не удалось получить трек: \`${err.message}\``);
    }

    if (!result) {
      return interaction.editReply('❌ Ничего не найдено по вашему запросу.');
    }

    // ── Получить / создать очередь ─────────────────────────────────────────
    let queue = client.queues.get(interaction.guildId);
    if (!queue) {
      queue = new GuildQueue(interaction.guildId, client);
      client.queues.set(interaction.guildId, queue);
    }

    // ── Autoplay опция ────────────────────────────────────────────────────
    const autoplayOpt = interaction.options.getBoolean('autoplay');
    if (autoplayOpt !== null) {
      queue.autoplay = autoplayOpt;
    }

    // ── Подключиться к каналу (если нужно) ────────────────────────────────
    if (!queue.connection || queue.voiceChannel?.id !== voiceChannel.id) {
      try {
        await queue.join(voiceChannel);
      } catch (err) {
        client.queues.delete(interaction.guildId);
        return interaction.editReply(`❌ ${err.message}`);
      }
    }

    // ── Заметка о смене autoplay ───────────────────────────────────────────
    const autoplayNote = autoplayOpt === true
      ? '\n🔁 Автовоспроизведение **включено**'
      : autoplayOpt === false
        ? '\n🔁 Автовоспроизведение **выключено**'
        : '';

    // ── Активировать плеер автоматически ────────────────────────────────
    if (!queue._playerChannelId) {
      queue.setPlayerChannel(interaction.channelId);
      queue._startPlayerInterval();
    }

    // ── Добавить треки ─────────────────────────────────────────────────────
    if (result.type === 'playlist') {
      const wasIdle = !queue.currentTrack && queue.tracks.length === 0;
      await queue.addTracks(result.tracks);

      return interaction.editReply(
        `✅ Добавлено **${result.tracks.length}** треков из плейлиста **${result.name}**` +
        (wasIdle ? '\n▶ Воспроизведение началось!' : '') +
        autoplayNote,
      );
    }

    // single
    const wasIdle = !queue.currentTrack && queue.tracks.length === 0;
    await queue.addTrack(result.track);

    return interaction.editReply(
      (wasIdle
        ? `▶ Воспроизведение началось: **${result.track.title}**`
        : `✅ Добавлено в очередь: **${result.track.title}** (${result.track.duration})`) +
      autoplayNote,
    );
  },
};
