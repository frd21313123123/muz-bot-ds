'use strict';

const { SlashCommandBuilder, PermissionsBitField, MessageFlags } = require('discord.js');
const GuildQueue = require('../utils/GuildQueue');
const { resolveQuery } = require('../utils/resolve');
const INFINITE_START_BATCH = 25;

module.exports = {
  data: new SlashCommandBuilder()
    .setName('play')
    .setDescription('▶ Воспроизвести трек')
    .addStringOption(opt =>
      opt.setName('query')
        .setDescription('Ссылка YouTube / YouTube Music или название трека')
        .setRequired(true)
        .setMinLength(1)
        .setMaxLength(500),
    )
    .addBooleanOption(opt =>
      opt.setName('infinite')
        .setDescription('♾ Бесконечное воспроизведение: после конца очереди добавлять ещё 25 рекомендаций'),
    ),

  /** @param {import('discord.js').ChatInputCommandInteraction} interaction */
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });

    const voiceChannel = interaction.member.voice.channel;
    if (!voiceChannel) {
      return interaction.editReply('❌ Вы должны находиться в голосовом канале!');
    }

    const perms = voiceChannel.permissionsFor(interaction.client.user);
    if (
      !perms.has(PermissionsBitField.Flags.Connect) ||
      !perms.has(PermissionsBitField.Flags.Speak)
    ) {
      return interaction.editReply('❌ У меня нет прав для подключения к вашему голосовому каналу!');
    }

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

    let queue = client.queues.get(interaction.guildId);
    if (!queue) {
      queue = new GuildQueue(interaction.guildId, client);
      client.queues.set(interaction.guildId, queue);
    }

    // Параметр бесконечного воспроизведения.
    const infiniteOpt = interaction.options.getBoolean('infinite');
    if (infiniteOpt !== null) {
      queue.setAutoplay(infiniteOpt);
    }

    if (!queue.connection || queue.voiceChannel?.id !== voiceChannel.id) {
      try {
        await queue.join(voiceChannel);
      } catch (err) {
        client.queues.delete(interaction.guildId);
        return interaction.editReply(`❌ ${err.message}`);
      }
    }

    const infiniteNote = infiniteOpt === true
      ? '\n♾ Бесконечное воспроизведение **включено**'
      : infiniteOpt === false
        ? '\n♾ Бесконечное воспроизведение **выключено**'
        : '';

    if (!queue._playerChannelId) {
      queue.setPlayerChannel(interaction.channelId);
      queue._startPlayerInterval();
    }

    if (result.type === 'playlist') {
      if (!result.tracks.length) {
        return interaction.editReply('❌ Плейлист пуст.');
      }

      const wasIdle = !queue.currentTrack && queue.tracks.length === 0;

      if (queue.autoplay) {
        const initialTracks = result.tracks.slice(0, INFINITE_START_BATCH);
        await queue.addTracks(initialTracks);
        return interaction.editReply(
          `✅ Добавлено **${initialTracks.length}** треков из плейлиста **${result.name}**` +
          (wasIdle ? '\n▶ Воспроизведение началось!' : '') +
          '\n♾ После конца очереди бот подберёт ещё 25 треков.' +
          infiniteNote,
        );
      }

      const firstTrack = result.tracks[0];
      await queue.addTrack(firstTrack);
      return interaction.editReply(
        `✅ По умолчанию добавлен только 1 трек из плейлиста **${result.name}**: **${firstTrack.title}**` +
        (wasIdle ? '\n▶ Воспроизведение началось!' : '') +
        '\nЧтобы сделать поток бесконечным, включите `infinite`.' +
        infiniteNote,
      );
    }

    const wasIdle = !queue.currentTrack && queue.tracks.length === 0;
    await queue.addTrack(result.track);

    return interaction.editReply(
      (wasIdle
        ? `▶ Воспроизведение началось: **${result.track.title}**`
        : `✅ Добавлено в очередь: **${result.track.title}** (${result.track.duration})`) +
      infiniteNote,
    );
  },
};
