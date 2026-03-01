'use strict';

const { EmbedBuilder } = require('discord.js');

const COLOR_PLAYING = 0x1DB954;  // Spotify green
const COLOR_QUEUE   = 0x5865F2;  // Blurple
const COLOR_ERROR   = 0xED4245;  // Red

const PAGE_SIZE = 10;

/**
 * Embed «Сейчас играет».
 * @param {import('./GuildQueue').Track} track
 * @param {boolean} autoplay
 */
function nowPlayingEmbed(track, autoplay = false) {
  const embed = new EmbedBuilder()
    .setColor(track.isAutoplay ? 0x57F287 : COLOR_PLAYING)
    .setTitle(track.isAutoplay ? '🤖 Автовоспроизведение' : '▶ Сейчас играет')
    .setDescription(`**[${track.title}](${track.url})**`)
    .addFields(
      { name: '⏱ Длительность', value: track.duration || '—', inline: true },
      { name: '👤 Запросил', value: track.requestedBy, inline: true },
    );

  if (autoplay && !track.isAutoplay) {
    embed.addFields({ name: '🔁', value: 'Автовоспроизведение включено', inline: true });
  }

  if (track.thumbnail) {
    embed.setThumbnail(track.thumbnail);
  }

  return embed;
}

/**
 * Embed очереди воспроизведения.
 * @param {import('./GuildQueue')} queue
 * @param {number} page
 */
function queueEmbed(queue, page = 1) {
  const totalPages = Math.max(1, Math.ceil(queue.tracks.length / PAGE_SIZE));
  page = Math.min(Math.max(1, page), totalPages);

  const start = (page - 1) * PAGE_SIZE;
  const slice = queue.tracks.slice(start, start + PAGE_SIZE);

  const embed = new EmbedBuilder()
    .setColor(COLOR_QUEUE)
    .setTitle('📋 Очередь воспроизведения');

  if (queue.currentTrack) {
    embed.addFields({
      name: '▶ Сейчас играет',
      value: `[${queue.currentTrack.title}](${queue.currentTrack.url}) — ${queue.currentTrack.duration}`,
    });
  }

  if (slice.length > 0) {
    const list = slice
      .map((t, i) => `\`${start + i + 1}.\` [${t.title}](${t.url}) — ${t.duration}`)
      .join('\n');
    embed.addFields({ name: `📃 Следующие треки (${queue.tracks.length} в очереди)`, value: list });
  } else if (!queue.currentTrack) {
    embed.setDescription('Очередь пуста. Добавьте трек командой `/play`.');
  }

  embed.setFooter({
    text: `Страница ${page}/${totalPages} • Автовоспроизведение: ${queue.autoplay ? '✅ вкл' : '❌ выкл'}`,
  });

  return embed;
}

module.exports = { nowPlayingEmbed, queueEmbed, COLOR_ERROR };
