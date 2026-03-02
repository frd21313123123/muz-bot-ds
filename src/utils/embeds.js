'use strict';

const { EmbedBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } = require('discord.js');

const COLOR_PLAYING = 0x1DB954;  // Spotify green
const COLOR_QUEUE   = 0x5865F2;  // Blurple
const COLOR_ERROR   = 0xED4245;  // Red
const COLOR_IDLE    = 0x99AAB5;  // Grey

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

// ─── Player helpers ────────────────────────────────────────────────────────

/**
 * Парсит строку длительности ("4:30", "1:02:15") в секунды.
 * @returns {number|null}
 */
function parseDurationToSeconds(str) {
  if (!str || str === '?' || str === '—') return null;
  const parts = str.split(':').map(Number);
  if (parts.some(isNaN)) return null;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 1) return parts[0];
  return null;
}

/**
 * Форматирует секунды в строку "m:ss" или "h:mm:ss".
 */
function formatSeconds(sec) {
  if (sec == null || sec < 0) return '0:00';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

/**
 * Строит прогресс-бар.
 */
function buildProgressBar(elapsed, total, length = 12) {
  if (!total || total <= 0) {
    return `🔴 ${formatSeconds(elapsed)} / Прямой эфир`;
  }
  const ratio = Math.min(elapsed / total, 1);
  const filled = Math.round(ratio * length);
  const empty = length - filled;
  const bar = '▰'.repeat(filled) + '▱'.repeat(empty);
  return `${bar}  ${formatSeconds(elapsed)} / ${formatSeconds(total)}`;
}

/**
 * Embed для интерактивного плеера.
 * @param {import('./GuildQueue')} queue
 */
function playerEmbed(queue) {
  const { AudioPlayerStatus } = require('@discordjs/voice');

  if (!queue.currentTrack) {
    return new EmbedBuilder()
      .setColor(COLOR_IDLE)
      .setTitle('⏹ Ничего не играет')
      .setDescription('Добавьте трек командой `/play`.');
  }

  const track = queue.currentTrack;
  const elapsed = queue.getElapsedSeconds();
  const totalSec = parseDurationToSeconds(track.duration);
  const isPaused = queue.player.state.status === AudioPlayerStatus.Paused;

  const statusIcon = isPaused ? '⏸' : (track.isAutoplay ? '🤖' : '▶');
  const statusText = isPaused ? 'На паузе' : (track.isAutoplay ? 'Автовоспроизведение' : 'Сейчас играет');

  const embed = new EmbedBuilder()
    .setColor(isPaused ? COLOR_IDLE : (track.isAutoplay ? 0x57F287 : COLOR_PLAYING))
    .setTitle(`${statusIcon} ${statusText}`)
    .setDescription(`**[${track.title}](${track.url})**\n\n${buildProgressBar(elapsed, totalSec)}`)
    .addFields(
      { name: '🔊 Громкость', value: `${Math.round(queue.volume * 100)}%`, inline: true },
      { name: '🔁 Авто', value: queue.autoplay ? 'Вкл' : 'Выкл', inline: true },
      { name: '📋 В очереди', value: `${queue.tracks.length}`, inline: true },
    );

  if (track.thumbnail) {
    embed.setThumbnail(track.thumbnail);
  }

  // Footer: следующий трек
  if (queue.tracks.length > 0) {
    const next = queue.tracks[0];
    embed.setFooter({ text: `Далее: ${next.title} (${next.duration})` });
  } else if (queue.autoplay) {
    embed.setFooter({ text: 'Далее: автоподбор по рекомендациям' });
  }

  return embed;
}

/**
 * ActionRow с кнопками управления плеером.
 * @param {import('./GuildQueue')} queue
 */
function playerActionRow(queue) {
  const { AudioPlayerStatus } = require('@discordjs/voice');
  const isPaused = queue.player.state.status === AudioPlayerStatus.Paused;

  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId('player_playpause')
      .setEmoji(isPaused ? '▶' : '⏸')
      .setStyle(isPaused ? ButtonStyle.Success : ButtonStyle.Secondary),
    new ButtonBuilder()
      .setCustomId('player_skip')
      .setEmoji('⏭')
      .setStyle(ButtonStyle.Primary),
    new ButtonBuilder()
      .setCustomId('player_stop')
      .setEmoji('⏹')
      .setStyle(ButtonStyle.Danger),
    new ButtonBuilder()
      .setCustomId('player_autoplay')
      .setEmoji('🔁')
      .setLabel(queue.autoplay ? 'Вкл' : 'Выкл')
      .setStyle(queue.autoplay ? ButtonStyle.Success : ButtonStyle.Secondary),
  );
}

module.exports = {
  nowPlayingEmbed,
  queueEmbed,
  playerEmbed,
  playerActionRow,
  parseDurationToSeconds,
  formatSeconds,
  buildProgressBar,
  COLOR_ERROR,
};
