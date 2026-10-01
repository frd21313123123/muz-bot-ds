import { ActionRowBuilder, ButtonBuilder, ButtonStyle, EmbedBuilder, escapeMarkdown } from 'discord.js';
import type { Track } from '../types.js';
import type { GuildQueue } from './GuildQueue.js';
import { formatDuration, parseDuration } from './duration.js';

const cut = (value: string, max: number): string => value.length > max ? `${value.slice(0, max - 1)}…` : value;

export function nowPlayingEmbed(track: Track, autoplay = false, speed = 1): EmbedBuilder {
  const embed = new EmbedBuilder()
    .setColor(track.isAutoplay ? 0x57f287 : 0x1db954)
    .setTitle(track.isAutoplay ? '🤖 Автовоспроизведение' : '▶ Сейчас играет')
    .setDescription(`**[${escapeMarkdown(cut(track.title, 200))}](${track.url})**`)
    .addFields(
      { name: '⏱ Длительность', value: track.isLive ? '🔴 Прямой эфир'
        : parseDuration(track.duration) === null ? 'Длительность неизвестна' : track.duration, inline: true },
      { name: '👤 Запросил', value: escapeMarkdown(cut(track.requestedBy, 100)), inline: true },
      { name: '⏩ Скорость', value: `${speed}×`, inline: true },
    );
  if (autoplay) embed.addFields({ name: '♾', value: 'Бесконечное воспроизведение включено' });
  if (track.thumbnail) embed.setThumbnail(track.thumbnail);
  return embed;
}

export function queueEmbed(queue: GuildQueue, page = 1): EmbedBuilder {
  const totalPages = Math.max(1, Math.ceil(queue.tracks.length / 10));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const start = (safePage - 1) * 10;
  const embed = new EmbedBuilder().setColor(0x5865f2).setTitle('📋 Очередь воспроизведения');
  if (queue.currentTrack) {
    embed.addFields({ name: '▶ Сейчас играет', value: `[${escapeMarkdown(cut(queue.currentTrack.title, 160))}](${queue.currentTrack.url})` });
  }
  const lines = queue.tracks.slice(start, start + 10)
    .map((track, i) => `\`${start + i + 1}.\` [${escapeMarkdown(cut(track.title, 100))}](${track.url}) — ${track.duration}`);
  if (lines.length) embed.addFields({ name: '📃 Следующие треки', value: cut(lines.join('\n'), 1024) });
  if (!lines.length && !queue.currentTrack) embed.setDescription('Очередь пуста. Добавьте трек командой `/play`.');
  return embed.setFooter({ text: `Страница ${safePage}/${totalPages} • Бесконечное: ${queue.autoplay ? 'вкл' : 'выкл'}` });
}

export function playerEmbed(queue: GuildQueue): EmbedBuilder {
  const track = queue.currentTrack;
  if (!track) return new EmbedBuilder().setColor(0x99aab5).setTitle('⏹ Ничего не играет');
  const paused = queue.isPaused;
  const duration = parseDuration(track.duration);
  const elapsed = Math.max(0, queue.getElapsedSeconds());
  const filled = duration ? Math.min(12, Math.round(elapsed / duration * 12)) : 0;
  const progress = track.isLive ? '🔴 Прямой эфир' : duration
    ? `${'▰'.repeat(filled)}${'▱'.repeat(12 - filled)}\n${formatDuration(Math.min(elapsed, duration))} / ${formatDuration(duration)}`
    : `${formatDuration(elapsed)} / длительность неизвестна`;
  const embed = new EmbedBuilder()
    .setColor(paused ? 0x99aab5 : 0x1db954)
    .setTitle(paused ? '⏸ На паузе' : track.isAutoplay ? '🤖 Автовоспроизведение' : '▶ Сейчас играет')
    .setDescription(`**[${escapeMarkdown(cut(track.title, 200))}](${track.url})**\n\n${progress}`)
    .addFields(
      { name: '🔊 Громкость', value: `${Math.round(queue.volume * 100)}%`, inline: true },
      { name: '⏩ Скорость', value: `${queue.speed}×`, inline: true },
      { name: '♾ Бесконечное', value: queue.autoplay ? 'Вкл' : 'Выкл', inline: true },
      { name: '📋 В очереди', value: `${queue.tracks.length}`, inline: true },
    );
  if (track.thumbnail) embed.setThumbnail(track.thumbnail);
  if (queue.tracks[0]) embed.setFooter({ text: cut(`Далее: ${queue.tracks[0].title}`, 200) });
  return embed;
}

export function playerActionRow(queue: GuildQueue): ActionRowBuilder<ButtonBuilder> {
  const paused = queue.isPaused;
  return new ActionRowBuilder<ButtonBuilder>().addComponents(
    new ButtonBuilder().setCustomId('player_playpause').setEmoji(paused ? '▶' : '⏸').setStyle(ButtonStyle.Secondary),
    new ButtonBuilder().setCustomId('player_skip').setEmoji('⏭').setStyle(ButtonStyle.Primary),
    new ButtonBuilder().setCustomId('player_stop').setEmoji('⏹').setStyle(ButtonStyle.Danger),
    new ButtonBuilder().setCustomId('player_autoplay').setEmoji('♾️')
      .setLabel(queue.autoplay ? 'Беск: Вкл' : 'Беск: Выкл').setStyle(ButtonStyle.Secondary),
    new ButtonBuilder().setCustomId('player_loop').setEmoji('🔂')
      .setLabel(queue.loopCurrent ? '1 трек: Вкл' : '1 трек: Выкл').setStyle(ButtonStyle.Secondary),
  );
}
