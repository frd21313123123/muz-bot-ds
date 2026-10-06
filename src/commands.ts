import {
  InviteTargetType, MessageFlags, PermissionsBitField, SlashCommandBuilder, escapeMarkdown,
  type ChatInputCommandInteraction,
} from 'discord.js';
import type { Command, MusicClient } from './types.js';
import { GuildQueue } from './utils/GuildQueue.js';
import { nowPlayingEmbed, playerActionRow, playerEmbed, queueEmbed } from './utils/embeds.js';
import { resolveQuery } from './utils/resolve.js';
import { canControl, memberVoiceChannel } from './utils/voiceAccess.js';
import { MIN_SPEED, MAX_SPEED } from './utils/audio.js';
import { extractRadioRequest, findRadioStation, type RadioStation } from './utils/radio.js';

async function startRadio(interaction: ChatInputCommandInteraction, client: MusicClient, station: RadioStation): Promise<unknown> {
  if (!interaction.guildId) return interaction.editReply('❌ Команда работает только на сервере.');
  const channel = memberVoiceChannel(interaction);
  if (!channel) return interaction.editReply('❌ Войдите в голосовой канал.');
  const me = interaction.guild?.members.me ?? await interaction.guild?.members.fetchMe().catch(() => null);
  const permissions = me && channel.permissionsFor(me);
  if (!permissions?.has(PermissionsBitField.Flags.Connect) || !permissions.has(PermissionsBitField.Flags.Speak)) {
    return interaction.editReply('❌ Нет прав на подключение и речь в этом канале.');
  }
  let queue = client.queues.get(interaction.guildId);
  if (queue?.voiceChannel && queue.voiceChannel.id !== channel.id) {
    return interaction.editReply('❌ Войдите в тот же голосовой канал, что и бот.');
  }
  const created = !queue;
  if (!queue) { queue = new GuildQueue(interaction.guildId, client); client.queues.set(interaction.guildId, queue); }
  try {
    await queue.join(channel);
    const valid = (): boolean => memberVoiceChannel(interaction)?.id === channel.id && !queue!.closed;
    const requestedBy = interaction.member && 'displayName' in interaction.member
      ? interaction.member.displayName : interaction.user.username;
    const changed = await queue.playRadio(station, requestedBy, { valid, userId: interaction.user.id });
    if (!changed) return interaction.editReply('ℹ️ Запрос радио отменён.');
    queue.setPlayerChannel(interaction.channelId);
    return interaction.editReply({ content: `📻 Включено **${escapeMarkdown(station.name)}** — прямой эфир. Очередь очищена.`,
      allowedMentions: { parse: [] } });
  } catch {
    if (created && !queue.currentTrack && !queue.tracks.length) await queue.stop().catch(() => {});
    return interaction.editReply('❌ Не удалось включить радио. Эфир недоступен; попробуйте другую станцию.');
  }
}

const radio: Command = {
  data: new SlashCommandBuilder().setName('radio').setDescription('📻 Включить прямой эфир радиостанции')
    .addStringOption(option => option.setName('station').setDescription('Название радиостанции')
      .setRequired(true).setAutocomplete(true).setMinLength(1).setMaxLength(100)),
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    const station = findRadioStation(interaction.options.getString('station', true));
    if (!station) return interaction.editReply('❌ Радиостанция не поддерживается. Выберите станцию из подсказок /radio.');
    return startRadio(interaction, client, station);
  },
};

const privateReply = (interaction: ChatInputCommandInteraction, content: string): Promise<unknown> =>
  interaction.reply({ content, flags: MessageFlags.Ephemeral, allowedMentions: { parse: [] } });

const voice: Command = {
  data: new SlashCommandBuilder().setName('voice').setDescription('🎙 Голосовое управление')
    .addSubcommand((command) => command.setName('on').setDescription('Включить прослушивание имени'))
    .addSubcommand((command) => command.setName('off').setDescription('Отключить прослушивание'))
    .addSubcommand((command) => command.setName('status').setDescription('Состояние голосового управления'))
    .addSubcommand((command) => command.setName('name').setDescription('Изменить обращение; без имени — вернуть имя Discord')
      .addStringOption((option) => option.setName('value').setDescription('Новое обращение к боту').setMinLength(2).setMaxLength(64))),
  async execute(interaction, client) {
    if (!interaction.guildId) return privateReply(interaction, '❌ Команда работает только на сервере.');
    const queue = client.queues.get(interaction.guildId);
    const action = interaction.options.getSubcommand();
    if (action === 'status') {
      const name = queue?.wakeName ?? client.voiceSettings?.get(interaction.guildId)
        ?? interaction.guild?.members.me?.displayName ?? client.user?.username ?? '';
      const runtime = client.voiceRuntime;
      const stt = runtime?.sttProvider === 'groq' ? 'Groq · Whisper Large V3 Turbo' : 'локальная Whisper';
      return privateReply(interaction, `🎙 ${queue?.voice?.enabled ? 'Прослушивание включено' : 'Прослушивание выключено'}. Обращение: **${escapeMarkdown(name)}**.\nПрофиль: ${runtime?.profile ?? 'light'}. Распознавание: ${stt}. Обработчик ${runtime?.ready ? 'готов' : runtime?.prepared ? 'запустится при включении прослушивания' : 'не подготовлен — проверьте .env и выполните npm run setup:voice'}.`);
    }
    if (!await requireController(interaction, queue)) return;
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    if (action === 'name') {
      const value = interaction.options.getString('value');
      if (!client.voiceSettings) return interaction.editReply('❌ Хранилище настроек недоступно.');
      try {
        queue!.voice?.reset();
        await client.voiceSettings.set(interaction.guildId, value);
        queue!.voice?.reset();
        return interaction.editReply({ content: `🎙 Новое обращение: **${escapeMarkdown(queue!.wakeName)}**.`, allowedMentions: { parse: [] } });
      } catch { return interaction.editReply('❌ Не удалось сохранить имя. Используйте 2–64 символа с буквами или цифрами.'); }
    }
    const enabled = action === 'on';
    if (!await queue!.setVoiceEnabled(enabled)) return interaction.editReply('❌ Голосовой обработчик недоступен. Проверьте .env, выполните npm run setup:voice и повторите /voice on.');
    return interaction.editReply({ content: enabled
      ? `🎙 Прослушивание включено. Назовите **${escapeMarkdown(queue!.wakeName)}**, дождитесь сигнала и произнесите команду.`
      : '🎙 Прослушивание выключено.', allowedMentions: { parse: [] } });
  },
};

async function requireController(interaction: ChatInputCommandInteraction, queue: GuildQueue | undefined): Promise<boolean> {
  if (!queue?.voiceChannel) {
    await privateReply(interaction, 'ℹ️ Бот сейчас не в голосовом канале.');
    return false;
  }
  if (!canControl(interaction, queue.voiceChannel.id)) {
    await privateReply(interaction, '❌ Войдите в тот же голосовой канал, что и бот.');
    return false;
  }
  return true;
}

const join: Command = {
  data: new SlashCommandBuilder().setName('join').setDescription('🎙 Войти в ваш голосовой канал и слушать команды'),
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    if (!interaction.guildId) return interaction.editReply('❌ Команда работает только на сервере.');
    const channel = memberVoiceChannel(interaction);
    if (!channel) return interaction.editReply('❌ Войдите в голосовой канал.');
    const me = interaction.guild?.members.me ?? await interaction.guild?.members.fetchMe().catch(() => null);
    const permissions = me && channel.permissionsFor(me);
    if (!permissions?.has(PermissionsBitField.Flags.Connect) || !permissions.has(PermissionsBitField.Flags.Speak)) {
      return interaction.editReply('❌ Нет прав на подключение и речь в этом канале.');
    }
    if (!client.voiceRuntime || !await client.voiceRuntime.start()) {
      return interaction.editReply('❌ Голосовой обработчик недоступен. Проверьте .env, выполните npm run setup:voice и повторите /join.');
    }
    // Startup and member fetching may take time; use the latest channel state.
    if (memberVoiceChannel(interaction)?.id !== channel.id) {
      return interaction.editReply('❌ Вы вышли из голосового канала или сменили его. Повторите /join.');
    }
    let queue = client.queues.get(interaction.guildId);
    if (queue?.voiceChannel && queue.voiceChannel.id !== channel.id) {
      return interaction.editReply('❌ Бот уже работает в другом голосовом канале.');
    }
    const created = !queue;
    if (!queue) {
      queue = new GuildQueue(interaction.guildId, client);
      client.queues.set(interaction.guildId, queue);
    }
    try {
      await queue.join(channel);
      if (memberVoiceChannel(interaction)?.id !== channel.id) {
        if (created) await queue.stop();
        return interaction.editReply('❌ Вы вышли из голосового канала или сменили его. Повторите /join.');
      }
      if (!await queue.setVoiceEnabled(true)) {
        if (created) await queue.stop();
        return interaction.editReply('❌ Не удалось включить прослушивание. Проверьте /voice status и повторите /join.');
      }
      queue.setPlayerChannel(interaction.channelId);
      return interaction.editReply({
        content: `🎙 Прослушивание включено. Назовите **${escapeMarkdown(queue.wakeName)}**, дождитесь сигнала и произнесите команду, например «включи Numb».`,
        allowedMentions: { parse: [] },
      });
    } catch (error) {
      if (created) await queue.stop().catch(() => {});
      console.error('[join]', error);
      return interaction.editReply('❌ Не удалось подключиться или включить прослушивание. Повторите /join.');
    }
  },
};

const play: Command = {
  data: new SlashCommandBuilder().setName('play').setDescription('▶ Воспроизвести трек или включить радио')
    .addStringOption((option) => option.setName('query').setDescription('Ссылка YouTube, название песни или радиостанции')
      .setRequired(true).setMinLength(1).setMaxLength(500))
    .addBooleanOption((option) => option.setName('infinite').setDescription('Добавлять рекомендации после очереди')),
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    const radioRequest = extractRadioRequest(interaction.options.getString('query', true), true);
    if (radioRequest?.kind === 'station') return startRadio(interaction, client, radioRequest.station);
    if (radioRequest) return interaction.editReply(radioRequest.kind === 'unsupported'
      ? '❌ Радиостанция не поддерживается. Выберите станцию из подсказок /radio.' : '❌ Укажите одну станцию или песню без отрицания.');
    if (!interaction.guildId) return interaction.editReply('❌ Команда работает только на сервере.');
    const channel = memberVoiceChannel(interaction);
    if (!channel) return interaction.editReply('❌ Войдите в голосовой канал.');
    const me = interaction.guild?.members.me ?? await interaction.guild?.members.fetchMe().catch(() => null);
    const permissions = me && channel.permissionsFor(me);
    if (!permissions?.has(PermissionsBitField.Flags.Connect) || !permissions.has(PermissionsBitField.Flags.Speak)) {
      return interaction.editReply('❌ Нет прав на подключение и речь в этом канале.');
    }
    let queue = client.queues.get(interaction.guildId);
    if (queue?.voiceChannel && queue.voiceChannel.id !== channel.id) {
      return interaction.editReply('❌ Бот уже работает в другом голосовом канале.');
    }
    const created = !queue;
    if (!queue) {
      queue = new GuildQueue(interaction.guildId, client);
      client.queues.set(interaction.guildId, queue);
    }
    try {
      await queue.join(channel);
      const infinite = interaction.options.getBoolean('infinite');
      const result = await resolveQuery(interaction.options.getString('query', true),
        interaction.member && 'displayName' in interaction.member ? interaction.member.displayName : interaction.user.username,
        client.ytdlp, (infinite ?? queue.autoplay) ? 25 : 1);
      if (!result) {
        if (created) await queue.stop();
        return interaction.editReply('❌ Ничего не найдено.');
      }
      if (memberVoiceChannel(interaction)?.id !== channel.id || queue.closed) {
        if (created && !queue.closed && !queue.currentTrack && !queue.tracks.length) await queue.stop();
        return interaction.editReply('❌ Вы вышли из голосового канала или сменили его. Повторите /play.');
      }
      queue.setPlayerChannel(interaction.channelId);
      const wasIdle = queue.isRadio || (!queue.currentTrack && queue.tracks.length === 0);
      if (result.type === 'playlist') {
        await queue.addTracks(result.tracks);
        if (infinite !== null) queue.setAutoplay(infinite);
        return interaction.editReply({
          content: `✅ Добавлено ${result.tracks.length} треков из **${escapeMarkdown(result.name)}**${wasIdle ? ' — воспроизведение началось.' : '.'}`,
          allowedMentions: { parse: [] },
        });
      }
      await queue.addTrack(result.track);
      if (infinite !== null) queue.setAutoplay(infinite);
      return interaction.editReply({
        content: `${wasIdle ? '▶ Воспроизведение началось' : '✅ Добавлено в очередь'}: **${escapeMarkdown(result.track.title)}**`,
        allowedMentions: { parse: [] },
      });
    } catch (error) {
      if (created) await queue.stop().catch(() => {});
      console.error('[play]', error);
      return interaction.editReply(`❌ Не удалось запустить трек: ${error instanceof Error ? error.message : String(error)}`);
    }
  },
};

const skip: Command = {
  data: new SlashCommandBuilder().setName('skip').setDescription('⏭ Пропустить текущий трек'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    if (!queue?.currentTrack) return privateReply(interaction, '❌ Сейчас ничего не играет.');
    const title = queue.currentTrack.title;
    queue.skip();
    return privateReply(interaction, `⏭ Пропущено: **${escapeMarkdown(title)}**`);
  },
};

const stop: Command = {
  data: new SlashCommandBuilder().setName('stop').setDescription('⏹ Остановить воспроизведение и выйти'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    await queue!.stop();
    return privateReply(interaction, '⏹ Воспроизведение остановлено.');
  },
};

const pause: Command = {
  data: new SlashCommandBuilder().setName('pause').setDescription('⏸ Пауза'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    return privateReply(interaction, queue!.pause() ? '⏸ Пауза.' : '❌ Воспроизведение уже на паузе.');
  },
};

const resume: Command = {
  data: new SlashCommandBuilder().setName('resume').setDescription('▶ Продолжить воспроизведение'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    return privateReply(interaction, queue!.resume() ? '▶ Воспроизведение продолжено.' : '❌ Бот не на паузе.');
  },
};

const clear: Command = {
  data: new SlashCommandBuilder().setName('clear').setDescription('🗑 Очистить очередь'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    return privateReply(interaction, `🗑 Удалено треков: ${queue!.clearQueue()}.`);
  },
};

const volume: Command = {
  data: new SlashCommandBuilder().setName('volume').setDescription('🔊 Установить громкость')
    .addIntegerOption((option) => option.setName('level').setDescription('От 1 до 150%')
      .setRequired(true).setMinValue(1).setMaxValue(150)),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    const level = interaction.options.getInteger('level', true);
    queue!.setVolume(level);
    return privateReply(interaction, `🔊 Громкость: ${level}%`);
  },
};

const autoplay: Command = {
  data: new SlashCommandBuilder().setName('autoplay').setDescription('♾ Переключить рекомендации'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    if (queue!.isRadio) return privateReply(interaction, 'ℹ️ Рекомендации недоступны во время радио.');
    return privateReply(interaction, `♾ Бесконечное воспроизведение ${queue!.toggleAutoplay() ? 'включено' : 'выключено'}.`);
  },
};

const speed: Command = {
  data: new SlashCommandBuilder().setName('speed').setDescription('⏩ Изменить скорость и тональность музыки')
    .addNumberOption((option) => option.setName('value').setDescription('От 0.5 до 2; 1 — обычная скорость')
      .setRequired(true).setMinValue(MIN_SPEED).setMaxValue(MAX_SPEED)),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!await requireController(interaction, queue)) return;
    const value = interaction.options.getNumber('value', true);
    try {
      queue!.setSpeed(value);
      return privateReply(interaction, `⏩ Скорость: ${value}×. ${value === 1 ? 'Обычная тональность.'
        : value < 1 ? 'Музыка медленнее, тональность ниже.' : 'Музыка быстрее, тональность выше.'}`);
    } catch (error) {
      return privateReply(interaction, `❌ ${error instanceof Error ? error.message : 'Не удалось изменить скорость.'}`);
    }
  },
};

const nowplaying: Command = {
  data: new SlashCommandBuilder().setName('nowplaying').setDescription('🎵 Текущий трек'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!queue?.currentTrack) return privateReply(interaction, '❌ Сейчас ничего не играет.');
    return interaction.reply({ embeds: [nowPlayingEmbed(queue.currentTrack, queue.autoplay, queue.speed)], flags: MessageFlags.Ephemeral });
  },
};

const queueCommand: Command = {
  data: new SlashCommandBuilder().setName('queue').setDescription('📋 Показать очередь')
    .addIntegerOption((option) => option.setName('page').setDescription('Номер страницы').setMinValue(1)),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!queue?.currentTrack && !queue?.tracks.length) return privateReply(interaction, '📋 Очередь пуста.');
    return interaction.reply({ embeds: [queueEmbed(queue, interaction.options.getInteger('page') ?? 1)], flags: MessageFlags.Ephemeral });
  },
};

const player: Command = {
  data: new SlashCommandBuilder().setName('player').setDescription('🎛 Публичный плеер с кнопками'),
  async execute(interaction, client) {
    const queue = client.queues.get(interaction.guildId!);
    if (!queue?.connection) return privateReply(interaction, 'ℹ️ Бот не в голосовом канале.');
    await interaction.deferReply();
    const message = await interaction.editReply({ embeds: [playerEmbed(queue)], components: [playerActionRow(queue)] });
    await queue.showPlayer(message, interaction.channelId);
  },
};

const watch: Command = {
  data: new SlashCommandBuilder().setName('watch').setDescription('📺 Запустить Watch Together'),
  async execute(interaction, client) {
    await interaction.deferReply({ flags: MessageFlags.Ephemeral });
    const channel = memberVoiceChannel(interaction);
    if (!channel || !interaction.guild) return interaction.editReply('❌ Войдите в голосовой канал.');
    const me = interaction.guild.members.me ?? await interaction.guild.members.fetchMe();
    const permissions = channel.permissionsFor(me);
    if (!permissions?.has(PermissionsBitField.Flags.ViewChannel)
      || !permissions.has(PermissionsBitField.Flags.CreateInstantInvite)
      || !permissions.has(PermissionsBitField.Flags.UseEmbeddedActivities)) {
      return interaction.editReply('❌ Нужны права на просмотр канала, приглашения и активности.');
    }
    try {
      const invite = await channel.createInvite({
        maxAge: 0, maxUses: 0, targetType: InviteTargetType.EmbeddedApplication,
        targetApplication: process.env.WATCH_TOGETHER_APP_ID || '880218394199220334',
        reason: `Watch Together requested by ${interaction.user.tag}`,
      });
      const queue = client.queues.get(interaction.guildId!);
      const track = queue?.voiceChannel?.id === channel.id ? queue.currentTrack : null;
      return interaction.editReply(`📺 Совместный просмотр: ${invite.url}${track ? `\n🎵 Текущий трек: ${track.url}` : ''}`);
    } catch (error) {
      console.error('[watch]', error);
      return interaction.editReply('❌ Не удалось запустить Watch Together.');
    }
  },
};

export const commands: Command[] = [
  join, play, radio, skip, stop, pause, resume, clear, volume, speed, autoplay, nowplaying, queueCommand, player, watch, voice,
];
