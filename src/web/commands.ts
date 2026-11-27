import type { PlayerApi } from './player.js';
import { extractRadioRequest } from '../utils/radio.js';
import { extractMusicRequest } from '../voice/music.js';
import { modeCommand, normalizeSpeech, ruleIntent } from '../voice/intents.js';
import { formatDuration, parseDuration } from '../utils/duration.js';
import type { CommandResult, QueueView, WebTarget } from './types.js';

const EXIT = /^(?:хватит|выход|выйти|пока|до свидания|закончить|завершить|закрыть|отмена)$/u;
const HELP = /^(?:помощь|что ты умеешь|что умеешь|команды|как пользоваться|как тебя использовать)$/u;
const GREETING = /^(?:привет|приветик|здравствуй|здравствуйте|доброе утро|добрый день|добрый вечер|доброй ночи|хай|ку|йоу)$/u;
const STATUS = /^(?:что (?:сейчас )?играет(?: сейчас)?|что за (?:трек|песня|композиция)(?: играет)?|какой (?:трек|песня|композиция)(?: играет)?|какая (?:песня|композиция|музыка)(?: играет| звучит)?|что в очереди|статус|что происходит)$/u;
const RESUME_PREFIX = /^(?:включи|включай|включить|продолжи|продолжай|возобнови)(?: (?:музыку|воспроизведение|игру|плеер))?$/u;

const PAIR = [
  /^(?:подключи|подключить|привяжи|привязать|свяжи|связать)(?: (?:колонку|колонка|колонке|устройство|алису|динамик|спикер))?(?: (?:к|по|с) ?(?:коду|кодом))?(?: код)? (\d{6})$/u,
  /^(?:код|пароль) (\d{6})$/u,
  /^(\d{6})$/u,
];

/** Recognises a spoken pairing code from the website. */
export function parsePairingCode(text: string): string | null {
  const words = normalizeSpeech(text);
  for (const pattern of PAIR) {
    const match = pattern.exec(words);
    if (match?.[1]) return match[1];
  }
  return null;
}

export function greeting(skillName: string): string {
  return `Это пульт ${skillName}. Скажите «подключи колонку» и код с сайта, а потом просто называйте музыку.`
    + ' Например: включи Кино, следующий трек, пауза, что играет.';
}

export function helpText(skillName: string): string {
  return `Я включаю музыку в Discord. Скажите: включи Кино, следующий трек, пауза, продолжай, стоп,`
    + ` громкость пятьдесят, что играет. Колонка привязывается кодом с сайта пульта ${skillName}.`;
}

function clip(value: string, max = 120): string {
  const text = value.replace(/\s+/gu, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function failure(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : '';
  if (/ничего не найдено|не нашл/iu.test(message)) return 'Ничего не нашла.';
  if (/другом голосовом канале/iu.test(message)) return 'Бот занят в другом голосовом канале.';
  if (/прав/iu.test(message)) return 'Нет прав на подключение и речь в этом канале.';
  if (/рекомендации|повтор недоступен/iu.test(message)) return 'Сейчас это недоступно.';
  if (/радио|эфир/iu.test(message)) return 'Радиостанция сейчас недоступна.';
  if (/сервер discord не найден/iu.test(message)) return 'Сервер Discord не найден.';
  if (/голосовой канал не найден/iu.test(message)) return 'Голосовой канал не найден. Проверьте настройки колонки.';
  return fallback;
}

function status(view: QueueView | null): string {
  if (!view) return 'Сейчас ничего не играет.';
  if (!view.current) return `Сейчас ничего не играет. В очереди ${view.upcomingCount}. Громкость ${view.volume} процентов.`;
  const track = view.current;
  if (track.isLive) {
    return `${view.paused ? 'На паузе' : 'Прямой эфир'}: ${clip(track.title)}. Громкость ${view.volume} процентов.`;
  }
  const total = parseDuration(track.duration);
  const position = total ? `, ${formatDuration(Math.min(view.elapsed, total))} из ${formatDuration(total)}` : '';
  const tail = view.upcomingCount ? ` В очереди ${view.upcomingCount}.` : '';
  return `${view.paused ? 'На паузе' : 'Сейчас играет'}: ${clip(track.title)}${position}.${tail}`.trim();
}

async function executeIntent(player: PlayerApi, target: WebTarget, intent: ReturnType<typeof ruleIntent>): Promise<CommandResult> {
  const reply = (text: string, ok = true): CommandResult => ({ text, keepSession: true, ok });
  switch (intent.action) {
    case 'volume_set': {
      const result = await player.act(target, 'volume', intent.level);
      return result.changed ? reply(`Громкость ${result.level ?? intent.level} процентов.`)
        : reply('Бот сейчас не играет.', false);
    }
    case 'volume_up':
    case 'volume_down': {
      const view = player.queue(target);
      if (!view || !view.current) return reply('Бот сейчас не играет.', false);
      const next = Math.max(1, Math.min(150, view.volume + (intent.action === 'volume_up' ? 10 : -10)));
      const result = await player.act(target, 'volume', next);
      return reply(`Громкость ${result.level ?? next} процентов.`);
    }
    case 'pause': {
      const result = await player.act(target, 'pause');
      return result.changed ? reply('Пауза.') : reply('Сейчас нечего ставить на паузу.', false);
    }
    case 'resume': {
      const result = await player.act(target, 'resume');
      return result.changed ? reply('Продолжаю.') : reply('Сейчас нечего продолжать.', false);
    }
    case 'skip': {
      const result = await player.act(target, 'skip');
      return result.changed ? reply('Следующий трек.') : reply('Очередь пуста.', false);
    }
    case 'stop': {
      const result = await player.act(target, 'stop');
      return reply(result.changed ? 'Останавливаю музыку.' : 'Музыка уже остановлена.');
    }
    case 'queue_clear': {
      const result = await player.act(target, 'clear');
      return reply(result.count ? `Убрала из очереди ${result.count}.` : 'Очередь уже пуста.');
    }
    case 'autoplay_on':
    case 'autoplay_off': {
      if (!player.queue(target)?.current) return reply('Сначала включите музыку.', false);
      const enabled = intent.action === 'autoplay_on';
      await player.act(target, 'autoplay', enabled);
      return reply(`Бесконечный режим ${enabled ? 'включён' : 'выключен'}.`);
    }
    case 'loop_on':
    case 'loop_off': {
      if (!player.queue(target)?.current) return reply('Сначала включите музыку.', false);
      const enabled = intent.action === 'loop_on';
      await player.act(target, 'loop', enabled);
      return reply(`Повтор трека ${enabled ? 'включён' : 'выключен'}.`);
    }
    case 'voice_on':
    case 'voice_off':
      return reply('Голосовое управление в Discord настраивается отдельно, командой голос.');
    default:
      return reply('Не поняла команду.', false);
  }
}

/**
 * Executes one spoken request against one paired speaker target. The function
 * has no HTTP or Discord types, so it is shared by the Alice webhook and the
 * local bridge API and can be tested with a stub player.
 */
export async function runVoiceCommand(player: PlayerApi, target: WebTarget, text: string,
  skillName = 'Муза'): Promise<CommandResult> {
  // A local bridge often forwards "Алиса, включи Кино в дискорде"; the wake
  // word and destination are not part of the search query.
  const raw = text.trim()
    .replace(/^\s*(?:алиса|алеса)[\s,.:;!—–-]+/iu, '')
    .replace(/\s+(?:в|во|через)\s+(?:дискорде|боте|бота|колонке|колонку)\s*$/iu, '');
  if (!raw) return { text: helpText(skillName), keepSession: true, ok: true };
  const words = normalizeSpeech(raw);
  if (EXIT.test(words)) return { text: 'До встречи.', keepSession: false, ok: true };
  if (HELP.test(words)) return { text: helpText(skillName), keepSession: true, ok: true };
  if (GREETING.test(words)) return { text: greeting(skillName), keepSession: true, ok: true };
  if (STATUS.test(words)) return { text: status(player.queue(target)), keepSession: true, ok: true };

  const intent = ruleIntent(raw);
  if (intent.action !== 'unknown') {
    try { return await executeIntent(player, target, intent); }
    catch (error) { return { text: failure(error, 'Не получилось выполнить команду.'), keepSession: true, ok: false }; }
  }

  // "Включи" without a title resumes only a paused player, matching the voice
  // control behaviour. Otherwise the request continues to the search below.
  const view = player.queue(target);
  if (RESUME_PREFIX.test(words) && view?.paused) {
    try {
      const result = await player.act(target, 'resume');
      if (result.changed) return { text: 'Продолжаю.', keepSession: true, ok: true };
    } catch { /* fall through to the generic reply */ }
  }

  const radio = extractRadioRequest(raw, true);
  if (radio?.kind === 'station') {
    try {
      await player.radio(target, radio.station, 'Алиса');
      return { text: `Включаю ${radio.station.name}.`, keepSession: true, ok: true };
    } catch (error) {
      return { text: failure(error, 'Не удалось включить радио.'), keepSession: true, ok: false };
    }
  }
  if (radio?.kind === 'unsupported') return { text: 'Такой радиостанции у меня нет.', keepSession: true, ok: false };
  if (radio?.kind === 'rejected') return { text: 'Не поняла команду.', keepSession: true, ok: false };

  const request = extractMusicRequest(raw, true);
  if (request) {
    try {
      const result = await player.play(target, request.query, 'Алиса', { source: 'youtube', fast: true });
      const action = result.started ? 'Включаю' : 'Добавила в очередь';
      return { text: `${action}: ${clip(result.track.title)}.`, keepSession: true, ok: true };
    } catch (error) {
      return { text: failure(error, 'Не получилось включить трек.'), keepSession: true, ok: false };
    }
  }

  // A mode request that `ruleIntent` could not confirm still deserves a clear
  // answer rather than a YouTube search for a control phrase.
  const mode = modeCommand(raw);
  if (mode && mode.action === 'unknown') return { text: 'Не поняла команду.', keepSession: true, ok: false };
  return { text: 'Не поняла. Скажите «включи» и название песни или исполнителя.', keepSession: true, ok: false };
}
