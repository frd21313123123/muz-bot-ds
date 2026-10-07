import { extraVoiceCommand } from './phrases.js';

export type VoiceAction = 'skip' | 'pause' | 'resume' | 'stop' | 'volume_set' | 'volume_up' | 'volume_down'
  | 'autoplay_on' | 'autoplay_off' | 'loop_on' | 'loop_off' | 'queue_clear' | 'voice_on' | 'voice_off' | 'unknown';
export type VoiceIntent = { action: 'volume_set'; level: number }
  | { action: Exclude<VoiceAction, 'volume_set'> };
export interface VoiceDecision { action: VoiceAction; confidence: number }

export function normalizeSpeech(value: string): string {
  return value.normalize('NFKC').toLocaleLowerCase('ru').replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}]+/gu, ' ').trim().replace(/\s+/g, ' ');
}

export function validateWakeName(value: string): string {
  const name = value.trim();
  if (name.length > 64 || normalizeSpeech(name).length < 2) {
    throw new Error('Имя должно содержать от 2 до 64 символов, включая буквы или цифры.');
  }
  return name;
}

export function isWakePhrase(text: string, name: string): boolean {
  const key = wakeSpelling(name);
  return key.length >= 2 && wakeSpelling(text) === key;
}

// Whisper can spell a Russian name in Latin letters. Accept equivalent
// spelling without broadening the match to similar-sounding Russian words.
const LATIN: Record<string, string> = {
  а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ж: 'zh', з: 'z', и: 'i', й: 'y',
  к: 'k', л: 'l', м: 'm', н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u',
  ф: 'f', х: 'kh', ц: 'ts', ч: 'ch', ш: 'sh', щ: 'shch', ъ: '', ы: 'y', ь: '', э: 'e', ю: 'yu', я: 'ya',
};
function wakeSpelling(value: string): string {
  return normalizeSpeech(value).replace(/[а-я]/g, (letter) => LATIN[letter]!);
}

export function wakeDistance(text: string, name: string): number {
  const actual = wakeSpelling(text).slice(0, 128), expected = wakeSpelling(name);
  let previous = Array.from({ length: expected.length + 1 }, (_, i) => i);
  for (let i = 0; i < actual.length; i++) {
    const next = [i + 1];
    for (let j = 0; j < expected.length; j++) next.push(Math.min(next[j]! + 1, previous[j + 1]! + 1,
      previous[j]! + (actual[i] === expected[j] ? 0 : 1)));
    previous = next;
  }
  return previous[expected.length]!;
}

export function commandLeadIn(text: string): string {
  return text.trim().replace(/^(?:(?:бот|bot|вот|ну|пожалуйста|слушай|давай)[\s,.:;!—-]+){1,3}/iu, '');
}

// Modes have explicit on/off semantics. Parse complete phrases before music
// extraction; a model trained only on player controls cannot supply these labels.
export function modeCommand(text: string): VoiceIntent | null {
  const words = normalizeSpeech(commandLeadIn(text)).replace(/ (?:пожалуйста|please)$/u, '');
  // Small Whisper sometimes spells the quiet initial consonant as "хлючи".
  // Accept it only with a complete, known mode target below.
  const on = '(?:включи|включить|включай|ключи|ключить|хлючи|запусти|запустить|поставь|поставить|активируй|enable|turn on)';
  const off = '(?:выключи|выключить|отключи|отключить|отключай|деактивируй|убери|останови|disable|turn off)';
  const auto = '(?:бесконечный режим(?: воспроизведения)?|режим (?:бесконечного воспроизведения|автоподбора|автоплея|рекомендаций)|бесконечное воспроизведение|бесконечную музыку|музыку бесконечно|'
    + 'авто ?плей|автопли|автоплэй|автоподбор(?: песен| музыки)?|автовоспроизведение|автоматический подбор(?: песен| музыки)?|'
    + 'автоматическое воспроизведение|автоматический режим|рекомендации(?: песен| музыки)?|autoplay|infinite mode)'
    + '(?: (?:от|с|на) (?:youtube|ютуба|ютубе))?';
  const loop = '(?:(?:повтор|повторение)(?: (?:трека|песни|этого трека|этой песни))?|режим (?:повтора|повторения)(?: трека| песни)?|'
    + 'зацикливание(?: трека| песни)?|repeat|loop)';
  const voice = '(?:голосовое управление|голосовой режим|режим голосового управления|прослушивание)';
  const match = (verb: string, target: string): boolean => new RegExp(`^${verb} ${target}$`, 'u').test(words);
  const unsafe = text.includes('?') || /(?:^| )(?:не|нет|если|и|или|потом|затем|not|never|and|or|then)(?: |$)/u.test(words);
  if (!unsafe) {
    const extra = extraVoiceCommand(words);
    if (extra) {
      const direct = modeCommand(extra);
      if (direct) return direct;
    }
    if (new RegExp(`^${on} громкость(?: |$)`, 'u').test(words)) {
      const level = parseVolume(text);
      return level === null ? { action: 'unknown' } : { action: 'volume_set', level };
    }
    if (match(on, auto)) return { action: 'autoplay_on' };
    if (match(off, auto)) return { action: 'autoplay_off' };
    if (match(on, loop) || /^(?:повторяй|зацикли) (?:этот трек|эту песню|трек|песню)$/u.test(words)) return { action: 'loop_on' };
    if (match(off, loop) || /^перестань повторять (?:этот трек|эту песню|трек|песню)$/u.test(words)) return { action: 'loop_off' };
    if (match(on, voice)) return { action: 'voice_on' };
    if (match(off, voice)) return { action: 'voice_off' };
    if (match(on, '(?:паузу|режим паузы)')) return { action: 'pause' };
    if (match(off, '(?:паузу|режим паузы)')) return { action: 'resume' };
    if (/^включи (?:следующий трек|следующую песню)$/u.test(words)) return { action: 'skip' };
    if (/^(?:очисти|очистить|сбрось|сбросить) очередь(?: песен| треков)?$|^удали (?:все )?(?:песни|треки) из очереди$/u.test(words)) return { action: 'queue_clear' };
  }
  // Reserve mode requests even when negated, compound, incomplete or unknown.
  // Explicit titles ("включи песню Бесконечный режим") stay music queries.
  const target = words.replace(new RegExp(`^(?:(?:не|нет) )?(?:${on}|${off}) `, 'u'), '');
  if (new RegExp(`^(?:(?:${auto}|${loop}|${voice})(?: |$)|режим(?: |$)|[\\p{L}]+ режим(?: |$)|паузу(?: |$))`, 'u').test(target)
    || (target !== words && /^громкость(?: |$)/u.test(target))
    || /^(?:очисти|очистить|сбрось|сбросить|удали|удалить|повторяй|зацикли|перестань повторять)(?: |$)/u.test(words)) {
    return { action: 'unknown' };
  }
  return null;
}

// Match whole requests so that titles, negations and multiple commands stay intact.
// Both voice profiles receive the same familiar command after canonicalization.
const CONTROL_ALIASES: readonly (readonly [RegExp, string])[] = [
  [/^приостанови(?: (?:музыку|воспроизведение|песню|трек))?$/u, 'Поставь на паузу'],
  [/^(?:продолжи|продолжай|возобнови|продолжить|возобновить) (?:песню|трек)$/u, 'Продолжи музыку'],
  [/^(?:следующ(?:ий|ая|ее|ую)|следущий|следующе|далее|дальше|переключи|переключись|скип|скипни|скипнуть|некст|next|skip)(?: (?:трек|песню|песня|композицию|track|song))?$/u, 'Следующий трек'],
  [/^(?:включи|включить|включай|ключи|поставь|поставить|запусти|сыграй) (?:следующ(?:ий|ая|ее|ую)(?: (?:трек|песню|песня|композицию))?|другой трек|другую песню)$/u, 'Следующий трек'],
  [/^(?:пропусти|пропустить|пропускай|скипни)(?: (?:этот трек|эту песню|эту композицию|этот|эту|это|текущий трек|текущую песню|трек|песню|музыку))?$/u, 'Следующий трек'],
  [/^(?:переключи|переключить|переключай|переключись|поменяй|смени)(?: (?:трек|песню|музыку|композицию))?(?: на следующ(?:ий|ую)| на следующий трек| на следующую песню)?$/u, 'Следующий трек'],
  [/^(?:перейди|переходи) к следующ(?:ему треку|ей песне)$|^дальше по очереди$/u, 'Следующий трек'],
  [/^(?:pause|пауза|паузу|пауза на музыку|поставь паузу|сделай паузу|нажми паузу|поставь (?:музыку|трек|песню|воспроизведение) на паузу|приостановить(?: (?:музыку|воспроизведение))?|притормози(?: (?:музыку|трек|песню))?)$/u, 'Поставь на паузу'],
  [/^(?:resume|continue|сними паузу|убери паузу|убери с паузы|сними (?:музыку|трек|песню) с паузы|снять с паузы|снять паузу|продолжить(?: (?:музыку|воспроизведение|играть))?|возобновить(?: (?:музыку|воспроизведение))?|возобнови проигрывание|продолжи воспроизведение трека)$/u, 'Продолжи музыку'],
  [/^(?:stop|стоп музыка|остановить(?: (?:музыку|воспроизведение|бота|плеер))?|останови (?:плеер|проигрывание|бота|музыку)|выключить (?:музыку|бота|плеер)|выключай музыку|выключи (?:музыку|бота|плеер)|отключи (?:музыку|бота|плеер)|хватит музыки|хватит играть|перестань играть|прекрати (?:музыку|воспроизведение)|выйти|выйди|выходи|выйти из (?:канала|войса|голосового(?: канала)?)|выйди из (?:канала|войса|голосового(?: канала)?)|выходи из (?:канала|войса)|выйди отсюда|выйти отсюда|покинь (?:канал|войс|голосовой канал)|покинуть (?:канал|войс)|уйди(?: из канала)?|уходи(?: из канала)?|отключись(?: от (?:канала|войса|голосового канала))?|отсоединись)$/u, 'Останови музыку'],
  [/^(?:louder|прибавь (?:громкость|звук|громкости|звука)|добавь (?:громкость|звук|громкости|звука)|погромче сделай|сделай (?:музыку|трек|песню|звук) (?:громче|погромче)|увеличить громкость)$/u, 'Сделай громче'],
  [/^(?:quieter|убавь (?:громкость|звук|громкости|звука)|потише сделай|сделай (?:музыку|трек|песню|звук) (?:тише|потише)|уменьшить громкость)$/u, 'Сделай тише'],
];

function controlAlias(text: string): string | null {
  if (text.includes('?')) return null;
  const words = normalizeSpeech(commandLeadIn(text)).replace(/ (?:пожалуйста|please)$/u, '');
  const volume = /^(?:(?:выставь|задай|сделай|поставь|установи) (?:громкость|звук)|звук)(?: (?:на|до))? (.+?)(?: (?:процентов|процента|процент))?$/u.exec(words);
  if (volume && volume[1]!.split(' ').every(token => /^\d+$/u.test(token) || ONES[token] !== undefined)) {
    const level = parseVolume(text);
    if (level !== null) return `Громкость ${level}`;
  }
  return CONTROL_ALIASES.find(([pattern]) => pattern.test(words))?.[1] ?? extraVoiceCommand(words);
}

// Closed control prefixes take priority over a free-form YouTube query.
export function playerControlRequest(text: string): boolean {
  if (modeCommand(text)) return true;
  const words = normalizeSpeech(controlAlias(text) ?? commandLeadIn(text));
  return /^(?:следующ\p{L}*|пропуст\p{L}*|скип\p{L}*|пауз\p{L}*|продолж\p{L}*|возобнов\p{L}*|приостанов\p{L}*|останов\p{L}*|стоп|выключ\p{L}*|отключ\p{L}*|отсоедин\p{L}*|выйди|выйти|выход\p{L}*|покин\p{L}*|уйд\p{L}*|уход\p{L}*|громк\p{L}*|громче|тише|погромче|потише|установ\p{L}*|выстав\p{L}*|задай|звук|увелич\p{L}*|уменьш\p{L}*|сдела\p{L}*)(?: |$)/u.test(words)
    || /^(?:поставь|поставить) (?:(?:на )?паузу|громкость|звук)(?: |$)|^сними с паузы(?: |$)|^хватит играть(?: |$)|^хватит музыки(?: |$)|^на паузу(?: |$)|^музыка на паузе(?: |$)/u.test(words);
}

// Generic "turn it on" means resume only when an existing track is paused.
// Keep song titles and any extra words outside this closed set.
export function contextualCommand(text: string, paused: boolean, radio = false): string {
  const cleaned = commandLeadIn(text);
  // Explicit mode commands already have a direct route without classification.
  if (modeCommand(cleaned)) return cleaned;
  const alias = controlAlias(cleaned);
  if (alias) return alias;
  const words = normalizeSpeech(cleaned);
  if (radio && !text.includes('?') && /^(?:останови|остановить|выключи|выключить) радио(?: пожалуйста)?$/u.test(words)) return 'Останови музыку';
  if (paused && !text.includes('?') && (
    /^(включи|включить|включай)( музыку| воспроизведение)?( снова| обратно| дальше)?$/.test(words)
    || /^(продолжай|продолжи|продолжить|возобнови|возобновить)( музыку| воспроизведение| играть)?$/.test(words)
  )) {
    return 'Продолжи музыку';
  }
  return cleaned;
}

// A closed decision model cannot supply arbitrary arguments. Parse numbers in code.
export const ONES: Record<string, number> = {
  ноль: 0, один: 1, одна: 1, два: 2, две: 2, три: 3, четыре: 4, пять: 5,
  шесть: 6, семь: 7, восемь: 8, девять: 9, десять: 10, одиннадцать: 11,
  двенадцать: 12, тринадцать: 13, четырнадцать: 14, пятнадцать: 15,
  шестнадцать: 16, семнадцать: 17, восемнадцать: 18, девятнадцать: 19,
  двадцать: 20, тридцать: 30, сорок: 40, пятьдесят: 50, шестьдесят: 60,
  семьдесят: 70, восемьдесят: 80, девяносто: 90, сто: 100,
};

export function isNumberWord(word: string): boolean {
  return /^\d+$/u.test(word) || ONES[word] !== undefined
    || /^(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)$/u.test(word);
}

export function isMathExpression(text: string): boolean {
  if (/https?:\/\//iu.test(text)) return false;
  if (/[\+^÷×]/u.test(text)) return true;
  if (/=\s*\d|\d\s*=/u.test(text)) return true;
  if (/\d\s*[\/\-−]\s*\d/u.test(text)) return true;
  if (/\b\d+\s*[xх*]\s*\d+\b/iu.test(text)) return true;

  const normalized = normalizeSpeech(text);
  const words = normalized.split(' ').filter(Boolean);
  if (!words.length) return false;

  if (words.some((w) => /^(?:умножить|умножь|умноженное|умноженная|поделить|разделить|раздели|деленное|квадратный|факториал|синус|косинус|тангенс|логарифм)$/u.test(w))) return true;
  if ((words.includes('плюс') || words.includes('минус') || words.includes('прибавить') || words.includes('прибавь') || words.includes('отнять') || words.includes('вычесть') || words.includes('равно'))
    && words.some(isNumberWord)) return true;

  if (/^(?:сколько будет|посчитай|вычисли|реши|решите|сосчитай|пересчитай|калькулятор)(?: |$)/u.test(normalized)) return true;
  if (/^(?:раз|раз два|раз два три|раз раз)(?: |$)/u.test(normalized)) return true;

  if (words.every(isNumberWord)) return true;

  return false;
}

export function parseVolume(text: string): number | null {
  // Reject decimals, signs and multiple separate values instead of clamping a guess.
  if (/\d\s*[.,]\s*\d|[-−+]\s*\d/u.test(text)) return null;
  const tokens = normalizeSpeech(text).split(' ');
  const groups: number[][] = [];
  let group: number[] = [];
  for (const token of tokens) {
    const value = /^\d+$/.test(token) ? Number(token) : ONES[token];
    if (value !== undefined) group.push(value);
    else if (group.length) { groups.push(group); group = []; }
  }
  if (group.length) groups.push(group);
  if (groups.length !== 1) return null;
  const parts = groups[0]!;
  const numberTokens = tokens.filter((token) => /^\d+$/.test(token) || ONES[token] !== undefined);
  if (numberTokens.some((token) => /^\d+$/.test(token)) && numberTokens.length !== 1) return null;
  if (parts.length > 3 || parts.some((n, i) => i > 0 && (n <= 0 || n >= parts[i - 1]!
    || parts[i - 1]! < 20 || (parts[i - 1]! < 100 && (n >= 10 || parts[i - 1]! % 10 !== 0))))) return null;
  const level = parts.reduce((sum, n) => sum + n, 0);
  return Number.isInteger(level) && level >= 1 && level <= 150 ? level : null;
}

export function unsupportedSpeech(text: string): boolean {
  const mode = modeCommand(text);
  if (mode) return mode.action === 'unknown';
  if (isMathExpression(text)) return true;
  const words = normalizeSpeech(text).split(' ');
  if (!words[0] || words.length > 35
    || words.some((word) => /^(не|нет|если|потом|затем|сначала|и|или|но|not|no|never|dont|and|or|then|but)$/.test(word))
    || /^(?:don t|do not|please don t)(?: |$)/u.test(words.join(' '))) return true;
  if (/\?/u.test(text) || /^(как|кто|когда|почему|зачем|расскажи|покажи|привет|спасибо|мне нравится|музыка на паузе|что|че|чо|где|куда|откуда|сколько|какой|какая|какое|какие|чей|посчитай|вычисли|реши|скажи|объясни|переведи|помоги|погода|новости|анекдот|шутка|здравствуй|здравствуйте|пока|до свидания|ладно|хорошо|ясно|понятно|ок|окей|проверка|тест)(?: |$)/u.test(words.join(' '))) return true;
  if (words.some((word) => /^(добавь|найди|поиск|сыграй|воспроизведи|очисти|плейлист|песн\p{L}*)$/u.test(word))) return true;
  if (words.includes('включи') || words.includes('включить')) return true;
  if (words.includes('поставь') || words.includes('поставить')) {
    if (!/^(поставь|поставить) (на )?паузу$/.test(words.join(' ')) && !words.includes('громкость')) return true;
  }
  if (words.some((word) => /^трек\p{L}*$/u.test(word)) && !words.some((word) => /^(следующий|пропусти|пропустить|скипни)$/.test(word))) return true;
  return false;
}

export function validateIntent(text: string, decision: VoiceDecision, threshold = 0.6): VoiceIntent {
  const direct = modeCommand(text);
  if (direct) return direct;
  if (unsupportedSpeech(text) || !Number.isFinite(decision.confidence)
    || decision.confidence < threshold || decision.confidence > 1) return { action: 'unknown' };
  const allowed: VoiceAction[] = ['skip', 'pause', 'resume', 'stop', 'volume_set', 'volume_up', 'volume_down', 'unknown'];
  if (!allowed.includes(decision.action)) return { action: 'unknown' };
  // Probabilities are not calibrated: never execute an action contradicting an explicit verb.
  const words = normalizeSpeech(text).split(' ');
  const incompatible: Partial<Record<VoiceAction, RegExp>> = {
    skip: /^(пауз\p{L}*|продолж\p{L}*|возобнов\p{L}*|останов\p{L}*|отключ\p{L}*|громк\p{L}*|громче|тише)$/u,
    pause: /^(следующ\p{L}*|пропуст\p{L}*|пропусти|продолж\p{L}*|возобнов\p{L}*|сними|останов\p{L}*|отключ\p{L}*|громче|тише)$/u,
    resume: /^(следующ\p{L}*|пропуст\p{L}*|пропусти|приостанов\p{L}*|останов\p{L}*|отключ\p{L}*|громче|тише)$/u,
    stop: /^(следующ\p{L}*|пропуст\p{L}*|пропусти|приостанов\p{L}*|пауз\p{L}*|продолж\p{L}*|возобнов\p{L}*|сними|громче|тише)$/u,
  };
  const contradiction = incompatible[decision.action];
  if (contradiction && words.some((word) => contradiction.test(word))) return { action: 'unknown' };
  if (decision.action === 'resume' && words.some((word) => /^постав\p{L}*$/u.test(word))) return { action: 'unknown' };
  if (decision.action === 'stop' && !words.some((word) => /^(музык\p{L}*|воспроизвед\p{L}*|бот\p{L}*|канал\p{L}*|войс\p{L}*|плеер\p{L}*|звук)$/u.test(word))
    && !/^(стоп|останови|остановить|отключись|отсоединись|выключись|выйди|выйти|выходи|покинь|уйди|уходи|хватит играть|хватит музыки)$/.test(words.join(' '))
    && !/^(?:выйди|выйти|выходи|покинь|уйди|уходи) (?:из (?:канала|войса|голосового(?: канала)?)|отсюда)$/.test(words.join(' '))) return { action: 'unknown' };
  const hasNumber = words.some((word) => /^\d+$/.test(word) || ONES[word] !== undefined);
  if (decision.action.startsWith('volume_')) {
    if (words.some((word) => /^(следующ\p{L}*|пропуст\p{L}*|пропусти|приостанов\p{L}*|пауз\p{L}*|продолж\p{L}*|возобнов\p{L}*|останов\p{L}*|отключ\p{L}*)$/u.test(word))) return { action: 'unknown' };
    if (hasNumber && !words.includes('до') && words.some((word) => /^(громче|тише|увелич\p{L}*|уменьш\p{L}*)$/u.test(word))) return { action: 'unknown' };
  }
  if (decision.action === 'volume_set') {
    const level = parseVolume(text);
    return level === null ? { action: 'unknown' } : { action: 'volume_set', level };
  }
  if ((decision.action === 'volume_up' || decision.action === 'volume_down') && hasNumber) return { action: 'unknown' };
  return { action: decision.action };
}

// Complete commands only: a title containing "stop" or an arbitrary extra
// word must never become a player action. Light mode needs no NLI model.
export function ruleIntent(text: string): VoiceIntent {
  const canonical = contextualCommand(text, false);
  const direct = modeCommand(canonical);
  if (direct) return direct;
  const words = normalizeSpeech(canonical).replace(/ (?:пожалуйста|please)$/u, '');
  if (canonical.includes('?') || /(?:^| )(?:не|нет|если|и|или|потом|затем|но|not|no|never|dont|and|or|then|but)(?: |$)/u.test(words)) return { action: 'unknown' };
  const target = '(?:музыку|воспроизведение|песню|трек)';
  if (/^(?:следующий трек|следующая песня|пропусти(?: (?:текущий |текущую )?(?:трек|песню|музыку))?|переключи(?: (?:трек|песню))?)$/u.test(words)) return { action: 'skip' };
  if (new RegExp(`^(?:поставь(?: на)? паузу|на паузу|пауза|приостанови(?: ${target})?)$`, 'u').test(words)) return { action: 'pause' };
  if (new RegExp(`^(?:сними с паузы|(?:продолжи|продолжай|возобнови)(?: ${target}| играть)?)$`, 'u').test(words)) return { action: 'resume' };
  if (/^(?:стоп|стоп музыка|останови(?: (?:музыку|воспроизведение|бота|плеер))?|остановить(?: (?:музыку|воспроизведение|бота|плеер))?|выключи (?:музыку|бота|плеер)|выключить (?:музыку|бота|плеер)|отключись(?: от (?:канала|войса|голосового канала))?|отсоединись|выйди(?: (?:из (?:канала|войса|голосового(?: канала)?)|отсюда))?|выйти(?: (?:из (?:канала|войса|голосового(?: канала)?)|отсюда))?|выходи(?: (?:из (?:канала|войса))?)?|покинь (?:канал|войс|голосовой канал)|уйди(?: из (?:канала|войса))?|уходи(?: из (?:канала|войса))?|хватит играть|хватит музыки)$/u.test(words)) return { action: 'stop' };
  if (/^(?:сделай (?:громче|погромче)|громче|погромче|(?:увеличь|повысь) (?:громкость|звук))$/u.test(words)) return { action: 'volume_up' };
  if (/^(?:сделай (?:тише|потише)|тише|потише|(?:уменьши|снизь) (?:громкость|звук))$/u.test(words)) return { action: 'volume_down' };
  const volume = /^(?:громкость|(?:установи|поставь|сделай) громкость|(?:увеличь|уменьши) громкость до) (?:на )?(.+?)(?: процентов| процента| процент)?$/u.exec(words);
  if (volume) {
    const tokens = volume[1]!.split(' ');
    if (tokens.every(token => /^\d+$/u.test(token) || ONES[token] !== undefined)) {
      const level = parseVolume(canonical);
      if (level !== null) return { action: 'volume_set', level };
    }
  }
  return { action: 'unknown' };
}
