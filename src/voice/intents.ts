export type VoiceAction = 'skip' | 'pause' | 'resume' | 'stop' | 'volume_set' | 'volume_up' | 'volume_down' | 'unknown';
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
  const key = normalizeSpeech(name);
  return key.length >= 2 && normalizeSpeech(text) === key;
}

// A closed decision model cannot supply arbitrary arguments. Parse numbers in code.
const ONES: Record<string, number> = {
  ноль: 0, один: 1, одна: 1, два: 2, две: 2, три: 3, четыре: 4, пять: 5,
  шесть: 6, семь: 7, восемь: 8, девять: 9, десять: 10, одиннадцать: 11,
  двенадцать: 12, тринадцать: 13, четырнадцать: 14, пятнадцать: 15,
  шестнадцать: 16, семнадцать: 17, восемнадцать: 18, девятнадцать: 19,
  двадцать: 20, тридцать: 30, сорок: 40, пятьдесят: 50, шестьдесят: 60,
  семьдесят: 70, восемьдесят: 80, девяносто: 90, сто: 100,
};

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
  const words = normalizeSpeech(text).split(' ');
  if (!words[0] || words.length > 35
    || words.some((word) => /^(не|нет|если|потом|затем|сначала|и|или|но)$/.test(word))) return true;
  if (/\?/u.test(text) || /^(как|кто|когда|почему|зачем|расскажи|покажи|привет|спасибо|мне нравится|музыка на паузе)(?: |$)/u.test(words.join(' '))) return true;
  if (words.some((word) => /^(добавь|найди|поиск|сыграй|воспроизведи|очисти|плейлист|песн\p{L}*)$/u.test(word))) return true;
  if (words.includes('включи') || words.includes('включить')) return true;
  if (words.includes('поставь') || words.includes('поставить')) {
    if (!/^(поставь|поставить) (на )?паузу$/.test(words.join(' ')) && !words.includes('громкость')) return true;
  }
  if (words.some((word) => /^трек\p{L}*$/u.test(word)) && !words.some((word) => /^(следующий|пропусти|пропустить|скипни)$/.test(word))) return true;
  return false;
}

export function validateIntent(text: string, decision: VoiceDecision, threshold = 0.6): VoiceIntent {
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
  if (decision.action === 'stop' && !words.some((word) => /^(музык\p{L}*|воспроизвед\p{L}*|бот\p{L}*|канал\p{L}*|плеер\p{L}*|звук)$/u.test(word))
    && !/^(стоп|останови|остановить|отключись|выключись|выйди|хватит играть)$/.test(words.join(' '))) return { action: 'unknown' };
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
