import type { RadioTrack } from '../types.js';
import { commandLeadIn, normalizeSpeech } from '../voice/intents.js';

export interface RadioStation {
  id: string;
  name: string;
  aliases: readonly string[];
  website: string;
  streams: readonly string[];
}

// Main Russian broadcasts, taken from the broadcasters' web players.
// Keep stream addresses separate from the public website used in embeds.
export const radioStations: readonly RadioStation[] = [
  { id: 'europa-plus', name: 'Европа Плюс', aliases: ['europa plus', 'europe plus', 'европу плюс', 'европаплюс'],
    website: 'https://europaplus.ru/', streams: ['https://hls-02-europaplus.emgsound.ru/11/playlist.m3u8'] },
  { id: 'retro-fm', name: 'Ретро FM', aliases: ['retro fm', 'ретро фм', 'ретрофм', 'retrofm'],
    website: 'https://retrofm.ru/', streams: ['https://hls-01-retro.emgsound.ru/12/playlist.m3u8'] },
  { id: 'dorognoe', name: 'Дорожное радио', aliases: ['дорожное', 'dorognoe radio', 'dorozhnoe radio'],
    website: 'https://dorognoe.ru/', streams: ['https://hls-01-dorognoe.emgsound.ru/15/playlist.m3u8'] },
  { id: 'russkoe', name: 'Русское радио', aliases: ['русское', 'russkoe radio', 'russian radio'],
    website: 'https://rusradio.ru/', streams: ['https://hls-01-rusradio.hostingradio.ru/rusradio/playlist.m3u8'] },
  { id: 'avtoradio', name: 'Авторадио', aliases: ['авто радио', 'avtoradio', 'auto radio'],
    website: 'https://www.avtoradio.ru/', streams: ['https://pub0201.101.ru:8443/stream/air/aac/64/100',
      'https://pub0102.101.ru:8443/stream/air/aac/64/100'] },
  { id: 'dfm', name: 'DFM', aliases: ['дфм', 'dfм', 'ди фм', 'ди эф эм', 'диэфэм', 'd fm'],
    website: 'https://dfm.ru/', streams: ['https://hls-01-dfm.hostingradio.ru/dfm/playlist.m3u8'] },
  { id: 'energy', name: 'Радио ENERGY', aliases: ['energy', 'radio energy', 'энерджи', 'энергия', 'энергии', 'энергию', 'nrj', 'нрж', 'энэрджи'],
    website: 'https://www.energyfm.ru/', streams: ['https://pub0201.101.ru:8443/stream/air/aac/64/99',
      'https://pub0102.101.ru:8443/stream/air/aac/64/99'] },
  { id: 'love-radio', name: 'Love Radio', aliases: ['лав радио', 'лов радио', 'лаврадио', 'love', 'лав'],
    website: 'https://www.loveradio.ru/', streams: ['https://stream2.n340.com/12_love_64_reg_44?type=aac'] },
  { id: 'radio-dacha', name: 'Радио Дача', aliases: ['дача', 'дачу', 'дачи', 'radio dacha'],
    website: 'https://www.radiodacha.ru/', streams: ['https://stream2.n340.com/12_dacha_64_reg_1093?type=aac'] },
  { id: 'nashe', name: 'Наше радио', aliases: ['наше', 'nashe radio'],
    website: 'https://nashe.ru/', streams: ['https://nashe1.hostingradio.ru/nashe-128.mp3'] },
  { id: 'hit-fm', name: 'Хит FM', aliases: ['hit fm', 'хит фм', 'хитфм', 'hitfm'],
    website: 'https://hitfm.ru/', streams: ['https://hls-01-hitfm.hostingradio.ru/hitfm/playlist.m3u8'] },
];

const key = (text: string): string => normalizeSpeech(text)
  .replace(/^(?:радиостанци\p{L}*|прямой\s+эфир|эфир\s+радио|эфир|станци\p{L}*|радио)\s+/u, '')
  .replace(/^(?:радио\s+)+/u, '')
  .replace(/\s+/g, '');
export function findRadioStation(name: string): RadioStation | null {
  const normalized = key(name);
  return radioStations.find(station => [station.id, station.name, ...station.aliases]
    .some(alias => key(alias) === normalized)) ?? null;
}

export type RadioRequest = { kind: 'station'; station: RadioStation }
  | { kind: 'unsupported' | 'rejected' };

// null means an ordinary music request. Explicit radio station commands or test
// sentinels reject unsupported stations; ordinary play requests fall through to music.
export function extractRadioRequest(message: string, allowBare = false): RadioRequest | null {
  if (message.length > 1000) return { kind: 'rejected' };
  const cleaned = commandLeadIn(message);
  const words = normalizeSpeech(cleaned);
  const match = /^(?:(?:можешь|можете)\s+)?(?:включи|включить|включай|ключи|ключить|поставь|поставить|запусти|запустить|хочу послушать|послушаем)(?:\s*[,：:\u2014-]\s*|\s+)(.+)$/iu.exec(cleaned);
  const target = normalizeSpeech(match?.[1] ?? cleaned).replace(/ (?:пожалуйста|please)$/u, '');
  if (/^(?:песню|песня|трек)(?: |$)/u.test(target)) return null;
  const explicit = /^(?:радиостанци\p{L}*|прямой\s+эфир|эфир\s+радио|эфир|станци\p{L}*)(?: |$)/u.test(target)
    || /^радио\s+(?:неизвестн\p{L}*|росси\p{L}*)$/u.test(target);
  const unsafe = cleaned.includes('?') || /(?:^| )(?:не|нет|если|потом|затем|сначала|и|или|но|not|no|never|dont|and|or|then|but)(?: |$)/u.test(words)
    || /^(?:как|кто|когда|почему|зачем|расскажи|покажи|do not|don t)(?: |$)/u.test(words);
  if (unsafe) {
    const mentionsRadio = /(?:^| )радио(?: |$)/u.test(words) || radioStations.some(station =>
      [station.name, ...station.aliases].some(alias => ` ${words} `.includes(` ${normalizeSpeech(alias)} `)));
    return mentionsRadio ? { kind: 'rejected' } : null;
  }
  if (!match && !allowBare) return null;
  const station = findRadioStation(target);
  return station ? { kind: 'station', station } : explicit ? { kind: 'unsupported' } : null;
}

export function radioTrack(station: RadioStation, requestedBy: string, streamUrl = station.streams[0]!): RadioTrack {
  return { source: 'radio', stationId: station.id, streamUrl, url: station.website,
    title: station.name, duration: '', thumbnail: null, requestedBy, isLive: true };
}

export function radioSuggestions(query: string): { name: string; value: string }[] {
  const normalized = key(query);
  return radioStations.filter(station => [station.name, station.id, ...station.aliases]
    .some(alias => key(alias).includes(normalized))).slice(0, 25)
    .map(station => ({ name: station.name, value: station.id }));
}
