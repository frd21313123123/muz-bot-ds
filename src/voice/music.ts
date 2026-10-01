import type { Track } from '../types.js';
import { commandLeadIn, normalizeSpeech, playerControlRequest } from './intents.js';
import { toTrack, type VideoInfo } from '../utils/ytdlp.js';

export type MusicVariant = 'live' | 'remix' | 'cover' | 'acoustic' | 'instrumental';
export interface MusicPlayerState { connected: boolean; playing: boolean; paused: boolean; autoplay: boolean; queue_length: number }
export interface MusicState {
  message: string; selected_track: null;
  parsed_request?: { kind: 'play'; search_query: string | null; url?: string; request_variant: MusicVariant | null };
  player?: MusicPlayerState;
}
export interface MusicDecision {
  next_tool: 'youtube_music_search' | 'direct_youtube_video' | 'player_control' | 'unknown';
  query_source: 'message';
  search_result_policy: 'play_first_result' | 'rerank_results';
  confidence: number;
  defer_result_policy?: boolean;
}
export interface MusicCandidate { index: number; title: string; artist: string; duration: string }
export interface MusicSelection { best_track: number; confidence: number; no_match?: boolean }
export interface MusicResultPolicy { search_result_policy: 'play_first_result' | 'rerank_results' | 'no_result'; confidence: number }
export interface MusicBackend {
  decideMusic(state: MusicState, signal: AbortSignal): Promise<MusicDecision>;
  rerankMusic(query: string, candidates: MusicCandidate[], signal: AbortSignal): Promise<MusicSelection>;
  decideMusicResults?(query: string, candidates: MusicCandidate[], signal: AbortSignal): Promise<MusicResultPolicy>;
}
export interface MusicMetadata {
  searchCandidates(query: string, limit: number, signal: AbortSignal): Promise<VideoInfo[]>;
  video(url: string, signal: AbortSignal): Promise<VideoInfo>;
}
export interface MusicDiagnostic {
  stage: 'routing' | 'search' | 'policy' | 'selection' | 'fallback' | 'rejected';
  ms?: number;
  candidates?: number;
  bestTrack?: number;
  confidence?: number;
  nextTool?: MusicDecision['next_tool'];
  policy?: MusicResultPolicy['search_result_policy'];
  reason?: 'model' | 'selection' | 'empty' | 'metadata' | 'no_result' | 'no_match' | 'routing' | 'confidence';
}
export type MusicRequest = { kind: 'search' | 'video'; query: string };

export function musicVariant(query: string): MusicVariant | null {
  const words = normalizeSpeech(query);
  for (const [variant, pattern] of [
    ['live', /(?:^| )(?:live|концертн\p{L}*)(?: |$)/u], ['remix', /(?:^| )(?:remix|ремикс\p{L}*)(?: |$)/u],
    ['cover', /(?:^| )(?:cover|кавер\p{L}*)(?: |$)/u], ['acoustic', /(?:^| )(?:acoustic|акустическ\p{L}*)(?: |$)/u],
    ['instrumental', /(?:^| )(?:instrumental|инструментальн\p{L}*)(?: |$)/u],
  ] as const) if (pattern.test(words)) return variant;
  return null;
}

export function musicState(message: string, request: MusicRequest, player?: MusicPlayerState): MusicState {
  return { message, selected_track: null, parsed_request: { kind: 'play',
    search_query: request.kind === 'search' ? request.query : null,
    ...(request.kind === 'video' ? { url: request.query } : {}), request_variant: musicVariant(request.query) },
    ...(player ? { player } : {}) };
}

// Extract from the original transcript: speech normalization would destroy URLs
// and punctuation in song titles. Only command boundaries are interpreted here.
export function extractMusicRequest(message: string, allowBare = false): MusicRequest | null {
  if (message.length > 1000) return null;
  // Conversational lead-ins and punctuation inserted by ASR are not part of
  // the title. Stay anchored: never fish a command out of a longer sentence.
  const cleaned = commandLeadIn(message);
  if (playerControlRequest(cleaned)) return null;
  // ASR can lose the quiet initial consonant in "включи". Interpret this
  // leading command form only; never rewrite words inside the query.
  const match = /^(?:(?:можешь|можете)\s+)?(?:включи|включить|включай|ключи|ключить|поставь|поставить|сыграй|сыграть|воспроизведи|воспроизвести|запусти|запустить|проиграй|проиграть|хочу послушать|послушаем)(?:\s*[,:\u2014-]\s*|\s+)(.+)$/iu.exec(cleaned);
  if (!match && !allowBare) return null;
  if (!match) {
    const bare = normalizeSpeech(cleaned);
    // Only the speaker's command window permits a bare title/artist. Do not
    // turn conversations, negations, questions or incomplete commands into music.
    if (!bare || bare.split(' ').length > 35 || cleaned.includes('?')
      || /^(?:не|нет|если|сначала|потом|затем|как|кто|когда|почему|зачем|расскажи|покажи|привет|спасибо|я|мы|ты|вы|он|она|мне нравится|добавь|найди|очисти)(?: |$)/u.test(bare)
      || /(?:^| )(?:включи|включить|включай|ключи|ключить|поставь|поставить|сыграй|сыграть|воспроизведи|воспроизвести|запусти|запустить|проиграй|проиграть)(?: |$)/u.test(bare)) return null;
  }
  const target = (match?.[1] ?? cleaned).trim();
  if (/^(?:песню|трек)\s+из[.!…]*$/iu.test(target)) return null;
  // "Song from Luntik" needs its music noun: searching only "from Luntik"
  // predominantly returns cartoon episodes, unlike an explicit song title.
  const query = /^(?:песню|трек)\s+из\s+/iu.test(target)
    ? target.replace(/^песню\s+/iu, 'песня ')
    : target.replace(/^(?:песню|трек)\s+/iu, '').trim();
  const words = normalizeSpeech(query);
  if (!words || /^(?:на паузу|паузу|музыку(?: снова| обратно| дальше)?|музыка|воспроизведение|снова|обратно|дальше|песню|песни|песня|трек|эту(?: песню)?|этот(?: трек)?|ее|его|это|то|ту|тот|первую|первый|следующую(?: песню| музыку)?|следующий(?: трек)?)$/u.test(words)) return null;
  // Conjunctions in names (e.g. "Numb и Encore") are valid. A second command
  // verb after a command separator is not a song name.
  if (/(?:^|\s)(?:и|или|потом|затем)\s+(?:не\s+)?(?:включ\p{L}*|ключи\p{L}*|постав\p{L}*|сыгра\p{L}*|воспроизвед\p{L}*|запуст\p{L}*|проигра\p{L}*|останов\p{L}*|останавли\p{L}*|выключ\p{L}*|пропуст\p{L}*|продолж\p{L}*|возобнов\p{L}*|сдела\p{L}*|пауза|следующий|громче|тише|стоп)(?:\s|$)/iu.test(words)) return null;
  if (/[,;.!]\s*(?:не\s+)?(?:включ\p{L}*|ключи\p{L}*|постав\p{L}*|сыгра\p{L}*|воспроизвед\p{L}*|запуст\p{L}*|проигра\p{L}*|останов\p{L}*|выключ\p{L}*|пропуст\p{L}*|продолж\p{L}*|возобнов\p{L}*|сдела\p{L}*|пауза|следующий|громче|тише|стоп)(?:\s|$)/iu.test(query)) return null;
  if (/https?:\/\//iu.test(query)) {
    let url: URL;
    try { url = new URL(query.replace(/[.!…]+$/u, '')); } catch { return null; }
    if (!['http:', 'https:'].includes(url.protocol)
      || !['youtube.com', 'www.youtube.com', 'music.youtube.com', 'm.youtube.com', 'youtu.be'].includes(url.hostname.toLowerCase())) return null;
    const id = url.hostname.toLowerCase() === 'youtu.be' ? url.pathname.slice(1).split('/')[0] : url.searchParams.get('v');
    if (!id || !/^[\w-]{11}$/.test(id)) return null;
    return { kind: 'video', query: `https://www.youtube.com/watch?v=${id}` };
  }
  return { kind: 'search', query };
}

export function validConfidence(value: number): boolean {
  return Number.isFinite(value) && value >= 0.6 && value <= 1;
}

export async function resolveMusicRequest(message: string, requestedBy: string, _backend: MusicBackend,
  metadata: MusicMetadata, signal: AbortSignal, diagnostic?: (event: MusicDiagnostic) => void,
  _player?: MusicPlayerState): Promise<Track | null> {
  const request = extractMusicRequest(message, true);
  if (!request) return null;
  signal.throwIfAborted();
  // A parsed music request is a search query, not a label the model must know.
  // Laya is reserved for player controls; artist/title queries go straight to
  // YouTube with its own spelling correction and relevance order.
  diagnostic?.({ stage: 'routing', ms: 0, nextTool: request.kind === 'video' ? 'direct_youtube_video' : 'youtube_music_search',
    policy: 'play_first_result' });
  let candidates: VideoInfo[];
  const searchStart = performance.now();
  try {
    candidates = request.kind === 'video' ? [await metadata.video(request.query, signal)]
      : await metadata.searchCandidates(request.query, 5, signal);
    signal.throwIfAborted();
  } catch {
    signal.throwIfAborted();
    diagnostic?.({ stage: 'search', reason: 'metadata', ms: Math.round(performance.now() - searchStart) });
    return null;
  }
  const seen = new Set<string>();
  const usable = candidates.flatMap((info) => {
    try {
      const track = toTrack(info, requestedBy);
      if (seen.has(track.videoId)) return [];
      seen.add(track.videoId);
      return [{ info, track }];
    } catch { return []; }
  }).slice(0, 5);
  diagnostic?.({ stage: 'search', candidates: usable.length, ms: Math.round(performance.now() - searchStart),
    ...(usable.length ? {} : { reason: 'empty' as const }) });
  if (!usable.length) return null;
  // YouTube's relevance order is authoritative, including for live/remix/etc.
  // Neither routing nor candidates need a model decision for a parsed query.
  signal.throwIfAborted();
  return usable[0]!.track;
}
