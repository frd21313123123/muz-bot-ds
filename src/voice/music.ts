import type { Track } from '../types.js';
import { normalizeSpeech } from './intents.js';
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
  stage: 'routing' | 'search' | 'policy' | 'selection' | 'fallback';
  ms?: number;
  candidates?: number;
  bestTrack?: number;
  confidence?: number;
  nextTool?: MusicDecision['next_tool'];
  policy?: MusicResultPolicy['search_result_policy'];
  reason?: 'model' | 'selection' | 'empty' | 'metadata';
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
export function extractMusicRequest(message: string): MusicRequest | null {
  if (message.length > 1000) return null;
  const match = /^(?:включи|поставь|сыграй|воспроизведи)\s+(.+)$/iu.exec(message.trim());
  if (!match) return null;
  const query = match[1]!.replace(/^(?:песню|трек)\s+/iu, '').trim();
  const words = normalizeSpeech(query);
  if (!words || /^(?:на паузу|паузу|музыку(?: снова| обратно| дальше)?|снова|обратно|дальше|песню|трек|эту(?: песню)?|этот(?: трек)?|ее|его|это|ту|тот|первую|первый|следующую(?: песню| музыку)?|следующий(?: трек)?)$/u.test(words)) return null;
  // Conjunctions in names (e.g. "Numb и Encore") are valid. A second command
  // verb after a command separator is not a song name.
  if (/(?:^|\s)(?:и|или|потом|затем)\s+(?:не\s+)?(?:включ\p{L}*|постав\p{L}*|сыгра\p{L}*|воспроизвед\p{L}*|останов\p{L}*|останавли\p{L}*|выключ\p{L}*|пропуст\p{L}*|продолж\p{L}*|возобнов\p{L}*|сдела\p{L}*|пауза|следующий|громче|тише|стоп)(?:\s|$)/iu.test(words)) return null;
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

export async function resolveMusicRequest(message: string, requestedBy: string, backend: MusicBackend,
  metadata: MusicMetadata, signal: AbortSignal, diagnostic?: (event: MusicDiagnostic) => void,
  player?: MusicPlayerState): Promise<Track | null> {
  const request = extractMusicRequest(message);
  if (!request) return null;
  signal.throwIfAborted();
  let policy: MusicDecision['search_result_policy'] = 'play_first_result';
  let deferredPolicy = false;
  const started = performance.now();
  try {
    const decision = await backend.decideMusic(musicState(message, request, player), signal);
    signal.throwIfAborted();
    diagnostic?.({ stage: 'routing', ms: Math.round(performance.now() - started),
      nextTool: decision.next_tool, policy: decision.search_result_policy, confidence: decision.confidence });
    if (!validConfidence(decision.confidence) || decision.next_tool === 'unknown' || decision.next_tool === 'player_control') return null;
    // The model selects a route; code verifies that it matches the actual input.
    if (decision.next_tool !== (request.kind === 'video' ? 'direct_youtube_video' : 'youtube_music_search')) return null;
    policy = decision.search_result_policy;
    deferredPolicy = decision.defer_result_policy === true;
  } catch {
    signal.throwIfAborted();
    diagnostic?.({ stage: 'fallback', reason: 'model' });
  }
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
  const selectionCandidates = usable.map(({ info, track }, i) => ({
    index: i, title: track.title.slice(0, 240), artist: (info.artist || info.channel || info.uploader || '').slice(0, 120),
    duration: track.duration.slice(0, 32),
  }));
  if (request.kind === 'search' && deferredPolicy) {
    const policyStart = performance.now();
    try {
      if (!backend.decideMusicResults) throw new Error('Result policy unavailable');
      const decision = await backend.decideMusicResults(request.query, selectionCandidates, signal);
      signal.throwIfAborted();
      diagnostic?.({ stage: 'policy', policy: decision.search_result_policy, confidence: decision.confidence,
        ms: Math.round(performance.now() - policyStart) });
      if (!validConfidence(decision.confidence)) throw new Error('Weak result policy');
      if (decision.search_result_policy === 'no_result') return null;
      // The voice product takes the first candidate for a plain title. Model
      // reranking is available when the request explicitly qualifies a version.
      policy = musicVariant(request.query) ? decision.search_result_policy : 'play_first_result';
    } catch {
      signal.throwIfAborted();
      diagnostic?.({ stage: 'fallback', reason: 'model' });
      policy = 'play_first_result';
    }
  }
  let index = 0;
  if (request.kind === 'search' && policy === 'rerank_results' && usable.length > 1) {
    const selectionStart = performance.now();
    try {
      const selection = await backend.rerankMusic(request.query, selectionCandidates, signal);
      signal.throwIfAborted();
      if (selection.no_match === true && selection.best_track === -1 && validConfidence(selection.confidence)) return null;
      if (!Number.isInteger(selection.best_track) || !usable[selection.best_track] || !validConfidence(selection.confidence)) {
        throw new Error('Invalid selection');
      }
      index = selection.best_track;
    } catch {
      signal.throwIfAborted();
      diagnostic?.({ stage: 'fallback', reason: 'selection' });
    }
    diagnostic?.({ stage: 'selection', bestTrack: index, ms: Math.round(performance.now() - selectionStart) });
  }
  signal.throwIfAborted();
  return usable[index]!.track;
}
