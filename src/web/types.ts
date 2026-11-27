// Shared view models for the web control panel, the Alice webhook and the
// local speaker bridge. They intentionally contain no tokens or secrets.

export interface WebTarget {
  guildId: string;
  channelId: string;
}

export interface TrackView {
  title: string;
  url: string;
  thumbnail: string | null;
  duration: string;
  requestedBy: string;
  isAutoplay: boolean;
  isLive: boolean;
  source: 'youtube' | 'soundcloud' | 'radio';
}

export interface QueueView {
  guildId: string;
  channelId: string | null;
  channelName: string | null;
  playing: boolean;
  paused: boolean;
  radio: boolean;
  volume: number;
  speed: number;
  autoplay: boolean;
  loop: boolean;
  elapsed: number;
  current: TrackView | null;
  upcoming: TrackView[];
  upcomingCount: number;
  sampledAt: number;
}

export interface GuildView {
  id: string;
  name: string;
  icon: string | null;
  channels: { id: string; name: string; canJoin: boolean }[];
  queue: QueueView | null;
}

export interface BotView {
  id: string | null;
  tag: string;
  avatar: string | null;
  ready: boolean;
  uptimeMs: number;
}

export interface SpeakerView {
  id: string;
  name: string;
  kind: 'alice' | 'bridge';
  guildId: string;
  channelId: string;
  createdAt: number;
  lastSeenAt: number | null;
  online: boolean;
}

export interface LogEntryView {
  at: number;
  kind: string;
  text: string;
  ok: boolean;
}

export interface StateView {
  bot: BotView;
  guilds: GuildView[];
  speakers: SpeakerView[];
  lastTarget: WebTarget | null;
  radioStations: { id: string; name: string }[];
  skill: { name: string; publicUrl: string | null; configured: boolean };
  sampledAt: number;
  log: LogEntryView[];
}

export interface CommandResult {
  text: string;
  keepSession: boolean;
  ok: boolean;
}
