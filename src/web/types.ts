// View models for the web control panel. They contain no tokens or secrets.

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

export interface LogEntryView {
  at: number;
  kind: string;
  text: string;
  ok: boolean;
}

export interface StateView {
  bot: BotView;
  guilds: GuildView[];
  lastTarget: WebTarget | null;
  radioStations: { id: string; name: string }[];
  sampledAt: number;
  log: LogEntryView[];
}
