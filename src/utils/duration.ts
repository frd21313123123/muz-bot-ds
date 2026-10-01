export function parseDuration(value: string | undefined): number | null {
  const trimmed = value?.trim();
  if (!trimmed || !/^\d+(?::\d{1,2}){1,2}$/.test(trimmed)) return null;
  const parts = trimmed.split(':').map(Number);
  if (parts.slice(1).some(part => part >= 60)) return null;
  const seconds = parts.reduce((total, part) => total * 60 + part, 0);
  return Number.isSafeInteger(seconds) && seconds > 0 ? seconds : null;
}

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = String(total % 60).padStart(2, '0');
  return total >= 3600 ? `${Math.floor(total / 3600)}:${String(minutes % 60).padStart(2, '0')}:${remainder}`
    : `${minutes}:${remainder}`;
}

export function videoDuration(info: { duration?: number; duration_string?: string }): number | null {
  return Number.isFinite(info.duration) && (info.duration ?? 0) > 0
    ? Math.floor(info.duration!) : parseDuration(info.duration_string);
}
