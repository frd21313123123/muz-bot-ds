import { monitorEventLoopDelay, performance } from 'node:perf_hooks';
import { readFile, readdir } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

export type Metric = 'command.ack' | 'command.total' | 'metadata.wait' | 'metadata.work'
  | 'search' | 'audio.first' | 'audio.start' | 'audio.transition' | 'speech.wait'
  | 'speech.recognize' | 'voice.action' | 'voice.total';
const samples = new Map<Metric, number[]>();
const counters = new Map<string, number>();
const children = new Set<number>();
const previousCpu = new Map<string, number>();
let retiredCpu = 0;
const commands = new WeakMap<object, { started: number; acknowledged: boolean }>();
export function beginCommand(interaction: object): void { commands.set(interaction, { started: performance.now(), acknowledged: false }); }
export async function acknowledgeCommand<T>(interaction: object, reply: Promise<T>): Promise<T> {
  const value = await reply;
  const timing = commands.get(interaction);
  if (timing && !timing.acknowledged) { timing.acknowledged = true; recordMetric('command.ack', performance.now() - timing.started); }
  return value;
}
export function finishCommand(interaction: object): void {
  const timing = commands.get(interaction);
  if (timing) recordMetric('command.total', performance.now() - timing.started);
  commands.delete(interaction);
}
export const performanceEnabled = (): boolean => process.env.BOT_PERF === '1';
export function recordMetric(name: Metric, ms: number): void {
  if (!performanceEnabled() || !Number.isFinite(ms) || ms < 0) return;
  const values = samples.get(name) ?? [];
  if (values.length === 4096) values.shift();
  values.push(ms); samples.set(name, values);
}
export function countMetric(name: 'cache.hit' | 'cache.miss' | 'request.shared' | 'audio.prefetch.hit' | 'audio.prefetch.failed'): void {
  if (performanceEnabled()) counters.set(name, (counters.get(name) ?? 0) + 1);
}
export function metricSummary(): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [name, values] of samples) {
    const sorted = [...values].sort((a, b) => a - b);
    result[name] = { count: sorted.length, medianMs: sorted[Math.ceil(sorted.length * .5) - 1],
      p95Ms: sorted[Math.ceil(sorted.length * .95) - 1] };
  }
  return { metrics: result, counters: Object.fromEntries(counters) };
}
export function trackProcess(child: import('node:child_process').ChildProcess): void {
  const pid = child.pid;
  if (pid) children.add(pid);
  child.once('close', () => { if (pid) children.delete(pid); });
}

// Include descendants (e.g. yt-dlp's JavaScript runtime), without reading command lines.
export async function processUsage(): Promise<{ processes: number; rssBytes: number; cpuSeconds: number } | null> {
  if (process.platform === 'linux') {
    const entries = await readdir('/proc');
    const rows = (await Promise.all(entries.filter(name => /^\d+$/.test(name)).map(async name => {
      try {
        const [stat, status] = await Promise.all([readFile(`/proc/${name}/stat`, 'utf8'), readFile(`/proc/${name}/status`, 'utf8')]);
        const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
        return { pid: Number(name), parent: Number(fields[1]), identity: `${name}:${fields[19]}`, ticks: Number(fields[11]) + Number(fields[12]),
          rss: Number(status.match(/^VmRSS:\s+(\d+)/m)?.[1] ?? 0) * 1024 };
      } catch { return null; }
    }))).filter(row => row !== null);
    const ids = new Set([process.pid, ...children]);
    let changed = true;
    while (changed) { changed = false; for (const row of rows) if (ids.has(row.parent) && !ids.has(row.pid)) { ids.add(row.pid); changed = true; } }
    // Linux USER_HZ is conventionally 100; obtain the actual value once per report.
    const { stdout } = await promisify(execFile)('getconf', ['CLK_TCK'], { timeout: 2000 });
    const hz = Number(stdout.trim());
    if (!hz || !Number.isFinite(hz)) return null;
    const tree = rows.filter(row => ids.has(row.pid));
    const current = new Set(tree.map(row => row.identity));
    for (const [key, cpu] of previousCpu) if (!current.has(key)) { retiredCpu += cpu; previousCpu.delete(key); }
    for (const row of tree) previousCpu.set(row.identity, row.ticks / hz);
    return { processes: tree.length, rssBytes: tree.reduce((n, row) => n + row.rss, 0),
      cpuSeconds: retiredCpu + [...previousCpu.values()].reduce((n, seconds) => n + seconds, 0) };
  }
  if (process.platform === 'win32') {
    const ids = [process.pid, ...children].join(',');
    try {
      const { stdout } = await promisify(execFile)('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
        `Get-Process -Id ${ids} -ErrorAction SilentlyContinue | Select-Object Id,WorkingSet64,CPU | ConvertTo-Json -Compress`],
      { timeout: 4000, windowsHide: true });
      const value: unknown = JSON.parse(stdout);
      const rows = (Array.isArray(value) ? value : [value]) as { WorkingSet64: number; CPU: number }[];
      return { processes: rows.length, rssBytes: rows.reduce((n, row) => n + row.WorkingSet64, 0), cpuSeconds: rows.reduce((n, row) => n + (row.CPU ?? 0), 0) };
    } catch { return null; }
  }
  return null;
}

export function startPerformanceMonitor(): () => void {
  if (!performanceEnabled()) return () => {};
  const loop = monitorEventLoopDelay({ resolution: 20 }); loop.enable();
  let busy = false, stopped = false;
  let previous: { time: number; cpu: number } | null = null;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void processUsage().then(usage => {
      if (stopped) return;
      const now = performance.now();
      const cpuPercent = usage && previous ? Math.max(0, (usage.cpuSeconds - previous.cpu) * 100_000 / (now - previous.time)) : null;
      if (usage) previous = { time: now, cpu: usage.cpuSeconds };
      console.log('[Performance]', JSON.stringify({ ...metricSummary(), usage, cpuPercent,
        eventLoopP95Ms: loop.percentile(95) / 1e6, nodeRssBytes: process.memoryUsage().rss }));
      loop.reset();
    }).catch(() => {}).finally(() => { busy = false; });
  }, 10_000);
  timer.unref();
  return () => { stopped = true; clearInterval(timer); loop.disable(); };
}
