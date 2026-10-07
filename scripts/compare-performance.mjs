import { readFile } from 'node:fs/promises';
const [baselinePath, candidatePath] = process.argv.slice(2);
if (!baselinePath || !candidatePath) throw new Error('Usage: node scripts/compare-performance.mjs baseline.json candidate.json');
const [baseline, candidate] = await Promise.all([baselinePath, candidatePath].map(async file => JSON.parse(await readFile(file, 'utf8'))));
const names = ['search.cold.idle', 'search.cold.music', 'search.shared.music', 'voice.with-music', 'audio.prepared-transition'];
const checks = names.map(name => {
  const before = baseline.summary[name], after = candidate.summary[name];
  if (!after || after.count < 30 || (name !== 'audio.prepared-transition' && (!before || before.count < 30))) return { metric: name, result: 'insufficient_samples' };
  const improvementPercent = before?.count >= 30 ? (1 - after.p95Ms / before.p95Ms) * 100 : null;
  const passed = name === 'audio.prepared-transition' ? after.p95Ms <= 1000 : improvementPercent >= 20;
  return { metric: name, beforeP95Ms: before?.count >= 30 ? before.p95Ms : null, afterP95Ms: after.p95Ms, improvementPercent, result: passed ? 'pass' : 'target_not_met' };
});
const resourceSummary = report => {
  const usage = report.usage ?? [];
  if (!usage.length) return null;
  const cpu = usage.slice(1).map((row, i) => Math.max(0, (row.cpuSeconds - usage[i].cpuSeconds) * 100_000 / (row.atMs - usage[i].atMs)));
  const loops = usage.map(row => row.eventLoopP95Ms).filter(Number.isFinite).sort((a, b) => a - b);
  return { peakRssMiB: Math.max(...usage.map(row => row.rssBytes)) / 1048576,
    peakProcesses: Math.max(...usage.map(row => row.processes)),
    meanCpuPercent: cpu.length ? cpu.reduce((n, value) => n + value, 0) / cpu.length : null,
    eventLoopP95Ms: loops[Math.ceil(loops.length * .95) - 1] ?? null };
};
console.log(JSON.stringify({ checks, baselineResources: resourceSummary(baseline), candidateResources: resourceSummary(candidate),
  baselineFailures: baseline.failures, candidateFailures: candidate.failures,
  soakElapsedMs: candidate.soakElapsedMs, soakAudioMs: candidate.soakAudioMs }, null, 2));
if (checks.some(check => check.result !== 'pass') || Object.values(candidate.failures ?? {}).some(Boolean)
  || candidate.stage !== 'complete' || candidate.soakElapsedMs < 3_600_000) process.exitCode = 1;
