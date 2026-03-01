'use strict';

const { spawn, spawnSync, execFile } = require('child_process');

let cachedRuntime = null;

function getCandidates() {
  const fromEnv = process.env.YT_DLP_PATH?.trim();
  const candidates = [];

  if (fromEnv) {
    candidates.push({ command: fromEnv, baseArgs: [] });
  }

  candidates.push(
    { command: 'yt-dlp', baseArgs: [] },
    { command: 'yt-dlp.exe', baseArgs: [] },
    { command: 'python3', baseArgs: ['-m', 'yt_dlp'] },
    { command: 'python', baseArgs: ['-m', 'yt_dlp'] },
  );

  return candidates;
}

function resolveRuntime() {
  if (cachedRuntime) return cachedRuntime;

  for (const candidate of getCandidates()) {
    try {
      const probe = spawnSync(
        candidate.command,
        [...candidate.baseArgs, '--version'],
        { stdio: 'ignore', windowsHide: true },
      );

      if (probe.status === 0) {
        cachedRuntime = candidate;
        return cachedRuntime;
      }
    } catch (_) {}
  }

  cachedRuntime = { command: 'yt-dlp', baseArgs: [] };
  return cachedRuntime;
}

function spawnYtdlp(args, options = {}) {
  const runtime = resolveRuntime();
  return spawn(runtime.command, [...runtime.baseArgs, ...args], options);
}

function execYtdlp(args, options = {}, callback) {
  const runtime = resolveRuntime();
  return execFile(
    runtime.command,
    [...runtime.baseArgs, ...args],
    options,
    callback,
  );
}

function getYtdlpCommandForLogs() {
  const runtime = resolveRuntime();
  return `${runtime.command}${runtime.baseArgs.length ? ` ${runtime.baseArgs.join(' ')}` : ''}`;
}

module.exports = { spawnYtdlp, execYtdlp, getYtdlpCommandForLogs };
