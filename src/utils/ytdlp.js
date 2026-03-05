'use strict';

const path = require('path');
const { spawn, spawnSync, execFile } = require('child_process');

let cachedRuntime = null;

function getCandidates() {
  const fromEnv = process.env.YT_DLP_PATH?.trim();
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const userLocalBin = homeDir ? path.join(homeDir, '.local', 'bin', 'yt-dlp') : null;
  const projectLocalBin = path.join(process.cwd(), '.local', 'bin', 'yt-dlp');
  const rootLocalBin = path.join(path.parse(process.cwd()).root || path.sep, '.local', 'bin', 'yt-dlp');
  const fixedContainerLocalBin = '/home/container/.local/bin/yt-dlp';
  const candidates = [];

  if (fromEnv) {
    candidates.push({ command: fromEnv, baseArgs: [] });
  }

  if (userLocalBin) {
    candidates.push({ command: userLocalBin, baseArgs: [] });
  }
  candidates.push({ command: projectLocalBin, baseArgs: [] });
  candidates.push({ command: rootLocalBin, baseArgs: [] });
  candidates.push({ command: fixedContainerLocalBin, baseArgs: [] });

  candidates.push(
    { command: 'yt-dlp', baseArgs: [] },
    { command: 'yt-dlp.exe', baseArgs: [] },
    { command: 'python3', baseArgs: ['-m', 'yt_dlp'] },
    { command: 'python', baseArgs: ['-m', 'yt_dlp'] },
  );

  return candidates;
}

function canRun(candidate) {
  try {
    const probe = spawnSync(
      candidate.command,
      [...candidate.baseArgs, '--version'],
      { stdio: 'ignore', windowsHide: true, timeout: 15_000 },
    );

    return probe.status === 0;
  } catch (_) {
    return false;
  }
}

function installViaPipUser() {
  const installers = [
    { command: 'python3', args: ['-m', 'pip', 'install', '-U', '--no-cache-dir', '--user', 'yt-dlp'] },
    { command: 'python', args: ['-m', 'pip', 'install', '-U', '--no-cache-dir', '--user', 'yt-dlp'] },
    { command: 'pip3', args: ['install', '-U', '--no-cache-dir', '--user', 'yt-dlp'] },
    { command: 'pip', args: ['install', '-U', '--no-cache-dir', '--user', 'yt-dlp'] },
  ];

  for (const installer of installers) {
    try {
      const result = spawnSync(
        installer.command,
        installer.args,
        { stdio: 'ignore', windowsHide: true, timeout: 120_000 },
      );

      if (result.status === 0) {
        return true;
      }
    } catch (_) {}
  }

  return false;
}

function resolveRuntime() {
  if (cachedRuntime) return cachedRuntime;

  for (const candidate of getCandidates()) {
    if (canRun(candidate)) {
      cachedRuntime = candidate;
      return cachedRuntime;
    }
  }

  if (installViaPipUser()) {
    for (const candidate of getCandidates()) {
      if (canRun(candidate)) {
        cachedRuntime = candidate;
        return cachedRuntime;
      }
    }
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
