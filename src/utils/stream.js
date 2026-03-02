'use strict';

const { spawn, spawnSync } = require('child_process');
const { StreamType } = require('@discordjs/voice');
const { spawnYtdlp, getYtdlpCommandForLogs } = require('./ytdlp');

let cachedFfmpegCommand = null;

function resolveFfmpegCommand() {
  if (cachedFfmpegCommand) return cachedFfmpegCommand;

  const ffmpegStatic = require('ffmpeg-static');
  const candidates = [
    process.env.FFMPEG_PATH?.trim(),
    ffmpegStatic,
    'ffmpeg',
  ].filter(Boolean);

  for (const cmd of candidates) {
    try {
      const probe = spawnSync(cmd, ['-version'], { stdio: 'ignore', windowsHide: true });
      if (probe.status === 0) {
        cachedFfmpegCommand = cmd;
        return cachedFfmpegCommand;
      }
    } catch (_) {}
  }

  cachedFfmpegCommand = candidates[0] || 'ffmpeg';
  return cachedFfmpegCommand;
}

function createYtdlpStream(url) {
  const ytdlp = spawnYtdlp([
    '-f', 'bestaudio[ext=webm]/bestaudio',
    '-o', '-',
    '--quiet',
    '--no-warnings',
    '--no-check-certificates',
    url,
  ], {
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  const ffmpegCommand = resolveFfmpegCommand();
  const ffmpeg = spawn(ffmpegCommand, [
    '-i', 'pipe:0',
    '-analyzeduration', '0',
    '-loglevel', '0',
    '-f', 'ogg',
    '-acodec', 'libopus',
    '-ar', '48000',
    '-ac', '2',
    '-b:a', '128k',
    'pipe:1',
  ], {
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });

  ytdlp.stdout.pipe(ffmpeg.stdin);

  // Поглощаем ошибку "write after end" при уничтожении потока
  ffmpeg.stdin.on('error', () => {});

  let failed = false;
  let ytdlpStderr = '';
  let ffmpegStderr = '';
  const cap = 400;

  ytdlp.stderr?.on('data', (chunk) => {
    if (ytdlpStderr.length < cap) ytdlpStderr += String(chunk);
  });
  ffmpeg.stderr?.on('data', (chunk) => {
    if (ffmpegStderr.length < cap) ffmpegStderr += String(chunk);
  });

  const fail = (message) => {
    if (failed) return;
    failed = true;
    try { ffmpeg.stdin.end(); } catch (_) {}
    try { ytdlp.kill(); } catch (_) {}
    try { ffmpeg.kill(); } catch (_) {}
    ffmpeg.stdout.destroy(new Error(message));
  };

  ytdlp.on('error', (err) => {
    fail(`yt-dlp start failed (${getYtdlpCommandForLogs()}): ${err.message}`);
  });
  ffmpeg.on('error', (err) => {
    fail(`ffmpeg start failed (${ffmpegCommand}): ${err.message}`);
  });
  ytdlp.on('exit', (code) => {
    if (code !== 0) {
      fail(`yt-dlp exited with code ${code}: ${ytdlpStderr.trim() || 'no details'}`);
      return;
    }
    try { ffmpeg.stdin.end(); } catch (_) {}
  });
  ffmpeg.on('exit', (code) => {
    if (code !== 0) {
      fail(`ffmpeg exited with code ${code}: ${ffmpegStderr.trim() || 'no details'}`);
    }
  });

  ffmpeg.stdout._ytdlpProc = ytdlp;
  ffmpeg.stdout._ffmpegProc = ffmpeg;

  return {
    stream: ffmpeg.stdout,
    type: StreamType.OggOpus,
  };
}

function destroyStream(stream) {
  // Отвязываем pipe yt-dlp → ffmpeg.stdin перед убийством процессов
  try { stream._ytdlpProc?.stdout?.unpipe(); } catch (_) {}
  try { stream._ytdlpProc?.kill(); } catch (_) {}
  try { stream._ffmpegProc?.stdin?.end(); } catch (_) {}
  try { stream._ffmpegProc?.kill(); } catch (_) {}
  try { stream.destroy(); } catch (_) {}
}

module.exports = { createYtdlpStream, destroyStream };
