'use strict';

const { spawn } = require('child_process');
const { StreamType } = require('@discordjs/voice');

/**
 * Получает аудиопоток через yt-dlp + ffmpeg для YouTube-видео.
 * yt-dlp загружает аудио, ffmpeg перекодирует в Opus для Discord.
 *
 * @param {string} url — YouTube URL
 * @returns {{ stream: import('stream').Readable, type: StreamType }}
 */
function createYtdlpStream(url) {
  const ytdlp = spawn('yt-dlp', [
    '-f', 'bestaudio[ext=webm]/bestaudio',
    '-o', '-',          // output to stdout
    '--quiet',
    '--no-warnings',
    '--no-check-certificates',
    url,
  ], {
    stdio: ['ignore', 'pipe', 'ignore'],
    windowsHide: true,
  });

  // ffmpeg конвертирует в формат Opus/OGG — нативный для Discord
  const ffmpegPath = require('ffmpeg-static');
  const ffmpeg = spawn(ffmpegPath, [
    '-i', 'pipe:0',         // input from stdin
    '-analyzeduration', '0',
    '-loglevel', '0',
    '-f', 'opus',            // output format
    '-acodec', 'libopus',
    '-ar', '48000',          // Discord requires 48kHz
    '-ac', '2',              // stereo
    '-b:a', '128k',
    'pipe:1',                // output to stdout
  ], {
    stdio: ['pipe', 'pipe', 'ignore'],
    windowsHide: true,
  });

  ytdlp.stdout.pipe(ffmpeg.stdin);

  // Корректно завершаем процессы при ошибках
  ytdlp.on('error', () => { ffmpeg.kill(); });
  ffmpeg.on('error', () => { ytdlp.kill(); });
  ytdlp.on('exit', (code) => {
    if (code !== 0) ffmpeg.stdin.end();
  });

  // Прикрепляем ссылки для cleanup
  ffmpeg.stdout._ytdlpProc = ytdlp;
  ffmpeg.stdout._ffmpegProc = ffmpeg;

  return {
    stream: ffmpeg.stdout,
    type: StreamType.OggOpus,
  };
}

/**
 * Уничтожает stream и процессы, созданные createYtdlpStream.
 * @param {import('stream').Readable} stream
 */
function destroyStream(stream) {
  try { stream._ytdlpProc?.kill(); } catch (_) {}
  try { stream._ffmpegProc?.kill(); } catch (_) {}
  try { stream.destroy(); } catch (_) {}
}

module.exports = { createYtdlpStream, destroyStream };
