'use strict';

const {
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
  NoSubscriberBehavior,
} = require('@discordjs/voice');
const { execFile } = require('child_process');
const { createYtdlpStream, destroyStream } = require('./stream');

/** Таймаут бездействия до авто-отключения (5 минут) */
const IDLE_TIMEOUT_MS = 5 * 60 * 1_000;

/**
 * @typedef {Object} Track
 * @property {string} url
 * @property {string} videoId
 * @property {string} title
 * @property {string} duration
 * @property {string|null} thumbnail
 * @property {string} requestedBy
 * @property {boolean} [isAutoplay]
 */

class GuildQueue {
  /**
   * @param {string} guildId
   * @param {import('discord.js').Client} client
   */
  constructor(guildId, client) {
    this.guildId = guildId;
    this.client = client;

    /** @type {import('discord.js').VoiceChannel|null} */
    this.voiceChannel = null;

    /** @type {import('@discordjs/voice').VoiceConnection|null} */
    this.connection = null;

    this.player = createAudioPlayer({
      behaviors: { noSubscriber: NoSubscriberBehavior.Pause },
    });

    /** @type {Track[]} */
    this.tracks = [];

    /** @type {Track|null} */
    this.currentTrack = null;

    /** @type {import('stream').Readable|null} */
    this._currentStream = null;

    this.autoplay = false;
    this.volume = 0.5;

    this._advancing = false;
    this._idleTimer = null;

    this._setupPlayerListeners();
  }

  // ─── Внутренние методы ────────────────────────────────────────────────────

  _setupPlayerListeners() {
    this.player.on(AudioPlayerStatus.Idle, async () => {
      this._destroyCurrentStream();
      if (this._advancing) return;
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    });

    this.player.on('error', async (err) => {
      console.error(`[Queue:${this.guildId}] Player error: ${err.message}`);
      this._destroyCurrentStream();

      if (this._advancing) return;
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    });
  }

  async _advanceQueue() {
    if (this.tracks.length > 0) {
      await this._playTrack(this.tracks.shift());
    } else if (this.autoplay && this.currentTrack) {
      await this._playAutoplay();
    } else {
      this.currentTrack = null;
      this._startIdleTimer();
    }
  }

  /** @param {Track} track */
  async _playTrack(track) {
    this.currentTrack = track;

    const streamUrl = (track.url || '')
      .replace('music.youtube.com', 'www.youtube.com');

    if (!streamUrl || !streamUrl.startsWith('http')) {
      console.error(`[Queue:${this.guildId}] Bad URL for "${track.title}": "${streamUrl}"`);
      await this._advanceQueue();
      return;
    }

    try {
      const { stream, type } = createYtdlpStream(streamUrl);
      this._currentStream = stream;

      const resource = createAudioResource(stream, {
        inputType: type,
        inlineVolume: true,
      });
      resource.volume?.setVolume(this.volume);

      this._clearIdleTimer();
      this.player.play(resource);
      console.log(`[Queue:${this.guildId}] ▶ ${track.title}`);
    } catch (err) {
      console.error(`[Queue:${this.guildId}] Stream error:`, err.message);
      this._destroyCurrentStream();
      await this._advanceQueue();
    }
  }

  async _playAutoplay() {
    try {
      const related = await this._fetchRelatedTracks(this.currentTrack.videoId);
      if (!related || related.length === 0) {
        console.log(`[Queue:${this.guildId}] Нет рекомендаций — остановка автовоспроизведения`);
        this.currentTrack = null;
        this._startIdleTimer();
        return;
      }

      // Берём 2-й трек (1-й обычно текущий), если доступен
      const next = related.length > 1 ? related[1] : related[0];
      await this._playTrack(next);
    } catch (err) {
      console.error(`[Queue:${this.guildId}] Autoplay error:`, err.message);
      this.currentTrack = null;
      this._startIdleTimer();
    }
  }

  /**
   * Получает рекомендованные треки через yt-dlp Radio Mix (RD{videoId}).
   * @returns {Promise<Track[]>}
   */
  async _fetchRelatedTracks(videoId) {
    const mixUrl = `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`;

    return new Promise((resolve) => {
      execFile('yt-dlp', [
        '--flat-playlist',
        '--print', '%(id)s\t%(title)s\t%(duration_string)s',
        '--playlist-items', '2:6',  // пропускаем текущий (1), берём 2-6
        '--quiet',
        '--no-warnings',
        mixUrl,
      ], { timeout: 15_000, windowsHide: true }, (err, stdout) => {
        if (err || !stdout.trim()) {
          if (err) console.error('yt-dlp related error:', err.message);
          return resolve([]);
        }

        const tracks = stdout.trim().split('\n').map(line => {
          const [id, title, duration] = line.split('\t');
          return {
            url: `https://www.youtube.com/watch?v=${id}`,
            videoId: id,
            title: title || 'Без названия',
            duration: duration || '?',
            thumbnail: `https://img.youtube.com/vi/${id}/hqdefault.jpg`,
            requestedBy: '🤖 Автовоспроизведение',
            isAutoplay: true,
          };
        }).filter(t => t.videoId && t.videoId !== videoId);

        resolve(tracks);
      });
    });
  }

  _destroyCurrentStream() {
    if (this._currentStream) {
      destroyStream(this._currentStream);
      this._currentStream = null;
    }
  }

  _startIdleTimer() {
    this._clearIdleTimer();
    this._idleTimer = setTimeout(() => {
      if (!this.currentTrack) {
        this._cleanup();
      }
    }, IDLE_TIMEOUT_MS);
  }

  _clearIdleTimer() {
    if (this._idleTimer) {
      clearTimeout(this._idleTimer);
      this._idleTimer = null;
    }
  }

  _cleanup() {
    this._clearIdleTimer();
    this._destroyCurrentStream();
    this._advancing = false;
    this.currentTrack = null;
    this.tracks = [];
    this.player.stop(true);

    if (this.connection) {
      try { this.connection.destroy(); } catch (_) {}
      this.connection = null;
    }

    this.client.queues.delete(this.guildId);
  }

  // ─── Публичный API ────────────────────────────────────────────────────────

  async join(voiceChannel) {
    this.voiceChannel = voiceChannel;

    if (this.connection) {
      try { this.connection.destroy(); } catch (_) {}
    }

    this.connection = joinVoiceChannel({
      channelId: voiceChannel.id,
      guildId: voiceChannel.guild.id,
      adapterCreator: voiceChannel.guild.voiceAdapterCreator,
      selfDeaf: true,
    });

    this.connection.on(VoiceConnectionStatus.Disconnected, async () => {
      try {
        await Promise.race([
          entersState(this.connection, VoiceConnectionStatus.Signalling, 5_000),
          entersState(this.connection, VoiceConnectionStatus.Connecting, 5_000),
        ]);
      } catch {
        this._cleanup();
      }
    });

    try {
      await entersState(this.connection, VoiceConnectionStatus.Ready, 30_000);
    } catch {
      this.connection.destroy();
      this.connection = null;
      throw new Error('Не удалось подключиться к голосовому каналу в течение 30 секунд.');
    }

    this.connection.subscribe(this.player);
  }

  async addTrack(track) {
    const wasIdle = !this.currentTrack && this.tracks.length === 0;
    this.tracks.push(track);
    if (wasIdle && !this._advancing) {
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    }
  }

  async addTracks(tracks) {
    const wasIdle = !this.currentTrack && this.tracks.length === 0;
    this.tracks.push(...tracks);
    if (wasIdle && !this._advancing) {
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    }
  }

  skip() {
    if (!this.currentTrack) return false;
    this._destroyCurrentStream();
    this.player.stop(true);
    return true;
  }

  stop() {
    this._cleanup();
  }

  pause() {
    if (this.player.state.status !== AudioPlayerStatus.Playing) return false;
    this.player.pause();
    return true;
  }

  resume() {
    if (this.player.state.status !== AudioPlayerStatus.Paused) return false;
    this.player.unpause();
    return true;
  }

  setVolume(percent) {
    this.volume = Math.max(0, Math.min(percent / 100, 1.5));
    if (this.player.state.status === AudioPlayerStatus.Playing) {
      this.player.state.resource?.volume?.setVolume(this.volume);
    }
  }

  toggleAutoplay() {
    this.autoplay = !this.autoplay;
    return this.autoplay;
  }

  clearQueue() {
    this.tracks = [];
  }

  get status() {
    return this.player.state.status;
  }
}

module.exports = GuildQueue;
