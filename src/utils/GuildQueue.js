'use strict';

const {
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
  getVoiceConnection,
  NoSubscriberBehavior,
} = require('@discordjs/voice');
const { execYtdlp, getYtdlpCommandForLogs } = require('./ytdlp');
const { createYtdlpStream, destroyStream } = require('./stream');

/** Таймаут бездействия до авто-отключения (5 минут) */
const IDLE_TIMEOUT_MS = 5 * 60 * 1_000;

/** Параметры fade-out */
const FADE_DURATION_MS = 1500;
const FADE_STEPS = 15;
const FADE_BEFORE_END_MS = 2000;

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

    // ── Fade-out state ──────────────────────────────────────────────────
    this._fadeInterval = null;
    this._isFading = false;
    this._scheduledFadeTimer = null;

    // ── Time tracking (для плеера и scheduled fade) ─────────────────────
    this._trackStartedAt = null;
    this._totalPausedMs = 0;
    this._pauseStartedAt = null;

    // ── Player message ──────────────────────────────────────────────────
    this._playerMessage = null;
    this._playerInterval = null;
    this._playerChannelId = null;
    this._playerUpdating = false;

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
      this._trackStartedAt = null;
      this._updatePlayerMessage();
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

      // Сброс time tracking
      this._trackStartedAt = Date.now();
      this._totalPausedMs = 0;
      this._pauseStartedAt = null;

      this.player.play(resource);
      console.log(`[Queue:${this.guildId}] ▶ ${track.title}`);

      // Запланировать fade перед концом трека
      this._scheduleFadeOut(track.duration);

      // Обновить плеер
      this._updatePlayerMessage();
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
        this._trackStartedAt = null;
        this._updatePlayerMessage();
        this._startIdleTimer();
        return;
      }

      // Берём 2-й трек (1-й обычно текущий), если доступен
      const next = related.length > 1 ? related[1] : related[0];
      await this._playTrack(next);
    } catch (err) {
      console.error(`[Queue:${this.guildId}] Autoplay error:`, err.message);
      this.currentTrack = null;
      this._trackStartedAt = null;
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
      execYtdlp([
        '--flat-playlist',
        '--print', '%(id)s\t%(title)s\t%(duration_string)s',
        '--playlist-items', '2:6',  // пропускаем текущий (1), берём 2-6
        '--quiet',
        '--no-warnings',
        mixUrl,
      ], { timeout: 15_000, windowsHide: true }, (err, stdout) => {
        if (err || !stdout.trim()) {
          if (err) console.error(`yt-dlp related error (${getYtdlpCommandForLogs()}): ${err.message}`);
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

  // ─── Fade-out ─────────────────────────────────────────────────────────────

  /**
   * Плавное снижение громкости, потом callback.
   * @param {Function} callback
   * @param {number} duration — длительность fade в мс
   */
  _fadeOutAndExecute(callback, duration = FADE_DURATION_MS) {
    this._cancelFade();

    const resource = this.player.state.resource;
    if (!resource?.volume) {
      callback();
      return;
    }

    this._isFading = true;
    const startVol = resource.volume.volume;
    const step = startVol / FADE_STEPS;
    const interval = Math.floor(duration / FADE_STEPS);
    let remaining = FADE_STEPS;

    this._fadeInterval = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        this._cancelFade();
        resource.volume.setVolume(0);
        callback();
        return;
      }
      resource.volume.setVolume(Math.max(0, step * remaining));
    }, interval);
  }

  _cancelFade() {
    if (this._fadeInterval) {
      clearInterval(this._fadeInterval);
      this._fadeInterval = null;
    }
    this._isFading = false;
  }

  /**
   * Парсит строку длительности (например "4:30" или "1:02:15") в секунды.
   * @returns {number|null}
   */
  _parseDuration(str) {
    if (!str || str === '?' || str === '—') return null;
    const parts = str.split(':').map(Number);
    if (parts.some(isNaN)) return null;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 1) return parts[0];
    return null;
  }

  /**
   * Планирует fade-out за FADE_BEFORE_END_MS до конца трека.
   */
  _scheduleFadeOut(durationStr) {
    this._cancelScheduledFade();
    const totalSec = this._parseDuration(durationStr);
    if (!totalSec || totalSec < 5) return; // слишком короткий или неизвестный — не планируем

    const fadeStartMs = (totalSec * 1000) - FADE_BEFORE_END_MS;
    if (fadeStartMs <= 0) return;

    this._scheduledFadeTimer = setTimeout(() => {
      this._scheduledFadeTimer = null;
      // Проверяем, что ещё играет и не на паузе
      if (this.player.state.status !== AudioPlayerStatus.Playing) return;
      if (this._isFading) return;

      this._fadeOutAndExecute(() => {
        // Ничего не делаем — трек сам закончится и AudioPlayerStatus.Idle сработает
        // Просто восстановим громкость для следующего трека
      }, FADE_BEFORE_END_MS);
    }, fadeStartMs);
  }

  _cancelScheduledFade() {
    if (this._scheduledFadeTimer) {
      clearTimeout(this._scheduledFadeTimer);
      this._scheduledFadeTimer = null;
    }
  }

  // ─── Player message ───────────────────────────────────────────────────────

  /**
   * Возвращает прошедшее время трека в секундах с учётом пауз.
   */
  getElapsedSeconds() {
    if (!this._trackStartedAt) return 0;
    let elapsed = Date.now() - this._trackStartedAt - this._totalPausedMs;
    if (this._pauseStartedAt) {
      elapsed -= (Date.now() - this._pauseStartedAt);
    }
    return Math.max(0, Math.floor(elapsed / 1000));
  }

  /**
   * Устанавливает канал для плеер-сообщения.
   */
  setPlayerChannel(channelId) {
    this._playerChannelId = channelId;
  }

  /**
   * Отправляет или обновляет плеер-сообщение.
   */
  async _updatePlayerMessage() {
    if (!this._playerChannelId) return;
    if (this._playerUpdating) return; // пропускаем, если предыдущее обновление ещё идёт
    this._playerUpdating = true;

    try {
      const { playerEmbed, playerActionRow } = require('./embeds');
      const embed = playerEmbed(this);
      const row = playerActionRow(this);

      if (this._playerMessage) {
        try {
          await this._playerMessage.edit({ embeds: [embed], components: [row] });
          return;
        } catch (_) {
          // Сообщение удалено — создадим новое
          this._playerMessage = null;
        }
      }

      const channel = await this.client.channels.fetch(this._playerChannelId).catch(() => null);
      if (!channel) return;

      this._playerMessage = await channel.send({ embeds: [embed], components: [row] });
    } catch (err) {
      console.error(`[Queue:${this.guildId}] Player message error: ${err.message}`);
    } finally {
      this._playerUpdating = false;
    }
  }

  _startPlayerInterval() {
    this._stopPlayerInterval();
    this._playerInterval = setInterval(() => {
      this._updatePlayerMessage();
    }, 1_000);
  }

  _stopPlayerInterval() {
    if (this._playerInterval) {
      clearInterval(this._playerInterval);
      this._playerInterval = null;
    }
  }

  async _destroyPlayerMessage() {
    this._stopPlayerInterval();
    if (this._playerMessage) {
      try { await this._playerMessage.delete(); } catch (_) {}
      this._playerMessage = null;
    }
  }

  // ─── Cleanup ──────────────────────────────────────────────────────────────

  _cleanup() {
    this._clearIdleTimer();
    this._cancelFade();
    this._cancelScheduledFade();
    this._stopPlayerInterval();
    this._destroyPlayerMessage();
    this._destroyCurrentStream();
    this._advancing = false;
    this.currentTrack = null;
    this._trackStartedAt = null;
    this.tracks = [];

    try { this.player.stop(true); } catch (_) {}

    if (this.connection) {
      try { this.connection.destroy(); } catch (_) {}
      this.connection = null;
    }

    // Fallback: уничтожить orphaned connection через getVoiceConnection
    const orphaned = getVoiceConnection(this.guildId);
    if (orphaned) {
      try { orphaned.destroy(); } catch (_) {}
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

    this._cancelScheduledFade();

    if (this.player.state.status === AudioPlayerStatus.Playing && !this._isFading) {
      // Плавный skip с fade-out
      this._fadeOutAndExecute(() => {
        this._destroyCurrentStream();
        this.player.stop(true);
      });
    } else {
      // Уже fading (быстрый повторный skip) или не играет — сразу stop
      this._cancelFade();
      this._destroyCurrentStream();
      this.player.stop(true);
    }

    this._updatePlayerMessage();
    return true;
  }

  stop() {
    this._cancelScheduledFade();

    if (this.player.state.status === AudioPlayerStatus.Playing && !this._isFading) {
      this._fadeOutAndExecute(() => {
        this._cleanup();
      });
    } else {
      this._cancelFade();
      this._cleanup();
    }
  }

  pause() {
    if (this.player.state.status !== AudioPlayerStatus.Playing) return false;
    this._pauseStartedAt = Date.now();
    this.player.pause();
    this._updatePlayerMessage();
    return true;
  }

  resume() {
    if (this.player.state.status !== AudioPlayerStatus.Paused) return false;
    if (this._pauseStartedAt) {
      this._totalPausedMs += Date.now() - this._pauseStartedAt;
      this._pauseStartedAt = null;
    }
    this.player.unpause();
    this._updatePlayerMessage();
    return true;
  }

  setVolume(percent) {
    this.volume = Math.max(0, Math.min(percent / 100, 1.5));
    if (this.player.state.status === AudioPlayerStatus.Playing) {
      this.player.state.resource?.volume?.setVolume(this.volume);
    }
    this._updatePlayerMessage();
  }

  toggleAutoplay() {
    this.autoplay = !this.autoplay;
    this._updatePlayerMessage();
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
