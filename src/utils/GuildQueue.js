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
const { debugLog } = require('./debugLog');

/** Таймаут бездействия до авто-отключения (5 минут) */
const IDLE_TIMEOUT_MS = 5 * 60 * 1_000;

/** Параметры fade-out */
const FADE_DURATION_MS = 1500;
const FADE_STEPS = 15;
const PLAYER_UPDATE_INTERVAL_MS = 10_000;
const EARLY_IDLE_TOLERANCE_SEC = 2;
const EARLY_IDLE_MAX_WAIT_SEC = 15;
const RADIO_BATCH_SIZE = 25;
const RADIO_FETCH_TIMEOUT_MS = 15_000;
const VOICE_READY_TIMEOUT_MS = 30_000;
const VOICE_RECONNECT_TIMEOUT_MS = 5_000;
const VOICE_JOIN_ATTEMPTS = 3;
const VOICE_JOIN_RETRY_DELAY_MS = 1_000;

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
    this.loopCurrent = false;
    this.volume = 0.5;

    this._advancing = false;
    this._idleTimer = null;
    this._isJoining = false;
    this._joinPromise = null;

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
    this._playerUpdateQueued = false;

    this._setupPlayerListeners();
  }

  // ─── Внутренние методы ────────────────────────────────────────────────────

  _setupPlayerListeners() {
    this.player.on(AudioPlayerStatus.Idle, async () => {
      debugLog(`[Queue:${this.guildId}] player idle (current=${this.currentTrack?.title || 'none'})`);
      const idleTrack = this.currentTrack;
      this._destroyCurrentStream();
      if (this._advancing) return;
      this._advancing = true;

      const earlyIdleWaitMs = this._getEarlyIdleWaitMs(idleTrack);
      if (earlyIdleWaitMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, earlyIdleWaitMs));
        if (idleTrack !== this.currentTrack) {
          this._advancing = false;
          return;
        }
        if (this.player.state.status !== AudioPlayerStatus.Idle) {
          this._advancing = false;
          return;
        }
      }

      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    });

    this.player.on('stateChange', (_oldState, newState) => {
      debugLog(`[Queue:${this.guildId}] player state -> ${newState.status}`);
      if (newState.status !== AudioPlayerStatus.Playing) return;
      if (this._trackStartedAt || !this.currentTrack) return;

      this._trackStartedAt = Date.now();
      this._updatePlayerMessage();
    });

    this.player.on('error', async (err) => {
      debugLog(`[Queue:${this.guildId}] player error: ${err.message}`);
      console.error(`[Queue:${this.guildId}] Player error: ${err.message}`);
      this._destroyCurrentStream();

      if (this._advancing) return;
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    });
  }

  async _advanceQueue() {
    if (this.loopCurrent && this.currentTrack) {
      await this._playTrack(this.currentTrack);
    } else if (this.tracks.length > 0) {
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
    debugLog(`[Queue:${this.guildId}] _playTrack start title="${track?.title}" url="${track?.url}"`);

    const streamUrl = (track.url || '')
      .replace('music.youtube.com', 'www.youtube.com');

    if (!streamUrl || !streamUrl.startsWith('http')) {
      debugLog(`[Queue:${this.guildId}] bad stream url: "${streamUrl}"`);
      console.error(`[Queue:${this.guildId}] Bad URL for "${track.title}": "${streamUrl}"`);
      await this._advanceQueue();
      return;
    }

    try {
      const { stream, type } = createYtdlpStream(streamUrl);
      debugLog(`[Queue:${this.guildId}] stream created type=${type}`);
      this._currentStream = stream;

      const resource = createAudioResource(stream, {
        inputType: type,
        inlineVolume: true,
      });
      resource.volume?.setVolume(this.volume);

      this._clearIdleTimer();

      // Сброс time tracking
      this._trackStartedAt = null;
      this._totalPausedMs = 0;
      this._pauseStartedAt = null;

      this.player.play(resource);
      debugLog(`[Queue:${this.guildId}] player.play called`);
      console.log(`[Queue:${this.guildId}] ▶ ${track.title}`);

      // Обновить плеер
      this._updatePlayerMessage();
    } catch (err) {
      debugLog(`[Queue:${this.guildId}] stream setup error: ${err.message}`);
      console.error(`[Queue:${this.guildId}] Stream error:`, err.message);
      this._destroyCurrentStream();
      await this._advanceQueue();
    }
  }

  async _playAutoplay() {
    try {
      const seedVideoId = this._resolveVideoId(this.currentTrack);
      if (!seedVideoId) {
        console.error(`[Queue:${this.guildId}] Infinite mode stopped: current track has no videoId`);
        this.currentTrack = null;
        this._trackStartedAt = null;
        this._updatePlayerMessage();
        this._startIdleTimer();
        return;
      }

      const related = await this._fetchRelatedTracks(seedVideoId, RADIO_BATCH_SIZE);
      if (!related || related.length === 0) {
        console.log(`[Queue:${this.guildId}] Нет рекомендаций — остановка бесконечного воспроизведения`);
        this.currentTrack = null;
        this._trackStartedAt = null;
        this._updatePlayerMessage();
        this._startIdleTimer();
        return;
      }

      const [next, ...rest] = related;
      if (rest.length > 0) {
        this.tracks.push(...rest);
      }
      await this._playTrack(next);
    } catch (err) {
      console.error(`[Queue:${this.guildId}] Infinite mode error:`, err.message);
      this.currentTrack = null;
      this._trackStartedAt = null;
      this._updatePlayerMessage();
      this._startIdleTimer();
    }
  }

  /**
   * Получает рекомендованные треки через yt-dlp Radio Mix (RD{videoId}).
   * @param {number} [maxTracks]
   * @returns {Promise<Track[]>}
   */
  async _fetchRelatedTracks(videoId, maxTracks = RADIO_BATCH_SIZE) {
    if (!videoId) return [];

    const mixUrl = `https://www.youtube.com/watch?v=${videoId}&list=RD${videoId}`;
    const safeMax = Math.max(1, Math.min(maxTracks, 50));
    const playlistItems = `2:${safeMax + 1}`;

    return new Promise((resolve) => {
      execYtdlp([
        '--flat-playlist',
        '-J',
        '--playlist-items', playlistItems, // пропускаем текущий (1), берём 2..N+1
        '--quiet',
        '--no-warnings',
        mixUrl,
      ], { timeout: RADIO_FETCH_TIMEOUT_MS, windowsHide: true }, (err, stdout) => {
        if (err || !stdout.trim()) {
          if (err) console.error(`yt-dlp related error (${getYtdlpCommandForLogs()}): ${err.message}`);
          return resolve([]);
        }

        let payload;
        try {
          payload = JSON.parse(stdout);
        } catch (parseErr) {
          console.error(`[Queue:${this.guildId}] Failed to parse yt-dlp related JSON: ${parseErr.message}`);
          return resolve([]);
        }

        const entries = Array.isArray(payload?.entries) ? payload.entries : [];
        const seen = new Set([videoId]);
        const tracks = [];

        for (const entry of entries) {
          const id = typeof entry?.id === 'string' ? entry.id.trim() : '';
          if (!id || seen.has(id)) continue;
          seen.add(id);

          const title = (typeof entry?.title === 'string' && entry.title.trim())
            ? entry.title.trim()
            : 'Без названия';

          let duration = (typeof entry?.duration_string === 'string' && entry.duration_string.trim())
            ? entry.duration_string.trim()
            : '?';
          if (duration === '?' && Number.isFinite(entry?.duration) && entry.duration > 0) {
            duration = this._formatDuration(Math.floor(entry.duration));
          }

          tracks.push({
            url: `https://www.youtube.com/watch?v=${id}`,
            videoId: id,
            title,
            duration,
            thumbnail: `https://img.youtube.com/vi/${id}/hqdefault.jpg`,
            requestedBy: '🤖 Бесконечное',
            isAutoplay: true,
          });
        }

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

  /**
   * Returns index where manual tracks should be inserted:
   * after existing manual items, before first autoplay item.
   * @returns {number}
   */
  _getManualInsertIndex() {
    const firstAutoplayIndex = this.tracks.findIndex((track) => track?.isAutoplay);
    return firstAutoplayIndex === -1 ? this.tracks.length : firstAutoplayIndex;
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
   * Formats seconds into m:ss or h:mm:ss.
   * @param {number} totalSeconds
   * @returns {string}
   */
  _formatDuration(totalSeconds) {
    if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return '?';

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
  }

  /**
   * Tries to get a valid video id from track metadata.
   * @param {Track|null} track
   * @returns {string}
   */
  _resolveVideoId(track) {
    const directId = typeof track?.videoId === 'string' ? track.videoId.trim() : '';
    if (directId) return directId;

    if (!track?.url) return '';
    try {
      return new URL(track.url).searchParams.get('v') || '';
    } catch (_) {
      return '';
    }
  }

  _cancelScheduledFade() {
    if (this._scheduledFadeTimer) {
      clearTimeout(this._scheduledFadeTimer);
      this._scheduledFadeTimer = null;
    }
  }

  /**
   * Если плеер ушёл в Idle немного раньше длительности трека,
   * ждём остаток, чтобы не перескакивать на следующий трек преждевременно.
   * @param {Track|null} track
   */
  _getEarlyIdleWaitMs(track) {
    if (!track) return 0;
    const totalSec = this._parseDuration(track.duration);
    if (!totalSec) return 0;

    const elapsedSec = this.getElapsedSeconds();
    const remainingSec = totalSec - elapsedSec;

    if (remainingSec <= EARLY_IDLE_TOLERANCE_SEC) return 0;
    if (remainingSec > EARLY_IDLE_MAX_WAIT_SEC) return 0;

    return Math.ceil(remainingSec * 1000);
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
    if (this._playerUpdating) {
      this._playerUpdateQueued = true;
      return;
    }
    this._playerUpdating = true;
    this._playerUpdateQueued = false;

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
      if (this._playerUpdateQueued) {
        this._playerUpdateQueued = false;
        setImmediate(() => this._updatePlayerMessage());
      }
    }
  }

  _startPlayerInterval() {
    this._stopPlayerInterval();
    this._playerInterval = setInterval(() => {
      this._updatePlayerMessage();
    }, PLAYER_UPDATE_INTERVAL_MS);
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
    this._isJoining = false;
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

  /**
   * Bind connection lifecycle handlers once the connection is ready.
   * @param {import('@discordjs/voice').VoiceConnection} connection
   */
  _bindConnectionHandlers(connection) {
    if (!connection || connection.__muzHandlersBound) return;
    connection.__muzHandlersBound = true;

    connection.on(VoiceConnectionStatus.Disconnected, async () => {
      if (this.connection !== connection) return;
      debugLog(`[Queue:${this.guildId}] connection disconnected`);

      try {
        await Promise.race([
          entersState(connection, VoiceConnectionStatus.Signalling, VOICE_RECONNECT_TIMEOUT_MS),
          entersState(connection, VoiceConnectionStatus.Connecting, VOICE_RECONNECT_TIMEOUT_MS),
        ]);
        debugLog(`[Queue:${this.guildId}] connection is reconnecting`);
      } catch (err) {
        if (this.connection !== connection) return;
        debugLog(`[Queue:${this.guildId}] reconnect window missed: ${err?.message || err}`);

        try {
          connection.rejoin();
          await entersState(connection, VoiceConnectionStatus.Ready, VOICE_READY_TIMEOUT_MS);
          debugLog(`[Queue:${this.guildId}] connection rejoin success`);
          return;
        } catch (rejoinErr) {
          debugLog(`[Queue:${this.guildId}] connection rejoin failed: ${rejoinErr?.message || rejoinErr}`);
        }

        if (this.connection === connection) {
          this._cleanup();
        }
      }
    });
  }

  async join(voiceChannel) {
    if (this._joinPromise) {
      debugLog(`[Queue:${this.guildId}] join already in progress, waiting`);
      return this._joinPromise;
    }

    this.voiceChannel = voiceChannel;
    debugLog(
      `[Queue:${this.guildId}] join requested channel=${voiceChannel?.id} name="${voiceChannel?.name || ''}"`,
    );

    this._joinPromise = (async () => {
      this._isJoining = true;

      try {
        if (this.connection) {
          try { this.connection.destroy(); } catch (_) {}
          this.connection = null;
        }

        const orphaned = getVoiceConnection(this.guildId);
        if (orphaned) {
          try { orphaned.destroy(); } catch (_) {}
        }

        for (let attempt = 1; attempt <= VOICE_JOIN_ATTEMPTS; attempt++) {
          debugLog(`[Queue:${this.guildId}] join attempt ${attempt}/${VOICE_JOIN_ATTEMPTS}`);
          const activeConnection = joinVoiceChannel({
            channelId: voiceChannel.id,
            guildId: voiceChannel.guild.id,
            adapterCreator: voiceChannel.guild.voiceAdapterCreator,
            selfDeaf: true,
          });

          try {
            await entersState(activeConnection, VoiceConnectionStatus.Ready, VOICE_READY_TIMEOUT_MS);
            debugLog(`[Queue:${this.guildId}] connection ready`);
            this.connection = activeConnection;
            this._bindConnectionHandlers(activeConnection);
            this.connection.subscribe(this.player);
            debugLog(`[Queue:${this.guildId}] connection subscribed to player`);
            return;
          } catch (err) {
            debugLog(`[Queue:${this.guildId}] join attempt failed: ${err.message}`);
            console.error(
              `[Queue:${this.guildId}] Voice join attempt ${attempt}/${VOICE_JOIN_ATTEMPTS} failed: ${err.message}`,
            );
            try { activeConnection.destroy(); } catch (_) {}
            if (attempt < VOICE_JOIN_ATTEMPTS) {
              await new Promise((resolve) => setTimeout(resolve, VOICE_JOIN_RETRY_DELAY_MS));
            }
          }
        }

        debugLog(`[Queue:${this.guildId}] join failed after retries`);
        throw new Error('Failed to connect to the voice channel. Check permissions, channel region, and host UDP networking.');
      } finally {
        this._isJoining = false;
      }
    })();

    try {
      await this._joinPromise;
    } finally {
      this._joinPromise = null;
    }
  }

  async addTrack(track) {
    const wasIdle = !this.currentTrack && this.tracks.length === 0;
    if (track?.isAutoplay) {
      this.tracks.push(track);
    } else {
      const insertIndex = this._getManualInsertIndex();
      this.tracks.splice(insertIndex, 0, track);
    }
    if (wasIdle && !this._advancing) {
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    }
  }

  async addTracks(tracks) {
    const wasIdle = !this.currentTrack && this.tracks.length === 0;
    const manualTracks = tracks.filter((track) => !track?.isAutoplay);
    const autoplayTracks = tracks.filter((track) => track?.isAutoplay);

    if (manualTracks.length > 0) {
      const insertIndex = this._getManualInsertIndex();
      this.tracks.splice(insertIndex, 0, ...manualTracks);
    }
    if (autoplayTracks.length > 0) {
      this.tracks.push(...autoplayTracks);
    }

    if (wasIdle && !this._advancing) {
      this._advancing = true;
      await this._advanceQueue().catch(console.error);
      this._advancing = false;
    }
  }

  skip() {
    if (!this.currentTrack) return false;
    this.loopCurrent = false;

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
    this.loopCurrent = false;
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

  setAutoplay(enabled) {
    this.autoplay = Boolean(enabled);
    this._updatePlayerMessage();
    return this.autoplay;
  }

  toggleAutoplay() {
    return this.setAutoplay(!this.autoplay);
  }

  toggleLoop() {
    this.loopCurrent = !this.loopCurrent;
    this._updatePlayerMessage();
    return this.loopCurrent;
  }

  clearQueue() {
    this.tracks = [];
  }

  get status() {
    return this.player.state.status;
  }
}

module.exports = GuildQueue;
