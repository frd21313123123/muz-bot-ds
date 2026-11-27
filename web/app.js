'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  data: null,
  authenticated: false,
  guildId: null,
  channelId: null,
  refreshing: false,
  modalOpen: false,
  baseElapsed: 0,
  baseAt: 0,
  basePaused: true,
  baseLive: false,
  durationSeconds: null,
  toastTimer: null,
  lastCode: null,
};

const el = {
  loginView: $('login-view'), appView: $('app-view'), loginForm: $('login-form'),
  password: $('password'), loginError: $('login-error'),
  botAvatar: $('bot-avatar'), botAvatarFallback: $('bot-avatar-fallback'), botName: $('bot-name'),
  botState: $('bot-state'), skillLabel: $('skill-label'), logout: $('logout'),
  guild: $('guild-select'), channel: $('channel-select'), targetHint: $('target-hint'),
  playerStatus: $('player-status'), cover: $('cover'), coverPlaceholder: $('cover-placeholder'),
  trackTitle: $('track-title'), trackMeta: $('track-meta'), progressFill: $('progress-fill'),
  elapsed: $('elapsed'), duration: $('duration'),
  pause: $('btn-pause'), skip: $('btn-skip'), stop: $('btn-stop'),
  autoplay: $('btn-autoplay'), loop: $('btn-loop'), clear: $('btn-clear'),
  volume: $('volume'), volumeValue: $('volume-value'),
  playForm: $('play-form'), query: $('query'), source: $('source'),
  radio: $('radio'), radioButton: $('btn-radio'),
  speakersEmpty: $('speakers-empty'), speakerList: $('speaker-list'), addSpeaker: $('btn-add-speaker'),
  queueCount: $('queue-count'), queueList: $('queue-list'), queueEmpty: $('queue-empty'),
  logList: $('log-list'),
  modal: $('modal'), modalClose: $('modal-close'), modalForm: $('modal-form'), modalCode: $('modal-code'),
  speakerName: $('speaker-name'), modalGuild: $('modal-guild'), modalChannel: $('modal-channel'),
  createCode: $('btn-create-code'), pairCode: $('pair-code'), pairEndpoint: $('pair-endpoint'),
  skillName: $('skill-name'), copyCode: $('btn-copy-code'), newCode: $('btn-new-code'),
  toast: $('toast'),
};

function isHttp(value) {
  return typeof value === 'string' && /^https?:\/\//iu.test(value);
}

function isHttps(value) {
  return typeof value === 'string' && /^https:\/\//iu.test(value);
}

function parseDuration(value) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!/^\d+(?::\d{1,2}){1,2}$/u.test(text)) return null;
  const parts = text.split(':').map(Number);
  if (parts.slice(1).some((part) => part >= 60)) return null;
  const seconds = parts.reduce((total, part) => total * 60 + part, 0);
  return Number.isSafeInteger(seconds) && seconds > 0 ? seconds : null;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  const rest = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${rest}` : `${minutes}:${rest}`;
}

function timeOf(timestamp) {
  try {
    return new Date(timestamp).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return '—'; }
}

function relative(timestamp) {
  if (!timestamp) return 'ещё не выходила на связь';
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return 'только что';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} мин назад`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} ч назад`;
  return timeOf(timestamp);
}

function toast(message, error = false) {
  el.toast.textContent = message;
  el.toast.className = error ? 'toast error' : 'toast ok';
  el.toast.hidden = false;
  if (state.toastTimer) clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { el.toast.hidden = true; }, 3800);
}

function saveSelection() {
  try {
    if (state.guildId) localStorage.setItem('muz.guild', state.guildId);
    if (state.channelId) localStorage.setItem('muz.channel', state.channelId);
  } catch { /* private mode */ }
}

function loadSelection(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

async function api(path, options = {}) {
  const method = options.method ?? 'GET';
  const headers = { Accept: 'application/json' };
  if (method !== 'GET' && method !== 'HEAD') {
    headers['Content-Type'] = 'application/json';
    headers['X-Requested-With'] = 'muz-web';
  }
  const response = await fetch(path, {
    method, headers, credentials: 'same-origin', cache: 'no-store',
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  let data = {};
  try { data = await response.json(); } catch { /* empty */ }
  if (response.status === 401) {
    showLogin();
    throw new Error(data.error ?? 'Требуется вход.');
  }
  if (!response.ok) throw new Error(data.error ?? `Ошибка ${response.status}`);
  return data;
}

function showLogin() {
  state.authenticated = false;
  state.data = null;
  el.appView.hidden = true;
  el.loginView.hidden = false;
  el.loginError.textContent = '';
  closeModal();
  setTimeout(() => el.password.focus(), 0);
}

function showApp() {
  state.authenticated = true;
  el.loginView.hidden = true;
  el.appView.hidden = false;
}

function selectedGuild() {
  return state.data?.guilds.find((guild) => guild.id === state.guildId) ?? null;
}

function selectedQueue() {
  return selectedGuild()?.queue ?? null;
}

function ensureSelection() {
  const guilds = state.data?.guilds ?? [];
  if (!guilds.length) {
    state.guildId = null;
    state.channelId = null;
    return;
  }
  const last = state.data?.lastTarget;
  let guild = guilds.find((item) => item.id === state.guildId)
    ?? guilds.find((item) => item.id === loadSelection('muz.guild'))
    ?? guilds.find((item) => item.id === last?.guildId)
    ?? guilds.find((item) => item.queue)
    ?? guilds[0];
  state.guildId = guild.id;
  const currentChannel = guild.channels.find((channel) => channel.id === state.channelId);
  if (!currentChannel?.canJoin) {
    const saved = loadSelection('muz.channel');
    const queueChannel = guild.queue?.channelId;
    state.channelId = guild.channels.find((channel) => channel.id === queueChannel && channel.canJoin)?.id
      ?? guild.channels.find((channel) => channel.id === saved && channel.canJoin)?.id
      ?? guild.channels.find((channel) => channel.id === last?.channelId && channel.canJoin)?.id
      ?? guild.channels.find((channel) => channel.canJoin)?.id
      ?? guild.channels[0]?.id
      ?? null;
  }
  saveSelection();
}

function fillSelect(select, items, value, signature) {
  const next = signature ?? JSON.stringify(items.map((item) => [item.value, item.label, Boolean(item.disabled)]));
  if (select.dataset.signature !== next) {
    select.replaceChildren(...items.map((item) => {
      const option = document.createElement('option');
      option.value = item.value;
      option.textContent = item.label;
      if (item.disabled) option.disabled = true;
      return option;
    }));
    select.dataset.signature = next;
  }
  if (items.some((item) => item.value === value)) select.value = value;
  else if (items.length) select.value = items[0].value;
}

function render() {
  if (!state.data) return;
  const data = state.data;
  const guild = selectedGuild();
  const queue = selectedQueue();

  el.botName.textContent = data.bot.tag || 'Бот';
  el.botState.textContent = data.bot.ready ? 'онлайн' : 'запускается';
  el.botState.className = `pill ${data.bot.ready ? 'playing' : ''}`;
  if (isHttps(data.bot.avatar)) {
    el.botAvatar.src = data.bot.avatar;
    el.botAvatar.hidden = false;
    el.botAvatarFallback.hidden = true;
  } else {
    el.botAvatar.hidden = true;
    el.botAvatarFallback.hidden = false;
  }
  el.skillLabel.textContent = data.skill.configured
    ? `навык «${data.skill.name}»`
    : 'навык Алисы не настроен';

  fillSelect(el.guild, (data.guilds ?? []).map((item) => ({
    value: item.id, label: item.name, disabled: false,
  })), state.guildId ?? '');
  fillSelect(el.channel, (guild?.channels ?? []).map((item) => ({
    value: item.id, label: item.canJoin ? item.name : `${item.name} (нет прав)`, disabled: !item.canJoin,
  })), state.channelId ?? '');
  fillSelect(el.radio, [{ value: '', label: 'Радио…', disabled: false },
    ...(data.radioStations ?? []).map((item) => ({ value: item.id, label: item.name, disabled: false }))],
  el.radio.value ?? '');

  if (queue?.channelId && queue.channelId !== state.channelId) {
    el.targetHint.textContent = `Бот сейчас в канале «${queue.channelName ?? 'другом'}». Следующее включение переведёт его в выбранный канал.`;
    el.targetHint.hidden = false;
  } else {
    el.targetHint.hidden = true;
  }

  renderNow(queue);
  renderControls(queue);
  renderQueue(queue);
  renderSpeakers();
  renderLog();

  if (state.modalOpen) {
    fillModalGuilds();
    fillModalChannels();
  }
}

function renderNow(queue) {
  const current = queue?.current ?? null;
  if (!current) {
    el.cover.hidden = true;
    el.cover.removeAttribute('src');
    el.coverPlaceholder.hidden = false;
    el.trackTitle.textContent = 'Ничего не играет';
    el.trackMeta.textContent = queue?.channelName ? `Канал «${queue.channelName}»` : 'Выберите канал и включите трек';
    el.playerStatus.textContent = 'тишина';
    el.playerStatus.className = 'pill';
    el.elapsed.textContent = '0:00';
    el.duration.textContent = '—';
    el.progressFill.style.width = '0%';
    state.baseElapsed = 0;
    state.baseAt = Date.now();
    state.basePaused = true;
    state.baseLive = false;
    state.durationSeconds = null;
    return;
  }

  if (isHttps(current.thumbnail)) {
    if (el.cover.getAttribute('src') !== current.thumbnail) el.cover.src = current.thumbnail;
    el.cover.hidden = false;
    el.coverPlaceholder.hidden = true;
  } else {
    el.cover.hidden = true;
    el.cover.removeAttribute('src');
    el.coverPlaceholder.hidden = false;
  }
  el.trackTitle.textContent = current.title;
  const parts = [];
  if (current.requestedBy) parts.push(`заказал(а) ${current.requestedBy}`);
  if (queue.paused) parts.push('пауза');
  else if (queue.radio) parts.push('прямой эфир');
  if (queue.speed !== 1) parts.push(`${queue.speed}×`);
  el.trackMeta.textContent = parts.join(' · ') || 'играет';

  el.playerStatus.textContent = queue.paused ? 'пауза' : queue.radio ? 'эфир' : 'играет';
  el.playerStatus.className = `pill ${queue.paused ? 'paused' : queue.radio ? 'live' : 'playing'}`;
  state.durationSeconds = parseDuration(current.duration);
  el.duration.textContent = current.isLive ? 'LIVE' : current.duration || '—';
  state.baseElapsed = queue.elapsed;
  state.baseAt = Date.now();
  state.basePaused = queue.paused;
  state.baseLive = Boolean(current.isLive);
  tick();
}

function tick() {
  if (!state.data) return;
  const queue = selectedQueue();
  if (!queue?.current) return;
  const elapsed = state.baseElapsed + (state.basePaused ? 0 : Math.max(0, (Date.now() - state.baseAt) / 1000));
  const shown = state.durationSeconds ? Math.min(elapsed, state.durationSeconds) : elapsed;
  el.elapsed.textContent = formatDuration(shown);
  el.progressFill.style.width = state.durationSeconds
    ? `${Math.min(100, (shown / state.durationSeconds) * 100)}%`
    : state.baseLive ? '100%' : '0%';
}

function renderControls(queue) {
  const current = queue?.current ?? null;
  el.pause.disabled = !current;
  el.pause.textContent = queue?.paused ? '▶' : '⏸';
  el.pause.title = queue?.paused ? 'Продолжить' : 'Пауза';
  el.pause.setAttribute('aria-label', el.pause.title);
  el.skip.disabled = !current;
  el.stop.disabled = !queue;
  el.autoplay.disabled = !queue || queue.radio;
  el.autoplay.classList.toggle('active', Boolean(queue?.autoplay));
  el.loop.disabled = !queue || queue.radio;
  el.loop.classList.toggle('active', Boolean(queue?.loop));
  el.clear.disabled = !queue || !queue.upcomingCount;
  el.volume.disabled = !queue;
  el.radioButton.disabled = !state.guildId || !state.channelId;
  if (queue && document.activeElement !== el.volume) {
    el.volume.value = String(queue.volume);
    el.volumeValue.textContent = `${queue.volume}%`;
  }
}

function renderQueue(queue) {
  const upcoming = queue?.upcoming ?? [];
  el.queueCount.textContent = String(queue?.upcomingCount ?? 0);
  el.queueEmpty.hidden = upcoming.length > 0;
  const fragment = document.createDocumentFragment();
  upcoming.forEach((track, index) => {
    const item = document.createElement('li');
    item.className = 'queue-item';
    const number = document.createElement('span');
    number.className = 'queue-index';
    number.textContent = String(index + 1);
    const title = document.createElement(isHttp(track.url) ? 'a' : 'span');
    title.className = 'queue-title';
    title.textContent = track.title;
    if (title instanceof HTMLAnchorElement) {
      title.href = track.url;
      title.target = '_blank';
      title.rel = 'noreferrer noopener';
    }
    const meta = document.createElement('span');
    meta.className = 'queue-meta';
    meta.textContent = [track.duration, track.requestedBy].filter(Boolean).join(' · ');
    item.append(number, title, meta);
    fragment.append(item);
  });
  el.queueList.replaceChildren(fragment);
}

function renderSpeakers() {
  const speakers = state.data?.speakers ?? [];
  el.speakersEmpty.hidden = speakers.length > 0;
  const guilds = new Map((state.data?.guilds ?? []).map((guild) => [guild.id, guild]));
  const fragment = document.createDocumentFragment();
  for (const speaker of speakers) {
    const guild = guilds.get(speaker.guildId);
    const channel = guild?.channels.find((item) => item.id === speaker.channelId);
    const item = document.createElement('li');
    item.className = 'speaker-item';

    const main = document.createElement('div');
    main.className = 'speaker-main';
    const nameLine = document.createElement('div');
    nameLine.className = 'speaker-name';
    const dot = document.createElement('span');
    dot.className = `dot ${speaker.online ? 'online' : ''}`;
    const name = document.createElement('span');
    name.textContent = speaker.name;
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = speaker.kind === 'alice' ? 'Алиса' : 'Мост';
    nameLine.append(dot, name, badge);
    const sub = document.createElement('div');
    sub.className = 'speaker-sub';
    const where = guild ? `${guild.name} · ${channel?.name ?? 'канал удалён'}` : 'сервер не найден';
    sub.textContent = `${where} · ${speaker.online ? 'в сети' : relative(speaker.lastSeenAt)}`;
    main.append(nameLine, sub);

    const actions = document.createElement('div');
    actions.className = 'speaker-actions';
    const rename = document.createElement('button');
    rename.type = 'button';
    rename.className = 'icon';
    rename.title = 'Переименовать';
    rename.textContent = '✎';
    rename.addEventListener('click', () => void renameSpeaker(speaker));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'icon';
    remove.title = 'Отключить';
    remove.textContent = '🗑';
    remove.addEventListener('click', () => void removeSpeaker(speaker));
    actions.append(rename, remove);

    item.append(main, actions);
    fragment.append(item);
  }
  el.speakerList.replaceChildren(fragment);
}

function renderLog() {
  const entries = state.data?.log ?? [];
  const fragment = document.createDocumentFragment();
  for (const entry of entries.slice(0, 25)) {
    const item = document.createElement('li');
    item.className = `log-item ${entry.ok ? '' : 'fail'}`;
    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = timeOf(entry.at);
    const message = document.createElement('span');
    message.className = 'text';
    const kind = document.createElement('span');
    kind.className = 'kind';
    kind.textContent = entry.kind;
    message.append(kind, document.createTextNode(entry.text));
    item.append(time, message);
    fragment.append(item);
  }
  el.logList.replaceChildren(fragment);
}

async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    const data = await api('/api/state');
    state.data = data;
    ensureSelection();
    render();
  } catch (error) {
    if (state.authenticated) toast(error.message, true);
  } finally {
    state.refreshing = false;
  }
}

function requireTarget() {
  if (!state.guildId || !state.channelId) throw new Error('Выберите сервер и голосовой канал.');
  return { guildId: state.guildId, channelId: state.channelId };
}

async function playerAction(action, extra = {}, options = {}) {
  try {
    const result = await api('/api/player', { method: 'POST', body: { ...requireTarget(), action, ...extra } });
    if (!options.silent) await refresh();
    return result;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

async function submitPlay(event) {
  event.preventDefault();
  const query = el.query.value.trim();
  if (!query) { el.query.focus(); return; }
  const button = el.playForm.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = 'Ищу…';
  const result = await playerAction('play', { query, source: el.source.value });
  button.disabled = false;
  button.textContent = 'Включить';
  if (result) {
    el.query.value = '';
    toast(result.started ? `Включаю: ${result.track.title}` : `В очереди: ${result.track.title}`);
  }
}

async function submitRadio() {
  if (!el.radio.value) { toast('Выберите радиостанцию.', true); return; }
  el.radioButton.disabled = true;
  const result = await playerAction('radio', { stationId: el.radio.value });
  el.radioButton.disabled = false;
  if (result) toast(`Радио: ${result.station}`);
}

async function renameSpeaker(speaker) {
  const name = window.prompt('Новое название колонки', speaker.name);
  if (name === null) return;
  try {
    await api(`/api/speakers/${speaker.id}`, { method: 'PATCH', body: { name } });
    toast('Название сохранено.');
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function removeSpeaker(speaker) {
  if (!window.confirm(`Отключить колонку «${speaker.name}»?`)) return;
  try {
    await api(`/api/speakers/${speaker.id}`, { method: 'DELETE' });
    toast('Колонка отключена.');
    await refresh();
  } catch (error) { toast(error.message, true); }
}

function openModal() {
  state.modalOpen = true;
  el.modal.hidden = false;
  el.modalForm.hidden = false;
  el.modalCode.hidden = true;
  el.speakerName.value = 'Колонка';
  el.pairCode.textContent = '000000';
  el.skillName.textContent = state.data?.skill.name ?? 'Муза';
  try {
    el.pairEndpoint.textContent = `${location.origin}/api/alice/pair`;
  } catch { el.pairEndpoint.textContent = '/api/alice/pair'; }
  fillModalGuilds();
  el.modalGuild.value = state.guildId ?? el.modalGuild.value;
  fillModalChannels();
  el.modalChannel.value = state.channelId ?? el.modalChannel.value;
  setTimeout(() => el.speakerName.select(), 0);
}

function closeModal() {
  state.modalOpen = false;
  el.modal.hidden = true;
  state.lastCode = null;
}

function fillModalGuilds() {
  const guilds = state.data?.guilds ?? [];
  fillSelect(el.modalGuild, guilds.map((guild) => ({ value: guild.id, label: guild.name, disabled: false })),
    el.modalGuild.value || state.guildId || '');
}

function fillModalChannels() {
  const guild = state.data?.guilds.find((item) => item.id === el.modalGuild.value);
  fillSelect(el.modalChannel, (guild?.channels ?? []).map((channel) => ({
    value: channel.id, label: channel.canJoin ? channel.name : `${channel.name} (нет прав)`, disabled: !channel.canJoin,
  })), el.modalChannel.value || (guild?.id === state.guildId ? state.channelId : '') || '');
}

async function createPairingCode() {
  const guildId = el.modalGuild.value;
  const channelId = el.modalChannel.value;
  if (!guildId || !channelId) { toast('Выберите сервер и канал.', true); return; }
  el.createCode.disabled = true;
  try {
    const result = await api('/api/speakers/code', {
      method: 'POST',
      body: { guildId, channelId, name: el.speakerName.value.trim() || 'Колонка' },
    });
    state.lastCode = result.code;
    el.pairCode.textContent = result.code;
    el.skillName.textContent = result.skillName ?? state.data?.skill.name ?? 'Муза';
    el.modalForm.hidden = true;
    el.modalCode.hidden = false;
  } catch (error) {
    toast(error.message, true);
  } finally {
    el.createCode.disabled = false;
  }
}

async function copyCode() {
  const code = state.lastCode;
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code);
    toast('Код скопирован.');
  } catch {
    toast(`Код: ${code}`);
  }
}

async function logout() {
  try { await api('/api/logout', { method: 'POST' }); } catch { /* ignore */ }
  state.data = null;
  showLogin();
}

function bind() {
  el.loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    el.loginError.textContent = '';
    try {
      await api('/api/login', { method: 'POST', body: { password: el.password.value } });
      el.password.value = '';
      showApp();
      await refresh();
    } catch (error) {
      el.loginError.textContent = error.message;
    }
  });
  el.logout.addEventListener('click', () => void logout());
  el.guild.addEventListener('change', () => {
    state.guildId = el.guild.value;
    const guild = selectedGuild();
    state.channelId = guild?.channels.find((channel) => channel.id === guild.queue?.channelId && channel.canJoin)?.id
      ?? guild?.channels.find((channel) => channel.canJoin)?.id
      ?? guild?.channels[0]?.id
      ?? null;
    saveSelection();
    render();
  });
  el.channel.addEventListener('change', () => {
    state.channelId = el.channel.value;
    saveSelection();
    render();
  });
  el.playForm.addEventListener('submit', (event) => void submitPlay(event));
  el.radioButton.addEventListener('click', () => void submitRadio());
  el.pause.addEventListener('click', () => {
    const queue = selectedQueue();
    if (queue) void playerAction(queue.paused ? 'resume' : 'pause');
  });
  el.skip.addEventListener('click', () => void playerAction('skip'));
  el.stop.addEventListener('click', () => {
    if (window.confirm('Остановить музыку и выйти из канала?')) void playerAction('stop');
  });
  el.clear.addEventListener('click', () => void playerAction('clear'));
  el.autoplay.addEventListener('click', () => {
    const queue = selectedQueue();
    if (queue) void playerAction('autoplay', { value: !queue.autoplay });
  });
  el.loop.addEventListener('click', () => {
    const queue = selectedQueue();
    if (queue) void playerAction('loop', { value: !queue.loop });
  });
  el.volume.addEventListener('input', () => {
    el.volumeValue.textContent = `${el.volume.value}%`;
  });
  el.volume.addEventListener('change', () => void playerAction('volume', { volume: Number(el.volume.value) }, { silent: true }));
  el.addSpeaker.addEventListener('click', openModal);
  el.modalClose.addEventListener('click', closeModal);
  el.modal.addEventListener('click', (event) => { if (event.target === el.modal) closeModal(); });
  el.modalGuild.addEventListener('change', fillModalChannels);
  el.createCode.addEventListener('click', () => void createPairingCode());
  el.copyCode.addEventListener('click', () => void copyCode());
  el.newCode.addEventListener('click', () => {
    el.modalForm.hidden = false;
    el.modalCode.hidden = true;
    void createPairingCode();
  });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeModal(); });
  el.cover.addEventListener('error', () => {
    el.cover.hidden = true;
    el.coverPlaceholder.hidden = false;
  });
}

async function init() {
  bind();
  setInterval(() => { if (!document.hidden && state.authenticated) void refresh(); }, 2500);
  setInterval(tick, 1000);
  try {
    const session = await api('/api/session');
    if (!session.authenticated) { showLogin(); return; }
    showApp();
    await refresh();
  } catch {
    showLogin();
  }
}

void init();
