const tracks = [
  { title: 'Midnight City', artist: 'M83 · Hurry Up, We’re Dreaming', next: 'Tame Impala — The Less I Know the Better', duration: 243 },
  { title: 'The Less I Know the Better', artist: 'Tame Impala · Currents', next: 'Daft Punk — Something About Us', duration: 216 },
  { title: 'Something About Us', artist: 'Daft Punk · Discovery', next: 'M83 — Midnight City', duration: 232 },
];
let current = 0;
let elapsed = 92;
let playing = true;
let repeat = false;
let autoplay = true;
let toastTimer;
const waveform = document.querySelector('.waveform');
for (let i = 0; i < 70; i++) {
  const bar = document.createElement('i');
  bar.style.height = `${12 + 73 * Math.abs(Math.sin(i * 1.9) * Math.cos(i * .28))}px`;
  bar.style.animationDelay = `${-(i % 13) * .16}s`;
  waveform.append(bar);
}
const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
function render() {
  const track = tracks[current];
  document.querySelector('#track-title').textContent = track.title;
  document.querySelector('#track-artist').textContent = track.artist;
  document.querySelector('#next-title').textContent = track.next;
  document.querySelector('#elapsed').textContent = formatTime(elapsed);
  document.querySelector('#duration').textContent = formatTime(track.duration);
  document.querySelector('.progress span').style.width = `${elapsed / track.duration * 100}%`;
  document.querySelector('.progress').setAttribute('aria-valuenow', elapsed);
  document.querySelector('.progress').setAttribute('aria-valuemax', track.duration);
  const play = document.querySelector('#play');
  play.setAttribute('aria-label', playing ? 'Pause demo' : 'Play demo');
  play.setAttribute('aria-pressed', String(playing));
  play.firstElementChild.textContent = playing ? 'Ⅱ' : '▶';
  waveform.classList.toggle('paused', !playing);
}
function changeTrack(direction) { current = (current + direction + tracks.length) % tracks.length; elapsed = 0; render(); }
document.querySelector('#play').addEventListener('click', () => { playing = !playing; render(); });
document.querySelector('#next').addEventListener('click', () => changeTrack(1));
document.querySelector('#previous').addEventListener('click', () => changeTrack(-1));
document.querySelector('#repeat').addEventListener('click', event => {
  repeat = !repeat;
  if (repeat) autoplay = false;
  updateModes();
});
document.querySelector('#autoplay').addEventListener('click', () => {
  autoplay = !autoplay;
  if (autoplay) repeat = false;
  updateModes();
});
function updateModes() {
  for (const [id, enabled] of [['repeat', repeat], ['autoplay', autoplay]]) {
    const button = document.querySelector(`#${id}`);
    button.classList.toggle('active', enabled);
    button.setAttribute('aria-pressed', String(enabled));
    button.setAttribute('aria-label', `${enabled ? 'Disable' : 'Enable'} demo ${id}`);
  }
}
setInterval(() => {
  if (!playing || document.hidden) return;
  if (elapsed < tracks[current].duration) elapsed++;
  else if (repeat) elapsed = 0;
  else if (autoplay) changeTrack(1);
  else playing = false;
  render();
}, 1000);
function toast(message) {
  const element = document.querySelector('#toast');
  element.textContent = message;
  element.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove('visible'), 2800);
}
document.querySelectorAll('[data-command]').forEach(button => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.command);
      toast(`${button.dataset.command} copied. Paste it in Discord.`);
    } catch {
      toast(`Type ${button.dataset.command} in your Discord server.`);
    }
  });
});
render();
