"""Generate fixed Russian replies with official Silero v5.5, offline at playback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / '.runtime' / 'tts'
MODEL = 'v5_5_ru'
SPEAKER = 'xenia'
URL = f'https://models.silero.ai/models/tts/ru/{MODEL}.pt'


def main():
    import torch
    torch.set_num_threads(4)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    checkpoint = RUNTIME / (MODEL + '.pt')
    if not checkpoint.exists():
        print('Downloading official Silero ' + MODEL, flush=True)
        temporary = checkpoint.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(checkpoint)
    model = torch.package.PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model')
    # Preparation runs once. Playback uses cached PCM and consumes no GPU memory.
    device = os.environ.get('VOICE_TTS_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('VOICE_TTS_DEVICE must be cpu or cuda')
    model.to(torch.device(device))
    phrases = json.loads((ROOT / 'src/voice/confirmations.json').read_text(encoding='utf-8'))
    stage = RUNTIME / 'silero-staging'
    stage.mkdir(exist_ok=True)
    timings = []
    for key, text in phrases.items():
        started = time.perf_counter()
        with torch.inference_mode():
            audio = model.apply_tts(text=text, speaker=SPEAKER, sample_rate=48000,
                                    put_accent=True, put_yo=True)
        audio = audio.detach().cpu().float()
        if not torch.isfinite(audio).all() or not audio.numel() or audio.abs().max() < .001:
            raise RuntimeError('Invalid synthesized audio: ' + key)
        wav = stage / (key + '.wav')
        with wave.open(str(wav), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            output.writeframes((audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes())
        pcm = stage / (key + '.pcm')
        subprocess.run([sys.argv[1], '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                        '-af', 'loudnorm=I=-18:TP=-6:LRA=7,aresample=48000',
                        '-ar', '48000', '-ac', '2', '-f', 's16le', str(pcm)], check=True)
        size = pcm.stat().st_size
        if not size or size % 4 or size > 48000 * 4 * 12:
            raise RuntimeError('Invalid confirmation: ' + key)
        seconds = time.perf_counter() - started
        timings.append({'phrase': key, 'audioSeconds': size / 192000, 'prepareSeconds': seconds})
        print(f'Silero {key}: {size / 192000:.2f}s audio, {seconds:.2f}s preparation', flush=True)
    # Keep the previous cache for comparison/rollback and publish only complete output.
    backup = RUNTIME / 'previous-cache'
    backup.mkdir(exist_ok=True)
    for name in ['ready.json', *[key + suffix for key in phrases for suffix in ('.pcm', '.wav')]]:
        old = RUNTIME / name
        if old.exists():
            shutil.copy2(old, backup / name)
    for key in phrases:
        for suffix in ('.pcm', '.wav'):
            (stage / (key + suffix)).replace(RUNTIME / (key + suffix))
    manifest = {'engine': 'silero', 'voice': MODEL + '/' + SPEAKER, 'modelUrl': URL,
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'device': device, 'sampleRate': 48000, 'channels': 2, 'phrases': phrases}
    ready = stage / 'ready.json'
    ready.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    ready.replace(RUNTIME / 'ready.json')
    (RUNTIME / 'silero-benchmark.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    print('Local Silero TTS ready', flush=True)


if __name__ == '__main__':
    main()
