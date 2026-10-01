"""Prepare local Piper confirmations. No Discord credentials or recordings used."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv
import wave

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / '.runtime' / 'tts'
PYTHON = RUNTIME / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
VOICE = 'ru_RU-irina-medium'
MODEL_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/irina/medium/'


def generate(ffmpeg):
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(str(RUNTIME / (VOICE + '.onnx')))
    phrases = json.loads((ROOT / 'src' / 'voice' / 'confirmations.json').read_text(encoding='utf-8'))
    for key, text in phrases.items():
        started = time.perf_counter()
        wav = RUNTIME / (key + '.wav')
        with wave.open(str(wav), 'wb') as output:
            voice.synthesize_wav(text, output, syn_config=SynthesisConfig(length_scale=1.05))
        temporary = RUNTIME / (key + '.pcm.tmp')
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                        '-af', 'loudnorm=I=-18:TP=-6:LRA=7,aresample=48000',
                        '-ar', '48000', '-ac', '2', '-f', 's16le', str(temporary)], check=True)
        size = temporary.stat().st_size
        if not size or size % 4 or size > 48000 * 4 * 12:
            raise RuntimeError('Invalid synthesized confirmation')
        temporary.replace(RUNTIME / (key + '.pcm'))
        print(f'Piper {key}: {size / 192000:.2f}s audio, {time.perf_counter() - started:.2f}s synthesis', flush=True)
    (RUNTIME / 'ready.json').write_text(json.dumps({'engine': 'piper', 'voice': VOICE,
        'sampleRate': 48000, 'channels': 2, 'phrases': phrases}, ensure_ascii=False), encoding='utf-8')


def main():
    ffmpeg = sys.argv[1]
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / 'ready.json').unlink(missing_ok=True)
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=True).create(RUNTIME)
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', 'piper-tts==1.8.0'], check=True)
    for name in [VOICE + '.onnx', VOICE + '.onnx.json', 'MODEL_CARD']:
        target = RUNTIME / name
        if not target.exists():
            print('Downloading Piper ' + name, flush=True)
            with urllib.request.urlopen(MODEL_BASE + name, timeout=120) as response:
                temporary = target.with_name(target.name + '.download')
                with temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(target)
    subprocess.run([str(PYTHON), str(Path(__file__).resolve()), ffmpeg, '--generate'], check=True)
    with (RUNTIME / 'requirements.lock.txt').open('w', encoding='utf-8') as output:
        subprocess.run([str(PYTHON), '-m', 'pip', 'freeze'], stdout=output, check=True)
    print('Local Piper TTS ready', flush=True)


if __name__ == '__main__':
    if '--generate' in sys.argv:
        generate(sys.argv[1])
    else:
        main()
