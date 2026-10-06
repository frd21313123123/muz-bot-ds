"""Install the isolated voice runtime and download models. Never handles Discord audio."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent
PROFILE = os.environ.get('VOICE_PROFILE', 'light').strip().lower()
if PROFILE not in ('light', 'heavy'):
    raise ValueError('VOICE_PROFILE must be light or heavy')
STT_PROVIDER = os.environ.get('STT_PROVIDER', 'groq').strip().lower() or 'groq'
if STT_PROVIDER not in ('groq', 'local'):
    raise ValueError('STT_PROVIDER must be groq or local')
RUNTIME = ROOT / '.runtime' / ('voice-light' if PROFILE == 'light' else 'voice')
PYTHON = RUNTIME / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    if STT_PROVIDER == 'groq' and PROFILE == 'light':
        print('Groq light profile requires only GROQ_API_KEY in the Node environment', flush=True)
        return
    if sys.version_info < (3, 11):
        raise RuntimeError("Voice setup requires Python 3.11+")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    models_only = '--models-only' in sys.argv
    if models_only and not PYTHON.exists():
        raise RuntimeError('Run full setup:voice before updating only models')
    if not PYTHON.exists():
        print(f"Creating {RUNTIME.relative_to(ROOT)} virtual environment", flush=True)
        venv.EnvBuilder(with_pip=True).create(RUNTIME)
    environment = dict(os.environ, HF_HOME=str(RUNTIME / "cache"), USE_TF="0",
                       TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8")
    environment.pop("HF_HUB_OFFLINE", None)
    environment.pop("TRANSFORMERS_OFFLINE", None)
    if not models_only:
        device = 'cpu' if PROFILE == 'light' else os.environ.get("VOICE_DEVICE", "cpu").strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("VOICE_DEVICE must be cpu or cuda")
        print(f"Installing {device} dependencies", flush=True)
        if PROFILE == 'heavy':
            torch_version = "2.11.0" if device == "cuda" else "2.14.0"
            subprocess.run([str(PYTHON), "-m", "pip", "install", "torch==" + torch_version,
                        "--index-url", "https://download.pytorch.org/whl/" + ("cu128" if device == "cuda" else "cpu")], check=True, env=environment)
        requirements = 'voice-nli-requirements.txt' if STT_PROVIDER == 'groq' else ('voice-light-requirements.txt' if PROFILE == 'light' else 'voice-requirements.txt')
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r",
                    str(ROOT / 'scripts' / requirements)], check=True, env=environment)
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "voice_worker.py"), "--prepare"], check=True, env=environment)
    with open(RUNTIME / "requirements.lock.txt", "w", encoding="utf-8") as lock:
        subprocess.run([str(PYTHON), "-m", "pip", "freeze"], stdout=lock, check=True, env=environment)
    print(f'Voice profile {PROFILE} ready', flush=True)


if __name__ == "__main__":
    main()
