"""Install the isolated voice runtime and download models. Never handles Discord audio."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime" / "voice"
PYTHON = RUNTIME / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("Voice setup requires Python 3.11+")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    models_only = '--models-only' in sys.argv
    if models_only and not PYTHON.exists():
        raise RuntimeError('Run full setup:voice before updating only models')
    if not PYTHON.exists():
        print("Creating .runtime/voice virtual environment", flush=True)
        venv.EnvBuilder(with_pip=True).create(RUNTIME)
    environment = dict(os.environ, HF_HOME=str(RUNTIME / "cache"), USE_TF="0",
                       TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8")
    environment.pop("HF_HUB_OFFLINE", None)
    environment.pop("TRANSFORMERS_OFFLINE", None)
    if not models_only:
        print("Installing CPU dependencies", flush=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "torch==2.14.0",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True, env=environment)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r",
                    str(ROOT / "scripts" / "voice-requirements.txt")], check=True, env=environment)
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / "voice_worker.py"), "--prepare"], check=True, env=environment)
    with open(RUNTIME / "requirements.lock.txt", "w", encoding="utf-8") as lock:
        subprocess.run([str(PYTHON), "-m", "pip", "freeze"], stdout=lock, check=True, env=environment)
    print("Local Whisper + Laya runtime ready", flush=True)


if __name__ == "__main__":
    main()
