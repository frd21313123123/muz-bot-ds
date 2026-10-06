"""
Live microphone wake-word listener (Keyword Spotter).
Listens for the wake-word (e.g. «бот») in real-time from the microphone,
just like smart assistants (Alice, Siri, Alexa), and emits a signal/event.
"""

import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import numpy as np

# Ensure wake-word modules are available in python path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / '.runtime/wake-validation'),
    str(ROOT / 'scripts/wakeword'),
    str(Path(__file__).parent)
]

try:
    from wake_audio import WakeDetector, N_SAMPLES, SAMPLE_RATE
except ImportError as err:
    print(f"Error importing wake_audio: {err}", file=sys.stderr)
    sys.exit(1)


def find_default_model():
    """Find the best available trained wake model."""
    candidates = [
        ROOT / 'models/wake-model',
        ROOT / 'word_training/runs/bot-v6-hardneg/wake-model-opt',
        ROOT / 'word_training/runs/bot-v6-hardneg/wake-model',
        ROOT / 'word_training/runs/bot-v5-acc90/wake-model',
        ROOT / 'word_training/runs/bot-v5-acc90-output/wake-model',
        ROOT / 'word_training/runs/bot-best-v3/model',
        ROOT / 'word_training/runs/bot-best-v3-output/wake-model',
        ROOT / 'word_training/runs/bot-binary-v3-output/wake-model',
        ROOT / 'word_training/runs/bot-v2/wake-model',
        ROOT / 'word_training/runs/bot-v2-output/wake-model',
    ]
    for c in candidates:
        if (c / 'wake_model.onnx').exists() and (c / 'wake_config.json').exists():
            return c
    return None


def play_chime():
    """Play an audio chime signal on detection (Windows beep in background thread)."""
    try:
        import winsound
        # Two-tone chime like a smart speaker activation
        winsound.Beep(880, 100)   # A5
        winsound.Beep(1318, 150)  # E6
    except Exception:
        pass


def on_wake_word_detected(word: str, prob: float, timestamp: float, chime: bool = True):
    """
    Callback executed when wake-word is recognized.
    Here you can trigger any downstream actions:
    - Start recording user command
    - Send event to Discord bot
    - Speak response ('Слушаю!', 'Да?')
    """
    ts_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
    print("\n" + "=" * 60)
    print(f"🔔 [{ts_str}] СИГНАЛ: Вейк-ворд «{word.upper()}» ОБНАРУЖЕН!")
    print(f"📊 Уверенность модели: {prob * 100:.2f}%")
    print("🎙️ Голосовой помощник активен — слушаю команду...")
    print("=" * 60 + "\n", flush=True)

    if chime:
        threading.Thread(target=play_chime, daemon=True).start()


def run_mic_listener(model_dir: Path, device: int | None = None, threshold_override: float | None = None,
                     hop_ms: int = 100, cooldown_sec: float = 1.5, enable_chime: bool = True):
    try:
        import sounddevice as sd
    except ImportError:
        print("Ошибка: библиотека sounddevice не установлена. Выполните: pip install sounddevice", file=sys.stderr)
        return

    detector = WakeDetector(model_dir)
    config = detector.config
    wake_word = config.get('wake_word_ru', 'бот')
    threshold = threshold_override if threshold_override is not None else float(config.get('threshold', 0.95))

    print("=" * 60)
    print(f"🚀 Запуск прослушивания вейк-ворда: «{wake_word}»")
    print(f"📁 Модель: {model_dir}")
    print(f"🎯 Порог срабатывания: {threshold:.4f}")
    print(f"⏱️ Окно детекции: {N_SAMPLES / SAMPLE_RATE:.1f} сек ({N_SAMPLES} отсчетов, {SAMPLE_RATE} Гц)")
    print(f"🔄 Шаг проверки: {hop_ms} мс")
    print("=" * 60)

    # Audio stream configuration
    hop_samples = int(SAMPLE_RATE * (hop_ms / 1000.0))
    buffer = np.zeros(N_SAMPLES, dtype=np.float32)
    audio_queue = queue.Queue()
    last_trigger_time = 0.0
    consecutive_hits = 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio warning: {status}]", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"🎤 Используемое аудиоустройство: [{device}] {dev_info.get('name')}")
    else:
        print("🎤 Используется микрофон по умолчанию (Default Input)")

    print("\n👂 Ожидание ключевого слова... Скажите «бот» в микрофон. (Нажмите Ctrl+C для выхода)\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=hop_samples, device=device, callback=audio_callback):
            while True:
                chunk = audio_queue.get()
                # Shift buffer left and append new chunk
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk

                now = time.time()
                # Skip detection during cooldown period
                if now - last_trigger_time < cooldown_sec:
                    consecutive_hits = 0
                    continue

                # Run neural network prediction
                prob = detector.score(buffer)

                # Visual indicator of probability in console (optional live bar)
                if prob > 0.3:
                    bars = int(prob * 20)
                    print(f"\r[Слушаю...] Вероятность «{wake_word}»: [{'#' * bars}{'.' * (20 - bars)}] {prob * 100:5.1f}%", end='', flush=True)

                if prob >= threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                # Require 2 consecutive frames (persistence) or strong peak (>= 0.9985)
                if (consecutive_hits >= 2 or prob >= 0.9985) and (now - last_trigger_time >= cooldown_sec):
                    last_trigger_time = now
                    consecutive_hits = 0
                    on_wake_word_detected(wake_word, prob, now, chime=enable_chime)

    except KeyboardInterrupt:
        print("\n\n🛑 Прослушивание остановлено пользователем.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=None,
                        help='Путь к папке модели с wake_model.onnx и wake_config.json')
    parser.add_argument('--device', type=int, default=None,
                        help='Индекс устройства ввода (микрофона). Запустите с --list-devices для списка.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Порог уверенности (0.0 .. 1.0). По умолчанию из конфигурации модели.')
    parser.add_argument('--hop', type=int, default=100,
                        help='Шаг проверки в миллисекундах (по умолчанию 100 мс)')
    parser.add_argument('--cooldown', type=float, default=1.5,
                        help='Период паузы после срабатывания в секундах (по умолчанию 1.5 с)')
    parser.add_argument('--no-chime', action='store_true',
                        help='Отключить звуковой сигнал (звук колокольчика/бип) при обнаружении')
    parser.add_argument('--list-devices', action='store_true',
                        help='Вывести список всех аудиоустройств и выйти')
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("\nДоступные аудиоустройства:")
        print(sd.query_devices())
        return

    model_dir = args.model or find_default_model()
    if not model_dir or not (model_dir / 'wake_model.onnx').exists():
        print("Ошибка: обученная модель не найдена!", file=sys.stderr)
        print("Укажите путь через --model или сначала запустите обучение.", file=sys.stderr)
        sys.exit(1)

    run_mic_listener(
        model_dir=model_dir,
        device=args.device,
        threshold_override=args.threshold,
        hop_ms=args.hop,
        cooldown_sec=args.cooldown,
        enable_chime=not args.no_chime
    )


if __name__ == '__main__':
    main()
