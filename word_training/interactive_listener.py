"""
Interactive Real-Time Wake-Word Monitor.
Visualizes microphone audio level, real-time "бот" probability,
and clearly indicates [ЕСТЬ] (detected) vs [НЕТ] (ordinary speech/no word).
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Colorama / ANSI styling
try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    RED = colorama.Fore.RED + colorama.Style.BRIGHT
    YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    CYAN = colorama.Fore.CYAN + colorama.Style.BRIGHT
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL
    DIM = colorama.Style.DIM
except ImportError:
    GREEN = RED = YELLOW = CYAN = WHITE = RESET = DIM = ""

# Audio import
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


def find_best_model():
    candidates = [
        ROOT / 'models/wake-model',
        ROOT / 'word_training/runs/bot-v6-hardneg/wake-model-opt',
        ROOT / 'word_training/runs/bot-v6-hardneg/wake-model',
        ROOT / 'word_training/runs/bot-v5-acc90/wake-model',
        ROOT / 'word_training/runs/bot-v5-acc90-output/wake-model',
        ROOT / 'word_training/runs/bot-best-v3/model',
        ROOT / 'word_training/runs/bot-v2/wake-model',
    ]
    for c in candidates:
        if (c / 'wake_model.onnx').exists() and (c / 'wake_config.json').exists():
            return c
    return None


def play_chime():
    """Smart speaker ding/chime on detection."""
    try:
        import winsound
        winsound.Beep(988, 80)    # B5
        winsound.Beep(1319, 120)  # E6
    except Exception:
        pass


def make_bar(value: float, length: int = 15, filled_char: str = "■", empty_char: str = "□") -> str:
    val = max(0.0, min(1.0, value))
    filled = int(round(val * length))
    return filled_char * filled + empty_char * (length - filled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=None)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--hop', type=int, default=80, help='Hop size in ms (default 80ms)')
    parser.add_argument('--cooldown', type=float, default=1.2, help='Cooldown in seconds after trigger')
    parser.add_argument('--no-chime', action='store_true')
    parser.add_argument('--list-devices', action='store_true')
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print(f"{RED}Ошибка:{RESET} sounddevice не установлен. Выполните: pip install sounddevice", file=sys.stderr)
        return

    if args.list_devices:
        print("\n" + "=" * 60)
        print("Доступные аудиоустройства ввода (микрофоны):")
        print("=" * 60)
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f" [{idx:2d}] {dev['name']} (входов: {dev['max_input_channels']})")
        print("=" * 60)
        return

    model_dir = args.model or find_best_model()
    if not model_dir or not (model_dir / 'wake_model.onnx').exists():
        print(f"{RED}Ошибка: обученная модель не найдена!{RESET}", file=sys.stderr)
        return

    detector = WakeDetector(model_dir)
    config = detector.config
    wake_word = config.get('wake_word_ru', 'бот')
    
    # If user didn't specify threshold, use 0.92 for comfortable interactive sensitivity,
    # or model's calibrated threshold if available
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(config.get('threshold', 0.995))

    # Device info
    if args.device is not None:
        dev_info = sd.query_devices(args.device)
        device_name = f"[{args.device}] {dev_info['name']}"
    else:
        default_in = sd.default.device[0]
        dev_info = sd.query_devices(default_in) if default_in is not None and default_in >= 0 else {}
        device_name = f"Default: {dev_info.get('name', 'Основной микрофон')}"

    # Print clear dashboard header
    os.system('cls' if os.name == 'nt' else 'clear')
    print(CYAN + "=" * 74)
    print(f"  🤖 НЕЙРОСЕТЕВОЙ МОНИТОР ВЕЙК-ВОРДА «{wake_word.upper()}» (Режим Алисы)")
    print("=" * 74 + RESET)
    print(f"  {WHITE}Модель:{RESET}     {model_dir.name} (INT8 ONNX)")
    print(f"  {WHITE}Микрофон:{RESET}   {device_name}")
    print(f"  {WHITE}Порог:{RESET}      {threshold * 100:.1f}%")
    print(f"  {WHITE}Окно:{RESET}       {N_SAMPLES / SAMPLE_RATE:.1f} сек (скользящее окно, шаг {args.hop} мс)")
    print(f"  {DIM}Управление: Говорите в микрофон. [Ctrl+C] — выход.{RESET}")
    print(CYAN + "-" * 74 + RESET)
    print(f"{WHITE}Журнал распознавания:{RESET}")
    print("-" * 74)

    hop_samples = int(SAMPLE_RATE * (args.hop / 1000.0))
    buffer = np.zeros(N_SAMPLES, dtype=np.float32)
    audio_queue = queue.Queue()
    last_trigger_time = 0.0
    detected_count = 0
    consecutive_hits = 0

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=hop_samples, device=args.device, callback=audio_callback):
            while True:
                chunk = audio_queue.get()
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk

                # Calculate volume (RMS)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                vol_pct = min(1.0, rms * 15.0)  # scale for mic visualization

                now = time.time()
                in_cooldown = (now - last_trigger_time) < args.cooldown

                if in_cooldown:
                    prob = 0.0
                    consecutive_hits = 0
                else:
                    prob = detector.score(buffer)

                if prob >= threshold and not in_cooldown:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                # Require 2 consecutive frames (160ms persistence) or strong peak (>= 0.9985)
                is_wake = (consecutive_hits >= 2 or prob >= 0.9985) and not in_cooldown

                # Format bars
                vol_bar = make_bar(vol_pct, length=10)
                prob_bar = make_bar(prob, length=12)

                # Status label
                if in_cooldown:
                    status_text = f"{YELLOW}[ ⏳ ПАУЗА ]{RESET} Остывание после срабатывания..."
                elif is_wake or prob >= threshold:
                    status_text = f"{GREEN}[ 🔥 ЕСТЬ! ] СЛОВО «{wake_word.upper()}» НАЙДЕНО!{RESET}"
                elif vol_pct > 0.12:
                    status_text = f"{RED}[ ❌ НЕТ  ]{RESET} Речь есть, слова «{wake_word}» НЕТ"
                else:
                    status_text = f"{DIM}[ 💤 ТИШ  ] Жду речь...{RESET}"

                # Live dashboard line (overwriting in place)
                prob_color = GREEN if (is_wake or prob >= threshold) else (YELLOW if prob > 0.4 else DIM)
                line = (
                    f"\rЗвук:[{vol_bar}] | "
                    f"«{wake_word}»:{prob_color}[{prob_bar}] {prob * 100:5.1f}%{RESET} | "
                    f"{status_text}   "
                )
                sys.stdout.write(line)
                sys.stdout.flush()

                # Handle detection trigger
                if is_wake:
                    last_trigger_time = now
                    consecutive_hits = 0
                    detected_count += 1
                    ts_str = time.strftime('%H:%M:%S', time.localtime(now))
                    
                    # Print persistent log line above live status
                    sys.stdout.write("\n" + GREEN + "=" * 74 + "\n")
                    sys.stdout.write(f"  🔔 [{ts_str}] # {detected_count} -> СЛОВО «{wake_word.upper()}» РАСПОЗНАНО! Уверенность: {prob * 100:.2f}%\n")
                    sys.stdout.write(f"  🎙️ Сигнал подан! Голосовой ассистент слушает команду...\n")
                    sys.stdout.write("=" * 74 + RESET + "\n")
                    sys.stdout.flush()

                    if not args.no_chime:
                        threading.Thread(target=play_chime, daemon=True).start()

    except KeyboardInterrupt:
        print("\n\n" + CYAN + "=" * 74)
        print(f"🛑 Монитор остановлен. Всего срабатываний: {detected_count}")
        print("=" * 74 + RESET)


if __name__ == '__main__':
    main()
