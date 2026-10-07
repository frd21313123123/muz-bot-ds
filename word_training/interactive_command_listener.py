"""
Interactive Real-Time Voice Command Tester.
Listens to your microphone and classifies spoken music bot commands in real time:
пауза, стоп, дальше/скип, громче, тише, повтор, автоплей, поиск песни и др.
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

try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    RED = colorama.Fore.RED + colorama.Style.BRIGHT
    YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    CYAN = colorama.Fore.CYAN + colorama.Style.BRIGHT
    MAGENTA = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL
    DIM = colorama.Style.DIM
except ImportError:
    GREEN = RED = YELLOW = CYAN = MAGENTA = WHITE = RESET = DIM = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / '.runtime/wake-validation'),
    str(ROOT / 'scripts/wakeword'),
    str(Path(__file__).parent)
]

from wake_audio import N_SAMPLES, SAMPLE_RATE
from command_audio import CommandDetector


ACTION_TITLES = {
    "pause": ("⏸ ПАУЗА", "Поставить музыку на паузу"),
    "resume": ("▶ ПРОДОЛЖИТЬ", "Возобновить воспроизведение"),
    "skip": ("⏭ СЛЕДУЮЩИЙ ТРЕК", "Пропустить текущий трек"),
    "stop": ("⏹ СТОП", "Остановить плеер и отключиться"),
    "volume_up": ("🔊 ГРОМЧЕ", "Увеличить громкость воспроизведения"),
    "volume_down": ("🔉 ТИШЕ", "Уменьшить громкость воспроизведения"),
    "loop_on": ("🔂 ПОВТОР ТРЕКА", "Включить зацикливание песни"),
    "autoplay_on": ("♾ АВТОПЛЕЙ", "Включить бесконечный режим"),
    "queue_clear": ("🗑 ОЧИСТКА ОЧЕРЕДИ", "Удалить все треки из очереди"),
    "play_search": ("🔍 ПОИСК ПЕСНИ (YouTube)", "Запрос на включение музыки через Groq STT"),
    "unknown": ("❓ НЕИЗВЕСТНО", "Обычная речь / не команда"),
}


def play_chime():
    try:
        import winsound
        winsound.Beep(1200, 100)
    except Exception:
        pass


def make_bar(value: float, length: int = 15, filled_char: str = "■", empty_char: str = "□") -> str:
    val = max(0.0, min(1.0, value))
    filled = int(round(val * length))
    return filled_char * filled + empty_char * (length - filled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/command-model')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None, help='Detection threshold (default from model config: 0.88)')
    parser.add_argument('--hop', type=int, default=120)
    parser.add_argument('--cooldown', type=float, default=2.0, help='Cooldown in seconds after command trigger (default 2.0s)')
    parser.add_argument('--no-chime', action='store_true')
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print(f"{RED}Ошибка:{RESET} sounddevice не установлен.", file=sys.stderr)
        return

    model_dir = args.model
    if not (model_dir / 'command_model.onnx').is_file():
        print(f"{RED}Ошибка:{RESET} Модель не найдена в {model_dir}", file=sys.stderr)
        return

    detector = CommandDetector(model_dir)
    threshold = float(args.threshold or detector.threshold)

    print(CYAN + "=" * 76)
    print("  🎧 НЕЙРОСЕТЕВОЙ КЛАССИФИКАТОР ГОЛОСОВЫХ КОМАНД (Whisper INT8)")
    print("=" * 76 + RESET)
    print(f"  {WHITE}Модель:{RESET}     {model_dir.name} (7.7 МБ, INT8 ONNX)")
    print(f"  {WHITE}Классов:{RESET}    {len(detector.classes)} команд плеера")
    print(f"  {WHITE}Порог:{RESET}      {threshold * 100:.1f}%")
    print(f"  {WHITE}Команды:{RESET}    пауза, продолжи, следующий, стоп, громче, тише, повтор,")
    print("              автоплей, очисти очередь, включи <песню>")
    print(CYAN + "-" * 76 + RESET)
    print(f"  {DIM}Говорите в микрофон команды. Для выхода нажмите Ctrl+C.{RESET}\n")

    audio_queue = queue.Queue(maxsize=100)
    audio_buffer = np.zeros(N_SAMPLES, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(mono)
        except queue.Full:
            pass

    hop_samples = int(SAMPLE_RATE * (args.hop / 1000.0))
    buffer_lock = threading.Lock()
    running = True

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=hop_samples,
            channels=1,
            dtype='float32',
            device=args.device,
            callback=audio_callback
        )
    except Exception as err:
        print(f"{RED}Ошибка аудио:{RESET} {err}", file=sys.stderr)
        return

    last_trigger_time = 0.0
    is_speaking = False
    speech_chunks = 0
    trailing_silence = 0

    VAD_SPEECH = 0.014      # RMS threshold for active speech
    TRAIL_CHUNKS = 2        # 2 hops of 120ms = 240ms silence to mark phrase end
    MIN_SPEECH_CHUNKS = 2   # Require at least 240ms speech (filters clicks/mic noise)

    with stream:
        try:
            while running:
                chunk = audio_queue.get()
                with buffer_lock:
                    audio_buffer = np.roll(audio_buffer, -len(chunk))
                    audio_buffer[-len(chunk):] = chunk

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                rms_bar = make_bar(min(1.0, rms * 15), 10)
                now = time.time()

                in_cooldown = (now - last_trigger_time) < args.cooldown
                if in_cooldown:
                    cooldown_rem = max(0.0, args.cooldown - (now - last_trigger_time))
                    status_line = f"\r  Звук: [{rms_bar}] | {YELLOW}[ ⏳ ПАУЗА ]{RESET} Команда принята, пауза {cooldown_rem:.1f}с..."
                    sys.stdout.write(status_line.ljust(78))
                    sys.stdout.flush()
                    continue

                # 1. Active speech detected: accumulate audio in rolling buffer
                if rms >= VAD_SPEECH:
                    if not is_speaking:
                        is_speaking = True
                        speech_chunks = 0
                    speech_chunks += 1
                    trailing_silence = 0

                    status_line = f"\r  Звук: [{rms_bar}] | {CYAN}[ 🎙️ СЛУШАЮ... ]{RESET} Говорите команду..."
                    sys.stdout.write(status_line.ljust(78))
                    sys.stdout.flush()
                    continue

                # 2. Silence / speech pause
                if is_speaking:
                    trailing_silence += 1

                    # If silence is brief (< 240ms), keep waiting for speech completion
                    if trailing_silence < TRAIL_CHUNKS:
                        status_line = f"\r  Звук: [{rms_bar}] | {CYAN}[ ⏳ ОБРАБОТКА... ]{RESET} Завершение фразы..."
                        sys.stdout.write(status_line.ljust(78))
                        sys.stdout.flush()
                        continue

                    # 3. Speech Boundary Reached! The full utterance is now in the buffer!
                    is_speaking = False
                    trailing_silence = 0

                    if speech_chunks >= MIN_SPEECH_CHUNKS:
                        res = detector.predict(audio_buffer)
                        conf = res["confidence"]
                        action = res["action"]
                        is_control = action in ["pause", "resume", "skip", "stop", "volume_up", "volume_down", "loop_on", "autoplay_on", "queue_clear", "play_search"]

                        if is_control and conf >= threshold:
                            last_trigger_time = now
                            title, desc = ACTION_TITLES.get(action, (action.upper(), ""))
                            color = GREEN if action != "play_search" else YELLOW
                            print(f"\r  {color}[РАСПОЗНАНО] {title:<20} ({conf*100:5.1f}%) -> {desc}{RESET}")

                            if not args.no_chime:
                                threading.Thread(target=play_chime, daemon=True).start()

                            # Clear buffer and drain queue
                            with buffer_lock:
                                audio_buffer.fill(0)
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                            continue
                        elif conf >= 0.40 and action != "unknown":
                            title, _ = ACTION_TITLES.get(action, (action, ""))
                            status_line = f"\r  Звук: [{rms_bar}] | {YELLOW}[ ⚠️ НЕДОСТАТОЧНО УВЕРЕННОСТИ ]{RESET} {title} ({conf*100:.1f}% < {threshold*100:.0f}%)"
                            sys.stdout.write(status_line.ljust(78))
                            sys.stdout.flush()
                            time.sleep(0.3)
                            continue

                # 4. Idle room silence
                status_line = f"\r  Звук: [{rms_bar}] | Ожидание команды..."
                sys.stdout.write(status_line.ljust(78))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print(f"\n\n{CYAN}Остановлено пользователем.{RESET}")


if __name__ == '__main__':
    main()
