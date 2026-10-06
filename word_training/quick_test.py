"""
Quick benchmark test script to verify model classification
on pre-recorded audio files: positive ('бот') vs negative ('кот', 'робот', speech, noise).
"""

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / '.runtime/wake-validation'),
    str(ROOT / 'scripts/wakeword'),
    str(Path(__file__).parent)
]

try:
    import colorama
    colorama.init()
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    RED = colorama.Fore.RED + colorama.Style.BRIGHT
    CYAN = colorama.Fore.CYAN + colorama.Style.BRIGHT
    YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = CYAN = YELLOW = WHITE = RESET = ""

from wake_audio import WakeDetector, read_wav


def main():
    model_dir = ROOT / 'word_training/runs/bot-v5-acc90/wake-model'
    if not (model_dir / 'wake_model.onnx').exists():
        print("Ошибка: модель bot-v5-acc90 не найдена.")
        return

    detector = WakeDetector(model_dir)
    threshold = 0.95

    print("\n" + CYAN + "=" * 70)
    print("  🧪 БЫСТРЫЙ ТЕСТ МОДЕЛИ НА АУДИОЗАПИСЯХ (БОТ vs ДРУГИЕ СЛОВА/ШУМ)")
    print("=" * 70 + RESET)
    print(f"  Модель: {model_dir.name} (INT8 ONNX)")
    print(f"  Порог детекции: {threshold * 100:.2f}%\n")

    dataset_wav_dir = ROOT / 'word_training/runs/bot-v2/dataset/wav'
    
    # Select test samples
    samples = [
        ("Слово «БОТ» (MMS rus)", dataset_wav_dir / "mms-facebook-mms-tts-rus-rus-1-00000.wav", True),
        ("Слово «БОТ» (Piper Denis)", dataset_wav_dir / "piper-piper-ru_RU-denis-medium-1-00088.wav", True),
        ("Слово «БОТ» во фразе (Silero Aidar)", dataset_wav_dir / "silero-v5_5_ru-aidar-1-00000.wav", True),
        ("Отрицательное слово (созвучное)", dataset_wav_dir / "mms-facebook-mms-tts-rus-rus-0-00000.wav", False),
        ("Отрицательное слово (Piper Denis)", dataset_wav_dir / "piper-piper-ru_RU-denis-medium-0-00000.wav", False),
        ("Фоновый шум (noise)", dataset_wav_dir / "noise-00000.wav", False),
    ]

    for title, wav_path, is_pos in samples:
        if not wav_path.exists():
            continue
        audio = read_wav(wav_path)
        prob = detector.score(audio)
        detected = prob >= threshold

        correct = (detected == is_pos)
        verdict = f"{GREEN}✅ ВЕРНО{RESET}" if correct else f"{RED}❌ ОШИБКА{RESET}"
        
        if detected:
            status = f"{GREEN}[ 🔥 ЕСТЬ: СЛОВО «БОТ» НАЙДЕНО ]{RESET}"
        else:
            status = f"{RED}[ ❌ НЕТ: СЛОВА «БОТ» НЕТ      ]{RESET}"

        expected_str = f"{GREEN}Ожидалось: ДА{RESET}" if is_pos else f"{WHITE}Ожидалось: НЕТ{RESET}"
        print(f"Тест: {WHITE}{title:<34s}{RESET} | {expected_str}")
        print(f"      Результат: {status} | Уверенность: {prob * 100:6.2f}% | {verdict}\n")

    print(CYAN + "=" * 70)
    print("  Тестирование завершено.")
    print("=" * 70 + RESET + "\n")


if __name__ == '__main__':
    main()
