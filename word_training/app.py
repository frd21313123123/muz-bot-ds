"""
Console Application for Wake-Word Spotter "БОТ" (Alice Mode).
Provides an interactive menu for running live microphone detection,
adjusting sensitivity, selecting devices, or running test recordings.
"""

import os
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
    CYAN = colorama.Fore.CYAN + colorama.Style.BRIGHT
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    MAGENTA = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL
    DIM = colorama.Style.DIM
except ImportError:
    CYAN = GREEN = YELLOW = MAGENTA = WHITE = RESET = DIM = ""

import interactive_listener
import interactive_command_listener
import quick_test


def show_menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(CYAN + "=" * 74)
    print("   🤖 НЕЙРОСЕТЕВОЙ ГОЛОСОВОЙ КОНТРОЛЛЕР (РЕЖИМ АЛИСЫ + КОМАНДЫ)")
    print("=" * 74 + RESET)
    print(f"  {WHITE}Вейк-ворд:{RESET}   «БОТ» (Recall ≥91%, защита от «боб»/«болт», INT8 7.7 МБ)")
    print(f"  {WHITE}Команды:{RESET}     11 классов (пауза, стоп, скип, громче, тише, повтор...)")
    print(CYAN + "-" * 74 + RESET)
    print(f"  {WHITE}Выберите режим работы:{RESET}\n")
    print(f"   {GREEN}[1]{RESET} Запустить детектор вейк-ворда «БОТ» (Калиброванный порог)")
    print(f"   {YELLOW}[2]{RESET} Запустить с повышенной чувствительностью для тихой речи (Порог 98%)")
    print(f"   {CYAN}[3]{RESET} Выбрать конкретный микрофон из списка устройств")
    print(f"   {WHITE}[4]{RESET} Быстрый тест на готовых аудиозаписях (без микрофона)")
    print(f"   {MAGENTA}[5]{RESET} 🎧 Живое распознавание команд плеера с микрофона (Offline INT8)")
    print(f"   {DIM}[0] Выход{RESET}")
    print(CYAN + "=" * 74 + RESET)


def safe_pause():
    try:
        input("\nНажмите Enter для возврата в меню...")
    except (EOFError, KeyboardInterrupt):
        pass


def main():
    while True:
        show_menu()
        try:
            choice = input("\nВведите номер пункта [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if choice == '1':
            sys.argv = ['interactive_listener.py']
            interactive_listener.main()
            safe_pause()
        elif choice == '2':
            sys.argv = ['interactive_listener.py', '--threshold', '0.98']
            interactive_listener.main()
            safe_pause()
        elif choice == '3':
            os.system('cls' if os.name == 'nt' else 'clear')
            try:
                import sounddevice as sd
                print(CYAN + "=" * 70)
                print("  🎤 ДОСТУПНЫЕ МИКРОФОНЫ:")
                print("=" * 70 + RESET)
                devices = sd.query_devices()
                inputs = []
                for idx, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        inputs.append(idx)
                        print(f"  [{idx:2d}] {dev['name']} (входов: {dev['max_input_channels']})")
                print(CYAN + "=" * 70 + RESET)
                dev_idx_str = input("\nВведите номер микрофона: ").strip()
                if dev_idx_str.isdigit() and int(dev_idx_str) in inputs:
                    sys.argv = ['interactive_listener.py', '--device', dev_idx_str]
                    interactive_listener.main()
                else:
                    print("Неверный номер устройства.")
            except Exception as e:
                print(f"Ошибка при работе с аудиоустройствами: {e}")
            safe_pause()
        elif choice == '4':
            quick_test.main()
            safe_pause()
        elif choice == '5':
            sys.argv = ['interactive_command_listener.py']
            interactive_command_listener.main()
            safe_pause()
        elif choice == '0':
            print("\nДо свидания!")
            break
        else:
            print("Неверный выбор, попробуйте снова.")


if __name__ == '__main__':
    main()
