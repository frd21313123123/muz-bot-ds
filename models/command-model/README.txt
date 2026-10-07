Spoken music command classifier (11 classes). Mono PCM16, 16000 Hz, two-second window.
Classes: pause, resume, skip, stop, volume_up, volume_down, loop_toggle, autoplay_toggle, queue_clear, play_search, unknown.
pip install numpy onnxruntime
python command_audio.py . command.wav
INT8 quantized Whisper-tiny backbone (~7.7 MB, ~13 ms CPU latency).
