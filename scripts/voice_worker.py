"""JSONL worker. Audio stays in RAM; stdout is exclusively the IPC protocol."""
import base64
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime" / "voice"
os.environ.setdefault("HF_HOME", str(RUNTIME / "cache"))
os.environ.setdefault("USE_TF", "0")

COMMAND_PROMPT = "Музыкальные команды: включи, включи музыку, продолжи музыку, на паузу, следующий трек, громче, тише, громкость."

QUESTIONS = {"action": {
    "type": "choice",
    "instructions": "Выбери ровно одно действие музыкального бота по просьбе пользователя. Непонятная, отрицательная, составная просьба или заказ музыки: unknown.",
    "criteria": {
        "skip": "Пропустить текущий трек, следующий трек",
        "pause": "Приостановить музыку, поставить на паузу",
        "resume": "Продолжить музыку после паузы",
        "stop": "Остановить музыку и отключить бота от голосового канала",
        "volume_set": "Установить громкость на указанное число процентов",
        "volume_up": "Сделать музыку громче без указания числа",
        "volume_down": "Сделать музыку тише без указания числа",
        "unknown": "Любая другая, неоднозначная или составная просьба; поиск или заказ песни",
    },
}}

# A shorter alternative helps short commands whose first decision is uncertain.
# Both passes use the same loaded checkpoint. A weak second decision abstains.
SHORT_QUESTIONS = {"action": {"type": "choice", "instructions": "Определи команду.", "criteria": {
    "skip": "следующий трек", "pause": "поставить на паузу", "resume": "снять с паузы",
    "stop": "остановить музыку и выйти", "volume_set": "громкость в процентах",
    "volume_up": "сделать громче", "volume_down": "сделать тише", "unknown": "другая просьба",
}}}


def classify(agent, text):
    # ASR casing/punctuation should not change the action of a short command.
    text = text.strip().rstrip('.!…').strip().lower()
    text = text[:1].upper() + text[1:]
    answer = agent.predict(text, QUESTIONS)["answers"]["action"]
    if answer["choice"] != "unknown" and answer["answer_confidence"] < 0.6:
        answer = agent.predict(text, SHORT_QUESTIONS)["answers"]["action"]
        if answer["answer_confidence"] < 0.75:
            return {"action": "unknown", "confidence": answer["answer_confidence"]}
    return {"action": answer["choice"], "confidence": answer["answer_confidence"]}


def models(prepare=False):
    import torch
    import laya
    from faster_whisper import WhisperModel
    import numpy as np

    class CommandWhisper(WhisperModel):
        # CTranslate2 4.8 accepts a shorter encoder window. faster-whisper still
        # pads every utterance to 30 seconds; keep the real audio and 1.5 s margin,
        # with at least 6 s of context, instead of encoding all that padding.
        encoder_frames = 3000

        def encode(self, features):
            return super().encode(features[..., :self.encoder_frames])

        def command(self, audio, prompt=None):
            frames = int(len(audio) / 16000 * 100) + 150
            self.encoder_frames = min(3000, max(600, (frames + 1) // 2 * 2))
            segments, _ = self.transcribe(audio, language="ru", beam_size=1, temperature=0,
                                          condition_on_previous_text=False, vad_filter=True,
                                          initial_prompt=prompt or COMMAND_PROMPT, without_timestamps=True, max_new_tokens=96)
            return " ".join(segment.text.strip() for segment in segments)
    torch.set_num_threads(max(1, min(8, int(os.environ.get("VOICE_CPU_THREADS", "4")))))
    marker = RUNTIME / "ready.json"
    manifest = {} if prepare else json.loads(marker.read_text(encoding="utf-8"))
    name = os.environ.get("WHISPER_MODEL", "small") if prepare else manifest["whisper"]
    whisper = CommandWhisper(name, device="cpu", compute_type="int8", cpu_threads=torch.get_num_threads(),
                           download_root=str(RUNTIME / "whisper"), local_files_only=not prepare)
    laya_model = "convaiinnovations/laya-multilingual"
    revision = manifest.get("laya_revision")
    if prepare:
        from huggingface_hub import model_info
        revision = model_info(laya_model).sha
    agent = laya.load(laya_model, device="cpu", revision=revision)
    # Warm encoder/decoder and VAD as well as Laya before the first real wake.
    whisper.encoder_frames = 600
    warm_segments, _ = whisper.transcribe(np.zeros(16000, dtype=np.float32), language="ru",
                                          beam_size=1, temperature=0, without_timestamps=True,
                                          max_new_tokens=8, condition_on_previous_text=False)
    list(warm_segments)
    whisper.command(np.zeros(16000, dtype=np.float32))
    agent.predict("Следующий трек", QUESTIONS)
    if prepare:
        marker.write_text(json.dumps({"whisper": name, "laya": laya_model, "laya_revision": revision}), encoding="utf-8")
    return whisper, agent


def main():
    prepare = "--prepare" in sys.argv
    with redirect_stdout(sys.stderr):
        whisper, agent = models(prepare)
    if prepare:
        return
    print(json.dumps({"ready": True}), flush=True)
    import numpy as np
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 700_000:
                raise ValueError("Request too large")
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if request["op"] == "transcribe":
                    pcm = base64.b64decode(request["pcm"], validate=True)
                    if not pcm or len(pcm) > 480_000 or len(pcm) % 2:
                        raise ValueError("Invalid audio")
                    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    wake_name = request.get("wakeName")
                    prompt = f"Обращение к боту: {wake_name}." if isinstance(wake_name, str) and len(wake_name) <= 64 else None
                    result = whisper.command(audio, prompt)
                elif request["op"] == "classify":
                    text = request["text"]
                    if not isinstance(text, str) or len(text) > 1000:
                        raise ValueError("Invalid text")
                    result = classify(agent, text)
                else:
                    raise ValueError("Unknown operation")
            print(json.dumps({"id": request["id"], "result": result}, ensure_ascii=False), flush=True)
        except Exception:
            # No exception details: they can contain audio, text or prompts.
            print(json.dumps({"id": request.get("id"), "error": True}), flush=True)


if __name__ == "__main__":
    main()
