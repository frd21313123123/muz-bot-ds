"""JSONL worker. Audio stays in RAM; stdout is exclusively the IPC protocol."""
import base64
from contextlib import redirect_stdout
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
PROFILE = os.environ.get('VOICE_PROFILE', 'light').strip().lower()
if PROFILE not in ('light', 'heavy'):
    raise ValueError('VOICE_PROFILE must be light or heavy')
STT_PROVIDER = os.environ.get('STT_PROVIDER', 'groq').strip().lower() or 'groq'
if STT_PROVIDER not in ('groq', 'local'):
    raise ValueError('STT_PROVIDER must be groq or local')
RUNTIME = ROOT / '.runtime' / ('voice-light' if PROFILE == 'light' else 'voice')
WHISPER_CACHE = ROOT / '.runtime' / 'voice' / 'whisper'
DEFAULT_LAYA_MODEL = "convaiinnovations/laya-multilingual"
os.environ.setdefault("HF_HOME", str(RUNTIME / "cache"))
os.environ.setdefault("USE_TF", "0")


def laya_source(manifest, prepare=False):
    configured = os.environ.get("LAYA_MODEL_PATH", "").strip()
    if configured:
        directory = Path(configured).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError("Configured local Laya checkpoint is missing")
        return str(directory), None
    return (DEFAULT_LAYA_MODEL, None) if prepare else (manifest.get("laya", DEFAULT_LAYA_MODEL), manifest.get("laya_revision"))

COMMAND_PROMPT = ("Музыкальный плеер: включи, выключи, бесконечный режим, автоплей, повтор трека, "
                  "очисти очередь, голосовое управление, поставь на паузу, продолжи музыку, "
                  "следующий трек, останови музыку, громкость, сделай громче, сделай тише. "
                  "Linkin Park, Numb, live, remix, cover. Монеточка, Земфира, Кино, Сплин, Баста, Би-2. "
                  "Радио: Европа Плюс, Europa Plus, Ретро FM, Retro FM, Дорожное радио, Русское радио, "
                  "Авторадио, DFM, Радио ENERGY, NRJ, Love Radio, Радио Дача, Наше радио.")

# Give the short, ambiguous name competing spellings instead of telling the
# recognizer that it must hear the bot's name. Similar words remain non-wakes.
WAKE_VOCABULARY = {"бот": "Бот, вот, кот, год, рот, борт, порт."}

MUSIC_QUESTIONS = {
    "next_tool": {
        "type": "choice",
        "instructions": "Определи команду.",
        "criteria": {
            "search": "включить песню",
            "url": "включить ссылку YouTube",
            "control": "пауза, продолжить, пропустить, громкость",
            "unknown": "другая просьба",
        },
    },
    "search_result_policy": {
        "type": "choice",
        "instructions": "Какая версия песни нужна?",
        "criteria": {
            "original": "обычная песня",
            "live": "live концертная версия",
            "remix": "remix ремикс",
            "cover": "cover кавер",
            "acoustic": "acoustic акустическая",
            "instrumental": "instrumental инструментальная",
        },
    },
}

MUSIC_SHORT_QUESTIONS = {"next_tool": {"type": "choice", "instructions": "Определи команду.", "criteria": {
    "search": "включить песню", "url": "включить ссылку YouTube",
    "control": "управлять плеером", "unknown": "другая просьба",
}}}

VERSION_ALIASES = (
    (r"\bконцертн\w*(?:\s+верси\w*)?\b", "live"),
    (r"\bремикс\w*\b", "remix"),
    (r"\bкавер\w*\b", "cover"),
    (r"\bакустическ\w*(?:\s+верси\w*)?\b", "acoustic"),
    (r"\bинструментальн\w*(?:\s+верси\w*)?\b", "instrumental"),
)

V3_SCHEMA = json.loads((ROOT / "scripts" / "laya_music_v3.json").read_text(encoding="utf-8"))


def is_music_v3(agent):
    return getattr(agent, "cfg", {}).get("model_name") == V3_SCHEMA["profile"]


def requested_variant(query):
    for pattern, name in ((r"\b(live|концертн\w*)\b", "live"), (r"\b(remix|ремикс\w*)\b", "remix"),
                          (r"\b(cover|кавер\w*)\b", "cover"), (r"\b(acoustic|акустическ\w*)\b", "acoustic"),
                          (r"\b(instrumental|инструментальн\w*)\b", "instrumental")):
        if re.search(pattern, query, re.IGNORECASE):
            return name
    return None


def model_music_query(query):
    for pattern, version in VERSION_ALIASES:
        query = re.sub(pattern, version, query, flags=re.IGNORECASE)
    query = re.sub(r"\b(live|remix|cover|acoustic|instrumental)\b", lambda match: match.group().lower(), query, flags=re.IGNORECASE)
    return query


def validate_candidates(query, candidates):
    if not isinstance(query, str) or not query or len(query) > 1000 or not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        raise ValueError("Invalid candidates")
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or item.get("index") != index:
            raise ValueError("Invalid index")
        for key, limit in (("title", 240), ("artist", 120), ("duration", 32)):
            if not isinstance(item.get(key), str) or len(item[key]) > limit:
                raise ValueError("Invalid candidate")


def v3_candidates(candidates):
    results = []
    for item in candidates:
        duration = item["duration"].split(":")
        seconds = 0
        if all(part.isdigit() for part in duration):
            for part in duration:
                seconds = seconds * 60 + int(part)
        title = item["title"]
        for pattern, version in VERSION_ALIASES:
            title = re.sub(pattern, version, title, flags=re.IGNORECASE)
        results.append({"id": f'c{item["index"]}', "title": title, "artist": item["artist"], "duration": seconds})
    return results


def v3_music_route(agent, state):
    parsed = state.get("parsed_request")
    if not isinstance(parsed, dict) or parsed.get("kind") != "play":
        raise ValueError("Missing parsed music request")
    query = parsed.get("search_query")
    if query is not None and (not isinstance(query, str) or not query or len(query) > 1000):
        raise ValueError("Invalid parsed query")
    model_state = {"phase": "request", "message": state["message"], "parsed_request": parsed,
                   "selected_track": None, "current_track": None,
                   "player": state.get("player", {"connected": True, "playing": False, "paused": False,
                                                  "autoplay": False, "queue_length": 0}),
                   "available_tools": ["youtube_music_search", "direct_youtube_video", "player_control"]}
    answers = agent.predict(model_state, V3_SCHEMA["routing"], max_len=2048, head_max_len=768)["answers"]
    route = answers["next_tool"]
    tool = route["choice"] if route["choice"] in model_state["available_tools"] else "unknown"
    confidence = route["answer_confidence"]
    if tool == "youtube_music_search":
        source = answers["query_source"]
        confidence = min(confidence, source["answer_confidence"])
        if source["choice"] != "parsed_search_query":
            tool = "unknown"
    return {"next_tool": tool, "query_source": "message", "search_result_policy": "play_first_result",
            "defer_result_policy": tool == "youtube_music_search", "confidence": confidence}


def music_result_policy(agent, query, candidates):
    validate_candidates(query, candidates)
    if not is_music_v3(agent):
        raise ValueError("Result-policy phase is only used by the v3 profile")
    state = {"phase": "tool_result", "tool": "youtube_music_search", "query": model_music_query(query),
             "request_variant": requested_variant(query), "results": v3_candidates(candidates)}
    answer = agent.predict(state, V3_SCHEMA["tool_result"], max_len=2048, head_max_len=768)["answers"]["search_result_policy"]
    return {"search_result_policy": answer["choice"], "confidence": answer["answer_confidence"]}


def v3_music_rerank(agent, query, candidates):
    items = v3_candidates(candidates)
    version = requested_variant(query)
    confidence = {}
    if version:
        # A trained best_track head can be confident about an incompatible
        # version. Confirm candidates' versions independently, as in the
        # multilingual adapter, before comparing otherwise compatible tracks.
        decisions = agent.predict_batch(candidates, {"version": MUSIC_QUESTIONS["search_result_policy"]},
                                        batch_size=5, max_len=2048, head_max_len=768)
        for item, decision in zip(items, decisions, strict=True):
            answer = decision["answers"]["version"]
            probability = answer["probabilities"].get(version, 0)
            if answer["choice"] == version and probability >= 0.6:
                confidence[item["id"]] = probability
        items = [item for item in items if item["id"] in confidence]
        if not items:
            return {"best_track": 0, "confidence": 0}
    # Metadata is already in the candidate state. Keep the option text to the
    # title, rather than changing the trained choice shape with extra suffixes.
    criteria = {item["id"]: item["title"] for item in items}
    criteria["none"] = V3_SCHEMA["rerank"]["none"]
    questions = {"best_track": {"type": "choice", "instructions": V3_SCHEMA["rerank"]["instructions"], "criteria": criteria}}
    state = {"phase": "rerank", "query": model_music_query(query), "constraints": {"variant": version}, "candidates": items}
    answer = agent.predict(state, questions, max_len=2048, head_max_len=768)["answers"]["best_track"]
    choice = answer["choice"]
    if choice == "none":
        return {"best_track": -1, "no_match": True, "confidence": answer["answer_confidence"]}
    if choice not in {item["id"] for item in items}:
        raise ValueError("Invalid model candidate")
    return {"best_track": int(choice[1:]), "confidence": min(answer["answer_confidence"], confidence.get(choice, 1))}


def music_route(agent, state):
    if not isinstance(state, dict) or not isinstance(state.get("message"), str) or len(state["message"]) > 1000 or state.get("selected_track") is not None:
        raise ValueError("Invalid state")
    if is_music_v3(agent):
        return v3_music_route(agent, state)
    # Canonicalize explicit version aliases for the decision model only. The
    # original transcript/query remains in TypeScript and is sent to yt-dlp.
    state = {"message": state["message"], "selected_track": None}
    for pattern, version in VERSION_ALIASES:
        state["message"] = re.sub(pattern, version, state["message"], flags=re.IGNORECASE)
    answers = agent.predict(state, MUSIC_QUESTIONS, max_len=2048, head_max_len=768)["answers"]
    route = answers["next_tool"]
    if route["choice"] != "unknown" and route["answer_confidence"] < 0.6:
        route = agent.predict(state, MUSIC_SHORT_QUESTIONS, max_len=2048, head_max_len=768)["answers"]["next_tool"]
    policy = answers["search_result_policy"]
    tools = {"search": "youtube_music_search", "url": "direct_youtube_video", "control": "player_control", "unknown": "unknown"}
    return {"next_tool": tools[route["choice"]], "query_source": "message",
            "search_result_policy": "rerank_results" if policy["choice"] != "original" and policy["answer_confidence"] >= 0.6 else "play_first_result",
            "confidence": route["answer_confidence"]}


def music_rerank(agent, query, candidates):
    validate_candidates(query, candidates)
    if is_music_v3(agent):
        return v3_music_rerank(agent, query, candidates)
    version = requested_variant(query)
    # First let Laya identify each candidate's actual version, without the
    # requested version in that state (it otherwise mistakes query text for
    # candidate metadata). Compare only model-confirmed versions in pass two.
    confidence = {}
    if version:
        results = agent.predict_batch(candidates, {"version": MUSIC_QUESTIONS["search_result_policy"]},
                                      batch_size=5, max_len=2048, head_max_len=768)
        for item, result in zip(candidates, results, strict=True):
            answer = result["answers"]["version"]
            probability = answer["probabilities"].get(version, 0)
            if answer["choice"] == version and probability >= 0.6:
                confidence[item["index"]] = probability
        candidates = [item for item in candidates if item["index"] in confidence]
        if not candidates:
            return {"best_track": 0, "confidence": 0}
        if len(candidates) == 1:
            index = candidates[0]["index"]
            return {"best_track": index, "confidence": confidence[index]}
    criteria = {str(item["index"]): f'{item["title"]} ({item["artist"]}; {item["duration"]})' for item in candidates}
    questions = {"best_track": {"type": "choice",
        "instructions": "Выбери песню.",
        "criteria": criteria}}
    answer = agent.predict({"query": query}, questions, max_len=2048, head_max_len=768)["answers"]["best_track"]
    index = int(answer["choice"])
    return {"best_track": index, "confidence": min(answer["answer_confidence"], confidence.get(index, 1))}

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
    if is_music_v3(agent) and text == "Сними с паузы":
        text = "Продолжи музыку"
    answer = agent.predict(text, QUESTIONS)["answers"]["action"]
    if answer["choice"] != "unknown" and answer["answer_confidence"] < 0.6:
        answer = agent.predict(text, SHORT_QUESTIONS)["answers"]["action"]
        if answer["answer_confidence"] < 0.75:
            return {"action": "unknown", "confidence": answer["answer_confidence"]}
    return {"action": answer["choice"], "confidence": answer["answer_confidence"]}


def whisper_names(manifest, prepare=False):
    if prepare:
        if PROFILE == 'light':
            name = os.environ.get('WHISPER_LIGHT_MODEL', 'small').strip()
            if name not in ('tiny', 'base', 'small'):
                raise ValueError('WHISPER_LIGHT_MODEL must be multilingual tiny, base or small')
            return name, name
        return (os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
                os.environ.get("WHISPER_WAKE_MODEL", "small"))
    # Old installations used one model for both phases.
    name, wake = manifest['whisper'], manifest.get('whisper_wake', manifest['whisper'])
    if PROFILE == 'light' and (manifest.get('profile') != 'light' or name not in ('tiny', 'base', 'small') or wake != name):
        raise ValueError('Prepare the light profile with setup:voice')
    return name, wake


def light_whisper_path(manifest, prepare=False):
    if PROFILE != 'light':
        return None
    configured = os.environ.get('WHISPER_LIGHT_MODEL_PATH', '').strip() if prepare else manifest.get('whisper_path')
    if not configured:
        return None
    directory = Path(configured).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    if not all((directory / filename).is_file() for filename in ('model.bin', 'config.json', 'tokenizer.json')):
        raise ValueError('WHISPER_LIGHT_MODEL_PATH must contain a complete CTranslate2 Whisper model')
    return directory


def load_wake_detector():
    wake_dir = os.environ.get("WAKE_MODEL_PATH", "").strip()
    if not wake_dir:
        for candidate in [
            ROOT / "models" / "wake-model",
            ROOT / "word_training" / "runs" / "bot-v5-acc90" / "wake-model",
        ]:
            if (candidate / "wake_model.onnx").is_file():
                wake_dir = str(candidate)
                break
    if not wake_dir or not (Path(wake_dir) / "wake_model.onnx").is_file():
        return None
    try:
        import sys
        if str(Path(wake_dir)) not in sys.path:
            sys.path.insert(0, str(Path(wake_dir)))
        from wake_audio import WakeDetector
        detector = WakeDetector(wake_dir)
        threshold_env = os.environ.get("WAKE_THRESHOLD", "").strip()
        if threshold_env:
            try:
                detector.config["threshold"] = float(threshold_env)
            except ValueError:
                pass
        else:
            detector.config["threshold"] = 0.95
        import numpy as np
        detector.detect(np.zeros(32000, dtype=np.float32))
        return detector
    except Exception as e:
        print(f"Warning: Failed to load WakeDetector: {e}", file=sys.stderr, flush=True)
        return None


def models(prepare=False):
    # PyTorch's Windows wheel bundles the CUDA/cuDNN DLLs needed by CTranslate2.
    # Keep DLL-directory handles alive for the lifetime of the worker.
    dll_handles = []
    if PROFILE == 'heavy':
        import torch
        import laya
        if os.name == 'nt':
            torch_lib = str(Path(torch.__file__).parent / 'lib')
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
            dll_handles.append(os.add_dll_directory(torch_lib))
    wake_detector = load_wake_detector()
    if STT_PROVIDER == 'groq':
        # STT runs in Node through Groq; this worker handles wake detection (and Laya in heavy profile)
        if PROFILE == 'light':
            if wake_detector is None:
                raise ValueError('Groq light profile requires a local wake detector')
            return None, None, None, wake_detector
        torch.set_num_threads(max(1, min(8, int(os.environ.get('VOICE_CPU_THREADS', '4')))))
        device = os.environ.get('VOICE_DEVICE', 'cpu').strip().lower()
        if device not in ('cpu', 'cuda'):
            raise ValueError('VOICE_DEVICE must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        marker = RUNTIME / 'ready.json'
        manifest = {} if prepare else json.loads(marker.read_text(encoding='utf-8'))
        laya_model, revision = laya_source(manifest, prepare)
        if prepare and not Path(laya_model).is_dir():
            from huggingface_hub import model_info
            revision = model_info(laya_model).sha
        agent = laya.load(laya_model, device=device, revision=revision)
        agent._voice_dll_handles = dll_handles
        agent.predict('Следующий трек', QUESTIONS)
        if prepare:
            temporary = marker.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'profile': PROFILE, 'stt_provider': 'groq',
                                            'laya': laya_model, 'laya_revision': revision}), encoding='utf-8')
            temporary.replace(marker)
        return None, None, agent, wake_detector
    from faster_whisper import WhisperModel
    import numpy as np

    class CommandWhisper(WhisperModel):
        # Heavy models retain standard context: shorter windows can repeat
        # words. Light commands pad short recordings to fifteen seconds and grow
        # the window for longer recordings; wakes use their own window.
        encoder_frames = 1500 if PROFILE == 'light' else 3000
        encoder_cache = None

        def detect_language(self, audio=None, features=None, **kwargs):
            # faster-whisper decodes shape[-1] - 1 content frames but language
            # detection includes the final boundary frame. Align that boundary
            # for a short command so both stages can reuse the exact encoding.
            if features is not None and 1 < features.shape[-1] < self.feature_extractor.nb_max_frames:
                features = features[..., :-1]
            return super().detect_language(audio=audio, features=features, **kwargs)

        def encode(self, features):
            features = features[..., :self.encoder_frames]
            # Automatic language detection and decoding can encode the same
            # window twice. Reuse it only when the feature arrays match exactly.
            if self.encoder_cache is not None:
                previous, output = self.encoder_cache
                if np.array_equal(previous, features):
                    return output
            output = super().encode(features)
            self.encoder_cache = (features.copy(), output)
            return output

        def command(self, audio, prompt=None, diagnostic=False):
            self.encoder_cache = None
            frames = int(len(audio) / 16000 * 100) + 150
            self.encoder_frames = (min(3000, max(1500, (frames + 1) // 2 * 2)) if PROFILE == 'light' else 3000) if prompt is None else min(3000, max(600, (frames + 1) // 2 * 2))
            # Wake words keep the fast greedy pass. Song/artist names benefit
            # from comparing several hypotheses, without a fixed artist list.
            segments, info = self.transcribe(audio, language=None if prompt is None else "ru",
                                          task="transcribe", beam_size=5 if prompt is None and PROFILE == 'heavy' else 1, temperature=0,
                                          condition_on_previous_text=False, vad_filter=True,
                                          initial_prompt=COMMAND_PROMPT if prompt is None else prompt,
                                          without_timestamps=True, max_new_tokens=96)
            segments = list(segments)
            accepted = [segment for segment in segments
                        if prompt is None or (segment.no_speech_prob < 0.6 and segment.avg_logprob >= -1.0)]
            text = " ".join(segment.text.strip() for segment in accepted)
            self.encoder_cache = None
            if diagnostic:
                return {"text": text, "metrics": {"vadMs": round(info.duration_after_vad * 1000),
                        "segments": len(segments), "rejectedSegments": len(segments) - len(accepted)}}
            return text
    threads = max(1, min(8, int(os.environ.get('VOICE_CPU_THREADS', '4'))))
    if PROFILE == 'heavy':
        torch.set_num_threads(threads)
    device = 'cpu' if PROFILE == 'light' else os.environ.get("VOICE_DEVICE", "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        raise ValueError("VOICE_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA PyTorch or use VOICE_DEVICE=cpu")
    compute_type = "int8_float16" if device == "cuda" else "int8"
    print(f"Voice inference: {device}, Whisper {compute_type}", file=sys.stderr, flush=True)
    marker = RUNTIME / "ready.json"
    manifest = {} if prepare else json.loads(marker.read_text(encoding="utf-8"))
    name, wake_name = whisper_names(manifest, prepare)
    custom_whisper = light_whisper_path(manifest, prepare)
    def load_whisper(model_name):
        model = CommandWhisper(str(custom_whisper) if custom_whisper else model_name, device=device, compute_type=compute_type, cpu_threads=threads,
                               download_root=str(WHISPER_CACHE), local_files_only=not prepare)
        model.runtime_name = model_name
        return model
    whisper = load_whisper(name)
    wake_whisper = whisper if name == wake_name else load_whisper(wake_name)
    agent, laya_model, revision = None, None, None
    if PROFILE == 'heavy':
        laya_model, revision = laya_source(manifest, prepare)
        if prepare and not Path(laya_model).is_dir():
            from huggingface_hub import model_info
            revision = model_info(laya_model).sha
        agent = laya.load(laya_model, device=device, revision=revision)
        agent._voice_dll_handles = dll_handles
        print(f'Laya inference: {agent.device}', file=sys.stderr, flush=True)
    # Warm encoder/decoder and VAD as well as Laya before the first real wake.
    for model in (whisper,) if whisper is wake_whisper else (whisper, wake_whisper):
        model.encoder_frames = (1500 if PROFILE == 'light' else 3000) if model is whisper else 600
        warm_segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="ru",
                                           beam_size=1, temperature=0, without_timestamps=True,
                                           max_new_tokens=8, condition_on_previous_text=False)
        list(warm_segments)
    whisper.command(np.zeros(16000, dtype=np.float32))
    wake_whisper.command(np.zeros(16000, dtype=np.float32), "")
    if agent is not None:
        agent.predict("Следующий трек", QUESTIONS)
    if prepare:
        temporary = marker.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({"profile": PROFILE, "whisper": name, "whisper_wake": wake_name,
                                        "whisper_path": str(custom_whisper.relative_to(ROOT)) if custom_whisper and custom_whisper.is_relative_to(ROOT) else str(custom_whisper) if custom_whisper else None,
                                        "laya": laya_model, "laya_revision": revision}), encoding="utf-8")
        temporary.replace(marker)
    return whisper, wake_whisper, agent, wake_detector


def main():
    prepare = "--prepare" in sys.argv
    with redirect_stdout(sys.stderr):
        whisper, wake_whisper, agent, wake_detector = models(prepare)
    if prepare:
        return
    wake_model_name = (wake_detector.config.get("recipe_version", "bot-v5-acc90") if wake_detector
                       else (wake_whisper.runtime_name if wake_whisper else 'whisper-large-v3-turbo'))
    print(json.dumps({"ready": True, "modelName": agent.cfg.get("model_name", "laya-multilingual") if agent else 'rules',
                      "sttModelName": whisper.runtime_name if whisper else 'whisper-large-v3-turbo',
                      "wakeModelName": wake_model_name,
                      "inferenceDevice": whisper.model.device if whisper else (str(agent.device).split(':')[0] if agent else 'cpu')}), flush=True)
    import numpy as np
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 700_000:
                raise ValueError("Request too large")
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if request["op"] == "transcribe":
                    if whisper is None:
                        raise ValueError('STT is handled by Groq in Node')
                    pcm = base64.b64decode(request["pcm"], validate=True)
                    if not pcm or len(pcm) > 480_000 or len(pcm) % 2:
                        raise ValueError("Invalid audio")
                    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    wake_name = request.get("wakeName")
                    # A single-name instruction biases noise toward that name.
                    # A contrastive vocabulary helps distinguish short words.
                    prompt = WAKE_VOCABULARY.get(wake_name.strip().lower(), "") if isinstance(wake_name, str) and len(wake_name) <= 64 else None
                    model = whisper if prompt is None else wake_whisper
                    result = model.command(audio, prompt, diagnostic=True)
                elif request["op"] == "wake_detect":
                    pcm = base64.b64decode(request["pcm"], validate=True)
                    if not pcm or len(pcm) > 480_000 or len(pcm) % 2:
                        raise ValueError("Invalid audio")
                    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    if wake_detector is not None:
                        threshold = request.get("threshold")
                        if isinstance(threshold, (int, float)):
                            prob = wake_detector.score(audio)
                            result = {"wake": prob >= float(threshold), "probability": prob}
                        else:
                            res = wake_detector.detect(audio)
                            result = {"wake": bool(res["wake"]), "probability": float(res["probability"])}
                    else:
                        result = {"wake": False, "probability": 0.0}
                elif agent is None:
                    raise ValueError('Light worker accepts only audio; actions are parsed in TypeScript')
                elif request["op"] == "classify":
                    text = request["text"]
                    if not isinstance(text, str) or len(text) > 1000:
                        raise ValueError("Invalid text")
                    result = classify(agent, text)
                elif request["op"] == "music_route":
                    result = music_route(agent, request["state"])
                elif request["op"] == "music_rerank":
                    result = music_rerank(agent, request["query"], request["candidates"])
                elif request["op"] == "music_policy":
                    result = music_result_policy(agent, request["query"], request["candidates"])
                else:
                    raise ValueError("Unknown operation")
            print(json.dumps({"id": request["id"], "result": result}, ensure_ascii=False), flush=True)
        except Exception:
            # No exception details: they can contain audio, text or prompts.
            print(json.dumps({"id": request.get("id"), "error": True}), flush=True)


if __name__ == "__main__":
    main()
