"""JSONL worker. Audio stays in RAM; stdout is exclusively the IPC protocol."""
import base64
from contextlib import redirect_stdout
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime" / "voice"
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

COMMAND_PROMPT = "Музыкальный поиск и управление: включи, поставь песню, хочу послушать, сыграй, воспроизведи, продолжи музыку, поставь на паузу, следующий трек, громче, тише, громкость. Название песни, имя исполнителя или описание музыки могут быть на русском или английском. Например: Монеточка, Земфира, Кино, Сплин, Баста, Би-2, Linkin Park, Numb, live, remix, cover."

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

        def command(self, audio, prompt=None, diagnostic=False):
            frames = int(len(audio) / 16000 * 100) + 150
            self.encoder_frames = min(3000, max(600, (frames + 1) // 2 * 2))
            # Wake words keep the fast greedy pass. Song/artist names benefit
            # from comparing several hypotheses, without a fixed artist list.
            segments, info = self.transcribe(audio, language="ru", beam_size=5 if prompt is None else 1, temperature=0,
                                          condition_on_previous_text=False, vad_filter=True,
                                          initial_prompt=COMMAND_PROMPT if prompt is None else prompt,
                                          without_timestamps=True, max_new_tokens=96)
            segments = list(segments)
            accepted = [segment for segment in segments
                        if prompt is None or (segment.no_speech_prob < 0.6 and segment.avg_logprob >= -1.0)]
            text = " ".join(segment.text.strip() for segment in accepted)
            if diagnostic:
                return {"text": text, "metrics": {"vadMs": round(info.duration_after_vad * 1000),
                        "segments": len(segments), "rejectedSegments": len(segments) - len(accepted)}}
            return text
    torch.set_num_threads(max(1, min(8, int(os.environ.get("VOICE_CPU_THREADS", "4")))))
    marker = RUNTIME / "ready.json"
    manifest = {} if prepare else json.loads(marker.read_text(encoding="utf-8"))
    name = os.environ.get("WHISPER_MODEL", "small") if prepare else manifest["whisper"]
    whisper = CommandWhisper(name, device="cpu", compute_type="int8", cpu_threads=torch.get_num_threads(),
                           download_root=str(RUNTIME / "whisper"), local_files_only=not prepare)
    laya_model, revision = laya_source(manifest, prepare)
    if prepare and not Path(laya_model).is_dir():
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
    print(json.dumps({"ready": True, "modelName": agent.cfg.get("model_name", "laya-multilingual")}), flush=True)
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
                    # A single-name instruction biases noise toward that name.
                    # A contrastive vocabulary helps distinguish short words.
                    prompt = WAKE_VOCABULARY.get(wake_name.strip().lower(), "") if isinstance(wake_name, str) and len(wake_name) <= 64 else None
                    result = whisper.command(audio, prompt, diagnostic=True)
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
