"""Neural TTS dataset generation; separate source voices before augmentation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import xml.sax.saxutils
import numpy as np
import torch
from scipy.signal import fftconvolve, resample_poly
from wake_audio import N_SAMPLES, SAMPLE_RATE, features, write_wav

SPLITS = ('train', 'validation', 'calibration', 'test')
RU_VOICES = {'train': ['eugene', 'kseniya'], 'validation': ['baya'], 'calibration': ['xenia'], 'test': ['aidar']}
EN_VOICES = {'train': [f'en_{i}' for i in range(12)], 'validation': [f'en_{i}' for i in range(12, 16)],
             'calibration': [f'en_{i}' for i in range(16, 20)], 'test': [f'en_{i}' for i in range(20, 24)]}
TTS = {'ru': ('v5_5_ru', 'https://models.silero.ai/models/tts/ru/v5_5_ru.pt'),
       'en': ('v3_en', 'https://models.silero.ai/models/tts/en/v3_en.pt')}
HARD_RU = ['вот', 'кот', 'год', 'рот', 'борт', 'порт', 'пот', 'тот', 'код', 'ход', 'лот', 'лёд', 'мёд',
           'болт', 'байт', 'бой', 'бок', 'быт', 'бод', 'бор', 'бота', 'боту', 'боты', 'робот', 'работа',
           'забота', 'суббота', 'ботинок', 'бутон', 'компот', 'капот', 'блок', 'бокс', 'борщ']
HARD_EN = ['boat', 'boot', 'but', 'bat', 'bit', 'bet', 'bite', 'bolt', 'box', 'boy', 'body', 'both',
           'pot', 'hot', 'dot', 'not', 'robot', 'bottle', 'bottom', 'button', 'about', 'bother']
RU_SHORT = ['да', 'нет', 'привет', 'спасибо', 'пожалуйста', 'следующий', 'пауза', 'продолжай', 'громче',
            'тише', 'стоп', 'музыка', 'радио', 'добрый день', 'доброе утро', 'добрый вечер', 'как дела',
            'почему', 'что случилось', 'подожди', 'идём домой', 'включи музыку', 'смотри сюда', 'не знаю']
EN_SHORT = ['yes', 'no', 'hello', 'thanks', 'please', 'next', 'pause', 'continue', 'louder', 'stop',
            'music', 'radio', 'good morning', 'how are you', 'wait', 'come here', 'I do not know']


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def download_models(work, languages):
    provenance = {}
    for language in languages:
        name, url = TTS[language]
        destination = work / 'tts' / (name + '.pt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as output:
                import shutil
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        provenance[language] = {'name': name, 'url': url, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                                'license_reference': 'https://github.com/snakers4/silero-models#licence'}
    atomic_json(work / 'tts_provenance.json', provenance)


def negative_texts(language, target, standalone, exclusions=()):
    if language == 'ru':
        phrases = [f'{verb} {noun}' for verb in ['включи', 'выключи', 'покажи', 'найди', 'добавь', 'убери', 'проверь', 'запусти']
                   for noun in ['песню', 'радио', 'музыку', 'трек', 'альбом', 'плеер', 'очередь', 'запись']]
        context = [f'этот {target}', f'наш {target}', f'где {target}', f'{target} работает', f'{target} молчит',
                   f'это {target}', f'{target} готов', f'новый {target}'] if standalone else []
        words = HARD_RU + RU_SHORT
    else:
        phrases = [f'{verb} {noun}' for verb in ['play', 'stop', 'find', 'show', 'start', 'check', 'add', 'remove']
                   for noun in ['music', 'radio', 'song', 'track', 'album', 'player', 'queue', 'audio']]
        context = [f'this {target}', f'our {target}', f'the {target}', f'where is {target}', f'{target} works',
                   f'new {target}', f'{target} is ready', f'hello {target}'] if standalone else []
        words = HARD_EN + EN_SHORT
    target = target.strip().lower()
    # Changing the target cannot turn that exact word into a negative example.
    words = [text for text in words if target not in text.lower().split()]
    phrases = [text for text in phrases if target not in text.lower().split()]
    excluded = {text.strip().casefold() for text in exclusions}
    return sorted(text for text in set(words + phrases + context) if text.strip().casefold() not in excluded)


def background(rng, length):
    kind = int(rng.integers(5))
    if kind == 0:
        return np.zeros(length, np.float32)
    white = rng.normal(0, 1, length)
    if kind == 1:
        sound = white
    elif kind == 2:
        sound = np.cumsum(white)
        sound -= sound.mean()
    elif kind == 3:
        t = np.arange(length) / SAMPLE_RATE
        sound = sum(np.sin(2 * np.pi * rng.uniform(70, 1200) * t + rng.uniform(0, 6.28)) for _ in range(4))
        sound *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 5) * t)
    else:
        sound = white * 0.05
        for _ in range(10):
            index = int(rng.integers(length - 100))
            sound[index:index + 100] += rng.uniform(-10, 10) * np.exp(-np.arange(100) / 20)
    return (sound / (np.sqrt(np.mean(sound ** 2)) + 1e-7)).astype(np.float32)


def prepare_source(audio):
    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-4:
        raise ValueError('TTS produced empty/silent audio')
    active = np.flatnonzero(np.abs(audio) > max(0.001, np.max(np.abs(audio)) * 0.02))
    if not len(active):
        raise ValueError('No audible speech in TTS output')
    return audio[max(0, active[0] - 800):min(len(audio), active[-1] + 801)]


def augment(audio, rng, label, preserve_context=False):
    # Resampling changes pitch and tempo together; no claim of independent voice generation.
    ratio = int(rng.integers(85, 116))
    speech = resample_poly(audio, ratio, 100).astype(np.float32)
    if len(speech) > N_SAMPLES - 800:
        if label or preserve_context:
            return None  # Never crop away the wake or context while retaining its label.
        start = int(rng.integers(len(speech) - (N_SAMPLES - 800) + 1))
        speech = speech[start:start + N_SAMPLES - 800]
    if rng.random() < 0.5:
        ir_length = int(rng.integers(800, 4000))
        ir = rng.normal(0, 0.03, ir_length) * np.exp(-np.arange(ir_length) / rng.uniform(200, 1500))
        ir[0] = 1
        speech = fftconvolve(speech, ir)[:len(speech)].astype(np.float32)
    speech *= 10 ** (rng.uniform(-24, -3) / 20) / (np.max(np.abs(speech)) + 1e-6)
    window = np.zeros(N_SAMPLES, np.float32)
    start = int(rng.integers(N_SAMPLES - len(speech) + 1))
    window[start:start + len(speech)] = speech
    noise = background(rng, N_SAMPLES)
    if np.any(noise):
        level = max(float(np.sqrt(np.mean(speech ** 2))), 1e-4) / 10 ** (rng.uniform(5, 30) / 20)
        window += noise * level
    if rng.random() < 0.2:
        # Bandwidth loss approximates a low-quality microphone; not a Discord Opus simulation.
        window = resample_poly(resample_poly(window, 1, 2), 2, 1)[:N_SAMPLES]
    peak = np.max(np.abs(window))
    if peak > 0.98:
        window *= 0.98 / peak
    return window.astype(np.float32)


def voice_jobs(config):
    jobs = []
    for language, splits in [('ru', RU_VOICES), ('en', EN_VOICES)]:
        if language == 'en' and not config['use_english']:
            continue
        for split, voices in splits.items():
            for voice in voices:
                jobs.append({'language': language, 'voice': voice, 'split': split})
    return jobs


def synthesize(model, text, speaker, device, rate=None):
    if rate:
        escaped = xml.sax.saxutils.escape(text)
        kwargs = {'ssml_text': f'<speak><prosody rate="{rate}">{escaped}</prosody></speak>'}
    else:
        kwargs = {'text': text}
    with torch.inference_mode():
        audio = model.apply_tts(**kwargs, speaker=speaker, sample_rate=24000)
    audio = audio.detach().cpu().float().numpy()
    return prepare_source(resample_poly(audio, 2, 3).astype(np.float32))


def generate(config, rank=0, world=1):
    work = Path(config['work_dir'])
    torch.set_num_threads(config.get('cpu_threads', 2))
    device = f'cuda:{rank}' if torch.cuda.is_available() and not config.get('force_cpu') else 'cpu'
    if device != 'cpu':
        torch.cuda.set_device(rank)
    jobs = voice_jobs(config)[rank::world]
    smoke = config.get('smoke', False)
    destination = work / 'dataset'
    for folder in ['wav', 'features', 'sources']:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    model, current_language = None, None
    started = time.monotonic()
    for job in jobs:
        language, speaker, split = job['language'], job['voice'], job['split']
        name = TTS[language][0]
        if language != current_language:
            if model is not None:
                del model
                if device != 'cpu': torch.cuda.empty_cache()
            model = torch.package.PackageImporter(str(work / 'tts' / (name + '.pt'))).load_pickle('tts_models', 'model')
            model.to(torch.device(device))
            current_language = language
        target = config['wake_word_ru' if language == 'ru' else 'wake_word_en'].strip()
        positive_texts = [target + mark for mark in ['.', '!', '?']]
        if not config['standalone_only']:
            positive_texts += [f'{target}, привет.' if language == 'ru' else f'{target}, hello.']
        negative_text = negative_texts(language, target, config['standalone_only'])
        sources = {0: [], 1: []}
        for label, texts in [(1, positive_texts), (0, negative_text[:6] if smoke else negative_text)]:
            rates = [None, 'slow', 'fast'] if label else [None]
            for text in texts:
                for rate in rates:
                    source_id = hashlib.sha256(f'{name}|{speaker}|{text}|{rate}'.encode()).hexdigest()[:24]
                    file = destination / 'sources' / (source_id + '.wav')
                    if file.exists():
                        from wake_audio import read_wav
                        audio = read_wav(file)
                    else:
                        audio = synthesize(model, text, speaker, device, rate)
                        write_wav(file, audio)
                        from wake_audio import read_wav
                        audio = read_wav(file)  # Consistent PCM16 quantization on rerun.
                    preserve = label == 1 or target.lower() in text.lower().replace(',', '').split()
                    sources[label].append((source_id, text, audio, preserve))
        for label in [1, 0]:
            count = config['positive_per_voice'] if label else config['negative_per_voice']
            rng = np.random.default_rng(config['seed'] + stable_seed(f'{language}:{speaker}:{label}'))
            attempts, written = 0, 0
            while written < count:
                attempts += 1
                if attempts > count * 30:
                    raise RuntimeError('Too many overlong wake/context examples; use a shorter word')
                source_id, text, audio, preserve = sources[label][int(rng.integers(len(sources[label])))]
                window = augment(audio, rng, label, preserve)
                if window is None: continue
                identifier = f'{language}-{speaker}-{label}-{written:05d}'
                wav = destination / 'wav' / (identifier + '.wav')
                write_wav(wav, window)
                from wake_audio import read_wav
                feat = features(read_wav(wav))
                np.save(destination / 'features' / (identifier + '.npy'), feat)
                rows.append({'id': identifier, 'label': label, 'split': split, 'language': language,
                             'speaker': f'{name}/{speaker}', 'source_id': source_id, 'text': text,
                             'wav': f'wav/{identifier}.wav', 'feature': f'features/{identifier}.npy', 'generator': 'neural_tts'})
                written += 1
        print(f'[TTS GPU {rank}] {language}/{speaker} -> {split}: {count + config["positive_per_voice"]} windows', flush=True)
    # Every noise seed belongs to one split; both labels receive the same noise process.
    for index in range(rank, config['noise_windows'], world):
        split = SPLITS[index % 4]
        rng = np.random.default_rng(config['seed'] + 100000 + index)
        audio = background(rng, N_SAMPLES) * 10 ** (rng.uniform(-45, -12) / 20)
        audio = np.clip(audio, -0.98, 0.98)
        identifier = f'noise-{index:05d}'
        write_wav(destination / 'wav' / (identifier + '.wav'), audio)
        from wake_audio import read_wav
        np.save(destination / 'features' / (identifier + '.npy'), features(read_wav(destination / 'wav' / (identifier + '.wav'))))
        rows.append({'id': identifier, 'label': 0, 'split': split, 'language': 'none', 'speaker': 'noise',
                     'source_id': identifier, 'text': '', 'wav': f'wav/{identifier}.wav',
                     'feature': f'features/{identifier}.npy', 'generator': 'procedural_noise'})
    shard = destination / f'manifest-rank-{rank}.jsonl'
    temporary = shard.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    temporary.replace(shard)
    print(f'[TTS rank {rank}] finished {len(rows)} windows in {time.monotonic() - started:.1f}s on {device}', flush=True)


def merge_manifest(config, world):
    work = Path(config['work_dir'])
    rows = []
    for rank in range(world):
        file = work / 'dataset' / f'manifest-rank-{rank}.jsonl'
        rows += [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line]
    identifiers, group_split, speakers = set(), {}, {}
    for row in rows:
        assert row['id'] not in identifiers, 'Duplicate generated ID'
        identifiers.add(row['id'])
        assert group_split.setdefault(row['source_id'], row['split']) == row['split'], 'Source audio leakage'
        if row['speaker'] != 'noise':
            assert speakers.setdefault(row['speaker'], row['split']) == row['split'], 'Speaker leakage'
        feature = np.load(work / 'dataset' / row['feature'])
        expected_shape = {'speech_embedding_v3': (1, 96, 16), 'whisper_encoder_v4': (1, 80, 200)}.get(config.get('architecture'), (1, 40, 201))
        assert feature.shape == expected_shape and np.isfinite(feature).all()
    for split in SPLITS:
        assert {row['label'] for row in rows if row['split'] == split} == {0, 1}
    rows.sort(key=lambda row: row['id'])
    (work / 'dataset' / 'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    summary = {split: {'positive': sum(r['split'] == split and r['label'] == 1 for r in rows),
                       'negative': sum(r['split'] == split and r['label'] == 0 for r in rows),
                       'voices': len({r['speaker'] for r in rows if r['split'] == split and r['speaker'] != 'noise'})} for split in SPLITS}
    atomic_json(work / 'dataset_summary.json', summary)
    return rows, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    args = parser.parse_args()
    generate(json.loads(Path(args.config).read_text(encoding='utf-8')), args.rank, args.world)
