"""Four local neural TTS families. Voice identities stay in one split across versions."""
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import xml.sax.saxutils
import numpy as np
import torch
from scipy.signal import resample_poly

SILERO_RU = {'train': ['eugene', 'kseniya'], 'validation': ['baya'], 'calibration': ['xenia'], 'test': ['aidar']}
SILERO_EN = {'train': [f'en_{i}' for i in range(16)], 'validation': ['en_16', 'en_17'],
             'calibration': ['en_18', 'en_19'], 'test': [f'en_{i}' for i in range(20, 24)]}
PIPER_RU = {'train': ['ru_RU-dmitri-medium', 'ru_RU-ruslan-medium', 'ru_RU-irina-medium'], 'validation': [],
            'calibration': [], 'test': ['ru_RU-denis-medium']}
PIPER_EN = {'train': ['en_US-amy-medium', 'en_US-lessac-medium', 'en_US-ryan-medium'],
            'validation': ['en_US-hfc_female-medium'], 'calibration': ['en_US-joe-medium'],
            'test': ['en_US-hfc_male-medium']}
KOKORO_EN = {'train': ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_aoede', 'af_kore', 'af_nova',
                       'am_fenrir', 'am_michael', 'am_puck', 'am_echo', 'am_onyx', 'bf_emma', 'bm_fable'],
             'validation': ['af_alloy', 'bm_george'], 'calibration': ['af_sky', 'bf_isabella'],
             'test': ['af_jessica', 'bm_lewis']}
SILERO_MODELS = {'v5_5_ru': 'https://models.silero.ai/models/tts/ru/v5_5_ru.pt',
                 'v4_ru': 'https://models.silero.ai/models/tts/ru/v4_ru.pt',
                 'v3_en': 'https://models.silero.ai/models/tts/en/v3_en.pt'}
KOKORO_BASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/'
PIPER_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/'


def file_sha256(file):
    digest = hashlib.sha256()
    with Path(file).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def voice_jobs(config):
    jobs = []
    def add(engine, model, language, voices):
        if language == 'en' and not config['use_english']: return
        if engine not in config.get('tts_engines', ['silero', 'piper', 'mms', 'kokoro']): return
        for split, speakers in voices.items():
            for speaker in speakers:
                # Different Silero versions are renderings of the same voice, not new speakers.
                jobs.append(dict(engine=engine, model=model, language=language, voice=speaker, split=split,
                                 speaker_group=f'{engine}/{language}/{speaker}'))
    silero_ru, piper_ru = SILERO_RU, PIPER_RU
    if config.get('gender_balanced_ru') is False:
        silero_ru = dict(train=['eugene', 'kseniya', 'baya'], validation=[], calibration=['xenia'], test=['aidar'])
        piper_ru = dict(train=['ru_RU-dmitri-medium', 'ru_RU-ruslan-medium'], validation=['ru_RU-irina-medium'], calibration=[], test=['ru_RU-denis-medium'])
    if config.get('piper_ru_splits') is not None:
        proposed = config['piper_ru_splits']
        known = {voice for voices in PIPER_RU.values() for voice in voices}
        if set(proposed) != {'train', 'validation', 'calibration', 'test'}:
            raise ValueError('Piper RU override requires all four splits')
        seen = set()
        for split, voices in proposed.items():
            if not isinstance(voices, list) or not voices:
                raise ValueError('Each Piper RU split requires at least one voice')
            for voice in voices:
                if voice not in known or voice in seen:
                    raise ValueError('Piper RU override contains an unknown or repeated voice')
                seen.add(voice)
        piper_ru = proposed
    for model in config.get('silero_ru_versions', ['v5_5_ru', 'v4_ru']): add('silero', model, 'ru', silero_ru)
    add('silero', 'v3_en', 'en', SILERO_EN)
    add('piper', 'piper', 'ru', piper_ru)
    add('piper', 'piper', 'en', PIPER_EN)
    add('kokoro', 'kokoro-v1.0', 'en', KOKORO_EN)
    add('mms', 'facebook/mms-tts-rus', 'ru', {'train': ['rus']})
    add('mms', 'facebook/mms-tts-eng', 'en', {'train': ['eng']})
    return jobs


def piper_relative(voice):
    locale, speaker, quality = voice.split('-')
    return f'{locale[:2]}/{locale}/{speaker}/{quality}/{voice}.onnx'


def download(url, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        temporary = file.with_suffix(file.suffix + '.download')
        with urllib.request.urlopen(url, timeout=240) as response, temporary.open('wb') as output:
            import shutil
            shutil.copyfileobj(response, output)
        temporary.replace(file)
    return dict(url=url, sha256=hashlib.sha256(file.read_bytes()).hexdigest(), bytes=file.stat().st_size)


def prepare_embedding_models(work):
    from wake_data import atomic_json
    directory = Path(work) / 'speech-embedding'
    provenance = {name: download('https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/' + name,
                                directory / name) for name in ['melspectrogram.onnx', 'embedding_model.onnx']}
    expected = {'melspectrogram.onnx': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f',
                'embedding_model.onnx': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'}
    for name, checksum in expected.items():
        if provenance[name]['sha256'] != checksum:
            raise ValueError(f'Pretrained frontend weights changed: {name}')
    provenance['reference'] = 'https://github.com/dscripka/openWakeWord#model-architecture'
    provenance['licence_note'] = 'Upstream describes Google speech_embedding as Apache-2.0; distributed pre-trained models have CC-BY-NC-SA terms. Keep upstream licence attribution; no commercial licence guarantee.'
    atomic_json(directory / 'provenance.json', provenance)
    return provenance


def prepare_whisper_encoder(work, repository='openai/whisper-tiny', revision=None):
    from huggingface_hub import snapshot_download, HfApi
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    from safetensors import safe_open
    from wake_data import atomic_json
    directory = Path(work) / 'whisper-encoder'
    directory.mkdir(parents=True, exist_ok=True)
    provenance_file = directory / 'provenance.json'
    previous = json.loads(provenance_file.read_text(encoding='utf-8')) if provenance_file.exists() else None
    if previous and previous['repository'] != repository:
        raise ValueError('Encoder directory belongs to another model')
    revision = revision or (previous['revision'] if previous else (
        '169d4a4341b33bc18d8881c4b69c2e104e1cc0af' if repository == 'openai/whisper-tiny' else HfApi().model_info(repository).sha))
    if previous and previous['revision'] != revision:
        raise ValueError('Encoder revision changed; use a new directory')
    if not all((directory / name).exists() for name in ['encoder_config.json', 'encoder_state.pt', 'whisper_mel_filters.npy']):
        source = directory / 'source'
        snapshot_download(repo_id=repository, revision=revision, local_dir=source,
                          allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
        encoder_config = WhisperConfig.from_pretrained(source, local_files_only=True)
        encoder_config._attn_implementation = 'eager'
        encoder = WhisperEncoder(encoder_config)
        # Read only encoder tensors; never instantiate the much larger decoder.
        with safe_open(str(source / 'model.safetensors'), framework='pt', device='cpu') as saved:
            state = {key.removeprefix('model.encoder.'): saved.get_tensor(key)
                     for key in saved.keys() if key.startswith('model.encoder.')}
        if not state:
            raise ValueError('No encoder tensors in Whisper checkpoint')
        encoder.load_state_dict(state)
        del state
        encoder.config.max_source_positions = 100
        encoder.max_source_positions = 100
        encoder.embed_positions = torch.nn.Embedding.from_pretrained(encoder.embed_positions.weight[:100].detach().clone(), freeze=True)
        encoder.config.to_json_file(str(directory / 'encoder_config.json'))
        torch.save(encoder.state_dict(), directory / 'encoder_state.pt')
        extractor = WhisperFeatureExtractor.from_pretrained(source, local_files_only=True)
        np.save(directory / 'whisper_mel_filters.npy', extractor.mel_filters.T.astype(np.float32))
    provenance = dict(repository=repository, revision=revision, window_adaptation='encoder positions truncated to 100 for 2-second audio',
                      license_reference=f'https://huggingface.co/{repository}',
                      files_sha256={name: file_sha256(directory / name)
                                    for name in ['encoder_config.json', 'encoder_state.pt', 'whisper_mel_filters.npy']})
    atomic_json(directory / 'provenance.json', provenance)
    return provenance


def download_models(work, config):
    if config.get('architecture') == 'speech_embedding_v3':
        prepare_embedding_models(work)
    if config.get('architecture') == 'whisper_encoder_v4':
        prepare_whisper_encoder(work)
    work = Path(work)
    provenance = {}
    jobs = voice_jobs(config)
    for model in sorted({j['model'] for j in jobs if j['engine'] == 'silero'}):
        provenance[model] = download(SILERO_MODELS[model], work / 'tts' / (model + '.pt'))
        provenance[model]['license_reference'] = 'https://github.com/snakers4/silero-models#licence'
    for voice in sorted({j['voice'] for j in jobs if j['engine'] == 'piper'}):
        relative = piper_relative(voice)
        provenance[voice] = download(PIPER_BASE + relative, work / 'tts/piper' / (voice + '.onnx'))
        provenance[voice]['config'] = download(PIPER_BASE + relative + '.json', work / 'tts/piper' / (voice + '.onnx.json'))
        card = relative.rsplit('/', 1)[0] + '/MODEL_CARD'
        provenance[voice]['license_reference'] = PIPER_BASE + card
    if any(j['engine'] == 'kokoro' for j in jobs):
        for file in ['kokoro-v1.0.onnx', 'voices-v1.0.bin']:
            provenance[file] = download(KOKORO_BASE + file, work / 'tts/kokoro' / file)
            provenance[file]['license_reference'] = 'https://huggingface.co/hexgrad/Kokoro-82M'
    if any(j['engine'] == 'mms' for j in jobs):
        from huggingface_hub import snapshot_download
        for repo in sorted({j['model'] for j in jobs if j['engine'] == 'mms'}):
            local = work / 'tts/mms' / repo.split('/')[-1]
            snapshot_download(repo_id=repo, local_dir=local,
                              allow_patterns=['*.json', '*.safetensors', 'pytorch_model.bin', '*.txt'])
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in local.iterdir() if p.is_file()}
            provenance[repo] = dict(repository=repo, files_sha256=hashes,
                                    license_reference=f'https://huggingface.co/{repo}')
    file = work / 'tts_provenance.json'
    file.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance


def resample(audio, sample_rate):
    common = math.gcd(16000, int(sample_rate))
    return resample_poly(np.asarray(audio, dtype=np.float32).reshape(-1), 16000 // common,
                         int(sample_rate) // common).astype(np.float32)


class Synthesizer:
    def __init__(self, work, job, device, threads=2):
        self.work, self.job, self.device = Path(work), job, device
        self.engine = job['engine']
        if self.engine == 'silero':
            self.model = torch.package.PackageImporter(str(self.work / 'tts' / (job['model'] + '.pt'))).load_pickle('tts_models', 'model')
            self.model.to(torch.device(device))
            self.provider = device
        elif self.engine == 'piper':
            import onnxruntime as ort
            from piper import PiperVoice
            from piper.config import PiperConfig
            file = self.work / 'tts/piper' / (job['voice'] + '.onnx')
            options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(file), options, providers=['CPUExecutionProvider'])
            self.model = PiperVoice(session=session, config=PiperConfig.from_dict(json.loads(Path(str(file) + '.json').read_text(encoding='utf-8'))))
            self.provider = 'CPUExecutionProvider'
        elif self.engine == 'kokoro':
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(self.work / 'tts/kokoro/kokoro-v1.0.onnx'), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            self.model = Kokoro.from_session(session, str(self.work / 'tts/kokoro/voices-v1.0.bin'))
            self.provider = 'CPUExecutionProvider'
        elif self.engine == 'mms':
            from transformers import VitsModel, VitsTokenizer
            local = self.work / 'tts/mms' / job['model'].split('/')[-1]
            self.tokenizer = VitsTokenizer.from_pretrained(local)
            self.model = VitsModel.from_pretrained(local).to(device).eval()
            self.provider = device
        else: raise ValueError(f'Unknown TTS engine: {self.engine}')

    def say(self, text, rate='normal', seed=0):
        speed = {'normal': 1., 'slow': .85, 'fast': 1.15}[rate]
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        if self.engine == 'silero':
            args = {'text': text} if rate == 'normal' else {'ssml_text':
                f'<speak><prosody rate="{rate}">{xml.sax.saxutils.escape(text)}</prosody></speak>'}
            with torch.inference_mode():
                audio = self.model.apply_tts(**args, speaker=self.job['voice'], sample_rate=24000).cpu().numpy()
            sample_rate = 24000
        elif self.engine == 'piper':
            from piper import SynthesisConfig
            chunks = list(self.model.synthesize(text, SynthesisConfig(length_scale=1 / speed, noise_scale=.667, noise_w_scale=.8)))
            if not chunks: raise ValueError('Empty Piper result')
            audio = np.concatenate([c.audio_float_array for c in chunks])
            sample_rate = chunks[0].sample_rate
        elif self.engine == 'kokoro':
            # 0.4.9 assumes int32 speed for input_ids exports; the released graph
            # uses float32. Build inputs from actual graph metadata instead.
            language = 'en-gb' if self.job['voice'].startswith('b') else 'en-us'
            phonemes = self.model.tokenizer.phonemize(text, language)
            tokens = self.model.tokenizer.tokenize(phonemes)
            if not tokens or len(tokens) > 510: raise ValueError('Kokoro text outside complete short-utterance limit')
            style = self.model.get_voice_style(self.job['voice'])
            style = style[min(len(tokens), len(style)) - 1]
            values = {'tokens': [[0, *tokens, 0]], 'input_ids': [[0, *tokens, 0]], 'style': style, 'speed': [speed]}
            dtypes = {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(int32)': np.int32}
            inputs = {item.name: np.asarray(values[item.name], dtype=dtypes[item.type]) for item in self.model.sess.get_inputs()}
            audio = self.model.sess.run(None, inputs)[0]
            sample_rate = 24000
        else:
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            old = self.model.speaking_rate
            self.model.speaking_rate = speed
            try:
                with torch.inference_mode(): audio = self.model(**inputs).waveform.cpu().numpy().reshape(-1)
            finally: self.model.speaking_rate = old
            sample_rate = self.model.config.sampling_rate
        return resample(audio, sample_rate)
