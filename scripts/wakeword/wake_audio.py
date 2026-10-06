"""Shared NumPy frontend used for training and exported CPU inference."""
from pathlib import Path
import json
import wave
import numpy as np

SAMPLE_RATE = 16000
N_SAMPLES = 32000
N_FFT = 512
HOP = 160
N_MELS = 40
N_FRAMES = 1 + N_SAMPLES // HOP


def mel_filters():
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(mel(50), mel(7600), N_MELS + 2) / 2595) - 1)
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    lower = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    upper = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    return np.maximum(0, np.minimum(lower, upper)).astype(np.float32)


MEL = mel_filters()
HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def features(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape != (N_SAMPLES,) or not np.isfinite(audio).all():
        raise ValueError('Expected exactly 32000 finite mono float32 samples at 16 kHz')
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * HANN, axis=-1)) ** 2
    logged = np.log(np.maximum(MEL @ power.T, 1e-6)).astype(np.float32)
    normalized = (logged - logged.mean()) / max(float(logged.std()), 1e-4)
    return normalized[None].astype(np.float32)


def read_wav(file):
    with wave.open(str(file), 'rb') as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError('Expected mono PCM16 WAV at 16000 Hz')
        return np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype(np.float32) / 32768


def write_wav(file, audio):
    with wave.open(str(file), 'wb') as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(SAMPLE_RATE)
        destination.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())


def right_aligned_window(audio):
    """Runtime contract: keep the most recent 2 seconds, left-pad short phrases."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) >= N_SAMPLES:
        return audio[-N_SAMPLES:].copy()
    return np.pad(audio, (N_SAMPLES - len(audio), 0))


class WakeDetector:
    def __init__(self, directory):
        import onnxruntime as ort
        directory = Path(directory)
        self.config = json.loads((directory / 'wake_config.json').read_text(encoding='utf-8'))
        if self.config['sample_rate'] != SAMPLE_RATE or self.config['window_samples'] != N_SAMPLES:
            raise ValueError('Incompatible frontend')
        import hashlib
        for name, expected in self.config.get('frontend_sha256', {}).items():
            if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Incompatible pretrained frontend hash')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'wake_model.onnx'), options, providers=['CPUExecutionProvider'])
        self.frontend = make_frontend(self.config, directory)

    def score(self, audio):
        tensor = self.frontend(right_aligned_window(audio))[None]
        return float(self.session.run(['probability'], {'log_mel': tensor})[0].reshape(-1)[0])

    def detect(self, audio):
        probability = self.score(audio)
        return {'wake': probability >= self.config['threshold'], 'probability': probability}


class SpeechEmbeddingFrontend:
    """Frozen Google speech features via the fixed openWakeWord v0.5.1 conversion.

    Full-window inference is used for both training and runtime; streaming buffers
    have different boundary behavior and are deliberately outside this contract.
    """
    def __init__(self, directory):
        import onnxruntime as ort
        directory = Path(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(str(directory / 'melspectrogram.onnx'), options, providers=['CPUExecutionProvider'])
        self.embedding = ort.InferenceSession(str(directory / 'embedding_model.onnx'), options, providers=['CPUExecutionProvider'])

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape != (N_SAMPLES,) or not np.isfinite(audio).all():
            raise ValueError('Expected exactly 32000 finite mono samples')
        pcm = (np.clip(audio, -1, 1) * 32768).clip(-32768, 32767).astype(np.int16).astype(np.float32)[None]
        mel = self.mel.run(None, {'input': pcm})[0].reshape(-1, 32) / 10 + 2
        windows = np.stack([mel[start:start+76] for start in range(0, len(mel)-75, 8)])[..., None]
        value = self.embedding.run(None, {'input_1': windows.astype(np.float32)})[0].reshape(-1, 96)
        if value.shape != (16, 96) or not np.isfinite(value).all():
            raise ValueError(f'Unexpected embedding shape: {value.shape}')
        return value.T[None].astype(np.float32)


def make_frontend(config, directory):
    if config.get('architecture') == 'speech_embedding_v3':
        return SpeechEmbeddingFrontend(directory)
    if config.get('architecture') == 'whisper_encoder_v4':
        return WhisperMelFrontend(directory)
    return features


class WhisperMelFrontend:
    """Whisper log-mel calculation for a fixed two-second, 200-frame window."""
    def __init__(self, directory):
        self.filters = np.load(Path(directory) / 'whisper_mel_filters.npy', allow_pickle=False)
        if self.filters.shape not in [(80, 201), (128, 201)]: raise ValueError('Invalid Whisper mel filters')
        self.window = (.5 - .5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape != (N_SAMPLES,) or not np.isfinite(audio).all(): raise ValueError('Expected two seconds of finite audio')
        padded = np.pad(audio, (200, 200), mode='reflect')
        frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
        power = np.abs(np.fft.rfft(frames * self.window, axis=-1)) ** 2
        value = np.log10(np.maximum(self.filters @ power.T, 1e-10))
        value = (np.maximum(value, value.max() - 8) + 4) / 4
        return value[None].astype(np.float32)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory')
    parser.add_argument('audio_wav')
    args = parser.parse_args()
    detector = WakeDetector(args.model_directory)
    print(json.dumps(detector.detect(read_wav(args.audio_wav))))
