"""CommandDetector for acoustic intent classification using quantized ONNX Whisper-tiny."""
from pathlib import Path
import json
import wave
import numpy as np

SAMPLE_RATE = 16000
N_SAMPLES = 32000

from wake_audio import WhisperMelFrontend, right_aligned_window, read_wav, write_wav


class CommandDetector:
    def __init__(self, directory):
        import onnxruntime as ort
        directory = Path(directory)
        self.directory = directory
        self.config = json.loads((directory / 'command_config.json').read_text(encoding='utf-8'))
        if self.config.get('sample_rate') != SAMPLE_RATE or self.config.get('window_samples') != N_SAMPLES:
            raise ValueError('Incompatible command detector audio format')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'command_model.onnx'), options, providers=['CPUExecutionProvider'])
        self.frontend = WhisperMelFrontend(directory)
        self.classes = self.config['classes']
        self.class_to_action = self.config.get('class_to_action', {})
        self.threshold = float(self.config.get('default_threshold', 0.85))

    def predict(self, audio):
        window = right_aligned_window(audio)
        tensor = self.frontend(window)[None]
        probs = self.session.run(['probabilities'], {'log_mel': tensor})[0][0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        cls_name = self.classes[idx]
        action = self.class_to_action.get(cls_name, cls_name)
        return {
            'class': cls_name,
            'action': action,
            'confidence': confidence,
            'matched': confidence >= self.threshold,
            'probabilities': {c: float(p) for c, p in zip(self.classes, probs)}
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory')
    parser.add_argument('audio_wav')
    args = parser.parse_args()
    detector = CommandDetector(args.model_directory)
    print(json.dumps(detector.predict(read_wav(args.audio_wav)), ensure_ascii=False, indent=2))
