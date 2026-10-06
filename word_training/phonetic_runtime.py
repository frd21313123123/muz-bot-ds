"""Compact keyword detector: acoustic posteriors -> complete BOT word probability."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from wake_audio import WhisperMelFrontend,right_aligned_window


class PhoneticDetector:
    def __init__(self,directory):
        directory=Path(directory)
        self.config=json.loads((directory/'wake_config.json').read_text(encoding='utf-8'))
        self.frontend=WhisperMelFrontend(directory)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(directory/'wake_model.onnx'),options,providers=['CPUExecutionProvider'])
    def score(self,audio):
        feature=self.frontend(right_aligned_window(np.asarray(audio,dtype=np.float32))).astype(np.float32)[None]
        return float(self.session.run(['probability'],{'log_mel':feature})[0].reshape(-1)[0])
