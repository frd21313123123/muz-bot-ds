"""Cache the deployed INT8 encoders' pooled features and logits at production batch size."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models/wake-model'))
from wake_audio import WhisperMelFrontend


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def taps(model, task):
    if any(x.name == 'adapt_w1' for x in model.graph.initializer):
        raise ValueError('Expected the original INT8 encoder. Set --base to the saved baseline directory.')
    nodes = model.graph.node
    pooled = next(n.input[0] for n in nodes if n.op_type == 'DynamicQuantizeLinear'
                  and n.input[0] == '/model/Concat_output_0')
    last_bias = 'model.head.' + ('3' if task == 'wake' else '5') + '.bias'
    logits = next(n.output[0] for n in nodes if n.op_type == 'Add' and last_bias in n.input)
    return pooled, logits


def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), opts, providers=['CPUExecutionProvider'])


def cache_features(work, base, task, rows, dataset_sha, deadline, threads):
    folder = base / (task + '-model')
    model_file = folder / (task + '_model.onnx')
    identity = [sha(model_file), sha(folder / 'whisper_mel_filters.npy'), dataset_sha, sha(work / 'windows.jsonl')]
    dest = work / (task + '-features.npz')
    if dest.exists():
        with np.load(dest, allow_pickle=False) as saved:
            if saved['identity'].tolist() == identity:
                return saved['indices'], saved['pooled'], saved['logits']
    ids = np.array([i for i, r in enumerate(rows) if r[task] is not None], dtype=np.int64)
    audio = np.load(work / 'audio.npy', mmap_mode='r')
    model = onnx.load(model_file)
    names = taps(model, task)
    for name in names:
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', None]))
    runner = session(model)
    frontend = WhisperMelFrontend(folder)
    def encode(i):
        mel = frontend(audio[i].astype(np.float32) / 32768)
        h, z = runner.run(list(names), {'log_mel': mel[None]})
        return h[0], z[0]
    features, logits = [], []
    with ThreadPoolExecutor(threads) as pool:
        for offset in range(0, len(ids), 256):
            if time.monotonic() >= deadline:
                raise TimeoutError('Feature extraction exhausted the training budget')
            for h, z in pool.map(encode, ids[offset:offset+256]):
                features.append(h)
                logits.append(z)
            if offset % 4096 == 0:
                print(task, 'encoder', offset, '/', len(ids), flush=True)
    features, logits = np.stack(features), np.stack(logits)
    if not (np.isfinite(features).all() and np.isfinite(logits).all()):
        raise ValueError('Invalid encoder output')
    temp = dest.with_suffix('.tmp.npz')
    np.savez(temp, indices=ids, pooled=features, logits=logits, identity=identity)
    temp.replace(dest)
    return ids, features, logits
