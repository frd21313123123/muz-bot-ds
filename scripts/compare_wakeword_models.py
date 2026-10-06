"""Compare fixed exported models on identical held-out data; recalibrate without test tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/wake-validation'), str(ROOT / 'scripts/wakeword')]
import numpy as np
import onnxruntime as ort
from wake_metrics import calibrate, confusion
from wake_audio import make_frontend, read_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-fpr', type=float, default=.005)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    selections = {split: [r for r in rows if r['split'] == split] for split in ['calibration', 'test']}
    configs = {name: json.loads((Path(directory) / 'wake_config.json').read_text(encoding='utf-8'))
               for name, directory in [('baseline', args.baseline), ('candidate', args.candidate)]}
    for field in ['sample_rate', 'window_samples', 'wake_word_ru', 'wake_word_en', 'standalone_only']:
        if configs['baseline'].get(field) != configs['candidate'].get(field):
            raise ValueError(f'Incompatible comparison: {field} differs')
    report = dict(protocol='Same held-out WAVs, each exported frontend; FPR constraints overall + ru + en; thresholds calibrated separately, never using test.',
                  target_fpr=args.target_fpr, real_positive_evaluation='not performed',
                  test_source_groups=len({r['source_id'] for r in selections['test']}),
                  test_speaker_groups=len({r.get('speaker_group', r['speaker']) for r in selections['test']}), models={})
    for name, model_directory in [('baseline', args.baseline), ('candidate', args.candidate)]:
        model_directory = Path(model_directory)
        config = configs[name]
        if hashlib.sha256((model_directory / 'wake_model.onnx').read_bytes()).hexdigest() != config['model_sha256']:
            raise ValueError(f'Model hash mismatch: {name}')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_directory / 'wake_model.onnx'), options, providers=['CPUExecutionProvider'])
        frontend = make_frontend(config, model_directory)
        predictions = {}
        for split, examples in selections.items():
            scores = []
            for start in range(0, len(examples), 64):
                features = np.stack([frontend(read_wav(dataset / r['wav'])) for r in examples[start:start+64]])
                scores.extend(session.run(['probability'], {'log_mel': features})[0].tolist())
            predictions[split] = np.array(scores)
        calibration = selections['calibration']; test = selections['test']
        threshold, cal = calibrate([r['label'] for r in calibration], predictions['calibration'], args.target_fpr,
                                   [r['language'] for r in calibration])
        labels = np.array([r['label'] for r in test]); languages = np.array([r['language'] for r in test])
        scores = predictions['test']
        report['models'][name] = dict(model_sha256=config['model_sha256'], threshold=threshold, calibration=cal,
            test=confusion(labels, scores, threshold), original_threshold=config['threshold'],
            original_threshold_test=confusion(labels, scores, config['threshold']),
            by_language={language: confusion(labels[languages == language], scores[languages == language], threshold)
                         for language in ['ru', 'en', 'none'] if np.any(languages == language)})
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
