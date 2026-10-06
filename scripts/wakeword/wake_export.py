"""Freeze calibration threshold, evaluate untouched test, check CPU ONNX parity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from wake_audio import SAMPLE_RATE, N_SAMPLES, N_MELS, N_FFT, HOP, make_frontend, read_wav
from wake_model import make_model, ProbabilityModel, EnsembleLogits
from wake_train import FeatureDataset, evaluate, confusion
from wake_data import atomic_json
from wake_metrics import calibrate as calibrate, operating_point


def export(config):
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(config.get('cpu_threads', 2))
    work = Path(config['work_dir'])
    members = config.get('ensemble_members', 1)
    models, bests = [], []
    for member in range(members):
        directory = work / 'checkpoints' / f'member-{member}' if members > 1 else work / 'checkpoints'
        best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
        model = make_model(config)
        model.load_state_dict(best['model'])
        models.append(model.eval())
        bests.append(best)
    selected_members, selection_report = list(range(members)), []
    if members > 1:
        validation = FeatureDataset(work / 'dataset', 'validation')
        candidates = [([i], model) for i, model in enumerate(models)]
        if config.get('allow_ensemble', True):
            candidates += [(list(range(members)), EnsembleLogits(models).eval())]
        highest = -float('inf')
        for indices, candidate in candidates:
            result = evaluate(candidate, DataLoader(validation, batch_size=128), torch.device('cpu'))
            point = operating_point(result, validation.rows, config['target_fpr'])
            score = point['macro_recall'] + .25 * point['worst_language_recall'] - .05 * result['loss']
            selection_report.append(dict(members=indices, score=score, validation=point))
            if score > highest:
                highest, selected_members = score, indices
        atomic_json(work / 'ensemble_selection.json', dict(selected_members=selected_members, candidates=selection_report,
                    decision_data='validation only; calibration/test not used'))
    chosen_models = [models[i] for i in selected_members]
    model = chosen_models[0] if len(chosen_models) == 1 else EnsembleLogits(chosen_models).eval()
    predictions = {}
    for split in ['calibration', 'test']:
        dataset = FeatureDataset(work / 'dataset', split)
        predictions[split] = (dataset, evaluate(model, DataLoader(dataset, batch_size=128, num_workers=0), torch.device('cpu')))
    test_dataset, test_predictions = predictions['test']
    smoke = config.get('smoke', False)
    output = work / ('wake-model-SMOKE-ONLY' if smoke else 'wake-model')
    output.mkdir(parents=True, exist_ok=True)
    probability_model = ProbabilityModel(model).eval()
    probe = torch.randn(3, *test_dataset[0][0].shape)
    onnx_file = output / 'wake_model.onnx'
    torch.onnx.export(probability_model, (probe,), str(onnx_file), input_names=['log_mel'], output_names=['probability'],
                      dynamic_axes={'log_mel': {0: 'batch'}, 'probability': {0: 'batch'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(onnx_file))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_file), options, providers=['CPUExecutionProvider'])
    covered, validation_rows = set(), []
    for row in test_dataset.rows:
        group = (row['generator'], row['language'], row['label'])
        if group not in covered:
            covered.add(group); validation_rows.append(row)
    frontend = make_frontend(config, work / ('whisper-encoder' if config.get('architecture') == 'whisper_encoder_v4' else 'speech-embedding'))
    tensors = np.stack([frontend(read_wav(work / 'dataset' / row['wav'])) for row in validation_rows])
    cached_tensors = np.stack([np.load(work / 'dataset' / row['feature']) for row in validation_rows])
    np.testing.assert_allclose(tensors, cached_tensors, rtol=1e-5, atol=1e-6)
    with torch.inference_mode(): expected = probability_model(torch.from_numpy(tensors)).numpy()
    actual = session.run(['probability'], {'log_mel': tensors})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    # Calibrate the actual exported runtime, using the same FP32 frontend as inference.
    for dataset, prediction in predictions.values():
        scores = []
        for feature, _ in DataLoader(dataset, batch_size=128, num_workers=0):
            scores.extend(session.run(['probability'], {'log_mel': feature.numpy()})[0].tolist())
        np.testing.assert_allclose(scores, prediction['scores'], rtol=1e-4, atol=1e-5)
        prediction['scores'] = scores
    calibration = predictions['calibration'][1]
    calibration_languages = [row['language'] for row in predictions['calibration'][0].rows] if config.get('balanced_sampling') else None
    threshold, calibration_metrics = calibrate(calibration['labels'], calibration['scores'], config['target_fpr'], calibration_languages)
    test_metrics = confusion(test_predictions['labels'], test_predictions['scores'], threshold)
    if config.get('export_pytorch_weights', True):
        torch.save(dict(architecture=config.get('architecture', 'legacy'), selected_members=selected_members,
                       models=[m.state_dict() for m in chosen_models]), output / 'wake_model_state.pt')
    for file in ['wake_audio.py', 'wake_model.py']:
        shutil.copy2(Path(__file__).parent / file, output / file)
    metadata = {'schema_version': 1, 'wake_word_ru': config['wake_word_ru'], 'wake_word_en': config['wake_word_en'] if config['use_english'] else None,
                'task': 'isolated_wake_word' if config['standalone_only'] else 'keyword_presence_in_window',
                'standalone_only': config['standalone_only'], 'threshold': threshold, 'sample_rate': SAMPLE_RATE,
                'window_samples': N_SAMPLES, 'input': 'log_mel float32 [batch,1,40,201]', 'output': 'probability float32 [batch]',
                'frontend': {'n_fft': N_FFT, 'hop': HOP, 'n_mels': N_MELS, 'fmin': 50, 'fmax': 7600,
                             'window': 'periodic Hann', 'center_pad': 'zero', 'normalization': 'per window mean/std'},
                'smoke_only': smoke, 'quality_validated_on_real_users': False,
                'threshold_allows_any_detection': threshold <= 1,
                'best_epoch': bests[selected_members[0]]['epoch'], 'best_epochs': [b['epoch'] for b in bests],
                'architecture': config.get('architecture', 'legacy'), 'selected_members': selected_members,
                'calibration_fpr_constraints': 'overall + Russian + English' if calibration_languages else 'overall',
                'model_sha256': hashlib.sha256(onnx_file.read_bytes()).hexdigest()}
    if config.get('architecture') == 'speech_embedding_v3':
        metadata['input'] = 'log_mel (legacy name): speech embeddings float32 [batch,1,96,16]'
        metadata['frontend'] = dict(type='google_speech_embedding_oww_v0.5.1', pcm_scale=32768,
                                    mel_transform='mel / 10 + 2', patch_frames=76, patch_step=8, full_window=True)
        metadata['frontend_sha256'] = {}
        for name in ['melspectrogram.onnx', 'embedding_model.onnx', 'provenance.json']:
            shutil.copy2(work / 'speech-embedding' / name, output / name)
            metadata['frontend_sha256'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    if config.get('architecture') == 'whisper_encoder_v4':
        metadata['input'] = 'log_mel float32 [batch,1,80,200]'
        metadata['frontend'] = dict(type='whisper_log_mel_2s', n_fft=400, hop=160, n_mels=80,
                                    center_pad='reflect', normalization='log10 dynamic range /4 +1', encoder_positions=100)
        metadata['frontend_sha256'] = {}
        for name in ['whisper_mel_filters.npy', 'provenance.json']:
            shutil.copy2(work / 'whisper-encoder' / name, output / name)
            metadata['frontend_sha256'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    atomic_json(output / 'wake_config.json', metadata)
    by_group = {}
    for field in ['language', 'generator', 'speaker', *(['kind'] if 'kind' in test_dataset.rows[0] else [])]:
        by_group[field] = {}
        for value in sorted({row[field] for row in test_dataset.rows}):
            indices = [i for i, row in enumerate(test_dataset.rows) if row[field] == value]
            by_group[field][value] = confusion([test_predictions['labels'][i] for i in indices],
                                              [test_predictions['scores'][i] for i in indices], threshold)
    # Windows overlap/correlate; these figures are NOT false activations per real-world hour.
    report = {'source': 'synthetic neural TTS + procedural noise', 'calibration_threshold': threshold,
              'target_fpr_per_window': config['target_fpr'], 'calibration': calibration_metrics, 'test': test_metrics,
              'test_groups': by_group, 'onnx_max_absolute_difference': float(np.max(np.abs(actual - expected))),
              'real_audio_evaluation': 'real negative words only' if any(r['generator'] == 'real_speech' for r in test_dataset.rows) else 'not performed',
              'smoke_only': smoke,
              'interpretation': 'Speaker-held-out windows; augmentation siblings are correlated. No real-world FAR/hour claim.',
              'architecture': config.get('architecture', 'legacy'), 'selected_members': selected_members,
              'ensemble_selection': selection_report, 'real_wake_positive_evaluation': 'not performed'}
    human_indices = [i for i, row in enumerate(test_dataset.rows) if row['generator'] == 'human_reviewed']
    if human_indices:
        report['reviewed_human_test'] = confusion([test_predictions['labels'][i] for i in human_indices],
                                                 [test_predictions['scores'][i] for i in human_indices], threshold)
        if report['reviewed_human_test']['positive_windows']:
            report['real_wake_positive_evaluation'] = 'reviewed human test recordings; see reviewed_human_test counts'
    if config.get('balanced_sampling'):
        report['source'] = 'Four local TTS families + procedural noise + optional real negative speech'
        report['threshold_profiles'] = []
        for target in [.002, .005, .01, .02]:
            boundary, metrics = calibrate(calibration['labels'], calibration['scores'], target, calibration_languages)
            report['threshold_profiles'].append(dict(target_fpr=target, threshold=boundary, calibration=metrics,
                test=confusion(test_predictions['labels'], test_predictions['scores'], boundary)))
        report['profiles_interpretation'] = 'All FPR targets fixed before evaluation. Primary target remains run_config.target_fpr; do not select using test.'
    # Publish auditable test scores, not just aggregate accuracy.
    records = [{'id': row['id'], 'speaker': row['speaker'], 'language': row['language'], 'label': row['label'],
                'generator': row['generator'], 'kind': row.get('kind'), 'text': row.get('text'),
                'probability': score, 'predicted_wake': score >= threshold} for row, score in zip(test_dataset.rows, test_predictions['scores'], strict=True)]
    atomic_json(work / 'test_predictions.json', records)
    atomic_json(output / 'test_predictions.json', records)
    session.run(['probability'], {'log_mel': tensors[:1]})
    started = time.perf_counter()
    for _ in range(100): session.run(['probability'], {'log_mel': tensors[:1]})
    report['cpu_onnx_ms_per_window_excluding_frontend'] = (time.perf_counter() - started) * 10
    if config.get('architecture') in ['speech_embedding_v3', 'whisper_encoder_v4']:
        started = time.perf_counter()
        audio = read_wav(work / 'dataset' / validation_rows[0]['wav'])
        for _ in range(20): session.run(['probability'], {'log_mel': frontend(audio)[None]})
        report['cpu_ms_per_window_including_pretrained_frontend'] = (time.perf_counter()-started) * 50
    atomic_json(output / 'training_report.json', report)
    if members > 1:
        histories = []
        for i in range(members):
            histories += [dict(member=i, **row) for row in json.loads((work / f'training_history-member-{i}.json').read_text(encoding='utf-8'))]
        atomic_json(work / 'training_history.json', histories)
    for file in ['training_history.json', 'training_run.json', 'dataset_summary.json', 'tts_provenance.json', 'run_config.json',
                 'ensemble_selection.json', 'real_provenance.json', 'dataset_groups.json']:
        if not (work / file).exists(): continue
        shutil.copy2(work / file, output / file)
    for file in work.glob('source_audit-*.json'): shutil.copy2(file, output / file.name)
    (output / 'README.txt').write_text(
        'Binary acoustic wake-word detector. Mono PCM16/float32, 16000 Hz, 2-second window.\n'
        'pip install numpy onnxruntime\npython wake_audio.py . example.wav\n'
        'This training uses synthetic speech. Measure recall and false alarms on real microphones before deployment.\n'
        'The bot integration is separate; this export does not modify the Discord bot.\n'
        'TTS sources and licence references are in tts_provenance.json.\n', encoding='utf-8')
    archive = shutil.make_archive(str(output), 'zip', root_dir=output)
    print(json.dumps({'archive': archive, 'test': test_metrics, 'threshold': threshold,
                      'onnx_parity': report['onnx_max_absolute_difference'], 'smoke_only': smoke}, indent=2), flush=True)
    return output, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    export(json.loads(Path(args.config).read_text(encoding='utf-8')))
