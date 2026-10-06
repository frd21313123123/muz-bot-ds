"""Quantize, calibrate each runtime, evaluate once, publish only allowed small artifacts."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from torch.utils.data import DataLoader
from wake_data import atomic_json
from wake_tts import file_sha256
from wake_train import FeatureDataset, confusion
from wake_metrics import calibrate, point_for_config, selection_score, source_group, threshold_audit

RUNTIME_FILES = ['wake_model.onnx', 'wake_audio.py', 'wake_config.json', 'whisper_mel_filters.npy', 'provenance.json', 'README.txt']


def export_student(config):
    """Validation-only selection/export: calibration and test remain unopened."""
    import torch
    import onnx
    import onnxruntime as ort
    from wake_train import evaluate
    from wake_model import make_model, ProbabilityModel
    from wake_audio import WhisperMelFrontend, read_wav
    torch.set_num_threads(config.get('cpu_threads', 2))
    work = Path(config['work_dir'])
    dataset = FeatureDataset(work / 'dataset', 'validation')
    highest, selection, chosen, selected_best = -float('inf'), [], None, None
    members = config.get('ensemble_members', 1)
    for member in range(members):
        directory = work / 'checkpoints' / f'member-{member}' if members > 1 else work / 'checkpoints'
        best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
        model = make_model(config).eval()
        model.load_state_dict(best['model'])
        result = evaluate(model, DataLoader(dataset, batch_size=64), torch.device('cpu'))
        point = point_for_config(result, dataset.rows, config)
        score = selection_score(point, result['loss'], config)
        selection.append(dict(member=member, validation=point, score=score, distill_weight=config['member_overrides'][str(member)]['distill_weight']))
        if config.get('threshold_diagnostics'):
            selection[-1].update(best_epoch=best['epoch'], training_recipe=config['member_overrides'][str(member)])
        if score > highest: highest, chosen, selected_best, selected_member = score, model, best, member
    atomic_json(work / 'ensemble_selection.json', dict(selected_members=[selected_member], candidates=selection,
                decision_data='validation only; single student, no ensemble'))
    output = work / ('wake-model-SMOKE-ONLY' if config.get('smoke') else 'wake-model')
    output.mkdir(exist_ok=True)
    wrapped = ProbabilityModel(chosen).eval()
    probe_rows, covered = [], set()
    for row in dataset.rows:
        key = row['generator'], row['language'], row['label']
        if key not in covered: covered.add(key); probe_rows.append(row)
    frontend = WhisperMelFrontend(work / 'whisper-encoder')
    probe = np.stack([frontend(read_wav(work / 'dataset' / row['wav'])) for row in probe_rows])
    cached = np.stack([np.load(work / 'dataset' / row['feature']) for row in probe_rows])
    np.testing.assert_allclose(probe, cached, rtol=1e-5, atol=1e-6)
    model_file = output / 'wake_model.onnx'
    torch.onnx.export(wrapped, (torch.from_numpy(probe[:1]),), str(model_file), input_names=['log_mel'], output_names=['probability'],
                      dynamic_axes={'log_mel': {0: 'batch'}, 'probability': {0: 'batch'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(model_file))
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_file), options, providers=['CPUExecutionProvider'])
    with torch.inference_mode(): expected = wrapped(torch.from_numpy(probe)).numpy()
    actual = session.run(['probability'], {'log_mel': probe})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    shutil.copy2(Path(__file__).parent / 'wake_audio.py', output / 'wake_audio.py')
    hashes = {}
    for name in ['whisper_mel_filters.npy', 'provenance.json']:
        shutil.copy2(work / 'whisper-encoder' / name, output / name)
        hashes[name] = file_sha256(output / name)
    metadata = dict(schema_version=1, wake_word_ru=config['wake_word_ru'], wake_word_en=config['wake_word_en'] if config['use_english'] else None,
                    task='isolated_wake_word' if config['standalone_only'] else 'keyword_presence_in_window',
                    standalone_only=config['standalone_only'], sample_rate=16000, window_samples=32000, threshold=.5,
                    architecture='whisper_encoder_v4', input='log_mel float32 [batch,1,80,200]', output='probability float32 [batch]',
                    frontend=dict(type='whisper_log_mel_2s', n_fft=400, hop=160, n_mels=80, center_pad='reflect'),
                    frontend_sha256=hashes, model_sha256=file_sha256(model_file), selected_members=[selected_member],
                    best_epoch=selected_best['epoch'], smoke_only=config.get('smoke', False), quality_validated_on_real_users=False,
                    recipe_version=config.get('recipe_version'),
                    source_group_calibration=config.get('source_groups', False) and config.get('source_group_fpr_constraint', True))
    atomic_json(output / 'wake_config.json', metadata)
    atomic_json(work / 'onnx_validation.json', dict(split='validation', wav_count=len(probe), max_difference=float(np.max(np.abs(actual - expected)))))
    histories = []
    for member in range(members):
        name = f'training_history-member-{member}.json' if members > 1 else 'training_history.json'
        histories.extend({**row, 'member': member} for row in json.loads((work / name).read_text(encoding='utf-8')))
    atomic_json(work / 'training_history.json', histories)
    (output / 'README.txt').write_text('Binary wake-word detector. Mono PCM16, 16000 Hz, two-second window.\n'
       'pip install numpy onnxruntime\npython wake_audio.py . call.wav\n'
       'Only the student is needed. Quality metrics and TTS licence references are in the accompanying reports.\n'
       'Validate on real microphones before deployment.\n', encoding='utf-8')
    print(f'Exported student {selected_member}; ONNX validation parity passed; calibration/test untouched', flush=True)


def runtime_scores(model_file, dataset, batch_size=64):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_file), options, providers=['CPUExecutionProvider'])
    scores = []
    for features, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        scores.extend(session.run(['probability'], {'log_mel': features.numpy()})[0].tolist())
    if len(scores) != len(dataset) or not np.isfinite(scores).all() or np.any(np.array(scores) < 0) or np.any(np.array(scores) > 1):
        raise ValueError('Exported scores are not finite probabilities')
    return dict(labels=[r['label'] for r in dataset.rows], scores=scores)


def quality_gate(metrics, language_metrics, target_recall, target_fpr, smoke=False):
    reasons = []
    for name, item in [('overall', metrics), *language_metrics.items()]:
        if name == 'none': continue
        if not item['positive_windows'] or not item['negative_windows']:
            reasons.append(f'{name}: missing positive or negative evaluation windows')
        elif item['recall'] < target_recall or item['false_positive_rate_per_window'] > target_fpr:
            reasons.append(f'{name}: recall/FPR target not met')
    if smoke: reasons.append('Smoke execution cannot certify quality')
    return dict(passed=not reasons, target_recall=target_recall, target_fpr_per_window=target_fpr, reasons=reasons,
                scope='Observed point estimates on this held-out benchmark; not real-user accuracy or statistical certification')


def zip_runtime(directory, file):
    directory, file = Path(directory), Path(file)
    with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in RUNTIME_FILES:
            path = directory / name
            if not path.is_file(): raise ValueError(f'Missing runtime artifact: {name}')
            archive.write(path, name)
    with zipfile.ZipFile(file) as archive:
        if set(archive.namelist()) != set(RUNTIME_FILES): raise ValueError('Unexpected archive content')


def package(config):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from wake_audio import WakeDetector, read_wav
    work = Path(config['work_dir'])
    output = Path(config['output_dir'])
    if output.resolve() == work.resolve() or work.resolve() in output.resolve().parents:
        raise ValueError('Published output must be separate from transient training workspace')
    output.mkdir(parents=True, exist_ok=True)
    # Only previous managed archives are replaced; scratch/data paths are never traversed.
    for name in ['wake-model.zip', 'wake-model-fp32-alternative.zip', 'wake-model-int8-alternative.zip']:
        (output / name).unlink(missing_ok=True)
    source = work / ('wake-model-SMOKE-ONLY' if config.get('smoke') else 'wake-model')
    float_config = json.loads((source / 'wake_config.json').read_text(encoding='utf-8'))
    runtime = work / 'runtime-int8'
    runtime.mkdir(exist_ok=True)
    for name in RUNTIME_FILES:
        if name != 'wake_model.onnx': shutil.copy2(source / name, runtime / name)
    # Constant-weight MatMul/Gemm only: keep signal frontend, convolutions and activations FP32.
    runtimes, quantization_error = [('fp32', source)], None
    try:
        quantize_dynamic(str(source / 'wake_model.onnx'), str(runtime / 'wake_model.onnx'),
                         weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul', 'Gemm'],
                         extra_options={'MatMulConstBOnly': True})
        onnx.checker.check_model(str(runtime / 'wake_model.onnx'))
        # Verify CPU operator support before adding this optional runtime.
        import onnxruntime as ort
        probe_dataset = FeatureDataset(work / 'dataset', 'validation')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(runtime / 'wake_model.onnx'), options, providers=['CPUExecutionProvider'])
        value = session.run(['probability'], {'log_mel': probe_dataset[0][0].numpy()[None]})[0]
        if not np.isfinite(value).all(): raise ValueError('INT8 probe returned non-finite scores')
        del session
        runtimes.append(('int8', runtime))
    except Exception as error:
        quantization_error = f'{type(error).__name__}: {error}'
        print(f'INT8 unavailable; continuing with verified FP32: {quantization_error}', flush=True)
        for name in ['training_report_int8.json', 'test_predictions_int8.json']:
            (output / name).unlink(missing_ok=True)
    validation = FeatureDataset(work / 'dataset', 'validation')
    val_points = {}
    for name, directory in runtimes:
        # Dynamic INT8 activation scales depend on the batch. Production scores one window.
        prediction = runtime_scores(directory / 'wake_model.onnx', validation, batch_size=1 if name == 'int8' else 64)
        val_points[name] = point_for_config(prediction, validation.rows, config)
    tolerance = config.get('quantization_recall_tolerance', .01)
    recall_keys = ['macro_recall', 'worst_language_recall']
    if config.get('source_groups'): recall_keys.append('worst_source_recall')
    acceptable_loss = 'int8' in val_points and all(val_points['int8'][key] >= val_points['fp32'][key] - tolerance for key in recall_keys)
    target = config.get('target_recall', .9)
    preserves_goal = 'int8' in val_points and (not all(val_points['fp32'][key] >= target for key in recall_keys) or all(val_points['int8'][key] >= target for key in recall_keys))
    selected = 'int8' if acceptable_loss and preserves_goal else 'fp32'
    # Selection is frozen before calibration/test are read.
    selection = dict(selected=selected, decision_data='validation only', validation=val_points,
                     recall_tolerance=tolerance, quantization_error=quantization_error,
                     sizes_bytes={name: (directory / 'wake_model.onnx').stat().st_size for name, directory in runtimes})
    atomic_json(output / 'compression_selection.json', selection)
    calibration = FeatureDataset(work / 'dataset', 'calibration')
    test = FeatureDataset(work / 'dataset', 'test')
    reports = {}
    for name, directory in runtimes:
        inference_batch = 1 if name == 'int8' else 64
        cal = runtime_scores(directory / 'wake_model.onnx', calibration, batch_size=inference_batch)
        cal_sources = [source_group(row) for row in calibration.rows] if config.get('source_groups') else None
        constrain_sources = config.get('source_group_fpr_constraint', True)
        threshold, cal_metrics = calibrate(cal['labels'], cal['scores'], config['target_fpr'],
                                           [r['language'] for r in calibration.rows], cal_sources if constrain_sources else None)
        calibration_sources = None
        if cal_sources is not None:
            cal_sources = np.array(cal_sources)
            cal_labels, cal_scores = np.array(cal['labels']), np.array(cal['scores'])
            calibration_sources = {group: confusion(cal_labels[cal_sources == group], cal_scores[cal_sources == group], threshold)
                                   for group in sorted(set(cal_sources.tolist()))}
        prediction = runtime_scores(directory / 'wake_model.onnx', test, batch_size=inference_batch)
        metrics = confusion(prediction['labels'], prediction['scores'], threshold)
        groups = {}
        for field in ['language', 'generator', 'speaker', 'kind']:
            groups[field] = {}
            for value in sorted({r[field] for r in test.rows}):
                indices = [i for i, row in enumerate(test.rows) if row[field] == value]
                groups[field][value] = confusion([prediction['labels'][i] for i in indices], [prediction['scores'][i] for i in indices], threshold)
        metadata = dict(copy.deepcopy(float_config), threshold=threshold, quantization=name, inference_batch_size=1,
                        model_sha256=file_sha256(directory / 'wake_model.onnx'))
        atomic_json(directory / 'wake_config.json', metadata)
        detector = WakeDetector(directory)
        covered, maximum_difference = set(), 0.
        for index, row in enumerate(test.rows):
            key = row['generator'], row['language'], row['label']
            if key in covered: continue
            covered.add(key)
            score = detector.score(read_wav(work / 'dataset' / row['wav']))
            maximum_difference = max(maximum_difference, abs(score - prediction['scores'][index]))
        if maximum_difference > 1e-4: raise ValueError('WAV frontend and published ONNX scores disagree')
        gate = quality_gate(metrics, groups['language'], config.get('target_recall', .9), config['target_fpr'], config.get('smoke', False))
        human_indices = [i for i, row in enumerate(test.rows) if row['generator'] == 'human_reviewed']
        human_metrics = confusion([prediction['labels'][i] for i in human_indices], [prediction['scores'][i] for i in human_indices], threshold) if human_indices else None
        report = dict(calibration=cal_metrics, calibration_source_groups=calibration_sources,
                      threshold=threshold, test=metrics, test_groups=groups,
                      quality_gate=gate, selected_on_validation=name == selected, model_bytes=(directory / 'wake_model.onnx').stat().st_size,
                      wav_runtime_max_difference=maximum_difference, reviewed_human_test=human_metrics,
                      real_user_validation='not performed' if not human_metrics or not human_metrics['positive_windows'] else 'reviewed human test subset; see counts',
                      interpretation='Known V4 voice holdouts reused as regression benchmark, with new synthesis seed. No FAR/hour claim. No test-based threshold/model selection.')
        reports[name] = report
        if config.get('threshold_diagnostics'):
            report['calibration_threshold_audit'] = threshold_audit(cal['labels'], cal['scores'], config['target_fpr'],
                [row['language'] for row in calibration.rows], cal_sources, constrain_sources)
            if report['calibration_threshold_audit']['threshold'] != threshold:
                raise ValueError('Threshold audit and runtime calibration disagree')
        atomic_json(output / f'training_report_{name}.json', report)
        if config.get('export_calibration_predictions'):
            records = [dict(id=row['id'], language=row['language'], speaker=row['speaker'], generator=row['generator'],
                            label=row['label'], text=row['text'], kind=row.get('kind'), source_id=row.get('source_id'),
                            probability=score, predicted_wake=score >= threshold)
                       for row, score in zip(calibration.rows, cal['scores'], strict=True)]
            atomic_json(output / f'calibration_predictions_{name}.json', records)
        records = [dict(id=row['id'], language=row['language'], speaker=row['speaker'], generator=row['generator'],
                        label=row['label'], text=row['text'], probability=score, predicted_wake=score >= threshold)
                   for row, score in zip(test.rows, prediction['scores'], strict=True)]
        if config.get('export_sample_metadata'):
            for record, row in zip(records, test.rows, strict=True):
                record.update(kind=row.get('kind'), source_id=row.get('source_id'))
        atomic_json(output / f'test_predictions_{name}.json', records)
        zip_name = 'wake-model.zip' if name == selected else f'wake-model-{name}-alternative.zip'
        zip_runtime(directory, output / zip_name)
    for name in ['environment.json', 'source_sha256.json', 'run_config.json', 'dataset_recipe.json', 'dataset_summary.json',
                 'dataset_groups.json', 'tts_provenance.json', 'real_provenance.json', 'distillation_report.json',
                 'teacher_provenance.json', 'ensemble_selection.json', 'training_history.json', 'training_run.json', 'onnx_validation.json']:
        if (work / name).is_file(): shutil.copy2(work / name, output / name)
    for name in ['training_history.json', 'training_run.json']:
        if (work / 'teacher' / name).is_file(): shutil.copy2(work / 'teacher' / name, output / ('teacher_' + name))
    if config.get('export_sample_metadata'):
        for audit in work.glob('source_audit-*.json'): shutil.copy2(audit, output / audit.name)
    selected_report = reports[selected]
    atomic_json(output / 'training_report.json', selected_report)
    summary = dict(recommended_archive='wake-model.zip', format=selected, quality_gate=selected_report['quality_gate'],
                   model_bytes=selected_report['model_bytes'], output_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                   dataset_exported=False, teacher_exported=False, real_user_validation=selected_report['real_user_validation'])
    atomic_json(output / 'output_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return reports, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--action', choices=['export', 'package'], default='package')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.action == 'export': export_student(config)
    else: package(config)
