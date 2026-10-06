"""Offline boundary checks; no model training, external service or model download."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/wake-validation'), str(ROOT / 'scripts/wakeword')]
import numpy as np
import torch
from wake_audio import WhisperMelFrontend, N_SAMPLES
from wake_train import FeatureDataset, evaluate, evaluate_shared, merge_evaluations
from wake_tts import file_sha256, prepare_whisper_encoder
from wake_model import WhisperWakeModel
from wake_distill import DistillationDataset, distillation_loss, merge_teacher
from wake_package_v5 import quality_gate, RUNTIME_FILES, zip_runtime


class DistillationChecks(unittest.TestCase):
    def test_evaluation_merges_uneven_shards_in_manifest_order(self):
        shards = [
            dict(indices=[0, 2, 4], prediction=dict(labels=[0, 0, 0], scores=[.1, .3, .5], logits=[1., 3., 5.], loss=2.)),
            dict(indices=[1, 3], prediction=dict(labels=[1, 1], scores=[.2, .4], logits=[2., 4.], loss=4.)),
        ]
        result = merge_evaluations(shards, 5)
        self.assertEqual(result['labels'], [0, 1, 0, 1, 0])
        self.assertEqual(result['scores'], [.1, .2, .3, .4, .5])
        self.assertEqual(result['logits'], [1., 2., 3., 4., 5.])
        self.assertAlmostEqual(result['loss'], 2.8)
        with self.assertRaisesRegex(ValueError, 'duplicate'): merge_evaluations(shards + [shards[0]], 5)
        with self.assertRaisesRegex(ValueError, 'cover'): merge_evaluations(shards[:1], 5)

    def test_shared_evaluation_single_process_matches_original(self):
        features = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
        dataset = torch.utils.data.TensorDataset(features, torch.tensor([0., 1., 0., 1., 0.]))
        model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(start_dim=0))
        device = torch.device('cpu')
        expected = evaluate(model, torch.utils.data.DataLoader(dataset, batch_size=2), device)
        actual = evaluate_shared(model, dataset, 2, device, amp=True)
        self.assertEqual(actual, expected)

    def test_encoder_extraction_ignores_decoder_and_accepts_large_v3_mel_shape(self):
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        from safetensors.torch import save_file
        # Small random fixture has Large V3's mel contract; no real teacher download/training.
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            source = root / 'whisper-encoder/source'
            source.mkdir(parents=True)
            config = WhisperConfig(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                                   num_mel_bins=128, max_source_positions=100,
                                   decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=64)
            config._attn_implementation = 'eager'
            config.to_json_file(str(source / 'config.json'))
            WhisperFeatureExtractor(feature_size=128).save_pretrained(source)
            encoder = WhisperEncoder(config)
            tensors = {'model.encoder.' + key: value.contiguous() for key, value in encoder.state_dict().items()}
            tensors['model.decoder.fixture'] = torch.ones(1)
            save_file(tensors, str(source / 'model.safetensors'))
            with patch('huggingface_hub.snapshot_download') as download:
                provenance = prepare_whisper_encoder(root, 'openai/whisper-large-v3', revision='fixture-commit')
            self.assertEqual(download.call_args.kwargs['repo_id'], 'openai/whisper-large-v3')
            self.assertEqual(provenance['revision'], 'fixture-commit')
            self.assertEqual(np.load(root / 'whisper-encoder/whisper_mel_filters.npy').shape, (128, 201))
            model = WhisperWakeModel(dict(work_dir=str(root), encoder_train_layers=1)).eval()
            with torch.inference_mode(): output = model(torch.zeros(2, 1, 128, 200))
            self.assertEqual(output.shape, (2,))
            self.assertTrue(torch.isfinite(output).all())
            self.assertFalse(any('decoder' in name for name in model.state_dict()))

    def test_teacher_disagreement_has_no_distillation_gradient(self):
        student = torch.tensor([0., 0.], requires_grad=True)
        loss = distillation_loss(student, torch.tensor([-8., 8.]), torch.tensor([1., 0.]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.all(student.grad == 0))

    def test_teacher_gradient_points_to_labels_and_equal_logits_have_zero_kl(self):
        student = torch.tensor([0., 0.], requires_grad=True)
        teacher, labels = torch.tensor([8., -8.]), torch.tensor([1., 0.])
        loss = distillation_loss(student, teacher, labels)
        loss.backward()
        self.assertLess(student.grad[0].item(), 0.)
        self.assertGreater(student.grad[1].item(), 0.)
        self.assertLess(distillation_loss(teacher, teacher, labels).item(), 1e-5)
        with self.assertRaises(ValueError): distillation_loss(student, teacher, labels, 0)

    def test_large_v3_and_tiny_frontends_preserve_distinct_channel_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            for channels in [80, 128]:
                np.save(root / 'whisper_mel_filters.npy', np.ones((channels, 201), np.float32) / 201)
                feature = WhisperMelFrontend(root)(np.sin(np.arange(N_SAMPLES) * .17).astype(np.float32) * .1)
                self.assertEqual(feature.shape, (1, channels, 200))
                self.assertTrue(np.isfinite(feature).all())

    def test_cache_rejects_feature_edits_and_non_training_data(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            np.save(root / 'sample.npy', np.zeros((1, 80, 200), np.float32))
            row = dict(id='train-0', feature='sample.npy', label=1, split='train')
            (root / 'manifest.jsonl').write_text(json.dumps(row), encoding='utf-8')
            file = root / 'targets.npz'
            np.savez(file, ids=np.array(['train-0']), logits=np.array([2.], np.float32),
                     student_feature_sha256=np.array([file_sha256(root / 'sample.npy')]),
                     manifest_sha256=np.array(file_sha256(root / 'manifest.jsonl')))
            data = FeatureDataset(root, 'train')
            cached = DistillationDataset(data, file)
            self.assertEqual(len(cached[0]), 3)
            np.save(root / 'sample.npy', np.ones((1, 80, 200), np.float32))
            with self.assertRaisesRegex(ValueError, 'feature changed'): DistillationDataset(data, file)
            data.rows[0]['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'training-only'): DistillationDataset(data, file)

    def test_duplicate_or_missing_teacher_cache_rows_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            (root / 'dataset').mkdir()
            (root / 'teacher/checkpoints').mkdir(parents=True)
            (root / 'teacher/checkpoints/best.pt').write_bytes(b'fixture-checkpoint-not-executed')
            row = dict(id='train-0', feature='sample.npy', label=1, split='train')
            (root / 'dataset/manifest.jsonl').write_text(json.dumps(row), encoding='utf-8')
            np.save(root / 'dataset/sample.npy', np.zeros((1, 80, 200), np.float32))
            digest = file_sha256(root / 'teacher/checkpoints/best.pt')
            for rank in range(2):
                np.savez(root / f'teacher-targets-rank-{rank}.npz', indices=np.array([0]), logits=np.array([2.]), teacher_checkpoint_sha256=np.array(digest))
            with self.assertRaisesRegex(ValueError, 'exactly once'): merge_teacher(dict(work_dir=str(root)), 2)

    def test_quality_gate_does_not_hide_russian_failure_in_pooled_accuracy(self):
        good = dict(positive_windows=100, negative_windows=1000, recall=.95, false_positive_rate_per_window=.001)
        bad = dict(good, recall=.6, false_positive_rate_per_window=.02)
        self.assertFalse(quality_gate(good, dict(ru=bad, en=good), .9, .005)['passed'])
        self.assertTrue(quality_gate(good, dict(ru=good, en=good), .9, .005)['passed'])
        self.assertFalse(quality_gate(good, dict(ru=good, en=good), .9, .005, smoke=True)['passed'])

    def test_runtime_zip_excludes_dataset_teacher_checkpoints_and_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            for name in RUNTIME_FILES: (root / name).write_bytes(b'runtime-fixture')
            for name in ['dataset.zip', 'teacher.pt', 'wake_model_state.pt']: (root / name).write_bytes(b'exclude-me')
            output = root / 'wake-model.zip'
            zip_runtime(root, output)
            with zipfile.ZipFile(output) as archive: self.assertEqual(set(archive.namelist()), set(RUNTIME_FILES))


if __name__ == '__main__': unittest.main()
