"""Offline checks for consequential label, split and calibration boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/wake-validation'), str(ROOT / 'scripts/wakeword')]
import numpy as np
from wake_audio import features, N_SAMPLES, right_aligned_window, read_wav, write_wav
from wake_data import augment, negative_texts, merge_manifest, voice_jobs
from wake_export import calibrate
from wake_train import confusion
from wake_train import BalancedSampler
from wake_tts import voice_jobs as multi_voice_jobs
from wake_data_v2 import augment as multi_augment, negative_kind, speaker_split, prepare_human


class WakeChecks(unittest.TestCase):
    def test_language_constraint_cannot_hide_russian_false_positives(self):
        labels = [0] * 2000 + [0] * 10 + [1, 1]
        scores = [.1] * 2000 + [.9] * 10 + [.99, .99]
        languages = ['en'] * 2000 + ['ru'] * 10 + ['ru', 'en']
        pooled, _ = calibrate(labels, scores, .005)
        balanced, metrics = calibrate(labels, scores, .005, languages)
        self.assertLess(pooled, .9)
        self.assertGreater(balanced, .9)
        self.assertEqual(metrics['fp'], 0)
        self.assertEqual(metrics['recall'], 1)

    def test_versions_share_voice_split_and_test_has_new_engines(self):
        jobs = multi_voice_jobs({'use_english': True})
        self.assertEqual(len(jobs), 66)
        assignment = {}
        for job in jobs:
            self.assertEqual(assignment.setdefault(job['speaker_group'], job['split']), job['split'])
        self.assertEqual(len(assignment), 61)
        self.assertEqual({j['engine'] for j in jobs}, {'silero', 'piper', 'mms', 'kokoro'})
        self.assertEqual({j['engine'] for j in jobs if j['split'] == 'test'}, {'silero', 'piper', 'kokoro'})
        russian_train = {j['speaker_group'] for j in jobs if j['language'] == 'ru' and j['split'] == 'train'}
        self.assertEqual(len(russian_train), 6)
        self.assertTrue(any('irina' in voice for voice in russian_train))
        self.assertTrue(all(j['split'] == 'validation' for j in jobs if j['language'] == 'ru' and j['voice'] == 'baya'))

    def test_balanced_sampler_distributes_same_stream_and_mining_preserves_mass(self):
        import torch
        rows = [dict(language=language, label=label, generator=engine) for language in ['ru', 'en']
                for label in [0, 1] for engine in ['silero', 'piper'] for _ in range(3 if language == 'en' else 1)]
        all_samples = BalancedSampler(rows, 42)
        rank0, rank1 = BalancedSampler(rows, 42, 0, 2), BalancedSampler(rows, 42, 1, 2)
        rank0.set_epoch(3); rank1.set_epoch(3); all_samples.set_epoch(3)
        expected = list(all_samples)
        self.assertEqual(list(rank0), expected[::2]); self.assertEqual(list(rank1), expected[1::2])
        positive = torch.tensor([r['label'] == 1 for r in rows])
        self.assertAlmostEqual(float(all_samples.weights[positive].sum()), .5)
        all_samples.mine(np.linspace(0, 1, len(rows)))
        self.assertAlmostEqual(float(all_samples.weights[positive].sum()), .5)
        for key in set(all_samples.keys):
            selected = torch.tensor([k == key for k in all_samples.keys])
            self.assertAlmostEqual(float(all_samples.weights[selected].sum()), float(all_samples.base[selected].sum()))

    def test_multi_engine_augmentation_keeps_labels_and_real_speaker_groups(self):
        too_long = np.ones(N_SAMPLES * 2, np.float32) * .1
        self.assertIsNone(multi_augment(too_long, np.random.default_rng(2), 1))
        self.assertIsNone(multi_augment(too_long, np.random.default_rng(2), 0, preserve_context=True))
        self.assertEqual(negative_kind('этот бот', 'бот'), 'context')
        self.assertEqual(negative_kind('ботинок', 'бот'), 'hard_word')
        self.assertEqual(speaker_split('knownspeaker'), speaker_split('knownspeaker'))

    def test_reviewed_human_recordings_reject_speaker_leakage_and_automatic_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            (root / 'real_sources.json').write_text('[]')
            write_wav(root / 'call.wav', np.sin(np.arange(4000) * .12).astype(np.float32) * .2)
            record = dict(wav='call.wav', label=1, language='ru', speaker='person', split='train', text='бот', reviewed=True)
            manifest = root / 'human.jsonl'
            manifest.write_text(json.dumps(record), encoding='utf-8')
            config = dict(work_dir=str(root), real_wake_manifest=str(manifest))
            prepare_human(config)
            rows = json.loads((root / 'real_sources.json').read_text(encoding='utf-8'))
            self.assertEqual(rows[0]['label'], 1)
            self.assertEqual(rows[0]['generator'], 'human_reviewed')
            manifest.write_text(json.dumps(dict(record, label=0, split='calibration')), encoding='utf-8')
            prepare_human(config)
            refreshed = json.loads((root / 'real_sources.json').read_text(encoding='utf-8'))
            self.assertEqual(refreshed[0]['label'], 0)
            self.assertEqual(refreshed[0]['split'], 'calibration')
            record['reviewed'] = False
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'reviewed'): prepare_human(config)
            record['reviewed'] = True
            manifest.write_text(json.dumps(record) + '\n' + json.dumps(dict(record, split='test')))
            with self.assertRaisesRegex(ValueError, 'Human speaker leakage'): prepare_human(config)

    def test_calibration_respects_runtime_comparison_and_tied_negatives(self):
        threshold, metrics = calibrate([0, 0, 0, 1, 1], [.8, .8, .1, .9, .95], 1 / 3)
        self.assertGreater(threshold, .8)
        self.assertEqual(metrics['fp'], 0)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(confusion([0, 1], [.49, .5], .5)['tp'], 1)
        with self.assertRaises(ValueError): calibrate([1, 1], [.8, .9], .01)

    def test_changed_word_cannot_be_an_exact_negative(self):
        for language, word in [('ru', 'кот'), ('en', 'boat')]:
            negatives = negative_texts(language, word, False)
            self.assertFalse(any(word in text.lower().split() for text in negatives))
            self.assertTrue(any(word in text.lower().split() for text in negative_texts(language, word, True)))

    def test_positive_and_context_audio_is_not_truncated_with_stale_labels(self):
        rng = np.random.default_rng(17)
        too_long = np.ones(N_SAMPLES * 2, np.float32) * .1
        self.assertIsNone(augment(too_long, rng, 1))
        self.assertIsNone(augment(too_long, rng, 0, preserve_context=True))
        negative = augment(too_long, rng, 0)
        self.assertEqual(negative.shape, (N_SAMPLES,))
        self.assertTrue(np.isfinite(negative).all())
        self.assertLessEqual(float(np.max(np.abs(negative))), .981)

    def test_frontend_matches_export_window_contract_and_pcm(self):
        short = np.linspace(-.4, .4, 4000, dtype=np.float32)
        window = right_aligned_window(short)
        np.testing.assert_array_equal(window[-len(short):], short)
        self.assertFalse(np.any(window[:-len(short)]))
        self.assertEqual(features(window).shape, (1, 40, 201))
        self.assertTrue(np.isfinite(features(np.zeros(N_SAMPLES, np.float32))).all())
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            path = Path(directory) / 'audio.wav'
            write_wav(path, window)
            np.testing.assert_allclose(read_wav(path), window, atol=5e-5)

    def test_voice_assignments_and_source_leakage_are_enforced(self):
        jobs = voice_jobs({'use_english': True})
        self.assertEqual(len(jobs), 29)
        self.assertEqual(len({(j['language'], j['voice']) for j in jobs}), len(jobs))
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as directory:
            root = Path(directory)
            data = root / 'dataset'
            data.mkdir()
            np.save(data / 'sample.npy', np.zeros((1, 40, 201), np.float32))
            rows = []
            for split in ['train', 'validation', 'calibration', 'test']:
                for label in [0, 1]:
                    rows.append({'id': f'{split}-{label}', 'split': split, 'label': label, 'speaker': split,
                                 'source_id': f'{split}-{label}', 'feature': 'sample.npy'})
            manifest = data / 'manifest-rank-0.jsonl'
            write = lambda: manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            write()
            merge_manifest({'work_dir': directory}, 1)
            rows[-1]['source_id'] = rows[0]['source_id']
            write()
            with self.assertRaisesRegex(AssertionError, 'Source audio leakage'): merge_manifest({'work_dir': directory}, 1)
            rows[-1]['source_id'] = 'fixed-source'
            rows[-1]['speaker'] = 'train'
            write()
            with self.assertRaisesRegex(AssertionError, 'Speaker leakage'): merge_manifest({'work_dir': directory}, 1)


if __name__ == '__main__':
    unittest.main()
