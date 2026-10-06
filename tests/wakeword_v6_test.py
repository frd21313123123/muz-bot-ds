"""Offline checks for V6 source coverage, calibration and recipe boundaries; no training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/wake-validation'), str(ROOT / 'scripts/wakeword'), str(ROOT / 'scripts')]
from wake_tts import voice_jobs
from wake_data import negative_texts
from wake_metrics import calibrate, point_for_config, selection_score, source_group
from build_wakeword_v6_notebook import build_v6
import numpy as np

SPLITS = {'train':['ru_RU-irina-medium'], 'validation':['ru_RU-ruslan-medium'],
          'calibration':['ru_RU-dmitri-medium'], 'test':['ru_RU-denis-medium']}


class V6Checks(unittest.TestCase):
    def test_piper_russian_coverage_has_disjoint_voices_in_all_splits(self):
        jobs = voice_jobs(dict(use_english=True, piper_ru_splits=SPLITS))
        seen = {}
        for job in jobs:
            self.assertEqual(seen.setdefault(job['speaker_group'], job['split']), job['split'])
        for split in SPLITS:
            selected = [job['voice'] for job in jobs if job['engine']=='piper' and job['language']=='ru' and job['split']==split]
            self.assertEqual(selected, SPLITS[split])
        self.assertEqual(len(seen), 61)
        self.assertFalse(any(job['voice']=='ru_RU-denis-medium' and job['split']!='test' for job in jobs))
        # Default V5 coverage remains reproducible; only V6 opts into the new split.
        self.assertFalse(any(job['engine']=='piper' and job['language']=='ru' and job['split']=='calibration'
                             for job in voice_jobs(dict(use_english=True))))

    def test_override_rejects_repeated_unknown_and_empty_voice_splits(self):
        for mutation in ['repeat', 'unknown', 'empty']:
            splits = copy.deepcopy(SPLITS)
            splits['validation'] = {'repeat':SPLITS['train'], 'unknown':['invented-voice'], 'empty':[]}[mutation]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                voice_jobs(dict(use_english=True, piper_ru_splits=splits))

    def test_exclusion_is_explicit_and_keeps_distinguishable_hard_negatives(self):
        original = negative_texts('ru', 'бот', True)
        adjusted = negative_texts('ru', 'бот', True, exclusions=[' БОД '])
        self.assertIn('бод', original)
        self.assertNotIn('бод', adjusted)
        self.assertEqual(set(original)-set(adjusted), {'бод'})
        for text in ['боту','бок','борт','этот бот']: self.assertIn(text, adjusted)

    def test_calibration_cannot_hide_minority_generator_false_positives(self):
        labels, scores = [0]*2000+[0]*4+[1,1], [.1]*2000+[.9]*4+[.99,.99]
        languages = ['ru']*len(labels)
        groups = ['ru|silero|v5_5_ru']*2000+['ru|piper|piper']*4+['ru|silero|v5_5_ru','ru|piper|piper']
        threshold, _ = calibrate(labels, scores, .005, languages)
        strict, result = calibrate(labels, scores, .005, languages, groups)
        self.assertLess(threshold,.9)
        self.assertGreater(strict,.9)
        self.assertEqual(result['fp'],0)
        self.assertEqual(result['recall'],1)
        with self.assertRaisesRegex(ValueError,'groups must match'): calibrate(labels,scores,.005,languages,groups[:2])

    def test_silero_renderings_are_distinct_and_worst_source_affects_selection(self):
        v4 = dict(language='ru',generator='silero',speaker='v4_ru/baya')
        v5 = dict(language='ru',generator='silero',speaker='v5_5_ru/baya')
        self.assertNotEqual(source_group(v4),source_group(v5))
        rows = [v4,v5,v4,v5]
        prediction = dict(labels=[0,0,1,1],scores=[.2,.2,.1,.9])
        default = point_for_config(prediction,rows,dict(target_fpr=.005))
        grouped = point_for_config(prediction,rows,dict(target_fpr=.005,source_groups=True))
        self.assertNotIn('source_groups',default)
        self.assertEqual(grouped['worst_source_recall'],0.)
        good = dict(grouped,worst_source_recall=.8)
        self.assertGreater(selection_score(good,.1,dict(source_groups=True)),
                           selection_score(grouped,.1,dict(source_groups=True)))

    def test_notebook_parameters_and_embedded_modules_execute_without_training(self):
        notebook = build_v6(output=None)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory, patch.dict(os.environ):
            scratch = Path(directory)
            namespace = {}
            cell = next(cell for cell in notebook['cells'] if 'parameters' in cell['metadata'].get('tags',[]))
            source = cell['source'].replace('/kaggle/temp/wakeword-bot-v6',str(scratch/'work').replace('\\','/'))
            source = source.replace('/kaggle/working/wakeword-bot-v6',str(scratch/'output').replace('\\','/'))
            exec(compile(source,'parameters','exec'),namespace)
            config = namespace['CONFIG']
            self.assertEqual(config['recipe_version'],6)
            self.assertEqual(config['piper_ru_splits'],SPLITS)
            self.assertEqual(config['negative_exclusions'],{'ru':['бод']})
            self.assertTrue(config['source_groups'] and config['positive_bare_word'])
            self.assertEqual(config['member_seed_stride'],0)
            for member in ['0','1']:
                self.assertEqual(config['member_overrides'][member]['encoder_train_layers'],1)
                self.assertEqual(config['member_overrides'][member]['encoder_learning_rate'],3e-6)
            recipe=json.loads((scratch/'work/dataset_recipe.json').read_text(encoding='utf-8'))
            self.assertEqual(recipe['negative_exclusions'],{'ru':['бод']})
            self.assertEqual(recipe['piper_ru_splits'],SPLITS)
            self.assertTrue(recipe['positive_bare_word'])
            for cell in notebook['cells']:
                if any(tag.startswith('embedded-') for tag in cell['metadata'].get('tags',[])):
                    exec(compile(cell['source'],'embedded','exec'),namespace)
            for name,expected in notebook['metadata']['wakeword']['source_sha256'].items():
                self.assertEqual(hashlib.sha256((scratch/'work/code'/name).read_bytes()).hexdigest(),expected)
            self.assertEqual(notebook['metadata']['wakeword']['version'],6)
            self.assertEqual(config['teacher_repository'],'openai/whisper-large-v3')
            self.assertFalse(any(cell.get('outputs') for cell in notebook['cells']))

    def test_generator_passes_bare_positive_and_excludes_ambiguous_negative(self):
        from wake_data_v2 import generate
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory:
            root=Path(directory)
            job=dict(engine='piper',model='piper',language='ru',voice='ru_RU-irina-medium',
                     split='train',speaker_group='piper/ru/ru_RU-irina-medium')
            config=dict(work_dir=str(root),force_cpu=True,architecture='temporal_v2',cpu_threads=1,
                        seed=345,smoke=False,wake_word_ru='бот',wake_word_en='bot',standalone_only=True,
                        positive_bare_word=True,negative_exclusions={'ru':['бод']},independent_pitch=False,
                        positive_tts_variants=1,negative_tts_variants=1,positive_per_voice=6,negative_per_voice=8,
                        russian_multiplier=1,noise_windows=0)
            with patch('wake_data_v2.voice_jobs',return_value=[job]), patch('wake_data_v2.Synthesizer') as synth:
                synth.return_value.provider='fixture-only'
                synth.return_value.say.return_value=np.sin(np.arange(4000)*.12).astype(np.float32)*.2
                generate(config)
                texts=[call.args[0] for call in synth.return_value.say.call_args_list]
            self.assertIn('бот',texts)
            self.assertNotIn('бод',texts)
            self.assertIn('боту',texts)
            rows=[json.loads(line) for line in (root/'dataset/manifest-rank-0.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(row['label']==1 for row in rows),6)
            self.assertEqual(sum(row['label']==0 for row in rows),8)
            self.assertTrue(all(row['split']=='train' and row['kind'] in ['positive','hard_word','context','ordinary'] for row in rows))
            self.assertTrue(all(np.isfinite(np.load(root/'dataset'/row['feature'])).all() for row in rows))


if __name__=='__main__': unittest.main()
