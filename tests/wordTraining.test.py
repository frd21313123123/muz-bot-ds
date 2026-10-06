"""Word-boundary labels and full-recording inference boundaries, without training."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'scripts/wakeword'),str(ROOT/'word_training')]
import numpy as np
from data import contains_word,text_sets,prepare_real_audio
from detect import scan
from train import configuration,same_recipe,run_training
from wake_tts import voice_jobs


class WordTrainingChecks(unittest.TestCase):
    def test_mixed_storage_reads_the_requested_frame_not_the_whole_pack(self):
        from improve import PackedFeatureDataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=np.full((1,80,200),.2,np.float32)
            b=np.full((1,80,200),.8,np.float16)
            np.save(root/'single.npy',a);np.save(root/'pack.npy',np.stack([a,b]))
            rows=[dict(feature='single.npy',label=0,split='train'),dict(feature='pack.npy',feature_index=1,label=1,split='train')]
            (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            for preload in [False,True]:
                dataset=PackedFeatureDataset(root,'train',preload)
                self.assertEqual(tuple(dataset[1][0].shape),(1,80,200))
                np.testing.assert_array_equal(dataset[0][0].numpy(),a)
                np.testing.assert_array_equal(dataset[1][0].numpy(),b.astype(np.float32))
                self.assertEqual(dataset[1][1].item(),1.)
                for mapped in dataset.packed.values():mapped._mmap.close()
    def test_training_restores_grad_mode_after_inference_only_tts(self):
        import torch
        gradients=[]
        def step(config):
            layer=torch.nn.Linear(2,1)
            layer(torch.ones(1,2)).sum().backward()
            gradients.append(layer.weight.grad.clone())
        with torch.inference_mode(),torch.no_grad(),patch('wake_train.train',side_effect=step):
            run_training({})
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(torch.equal(gradients[0],torch.ones(1,2)))
    def test_resume_ignores_checkpoint_location_but_rejects_changed_word_or_batch(self):
        config=configuration(ROOT/'word_training/runs/fixture','бот')
        self.assertTrue(same_recipe(config,dict(config,resume_checkpoint='checkpoints/latest.pt')))
        self.assertFalse(same_recipe(config,dict(config,wake_word_ru='кот')))
        self.assertFalse(same_recipe(config,dict(config,batch_per_gpu=128)))
    def test_quiet_real_recording_is_normalized_and_empty_audio_is_rejected(self):
        quiet=np.sin(np.arange(1600)*.1).astype(np.float32)*.0005
        normalized=prepare_real_audio(quiet)
        self.assertGreater(float(np.max(np.abs(normalized))),.9)
        with self.assertRaises(ValueError): prepare_real_audio(np.zeros(1600,np.float32))
    def test_whole_word_anywhere_has_no_old_negative_context_labels(self):
        for text in ['БОТ','бот, включи музыку','это бот','слушай, бот!']:
            self.assertTrue(contains_word(text,'бот'))
        for text in ['робот','боту','работа','борт','вот','ботинок']:
            self.assertFalse(contains_word(text,'бот'))
        positives,negatives=text_sets('бот')
        self.assertIn('бот включи музыку',positives)
        self.assertIn('это бот',positives)
        self.assertTrue(all(contains_word(text,'бот') for text in positives))
        self.assertTrue(all(not contains_word(text,'бот') for text in negatives))
        self.assertIn('борт',negatives)

    def test_ru_gpu_recipe_has_disjoint_speakers_and_adapts_four_layers(self):
        config=configuration(ROOT/'word_training/runs/fixture','бот')
        groups={}
        for job in voice_jobs(config):
            self.assertEqual(groups.setdefault(job['speaker_group'],job['split']),job['split'])
        self.assertFalse(config['standalone_only'] or config['use_english'])
        self.assertEqual(config['encoder_train_layers'],4)
        self.assertEqual(config['isolated_positive_fraction'],.5)
        self.assertTrue(config['mixed_precision'])
        self.assertEqual(config['loader_workers'],0)

    def test_scan_covers_word_at_start_middle_and_partial_final_window(self):
        class Detector:
            config=dict(threshold=.5,wake_word_ru='бот')
            def score(self,audio): return float(np.any(audio==1))
        for size,index in [(100,99),(45001,0),(45001,26000),(45001,45000)]:
            audio=np.zeros(size,np.float32); audio[index]=1
            self.assertTrue(scan(Detector(),audio)['detected'])
        self.assertFalse(scan(Detector(),np.zeros(40000,np.float32))['detected'])

    def test_scan_rejects_transient_prefix_before_negative_following_context(self):
        class Detector:
            config=dict(threshold=.5,wake_word_ru='бот')
            def score(self,audio): return float(len(audio)<=4800)
        self.assertFalse(scan(Detector(),np.ones(16000,np.float32))['detected'])


if __name__=='__main__': unittest.main()
