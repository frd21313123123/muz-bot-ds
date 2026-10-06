"""Policy and adaptation boundary checks; no training or model downloads."""
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
import numpy as np
import torch
from wake_metrics import calibrate, point_for_config, threshold_audit
from wake_model import WhisperWakeModel
from build_wakeword_v7_notebook import build_v7
from build_wakeword_v6_notebook import build_v6


class V7Checks(unittest.TestCase):
    def test_monitoring_does_not_silently_add_source_fpr_constraint(self):
        silero = dict(language='ru',generator='silero',speaker='v5_5_ru/kseniya')
        piper = dict(language='ru',generator='piper',speaker='piper/irina')
        rows = [silero]*2000 + [piper]*4 + [silero,piper]
        prediction = dict(labels=[0]*2004+[1,1],scores=[.1]*2000+[.9]*4+[.99,.5])
        config = dict(target_fpr=.005,source_groups=True,threshold_diagnostics=True)
        legacy = point_for_config(prediction,rows,config)
        revised = point_for_config(prediction,rows,dict(config,source_group_fpr_constraint=False))
        self.assertGreater(legacy['threshold'],.9)
        self.assertEqual(legacy['metrics']['recall'],.5)
        self.assertEqual(revised['metrics']['recall'],1)
        self.assertLessEqual(revised['groups']['ru']['false_positive_rate_per_window'],.005)
        # Every source remains visible, including a deliberately failing tiny source.
        self.assertEqual(revised['source_groups']['ru|piper|piper']['fp'],4)
        self.assertEqual(revised['source_groups']['ru|piper|piper']['false_positive_rate_per_window'],1)
        self.assertEqual(revised['worst_source_recall'],1)
        self.assertTrue(all(not r['applied'] for r in revised['threshold_audit']['boundaries'] if r['scope']=='source'))

    def test_audit_language_constraint_and_negative_only_sources(self):
        labels=[0]*4000+[0]*4+[1,1]
        scores=[.1]*4000+[.9]*4+[.95,.99]
        languages=['en']*4000+['ru']*4+['ru','en']
        sources=['en|silero|v3_en']*4000+['ru|piper|piper']*4+['ru|silero|v4_ru','en|silero|v3_en']
        audit=threshold_audit(labels,scores,.005,languages,sources,False)
        threshold,metrics=calibrate(labels,scores,.005,languages)
        self.assertEqual(audit['threshold'],threshold)
        self.assertEqual(metrics['fp'],0)
        self.assertTrue(any(r['scope']=='language' and r['group']=='ru' and r['limits_threshold'] for r in audit['boundaries']))
        negative_only=next(r for r in audit['boundaries'] if r['group']=='ru|piper|piper')
        self.assertEqual(negative_only['negative_windows'],4)
        self.assertEqual(negative_only['allowed_fp'],0)
        self.assertFalse(negative_only['applied'])
        json.dumps(audit,allow_nan=False)

    def test_four_layer_adaptation_keeps_export_shape_and_changes_trainable_parameters(self):
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory:
            root=Path(directory); encoder=root/'whisper-encoder'; encoder.mkdir()
            config=WhisperConfig(d_model=16,encoder_layers=4,encoder_attention_heads=2,encoder_ffn_dim=32,
                                 num_mel_bins=80,max_source_positions=100,decoder_layers=1,decoder_attention_heads=2,decoder_ffn_dim=32)
            config._attn_implementation='eager'
            config.to_json_file(str(encoder/'encoder_config.json'))
            torch.save(WhisperEncoder(config).state_dict(),encoder/'encoder_state.pt')
            control=WhisperWakeModel(dict(work_dir=str(root),encoder_train_layers=1)).eval()
            adapted=WhisperWakeModel(dict(work_dir=str(root),encoder_train_layers=4)).eval()
            adapted.load_state_dict(control.state_dict())
            self.assertEqual([all(p.requires_grad for p in layer.parameters()) for layer in control.encoder.layers],[False,False,False,True])
            self.assertTrue(all(all(p.requires_grad for p in layer.parameters()) for layer in adapted.encoder.layers))
            self.assertEqual({k:v.shape for k,v in control.state_dict().items()},{k:v.shape for k,v in adapted.state_dict().items()})
            with torch.inference_mode():
                x=torch.zeros(2,1,80,200)
                self.assertTrue(torch.equal(control(x),adapted(x)))

    def test_notebook_executes_parameters_and_embedded_sources_with_v6_data_recipe(self):
        notebook=build_v7(output=None)
        def parameters(notebook,scratch):
            namespace={}
            cell=next(c for c in notebook['cells'] if 'parameters' in c['metadata'].get('tags',[]))
            source=cell['source']
            for version in [6,7]:
                source=source.replace(f'/kaggle/temp/wakeword-bot-v{version}',str(scratch/'work').replace('\\','/'))
                source=source.replace(f'/kaggle/working/wakeword-bot-v{version}',str(scratch/'output').replace('\\','/'))
            exec(compile(source,'parameters','exec'),namespace)
            return namespace
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory,patch.dict(os.environ):
            root=Path(directory)
            v7=parameters(notebook,root/'v7'); v6=parameters(build_v6(output=None),root/'v6')
            config=v7['CONFIG']; old=v6['CONFIG']
            for key in ['seed','piper_ru_splits','negative_exclusions','positive_bare_word','positive_per_voice',
                        'negative_per_voice','positive_tts_variants','negative_tts_variants','negative_kind_weights']:
                self.assertEqual(config[key],old[key],key)
            self.assertEqual(config['ensemble_members'],3)
            self.assertEqual(config['member_seed_stride'],0)
            self.assertEqual(config['recipe_version'],7)
            self.assertFalse(config['source_group_fpr_constraint'])
            self.assertTrue(config['source_groups'] and config['export_calibration_predictions'] and config['threshold_diagnostics'])
            self.assertEqual([config['member_overrides'][str(i)]['encoder_train_layers'] for i in range(3)],[1,4,4])
            self.assertEqual([config['member_overrides'][str(i)]['distill_weight'] for i in range(3)],[0,0,.4])
            self.assertEqual({config['member_overrides'][str(i)]['encoder_learning_rate'] for i in range(3)},{3e-6})
            self.assertEqual(config['teacher_repository'],'openai/whisper-large-v3')
            for cell in notebook['cells']:
                if any(tag.startswith('embedded-') for tag in cell['metadata'].get('tags',[])):
                    exec(compile(cell['source'],'embedded','exec'),v7)
            for name,expected in notebook['metadata']['wakeword']['source_sha256'].items():
                self.assertEqual(hashlib.sha256((root/'v7/work/code'/name).read_bytes()).hexdigest(),expected)
            self.assertEqual(notebook['metadata']['wakeword']['version'],7)
            self.assertFalse(any(cell.get('outputs') for cell in notebook['cells']))


if __name__=='__main__': unittest.main()
