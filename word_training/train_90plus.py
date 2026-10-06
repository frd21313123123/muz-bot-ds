"""
Train a new high-accuracy Russian wake-word detector ("бот")
targeting >= 90% recall while preserving low false alarms.
Uses all available acoustic voices (Silero, Piper, MMS, Kokoro, etc.) in training,
with utterance-level held-out validation, calibration and test sets.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / '.runtime/wake-validation'),
    str(ROOT / 'scripts/wakeword'),
    str(Path(__file__).parent)
]

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
import torch
from torch.utils.data import Sampler
import wake_train
from improve import PackedFeatureDataset
from train import cache_files
from wake_data import atomic_json


class PatchedBalancedSampler(Sampler):
    """Balanced multi-language & multi-generator sampler supporting Russian, Ukrainian, English & Noise."""
    def __init__(self, rows, seed, rank=0, world=1):
        self.rows, self.seed, self.rank, self.world, self.epoch = rows, seed, rank, world, 0
        self.keys = [(r['language'], r['label'], r['generator']) for r in rows]
        counts = {}
        for key in self.keys:
            counts[key] = counts.get(key, 0) + 1
        groups = {}
        for language, label, engine in counts:
            groups[(language, label)] = groups.get((language, label), 0) + 1
        weights = []
        for key in self.keys:
            language, label, _ = key
            if label:
                language_mass = {'ru': .55, 'uk': .15, 'en': .30}.get(language, 0.05)
            else:
                language_mass = {'ru': .50, 'uk': .15, 'en': .25, 'none': .10}.get(language, 0.05)
            weights.append(language_mass / max(1e-6, counts[key] * groups[(language, label)]))
        self.base = torch.tensor(weights, dtype=torch.double)
        for label in [0, 1]:
            selected = torch.tensor([r['label'] == label for r in rows])
            s = self.base[selected].sum()
            if s > 0:
                self.base[selected] *= .5 / s
        self.weights = self.base.clone()
        self.size = (len(rows) + world - 1) // world

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        w = torch.nan_to_num(self.weights, nan=1e-8, posinf=1.0, neginf=1e-8)
        w = torch.clamp(w, min=1e-8)
        indices = torch.multinomial(w, self.size * self.world, replacement=True, generator=generator).tolist()
        return iter(indices[self.rank::self.world])

    def mine(self, scores):
        scores = np.asarray(scores)
        difficulty = torch.tensor([1 + min(2., 2 * (1 - s if r['label'] else s)) for r, s in zip(self.rows, scores, strict=True)], dtype=torch.double)
        self.weights = self.base * difficulty
        for key in set(self.keys):
            selected = torch.tensor([k == key for k in self.keys])
            w_sum = self.weights[selected].sum()
            b_sum = self.base[selected].sum()
            if w_sum > 0 and b_sum > 0:
                self.weights[selected] *= b_sum / w_sum
        self.weights = torch.nan_to_num(self.weights, nan=1e-8, posinf=1.0, neginf=1e-8)
        self.weights = torch.clamp(self.weights, min=1e-8)


def prepare_balanced_splits(source_manifest: Path, target_manifest: Path):
    """
    Split utterances deterministically by source_id so all voices are learned
    in train, while testing on completely unseen held-out utterances.
    """
    print(f"Loading source manifest from {source_manifest}...")
    rows = [json.loads(line) for line in source_manifest.read_text(encoding='utf-8').splitlines()]
    
    source_dataset_dir = source_manifest.parent
    for row in rows:
        for key in ['feature', 'wav']:
            if not os.path.isabs(row[key]):
                row[key] = str((source_dataset_dir / row[key]).resolve())

    # Auxiliary voices (Kokoro, Piper UK) stay 100% in train.
    # Russian voices (Silero, Piper RU, MMS) and real_speech/noise are partitioned:
    # 75% train, 10% validation, 5% calibration, 10% test
    def assign_split(r: dict) -> str:
        if r['generator'] in ['kokoro', 'piper_uk']:
            return 'train'
        h = int(hashlib.md5(r['source_id'].encode('utf-8')).hexdigest()[:8], 16) % 100
        if h < 75:
            return 'train'
        elif h < 85:
            return 'validation'
        elif h < 90:
            return 'calibration'
        else:
            return 'test'

    for row in rows:
        row['split'] = assign_split(row)

    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = target_manifest.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    temp.replace(target_manifest)

    # Print summary
    counts = {}
    for r in rows:
        s = r['split']
        counts.setdefault(s, {'pos': 0, 'neg': 0})
        if r['label'] == 1:
            counts[s]['pos'] += 1
        else:
            counts[s]['neg'] += 1
    
    print("\nSplit statistics:")
    for s, c in counts.items():
        total = c['pos'] + c['neg']
        print(f"  {s:12s}: Total={total:5d} (Pos={c['pos']:5d}, Neg={c['neg']:5d}, Pos%={c['pos']/total*100:4.1f}%)")
    return rows


def make_config(work_dir: Path, source_run_dir: Path):
    return dict(
        work_dir=str(work_dir),
        output_dir=str(work_dir.parent / (work_dir.name + '-output')),
        dataset_dir=str(work_dir / 'dataset'),
        encoder_dir=str(work_dir / 'whisper-encoder'),
        wake_word_ru='бот',
        wake_word_en='bot',
        use_english=True,
        standalone_only=False,
        recipe_version='bot-v5-acc90',
        seed=202610065,
        architecture='whisper_encoder_v4',
        encoder_train_layers=4,
        encoder_learning_rate=3e-6,
        learning_rate=0.0003,
        batch_per_gpu=64,
        epochs=12,
        min_epochs=6,
        patience=5,
        loader_workers=0,
        cpu_threads=4,
        mixed_precision=True,
        ema=True,
        balanced_sampling=True,
        preload_features=False,
        mining_interval=4,
        ensemble_members=1,
        member_overrides={'0': dict(distill_weight=0, encoder_train_layers=4, encoder_learning_rate=3e-6)},
        target_fpr=0.005,
        target_recall=0.90,
        source_groups=False,
        source_group_fpr_constraint=False,
        threshold_diagnostics=True,
        export_sample_metadata=True,
        export_calibration_predictions=True,
        quantization_recall_tolerance=0.01,
        smoke=False,
        persist_latest=True,
        resume_checkpoint=None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='bot-v5-acc90', help='Run name')
    parser.add_argument('--source-run', default='bot-data-v4', help='Source data run name')
    parser.add_argument('--initial', default='bot-binary-v3/checkpoints/best.pt', help='Initial weights checkpoint')
    parser.add_argument('--export-only', action='store_true', help='Skip training and export model')
    args = parser.parse_args()

    work = ROOT / 'word_training/runs' / args.run
    source_run = ROOT / 'word_training/runs' / args.source_run
    work.mkdir(parents=True, exist_ok=True)
    (work / 'dataset').mkdir(parents=True, exist_ok=True)

    # 1. Prepare dataset manifest
    manifest = work / 'dataset/manifest.jsonl'
    prepare_balanced_splits(source_run / 'dataset/manifest.jsonl', manifest)

    # 2. Copy/link whisper-encoder
    cache_files(source_run / 'whisper-encoder', work / 'whisper-encoder')

    # 3. Build config
    config = make_config(work, source_run)
    atomic_json(work / 'run_config.json', config)

    # 4. Patch FeatureDataset and BalancedSampler
    wake_train.FeatureDataset = PackedFeatureDataset
    wake_train.BalancedSampler = PatchedBalancedSampler
    import wake_package_v5
    wake_package_v5.FeatureDataset = PackedFeatureDataset

    # 5. Initialize model from pre-trained checkpoint
    initial_ckpt_path = (ROOT / 'word_training/runs' / args.initial).resolve()
    print(f"Loading initial model weights from {initial_ckpt_path}...")
    orig_make_model = wake_train.make_model
    
    def initialized_make_model(configuration):
        m = orig_make_model(configuration)
        saved = torch.load(initial_ckpt_path, map_location='cpu', weights_only=False)
        m.load_state_dict(saved['model'], strict=False)
        return m

    wake_train.make_model = initialized_make_model
    import wake_model
    wake_model.make_model = initialized_make_model

    # 6. Run training
    if not args.export_only:
        print("\n" + "=" * 60)
        print("🚀 Starting training for high-recall model (target >= 90%)...")
        print("=" * 60)
        with torch.inference_mode(False), torch.enable_grad():
            wake_train.train(config)

    # 7. Export student and package
    print("\n" + "=" * 60)
    print("📦 Exporting student model to ONNX & evaluating test set...")
    print("=" * 60)
    from wake_package_v5 import export_student, package, quality_gate

    export_student(config)
    reports, selection = package(config)

    report = reports[selection['selected']]
    gate = quality_gate(report['test'], {'ru': report['test_groups']['language']['ru']}, target_recall=0.90, target_fpr=0.005)
    report['quality_gate'] = gate

    output = Path(config['output_dir'])
    atomic_json(output / 'training_report.json', report)
    for name in ['training_history.json', 'training_run.json', 'run_config.json']:
        if (work / name).exists():
            shutil.copy2(work / name, output / name)

    summary = dict(
        test=report['test'],
        quality_gate=gate,
        model_bytes=report['model_bytes'],
        archive=str(output / 'wake-model.zip'),
        word='бот',
        task='whole_word_anywhere',
        format=selection['selected']
    )
    atomic_json(output / 'output_summary.json', summary)

    # 8. Unpack model for immediate inference
    model_export_dir = work / 'wake-model'
    model_export_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(output / 'wake-model.zip', 'r') as z:
        z.extractall(model_export_dir)
    print(f"Model extracted to {model_export_dir}")

    print("\n" + "=" * 60)
    print("🎯 FINAL TRAINING & TEST SUMMARY:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == '__main__':
    main()
