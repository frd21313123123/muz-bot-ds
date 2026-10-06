"""Train-only knowledge transfer from a supervised Whisper Large V3 keyword teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from wake_data import atomic_json
from wake_tts import file_sha256, prepare_whisper_encoder


def distillation_loss(student, teacher, labels, temperature=2.):
    """Bernoulli KL; disagreement never replaces annotation with a teacher's guess."""
    if temperature <= 0:
        raise ValueError('Distillation temperature must be positive')
    teacher = teacher.detach().float()
    probability = teacher.sigmoid()
    agreement = (probability >= .5) == labels.bool()
    weight = (2 * probability - 1).abs() * agreement.float()
    soft = (teacher / temperature).sigmoid().clamp(1e-6, 1 - 1e-6)
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(student.float() / temperature, soft, reduction='none')
    entropy = -(soft * soft.log() + (1 - soft) * (1 - soft).log())
    return ((cross_entropy - entropy).clamp_min(0) * weight).mean() * temperature ** 2


class DistillationDataset(Dataset):
    def __init__(self, dataset, targets_file):
        self.dataset, self.rows, self.root = dataset, dataset.rows, dataset.root
        if any(row['split'] != 'train' for row in self.rows):
            raise ValueError('Distillation targets are training-only')
        with np.load(targets_file, allow_pickle=False) as saved:
            self.logits = saved['logits'].copy()
            ids = saved['ids'].tolist()
            feature_hashes = saved['student_feature_sha256'].tolist()
            expected_manifest = str(saved['manifest_sha256'].item())
        if expected_manifest != file_sha256(self.root / 'manifest.jsonl'):
            raise ValueError('Teacher cache belongs to another dataset manifest')
        if ids != [row['id'] for row in self.rows] or self.logits.shape != (len(self.rows),) or not np.isfinite(self.logits).all():
            raise ValueError('Teacher cache IDs or logits are invalid')
        for row, expected in zip(self.rows, feature_hashes, strict=True):
            if expected != file_sha256(self.root / row['feature']):
                raise ValueError('Student feature changed after teacher cache generation')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        feature, label = self.dataset[index]
        return feature, label, torch.tensor(self.logits[index], dtype=torch.float32)


def prepare_teacher(config):
    """Teacher sees only train/validation; its mel-128 never enters the mel-80 student."""
    from wake_audio import WhisperMelFrontend, read_wav
    work = Path(config['work_dir'])
    teacher_work = work / 'teacher'
    provenance = prepare_whisper_encoder(teacher_work, config['teacher_repository'], config.get('teacher_revision'))
    frontend = WhisperMelFrontend(teacher_work / 'whisper-encoder')
    student_rows = [json.loads(line) for line in (work / 'dataset/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    destination = teacher_work / 'dataset'
    (destination / 'features').mkdir(parents=True, exist_ok=True)
    records = []
    for row in student_rows:
        if row['split'] not in ['train', 'validation']: continue
        feature = destination / row['feature']
        # Always regenerate: source audio/annotation edits cannot silently reuse stale mel-128.
        np.save(feature, frontend(read_wav(work / 'dataset' / row['wav'])))
        records.append(row)
    (destination / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    teacher_config = dict(config, work_dir=str(teacher_work), encoder_dir=None, dataset_dir=None,
                         ensemble_members=1, member_overrides={}, teacher_targets=None, distill_weight=0,
                         encoder_train_layers=config.get('teacher_train_layers', 2),
                         encoder_learning_rate=config.get('teacher_encoder_lr', 3e-6), learning_rate=3e-4,
                         batch_per_gpu=config.get('teacher_batch_per_gpu', 16), preload_features=False, ema=False,
                         epochs=2 if config.get('smoke') else config.get('teacher_epochs', 18),
                         min_epochs=1 if config.get('smoke') else 6, patience=6, mining_interval=6,
                         persist_latest=False, resume_checkpoint=None)
    atomic_json(teacher_work / 'run_config.json', teacher_config)
    atomic_json(work / 'teacher_provenance.json', provenance)
    print(f'Teacher: {provenance["repository"]}, revision {provenance["revision"]}; '
          f'{len(records)} train/validation mel-{frontend.filters.shape[0]} windows', flush=True)
    return teacher_config


@torch.inference_mode()
def cache_teacher(config, rank=0, world=1):
    from wake_train import FeatureDataset
    from wake_model import make_model
    work = Path(config['work_dir'])
    teacher_work = work / 'teacher'
    teacher_config = json.loads((teacher_work / 'run_config.json').read_text(encoding='utf-8'))
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and not config.get('force_cpu') else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(rank)
    torch.set_num_threads(config.get('cpu_threads', 2))
    checkpoint = teacher_work / 'checkpoints/best.pt'
    model = make_model(teacher_config)
    model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location='cpu')['model'])
    model.to(device).eval()
    dataset = FeatureDataset(teacher_work / 'dataset', 'train')
    indices = list(range(rank, len(dataset), world))
    scores = []
    for feature, _ in DataLoader(torch.utils.data.Subset(dataset, indices),
                                batch_size=config.get('teacher_batch_per_gpu', 16), num_workers=0):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            scores.extend(model(feature.to(device)).float().cpu().tolist())
    np.savez(work / f'teacher-targets-rank-{rank}.npz', indices=np.array(indices), logits=np.array(scores, np.float32),
             teacher_checkpoint_sha256=np.array(file_sha256(checkpoint)))
    print(f'Teacher targets rank {rank}: {len(scores)} TRAIN windows on {device}', flush=True)


def merge_teacher(config, world):
    from wake_train import FeatureDataset, confusion
    work = Path(config['work_dir'])
    dataset = FeatureDataset(work / 'dataset', 'train')
    logits = np.full(len(dataset), np.nan, np.float32)
    counts = np.zeros(len(dataset), np.int32)
    checkpoint_hash = file_sha256(work / 'teacher/checkpoints/best.pt')
    for rank in range(world):
        with np.load(work / f'teacher-targets-rank-{rank}.npz', allow_pickle=False) as shard:
            indices = shard['indices']
            if str(shard['teacher_checkpoint_sha256'].item()) != checkpoint_hash:
                raise ValueError('Teacher checkpoint changed between cache shards')
            if len(indices) != len(shard['logits']) or np.any(indices < 0) or np.any(indices >= len(dataset)):
                raise ValueError('Teacher cache indices are invalid')
            logits[indices] = shard['logits']
            np.add.at(counts, indices, 1)
    if not np.all(counts == 1) or not np.isfinite(logits).all():
        raise ValueError('Teacher cache must cover each train row exactly once')
    file = work / 'teacher-targets.npz'
    np.savez_compressed(file, ids=np.array([r['id'] for r in dataset.rows]), logits=logits,
                        student_feature_sha256=np.array([file_sha256(dataset.root / r['feature']) for r in dataset.rows]),
                        manifest_sha256=np.array(file_sha256(dataset.root / 'manifest.jsonl')),
                        teacher_checkpoint_sha256=np.array(checkpoint_hash))
    labels = np.array([r['label'] for r in dataset.rows])
    probabilities = torch.tensor(logits).sigmoid().numpy()
    report = dict(teacher_repository=config['teacher_repository'], teacher_checkpoint_sha256=checkpoint_hash,
                  targets_sha256=file_sha256(file), target_rows=len(dataset), splits=['train'],
                  teacher_train_diagnostic=confusion(labels, probabilities),
                  excluded_disagreements=int(np.sum((probabilities >= .5) != labels.astype(bool))),
                  interpretation='Train diagnostics only, not independent teacher accuracy. Human/synthetic hard labels retained.')
    atomic_json(work / 'distillation_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--action', choices=['prepare', 'cache', 'merge'], required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.action == 'prepare': prepare_teacher(config)
    elif args.action == 'cache': cache_teacher(config, args.rank, args.world)
    else: merge_teacher(config, args.world)
