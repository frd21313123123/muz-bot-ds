"""Adapt the two deployed acoustic models from the same leakage-checked local corpus."""
import argparse
import json
import os
from pathlib import Path
import time

from local_features import ROOT, cache_features, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / '.runtime/local-voice')
    parser.add_argument('--base', type=Path, help='Original INT8 model pair (defaults to the saved baseline, then models/)')
    parser.add_argument('--minutes', type=float, default=18)
    parser.add_argument('--threads', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.minutes <= 20 or not 1 <= args.threads <= 32:
        parser.error('Expected at most 20 minutes and 1–32 CPU encoder workers')
    start = time.monotonic()
    deadline = start + args.minutes * 60
    work = args.work.resolve()
    base = args.base or (work / 'baseline' if (work / 'baseline').is_dir() else ROOT / 'models')
    with (work / 'windows.jsonl').open(encoding='utf8') as src:
        rows = [json.loads(line) for line in src]
    if any(r['index'] != i for i, r in enumerate(rows)):
        raise ValueError('Inconsistent window manifest')
    digest = sha(work / 'audio.npy')
    caches = {task: cache_features(work, base, task, rows, digest, deadline, args.threads)
              for task in ('wake', 'command')}
    cache_seconds = time.monotonic() - start
    if args.cache_only:
        print(f'Encoder preparation: {cache_seconds:.1f} seconds', flush=True)
        return
    from local_adapter import fit_and_export
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required')
    torch.set_num_threads(2)
    report = dict(gpu=torch.cuda.get_device_name(), dataset_sha256=digest, encoder_seconds=cache_seconds)
    for task, cache in caches.items():
        report[task] = fit_and_export(work, base, task, rows, cache, deadline)
    report['total_seconds'] = time.monotonic() - start
    (work / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(f'Finished in {report["total_seconds"]:.1f} seconds on {report["gpu"]}', flush=True)


if __name__ == '__main__':
    main()
