"""Two-process inference/mining regression. No training or model downloads."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/wake-validation'), str(ROOT / 'scripts/wakeword')]
import numpy as np
import torch
import torch.distributed as dist
from wake_train import BalancedSampler, evaluate, evaluate_shared


class AuditDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.features = torch.arange(size * 3, dtype=torch.float32).reshape(size, 3) / 8
        self.labels = torch.tensor([i % 2 for i in range(size)], dtype=torch.float32)
        self.seen = []

    def __len__(self): return len(self.labels)

    def __getitem__(self, index):
        self.seen.append(index)
        return self.features[index], self.labels[index]


class FixtureModel(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, feature): return feature.sum(dim=1) * self.scale - 1.


def worker(rank, directory):
    print(f'Rank {rank}: inference worker ready', flush=True)
    torch.set_num_threads(1)
    timeout = timedelta(seconds=30)
    backend = 'gloo'
    if os.name == 'nt':
        # Windows QA: explicitly choose loopback instead of hostname discovery.
        def create_gloo(store, process_rank, world, limit):
            options = dist.ProcessGroupGloo._Options()
            options._devices = [dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')]
            options._timeout = limit
            return dist.ProcessGroupGloo(store, process_rank, world, options)
        backend = 'gloo_loopback'
        dist.Backend.register_backend(backend, create_gloo, devices=['cpu'])
    dist.init_process_group(backend, store=dist.FileStore(str(directory / 'store'), 2),
                            rank=rank, world_size=2, timeout=timeout)
    print(f'Rank {rank}: process group ready', flush=True)
    try:
        device = torch.device('cpu')
        checked = []
        for size in [7, 1, 5]:
            for scale in [1., .9]:  # Raw and EMA-like validation forwards.
                model, data = FixtureModel(scale), AuditDataset(size)
                expected = evaluate(model, torch.utils.data.DataLoader(AuditDataset(size), batch_size=2), device)
                actual = evaluate_shared(model, data, 2, device)
                assert data.seen == list(range(rank, size, 2)), (rank, data.seen)
                for key in ['labels', 'scores', 'logits']:
                    np.testing.assert_allclose(actual[key], expected[key], atol=1e-7, rtol=0)
                np.testing.assert_allclose(actual['loss'], expected['loss'], atol=1e-6, rtol=0)
            checked.append(size)
        # Mining follows both validation collectives, then every sampler sees identical weights.
        data = AuditDataset(7)
        mined = evaluate_shared(FixtureModel(), data, 2, device)
        rows = [dict(language='ru' if i < 4 else 'en', label=i % 2, generator='fixture') for i in range(7)]
        sampler = BalancedSampler(rows, seed=123, rank=rank, world=2)
        sampler.mine(mined['scores'])
        weights = [None, None]
        dist.all_gather_object(weights, sampler.weights.tolist())
        np.testing.assert_array_equal(weights[0], weights[1])
        marker = torch.tensor(1.)
        dist.all_reduce(marker)
        assert marker.item() == 2.
        (directory / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank, sizes=checked, mining=True)), encoding='utf-8')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedEvaluationChecks(unittest.TestCase):
    def test_two_process_validation_empty_shard_and_mining(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temporary:
            directory = Path(temporary)
            processes, streams = [], []
            try:
                for rank in range(2):
                    stream = (directory / f'rank-{rank}.log').open('w', encoding='utf-8')
                    streams.append(stream)
                    # Avoid the Windows venv redirector: terminate/wait must own the actual worker.
                    executable = sys._base_executable if os.name == 'nt' else sys.executable
                    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
                    processes.append(subprocess.Popen([executable, str(Path(__file__).resolve()), '--worker',
                                                       str(rank), str(directory)], stdout=stream, stderr=stream, env=environment))
                deadline = time.monotonic() + 150
                while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
                    for process in processes:
                        try: process.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                for process in processes:
                    self.assertEqual(process.poll(), 0, '\n'.join(
                        file.read_text(encoding='utf-8') for file in directory.glob('*.log')))
                results = [json.loads((directory / f'rank-{rank}.json').read_text(encoding='utf-8')) for rank in range(2)]
                self.assertTrue(all(result['sizes'] == [7, 1, 5] and result['mining'] for result in results))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                for stream in streams: stream.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': worker(int(sys.argv[2]), Path(sys.argv[3]))
    else: unittest.main()
