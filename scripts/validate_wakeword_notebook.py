"""Execute the notebook on a small real-TTS dataset; never label smoke weights production-ready.

Uses the existing local PyTorch runtime, optional packages in .runtime/wake-validation,
one available GPU for neural TTS/training, plus two CPU Gloo processes to check DDP.
The unmodified handoff notebook requires two GPUs and remains output-free.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/wake-validation'))


def validate_static(notebook):
    import nbformat
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code': ast.parse(cell.source)
    assert all(not cell.get('outputs') and cell.get('execution_count') is None for cell in notebook.cells if cell.cell_type == 'code')
    for filename, expected in notebook.metadata.wakeword.source_sha256.items():
        assert hashlib.sha256((ROOT / 'scripts/wakeword' / filename).read_text(encoding='utf-8').encode()).hexdigest() == expected
    assert 'gsk_' not in json.dumps(notebook), 'No API credentials belong in the notebook'
    print('Structure, all code cells, embedded hashes and clean outputs: PASS', flush=True)


def main():
    import nbformat
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebooks/wakeword_bot_kaggle.ipynb')
    parser.add_argument('--reuse', help='Existing QA work directory (for repeated validation)')
    parser.add_argument('--ddp-only', action='store_true', help='Reuse an executed smoke notebook; recheck DDP with current training code')
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    validate_static(notebook)
    if args.static: return
    if notebook.metadata.wakeword.get('version', 0) >= 5:
        raise RuntimeError('V5+ require Kaggle T4 x2 and Large V3. Use --static here, then Run All in Kaggle; no local training is launched.')
    import torch
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    if not torch.cuda.is_available(): raise RuntimeError('Real neural TTS smoke requires a local GPU')
    work = Path(args.reuse).resolve() if args.reuse else ROOT / '.runtime' / f'wake-notebook-qa-{int(time.time())}'
    work.mkdir(parents=True, exist_ok=True)
    dependencies = str(ROOT / '.runtime/wake-validation')
    environment = dict(os.environ, PYTHONPATH=dependencies, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', USE_LIBUV='0',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    # Keep the local test kernel specification inside the ignored workspace.
    kernels = work / 'kernels'
    kernel = kernels / 'wake-qa'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                                 'display_name': 'Wake QA', 'language': 'python'}))
    manager = KernelManager(kernel_name='wake-qa', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    runtime = work / 'run'
    (runtime / 'tts').mkdir(parents=True, exist_ok=True)
    cached = ROOT / '.runtime/tts/v5_5_ru.pt'
    if cached.exists() and not (runtime / 'tts/v5_5_ru.pt').exists():
        shutil.copy2(cached, runtime / 'tts/v5_5_ru.pt')
    cached_tts = ROOT / '.runtime/wakeword-v2-local/tts'
    if cached_tts.exists(): shutil.copytree(cached_tts, runtime / 'tts', dirs_exist_ok=True)
    for cell in notebook.cells:
        tags = cell.metadata.get('tags', [])
        if 'parameters' in tags:
            cell.source = cell.source.replace('REQUIRE_TWO_GPUS = True', 'REQUIRE_TWO_GPUS = False')
            cell.source = cell.source.replace('SMOKE = False', 'SMOKE = True')
            cell.source = cell.source.replace("Path('/kaggle/working/wakeword-bot-v4')", repr(runtime))
            # repr(Path) is WindowsPath(...); use a portable string literal instead.
            cell.source = cell.source.replace(repr(runtime), f'Path({str(runtime)!r})')
            cell.source += '\nCONFIG["loader_workers"] = 0\nCONFIG_FILE.write_text(json.dumps(CONFIG), encoding="utf-8")\n'
        elif 'dependencies' in tags:
            source = cell.source
            start, end = source.index('subprocess.run('), source.index('    import torch') if '    import torch' in source else source.index('\nimport torch')
            cell.source = source[:start] + '# Local QA: dependencies already isolated; no package install in the kernel.\n' + source[end:]
        elif 'generate-data' in tags:
            cell.source = cell.source.replace('time.sleep(30)', 'time.sleep(5)')
        elif 'train-ddp' in tags and os.name == 'nt':
            # Windows' torchrun rendezvous is built without libuv; single-GPU QA
            # runs the same training entry point directly. Linux Kaggle stays unchanged.
            cell.source = cell.source.replace("'-m', 'torch.distributed.run', '--standalone',", '')
            cell.source = cell.source.replace("f'--nproc_per_node={GPU_COUNT}',", '')
        ast.parse(cell.source) if cell.cell_type == 'code' else None
    if args.ddp_only:
        assert (work / 'wakeword_bot_executed_SMOKE_ONLY.ipynb').exists(), 'Execute the smoke notebook first'
        shutil.copy2(ROOT / 'scripts/wakeword/wake_train.py', runtime / 'code/wake_train.py')
    else:
        try:
            client = NotebookClient(notebook, km=manager, timeout=1200, resources={'metadata': {'path': str(ROOT)}}, record_timing=True)
            client.execute(env=environment)
        finally:
            nbformat.write(notebook, work / 'wakeword_bot_executed_SMOKE_ONLY.ipynb')
        print('Actual neural TTS, GPU training, calibration, CPU ONNX and fresh-process inference: PASS', flush=True)
    # Inspect and test the exact embedded modules that executed in the notebook.
    sys.path.insert(0, str(runtime / 'code'))
    from wake_export import calibrate
    from wake_train import confusion
    labels, scores = [0, 0, 1, 1], [0.8, 0.8, 0.9, 0.95]
    threshold, metrics = calibrate(labels, scores, 0)
    assert threshold > .8 and metrics['fp'] == 0 and metrics['recall'] == 1
    assert confusion([0, 1], [0.2, 0.5], .5)['tp'] == 1
    config = json.loads((runtime / 'run_config.json').read_text(encoding='utf-8'))
    config.update(force_cpu=True, epochs=2, stop_after_epoch=1, resume_checkpoint=None, loader_workers=0, mining_interval=1)
    ddp = work / 'ddp-cpu'
    ddp.mkdir(exist_ok=True)
    shutil.copytree(runtime / 'dataset', ddp / 'dataset', dirs_exist_ok=True)
    if config.get('architecture') == 'whisper_encoder_v4':
        (ddp / 'whisper-encoder').mkdir(exist_ok=True)
        for name in ['encoder_config.json', 'encoder_state.pt']:
            shutil.copy2(runtime / 'whisper-encoder' / name, ddp / 'whisper-encoder' / name)
    config['work_dir'] = str(ddp)
    config_file = ddp / 'run_config.json'
    config_file.write_text(json.dumps(config))
    def ddp_run():
        import socket
        import uuid
        store_file = str(ddp / f'store-{uuid.uuid4().hex}')
        # Local Windows QA only: PyTorch's built-in Gloo constructor ignores
        # pg_options and chooses this machine's unusable hostname. Register the
        # same compiled Gloo backend with explicit loopback devices + FileStore.
        bootstrap = '''
import os, sys, runpy
import torch.distributed as dist
def create_gloo(store, rank, world, timeout):
    options = dist.ProcessGroupGloo._Options()
    options._devices = [dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')]
    options._timeout = timeout
    return dist.ProcessGroupGloo(store, rank, world, options)
dist.Backend.register_backend('gloo_loopback', create_gloo, devices=['cpu'])
original_init = dist.init_process_group
def initialize(backend, **kwargs):
    if backend == 'gloo':
        backend = 'gloo_loopback'
        rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        kwargs.update(store=dist.FileStore(os.environ['QA_GLOO_STORE'], world), rank=rank, world_size=world)
    return original_init(backend, **kwargs)
dist.init_process_group = initialize
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        with socket.socket() as port_probe:
            port_probe.bind(('127.0.0.1', 0))
            port = port_probe.getsockname()[1]
        children = []
        try:
            for rank in range(2):
                env = dict(environment, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                           MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), USE_LIBUV='0', QA_GLOO_STORE=store_file)
                command = [sys.executable]
                if os.name == 'nt': command += ['-c', bootstrap]
                command += [str(runtime / 'code/wake_train.py'), '--config', str(config_file)]
                children.append(subprocess.Popen(command, env=env))
            for process in children:
                if process.wait(timeout=300): raise RuntimeError('Two-process CPU DDP validation failed')
        finally:
            for process in children:
                if process.poll() is None: process.terminate(); process.wait(timeout=15)
    ddp_run()
    first = torch.load(ddp / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
    assert first['epoch'] == 0 and first['world_size'] == 2
    config.update(stop_after_epoch=2, resume_checkpoint=str(ddp / 'checkpoints/latest.pt'))
    config_file.write_text(json.dumps(config))
    ddp_run()
    resumed = torch.load(ddp / 'checkpoints/latest.pt', weights_only=False, map_location='cpu')
    assert resumed['epoch'] == 1 and len(resumed['history']) == 2
    print('Two-process DDP backward + checkpoint resume + threshold ties: PASS', flush=True)
    summary = {'notebook': str(work / 'wakeword_bot_executed_SMOKE_ONLY.ipynb'),
               'local_gpus': torch.cuda.device_count(), 'smoke_only': True,
               'gpu_2_hardware_validation': 'not performed locally; Kaggle T4 x2 required',
               'ddp_two_cpu_processes': 'passed', 'source_notebook': str(ROOT / 'notebooks/wakeword_bot_kaggle.ipynb')}
    (work / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
