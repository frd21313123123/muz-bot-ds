"""Run the embedded V3 notebook locally: full Action only, with persistent logs/state."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".runtime/training/lia-action-v3"
NOTEBOOK = ROOT / ".runtime/datasets/lia_kaggle_v3_embedded.ipynb"
REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
CACHE = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots" / REVISION
sys.path.insert(0, str(ROOT / ".runtime/notebook-validation"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(WORK / "matplotlib-cache")
    os.environ["HF_HOME"] = str(ROOT / ".runtime/voice/cache")
    os.environ["USE_TF"] = "0"
    if not (CACHE / "model.safetensors").is_file():
        raise FileNotFoundError("Pinned base weights are not in the prepared local cache")
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    namespace, figures = {}, []
    def save_plot():
        path = WORK / "diagnostics" / f"action-figure-{len(figures) + 1}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(path, dpi=140, bbox_inches="tight")
        figures.append(str(path)); plt.close()
    plt.show = save_plot
    finish_only = "--finish-only" in sys.argv
    cpu_finish = "--cpu" in sys.argv
    if cpu_finish and not finish_only: raise ValueError("--cpu is for final checks only")
    status = {"state": "starting", "pid": os.getpid(), "stage": "action", "full_training": True,
              "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
              "base_revision": REVISION, "precision": "FP16 autocast; FP32 weights", "workdir": str(WORK),
              "started_at_unix": time.time(), "bot_model_changed": False}
    if finish_only:
        status.update(recovery="finish completed training without optimizer restore or more epochs",
                      checks_device="cpu" if cpu_finish else "cuda")
    def save_status():
        temporary = WORK / "run-status.json.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(WORK / "run-status.json")
    save_status()
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code": continue
            tags, source = cell["metadata"].get("tags", []), cell["source"]
            ast.parse(source)
            if "install" in tags or "train-semantic" in tags: continue
            if "parameters" in tags:
                source = source.replace('WORKDIR = Path("/kaggle/working/lia-v3")', f'WORKDIR = Path({str(WORK)!r})')
                source = source.replace('BASE_MODEL = "convaiinnovations/laya-multilingual"', f'BASE_MODEL = {str(CACHE)!r}')
                source = source.replace(f'BASE_REVISION = "{REVISION}"', 'BASE_REVISION = None')
                source = source.replace('TRAIN_SEMANTIC = True', 'TRAIN_SEMANTIC = False')
            status.update(state="running", phase=tags[0] if tags else cell["id"])
            save_status()
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "EXECUTE", cell["id"], tags, flush=True)
            exec(compile(source, cell["id"], "exec"), namespace)
            if "environment" in tags:
                if cpu_finish:
                    namespace["DEVICE"] = "cpu"
                    print("RECOVERY: final checks on CPU; trained weights preserved", flush=True)
                original_load = namespace["laya"].load
                def load_fp16(*args, **kwargs):
                    if finish_only and args and args[0] == namespace["BASE_MODEL"]:
                        args = (str(WORK / "candidates/laya-muz-bot-controls-v3"), *args[1:])
                        kwargs.pop("revision", None)
                    agent = original_load(*args, **kwargs)
                    if agent.device.type == "cuda": agent.dtype = namespace["torch"].float16
                    return agent
                namespace["laya"].load = load_fp16
                status["gpu"] = namespace["torch"].cuda.get_device_name(0)
            if "contract" in tags:
                original_indices = namespace["v3"].balanced_indices
                def sampling_progress(items, draws, seed):
                    print("FULL ACTION EPOCH TRAINING: draws=", draws, "seed=", seed, flush=True)
                    return original_indices(items, draws, seed)
                namespace["v3"].balanced_indices = sampling_progress
                if finish_only:
                    def finished_training(agent, train_items, validation_items, directory, model_name, state_path, **kwargs):
                        saved = namespace["torch"].load(state_path, map_location="cpu", weights_only=True)
                        expected = {"signature": kwargs["signature"], "epochs": kwargs["epochs"], "draws": kwargs["draws"],
                            "batch": kwargs["batch_size"], "accumulation": kwargs["accumulation"], "head_lr": kwargs["head_lr"],
                            "encoder_lr": kwargs["encoder_lr"], "layers": kwargs["last_layers"], "seed": kwargs["seed"]}
                        if saved["config"] != expected or saved["epoch"] != kwargs["epochs"]:
                            raise ValueError("Finish-only requires all epochs completed and original data/parameters")
                        namespace["legacy"].configure_trainable(agent, kwargs["last_layers"])
                        info = {"history": saved["history"], "baseline_validation": saved["baseline"], "selected_score": saved["best"],
                            "sampling_audit": saved["audits"], "last_encoder_layers": kwargs["last_layers"],
                            "sampling": "equal task/class; equal negative subtype; equal family/text/variant",
                            "trainable_encoder_parameters": sum(p.numel() for n,p in agent.model.named_parameters() if p.requires_grad and n.startswith("encoder.")),
                            "trainable_head_parameters": sum(p.numel() for n,p in agent.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")),
                            "optimizer_updates": sum(r["optimizer_updates"] for r in saved["history"]),
                            "skipped_updates": sum(r["skipped_updates"] for r in saved["history"]), "resumed_from_epoch": saved["epoch"],
                            "recovery": "final evaluation of saved best checkpoint; no training steps repeated"}
                        del saved
                        for parameter in agent.model.parameters(): parameter.requires_grad_(False)
                        agent.model.eval()
                        print("RECOVERY: saved training state verified; completed epochs", kwargs["epochs"], flush=True)
                        return info
                    namespace["v3"].train = finished_training
            if "stage-functions" in tags:
                original_raw_logits = namespace["training"].raw_logits
                def measured_logits(agent, items, batch_size=8):
                    status["phase"] = "evaluation_questions_" + str(len(items))
                    save_status()
                    print("EVALUATION START", len(items), "questions on", str(agent.device), flush=True)
                    values = original_raw_logits(agent, items, batch_size)
                    print("EVALUATION DONE", len(items), "questions", flush=True)
                    return values
                namespace["training"].raw_logits = measured_logits
                original_runtime_evaluation = namespace["training"].runtime_evaluation
                def measured_pipeline(agent, commands, worker, bridge_script):
                    status["phase"] = "pipeline_commands_" + str(len(commands))
                    save_status()
                    print("PIPELINE START", len(commands), "commands on", str(agent.device), flush=True)
                    values = original_runtime_evaluation(agent, commands, worker, bridge_script)
                    print("PIPELINE DONE", len(commands), "commands", flush=True)
                    return values
                namespace["training"].runtime_evaluation = measured_pipeline
        report = namespace["reports"]["action"]
        assert not report["smoke_only"] and set(namespace["reports"]) == {"action"}
        status.update(state="completed", completed_at_unix=time.time(),
                      qualified=report["qualified_on_synthetic_holdouts"],
                      archive=report["export_archive"], figures=figures,
                      optimizer_updates=report["training"]["optimizer_updates"])
        print("FULL ACTION COMPLETED", json.dumps(status, ensure_ascii=False), flush=True)
    except BaseException as error:
        status.update(state="failed", failed_at_unix=time.time(), error=type(error).__name__, message=str(error))
        traceback.print_exc()
        raise
    finally:
        save_status()


if __name__ == "__main__": main()
