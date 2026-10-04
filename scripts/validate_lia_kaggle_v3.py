"""Execute V3 notebook with cached weights, GPU FP16 smoke, and actual resume load."""
import ast
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime/notebook-validation"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    work = ROOT / ".runtime/kaggle-v3-validation-final"
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib-cache")
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = ROOT / ".runtime/datasets/lia_kaggle_v3_embedded.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    work.mkdir(parents=True, exist_ok=True)
    cached = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    namespace, figures = {}, []
    def save_plot():
        filename = work / f"figure-{len(figures) + 1}.png"
        plt.gcf().savefig(filename, dpi=120, bbox_inches="tight"); figures.append(str(filename)); plt.close()
    plt.show = save_plot
    start = time.monotonic()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        source, tags = cell["source"], cell["metadata"].get("tags", [])
        ast.parse(source)
        if "install" in tags: continue
        if "parameters" in tags:
            source = source.replace('WORKDIR = Path("/kaggle/working/lia-v3")', f'WORKDIR = Path({str(work)!r})')
            source = source.replace('SMOKE_TEST = False', 'SMOKE_TEST = True')
            source = source.replace('BASE_MODEL = "convaiinnovations/laya-multilingual"', f'BASE_MODEL = {str(cached)!r}')
            source = source.replace('BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"', 'BASE_REVISION = None')
        if "data-loading" in tags:
            source = source.replace('EPOCHS = 1; ACTION_DRAWS', 'EPOCHS = 2; ACTION_DRAWS')
        print("Execute", cell["id"], tags, flush=True)
        exec(compile(source, cell["id"], "exec"), namespace)
        if "environment" in tags and "--fp16" in sys.argv:
            original_load = namespace["laya"].load
            def load_fp16(*args, **kwargs):
                agent = original_load(*args, **kwargs)
                if agent.device.type == "cuda": agent.dtype = namespace["torch"].float16
                return agent
            namespace["laya"].load = load_fp16
        if "contract" in tags:
            original_train = namespace["v3"].train
            def one_epoch(*args, **kwargs):
                return original_train(*args, **kwargs, _max_new_epochs=1)
            namespace["v3"].train = one_epoch
    assert set(namespace["reports"]) == {"action", "semantic"}
    for stage, report in namespace["reports"].items():
        assert report["smoke_only"] and report["checkpoint_roundtrip"] == "passed"
        assert report["training"]["optimizer_updates"] > 0
        assert report["training"]["last_encoder_layers"] == 4
        assert report["training"]["trainable_encoder_parameters"] > 0
    assert not list(work.glob("laya-muz-bot-*.zip")), "Do not publish smoke weights"
    # Real restoration AND another epoch with optimizer/scaler/scheduler restored for both stages.
    namespace["v3"].train = original_train
    for stage, records, draws in (("action", namespace["action_records"], namespace["ACTION_DRAWS"]),
                                  ("semantic", namespace["semantic_records"], namespace["SEMANTIC_DRAWS"])):
        resumed = namespace["run_stage"](stage, records, draws)
        assert resumed["training"]["resumed_from_epoch"] == 1
        assert resumed["training"]["optimizer_updates"] > namespace["reports"][stage]["training"]["optimizer_updates"]
    # Last cell recovers from files with no WORKDIR/reports globals.
    finish = next(c["source"] for c in notebook["cells"] if "diagnostics" in c.get("metadata", {}).get("tags", []))
    finish = finish.replace('"/kaggle/working/lia-v3"', repr(str(work)))
    exec(compile(finish, "fresh-final-cell", "exec"), {})
    result = {"schema": "nbformat.validate", "mode": "GPU short training; actual partial encoder backward for both stages",
        "forced_fp16_for_t4_path": "--fp16" in sys.argv, "full_training_executed": False,
        "install_cell": "skipped, existing pinned packages", "checkpoint_roundtrip": "passed for both models",
        "resume_optimizer_scaler_rng": "passed", "empty_globals_final_cell": "passed",
        "policy_fixture_count": namespace["policy_report"]["fixtures"], "data_integrity_and_split": "passed",
        "bot_logs_preserved": namespace["DATA_MANIFEST"]["all_bot_log_events"],
        "figures": figures, "seconds": round(time.monotonic() - start, 1)}
    (work / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
