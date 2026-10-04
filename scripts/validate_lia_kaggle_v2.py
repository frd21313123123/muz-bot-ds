"""Execute the V2 notebook top to bottom, using cached weights and bounded smoke data."""
import ast
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime/notebook-validation"))


def main():
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime/kaggle-v2-validation/matplotlib-cache")
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    notebook_path = ROOT / ".runtime/datasets/lia_kaggle_v2_embedded.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    work = ROOT / ".runtime/kaggle-v2-validation"
    work.mkdir(parents=True, exist_ok=True)
    cached = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    namespace, figures = {}, []
    def save_plot():
        path = work / f"figure-{len(figures) + 1}.png"
        plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
        figures.append(str(path))
        plt.close()
    plt.show = save_plot
    start = time.monotonic()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        source, tags = cell["source"], cell["metadata"].get("tags", [])
        ast.parse(source)
        if "install" in tags: continue  # Installed pinned bot dependencies, no network.
        if "parameters" in tags:
            source = source.replace('WORKDIR = Path("/kaggle/working/lia-v2")', f'WORKDIR = Path({str(work)!r})')
            source = source.replace('SMOKE_TEST = False', 'SMOKE_TEST = True')
            source = source.replace('BASE_MODEL = "convaiinnovations/laya-multilingual"', f'BASE_MODEL = {str(cached)!r}')
            source = source.replace('BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"', 'BASE_REVISION = None')
        print("Execute", cell["id"], tags, flush=True)
        exec(compile(source, cell["id"], "exec"), namespace)
        if "environment" in tags and "--fp16" in sys.argv:
            original_load = namespace["laya"].load
            def load_fp16(*args, **kwargs):
                agent = original_load(*args, **kwargs)
                if agent.device.type == "cuda": agent.dtype = namespace["torch"].float16
                return agent
            namespace["laya"].load = load_fp16
    assert set(namespace["reports"]) == {"action", "planner"}
    assert all(r["checkpoint_roundtrip"] == "passed" and r["smoke_only"] for r in namespace["reports"].values())
    assert not list(work.glob("laya-muz-bot-*.zip")), "Smoke weights must not be exported"
    assert namespace["DATA_MANIFEST"]["bot_log_rows"] == 44
    result = {"schema": "nbformat.validate", "mode": "GPU smoke, partial encoder backwards for both stages",
              "full_training_executed": False, "install_cell": "skipped, existing pinned packages",
              "checkpoint_roundtrip": "passed for action and planner", "smoke_export_blocked": True,
              "data_integrity_and_joint_split": "passed", "figures": figures, "seconds": round(time.monotonic() - start, 1)}
    result["forced_fp16_for_t4_path"] = "--fp16" in sys.argv
    (work / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
