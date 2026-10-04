"""Run all computational notebook cells on the local cached model, without installs.

Artificial smoke data only. Outputs are saved in ignored .runtime, never shipped
as quality measurements. Optional plots require matplotlib in the chosen Python.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation_packages = ROOT / ".runtime/notebook-validation"
    if validation_packages.is_dir():
        sys.path.insert(0, str(validation_packages))
    notebook = json.loads((ROOT / "notebooks/laya_training_colab_kaggle.ipynb").read_text(encoding="utf-8"))
    schema = "structural/syntax only"
    if importlib.util.find_spec("nbformat"):
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        schema = "nbformat.validate passed"
    marker = json.loads((ROOT / ".runtime/voice/ready.json").read_text(encoding="utf-8"))
    cached = ROOT / ".runtime/voice/cache/hub/models--convaiinnovations--laya-multilingual/snapshots" / marker["laya_revision"]
    if not cached.is_dir():
        raise FileNotFoundError("Prepare the multilingual base cache with npm run setup:voice first")
    namespace = {}
    workdir = ROOT / ".runtime/notebook-smoke"
    figures = []
    start = time.monotonic()
    skipped = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse(cell["source"])
        tags = cell["metadata"].get("tags", [])
        if "install" in tags:
            skipped.append({"cell": i, "reason": "Use already installed bot dependencies; no network/install"})
            continue
        if "plots" in tags and importlib.util.find_spec("matplotlib") is None:
            skipped.append({"cell": i, "reason": "matplotlib unavailable; plot syntax checked, rendering unverified"})
            continue
        if "plots" in tags:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            def save_plot():
                path = workdir / f"figure-{len(figures) + 1}.png"
                plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
                figures.append(str(path))
                plt.close()
            plt.show = save_plot
        print(f"Executing cell {i}", flush=True)
        source = cell["source"]
        if "parameters" in tags:
            source = source.replace('SMOKE_TEST = False', 'SMOKE_TEST = True')
            source = source.replace('WORKDIR = Path("/kaggle/working/lia-training" if Path("/kaggle/working").exists() else "/content/lia-training")',
                                    f'WORKDIR = Path({str(workdir)!r})')
        exec(compile(source, f"notebook-cell-{i}", "exec"), namespace)
        if "parameters" in tags:
            namespace.update(SMOKE_TEST=True, WORKDIR=workdir, BASE_MODEL=str(cached), BASE_REVISION=None)
            workdir.mkdir(parents=True, exist_ok=True)
    result = {"elapsed_seconds": round(time.monotonic() - start, 1), "smoke_only": True,
              "skipped": skipped, "archive": namespace["archive_path"], "checkpoint_roundtrip": "passed",
              "schema": schema, "figures": figures}
    (workdir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
