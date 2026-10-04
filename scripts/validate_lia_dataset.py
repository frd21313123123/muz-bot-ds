"""Validate the delivered ZIP by running its notebook's upload/data cells offline."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".runtime/datasets/lia-starter-v1"
WORK = ROOT / ".runtime/dataset-validation"


def main():
    global DATA, WORK
    embedded = "--embedded" in sys.argv
    if embedded:
        WORK = ROOT / ".runtime/embedded-dataset-validation"
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    archive_path = DATA.with_suffix(".zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    notebook_path = DATA.parent / "lia_training_with_embedded_dataset.ipynb" if embedded else DATA / "lia_dataset_training_colab_kaggle.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    validation_packages = ROOT / ".runtime/notebook-validation"
    if validation_packages.is_dir():
        sys.path.insert(0, str(validation_packages))
    schema = "structure + Python AST"
    if importlib.util.find_spec("nbformat"):
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        schema = "nbformat.validate"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    namespace = {"Path": Path, "os": os, "json": json, "importlib": importlib}
    fake_input = WORK / "input"
    fake_input.mkdir(parents=True, exist_ok=True)
    if not embedded:
        shutil.copyfile(archive_path, fake_input / archive_path.name)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source, tags = cell["source"], cell["metadata"].get("tags", [])
        if "parameters" in tags:
            source = source.replace('WORKDIR = Path("/kaggle/working/lia-training" if Path("/kaggle/working").exists() else "/content/lia-training")',
                                    f'WORKDIR = Path({str(WORK / "run")!r})')
        elif "dataset-upload" in tags:
            source = source.replace('Path("/kaggle/input")', f'Path({str(fake_input)!r})')
            source = source.replace('Path("/content")', f'Path({str(WORK / "missing-colab")!r})')
        elif "dataset-embedded" in tags:
            pass
        elif not (source.startswith("SNAPSHOT =") or source.startswith("if SMOKE_TEST:") or source.startswith("if PLANNER_PATHS:")):
            continue
        exec(compile(source, cell["id"], "exec"), namespace)
        if "dataset-embedded" in tags:
            DATA = namespace["EMBEDDED_DATA_DIR"]
            manifest = namespace["embedded_manifest"]
    training = namespace["training"]
    for filename, expected in manifest["file_sha256"].items():
        assert training.sha256_file(DATA / filename) == expected, filename
    actual_split = {name: {"runtime_commands": len(namespace["command_splits"][name]),
                           "planner_scenarios": len(namespace["planner_splits"][name]),
                           "bot_log_commands": sum(r["origin"] == "bot_log" for r in namespace["command_splits"][name])}
                    for name in training.SPLITS}
    assert actual_split == manifest["validated_grouped_split"]
    raw = training.read_jsonl([DATA / "bot-logs-unreviewed.jsonl"])
    labelled = training.read_jsonl([DATA / "bot-commands-labeled.jsonl"])
    pending = training.read_jsonl([DATA / "bot-review-queue.jsonl"])
    assert {r["event_id"] for r in raw} == {r["event_id"] for r in labelled + pending}
    assert len(raw) == len(labelled) + len(pending) == manifest["bot_log_rows"]
    forbidden = {"model_decision", "suggested_intent", "validated_intent", "canonical_message", "outcome", "diagnostics"}
    assert all(not (forbidden & set(row)) and not (forbidden & set(row.get("state", {}))) for row in raw + labelled + pending)
    location = {}
    text_location = {}
    for name, rows in namespace["all_splits"].items():
        for row in rows:
            group = row["group_id"]
            assert group not in location or location[group] == name
            location[group] = name
            for key in ("message", "canonical"):
                text = training.normalized(row[key])
                assert text not in text_location or text_location[text] == name
                text_location[text] = name
    assert all(r["reviewed"] is False for r in training.read_jsonl([DATA / "commands-synthetic.jsonl"]) + labelled)
    assert namespace["ALLOW_AUTOMATIC_LABELS"] is True
    result = {"schema": schema, "zip_integrity": "passed", "notebook_upload_and_loading": "passed",
              "embedded_dataset": embedded,
              "log_coverage": len(raw), "automatic_bot_labels": len(labelled), "pending_bot_labels": len(pending),
              "leakage_checks": "family + raw/canonical across runtime/planner passed", "splits": actual_split,
              "training_executed": False}
    (WORK / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
