"""Final notebook cell also works without any live training variables."""
from pathlib import Path
import json
import shutil
import zipfile

default_root = (Path("/kaggle/working") if Path("/kaggle").exists() else
                Path("/content") if Path("/content").exists() else Path.cwd() / ".runtime/training")
result_root = Path(globals().get("WORKDIR", default_root / "lia-action-v4"))
result_root.mkdir(parents=True, exist_ok=True)
diagnostics_dir = result_root / "diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)
saved_reports = list((result_root / "candidates").glob("*/training_report.json"))
for report_path in saved_reports:
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    print("Action:", "SMOKE" if saved_report["smoke_only"] else "QUALIFIED" if saved_report["qualified_on_synthetic_holdouts"] else "CANDIDATE")
    print("Epoch selected:", saved_report["training"]["selected_epoch"], "updates:", saved_report["training"]["optimizer_updates"])
    print("Test macro-F1:", {k: round(v["macro_f1"], 4) for k, v in saved_report["metrics"]["test"].items()})
    shutil.copyfile(report_path, diagnostics_dir / (report_path.parent.name + "-training_report.json"))
for folder in ("resume", "failures"):
    for report_path in (result_root / folder).glob("*.json"):
        shutil.copyfile(report_path, diagnostics_dir / (folder + "-" + report_path.name))
diagnostics_zip = shutil.make_archive(str(result_root / "lia-action-v4-diagnostics"), "zip", root_dir=diagnostics_dir)
with zipfile.ZipFile(diagnostics_zip) as archive:
    assert archive.testzip() is None
print("Diagnostics:", diagnostics_zip)
print("Model archives:", [p.name for p in result_root.glob("laya-muz-bot-controls-v4*.zip")])
if not saved_reports: print("No completed report. Run cells from the top; inspect failures/ and resume/ if a prior run stopped.")
print("CANDIDATE and SMOKE are not deployable. Preserve candidates/ + resume/ to continue training.")
