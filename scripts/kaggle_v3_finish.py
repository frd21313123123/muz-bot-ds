"""Independent final cell: no live WORKDIR/reports variables required."""
from pathlib import Path
import json, shutil

result_root = Path(globals().get("WORKDIR", "/kaggle/working/lia-v3"))
diagnostics = result_root / "diagnostics"
diagnostics.mkdir(parents=True, exist_ok=True)
paths = list((result_root / "candidates").glob("*/training_report.json"))
paths += list((result_root / "resume").glob("*.json"))
paths += list((result_root / "failures").glob("*.json"))
if not paths:
    raise RuntimeError("В " + str(result_root) + " нет сохранённых результатов. Запустите Save Version → Save & Run All с первой ячейки. "
                       "После удаления среды используйте сохранённый Output; из одного графика веса восстановить нельзя.")
for path in paths:
    shutil.copyfile(path, diagnostics / (path.parent.name + "-" + path.name))
    if path.name == "training_report.json":
        report = json.loads(path.read_text(encoding="utf-8"))
        print(report["stage"], "qualified:", report["qualified_on_synthetic_holdouts"], "smoke:", report["smoke_only"])
diagnostics_zip = shutil.make_archive(str(result_root / "lia-v3-diagnostics"), "zip", root_dir=diagnostics)
print("Diagnostics:", diagnostics_zip)
print("Model archives:", [p.name for p in result_root.glob("laya-muz-bot-*.zip")])
print("Resume states:", [p.name for p in (result_root / "resume").glob("*.pt")])
print("CANDIDATE и SMOKE-ONLY не подключайте к боту. Для продолжения сохраните весь Output: candidates/ + resume/.")
