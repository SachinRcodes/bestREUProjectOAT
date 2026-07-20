"""
Train YOLOv8n and YOLOv8m on a 2-class door / door-frame dataset.

Dataset layout (already labeled & split):
    <DATASET_ROOT>/
    ├── data.yaml            (optional; if missing, one is generated)
    ├── train/ images/ labels/
    ├── valid/ images/ labels/   ("val" also handled)
    └── test/  images/ labels/

Each model trains into its own folder; YOLO writes the confusion matrix and
PR/P/R/F1 curves automatically, and a final eval is run on the test split.
"""

from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────
DATASET_ROOT = r"/home/nsr59/Door_detection/Dataset_All/door_doorframe_dataset"  # EDIT to your dataset
MODELS  = ["yolov8n.pt", "yolov8m.pt"]
EPOCHS  = 100
BATCH   = 16            # lower for yolov8m if GPU OOM
IMGSZ   = 640           # set 224 to match older runs (faster on CPU)
PATIENCE = 20
PROJECT = r"/home/nsr59/Door_detection/runs"
DEVICE  = 0 if torch.cuda.is_available() else "cpu"

DEFAULT_NAMES = ["door", "door frame"]   # used only if dataset has no data.yaml
# ─────────────────────────────────────────────────────────


def prepare_yaml(root: str) -> str:
    root = Path(root)
    val_dir = "valid" if (root / "valid").exists() else "val"

    names = DEFAULT_NAMES
    src = root / "data.yaml"
    if src.exists():
        with open(src) as f:
            orig = yaml.safe_load(f) or {}
        names = orig.get("names", names)

    data = {
        "path":  str(root),
        "train": "train/images",
        "val":   f"{val_dir}/images",
        "test":  "test/images",
        "nc":    len(names),
        "names": names,
    }
    out = root / "data_fixed.yaml"
    with open(out, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"[yaml] {len(names)} classes {names} -> {out}")
    return str(out)


def main():
    data_yaml = prepare_yaml(DATASET_ROOT)
    print(f"[device] {DEVICE}")
    for weights in MODELS:
        tag = Path(weights).stem
        name = f"{tag}_doorframe_{IMGSZ}px_{EPOCHS}e_{BATCH}b"
        print("\n" + "=" * 55 + f"\n  Training {tag}\n" + "=" * 55)
        model = YOLO(weights)
        model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                    device=DEVICE, project=PROJECT, name=name, patience=PATIENCE, verbose=True)
        model.val(data=data_yaml, imgsz=IMGSZ, split="test",
                  project=PROJECT, name=name + "_test", plots=True, verbose=True)
        print(f"[done] {tag}")


if __name__ == "__main__":
    main()
