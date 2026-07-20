"""
train_yolo_finetune.py

Fine-tune a YOLOv8n model starting from existing trained weights (best.pt)
on a NEW/additional dataset.

Your new dataset only has flat "images" and "labels" folders (no train/val/test
split yet), so this script:
  1. Splits it into train/val/test folders (matching image<->label pairs by
     filename stem, same YOLO structure Ultralytics expects)
  2. Auto-generates a data.yaml for the split
  3. Fine-tunes from PRETRAINED_WEIGHTS on the newly split dataset

Usage:
    python train_yolo_finetune.py

Just edit the CONFIG block below before running.
"""

import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG – edit these before running
# ─────────────────────────────────────────────

# Path to your existing trained weights (the model you're building on top of)
PRETRAINED_WEIGHTS = r"C:\Users\golir\Door_detection\weights\best.pt"

# Root folder containing your NEW, unsplit dataset — must contain
# subfolders named exactly "images" and "labels" with matching filenames
# (e.g. images/foo.jpg <-> labels/foo.txt)
NEW_DATASET_ROOT = r"C:\Users\golir\Door_detection\new_dataset"

# Where the split dataset gets written (train/val/test + data.yaml)
SPLIT_OUTPUT_ROOT = r"C:\Users\golir\Door_detection\new_dataset_split"

# Split ratios — must sum to 1.0
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

SPLIT_SEED = 42   # fixed seed so the split is reproducible across reruns

# Class names, in the SAME order as PRETRAINED_WEIGHTS was originally trained on.
# This MUST match best.pt's class head order or fine-tuning will break/reinit
# the detection head silently. Edit to match your project's real class list.
CLASS_NAMES = ["door"]

# Image extensions to look for
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Set to False if you've already run the split once and just want to re-train
# on the existing split without re-splitting (e.g. after a crash mid-training)
RUN_SPLIT = True

# Where run folders get created (matches your naming convention)
PROJECT_DIR = r"C:\Users\golir\Door_detection\runs"
RUN_NAME    = "yolov8n_finetune_640px_50e_8b"

# Training hyperparameters
IMG_SIZE    = 640
EPOCHS      = 50
BATCH_SIZE  = 8
DEVICE      = "cpu"    # set to "0" if a GPU is available (e.g. Colab)
PATIENCE    = 15
WORKERS     = 4

FREEZE_BACKBONE = False
FREEZE_LAYERS   = 10

# ─────────────────────────────────────────────


def split_dataset():
    """Split flat images/ + labels/ folders into train/val/test, matching
    pairs by filename stem. Copies files (originals are left untouched)."""
    src_root = Path(NEW_DATASET_ROOT)
    images_dir = src_root / "images"
    labels_dir = src_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Expected an 'images' folder inside {src_root}, not found.")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Expected a 'labels' folder inside {src_root}, not found.")

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0"

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    # Match each image to its label by filename stem; skip + warn on mismatches
    pairs = []
    missing_labels = []
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} image(s) have no matching label file and will be skipped:")
        for name in missing_labels[:10]:
            print(f"  - {name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    print(f"Found {len(pairs)} matched image/label pairs.")

    random.seed(SPLIT_SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    # remainder goes to test, so rounding doesn't drop images
    n_test  = n_total - n_train - n_val

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    out_root = Path(SPLIT_OUTPUT_ROOT)
    for split_name, split_pairs in splits.items():
        img_out = out_root / split_name / "images"
        lbl_out = out_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

        print(f"  {split_name}: {len(split_pairs)} images")

    # Write data.yaml
    data_yaml_path = out_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {out_root}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\ndata.yaml written to: {data_yaml_path}")
    return data_yaml_path


def train(data_yaml_path: Path):
    weights_path = Path(PRETRAINED_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    print(f"\nLoading pretrained weights from: {weights_path}")
    model = YOLO(str(weights_path))

    train_kwargs = dict(
        data=str(data_yaml_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    if FREEZE_BACKBONE:
        train_kwargs["freeze"] = FREEZE_LAYERS

    print(f"Starting fine-tuning: {RUN_NAME}")
    print(f"  data:   {data_yaml_path}")
    print(f"  imgsz:  {IMG_SIZE}")
    print(f"  epochs: {EPOCHS}")
    print(f"  batch:  {BATCH_SIZE}")
    print(f"  device: {DEVICE}")
    print(f"  freeze_backbone: {FREEZE_BACKBONE}"
          + (f" (first {FREEZE_LAYERS} layers)" if FREEZE_BACKBONE else ""))

    model.train(**train_kwargs)

    run_dir = Path(PROJECT_DIR) / RUN_NAME
    print(f"\nDone. Results saved to: {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")


def main():
    if RUN_SPLIT:
        data_yaml_path = split_dataset()
    else:
        data_yaml_path = Path(SPLIT_OUTPUT_ROOT) / "data.yaml"
        print(f"RUN_SPLIT is False — using existing split at: {data_yaml_path}")

    train(data_yaml_path)


if __name__ == "__main__":
    main()