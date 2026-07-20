"""
train_rcnn_finetune.py

Fine-tune a Faster R-CNN model starting from existing trained weights
(best.pt) on a NEW/additional dataset — mirrors train_yolo_finetune.py.

Your dataset is in YOLO format (flat "images" and "labels" folders, one
.txt per image with "<class_id> <xc> <yc> <w> <h>" normalized lines).
This script:
  1. Splits it into train/val/test (matching image<->label pairs by
     filename stem), reusing the same split logic as the YOLO script
  2. Converts YOLO-format labels to Faster R-CNN's expected target format
     (absolute-pixel xyxy boxes + integer class labels) on the fly
  3. Fine-tunes from PRETRAINED_WEIGHTS on the newly split dataset
  4. Saves best.pt (lowest validation loss) into PROJECT_DIR/RUN_NAME

Usage:
    python train_rcnn_finetune.py

Just edit the CONFIG block below before running.
"""

import random
import shutil
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ─────────────────────────────────────────────
# CONFIG – edit these before running
# ─────────────────────────────────────────────

# Path to your existing trained Faster R-CNN weights (state_dict .pt)
PRETRAINED_WEIGHTS = r"C:\Users\golir\Door_detection\Dataset_All\labeled_all_new\Dataset_ALL\results\RCNN_doors_640px_50e_8b\best.pt"

# Root folder containing your NEW, unsplit dataset — must contain
# subfolders named exactly "images" and "labels" with matching filenames
# in YOLO format (e.g. images/foo.jpg <-> labels/foo.txt)
# Root folder containing your dataset. If RUN_SPLIT=True, this must contain
# flat "images" and "labels" subfolders. If RUN_SPLIT=False (your case —
# already split), this must contain train/val/test subfolders, each with
# their own "images" and "labels" subfolders:
#   NEW_DATASET_ROOT/train/images, NEW_DATASET_ROOT/train/labels
#   NEW_DATASET_ROOT/val/images,   NEW_DATASET_ROOT/val/labels
#   NEW_DATASET_ROOT/test/images,  NEW_DATASET_ROOT/test/labels
NEW_DATASET_ROOT = r"C:\Users\golir\Door_detection\new_dataset"

# Where the split dataset gets written if RUN_SPLIT=True (ignored otherwise)
SPLIT_OUTPUT_ROOT = r"C:\Users\golir\Door_detection\new_dataset_split_rcnn"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SPLIT_SEED  = 42

# Your dataset is already split into train/val/test, so skip the split step
# and read directly from NEW_DATASET_ROOT.
RUN_SPLIT = False

# Class names, in the SAME order as PRETRAINED_WEIGHTS was trained on.
# Index 0 MUST be "__background__" per torchvision Faster R-CNN convention.
# This must match the checkpoint's head shape exactly (see the live
# detection script — we confirmed your current best.pt has 5 classes:
# background + door, door frame, door handle, lever).
CLASS_NAMES = ["__background__", "door", "door frame", "door handle", "lever"]
NUM_CLASSES = len(CLASS_NAMES)

# YOLO class_id -> Faster R-CNN label id. YOLO ids are 0-indexed with no
# background slot; Faster R-CNN reserves 0 for background, so normally
# rcnn_label = yolo_class_id + 1. Edit if your YOLO class order doesn't
# line up 1:1 with CLASS_NAMES[1:].
def yolo_id_to_rcnn_label(yolo_id: int) -> int:
    return yolo_id + 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Where run folders get created
PROJECT_DIR = r"C:\Users\golir\Door_detection\runs_rcnn"

# Training hyperparameters
IMG_SIZE     = 640     # images are resized so their longer side = IMG_SIZE before feeding the model
EPOCHS       = 20
BATCH_SIZE   = 4       # Faster R-CNN is memory-heavy; keep this modest on CPU
LEARNING_RATE = 0.0005
DEVICE       = "cpu"   # set to "cuda" if a GPU is available (e.g. Colab)
NUM_WORKERS  = 2
PATIENCE     = 5       # early stopping: stop if val loss doesn't improve for this many epochs

# Auto-generated run name — encodes hyperparams so it always matches what
# actually ran, even if you change EPOCHS/BATCH_SIZE/IMG_SIZE above later.
# e.g. "rcnn_finetune_640px_20e_4b"
RUN_NAME = f"rcnn_finetune_{IMG_SIZE}px_{EPOCHS}e_{BATCH_SIZE}b"

# Preprocessing — should match your training pipeline (grayscale + CLAHE)
GRAYSCALE = True
USE_CLAHE = True
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# If True, freezes the backbone (only the head trains) — faster and less
# prone to overfitting on small new datasets.
FREEZE_BACKBONE = False

# ─────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE) if USE_CLAHE else None


# ─────────────────────────────────────────────
# Dataset splitting (mirrors train_yolo_finetune.py)
# ─────────────────────────────────────────────
def split_dataset():
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

    return out_root


# ─────────────────────────────────────────────
# Dataset class: reads YOLO-format labels, returns Faster R-CNN targets
# ─────────────────────────────────────────────
class YoloFormatDetectionDataset(Dataset):
    def __init__(self, split_root: Path, split_name: str):
        self.images_dir = split_root / split_name / "images"
        self.labels_dir = split_root / split_name / "labels"
        self.image_paths = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        orig_h, orig_w = image.shape[:2]

        # match training preprocessing: grayscale + CLAHE
        if GRAYSCALE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if USE_CLAHE:
                gray = _clahe.apply(gray)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # resize so the longer side == IMG_SIZE, keep aspect ratio
        scale = IMG_SIZE / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image = cv2.resize(image, (new_w, new_h))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    yolo_cls = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])

                    # YOLO normalized -> absolute pixels in the RESIZED image
                    box_w = w * new_w
                    box_h = h * new_h
                    x1 = (xc * new_w) - box_w / 2
                    y1 = (yc * new_h) - box_h / 2
                    x2 = x1 + box_w
                    y2 = y1 + box_h

                    # clamp
                    x1 = max(0.0, min(new_w, x1))
                    y1 = max(0.0, min(new_h, y1))
                    x2 = max(0.0, min(new_w, x2))
                    y2 = max(0.0, min(new_h, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue  # degenerate box, skip

                    boxes.append([x1, y1, x2, y2])
                    labels.append(yolo_id_to_rcnn_label(yolo_cls))

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Faster R-CNN needs valid-shaped empty tensors for images with no boxes
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_model(weights_path: str, num_classes: int, device: str, freeze_backbone: bool):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(device)
    return model


# ─────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if train:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
        else:
            # torchvision detection models only return losses in train() mode,
            # so temporarily switch to train() for loss computation without
            # updating weights (no_grad keeps it from actually training).
            model.train()
            with torch.no_grad():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            model.eval()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    if RUN_SPLIT:
        split_root = split_dataset()
    else:
        split_root = Path(NEW_DATASET_ROOT)
        print(f"RUN_SPLIT is False — reading existing train/val/test split directly from: {split_root}")

        for split_name in ("train", "val", "test"):
            img_dir = split_root / split_name / "images"
            lbl_dir = split_root / split_name / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(
                    f"Expected '{img_dir}' and '{lbl_dir}' to both exist. "
                    f"Check that NEW_DATASET_ROOT points to a folder with "
                    f"train/val/test subfolders, each containing their own "
                    f"images/ and labels/ subfolders."
                )

    train_ds = YoloFormatDetectionDataset(split_root, "train")
    val_ds   = YoloFormatDetectionDataset(split_root, "val")
    print(f"train images: {len(train_ds)}   val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    print(f"Loading pretrained weights from: {PRETRAINED_WEIGHTS}")
    model = load_model(PRETRAINED_WEIGHTS, NUM_CLASSES, DEVICE, FREEZE_BACKBONE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    run_dir = Path(PROJECT_DIR) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Starting fine-tuning: {RUN_NAME}")
    print(f"  epochs: {EPOCHS}  batch: {BATCH_SIZE}  lr: {LEARNING_RATE}  device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
        val_loss   = run_epoch(model, val_loader, optimizer, DEVICE, train=False)

        print(f"[epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  -> new best (val_loss={val_loss:.4f}), saved to {run_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  No improvement for {PATIENCE} epochs — stopping early.")
                break

    torch.save(model.state_dict(), run_dir / "last.pt")
    print(f"\nDone. Results saved to: {run_dir}")
    print(f"Best weights: {run_dir / 'best.pt'}")
    print(f"Last weights: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()