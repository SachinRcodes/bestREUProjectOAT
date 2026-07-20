"""
Auto-label doors -> YOLOv8 label files using Roboflow's HOSTED HTTP API.
=======================================================================
No SDK needed (works on Python 3.13). Each image is sent to a public
Roboflow detection model; the returned boxes are written as YOLO labels.

Output:
    OUTPUT_FOLDER/
      images/      <- copies of the originals
      labels/      <- one .txt per image:  `class cx cy w h`  (normalized 0..1)
      classes.txt  <- class names, one per line, in index order
      preview/     <- first N images with boxes drawn (sanity check; optional)

Needs only: requests, opencv-python, tqdm   (all already installed)

Run:
    PowerShell:  $env:ROBOFLOW_API_KEY="your_key_here"
    python autolabel_doors_yolo.py
"""

import os
import sys
import base64
import shutil
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────
INPUT_FOLDER  = r"C:\Users\golir\Door_detection\Dataset_All\labeled_all_new\Dataset_ALL\images"
OUTPUT_FOLDER = r"C:\Users\golir\Door_detection\Dataset_All\auto_labeled_doors"

# A PUBLIC Roboflow Universe door-detection model, as "project-slug/version".
# Confirm it exists on https://universe.roboflow.com (search "door detection");
# the id is in the model URL + version dropdown -> "<project-slug>/<version>".
MODEL_ID = "door-detection-zxjly/1"

# Hosted inference endpoint (cloud). Uses your API key + Roboflow quota.
API_URL = "https://detect.roboflow.com"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# Only keep detections at/above this confidence (0..1).
CONFIDENCE = 0.40

# Map the model's class names -> your YOLO class indices.
# Order here also defines classes.txt. Any class the model returns that is NOT
# listed here is skipped. (Run the script on a few images and check the printed
# class names if labels come out empty — the public model may name them differently.)
CLASS_MAP = {
    "door": 0,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# How many annotated sanity-check images to write to preview/ (0 = none).
PREVIEW_COUNT = 20

# Network timeout per image (seconds).
TIMEOUT = 60
# ─────────────────────────────────────────────


def get_image_paths(folder: str):
    folder = Path(folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Input folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"[INFO] Found {len(paths)} images in {folder}")
    return paths


def infer(image_path: Path):
    """POST the image to Roboflow's hosted model; return a list of prediction dicts.
    Each dict has center x,y and width,height in PIXELS, plus 'class' and 'confidence'."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        f"{API_URL}/{MODEL_ID}",
        params={"api_key": API_KEY, "confidence": CONFIDENCE, "format": "json"},
        data=b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("predictions", [])


def to_yolo_line(pred: dict, w: int, h: int):
    """Roboflow prediction (center x/y, width/height px) -> 'cls cx cy bw bh' or None."""
    name = pred.get("class") or pred.get("class_name") or ""
    if name not in CLASS_MAP:
        return None
    if pred.get("width", 0) <= 0 or pred.get("height", 0) <= 0:
        return None
    cls = CLASS_MAP[name]
    clamp = lambda v: max(0.0, min(1.0, v))
    cx = clamp(pred["x"] / w)
    cy = clamp(pred["y"] / h)
    bw = clamp(pred["width"] / w)
    bh = clamp(pred["height"] / h)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    if not API_KEY:
        sys.exit('[ERROR] Set your key first:  $env:ROBOFLOW_API_KEY="your_key_here"')

    image_paths = get_image_paths(INPUT_FOLDER)
    if not image_paths:
        print("[ERROR] No images to process.")
        return

    out = Path(OUTPUT_FOLDER)
    images_out = out / "images"
    labels_out = out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if PREVIEW_COUNT > 0:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    # classes.txt in index order
    ordered = sorted(CLASS_MAP.items(), key=lambda kv: kv[1])
    (out / "classes.txt").write_text("\n".join(name for name, _ in ordered) + "\n",
                                      encoding="utf-8")

    total_boxes = 0
    labeled_imgs = 0
    preview_saved = 0
    seen_classes = set()

    for img_path in tqdm(image_paths, desc="Labeling", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ! could not read {img_path.name}, skipped")
            continue
        h, w = img.shape[:2]

        try:
            preds = infer(img_path)
        except Exception as e:
            print(f"  ! inference failed on {img_path.name}: {e}")
            preds = []

        lines, kept = [], []
        for p in preds:
            seen_classes.add(p.get("class", ""))
            line = to_yolo_line(p, w, h)
            if line:
                lines.append(line)
                kept.append(p)

        # write label file (empty file if no detections, to keep pairs intact)
        (labels_out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        shutil.copy(img_path, images_out / img_path.name)

        if lines:
            labeled_imgs += 1
            total_boxes += len(lines)

        if PREVIEW_COUNT > 0 and preview_saved < PREVIEW_COUNT:
            vis = img.copy()
            for p in kept:
                x1 = int(p["x"] - p["width"] / 2); y1 = int(p["y"] - p["height"] / 2)
                x2 = int(p["x"] + p["width"] / 2); y2 = int(p["y"] + p["height"] / 2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(vis, f'{p.get("class","")} {p.get("confidence",0):.2f}',
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            cv2.imwrite(str(out / "preview" / img_path.name), vis)
            preview_saved += 1

    print(f"\n{'─'*50}")
    print(f"✅ Done. Output: {out.resolve()}")
    print(f"   images/   {len(image_paths)} files")
    print(f"   labels/   {len(image_paths)} files  ({labeled_imgs} with doors, {total_boxes} boxes)")
    print(f"   classes.txt, preview/ ({preview_saved} images)")
    print(f"   class names the model returned: {sorted(c for c in seen_classes if c)}")
    if not total_boxes:
        print("[HINT] 0 boxes. Check that the names above are keys in CLASS_MAP,")
        print("       and that MODEL_ID is a valid public model your key can access.")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
