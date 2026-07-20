"""
Roboflow Door Detection - Batch Folder Processing
Runs the 'general-segmentation-api' workflow on all images in a folder
and saves YOLO-format .txt label files alongside a summary CSV.
Also saves annotated previews so you can verify the auto-labels.

YOLO label format (one line per detection):
  <class_id> <x_center> <y_center> <width> <height>
All values are normalized to [0, 1] relative to image dimensions.
Classes: 0 = door
"""

import os
import csv
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

# ─────────────────────────────────────────────
# CONFIG  –  edit these before running
# ─────────────────────────────────────────────
INPUT_FOLDER   = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\train\images_grayscale"   # folder with your images
OUTPUT_FOLDER  = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\train\labels_doors"   # where YOLO .txt files go
PREVIEW_FOLDER = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\train\preview"  # annotated previews (first PREVIEW_N)

# NEW: second preview folder that gets a random sample of ALL processed images,
# independent of PREVIEW_N — useful for spot-checking the whole batch rather
# than just the first few.
SECONDARY_PREVIEW_FOLDER = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\train\preview_sample_50pct"
SECONDARY_PREVIEW_RATE   = 0.5   # ~50% of all images get a copy here too

# Read the key from an env var if set (safer than hard-coding); falls back to the literal.
API_KEY     = os.environ.get("ROBOFLOW_API_KEY", "4HVu7elVLOFPFaIU1WXp")
WORKSPACE   = "golnooshs-workspace"
WORKFLOW_ID = "general-segmentation-api"
CLASSES     = "door"          # ask the workflow for both classes

# Class name → YOLO integer ID mapping (matched case-insensitively).
# Edit to match your project's class order exactly.
CLASS_MAP = {
    "door":         0,
}
CLASS_NAMES = ["door"]     # index = YOLO id (for preview labels)
PREVIEW_COLORS = ["lime"]      # door=lime
PREVIEW_N = 20                           # how many annotated previews to save to PREVIEW_FOLDER

# Image extensions to process
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────

# class names returned by the model that we couldn't map (collected for a warning)
UNMAPPED = set()


def setup_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY,
    )


def get_image_paths(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    paths = [p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    return paths


def run_detection(client, image_path: Path):
    return client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={"image": str(image_path)},
        parameters={"classes": CLASSES},
        use_cache=True,
    )


def get_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image without loading pixels."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def extract_predictions(result) -> list[dict]:
    """
    Pull the flat list of prediction dicts from a Roboflow workflow response.
    Roboflow returns a list of output dicts; predictions are usually nested
    under result[0]['predictions'] or result[0]['predictions']['predictions'].
    """
    if not isinstance(result, list) or len(result) == 0:
        return []

    first = result[0]

    for key in ("predictions", "detections", "output"):
        if key not in first:
            continue
        val = first[key]

        if isinstance(val, list):
            return val

        if isinstance(val, dict):
            inner = val.get("predictions", [])
            if isinstance(inner, list):
                return inner

    return []


def resolve_class_id(class_name: str):
    """Case-insensitive class lookup. Returns YOLO id, or None if unmapped
    (unmapped names are tracked and skipped, not mislabeled)."""
    key = str(class_name).strip().lower()
    if key in CLASS_MAP:
        return CLASS_MAP[key]
    UNMAPPED.add(class_name)
    return None


def prediction_to_yolo(pred: dict, img_w: int, img_h: int) -> str | None:
    """
    Convert one Roboflow prediction dict to a YOLO label line.

    Roboflow bounding-box fields:
      x, y          – center of box in pixels
      width, height – box size in pixels
      class         – class name string
    """
    try:
        x_center_px = float(pred["x"])
        y_center_px = float(pred["y"])
        box_w_px    = float(pred["width"])
        box_h_px    = float(pred["height"])
        class_name  = pred.get("class", "")
    except (KeyError, TypeError, ValueError):
        return None

    class_id = resolve_class_id(class_name)
    if class_id is None:
        return None

    # Normalize to [0, 1]
    x_norm = x_center_px / img_w
    y_norm = y_center_px / img_h
    w_norm = box_w_px    / img_w
    h_norm = box_h_px    / img_h

    # Clamp to valid range
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


def save_yolo_labels(lines: list[str], out_path: Path):
    """Write YOLO label file. Empty list → empty file (image with no detections)."""
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def save_preview(image_path: Path, predictions: list[dict], out_path: Path,
                  secondary_out_dir: Path = None, secondary_rate: float = 0.0):
    """Draw boxes + class labels on a copy of the image so labels can be eyeballed.

    If secondary_out_dir is given, ~secondary_rate fraction of calls will also
    save a copy there (e.g. a random spot-check subset separate from the
    always-save-first-N preview folder).
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for pred in predictions:
            cid = resolve_class_id(pred.get("class", ""))
            if cid is None:
                continue
            try:
                x, y = float(pred["x"]), float(pred["y"])
                w, h = float(pred["width"]), float(pred["height"])
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            color = PREVIEW_COLORS[cid % len(PREVIEW_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            conf = pred.get("confidence", 0)
            draw.text((x1 + 2, max(0, y1 - 12)),
                      f"{CLASS_NAMES[cid]} {float(conf):.2f}", fill=color)

        if out_path is not None:
            im.save(out_path)

        if secondary_out_dir is not None and random.random() < secondary_rate:
            secondary_out_dir = Path(secondary_out_dir)
            secondary_out_dir.mkdir(parents=True, exist_ok=True)
            name = out_path.name if out_path is not None else (image_path.stem + "_preview.jpg")
            im.save(secondary_out_dir / name)


def process_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)
    os.makedirs(SECONDARY_PREVIEW_FOLDER, exist_ok=True)

    client = setup_client()
    images = get_image_paths(INPUT_FOLDER)
    total  = len(images)

    if total == 0:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Found {total} image(s) in {INPUT_FOLDER}")
    print(f"YOLO labels will be saved to {OUTPUT_FOLDER}")
    print(f"Previews (first {PREVIEW_N}) will be saved to {PREVIEW_FOLDER}")
    print(f"~{int(SECONDARY_PREVIEW_RATE*100)}% random sample of ALL images will also be saved to {SECONDARY_PREVIEW_FOLDER}\n")

    csv_path = Path(OUTPUT_FOLDER) / "summary.csv"
    csv_rows = []
    previews_saved = 0

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{total}] {img_path.name} ...", end=" ", flush=True)
        status     = "ok"
        n_detected = 0
        error_msg  = ""

        try:
            img_w, img_h = get_image_size(img_path)
            result       = run_detection(client, img_path)
            predictions  = extract_predictions(result)
            n_detected   = len(predictions)

            yolo_lines = []
            for pred in predictions:
                line = prediction_to_yolo(pred, img_w, img_h)
                if line:
                    yolo_lines.append(line)

            label_out = Path(OUTPUT_FOLDER) / (img_path.stem + ".txt")
            save_yolo_labels(yolo_lines, label_out)

            # Decide preview behavior for this image:
            #  - first PREVIEW_N images always get a copy in PREVIEW_FOLDER
            #  - EVERY image has an independent ~50% chance of also landing
            #    in SECONDARY_PREVIEW_FOLDER, regardless of PREVIEW_N
            preview_main_path = None
            if previews_saved < PREVIEW_N:
                preview_main_path = Path(PREVIEW_FOLDER) / (img_path.stem + "_preview.jpg")
                previews_saved += 1

            if preview_main_path is not None or SECONDARY_PREVIEW_RATE > 0:
                save_preview(
                    img_path, predictions,
                    out_path=preview_main_path,
                    secondary_out_dir=SECONDARY_PREVIEW_FOLDER,
                    secondary_rate=SECONDARY_PREVIEW_RATE,
                )

            print(f"✓  {len(yolo_lines)} kept / {n_detected} raw → {label_out.name}")

        except Exception as e:
            status    = "error"
            error_msg = str(e)
            print(f"✗  ERROR: {e}")

        csv_rows.append({
            "image":      img_path.name,
            "status":     status,
            "detections": n_detected,
            "error":      error_msg,
        })

        time.sleep(0.2)  # be polite to the API

    # data.yaml so the labeled set is training-ready
    with open(Path(OUTPUT_FOLDER) / "data.yaml", "w") as f:
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

    # Summary CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "status", "detections", "error"])
        writer.writeheader()
        writer.writerows(csv_rows)

    ok_count  = sum(1 for r in csv_rows if r["status"] == "ok")
    err_count = total - ok_count
    print(f"\n{'─'*50}")
    print(f"Done.  {ok_count} succeeded, {err_count} failed.")
    print(f"YOLO labels → {OUTPUT_FOLDER}/")
    print(f"Previews    → {PREVIEW_FOLDER}/")
    print(f"50% sample  → {SECONDARY_PREVIEW_FOLDER}/")
    print(f"Summary CSV → {csv_path}")
    if UNMAPPED:
        print(f"\n[!] model returned class names NOT in CLASS_MAP: {sorted(UNMAPPED)}")
        print("    Add them to CLASS_MAP (mapping to the right YOLO id) and re-run.")


if __name__ == "__main__":
    process_folder()