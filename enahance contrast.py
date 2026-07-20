"""
clahe_enhance.py
----------------
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to every
image found inside INPUT_FOLDER, then saves the results to OUTPUT_FOLDER.

Requirements
------------
  pip install opencv-python
"""

import os
import cv2

# ── CHANGE THESE TWO PATHS ────────────────────────────────────────────────────
INPUT_FOLDER  = r"C:\Users\golir\Door_detection\GitHub_Dataset_ready\Images\Train_GrayScale"        # folder with original images
OUTPUT_FOLDER = r"C:\Users\golir\Door_detection\GitHub_Dataset_ready\Images\Train_GrayScale_enhanced"  # folder to save results
# ─────────────────────────────────────────────────────────────────────────────

# ── CLAHE settings (optional to tweak) ───────────────────────────────────────
CLIP_LIMIT = 2.0     # contrast limit; raise for stronger enhancement (e.g. 3.0–4.0)
TILE_GRID  = (8, 8)  # grid of local regions; (8,8) is a sensible default
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def apply_clahe(image_bgr):
    """
    Apply CLAHE to a BGR image.
    - Colour images: converted to LAB, CLAHE applied only to the Lightness
      channel so colours stay intact, then converted back to BGR.
    - Grayscale images: CLAHE applied directly.
    """
    if len(image_bgr.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
        return clahe.apply(image_bgr)

    # Colour image → LAB colour space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def process_folder(input_dir, output_dir):
    """Enhance every image in input_dir and save to output_dir."""

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        print(f"No supported image files found in: {input_dir}")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTS))}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFound {len(image_files)} image(s).")
    print(f"Output folder: {output_dir}\n")

    success, failed = 0, 0

    for filename in image_files:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP]  Could not read: {filename}")
            failed += 1
            continue

        enhanced = apply_clahe(img)
        cv2.imwrite(dst_path, enhanced)
        print(f"  [OK]    {filename}")
        success += 1

    print(f"\nDone. {success} enhanced, {failed} skipped.")


# ── Run ───────────────────────────────────────────────────────────────────────
process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
