"""
convert_to_grayscale.py
-----------------------
Scans a folder for images, converts each one to grayscale,
and saves them all into a new output folder.

HOW TO USE:
  1. Change INPUT_FOLDER to the path of your image folder.
  2. Change OUTPUT_FOLDER to wherever you want the grayscale images saved.
  3. Run: python convert_to_grayscale.py

REQUIREMENTS:
  pip install Pillow
"""

from pathlib import Path
from PIL import Image

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FOLDER  = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\test\images"   # <-- change this
OUTPUT_FOLDER = r"C:\Users\golir\Door_detection\Door detection.v2i.yolov8\test\images_grayscale" \
"" # <-- change this
# ──────────────────────────────────────────────────────────────────────────────

# Image file types to look for (case-insensitive)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def convert_images_to_grayscale(input_folder: str, output_folder: str) -> None:
    input_path  = Path(input_folder)
    output_path = Path(output_folder)

    # Make sure the input folder actually exists
    if not input_path.exists():
        print(f"ERROR: Input folder not found: {input_path}")
        return

    # Create the output folder (and any parent folders) if it doesn't exist yet
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder ready: {output_path}\n")

    # Collect every image file in the input folder (not recursive)
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print("No supported image files found in the input folder.")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"Found {len(image_files)} image(s). Converting...\n")

    converted = 0
    skipped   = 0

    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                # Convert to grayscale ('L' = 8-bit pixels, black and white)
                grayscale_img = img.convert("L")

                # Save to the output folder with the same filename
                save_path = output_path / img_file.name
                grayscale_img.save(save_path)

            print(f"  ✓  {img_file.name}")
            converted += 1

        except Exception as e:
            print(f"  ✗  {img_file.name}  →  SKIPPED ({e})")
            skipped += 1

    # Summary
    print(f"\nDone! {converted} image(s) converted, {skipped} skipped.")
    print(f"Grayscale images saved to: {output_path}")


if __name__ == "__main__":
    convert_images_to_grayscale(INPUT_FOLDER, OUTPUT_FOLDER)
