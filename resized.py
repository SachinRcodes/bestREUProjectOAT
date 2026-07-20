from pathlib import Path
from PIL import Image

# Folder containing images
image_dir = Path(r"C:\Users\golir\Door_detection\Dataset_All\larger_images\images")

# Output size
target_size = (640, 640)

# Supported image extensions
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

for image_path in image_dir.iterdir():
    if image_path.suffix.lower() in extensions and image_path.is_file():
        try:
            with Image.open(image_path) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(image_path)
                print(f"Resized: {image_path.name}")
        except Exception as e:
            print(f"Failed: {image_path.name} -> {e}")

print("Done.")
