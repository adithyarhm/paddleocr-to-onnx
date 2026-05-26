"""Parse CORD-1000 JSON annotations → PaddleOCR rec format.

CORD JSON structure (per image):
  valid_line[]
    └── words[]
          ├── quad: {x1,y1,x2,y2,x3,y3,x4,y4}   <- bounding box (polygon)
          └── text: str                             <- ground truth text

Output:
  data/rec_train/
    images/   <- cropped word images
    rec_gt_train.txt  <- "images/xxx.jpg\ttext"
  data/rec_val/
    images/
    rec_gt_val.txt
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

CORD_ROOT = Path("data/cord_1000")
OUT_TRAIN = Path("data/rec_train")
OUT_VAL = Path("data/rec_val")

MIN_TEXT_LEN = 1        # skip empty/whitespace-only labels
MIN_CROP_SIZE = 4       # skip degenerate crops (px)


def quad_to_bbox(quad: dict) -> tuple:
    """Convert CORD quad dict to (x_min, y_min, x_max, y_max)."""
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def crop_and_save(img: np.ndarray, quad: dict, out_path: Path) -> bool:
    """Crop word region from image using axis-aligned bounding box."""
    x1, y1, x2, y2 = quad_to_bbox(quad)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return False
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)
    return True


def process_split(split: str, out_dir: Path):
    img_dir = CORD_ROOT / split / "image"
    json_dir = CORD_ROOT / split / "json"
    crop_dir = out_dir / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    label_file = out_dir / f"rec_gt_{split}.txt"
    lines = []
    skipped = 0

    json_files = sorted(json_dir.glob("*.json"))
    for jf in tqdm(json_files, desc=f"Processing {split}"):
        ann = json.loads(jf.read_text())
        img_name = jf.stem + ".png"
        img_path = img_dir / img_name
        # CORD images may be .png or .jpg
        if not img_path.exists():
            img_path = img_dir / (jf.stem + ".jpg")
        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        for line_idx, valid_line in enumerate(ann.get("valid_line", [])):
            for word_idx, word in enumerate(valid_line.get("words", [])):
                text = word.get("text", "").strip()
                if len(text) < MIN_TEXT_LEN:
                    continue

                out_name = f"{jf.stem}_l{line_idx}_w{word_idx}.jpg"
                out_path = crop_dir / out_name

                if "quad" not in word:
                    skipped += 1
                    continue

                ok = crop_and_save(img, word["quad"], out_path)
                if ok:
                    rel = f"images/{out_name}"
                    lines.append(f"{rel}\t{text}")
                else:
                    skipped += 1

    label_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{split}] Saved {len(lines)} samples → {label_file}")
    print(f"[{split}] Skipped {skipped} entries")


if __name__ == "__main__":
    process_split("train", OUT_TRAIN)
    process_split("dev", OUT_VAL)
    print("\nDone! Sekarang jalankan train.py")
