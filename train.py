"""Entry point fine-tune PP-OCRv4 rec pada CORD-1000.

Pastikan sudah:
1. python data/prepare_cord.py
2. Clone PaddleOCR dan download pretrained model

Run: python train.py
"""

import subprocess
import sys
from pathlib import Path

CONFIG = "configs/rec_ppocr_v4_cord.yml"
PADDLEOCR_DIR = Path("PaddleOCR")


def check_prerequisites():
    errors = []
    if not PADDLEOCR_DIR.exists():
        errors.append("Folder 'PaddleOCR/' tidak ditemukan. Jalankan:\n  git clone https://github.com/PaddlePaddle/PaddleOCR.git")
    if not Path("data/rec_train/rec_gt_train.txt").exists():
        errors.append("Label train tidak ditemukan. Jalankan:\n  python data/prepare_cord.py")
    if not Path("pretrained_models/en_PP-OCRv4_rec_train/best_accuracy.pdparams").exists():
        errors.append(
            "Pretrained model tidak ditemukan. Jalankan:\n"
            "  wget -P pretrained_models https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar\n"
            "  tar -xf pretrained_models/en_PP-OCRv4_rec_train.tar -C pretrained_models/"
        )
    if errors:
        print("[ERROR] Prerequisites tidak terpenuhi:")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)


def main():
    check_prerequisites()
    cmd = [
        sys.executable,
        str(PADDLEOCR_DIR / "tools" / "train.py"),
        "-c", CONFIG,
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
