"""Export fine-tuned .pdparams → Paddle static graph (.pdmodel + .pdiparams).

Harus dijalankan SEBELUM convert_to_onnx.py.
Output disimpan di: ./inference/rec_infer/

Run: python export_inference.py
Opsi override: python export_inference.py --epoch 30
"""

import subprocess
import sys
import argparse
from pathlib import Path

PADDLEOCR_DIR = Path("PaddleOCR")
CONFIG = "configs/rec_ppocr_v4_cord.yml"
OUTPUT_TRAIN_DIR = "./output/rec_ppocr_v4_cord"
INFER_SAVE_DIR = "./inference/rec_infer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=str, default=None,
                        help="Epoch checkpoint tertentu (default: best_accuracy)")
    args = parser.parse_args()

    if args.epoch:
        pretrained = f"{OUTPUT_TRAIN_DIR}/iter_epoch_{args.epoch}"
    else:
        pretrained = f"{OUTPUT_TRAIN_DIR}/best_accuracy"

    cmd = [
        sys.executable,
        str(PADDLEOCR_DIR / "tools" / "export_model.py"),
        "-c", CONFIG,
        "-o",
        f"Global.pretrained_model={pretrained}",
        f"Global.save_inference_dir={INFER_SAVE_DIR}",
    ]
    print(f"[INFO] Exporting model ke static graph...")
    print(f"[INFO] Checkpoint: {pretrained}")
    print(f"[INFO] Output dir: {INFER_SAVE_DIR}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[OK] Static graph tersimpan di {INFER_SAVE_DIR}/")
        print("     inference.pdmodel")
        print("     inference.pdiparams")
    else:
        print("[ERROR] Export gagal.")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
