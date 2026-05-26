"""Convert Paddle static graph → ONNX menggunakan paddle2onnx.

Harus dijalankan SETELAH export_inference.py.
Output: ./inference/rec_onnx/model.onnx

Run: python convert_to_onnx.py
"""

import subprocess
import sys
from pathlib import Path

INFER_DIR = Path("./inference/rec_infer")
ONNX_SAVE = "./inference/rec_onnx/model.onnx"
OPSET = 11  # Opset 11 stabil untuk PP-OCRv4 rec (SVTR_LCNet)


def check_prerequisites():
    pdmodel = INFER_DIR / "inference.pdmodel"
    pdiparams = INFER_DIR / "inference.pdiparams"
    if not pdmodel.exists() or not pdiparams.exists():
        print("[ERROR] Static graph tidak ditemukan di:", INFER_DIR)
        print("  Jalankan terlebih dahulu: python export_inference.py")
        sys.exit(1)


def convert():
    Path(ONNX_SAVE).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "paddle2onnx",
        "--model_dir", str(INFER_DIR),
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", ONNX_SAVE,
        "--opset_version", str(OPSET),
        "--enable_onnx_checker", "True",
    ]
    print(f"[INFO] Konversi ke ONNX (opset={OPSET})...")
    print(f"[INFO] Input : {INFER_DIR}")
    print(f"[INFO] Output: {ONNX_SAVE}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[OK] ONNX model tersimpan: {ONNX_SAVE}")
        # Optimasi input shape agar dynamic batch
        optimize_shape()
    else:
        print("[ERROR] Konversi gagal.")
    sys.exit(result.returncode)


def optimize_shape():
    """Set dynamic input shape supaya bisa inference berbagai ukuran gambar."""
    cmd = [
        sys.executable, "-m", "paddle2onnx.optimize",
        "--input_model", ONNX_SAVE,
        "--output_model", ONNX_SAVE,
        "--input_shape_dict", "{'x': [-1, 3, 48, -1]}",
    ]
    print("[INFO] Optimasi input shape ke dynamic...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("[OK] Shape optimization selesai.")
    else:
        print("[WARN] Shape optimization gagal (opsional, tidak mempengaruhi konversi).")


if __name__ == "__main__":
    check_prerequisites()
    convert()
