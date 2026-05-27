"""Pipeline lengkap: gambar struk → OCR → structured JSON (Fitur 2 AI-DS-SPEC).

Menggabungkan:
  - inference_onnx.py  (OCR engine, ONNX + ONNXRuntime CPU)
  - parse_receipt.py   (post-processing, ekstrak field dari raw text)

Usage:
  python ocr_receipt.py --image_path path/to/struk.jpg
  python ocr_receipt.py --image_path struk.jpg --output_json result.json

Output JSON:
  {
    "merchant": "Indomaret",
    "date": "2026-05-27",
    "total": 15000,
    "items": ["Indomie Goreng", "Teh Botol x1"],
    "category": "makanan",
    "confidence": 0.85
  }
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from inference_onnx import load_session, preprocess, CTCDecoder
from parse_receipt import parse_receipt


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_REC_MODEL = "./inference/rec_onnx/model.onnx"
DICT_PATH = "./PaddleOCR/ppocr/utils/en_dict.txt"


def ocr_full_image(img_path: str, session: ort.InferenceSession, decoder: CTCDecoder) -> str:
    """Deteksi teks dari gambar struk secara full-page (sliding window horizontal).

    Strategi: potong gambar menjadi strip horizontal dengan tinggi = REC_H,
    overlap 10px, jalankan rec model di setiap strip. Hasilnya digabung per baris.

    Note: Untuk akurasi maksimal, gunakan PaddleOCR det model untuk deteksi
    bounding box terlebih dahulu. Sliding window ini adalah simplified approach
    yang cukup untuk struk dengan layout 1 kolom lurus.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")

    REC_H = 48
    STEP = REC_H - 10  # overlap 10px
    input_name = session.get_inputs()[0].name

    h, w = img.shape[:2]
    lines_text = []

    y = 0
    while y < h:
        y_end = min(y + REC_H, h)
        strip = img[y:y_end, :]
        if strip.shape[0] < 10:  # skip strip terlalu kecil
            break

        inp = preprocess(strip, target_h=REC_H)
        output = session.run(None, {input_name: inp})
        pred = output[0]
        if pred.ndim == 3:
            pred = pred[0]

        text = decoder.decode(pred).strip()
        if text:
            lines_text.append(text)

        y += STEP

    return '\n'.join(lines_text)


def process_receipt(image_path: str, rec_model: str = DEFAULT_REC_MODEL,
                    dict_path: str = DICT_PATH) -> dict:
    """End-to-end: gambar → OCR → JSON sesuai AI-DS-SPEC Fitur 2."""
    print(f"[INFO] Loading ONNX model: {rec_model}")
    session = load_session(rec_model)
    decoder = CTCDecoder(dict_path)

    print(f"[INFO] Running OCR pada: {image_path}")
    raw_text = ocr_full_image(image_path, session, decoder)

    print(f"[INFO] Raw OCR output:\n{'='*40}")
    print(raw_text)
    print('='*40)

    print("[INFO] Parsing structured fields...")
    result = parse_receipt(raw_text)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="OCR Struk/Invoice → JSON (AI-DS-SPEC Fitur 2)"
    )
    parser.add_argument("--image_path", required=True, help="Path ke gambar struk (JPG/PNG)")
    parser.add_argument("--rec_model", default=DEFAULT_REC_MODEL, help="Path ke .onnx model")
    parser.add_argument("--dict_path", default=DICT_PATH, help="Path ke character dict")
    parser.add_argument("--output_json", default=None, help="Simpan hasil ke file JSON (opsional)")
    args = parser.parse_args()

    result = process_receipt(args.image_path, args.rec_model, args.dict_path)

    print("\n[RESULT] Structured JSON output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[OK] Disimpan ke: {args.output_json}")


if __name__ == "__main__":
    main()
