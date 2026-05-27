"""FastAPI wrapper untuk OCR Struk/Invoice (AI-DS-SPEC Fitur 2).

Endpoint:
  POST /ocr/receipt
    - Body: multipart/form-data, field "file" (JPG/PNG)
    - Response: JSON sesuai AI-DS-SPEC Fitur 2

Usage:
  pip install fastapi uvicorn python-multipart
  uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Test via curl:
  curl -X POST http://localhost:8001/ocr/receipt \\
    -F "file=@path/to/struk.jpg"
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from inference_onnx import CTCDecoder, load_session
from ocr_receipt import ocr_full_image
from parse_receipt import parse_receipt

# ── Config ────────────────────────────────────────────────────────────────────
REC_MODEL = "./inference/rec_onnx/model.onnx"
DICT_PATH = "./PaddleOCR/ppocr/utils/en_dict.txt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE_MB = 10


# ── Response schema (sesuai AI-DS-SPEC) ─────────────────────────────────────────────────────
class ReceiptResponse(BaseModel):
    merchant: str
    date: Optional[str] = None       # YYYY-MM-DD atau null
    total: Optional[int] = None
    items: List[str] = []
    category: str
    confidence: float                 # 0.0 – 1.0


# ── App setup ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCR Receipt API",
    description="OCR Struk/Invoice → JSON (AI-DS-SPEC Fitur 2). Powered by PP-OCRv4 + ONNX.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model sekali saat startup (singleton)
_session = None
_decoder = None


@app.on_event("startup")
async def load_model():
    global _session, _decoder
    print(f"[INFO] Loading ONNX model: {REC_MODEL}")
    _session = load_session(REC_MODEL)
    _decoder = CTCDecoder(DICT_PATH)
    print("[INFO] Model loaded. API ready.")


# ── Endpoints ─────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _session is not None}


@app.post("/ocr/receipt", response_model=ReceiptResponse)
async def ocr_receipt(file: UploadFile = File(...)):
    """Upload gambar struk/invoice, kembalikan data terstruktur sesuai AI-DS-SPEC Fitur 2.

    - **file**: Gambar struk (JPG/PNG/BMP/WEBP, max 10MB)
    """
    # Validasi ekstensi
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {suffix}. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Baca konten file
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File terlalu besar (max {MAX_FILE_SIZE_MB}MB)")

    # Decode gambar dari bytes
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Gambar tidak dapat dibaca atau corrupt.")

    # Simpan ke tempfile (ocr_full_image butuh path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        raw_text = ocr_full_image(tmp_path, _session, _decoder)
        result = parse_receipt(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR gagal: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ReceiptResponse(**result)
