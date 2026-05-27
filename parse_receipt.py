"""Post-processing: raw OCR text → structured receipt JSON (Fitur 2 AI-DS-SPEC).

Output format sesuai spec:
{
  "merchant": str,
  "date": "YYYY-MM-DD" | null,
  "total": int | null,
  "items": [str, ...],
  "category": str,
  "confidence": float (0-1)
}

Strategy:
  1. merchant  → baris pertama teks yang mengandung nama toko (heuristic + keyword)
  2. date      → regex multi-format tanggal
  3. total     → regex nominal terbesar yang dekat keyword total/bayar/grand
  4. items     → baris yang mengandung pola "qty x harga" atau item keyword
  5. category  → keyword matching dari merchant + items
  6. confidence→ heuristic berdasarkan jumlah field yang berhasil diekstrak
"""

import re
from datetime import datetime
from typing import Optional

# ── Kategori valid sesuai spec ───────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "makanan": [
        "indomaret", "alfamart", "alfamidi", "lawson", "circle k",
        "mcdonald", "kfc", "pizza", "burger", "bakery", "cafe", "kopi",
        "warung", "resto", "restaurant", "mie", "nasi", "ayam",
        "supermarket", "hypermart", "giant", "hero", "carrefour",
        "indomie", "teh", "minuman", "snack", "roti", "es",
    ],
    "transport": [
        "gojek", "grab", "ojek", "taxi", "transjakarta", "kereta",
        "pertamina", "shell", "spbu", "bensin", "bbm", "parkir",
        "toll", "tol", "airport", "bandara",
    ],
    "belanja": [
        "tokopedia", "shopee", "lazada", "bukalapak", "blibli",
        "zalora", "uniqlo", "h&m", "zara", "miniso", "ikea",
        "ace hardware", "informa", "electronic", "erafone", "ibox",
    ],
    "kesehatan": [
        "apotek", "apotik", "kimia farma", "century", "guardian",
        "klinik", "rumah sakit", "rs ", "puskesmas", "dokter",
        "vitamin", "obat", "masker",
    ],
    "tagihan": [
        "pln", "listrik", "pdam", "air", "telkom", "indihome",
        "firstmedia", "biznet", "wifi", "internet", "pulsa", "token",
    ],
    "hiburan": [
        "cinema", "cgv", "cinepolis", "xxi", "bioskop", "studio",
        "spotify", "netflix", "youtube", "game", "playstation",
    ],
    "pendidikan": [
        "gramedia", "togamas", "buku", "alat tulis", "stationery",
        "kursus", "les", "sekolah", "universitas",
    ],
}

# ── Regex patterns ─────────────────────────────────────────────────────────────────────────
DATE_PATTERNS = [
    # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', '%d/%m/%Y'),
    # YYYY/MM/DD, YYYY-MM-DD
    (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', '%Y/%m/%d'),
    # DD Month YYYY (e.g. 23 April 2026)
    (r'(\d{1,2})\s+(Jan(?:uari)?|Feb(?:ruari)?|Mar(?:et)?|Apr(?:il)?|Mei|Jun(?:i)?|'
     r'Jul(?:i)?|Agu(?:stus)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?|Des(?:ember)?)\s+(\d{4})',
     'id_month'),
]

MONTH_ID = {
    'jan': 1, 'januari': 1, 'feb': 2, 'februari': 2, 'mar': 3, 'maret': 3,
    'apr': 4, 'april': 4, 'mei': 5, 'jun': 6, 'juni': 6, 'jul': 7, 'juli': 7,
    'agu': 8, 'agustus': 8, 'sep': 9, 'september': 9, 'okt': 10, 'oktober': 10,
    'nov': 11, 'november': 11, 'des': 12, 'desember': 12,
}

TOTAL_KEYWORDS = r'(?:total|grand\s*total|jumlah|bayar|pembayaran|tagihan|amount)'
NOMINAL_RE = re.compile(r'(?:rp\.?|idr)?\s*([0-9]{1,3}(?:[.,][0-9]{3})+|[0-9]{4,})', re.IGNORECASE)
ITEM_LINE_RE = re.compile(
    r'^(?:.*?\s)?(\d+)\s*[xX]\s*(?:rp\.?\s*)?([0-9.,]+)|'
    r'^([A-Za-z][\w\s/]{2,30})\s+(?:rp\.?\s*)?([0-9.,]+)$',
    re.IGNORECASE
)


# ── Helper functions ─────────────────────────────────────────────────────────────────────────
def _parse_nominal(s: str) -> Optional[int]:
    """Bersihkan string nominal Rupiah dan konversi ke int."""
    s = s.replace(',', '').replace('.', '').strip()
    try:
        return int(s)
    except ValueError:
        return None


def _extract_date(lines: list[str]) -> Optional[str]:
    for line in lines:
        for pat, fmt in DATE_PATTERNS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                try:
                    if fmt == 'id_month':
                        day, month_str, year = m.group(1), m.group(2).lower(), m.group(3)
                        month_num = MONTH_ID.get(month_str)
                        if month_num:
                            return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
                    elif '/' in fmt:
                        sep = re.search(r'[/\-\.]', m.group(0)).group()
                        raw = m.group(0).replace(sep, '/')
                        dt = datetime.strptime(raw, fmt)
                        return dt.strftime('%Y-%m-%d')
                except Exception:
                    continue
    return None


def _extract_total(lines: list[str]) -> Optional[int]:
    """Cari nominal terbesar di baris yang mengandung keyword total/bayar."""
    candidates = []
    for line in lines:
        if re.search(TOTAL_KEYWORDS, line, re.IGNORECASE):
            for m in NOMINAL_RE.finditer(line):
                val = _parse_nominal(m.group(1))
                if val and val >= 100:  # minimum Rp100
                    candidates.append(val)
    # Jika tidak ada keyword total, ambil nominal terbesar di seluruh struk
    if not candidates:
        for line in lines:
            for m in NOMINAL_RE.finditer(line):
                val = _parse_nominal(m.group(1))
                if val and val >= 1000:
                    candidates.append(val)
    return max(candidates) if candidates else None


def _extract_merchant(lines: list[str]) -> str:
    """Merchant biasanya ada di 3 baris pertama struk."""
    # Prioritaskan baris pertama non-kosong yang bukan nomor/tanggal
    for line in lines[:5]:
        line = line.strip()
        if len(line) >= 3 and not re.match(r'^\d', line):
            return line.title()
    return lines[0].strip().title() if lines else "Unknown"


def _extract_items(lines: list[str]) -> list[str]:
    """Ekstrak item belanja dari baris yang mengandung pola qty x harga."""
    items = []
    skip_keywords = re.compile(
        r'total|subtotal|bayar|kembalian|cash|tunai|diskon|ppn|pajak|tanggal|no\.?\s*struk',
        re.IGNORECASE
    )
    for line in lines:
        line = line.strip()
        if not line or re.search(skip_keywords, line):
            continue
        # Pola: "2 x Indomie Goreng 3000" atau "Indomie Goreng 3000"
        m = ITEM_LINE_RE.match(line)
        if m:
            if m.group(1):  # pola qty x
                # ambil nama item dari konteks sekitar (simplifikasi: pakai line asli)
                items.append(line.strip())
            elif m.group(3):  # pola nama harga
                items.append(m.group(3).strip())
    # Fallback: jika tidak ada item terdeteksi, ambil baris tengah (exclude 3 awal & akhir)
    if not items and len(lines) > 6:
        middle = lines[3:-3]
        items = [l.strip() for l in middle if l.strip() and not re.search(skip_keywords, l)]
    return items[:20]  # batasi 20 item


def _classify_category(merchant: str, items: list[str]) -> str:
    """Klasifikasi kategori berdasarkan keyword matching."""
    text = (merchant + ' ' + ' '.join(items)).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "lainnya"


def _compute_confidence(merchant: str, date, total, items: list) -> float:
    """Heuristic confidence berdasarkan kelengkapan field."""
    score = 0.0
    if merchant and merchant != "Unknown":
        score += 0.30
    if date:
        score += 0.25
    if total and total > 0:
        score += 0.30
    if items:
        score += min(len(items) * 0.03, 0.15)  # max 0.15 dari items
    return round(min(score, 1.0), 2)


# ── Main function ─────────────────────────────────────────────────────────────────────────
def parse_receipt(raw_text: str) -> dict:
    """Parse raw OCR text → structured receipt dict sesuai AI-DS-SPEC Fitur 2.

    Args:
        raw_text: String hasil penggabungan semua OCR output, satu baris per \n.

    Returns:
        dict dengan key: merchant, date, total, items, category, confidence.
    """
    lines = [l for l in raw_text.strip().split('\n') if l.strip()]

    merchant = _extract_merchant(lines)
    date = _extract_date(lines)
    total = _extract_total(lines)
    items = _extract_items(lines)
    category = _classify_category(merchant, items)
    confidence = _compute_confidence(merchant, date, total, items)

    return {
        "merchant": merchant,
        "date": date,
        "total": total,
        "items": items,
        "category": category,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # Contoh penggunaan
    sample = """
    INDOMARET
    Jl. Sudirman No. 10, Jakarta
    No. Struk: 0001-20260527
    Tanggal: 27/05/2026 14:30

    Indomie Goreng         3.000
    Teh Botol Sosro        5.000
    Aqua 600ml             4.000
    2 x Chiki Balls        4.000

    Subtotal              16.000
    Diskon                 1.000
    TOTAL                 15.000
    Tunai                 20.000
    Kembalian              5.000
    """
    import json
    result = parse_receipt(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
