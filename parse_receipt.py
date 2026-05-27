"""Post-processing: raw OCR text → structured receipt JSON (Fitur 2 AI-DS-SPEC).

Output format sesuai spec:
{
  "merchant": str,
  "date": "YYYY-MM-DD" | null,
  "total": int | null,
  "items": [{"name": str, "price": int}],
  "category": str,
  "confidence": float (0-1)
}

Strategy:
  - Det model menghasilkan baris terpisah per box. Format struk restoran/minimarket
    umumnya: nama item di satu baris, harga di baris berikutnya.
  - Parser pairing: jika baris saat ini adalah teks (nama) dan baris berikutnya adalah
    angka (harga), pasangkan sebagai satu item.
  - Merchant: baris teks terpanjang di 10 baris pertama yang bukan item/noise
  - Total: baris SETELAH keyword grand total / total, atau nominal terbesar
"""

import re
from datetime import datetime
from typing import Optional

# ── Kategori valid sesuai spec ────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "makanan": [
        "indomaret", "alfamart", "alfamidi", "lawson", "circle k",
        "mcdonald", "kfc", "pizza", "burger", "bakery", "cafe", "kopi",
        "warung", "resto", "restaurant", "mie", "nasi", "ayam", "bebek",
        "supermarket", "hypermart", "giant", "hero", "carrefour",
        "indomie", "teh", "minuman", "snack", "roti", "es", "milk", "ice",
        "tahu", "tempe", "goreng", "sambal", "organic",
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

# ── Regex patterns ───────────────────────────────────────────────────────────────
DATE_PATTERNS = [
    (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', '%d/%m/%Y'),
    (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', '%Y/%m/%d'),
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

TOTAL_KW_RE  = re.compile(r'(?:grand[-\s]?total|total|jumlah|bayar|pembayaran|tagihan|amount)', re.IGNORECASE)
SUBTOTAL_KW_RE = re.compile(r'(?:sub[-\s]?total|subtotal)', re.IGNORECASE)
SKIP_KW_RE   = re.compile(
    r'^(?:x|X|--|-)$|'
    r'(?:kembalian|cash|tunai|diskon|ppn|pajak|service|pb1|rounding|free)',
    re.IGNORECASE
)
NOMINAL_FULL = re.compile(r'^(?:rp\.?\s*)?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,2})?|[0-9]{4,})$', re.IGNORECASE)
NOMINAL_ANY  = re.compile(r'(?:rp\.?\s*)?([0-9]{1,3}(?:[.,][0-9]{3})+|[0-9]{4,})', re.IGNORECASE)
ITEM_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9\s/\-&\']{2,}$')
NOISE_RE     = re.compile(r'^[A-Za-z]{1,2}$|^-+$|^\.+$')  # single/double char dan separator


# ── Helper functions ────────────────────────────────────────────────────────────────────────
def _parse_nominal(s: str) -> Optional[int]:
    """Bersihkan string nominal Rupiah dan konversi ke int."""
    # Deteksi format: jika menggunakan titik sebagai ribuan dan koma sebagai desimal
    # contoh: 1.346.000 atau 1,346,000 atau 40000 atau 40.000
    s = s.strip().rstrip('.')
    # Hapus Rp prefix
    s = re.sub(r'^(?:rp\.?\s*)', '', s, flags=re.IGNORECASE)
    # Format titik sebagai pemisah ribuan (1.346.000 -> 1346000)
    if re.match(r'^[0-9]{1,3}(?:\.[0-9]{3})+$', s):
        return int(s.replace('.', ''))
    # Format koma sebagai pemisah ribuan (1,346,000 -> 1346000)
    if re.match(r'^[0-9]{1,3}(?:,[0-9]{3})+$', s):
        return int(s.replace(',', ''))
    # Integer polos
    try:
        return int(re.sub(r'[.,]', '', s))
    except ValueError:
        return None


def _is_price_line(line: str) -> bool:
    """Apakah baris ini adalah harga (angka saja, bisa dengan titik/koma ribuan)."""
    line = line.strip().rstrip('.')
    return bool(NOMINAL_FULL.match(line.lstrip('Rp').lstrip('rp').strip()))


def _is_item_name(line: str) -> bool:
    """Apakah baris ini nama item (teks, bukan angka, bukan noise)."""
    line = line.strip()
    if not line or NOISE_RE.match(line):
        return False
    if SKIP_KW_RE.search(line):
        return False
    if TOTAL_KW_RE.search(line) or SUBTOTAL_KW_RE.search(line):
        return False
    return bool(ITEM_NAME_RE.match(line))


def _extract_date(lines: list) -> Optional[str]:
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


def _extract_total(lines: list) -> Optional[int]:
    """
    Strategi:
    1. Cari keyword 'grand total' / 'total' - ambil nominal di baris YANG SAMA
       atau baris BERIKUTNYA (format struk: label dan harga terpisah baris).
    2. Hindari Sub-Total, hanya ambil Grand Total.
    3. Fallback: nominal terbesar di seluruh struk.
    """
    candidates = []

    for i, line in enumerate(lines):
        # Skip sub-total
        if SUBTOTAL_KW_RE.search(line):
            continue

        if TOTAL_KW_RE.search(line):
            # Cek nominal di baris yang sama
            for m in NOMINAL_ANY.finditer(line):
                val = _parse_nominal(m.group(1))
                if val and val >= 1000:
                    candidates.append((val, 2))  # (nilai, prioritas)
            # Cek baris berikutnya (format terpisah)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if _is_price_line(next_line):
                    val = _parse_nominal(next_line)
                    if val and val >= 1000:
                        candidates.append((val, 2))

    if candidates:
        # Ambil yang prioritas tertinggi, lalu terbesar
        candidates.sort(key=lambda x: (-x[1], -x[0]))
        return candidates[0][0]

    # Fallback: nominal terbesar di seluruh struk
    all_vals = []
    for line in lines:
        for m in NOMINAL_ANY.finditer(line):
            val = _parse_nominal(m.group(1))
            if val and val >= 1000:
                all_vals.append(val)
    return max(all_vals) if all_vals else None


def _extract_items_and_merchant(lines: list) -> tuple:
    """
    Pairing strategy untuk format struk restoran:
    - Nama item (teks) di satu baris
    - Harga di baris berikutnya
    - Kadang ada baris 'X' (separator qty) di antara harga dan nama berikutnya

    Juga deteksi merchant dari baris teks SEBELUM item pertama.

    Returns:
        (merchant: str, items: list[dict])
    """
    items = []
    merchant = "Unknown"
    first_item_idx = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip noise/separator/skip keywords
        if not line or NOISE_RE.match(line) or SKIP_KW_RE.search(line):
            i += 1
            continue

        # Jika baris ini adalah nama item...
        if _is_item_name(line):
            # Cari harga di baris-baris berikutnya (max lookahead 3)
            found_price = False
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if not next_line or NOISE_RE.match(next_line) or SKIP_KW_RE.search(next_line):
                    continue
                if _is_price_line(next_line):
                    price_val = _parse_nominal(next_line)
                    if price_val and price_val > 0:
                        # Rekam index item pertama untuk deteksi merchant
                        if first_item_idx is None:
                            first_item_idx = i
                        items.append({"name": line, "price": price_val})
                        i = j + 1
                        found_price = True
                        break
                elif _is_item_name(next_line):
                    # Baris berikutnya juga nama item, stop lookahead
                    break
            if not found_price:
                i += 1
        else:
            i += 1

    # ── Deteksi merchant ───────────────────────────────────────────────────
    # Merchant = baris teks terpanjang di 10 baris pertama, atau sebelum item pertama
    search_range = lines[:first_item_idx] if first_item_idx else lines[:10]
    candidate_merchants = [
        l.strip() for l in search_range
        if l.strip() and _is_item_name(l.strip()) and len(l.strip()) >= 4
    ]
    if candidate_merchants:
        # Pilih yang terpanjang (nama restoran biasanya lebih panjang dari item pertama)
        merchant = max(candidate_merchants, key=len).title()
    elif items:
        # Fallback: tidak ada merchant jelas, pakai nama item pertama sebagai indicator
        merchant = "Unknown"

    # Filter: hapus item yang harganya 0 (Free items tetap disimpan dengan price=0)
    # Hapus item yang harganya mencurigakan (terlalu kecil < 100)
    items_clean = [it for it in items if it["price"] >= 100 or it["price"] == 0]

    return merchant, items_clean


def _classify_category(merchant: str, items: list) -> str:
    item_names = ' '.join(it['name'] if isinstance(it, dict) else it for it in items)
    text = (merchant + ' ' + item_names).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return "lainnya"


def _compute_confidence(merchant: str, date, total, items: list) -> float:
    score = 0.0
    if merchant and merchant != "Unknown":
        score += 0.25
    if date:
        score += 0.20
    if total and total > 0:
        score += 0.30
    if items:
        score += min(len(items) * 0.025, 0.25)  # max 0.25 dari items
    return round(min(score, 1.0), 2)


# ── Main function ────────────────────────────────────────────────────────────────────────────
def parse_receipt(raw_text: str) -> dict:
    """Parse raw OCR text → structured receipt dict sesuai AI-DS-SPEC Fitur 2.

    Args:
        raw_text: String hasil penggabungan semua OCR output, satu baris per \n.

    Returns:
        dict dengan key: merchant, date, total, items, category, confidence.
    """
    lines = [l for l in raw_text.strip().split('\n') if l.strip()]

    merchant, items = _extract_items_and_merchant(lines)
    date     = _extract_date(lines)
    total    = _extract_total(lines)
    category = _classify_category(merchant, items)
    confidence = _compute_confidence(merchant, date, total, items)

    return {
        "merchant":   merchant,
        "date":       date,
        "total":      total,
        "items":      items,
        "category":   category,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # Test dengan output aktual dari receipt_00000.png
    sample = """75,000
Nasi Campur Bali
X
125,000
Bbk Bengil Nasi
X
37,000
MilkShake Starwb
X
24,000
Ice Lemon Tea
--
70,000
Nasi Ayam Dewata
m
X
0
Free Ice Tea
X
65,000
Organic Breen Sa
18,000
IceTea-
29,000
Ice Orange
85,000
Ayam Suir Bali
36,000
Tahu Goreng
36,000
Tempe Goreng
--
40000.
Tahu Telor Asin
--
70,000
Nasi Goreng Samb
--
366000
Bbk Panggang San
3
X
92,000
Ayam Sambal Hija
X
44000
HotTea
2
X
32000
Ice Kopi
X
40000
Tahu Telor Asin
A
0
Free Ice Tea
de
44.000
Bebek Street
X
18,000
Ice Tea Tawar
A
1,346,000
Sub-Total
100950
Service-
144695
PB1
-45
Rounding-
1,591,600
-
Grand Total
--"""
    import json
    result = parse_receipt(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
