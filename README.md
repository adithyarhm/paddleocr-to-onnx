# PaddleOCR v4 Fine-tune → ONNX (CORD-1000)

Pipeline fine-tuning PP-OCRv4 recognition model pada dataset CORD-1000 lalu export ke ONNX untuk inferensi dengan ONNXRuntime (CPU).

## Pipeline Overview

```
CORD-1000 dataset
    ↓
data/prepare_cord.py        # Parse JSON CORD → crop word images + rec_gt.txt
    ↓
train.py                    # Fine-tune PP-OCRv4 rec (wrapper tools/train.py)
    ↓
export_inference.py         # .pdparams → Paddle static graph (.pdmodel + .pdiparams)
    ↓
convert_to_onnx.py          # Paddle static graph → .onnx via paddle2onnx
    ↓
inference_onnx.py           # Inferensi dengan ONNXRuntime CPU
```

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** Install PaddlePaddle CPU terlebih dahulu sebelum install requirements.
> ```bash
> pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

## Quick Start

### 1. Siapkan dataset CORD-1000

Download dari [Kaggle](https://www.kaggle.com/datasets/lonelvino/cord-1000) dan ekstrak ke folder `data/cord_1000/`.
Struktur yang diharapkan:
```
data/cord_1000/
├── train/
│   ├── image/       # gambar receipt
│   └── json/        # anotasi ground truth
├── dev/
│   ├── image/
│   └── json/
└── test/
    ├── image/
    └── json/
```

Lalu jalankan script parser:
```bash
python data/prepare_cord.py
```
Output: `data/rec_train/` (cropped word images + `rec_gt_train.txt`) dan `data/rec_val/` (`rec_gt_val.txt`).

### 2. Clone PaddleOCR & Download pretrained model

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && pip install -e . && cd ..

mkdir -p pretrained_models
wget -P pretrained_models https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar
tar -xf pretrained_models/en_PP-OCRv4_rec_train.tar -C pretrained_models/
```

### 3. Fine-tune

```bash
python train.py
```

Atau langsung via PaddleOCR CLI:
```bash
python PaddleOCR/tools/train.py -c configs/rec_ppocr_v4_cord.yml
```

### 4. Export ke Paddle static graph

```bash
python export_inference.py
```

### 5. Convert ke ONNX

```bash
python convert_to_onnx.py
```

### 6. Inferensi dengan ONNXRuntime

```bash
python inference_onnx.py --image_path path/to/receipt.jpg
```

## Struktur Repo

```
├── data/
│   ├── prepare_cord.py         # Parser CORD JSON → PaddleOCR rec format
│   └── cord_1000/              # Dataset (letakkan di sini setelah download)
├── configs/
│   └── rec_ppocr_v4_cord.yml   # Config fine-tune PP-OCRv4 rec
├── train.py
├── export_inference.py
├── convert_to_onnx.py
├── inference_onnx.py
├── requirements.txt
└── README.md
```
