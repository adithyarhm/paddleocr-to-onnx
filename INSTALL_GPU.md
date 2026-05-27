# Panduan Install PaddlePaddle GPU

Untuk training di **NVIDIA RTX 4080 SUPER** dengan CUDA.

---

## ⚠️ Penting: CUDA 12.0 Tidak Didukung

PaddlePaddle 3.0.0 **tidak memiliki build untuk CUDA 12.0**.
Versi CUDA yang didukung secara resmi:

| CUDA Version | PaddlePaddle Index URL |
|---|---|
| 11.8 | `https://www.paddlepaddle.org.cn/packages/stable/cu118/` |
| 12.6 | `https://www.paddlepaddle.org.cn/packages/stable/cu126/` |

Kamu saat ini menggunakan **CUDA 12.0**, pilih salah satu opsi di bawah.

---

## Opsi A — Upgrade CUDA ke 12.6 (Direkomendasikan ✅)

RTX 4080 SUPER mendukung CUDA 12.6 dan ini adalah opsi paling mudah.

### 1. Download & Install CUDA 12.6
- https://developer.nvidia.com/cuda-12-6-0-download-archive
- cuDNN 9.x for CUDA 12.x: https://developer.nvidia.com/cudnn-downloads

### 2. Verifikasi
```bash
nvcc --version
# Cuda compilation tools, release 12.6, ...
```

### 3. Install PaddlePaddle GPU (CUDA 12.6)
```bash
python -m pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

---

## Opsi B — Install CUDA 11.8 Secara Paralel

Windows/Linux support multi-versi CUDA. Kamu bisa tetap punya CUDA 12.0 dan install CUDA 11.8 di direktori berbeda.

### 1. Download & Install CUDA 11.8
- https://developer.nvidia.com/cuda-11-8-0-download-archive
- cuDNN 8.6 for CUDA 11.x: https://developer.nvidia.com/rdp/cudnn-archive

### 2. Set environment variable agar Python pakai CUDA 11.8

**Windows (PowerShell):**
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

**Linux:**
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. Install PaddlePaddle GPU (CUDA 11.8)
```bash
python -m pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

---

## Verifikasi Setelah Install

```bash
python -c "import paddle; paddle.utils.run_check()"
# Output: PaddlePaddle is installed successfully!
python -c "import paddle; print(paddle.device.get_device())"
# Output: gpu:0
```

---

## Install Dependencies & Training

```bash
pip install -r requirements.txt
python train.py
```

Estimasi waktu training dengan RTX 4080 SUPER:
- ~50 epoch pada CORD-1000 (~10k samples): **± 10–15 menit**
- Dibanding CPU Ryzen 5 6600H: **~20–30x lebih cepat**

---

## Troubleshooting

**CUDA 12.0 error saat import paddle:**
```
Could not load dynamic library 'libcudart.so.11.0'
```
Ini konfirmasi bahwa CUDA 12.0 tidak kompatibel. Ikuti Opsi A atau B di atas.

**Error: CUDA out of memory**
Kurangi `batch_size_per_card` di `configs/rec_ppocr_v4_cord.yml`:
```yaml
loader:
  batch_size_per_card: 128  # turunkan dari 256
```

**GPU tidak terdeteksi:**
```bash
nvidia-smi   # pastikan driver aktif
python -c "import paddle; print(paddle.device.cuda.device_count())"
```
