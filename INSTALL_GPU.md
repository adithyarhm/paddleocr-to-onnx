# Panduan Install PaddlePaddle GPU

Untuk training di **NVIDIA RTX 4080 SUPER** dengan CUDA.

## 1. Cek versi CUDA

```bash
nvidia-smi
```

RTX 4080 SUPER mendukung CUDA 11.8 dan 12.x.
Disarankan pakai **CUDA 11.8** karena paling stabil dengan PaddlePaddle 3.0.

## 2. Install CUDA Toolkit + cuDNN

Download dari NVIDIA:
- CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- cuDNN 8.6 for CUDA 11.x: https://developer.nvidia.com/rdp/cudnn-archive

Verifikasi:
```bash
nvcc --version          # harus menampilkan 11.8
python -c "import ctypes; ctypes.CDLL('libcudnn.so.8')"  # Linux
```

## 3. Install PaddlePaddle GPU

```bash
# CUDA 11.8 (recommended)
python -m pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Alternatif CUDA 12.3:
```bash
python -m pip install paddlepaddle-gpu==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

Verifikasi GPU terdeteksi:
```bash
python -c "import paddle; paddle.utils.run_check()"
# Output: PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
python -c "import paddle; print(paddle.device.get_device())"
# Output: gpu:0
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Mulai training

```bash
python train.py
```

Estimasi waktu training dengan RTX 4080 SUPER:
- ~50 epoch pada CORD-1000 (~10k samples): **± 10–15 menit**
- Dibanding CPU Ryzen 5 6600H: **~20–30x lebih cepat**

## Troubleshooting

**Error: CUDA out of memory**
Kurangi `batch_size_per_card` di `configs/rec_ppocr_v4_cord.yml`:
```yaml
loader:
  batch_size_per_card: 128  # turunkan dari 256
```

**Error: libcudnn not found**
Pastikan cuDNN sudah di-install dan path-nya terdaftar:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**GPU tidak terdeteksi:**
```bash
nvidia-smi  # pastikan driver aktif
python -c "import paddle; print(paddle.device.cuda.device_count())"
```
