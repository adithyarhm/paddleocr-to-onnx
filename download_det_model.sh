#!/bin/bash
# Download PP-OCRv4 detection model (DBNet++) - pretrained, tidak perlu fine-tune
# Model ini generik dan bekerja baik untuk struk/invoice

MODEL_DIR="./inference/det_onnx"
mkdir -p $MODEL_DIR

echo "[INFO] Downloading PP-OCRv4 det model (English)..."

# Download Paddle inference model dulu
mkdir -p ./pretrained_models/det
wget -q --show-progress \
  https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar \
  -O ./pretrained_models/det/en_PP-OCRv4_det_infer.tar

tar -xf ./pretrained_models/det/en_PP-OCRv4_det_infer.tar \
  -C ./pretrained_models/det/

echo "[INFO] Converting det model ke ONNX..."
python3 -c "
import paddle2onnx
paddle2onnx.command(
    model_dir='./pretrained_models/det/en_PP-OCRv4_det_infer',
    model_filename='inference.pdmodel',
    params_filename='inference.pdiparams',
    save_file='$MODEL_DIR/model.onnx',
    opset_version=11,
    enable_dev_version=True,
)
print('[OK] Det model saved to $MODEL_DIR/model.onnx')
"

echo "[DONE] Detection model siap di: $MODEL_DIR/model.onnx"
