# GPU image for Raster2Seq inference. Build and run:
#   docker build -t raster2seq .
#   docker run --gpus all --rm -v "$PWD/data:/workspace/data" -v "$PWD/pred_outputs:/workspace/pred_outputs" \
#       raster2seq bash tools/predict_cc5k.sh
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# No GPU is visible during `docker build`: compile the op for these architectures (Volta -> Hopper).
ENV FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ops models/ops
RUN sh models/ops/make.sh && rm -rf models/ops/build

COPY . .
CMD ["python", "tools/check_cuda.py"]
