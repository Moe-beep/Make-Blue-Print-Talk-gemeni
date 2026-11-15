# Raster2Seq — inference only

Inference-only build of the official PyTorch implementation of
["Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction"](https://arxiv.org/abs/2602.09016) (SIGGRAPH'26),
by Hao Phung and Hadar Averbuch-Elor (Cornell University).
[[Page]](https://cornell-vailab.github.io/Raster2Seq/) [[Paper]](https://arxiv.org/abs/2602.09016) [[Hugging Face]](https://huggingface.co/haopt/Raster2Seq)

Raster2Seq converts a rasterized floorplan image into a vectorized floorplan, represented as a labeled polygon sequence.
Training, evaluation, data preprocessing and the VLM refinement step were removed from this copy; they are available
in the upstream repository and in this repository's git history.

```bibtex
@inproceedings{phung2026raster2seq,
   title={Raster2Seq: Polygon Sequence Generation for Floorplan Reconstruction},
   author={Phung, Hao and Averbuch-Elor, Hadar},
   booktitle={Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
   year= {2026},
}
```

## Pipeline

```
image folder ──► predict.py
                  ├─ datasets/transforms.py        resize + pad to image_size (default 256)
                  ├─ raster2seq_hub.py             resolve hf:<key> checkpoints from Hugging Face
                  ├─ models/                       Raster2Seq: ResNet backbone + deformable encoder
                  │                                + anchor-based autoregressive decoder (KV cache)
                  ├─ datasets/discrete_tokenizer.py  coordinate quantization / token types
                  ├─ engine.py                     generate: decode token sequences into
                  │                                room, door, window polygons
                  └─ util/plot_utils.py            overlay + floorplan rendering
                 ► <output_dir>/<ckpt>/  *_pred_room_map.png, *_pred_floorplan.png, jsons/, npy/
```

## Installation (NVIDIA GPU)

Requirements: Linux, an NVIDIA GPU with its driver, python 3.10. Tested with pytorch 2.3.1 + CUDA 11.8.

### Option A — Docker (recommended: nvcc and torch versions already match)

Needs Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```shell
docker build -t raster2seq .
docker run --gpus all --rm raster2seq                      # runs tools/check_cuda.py
docker run --gpus all --rm \
    -v "$PWD/data:/workspace/data" -v "$PWD/pred_outputs:/workspace/pred_outputs" \
    raster2seq bash tools/predict_cc5k.sh
```

The deformable-attention op is compiled at build time for compute capabilities 7.0–9.0 (V100, T4, A100,
A10/A6000/RTX 30xx, L4/RTX 40xx, H100). For other GPUs change `TORCH_CUDA_ARCH_LIST` in the `Dockerfile`.

### Option B — directly on the machine

Needs a CUDA toolkit (`nvcc`) with the same major version as the torch build (`cu118` -> CUDA 11.8).

```shell
conda create -n raster2seq python=3.10 && conda activate raster2seq
bash tools/setup_cuda.sh                                          # torch cu118 + requirements + build op + check
CUDA_VERSION=cu121 TORCH_VERSION=2.3.1 bash tools/setup_cuda.sh   # other CUDA builds
```

`tools/setup_cuda.sh` installs torch and `requirements.txt`, compiles the CUDA op with `models/ops/make.sh` and runs
`python tools/check_cuda.py`, which checks the GPU, the compiled op, and the op's output against the PyTorch reference.
To build without a visible GPU (e.g. on a login node), set `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST`.

If the op is not compiled, the model falls back to a pure-PyTorch deformable attention (with a warning on GPU):
same outputs, but slower. This also allows running on CPU with `--device cpu`.

## Checkpoints

Hosted at [haopt/Raster2Seq](https://huggingface.co/haopt/Raster2Seq), one subfolder per model. The predict scripts use
`hf:<key>` aliases and download the checkpoint on first use; `tools/download_checkpoints.sh` downloads all of them.

| Dataset                 | RoomF1 | Hugging Face key   |
| ----------------------- | ------ | ------------------ |
| Structured3D            | 99.6   | `s3d-bw`           |
| CubiCasa5K              | 88.7   | `cubicasa5k`       |
| Raster2Graph            | 97.0   | `raster2graph`     |
| Structured3D-DensityMap | 99.1   | `s3d-density`      |
| Raster2Graph (512x512)  | 98.1   | `raster2graph-512` |

## Inference

`predict.py` runs on every image found (recursively) under `--dataset_root`, on `--device cuda` by default.

| Dataset                 | Bash Script                    |
| ----------------------- | ------------------------------ |
| Structured3D            | `tools/predict_s3d.sh`         |
| CubiCasa5K              | `tools/predict_cc5k.sh`        |
| Raster2Graph            | `tools/predict_r2g.sh`         |
| WAFFLE                  | `tools/predict_waffle.sh`      |
| Structured3D-DensityMap | `tools/predict_s3d_density.sh` |

For WAFFLE (or any other floorplan images), the CubiCasa5K checkpoint is used. Useful flags:
`--save_pred` (JSON + NPY polygons), `--batch_size`, `--plot_text`, `--one_color`, `--crop_white_space`,
`--drop_wd` (drop doors/windows), `--measure_time`.

Inference takes about 0.52 s per image at batch size 1 on a single RTX A6000.

## License

Source code is licensed under the [MIT License](LICENSE). Third-party code retains its own license and notices.

The pretrained weights (`s3d-bw`, `s3d-density`, `cubicasa5k`, `raster2graph`, `raster2graph-512`) are licensed
separately under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — see [LICENSE-CHECKPOINTS](LICENSE-CHECKPOINTS).
Commercial use of these weights, including inference in a commercial SaaS product, is not authorized by this grant;
for permission inquiries contact the authors (htp26@cornell.edu).

## Acknowledgment

Built upon [RoomFormer](https://github.com/ywyue/RoomFormer) and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR),
with inspiration from [PolyFormer](https://github.com/amazon-science/polygon-transformer).
