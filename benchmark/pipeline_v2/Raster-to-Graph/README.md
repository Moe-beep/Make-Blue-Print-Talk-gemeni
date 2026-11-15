# Raster-to-Graph (inference only)

Inference pipeline from the official implementation of [Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer](https://wutomwu.github.io/publications/2024-Raster2Graph/paper.pdf) (EG 2024), by Sizhe Hu, Wenming Wu, Ruolin Su, Wanni Hou, Liping Zheng and Benzhu Xu.

Training, evaluation and dataset-preparation code have been removed. This repo only turns raster floorplan images into structural graphs (wall junctions, wall segments, room semantics) using a trained model.

## GPU setup (CUDA)
Requires an NVIDIA GPU: the multi-scale deformable attention op only runs on CUDA.

1. Install the NVIDIA driver and the **CUDA Toolkit** (`nvcc`) whose major version matches the PyTorch CUDA build below. On Windows you also need the *Visual Studio Build Tools* (C++ workload) and should run the build from the *x64 Native Tools* prompt.

2. Install PyTorch (>= 2.4) with CUDA, picking the CUDA version for your driver on [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

   Check it sees the GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
   ```

3. Install the other dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Build the deformable attention CUDA op (from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)):

   ```bash
   cd models/ops && python -m pip install --no-build-isolation -v . && cd ../..
   ```

   (`sh models/ops/make.sh` does the same.) To build where no GPU is visible (e.g. a docker build) set `FORCE_CUDA=1`, and set `TORCH_CUDA_ARCH_LIST` (e.g. `8.6`) to target a specific GPU architecture.

## Checkpoint
Download the [trained model](https://drive.google.com/file/d/1A_pyF0KHo-fja-fNqkoFkMUxo0W7GHhC/view) and place it at `output/trained_r2gmodel/checkpoint0299.pth` (or pass `--resume path/to/checkpoint.pth`).

## Usage
Put your floorplan images in `input/`, then run:

```bash
python inference.py
```

Options:

- `--input_dir` folder with the images (default `./input`)
- `--output_dir` where results are written (default `./output/inference`)
- `--resume` checkpoint path
- `--device` CUDA device, e.g. `cuda:1` (default `cuda`)
- `--num_workers` data loading workers (default `0`; keep `0` on Windows)
- `--allow_unsafe_load` load the checkpoint with full pickle, only if safe loading fails and you trust the file

Each image is scaled so its longer side is 512 px and centered on a white 512x512 canvas. Visualized results go to `<output_dir>/val_visualize_iter/epoch0`. The model was trained on centered 512x512 floorplans from the LIFULL HOME'S data; images that look very different may need retraining with the original repository.

## Citation
If you use our code or dataset, please cite Raster-to-Graph:

Bibtex:
```
@article{https://doi.org/10.1111/cgf.15007,
author = {Hu, Sizhe and Wu, Wenming and Su, Ruolin and Hou, Wanni and Zheng, Liping and Xu, Benzhu},
year = {2024},
title = {Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer},
journal = {Computer Graphics Forum},
volume = {43},
number = {2},
pages = {e15007},
keywords = {CCS Concepts, • Computing methodologies → Shape modeling, Computer vision},
doi = {https://doi.org/10.1111/cgf.15007},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15007},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.15007}
}
```
