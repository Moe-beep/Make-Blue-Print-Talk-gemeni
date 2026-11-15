# Running main.py on a Verda GPU VM

## 1. Create the instance (Verda console → Instances)

| Setting | Choice | Why |
| --- | --- | --- |
| GPU | **1× L40S** or **1× RTX 6000 Ada** (48 GB) | Inference only: the five models run in parallel and fit easily in 48 GB. Ada (sm_89) is supported by every torch build used here. |
| Fallback | 1× A100 40GB / 80GB | Same compatibility, more expensive. |
| Avoid | B200, B300, GB300, RTX PRO 6000 (Blackwell) | Raster2Seq's torch 2.3.1 has no Blackwell kernels. |
| Avoid | Tesla V100 | 16 GB and an old architecture; not worth it. |
| Avoid (not needed) | H100, H200 | They work, but cost far more for no gain here. |
| OS image | **Ubuntu** tab → **Ubuntu 24.04 + CUDA 12.x** | Raster2Seq's CUDA op must be built with nvcc 12 (torch 2.3.1+cu121). The CUDA 13 images make `setup.sh` stop. |
| OS volume | 100 GB | Five torch environments (~30 GB), code and weights. |
| SSH key | your public key | |

## 2. Copy the project (from Git Bash on Windows, project root)

Weights in `models/` are not in git, so copy the folder, without the Windows venvs
and the run outputs:

```bash
tar --exclude=.env --exclude=.venv --exclude=__pycache__ --exclude=work --exclude=facts --exclude=data --exclude=.git -czf - . | ssh root@<VM_IP> "mkdir -p ~/junction && tar -xzf - -C ~/junction"
```

## 3. Install (on the VM, ~15 minutes)

```bash
cd ~/junction && bash deploy/verda/setup.sh
```

It creates `.venv` in `Raster2Seq/`, `Raster-to-Graph/`, `models/2dplan2strct/`, `models/text_ocr/`
(YOLO26-Text + PaddleOCR) and the project root (`pipelines/runners.py` picks them up automatically), builds the two
deformable-attention ops and checks each env sees the GPU.

## 4. Run

```bash
.venv/bin/python main.py plans/ --device cuda          # all models: raster2seq raster2graph layout plan2strct text_ocr
```

`layout` needs the CubiCasa5K weights in `models/model_best_val_loss_var.pkl` (or `CUBICASA_WEIGHTS`). Results
per plan: `facts/<plan>/<model>.campi.json`, `<model>.building_graph.json` and `merged.campi.json`.

Copy the results back from Windows:

```bash
scp -r root@<VM_IP>:~/junction/facts ./facts
```

### Or run from your PC (`--remote`)

After steps 2-3, `main.py` on Windows can do the upload, run and download for you over SSH. Only the input
files are sent; the facts land in `--output` (default `facts/`) as if the run had been local:

Put the VM in `.env` in the project root (template: `.env.example`; `.env` is never copied to the VM):

```
JUNCTION_REMOTE=root@<VM_IP>
JUNCTION_SSH_KEY=~/.ssh/id_ed25519
```

then run as usual:

```bash
python main.py plans/ --device cuda --models raster2seq raster2graph
```

`--remote root@<VM_IP>` overrides `.env` for one run; `--local` ignores `JUNCTION_REMOTE`. Shell variables win over `.env`. The project folder on the
VM defaults to `~/junction` (`--remote-dir` or `JUNCTION_REMOTE_DIR` to change it). SSH must log in with your key
without a password prompt. After code changes, copy the project again (step 2).

Delete the instance in the Verda console when done: a stopped instance still bills for its volume.
