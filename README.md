# Distributed Data Parallel BraTS 2024 Documentation

## Setup

```bash
python -m venv venv
source venv/bin/activate

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download dataset:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='yuuricho/brats_preprocessing',
    repo_type='dataset',
    local_dir='datasets'
)
"
```

Split dataset:

```bash
python split_dataset.py --data_dir datasets/dataset --output data/data_split.json
```

---

## Skenario 1 — Pelatihan Penuh 300 Epoch

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp_unet.py \
    --comment 1gpu_strong_b12 \
    --num_workers 8 \
    --split_file data/data_split.json \
    --data_root datasets/dataset \
    --batch_size 12 \
    --patch_size 128 \
    --pos_ratio 2.0 \
    --neg_ratio 1.0 \
    --epochs 300 \
    --eval_freq 10 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --optim adamw \
    --scheduler warmup_cosine \
    --warmup_epochs 10 \
    --amp \
    --deep_supervision \
    --clip_grad \
    --sw_batch_size 4 \
    --save_model
```

Sesuaikan `--nproc_per_node`, `--batch_size`, dan `--comment` per konfigurasi:

| GPU | `--nproc_per_node` | `--batch_size` | `--comment` |
|-----|--------------------|----------------|-------------|
| 1   | `1`                | `12`           | `1gpu_strong_b12` |
| 2   | `2`                | `6`            | `2gpu_strong_b6`  |
| 4   | `4`                | `3`            | `4gpu_strong_b3`  |

Output: `exps/<comment>_*/` — berisi `best_model.pth` dan `timer.npz`.

---

## Skenario 2 — Scaling Benchmark 50 Epoch

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp_unet.py \
    --comment 1gpu_scaling_b12 \
    --num_workers 8 \
    --split_file data/data_split.json \
    --data_root datasets/dataset \
    --batch_size 12 \
    --patch_size 128 \
    --pos_ratio 2.0 \
    --neg_ratio 1.0 \
    --epochs 50 \
    --eval_freq 51 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --optim adamw \
    --scheduler cosine \
    --amp \
    --deep_supervision \
    --clip_grad \
    --sw_batch_size 2
```

Sesuaikan `--nproc_per_node`, `--batch_size`, dan `--comment` seperti tabel Skenario 1.

Output: `exps/<comment>_*/timer.npz`.

---

## Skenario 3 — Profiling Komponen Iterasi

Script berhenti otomatis setelah 5 iterasi warmup + 50 iterasi aktif.

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp_unet_profiler.py \
    --comment profiler_1gpu \
    --num_workers 8 \
    --split_file data/data_split.json \
    --data_root datasets/dataset \
    --batch_size 12 \
    --patch_size 128 \
    --pos_ratio 2.0 \
    --neg_ratio 1.0 \
    --epochs 50 \
    --eval_freq 51 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --optim adamw \
    --scheduler cosine \
    --amp \
    --deep_supervision \
    --clip_grad
```

Sesuaikan `--nproc_per_node`, `--batch_size`, dan `--comment` seperti tabel Skenario 1.

Output: `profiler_<N>gpu/profiler_timing.npz` dengan keys:
`data_transfer_ms`, `forward_ms`, `zero_grad_ms`, `grad_compute_ms`,
`amp_unscale_ms`, `grad_clip_ms`, `optimizer_ms`, `scaler_update_ms`,
`allreduce_ms`, `barrier_ms`, `total_ms`.
