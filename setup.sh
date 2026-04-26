set -e

apt update && apt install -y tmux

# 1. Create venv.
echo "[1/7] Creating virtual environment."
python -m venv venv
source venv/bin/activate

# 2. Install PyTorch (CUDA 12.8)
echo "[2/7] Installing PyTorch (CUDA 12.8)."
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
echo "[3/7] Installing project dependencies."
pip install -r requirements.txt

# 4. Verify installation
echo ""
echo "[4/7] Verifying installation."
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'GPUs    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i} : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)')
print(f'Status  : OK')
"

# 5. Download dataset
echo "[5/7] Downloading dataset from HuggingFace."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='yuuricho/brats_preprocessing',
    repo_type='dataset',
    local_dir='datasets'
)
"

# 6. Split dataset
echo "[6/7] Splitting dataset."
python split_dataset.py --data_dir "datasets/dataset" --output data/data_split.json

echo "python sample_subset.py \\"
echo "  --split_file data/data_split.json \\"
echo "  --output data/data_split_sample.json"

# 7. Training
echo "[7/7] Training commands."
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#                          SMOKE TEST"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp.py \\"
echo "    --comment smoke_test_b12 \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 12 \\"
echo "    --patch_size 128 \\"
echo "    --pos_ratio 2.0 \\"
echo "    --neg_ratio 1.0 \\"
echo "    --epochs 15 \\"
echo "    --eval_freq 15 \\"
echo "    --lr 3e-4 \\"
echo "    --weight-decay 1e-5 \\"
echo "    --optim adamw \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 5 \\"
echo "    --amp \\"
echo "    --deep_supervision \\"
echo "    --clip_grad \\"
echo "    --sw_batch_size 4 \\"
echo "    --save_model"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 1 GPU (batch 12 per GPU, global=12)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp.py \\"
echo "    --comment 1gpu_strong_b12 \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 12 \\"
echo "    --patch_size 128 \\"
echo "    --pos_ratio 2.0 \\"
echo "    --neg_ratio 1.0 \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --lr 3e-4 \\"
echo "    --weight-decay 1e-5 \\"
echo "    --optim adamw \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --amp \\"
echo "    --deep_supervision \\"
echo "    --clip_grad \\"
echo "    --sw_batch_size 4 \\"
echo "    --save_model"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 2 GPUs (batch 6 per GPU, global=12)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 train_ddp.py \\"
echo "    --comment 2gpu_strong_b6 \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 6 \\"
echo "    --patch_size 128 \\"
echo "    --pos_ratio 2.0 \\"
echo "    --neg_ratio 1.0 \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --lr 3e-4 \\"
echo "    --weight-decay 1e-5 \\"
echo "    --optim adamw \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --amp \\"
echo "    --deep_supervision \\"
echo "    --clip_grad \\"
echo "    --sw_batch_size 4 \\"
echo "    --save_model"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 4 GPUs (batch 3 per GPU, global=12)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 train_ddp.py \\"
echo "    --comment 4gpu_strong_b3 \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 3 \\"
echo "    --patch_size 128 \\"
echo "    --pos_ratio 2.0 \\"
echo "    --neg_ratio 1.0 \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --lr 3e-4 \\"
echo "    --weight-decay 1e-5 \\"
echo "    --optim adamw \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --amp \\"
echo "    --deep_supervision \\"
echo "    --clip_grad \\"
echo "    --sw_batch_size 4 \\"
echo "    --save_model"