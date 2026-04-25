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
echo "#           SMOKE TEST (1 epoch dulu sebelum full run)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp.py \\"
echo "    --comment smoke_test \\"
echo "    --num_workers 4 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 16 \\"
echo "    --amp \\"
echo "    --epochs 1 \\"
echo "    --eval_freq 1 \\"
echo "    --deep_supervision"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 1 GPU (batch 16 per GPU, global=16)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 train_ddp.py \\"
echo "    --comment 1gpu_strong_b16 \\"
echo "    --num_workers 4 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 16 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --deep_supervision"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 2 GPUs (batch 8 per GPU, global=16)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 train_ddp.py \\"
echo "    --comment 2gpu_strong_b8 \\"
echo "    --num_workers 4 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 8 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --deep_supervision"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 4 GPUs (batch 4 per GPU, global=16)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 train_ddp.py \\"
echo "    --comment 4gpu_strong_b4 \\"
echo "    --num_workers 4 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 4 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --deep_supervision"
echo ""
echo "# ———————————————————————————————————————————————————————————————"
echo "#           DDP 8 GPUs (batch 2 per GPU, global=16)"
echo "# ———————————————————————————————————————————————————————————————"
echo "OMP_NUM_THREADS=4 torchrun --nproc_per_node=8 train_ddp.py \\"
echo "    --comment 8gpu_strong_b2 \\"
echo "    --num_workers 4 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 2 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --deep_supervision"
echo ""
echo "tmux attach -t train"