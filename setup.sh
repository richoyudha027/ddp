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
print(f'GPU     : {torch.cuda.get_device_name(0)}')
print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
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


# 7. Training
echo "[7/7] Training."
echo "# ————————————————————————————————————————————————"
echo "#                   Single GPU"
echo "# ————————————————————————————————————————————————"
echo "python train_single_gpu.py \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 8 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10"
echo ""
echo "# ————————————————————————————————————————————————"
echo "#                   DDP 2 nodes"
echo "# ————————————————————————————————————————————————"
echo "python train_ddp.py \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 4 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --master_addr <MASTER_IP> \\"
echo "    --nnodes 2 \\"
echo "    --node_rank <0_or_1>"
echo ""
echo "# ————————————————————————————————————————————————"
echo "#                   DDP 4 nodes"
echo "# ————————————————————————————————————————————————"
echo "python train_ddp.py \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 2 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --master_addr <MASTER_IP> \\"
echo "    --nnodes 4 \\"
echo "    --node_rank <0_to_3>"
echo ""
echo "# ————————————————————————————————————————————————"
echo "#                    DDP 8 nodes"
echo "# ————————————————————————————————————————————————"
echo "python train_ddp.py \\"
echo "    --num_workers 8 \\"
echo "    --split_file data/data_split.json \\"
echo "    --data_root brats_data \\"
echo "    --batch_size 1 \\"
echo "    --amp \\"
echo "    --epochs 300 \\"
echo "    --eval_freq 10 \\"
echo "    --save_model \\"
echo "    --scheduler warmup_cosine \\"
echo "    --warmup_epochs 10 \\"
echo "    --master_addr <MASTER_IP> \\"
echo "    --nnodes 8 \\"
echo "    --node_rank <0_to_7>"

echo "tmux new -s train"
