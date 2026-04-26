#!/bin/bash
echo "=========================================="
echo "          HARDWARE CHECK"
echo "=========================================="
echo
echo "[1] CPU"
echo "  Cores: $(nproc)"
lscpu | grep -E "Model name|Thread|Core\(s\)|Socket" | sed 's/^/  /'
echo
echo "[2] RAM"
free -h | head -2 | sed 's/^/  /'
echo
echo "[3] GPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
echo
echo "[4] GPU Topology"
nvidia-smi topo -m 2>/dev/null | sed 's/^/  /'
echo
echo "[5] Disk Space"
df -h | grep -E "Filesystem|/$|workspace" | sed 's/^/  /'
echo
echo "[6] CUDA"
nvcc --version 2>/dev/null | grep release | sed 's/^/  /' || echo "  nvcc not found in PATH"
echo
echo "[7] Python"
echo "  $(python --version 2>&1)"
echo
echo "[8] Storage Speed Test (1 GB write)"
TEMP=$(mktemp /workspace/dd_test.XXXXXX 2>/dev/null || mktemp /tmp/dd_test.XXXXXX)
dd if=/dev/zero of="$TEMP" bs=1M count=1024 oflag=direct 2>&1 | tail -1 | sed 's/^/  /'
rm "$TEMP"
echo
echo "=========================================="