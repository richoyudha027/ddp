import os
import time
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.nvtx as nvtx                          # [NSYS] NVTX untuk annotation timeline
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_train_loader
from models import get_unet
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, initialization, is_main_process)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler

# ------------------------------------------------
#  [NSYS] Konfigurasi Profiling
#  Hanya 20 iterasi yang di-profile — cukup untuk
#  capture distribusi waktu fwd/bwd/AllReduce/opt.
#  Ganti PROFILE_WARMUP + PROFILE_ITERS sesuai kebutuhan.
# ------------------------------------------------
PROFILE_WARMUP = 5          # iterasi warmup (tidak di-profile, biarkan GPU warm)
PROFILE_ITERS  = 20         # iterasi yang aktif di-profile setelah warmup
TOTAL_ITERS    = PROFILE_WARMUP + PROFILE_ITERS


# ------------------------------------------------
#              Deep Supervision Loss
# ------------------------------------------------

def compute_deep_supervision_loss(preds, label, loss_fn, ds_weights=None):
    if not isinstance(preds, list):
        return loss_fn(preds, label)

    num_outputs = len(preds)
    if ds_weights is None:
        ds_weights = [1.0 - (i / (2 * num_outputs)) for i in range(num_outputs)]

    weight_sum = sum(ds_weights[:num_outputs])
    ds_weights = [w / weight_sum for w in ds_weights[:num_outputs]]

    total_bce = 0.0
    total_dsc = 0.0
    for pred, w in zip(preds, ds_weights):
        bce, dsc = loss_fn(pred, label)
        total_bce += w * bce
        total_dsc += w * dsc

    return total_bce, total_dsc


# ------------------------------------------------
#              DDP Setup & Cleanup
# ------------------------------------------------

def setup_multigpu_ddp(args):
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.rank       = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        timeout=timedelta(hours=2),
        device_id=torch.device(f'cuda:{args.local_rank}')
    )
    torch.cuda.set_device(args.local_rank)
    dist.barrier()

    if args.rank == 0:
        print(f"[NSYS] World size = {args.world_size} | Warmup = {PROFILE_WARMUP} iters | Profile = {PROFILE_ITERS} iters")


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------
#          Training Loop (Profiling Mode)
# ------------------------------------------------

def train_profile(args, model, train_loader, train_sampler, loss_fn,
                  optimizer, scaler, logger):
    """
    Training loop yang dimodifikasi khusus untuk Nsight Systems profiling.
    - Hanya jalan 1 epoch dengan TOTAL_ITERS iterasi
    - NVTX range di setiap komponen: data_transfer, forward, backward, optimizer
    - cudaProfilerStart dipanggil setelah warmup selesai
    - cudaProfilerStop dipanggil setelah PROFILE_ITERS selesai
    - Tidak ada validation, checkpointing, atau logging TensorBoard
    """
    model.train()
    train_sampler.set_epoch(0)

    loss_meter = AverageMeter('Loss', ':.4f')

    profiling_active = False

    for i, (image, label, _, _) in enumerate(train_loader):

        if i >= TOTAL_ITERS:
            break

        # ------------------------------------------
        # [NSYS] Warmup selesai → aktifkan profiler
        # ------------------------------------------
        if i == PROFILE_WARMUP and not profiling_active:
            torch.cuda.synchronize()
            torch.cuda.cudaProfilerStart()
            profiling_active = True
            if is_main_process(args):
                print(f"[NSYS] Profiler START at iteration {i}")

        # ------------------------------------------
        # [NSYS] NVTX: iterasi keseluruhan
        # ------------------------------------------
        iter_label = f"iter_{i:03d}"
        nvtx.range_push(iter_label)

        # ------------------------------------------
        # [NSYS] NVTX: data transfer CPU → GPU
        # ------------------------------------------
        nvtx.range_push("data_transfer")
        image = image.cuda(args.local_rank, non_blocking=True)
        label = label.float().cuda(args.local_rank, non_blocking=True)
        torch.cuda.synchronize()        # pastikan transfer selesai sebelum range ditutup
        nvtx.range_pop()                # end: data_transfer

        bsz = image.size(0)

        # ------------------------------------------
        # [NSYS] NVTX: forward pass
        # Termasuk: conv3d stack, instance norm,
        # deep supervision heads, loss computation
        # ------------------------------------------
        nvtx.range_push("forward_pass")
        with autocast('cuda', enabled=args.amp):
            preds = model(image)
            bce_loss, dsc_loss = compute_deep_supervision_loss(preds, label, loss_fn)
            loss = bce_loss + dsc_loss
        torch.cuda.synchronize()
        nvtx.range_pop()                # end: forward_pass

        optimizer.zero_grad()

        # ------------------------------------------
        # [NSYS] NVTX: backward pass
        # Termasuk: gradient computation (paralel)
        # DAN AllReduce NCCL (serial/komunikasi)
        # Keduanya terjadi di dalam .backward()
        # karena DDP hook AllReduce overlap dengan backward
        # ------------------------------------------
        nvtx.range_push("backward_pass")
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()        # tunggu AllReduce selesai sebelum range ditutup
        nvtx.range_pop()                # end: backward_pass

        # ------------------------------------------
        # [NSYS] NVTX: optimizer step
        # Termasuk: grad unscale, grad clip, param update
        # ------------------------------------------
        nvtx.range_push("optimizer_step")
        if args.amp and scaler is not None:
            if args.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        torch.cuda.synchronize()
        nvtx.range_pop()                # end: optimizer_step

        nvtx.range_pop()                # end: iter_XXX

        loss_meter.update(loss.item(), bsz)

        if is_main_process(args):
            print(f"[NSYS] iter {i:03d}/{TOTAL_ITERS-1} | loss={loss_meter.val:.4f} | "
                  f"{'[PROFILING]' if profiling_active else '[WARMUP]'}")

    # ------------------------------------------
    # [NSYS] Stop profiler setelah PROFILE_ITERS selesai
    # ------------------------------------------
    if profiling_active:
        torch.cuda.synchronize()
        torch.cuda.cudaProfilerStop()
        if is_main_process(args):
            print(f"[NSYS] Profiler STOP after {PROFILE_ITERS} profiled iterations")


# ------------------------------------------------
#                     Main
# ------------------------------------------------

def main():
    args = parse_seg_args()
    setup_multigpu_ddp(args)

    torch.backends.cudnn.benchmark = True

    logger, _ = initialization(args)   # writer=None, tidak perlu TensorBoard

    if is_main_process(args):
        print("=" * 60)
        print("NSIGHT SYSTEMS PROFILING MODE".center(60))
        print("=" * 60)
        print(f"World size    : {args.world_size} GPU")
        print(f"Warmup iters  : {PROFILE_WARMUP}")
        print(f"Profile iters : {PROFILE_ITERS}")
        print(f"Total iters   : {TOTAL_ITERS}")
        print(f"AMP           : {args.amp}")
        print("=" * 60)

    split = load_split(args.split_file)
    train_loader, train_sampler = get_train_loader(args, split['train'], distributed=True)

    model = get_unet(args).cuda(args.local_rank)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        gradient_as_bucket_view=True,
        broadcast_buffers=True,
        find_unused_parameters=False,
        static_graph=True,
    )

    optimizer  = get_optimizer(args, model)
    _          = get_scheduler(args, optimizer)   # scheduler tidak dipakai tapi init tetap diperlukan
    loss_fn    = SoftDiceBCEWithLogitsLoss().cuda(args.local_rank)
    scaler     = GradScaler('cuda') if args.amp else None

    if is_main_process(args):
        print("[NSYS] Starting profiling run...")

    train_profile(args, model, train_loader, train_sampler, loss_fn,
                  optimizer, scaler, logger)

    if is_main_process(args):
        print("[NSYS] Profiling complete. Output file: see --output flag in nsys command.")

    cleanup_ddp()


if __name__ == "__main__":
    main()