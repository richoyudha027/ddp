import os
import json
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_train_loader
from models import get_unet
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, initialization, is_main_process)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler

# ------------------------------------------------
#  Konfigurasi Profiling
# ------------------------------------------------
PROFILE_WARMUP = 5    # iterasi warmup (skip, biarkan GPU warm)
PROFILE_ITERS  = 20   # iterasi aktif yang di-profile
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
        print(f"[PROFILER] World size={args.world_size} | Warmup={PROFILE_WARMUP} | Profile={PROFILE_ITERS} iters")


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------
#  Training Loop dengan PyTorch Profiler
# ------------------------------------------------

def train_profile(args, model, train_loader, train_sampler, loss_fn, optimizer, scaler, output_dir):
    model.train()
    train_sampler.set_epoch(0)

    loss_meter = AverageMeter('Loss', ':.4f')

    # Kumpulkan timing manual per komponen per iterasi
    # untuk hitung f (parallel fraction) Amdahl
    timing_records = []

    # PyTorch Profiler hanya aktif di rank 0
    # rank lain tetap jalan normal (DDP butuh semua rank sync)
    prof_context = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=PROFILE_WARMUP,    # skip N iterasi pertama
            warmup=0,
            active=PROFILE_ITERS,  # profile N iterasi aktif
            repeat=1
        ),
        record_shapes=False,
        with_stack=False,
        on_trace_ready=lambda p: p.export_chrome_trace(
            os.path.join(output_dir, f"trace_rank{args.rank}.json")
        )
    ) if is_main_process(args) else None

    if prof_context:
        prof_context.__enter__()

    for i, (image, label, _, _) in enumerate(train_loader):
        if i >= TOTAL_ITERS:
            break

        # ── Data Transfer ──────────────────────────────
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        with record_function("data_transfer"):
            image = image.cuda(args.local_rank, non_blocking=True)
            label = label.float().cuda(args.local_rank, non_blocking=True)
            torch.cuda.synchronize()

        t1.record()
        torch.cuda.synchronize()
        dt_time = t0.elapsed_time(t1)  # ms

        # ── Forward Pass ───────────────────────────────
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t2.record()

        with record_function("forward_pass"):
            with autocast('cuda', enabled=args.amp):
                preds = model(image)
                bce_loss, dsc_loss = compute_deep_supervision_loss(preds, label, loss_fn)
                loss = bce_loss + dsc_loss
            torch.cuda.synchronize()

        t3.record()
        torch.cuda.synchronize()
        fwd_time = t2.elapsed_time(t3)  # ms

        optimizer.zero_grad()

        # ── Backward Pass + AllReduce ──────────────────
        # AllReduce NCCL terjadi di dalam .backward() via DDP hook
        # Selisih backward_time antara 1GPU dan 2GPU = AllReduce overhead
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        t4.record()

        with record_function("backward_pass"):
            if args.amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()  # tunggu AllReduce selesai

        t5.record()
        torch.cuda.synchronize()
        bwd_time = t4.elapsed_time(t5)  # ms

        # ── Optimizer Step ─────────────────────────────
        t6 = torch.cuda.Event(enable_timing=True)
        t7 = torch.cuda.Event(enable_timing=True)
        t6.record()

        with record_function("optimizer_step"):
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

        t7.record()
        torch.cuda.synchronize()
        opt_time = t6.elapsed_time(t7)  # ms

        total_time = dt_time + fwd_time + bwd_time + opt_time

        # Simpan hanya iterasi aktif (setelah warmup)
        if i >= PROFILE_WARMUP:
            timing_records.append({
                "iter": i,
                "data_transfer_ms": round(dt_time, 3),
                "forward_ms":       round(fwd_time, 3),
                "backward_ms":      round(bwd_time, 3),   # termasuk AllReduce
                "optimizer_ms":     round(opt_time, 3),
                "total_ms":         round(total_time, 3),
            })

        loss_meter.update(loss.item(), image.size(0))

        if is_main_process(args):
            phase = "[WARMUP]" if i < PROFILE_WARMUP else "[PROFILE]"
            print(f"{phase} iter {i:03d}/{TOTAL_ITERS-1} | "
                  f"fwd={fwd_time:.1f}ms bwd={bwd_time:.1f}ms opt={opt_time:.1f}ms | "
                  f"loss={loss_meter.val:.4f}")

        if prof_context:
            prof_context.step()

    if prof_context:
        prof_context.__exit__(None, None, None)

    # ── Simpan timing JSON ─────────────────────────────
    if is_main_process(args) and timing_records:
        timing_path = os.path.join(output_dir, "timing_breakdown.json")
        with open(timing_path, 'w') as f:
            json.dump(timing_records, f, indent=2)
        print(f"\n[PROFILER] Timing data saved to {timing_path}")

        # ── Hitung statistik & serial fraction ────────
        fwd_arr = np.array([r["forward_ms"]   for r in timing_records])
        bwd_arr = np.array([r["backward_ms"]  for r in timing_records])
        opt_arr = np.array([r["optimizer_ms"] for r in timing_records])
        dt_arr  = np.array([r["data_transfer_ms"] for r in timing_records])
        tot_arr = np.array([r["total_ms"]     for r in timing_records])

        print("\n" + "=" * 60)
        print("TIMING BREAKDOWN SUMMARY".center(60))
        print("=" * 60)
        print(f"{'Component':<20} {'Mean (ms)':>10} {'Median (ms)':>12} {'% of Total':>12}")
        print("-" * 60)
        for name, arr in [
            ("data_transfer", dt_arr),
            ("forward_pass",  fwd_arr),
            ("backward_pass", bwd_arr),
            ("optimizer_step",opt_arr),
        ]:
            pct = arr.mean() / tot_arr.mean() * 100
            print(f"{name:<20} {arr.mean():>10.1f} {np.median(arr):>12.1f} {pct:>11.1f}%")
        print("-" * 60)
        print(f"{'TOTAL':<20} {tot_arr.mean():>10.1f} {np.median(tot_arr):>12.1f} {'100.0':>11}%")
        print("=" * 60)
        print(f"\n[NOTE] backward_pass mencakup gradient compute + AllReduce NCCL")
        print(f"[NOTE] Untuk mengisolasi AllReduce: bandingkan backward_ms antara 1GPU dan 2GPU")
        print(f"[NOTE] AllReduce overhead ≈ backward_ms(2GPU) - backward_ms(1GPU)")
        print("=" * 60)


# ------------------------------------------------
#                     Main
# ------------------------------------------------

def main():
    args = parse_seg_args()
    setup_multigpu_ddp(args)
    torch.backends.cudnn.benchmark = True

    # Buat output dir untuk profiling
    output_dir = f"/workspace/profiler_{args.world_size}gpu"
    if is_main_process(args):
        os.makedirs(output_dir, exist_ok=True)
        print("=" * 60)
        print("PYTORCH PROFILER MODE".center(60))
        print("=" * 60)
        print(f"World size    : {args.world_size} GPU")
        print(f"Warmup iters  : {PROFILE_WARMUP}")
        print(f"Profile iters : {PROFILE_ITERS}")
        print(f"Output dir    : {output_dir}")
        print("=" * 60)

    logger, _ = initialization(args)

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

    optimizer = get_optimizer(args, model)
    _         = get_scheduler(args, optimizer)
    loss_fn   = SoftDiceBCEWithLogitsLoss().cuda(args.local_rank)
    scaler    = GradScaler('cuda') if args.amp else None

    train_profile(args, model, train_loader, train_sampler,
                  loss_fn, optimizer, scaler, output_dir)

    if is_main_process(args):
        print(f"\n[PROFILER] Done. Files saved to {output_dir}/")
        print(f"  - timing_breakdown.json  : per-iterasi timing fwd/bwd/opt")
        print(f"  - trace_rank0.json       : Chrome trace (buka di chrome://tracing)")

    cleanup_ddp()


if __name__ == "__main__":
    main()