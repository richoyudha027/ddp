import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
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
#  Konfigurasi Profiling
# ------------------------------------------------
PROFILE_WARMUP = 5
PROFILE_ITERS  = 50
TOTAL_ITERS    = PROFILE_WARMUP + PROFILE_ITERS


# ------------------------------------------------
#  CUDA Event Timer Helper
# ------------------------------------------------

def cuda_time(fn):
    """Jalankan fn(), return durasi dalam ms via CUDA event."""
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    fn()
    torch.cuda.synchronize()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1)


# ------------------------------------------------
#  AllReduce Timer Hook
#  Inject langsung ke DDP communication pipeline.
#  Timing aktual AllReduce NCCL per bucket per iterasi.
# ------------------------------------------------

class AllReduceTimer:
    def __init__(self):
        self._iter_buckets     = []
        self.iter_allreduce_ms = []
        self._active           = False

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def flush_iter(self):
        if not self._active or not self._iter_buckets:
            self._iter_buckets = []
            return
        torch.cuda.synchronize()
        total_ms = sum(s.elapsed_time(e) for s, e in self._iter_buckets)
        self.iter_allreduce_ms.append(total_ms)
        self._iter_buckets = []

    def hook_fn(self, process_group, bucket):
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        tensor  = bucket.buffer()
        t_start.record()
        fut = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, async_op=True).get_future()

        def callback(fut):
            t_end.record()
            if self._active:
                self._iter_buckets.append((t_start, t_end))
            result = fut.value()
            # DDP mengharapkan Tensor, bukan list
            return result[0] if isinstance(result, list) else result

        return fut.then(callback)


# ------------------------------------------------
#  Deep Supervision Loss
# ------------------------------------------------

def compute_deep_supervision_loss(preds, label, loss_fn, ds_weights=None):
    if not isinstance(preds, list):
        return loss_fn(preds, label)
    num_outputs = len(preds)
    if ds_weights is None:
        ds_weights = [1.0 - (i / (2 * num_outputs)) for i in range(num_outputs)]
    weight_sum = sum(ds_weights[:num_outputs])
    ds_weights = [w / weight_sum for w in ds_weights[:num_outputs]]
    total_bce, total_dsc = 0.0, 0.0
    for pred, w in zip(preds, ds_weights):
        bce, dsc = loss_fn(pred, label)
        total_bce += w * bce
        total_dsc += w * dsc
    return total_bce, total_dsc


# ------------------------------------------------
#  DDP Setup & Cleanup
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
        print(f"[PROFILER] World size={args.world_size} | "
              f"Warmup={PROFILE_WARMUP} | Active={PROFILE_ITERS} iters")


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------
#  Training Loop — Detail Profiling
# ------------------------------------------------

def train_profile(args, model, train_loader, train_sampler,
                  loss_fn, optimizer, scaler, ar_timer, output_dir):
    model.train()
    train_sampler.set_epoch(0)
    loss_meter = AverageMeter('Loss', ':.4f')
    timing_records = []

    for i, batch in enumerate(train_loader):
        if i >= TOTAL_ITERS:
            break

        # Aktifkan AllReduce timer setelah warmup
        if i == PROFILE_WARMUP:
            torch.cuda.synchronize()
            if ar_timer:
                ar_timer.activate()
            if is_main_process(args):
                print(f"\n[PROFILER] === Profiling START (iter {i}) ===\n")

        is_active = (i >= PROFILE_WARMUP)
        image_cpu, label_cpu = batch[0], batch[1]

        # ── 1. Data Transfer H2D ───────────────────────────────────────
        # non_blocking=True → transfer async, overlap dengan komputasi
        # synchronize() memastikan transfer selesai sebelum komputasi
        # Tipe: PARALEL (independent per rank, tidak ada komunikasi antar GPU)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        image = image_cpu.cuda(args.local_rank, non_blocking=True)
        label = label_cpu.float().cuda(args.local_rank, non_blocking=True)
        torch.cuda.synchronize()
        t1.record()
        torch.cuda.synchronize()
        dt_h2d_ms = t0.elapsed_time(t1)

        bsz = image.size(0)

        # ── 2. Forward Pass ────────────────────────────────────────────
        # Termasuk: seluruh 3D U-Net (encoder conv3d, instance norm,
        # leaky relu, decoder convtranspose3d, deep supervision heads)
        # + loss computation (BCE + Soft Dice)
        # Tipe: PARALEL (setiap rank forward pass independent)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t2.record()
        with autocast('cuda', enabled=args.amp):
            preds = model(image)
            bce_loss, dsc_loss = compute_deep_supervision_loss(preds, label, loss_fn)
            loss = bce_loss + dsc_loss
        torch.cuda.synchronize()
        t3.record()
        torch.cuda.synchronize()
        fwd_ms = t2.elapsed_time(t3)

        # ── 3. zero_grad ───────────────────────────────────────────────
        # Reset gradient buffer sebelum backward
        # Tipe: PARALEL (per rank, tidak ada komunikasi)
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        t4.record()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t5.record()
        torch.cuda.synchronize()
        zero_grad_ms = t4.elapsed_time(t5)

        # ── 4. Backward Pass (gradient compute) ───────────────────────
        # Gradient computation berjalan paralel di setiap rank.
        # DDP hook AllReduce di-overlap dengan backward per bucket —
        # artinya AllReduce bucket i berjalan bersamaan dengan
        # gradient computation layer i-1.
        # backward_ms = grad_compute_ms + allreduce_ms (overlap sebagian)
        # Isolasi AllReduce dilakukan via AllReduceTimer hook.
        # Tipe: CAMPURAN (grad compute=PARALEL, AllReduce=SERIAL)
        t6 = torch.cuda.Event(enable_timing=True)
        t7 = torch.cuda.Event(enable_timing=True)
        t6.record()
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()  # tunggu AllReduce selesai
        t7.record()
        torch.cuda.synchronize()
        bwd_ms = t6.elapsed_time(t7)

        # Finalize AllReduce timing untuk iterasi ini
        if ar_timer:
            ar_timer.flush_iter()

        # ── 5. Gradient Unscale (AMP) ──────────────────────────────────
        # Kembalikan gradient ke FP32 scale sebelum clipping
        # Tipe: PARALEL (per rank, tidak ada komunikasi)
        t8 = torch.cuda.Event(enable_timing=True)
        t9 = torch.cuda.Event(enable_timing=True)
        t8.record()
        if args.amp and scaler is not None and args.clip_grad:
            scaler.unscale_(optimizer)
        torch.cuda.synchronize()
        t9.record()
        torch.cuda.synchronize()
        unscale_ms = t8.elapsed_time(t9)

        # ── 6. Gradient Clipping ───────────────────────────────────────
        # Clip gradient norm untuk stabilitas training
        # Tipe: PARALEL (per rank, gradient sudah di-average via AllReduce)
        t10 = torch.cuda.Event(enable_timing=True)
        t11 = torch.cuda.Event(enable_timing=True)
        t10.record()
        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), 10)
        torch.cuda.synchronize()
        t11.record()
        torch.cuda.synchronize()
        clip_ms = t10.elapsed_time(t11)

        # ── 7. Optimizer Step ──────────────────────────────────────────
        # Update parameter berdasarkan gradient yang sudah di-average
        # Tipe: PARALEL (per rank, identik di semua rank karena gradient sama)
        t12 = torch.cuda.Event(enable_timing=True)
        t13 = torch.cuda.Event(enable_timing=True)
        t12.record()
        if args.amp and scaler is not None:
            scaler.step(optimizer)
        else:
            optimizer.step()
        torch.cuda.synchronize()
        t13.record()
        torch.cuda.synchronize()
        opt_ms = t12.elapsed_time(t13)

        # ── 8. AMP Scaler Update ───────────────────────────────────────
        # Update loss scale factor untuk iterasi berikutnya
        # Tipe: PARALEL (per rank)
        t14 = torch.cuda.Event(enable_timing=True)
        t15 = torch.cuda.Event(enable_timing=True)
        t14.record()
        if args.amp and scaler is not None:
            scaler.update()
        torch.cuda.synchronize()
        t15.record()
        torch.cuda.synchronize()
        scaler_update_ms = t14.elapsed_time(t15)

        # ── 9. dist.barrier() ──────────────────────────────────────────
        # Sinkronisasi global — semua rank harus selesai sebelum lanjut
        # Tipe: SERIAL (blocking, semua rank harus wait)
        t16 = torch.cuda.Event(enable_timing=True)
        t17 = torch.cuda.Event(enable_timing=True)
        t16.record()
        dist.barrier()
        torch.cuda.synchronize()
        t17.record()
        torch.cuda.synchronize()
        barrier_ms = t16.elapsed_time(t17)

        # AllReduce timing untuk iterasi ini
        # Di 1 GPU: paksa 0 karena tidak ada komunikasi antar GPU
        # Nilai ~0.55ms yang muncul di 1 GPU adalah hook overhead, bukan AllReduce
        ar_ms = 0.0
        if args.world_size > 1 and ar_timer and is_active and len(ar_timer.iter_allreduce_ms) > 0:
            ar_ms = ar_timer.iter_allreduce_ms[-1]

        # Gradient compute = backward total - AllReduce
        # (karena AllReduce overlap dengan backward di DDP)
        grad_compute_ms = max(bwd_ms - ar_ms, 0.0)

        total_ms = (dt_h2d_ms + fwd_ms + zero_grad_ms +
                    bwd_ms + unscale_ms + clip_ms +
                    opt_ms + scaler_update_ms + barrier_ms)

        if is_active:
            timing_records.append({
                "iter":              i,
                # ── PARALEL ──────────────────────────────────────
                "data_transfer_ms":  round(dt_h2d_ms,        3),
                "forward_ms":        round(fwd_ms,            3),
                "zero_grad_ms":      round(zero_grad_ms,      3),
                "grad_compute_ms":   round(grad_compute_ms,   3),
                "unscale_ms":        round(unscale_ms,        3),
                "grad_clip_ms":      round(clip_ms,           3),
                "optimizer_ms":      round(opt_ms,            3),
                "scaler_update_ms":  round(scaler_update_ms,  3),
                # ── SERIAL ───────────────────────────────────────
                "allreduce_ms":      round(ar_ms,             3),
                "barrier_ms":        round(barrier_ms,        3),
                # ── RAW ──────────────────────────────────────────
                "backward_raw_ms":   round(bwd_ms,            3),
                "total_ms":          round(total_ms,          3),
            })

        loss_meter.update(loss.item(), bsz)

        if is_main_process(args):
            phase = "[WARMUP] " if not is_active else "[PROFILE]"
            ar_str = f" AR={ar_ms:.1f}ms" if (is_active and ar_timer and args.world_size > 1) else ""
            print(f"{phase} iter {i:03d}/{TOTAL_ITERS-1} | "
                  f"fwd={fwd_ms:.1f}ms "
                  f"bwd={bwd_ms:.1f}ms"
                  f"{ar_str} "
                  f"opt={opt_ms:.1f}ms "
                  f"barrier={barrier_ms:.1f}ms | "
                  f"loss={loss_meter.val:.4f}")

    if ar_timer:
        ar_timer.deactivate()

    # ── Simpan & Cetak Summary ─────────────────────────────────────────
    if is_main_process(args) and timing_records:

        # Arrays per komponen
        dt_arr      = np.array([r["data_transfer_ms"]  for r in timing_records])
        fwd_arr     = np.array([r["forward_ms"]         for r in timing_records])
        zg_arr      = np.array([r["zero_grad_ms"]       for r in timing_records])
        gc_arr      = np.array([r["grad_compute_ms"]    for r in timing_records])
        us_arr      = np.array([r["unscale_ms"]         for r in timing_records])
        clip_arr    = np.array([r["grad_clip_ms"]       for r in timing_records])
        opt_arr     = np.array([r["optimizer_ms"]       for r in timing_records])
        su_arr      = np.array([r["scaler_update_ms"]   for r in timing_records])
        ar_arr      = np.array([r["allreduce_ms"]       for r in timing_records])
        bar_arr     = np.array([r["barrier_ms"]         for r in timing_records])
        tot_arr     = np.array([r["total_ms"]           for r in timing_records])

        # Simpan ke .npz — konsisten dengan timer.npz dari training
        timing_path = os.path.join(output_dir, "profiler_timing.npz")
        np.savez(
            timing_path,
            data_transfer_ms  = dt_arr,
            forward_ms        = fwd_arr,
            zero_grad_ms      = zg_arr,
            grad_compute_ms   = gc_arr,
            amp_unscale_ms    = us_arr,
            grad_clip_ms      = clip_arr,
            optimizer_ms      = opt_arr,
            scaler_update_ms  = su_arr,
            allreduce_ms      = ar_arr,
            barrier_ms        = bar_arr,
            total_ms          = tot_arr,
        )
        print(f"\n[PROFILER] Saved: {timing_path}")

        # Serial fraction = AllReduce + barrier
        serial_ms   = ar_arr + bar_arr
        parallel_ms = tot_arr - serial_ms
        # Gunakan median untuk robustness terhadap outlier infrastruktur
        f_val       = np.median(parallel_ms) / np.median(tot_arr)
        s_val       = np.median(serial_ms)   / np.median(tot_arr)

        print("\n" + "=" * 70)
        print(f"TIMING BREAKDOWN — {args.world_size} GPU".center(70))
        print("=" * 70)
        print(f"{'Component':<25} {'Tipe':<10} {'Mean ms':>9} {'Median ms':>10} {'%':>7}")
        print("-" * 70)

        rows = [
            ("data_transfer_H2D",  "PARALEL", dt_arr),
            ("forward_pass",       "PARALEL", fwd_arr),
            ("zero_grad",          "PARALEL", zg_arr),
            ("grad_compute",       "PARALEL", gc_arr),
            ("amp_unscale",        "PARALEL", us_arr),
            ("grad_clip",          "PARALEL", clip_arr),
            ("optimizer_step",     "PARALEL", opt_arr),
            ("scaler_update",      "PARALEL", su_arr),
            ("allreduce_nccl",     "SERIAL",  ar_arr),
            ("dist_barrier",       "SERIAL",  bar_arr),
        ]

        for name, tipe, arr in rows:
            pct = arr.mean() / tot_arr.mean() * 100
            marker = " ◄" if tipe == "SERIAL" else ""
            print(f"{name:<25} {tipe:<10} {arr.mean():>9.2f} {np.median(arr):>10.2f} {pct:>6.2f}%{marker}")

        print("-" * 70)
        print(f"{'TOTAL':<25} {'':<10} {tot_arr.mean():>9.2f} {np.median(tot_arr):>10.2f} {'100.00':>7}%")
        print("=" * 70)
        print(f"\n[Menggunakan median untuk robustness terhadap outlier]")
        print(f"Parallel fraction (f) = {f_val:.6f}  ({f_val*100:.4f}%)")
        print(f"Serial fraction   (s) = {s_val:.6f}  ({s_val*100:.4f}%)")
        print(f"  - allreduce_nccl    = {np.median(ar_arr):.2f}ms  ({np.median(ar_arr)/np.median(tot_arr)*100:.4f}%)")
        print(f"  - dist_barrier      = {np.median(bar_arr):.2f}ms  ({np.median(bar_arr)/np.median(tot_arr)*100:.4f}%)")

        if s_val > 0:
            s_max = 1.0 / s_val
            print(f"\nAmdahl Prediction (dari {args.world_size} GPU profiling):")
            for n in [2, 4, 8, 16, 32]:
                sn = 1.0 / (s_val + f_val / n)
                eff = sn / n * 100
                print(f"  S({n:>2})  = {sn:.4f}x  (efficiency={eff:.2f}%)")
            print(f"  S(∞)   = {s_max:.2f}x  [theoretical maximum]")
        else:
            print(f"\n[NOTE] AllReduce=0ms dan barrier≈0ms → 1 GPU baseline")
            print(f"[NOTE] Jalankan 2 GPU dan 4 GPU untuk mendapat serial fraction")

        print("=" * 70)


# ------------------------------------------------
#  Main
# ------------------------------------------------

def main():
    args = parse_seg_args()
    setup_multigpu_ddp(args)
    torch.backends.cudnn.benchmark = True

    output_dir = f"/workspace/profiler_{args.world_size}gpu"
    if is_main_process(args):
        os.makedirs(output_dir, exist_ok=True)
        print("=" * 70)
        print("PYTORCH PROFILER — FULL COMPONENT BREAKDOWN".center(70))
        print("=" * 70)
        print(f"World size    : {args.world_size} GPU")
        print(f"Warmup iters  : {PROFILE_WARMUP}")
        print(f"Profile iters : {PROFILE_ITERS}")
        print(f"Output dir    : {output_dir}")
        print("=" * 70)

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

    # Pasang AllReduce hook — aktif di semua konfigurasi
    # Di 1 GPU: hook terpasang tapi tidak pernah dipanggil → ar_ms = 0
    # Di 2/4 GPU: hook capture setiap bucket AllReduce NCCL
    ar_timer = AllReduceTimer()
    model.register_comm_hook(
        state=dist.group.WORLD,
        hook=ar_timer.hook_fn
    )

    optimizer = get_optimizer(args, model)
    _         = get_scheduler(args, optimizer)
    loss_fn   = SoftDiceBCEWithLogitsLoss().cuda(args.local_rank)
    scaler    = GradScaler('cuda') if args.amp else None

    train_profile(
        args, model, train_loader, train_sampler,
        loss_fn, optimizer, scaler,
        ar_timer if is_main_process(args) else None,
        output_dir
    )

    if is_main_process(args):
        print(f"\n[PROFILER] Complete.")

    cleanup_ddp()


if __name__ == "__main__":
    main()