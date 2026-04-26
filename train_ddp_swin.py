import os
import time
import warnings
from copy import deepcopy

from tqdm import tqdm

from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_train_loader, get_infer_loader
import utils.metrics as metrics
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, compute_eval_regions, initialization, is_main_process)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler
from utils.timer import TimerCollector


# ------------------------------------------------
#                   Utilities
# ------------------------------------------------

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return current, peak, reserved
    return 0.0, 0.0, 0.0


# ------------------------------------------------
#              Swin UNETR Builder
# ------------------------------------------------

def get_swin_unetr(args):
    feature_size = getattr(args, 'swin_feature_size', 48)
    use_checkpoint = getattr(args, 'swin_use_checkpoint', True)
    depths = tuple(getattr(args, 'swin_depths', (2, 2, 2, 2)))
    num_heads = tuple(getattr(args, 'swin_num_heads', (3, 6, 12, 24)))
    drop_rate = getattr(args, 'swin_drop_rate', 0.0)
    attn_drop_rate = getattr(args, 'swin_attn_drop_rate', 0.0)
    dropout_path_rate = getattr(args, 'swin_dropout_path_rate', 0.0)
    spatial_dims = 3

    assert args.patch_size % 32 == 0, (
        f"Swin UNETR requires patch_size divisible by 32, got {args.patch_size}. "
        f"Common valid values: 96, 128, 160."
    )

    model = SwinUNETR(
        img_size=(args.patch_size,) * 3,
        in_channels=4,
        out_channels=4,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )
    return model


# ------------------------------------------------
#              DDP Setup & Cleanup
# ------------------------------------------------

def setup_multigpu_ddp(args):
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(
        f"[GPU {args.rank}/{args.world_size}] Initializing DDP | "
        f"local_rank={args.local_rank} | backend={args.dist_backend}"
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        timeout=timedelta(hours=2)
    )

    torch.cuda.set_device(args.local_rank)

    dist.barrier()
    if args.rank == 0:
        print(f"[MASTER] All {args.world_size} GPUs connected successfully!")


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------
#              Loss (single-output only)
# ------------------------------------------------

def compute_loss(preds, label, loss_fn):
    if isinstance(preds, list):
        preds = preds[0]
    bce, dsc = loss_fn(preds, label)
    return bce, dsc


# ------------------------------------------------
#                   Training
# ------------------------------------------------

def train(args, epoch, model, train_loader, train_sampler, loss_fn,
          optimizer, scheduler, scaler, writer, logger, timer):
    model.train()
    train_sampler.set_epoch(epoch)

    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')

    num_samples = 0

    if is_main_process(args):
        timer.start_epoch()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}]", leave=True)
    else:
        pbar = train_loader

    for i, (image, label, _, _) in enumerate(pbar):
        image = image.cuda(args.local_rank, non_blocking=True)
        label = label.float().cuda(args.local_rank, non_blocking=True)
        bsz = image.size(0)
        num_samples += bsz

        if is_main_process(args):
            timer.start_iter()
            timer.start_fwd()

        # Forward
        with autocast('cuda', enabled=args.amp):
            preds = model(image)
            bce_loss, dsc_loss = compute_loss(preds, label, loss_fn)
            loss = bce_loss + dsc_loss

        fwd_time = timer.end_fwd() if is_main_process(args) else 0

        if epoch == 0 and i == 0 and is_main_process(args):
            logger.info(f"Model returns single tensor (Swin UNETR, no deep supervision)")

        # Backward
        if is_main_process(args):
            timer.start_bwd()

        optimizer.zero_grad()
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        bwd_time = timer.end_bwd() if is_main_process(args) else 0

        if is_main_process(args):
            timer.start_opt()

        # Optimizer step
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

        opt_time = timer.end_opt() if is_main_process(args) else 0

        if is_main_process(args):
            timer.end_iter(fwd_time, bwd_time, opt_time)

        bce_meter.update(bce_loss.item(), bsz)
        dsc_meter.update(dsc_loss.item(), bsz)
        loss_meter.update(loss.item(), bsz)

        if is_main_process(args):
            pbar.set_postfix(
                Loss=f"{loss_meter.val:.4f}",
                BCE=f"{bce_meter.val:.4f}",
                Dice=f"{dsc_meter.val:.4f}",
            )

    if scheduler is not None:
        scheduler.step()

    if is_main_process(args):
        epoch_time, _ = timer.end_epoch()
    else:
        epoch_time = 0

    gpu_throughput = num_samples / epoch_time if epoch_time > 0 else 0
    aggregate_throughput = gpu_throughput * args.world_size

    _, peak_mem, _ = get_gpu_memory_mb()

    if is_main_process(args):
        logger.info(
            f"Train [{epoch}], "
            f"Loss      = {loss_meter.avg:.4f}, "
            f"BCE       = {bce_meter.avg:.4f}, "
            f"Dice      = {dsc_meter.avg:.4f}, "
            f"LR        = {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time      = {format_time(epoch_time)}, "
            f"GPU_Thrpt = {gpu_throughput:.2f} samp/s, "
            f"AggrThrpt = {aggregate_throughput:.2f} samp/s, "
            f"Peak_GPU  = {peak_mem:.0f}MB, "
        )

    if is_main_process(args) and writer is not None:
        n = timer.epoch_iter_counts[-1]
        writer.add_scalar('train/bce_loss', bce_meter.avg, epoch)
        writer.add_scalar('train/dice_loss', dsc_meter.avg, epoch)
        writer.add_scalar('train/loss', loss_meter.avg, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)
        writer.add_scalar('train/gpu_throughput_samples_per_sec', gpu_throughput, epoch)
        writer.add_scalar('train/aggregate_throughput_samples_per_sec', aggregate_throughput, epoch)
        writer.add_scalar('train/peak_gpu_memory_mb', peak_mem, epoch)
        writer.add_scalar('train/iter_fwd_median_ms', np.median(timer.iter_fwd_times[-n:]) * 1000, epoch)
        writer.add_scalar('train/iter_bwd_median_ms', np.median(timer.iter_bwd_times[-n:]) * 1000, epoch)
        writer.add_scalar('train/iter_opt_median_ms', np.median(timer.iter_opt_times[-n:]) * 1000, epoch)
        writer.add_scalar('train/iter_total_median_ms', np.median(timer.iter_total_times[-n:]) * 1000, epoch)

    dist.barrier()

    return loss_meter.avg, epoch_time


# ------------------------------------------------
#                  Evaluation
# ------------------------------------------------

@torch.no_grad()
def evaluate(args, epoch, model, infer_loader, loss_fn, writer, logger, mode='val'):
    if not is_main_process(args):
        return {}

    model.eval()

    all_dice = []
    all_hd95 = []
    loss_meter = AverageMeter('Loss', ':.4f')

    eval_start = time.time()

    eval_model = model.module if hasattr(model, 'module') else model

    pbar = tqdm(infer_loader, desc=f"{mode.capitalize()} [{epoch}]", leave=True)
    for i, (image, label, _, _) in enumerate(pbar):
        image = image.cuda(args.local_rank, non_blocking=True)
        label = label.float().cuda(args.local_rank, non_blocking=True)
        bsz = image.size(0)

        seg_map = sliding_window_inference(
            inputs=image,
            predictor=eval_model,
            roi_size=(args.patch_size,) * 3,
            sw_batch_size=args.sw_batch_size,
            overlap=args.patch_overlap,
            mode=args.sliding_window_mode,
        )

        with autocast('cuda', enabled=args.amp):
            bce_loss, dsc_loss = loss_fn(seg_map, label)
            val_loss = bce_loss + dsc_loss
        loss_meter.update(val_loss.item(), bsz)

        seg_map = (torch.sigmoid(seg_map) > 0.5).float()

        seg_map_6 = compute_eval_regions(seg_map)
        label_6 = compute_eval_regions(label)

        dice = metrics.dice(seg_map_6, label_6)
        hd95 = metrics.hd95(seg_map_6, label_6)

        all_dice.append(dice)
        all_hd95.append(hd95)

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(
                Dice_ET=f"{dice[:, 2].mean():.3f}",
                Dice_TC=f"{dice[:, 4].mean():.3f}",
                Dice_WT=f"{dice[:, 5].mean():.3f}",
            )

    eval_time = time.time() - eval_start

    all_dice = np.concatenate(all_dice, axis=0)
    all_hd95 = np.concatenate(all_hd95, axis=0)

    region_names = ['NETC', 'SNFH', 'ET', 'RC', 'TC', 'WT']
    mean_dice = {f'Dice_{r}': float(all_dice[:, i].mean()) for i, r in enumerate(region_names)}
    mean_hd95 = {f'HD95_{r}': float(all_hd95[:, i].mean()) for i, r in enumerate(region_names)}
    std_dice = {f'Dice_{r}_std': float(all_dice[:, i].std()) for i, r in enumerate(region_names)}
    infer_metrics = {**mean_dice, **mean_hd95}

    logger.info(
        f"{mode.capitalize()} Summary [{epoch}], "
        f"Loss    = {loss_meter.avg:.4f}, "
        f"Dice_ET = {mean_dice['Dice_ET']:.4f} ± {std_dice['Dice_ET_std']:.4f}, "
        f"Dice_TC = {mean_dice['Dice_TC']:.4f} ± {std_dice['Dice_TC_std']:.4f}, "
        f"Dice_WT = {mean_dice['Dice_WT']:.4f} ± {std_dice['Dice_WT_std']:.4f}, "
        f"HD95_ET = {mean_hd95['HD95_ET']:.2f}, "
        f"HD95_TC = {mean_hd95['HD95_TC']:.2f}, "
        f"HD95_WT = {mean_hd95['HD95_WT']:.2f}, "
        f"Time    = {format_time(eval_time)}"
    )

    if writer is not None:
        for key, value in infer_metrics.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)
        for key, value in std_dice.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)
        writer.add_scalar(f'{mode}/loss', loss_meter.avg, epoch)
        writer.add_scalar(f'{mode}/eval_time_sec', eval_time, epoch)

    return infer_metrics


# ------------------------------------------------
#                     Main
# ------------------------------------------------

def main():
    args = parse_seg_args()

    if getattr(args, 'deep_supervision', False):
        if int(os.environ.get('RANK', 0)) == 0:
            print("[WARN] Swin UNETR does not support deep supervision. "
                  "Forcing args.deep_supervision = False.")
        args.deep_supervision = False

    setup_multigpu_ddp(args)

    torch.backends.cudnn.benchmark = True

    global_batch_size = args.batch_size * args.world_size

    logger, writer = initialization(args)

    timer = TimerCollector() if is_main_process(args) else None

    if is_main_process(args):
        logger.info("—" * 60)
        logger.info("EXPERIMENT CONFIGURATION MULTI-GPU DDP".center(60))
        logger.info("—" * 60)
        logger.info(f"Model           : Swin UNETR (MONAI)")
        logger.info(f"Feature size    : {getattr(args, 'swin_feature_size', 48)}")
        logger.info(f"Depths          : {tuple(getattr(args, 'swin_depths', (2, 2, 2, 2)))}")
        logger.info(f"Num heads       : {tuple(getattr(args, 'swin_num_heads', (3, 6, 12, 24)))}")
        logger.info(f"Use checkpoint  : {getattr(args, 'swin_use_checkpoint', True)}")
        logger.info(f"Drop rate       : {getattr(args, 'swin_drop_rate', 0.0)}")
        logger.info(f"Attn drop       : {getattr(args, 'swin_attn_drop_rate', 0.0)}")
        logger.info(f"DropPath rate   : {getattr(args, 'swin_dropout_path_rate', 0.0)}")
        logger.info(f"Patch size      : {args.patch_size}")
        logger.info(f"Batch size/GPU  : {args.batch_size}")
        logger.info(f"Global batch    : {global_batch_size}")
        logger.info(f"Scaling mode    : Strong Scaling")
        logger.info(f"Epochs          : {args.epochs}")
        logger.info(f"Optimizer       : {args.optim} (lr={args.lr}, wd={args.weight_decay})")
        logger.info(f"Scheduler       : {args.scheduler}")
        logger.info(f"AMP             : {args.amp}")
        logger.info(f"Deep superv.    : False (not supported by Swin UNETR)")
        logger.info(f"Seed            : {args.seed}")
        logger.info(f"Num GPUs        : {args.world_size}")
        if torch.cuda.is_available():
            logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info("—" * 60)

    split = load_split(args.split_file)
    train_loader, train_sampler = get_train_loader(args, split['train'], distributed=True)

    if is_main_process(args):
        val_loader, _ = get_infer_loader(args, split['val'], distributed=False)
        test_loader, _ = get_infer_loader(args, split['test'], distributed=False)
    else:
        val_loader = None
        test_loader = None

    if is_main_process(args):
        logger.info("—" * 60)
        logger.info("DATA SPLIT SUMMARY".center(60))
        logger.info("—" * 60)
        logger.info(f"Train                   : {len(split['train'])}")
        logger.info(f"Validation              : {len(split['val'])}")
        logger.info(f"Test                    : {len(split['test'])}")
        logger.info(f"Samples per GPU / epoch : ~{len(split['train']) // args.world_size}")
        logger.info(f"Iterations per epoch    : ~{len(split['train']) // global_batch_size}")
        logger.info("—" * 60)

    model = get_swin_unetr(args).cuda(args.local_rank)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        logger.info("—" * 60)
        logger.info("MODEL SUMMARY".center(60))
        logger.info("—" * 60)
        logger.info(f"Architecture         : Swin UNETR (MONAI)")
        logger.info(f"Total parameters     : {total_params:,}")
        logger.info(f"Trainable parameters : {trainable_params:,}")
        logger.info("—" * 60)

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
    scheduler = get_scheduler(args, optimizer)
    loss_fn = SoftDiceBCEWithLogitsLoss().cuda(args.local_rank)

    scaler = GradScaler('cuda') if args.amp else None
    if is_main_process(args) and args.amp:
        logger.info("Using AMP (Automatic Mixed Precision)")

    if args.weight_path is not None:
        state = torch.load(args.weight_path, map_location='cpu')
        model.module.load_state_dict(state['model'])
        if is_main_process(args):
            logger.info(f"Loaded weights from {args.weight_path}")

    torch.cuda.reset_peak_memory_stats()

    if is_main_process(args):
        logger.info("Start training.")

    best_epoch = 0
    best_dice = 0.0
    best_model = None
    total_train_start = time.time()

    for epoch in range(args.epochs):
        train_loss, epoch_time = train(
            args, epoch, model, train_loader, train_sampler, loss_fn,
            optimizer, scheduler, scaler, writer, logger, timer
        )

        if (epoch + 1) % args.eval_freq == 0:
            if is_main_process(args):
                timer.start_val()

            val_metrics = evaluate(
                args, epoch, model, val_loader, loss_fn, writer, logger, mode='val'
            )

            if is_main_process(args):
                timer.end_val(epoch)

            dist.barrier()

            obj_list = [val_metrics] if is_main_process(args) else [None]
            dist.broadcast_object_list(
                obj_list,
                src=0,
                device=torch.device(f'cuda:{args.local_rank}')
            )
            val_metrics = obj_list[0]

            mean_dice = np.mean([
                val_metrics['Dice_ET'],
                val_metrics['Dice_TC'],
                val_metrics['Dice_WT']
            ])

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch
                if is_main_process(args):
                    best_model = deepcopy(model.module.state_dict())
                    logger.info(f"NEW BEST MODEL at epoch [{epoch}] with mean Dice {best_dice:.4f}")

        torch.cuda.empty_cache()

    total_train_time = time.time() - total_train_start

    if is_main_process(args):
        _, peak_mem, _ = get_gpu_memory_mb()

        total_wallclock = total_train_time
        total_train_times = sum(timer.epoch_times)
        total_val_times = sum(timer.val_times)
        train_fraction = (total_train_times / total_wallclock * 100) if total_wallclock > 0 else 0

        avg_epoch_time = np.mean(timer.epoch_times) if timer.epoch_times else 0
        median_epoch_time = np.median(timer.epoch_times) if timer.epoch_times else 0

        logger.info("—" * 60)
        logger.info("TRAINING SUMMARY".center(60))
        logger.info("—" * 60)
        logger.info(f"Total wallclock          : {format_time(total_wallclock)}")
        logger.info(f"Train time               : {format_time(total_train_times)}")
        logger.info(f"Val time                 : {format_time(total_val_times)}")
        logger.info(f"Train fraction           : {train_fraction:.1f}%")
        logger.info(f"Avg epoch time           : {format_time(avg_epoch_time)}")
        logger.info(f"Median epoch time        : {format_time(median_epoch_time)}")
        logger.info(f"Min epoch time           : {format_time(min(timer.epoch_times))}")
        logger.info(f"Max epoch time           : {format_time(max(timer.epoch_times))}")
        logger.info(f"Peak GPU memory          : {peak_mem:.0f} MB")
        logger.info(f"Best epoch               : {best_epoch}")
        logger.info(f"Best mean Dice           : {best_dice:.4f}")
        logger.info(f"Total epochs trained     : {args.epochs}")
        logger.info(f"Num GPUs                 : {args.world_size}")
        logger.info(f"Global batch size        : {global_batch_size}")
        logger.info(f"Scaling mode             : Strong")
        logger.info("—" * 60)

        if writer is not None:
            writer.add_hparams(
                {
                    'model': 'SwinUNETR',
                    'swin_feature_size': getattr(args, 'swin_feature_size', 48),
                    'swin_use_checkpoint': getattr(args, 'swin_use_checkpoint', True),
                    'batch_size_per_gpu': args.batch_size,
                    'global_batch_size': global_batch_size,
                    'scaling_mode': 'strong',
                    'lr': args.lr,
                    'optimizer': args.optim,
                    'scheduler': args.scheduler,
                    'patch_size': args.patch_size,
                    'epochs': args.epochs,
                    'amp': args.amp,
                    'deep_supervision': False,
                    'num_gpus': args.world_size,
                },
                {
                    'hparam/best_dice': best_dice,
                    'hparam/best_epoch': best_epoch,
                    'hparam/total_wallclock': total_wallclock,
                    'hparam/total_train_times': total_train_times,
                    'hparam/total_val_times': total_val_times,
                    'hparam/peak_gpu_memory_mb': peak_mem,
                }
            )

    if is_main_process(args) and best_model is not None:
        logger.info(f"Testing best model from epoch [{best_epoch}] (mean Dice {best_dice:.4f})")
        model.module.load_state_dict(best_model)
        test_metrics = evaluate(
            args, best_epoch, model, test_loader, loss_fn, writer, logger, mode='test'
        )

        if args.save_model:
            _, peak_mem, _ = get_gpu_memory_mb()
            save_path = os.path.join(args.exp_dir, "best_model.pth")
            torch.save({
                'model': best_model,
                'epoch': best_epoch,
                'args': vars(args),
                'metrics': {
                    'best_dice': best_dice,
                    'test_metrics': test_metrics,
                    'total_wallclock': total_wallclock,
                    'total_train_times': total_train_times,
                    'total_val_times': total_val_times,
                    'peak_gpu_memory_mb': peak_mem,
                    'num_gpus': args.world_size,
                    'global_batch_size': global_batch_size,
                    'scaling_mode': 'strong',
                },
            }, save_path)
            logger.info(f"Saved to {save_path}")
    elif is_main_process(args):
        logger.warning("No best model found (no evaluation was run). Check eval_freq setting.")

    if is_main_process(args):
        timer_path = os.path.join(args.exp_dir, "timer.npz")
        timer.save(timer_path)
        logger.info(f"Saved raw timing data to {timer_path}")
        if writer is not None:
            writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()