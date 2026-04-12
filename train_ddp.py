import os
import time
import socket
import warnings
from copy import deepcopy
from datetime import timedelta

from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_train_loader, get_infer_loader
from models import get_unet
import utils.metrics as metrics
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, compute_eval_regions, initialization, is_main_process)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler


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
#              DDP Setup & Cleanup
# ------------------------------------------------

def setup_multinode_ddp(args):
    master_addr = os.environ.get('MASTER_ADDR', args.master_addr)
    master_port = os.environ.get('MASTER_PORT', args.master_port)
    world_size = int(os.environ.get('WORLD_SIZE', args.nnodes))
    rank = int(os.environ.get('RANK', args.node_rank))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    print(
        f"[Node {rank}/{world_size}] Initializing DDP\n"
        f"Host    = {hostname} ({ip_addr})\n"
        f"Master  = {master_addr}:{master_port}\n"
        f"Backend = {args.dist_backend}"
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=60),
    )

    torch.cuda.set_device(local_rank)

    dist.barrier()
    if rank == 0:
        print(f"[MASTER] All {world_size} nodes connected successfully!")


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------
#              Communication Helpers
# ------------------------------------------------

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def broadcast_dict(data: dict, src=0, device='cuda'):
    keys = sorted(data.keys())
    values = torch.tensor([data[k] for k in keys], dtype=torch.float32, device=device)
    dist.broadcast(values, src=src)
    return {k: v.item() for k, v in zip(keys, values)}


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
#                   Training
# ------------------------------------------------

def train(args, epoch, model, train_loader, train_sampler, loss_fn,
          optimizer, scheduler, scaler, writer, logger):
    model.train()
    train_sampler.set_epoch(epoch)

    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')
    comm_meter = AverageMeter('Comm', ':.4f')

    num_samples = 0
    epoch_start = time.time()
    compute_time_total = 0.0
    comm_time_total = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}]", leave=True) if is_main_process(args) else train_loader
    for i, (image, label, _, _) in enumerate(pbar):
        image = image.cuda(args.local_rank, non_blocking=True)
        label = label.float().cuda(args.local_rank, non_blocking=True)
        bsz = image.size(0)
        num_samples += bsz

        compute_start = time.time()

        with autocast('cuda', enabled=args.amp):
            preds = model(image)
            bce_loss, dsc_loss = compute_deep_supervision_loss(preds, label, loss_fn)
            loss = bce_loss + dsc_loss

        optimizer.zero_grad()
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        torch.cuda.synchronize()
        compute_time = time.time() - compute_start

        comm_start = time.time()
        reduced_loss = reduce_tensor(loss.detach(), args.world_size)
        reduced_bce = reduce_tensor(bce_loss.detach(), args.world_size)
        reduced_dsc = reduce_tensor(dsc_loss.detach(), args.world_size)
        torch.cuda.synchronize()
        comm_time = time.time() - comm_start

        compute_time_total += compute_time
        comm_time_total += comm_time

        bce_meter.update(reduced_bce.item(), bsz)
        dsc_meter.update(reduced_dsc.item(), bsz)
        loss_meter.update(reduced_loss.item(), bsz)
        comm_meter.update(comm_time, 1)

        if is_main_process(args) and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix(
                Loss=f"{loss_meter.val:.4f}",
                BCE=f"{bce_meter.val:.4f}",
                Dice=f"{dsc_meter.val:.4f}",
                Comm=f"{comm_meter.val:.3f}s",
            )

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - epoch_start
    node_throughput = num_samples / epoch_time
    aggregate_throughput = (num_samples * args.world_size) / epoch_time
    total_time = compute_time_total + comm_time_total
    compute_ratio = (compute_time_total / total_time * 100) if total_time > 0 else 0
    comm_ratio = (comm_time_total / total_time * 100) if total_time > 0 else 0

    _, peak_mem, _ = get_gpu_memory_mb()

    if is_main_process(args):
        logger.info(
            f"Train [{epoch}], "
            f"Loss      = {loss_meter.avg:.4f}, "
            f"BCE       = {bce_meter.avg:.4f}, "
            f"Dice      = {dsc_meter.avg:.4f}, "
            f"LR        = {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time      = {format_time(epoch_time)}, "
            f"NodeThrpt = {node_throughput:.2f} samp/s, "
            f"AggrThrpt = {aggregate_throughput:.2f} samp/s, "
            f"Compute   = {compute_ratio:.1f}%, "
            f"Comm      = {comm_ratio:.1f}%, "
            f"CommTime  = {comm_time_total:.2f}s, "
            f"Peak_GPU  = {peak_mem:.0f}MB, "
        )

    if is_main_process(args) and writer is not None:
        writer.add_scalar('train/bce_loss', bce_meter.avg, epoch)
        writer.add_scalar('train/dice_loss', dsc_meter.avg, epoch)
        writer.add_scalar('train/loss', loss_meter.avg, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)
        writer.add_scalar('train/node_throughput_samples_per_sec', node_throughput, epoch)
        writer.add_scalar('train/aggregate_throughput_samples_per_sec', aggregate_throughput, epoch)
        writer.add_scalar('train/compute_ratio_pct', compute_ratio, epoch)
        writer.add_scalar('train/comm_ratio_pct', comm_ratio, epoch)
        writer.add_scalar('train/comm_time_sec', comm_time_total, epoch)
        writer.add_scalar('train/peak_gpu_memory_mb', peak_mem, epoch)

    dist.barrier()

    return loss_meter.avg, epoch_time, comm_time_total


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
    mean_dice = {f'Dice_{r}': all_dice[:, i].mean() for i, r in enumerate(region_names)}
    mean_hd95 = {f'HD95_{r}': all_hd95[:, i].mean() for i, r in enumerate(region_names)}
    std_dice = {f'Dice_{r}_std': all_dice[:, i].std() for i, r in enumerate(region_names)}
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

    setup_multinode_ddp(args)

    global_batch_size = args.batch_size * args.world_size

    logger, writer = initialization(args)

    if is_main_process(args):
        logger.info("—" * 60)
        logger.info("EXPERIMENT CONFIGURATION (MULTI-NODE DDP)".center(60))
        logger.info("—" * 60)
        logger.info(f"Model           : {args.unet_arch} ({args.block} block)")
        logger.info(f"Channels        : {args.channels_list}")
        logger.info(f"Patch size      : {args.patch_size}")
        logger.info(f"Batch size/node : {args.batch_size}")
        logger.info(f"Global batch    : {global_batch_size}")
        logger.info(f"Scaling mode    : Strong (fixed global batch size)")
        logger.info(f"Epochs          : {args.epochs}")
        logger.info(f"Optimizer       : {args.optim} (lr={args.lr}, wd={args.weight_decay})")
        logger.info(f"Scheduler       : {args.scheduler}")
        logger.info(f"AMP             : {args.amp}")
        logger.info(f"Deep superv.    : {args.deep_supervision}")
        logger.info(f"Seed            : {args.seed}")
        logger.info(f"Num nodes       : {args.world_size}")
        logger.info(f"This node rank  : {args.rank}")
        if torch.cuda.is_available():
            logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory      : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        logger.info("—" * 60)

    split = load_split(args.split_file)
    train_loader, train_sampler = get_train_loader(args, split['train'], distributed=True)

    # Val and test loaders only needed on rank 0
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
        logger.info(f"Train                    : {len(split['train'])}")
        logger.info(f"Validation               : {len(split['val'])}")
        logger.info(f"Test                     : {len(split['test'])}")
        logger.info(f"Samples per node / epoch : ~{len(split['train']) // args.world_size}")
        logger.info(f"Iterations per epoch     : ~{len(split['train']) // global_batch_size}")
        logger.info("—" * 60)

    model = get_unet(args).cuda(args.local_rank)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        logger.info("—" * 60)
        logger.info("MODEL SUMMARY".center(60))
        logger.info("—" * 60)
        logger.info(f"Total parameters     : {total_params:,}")
        logger.info(f"Trainable parameters : {trainable_params:,}")
        logger.info("—" * 60)

    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        gradient_as_bucket_view=True,
        broadcast_buffers=True,
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
    epoch_times = []
    comm_times = []
    total_train_start = time.time()

    # Template for non-main nodes to participate in broadcast_dict
    _empty_val_metrics = {
        'Dice_NETC': 0.0, 'Dice_SNFH': 0.0, 'Dice_ET': 0.0,
        'Dice_RC': 0.0, 'Dice_TC': 0.0, 'Dice_WT': 0.0,
        'HD95_NETC': 0.0, 'HD95_SNFH': 0.0, 'HD95_ET': 0.0,
        'HD95_RC': 0.0, 'HD95_TC': 0.0, 'HD95_WT': 0.0,
    }

    for epoch in range(args.epochs):
        train_loss, epoch_time, comm_time = train(
            args, epoch, model, train_loader, train_sampler, loss_fn,
            optimizer, scheduler, scaler, writer, logger
        )
        epoch_times.append(epoch_time)
        comm_times.append(comm_time)

        if (epoch + 1) % args.eval_freq == 0:
            # Only rank 0 runs evaluation; others get empty dict
            val_metrics = evaluate(
                args, epoch, model, val_loader, loss_fn, writer, logger, mode='val'
            )

            # All nodes sync here while rank 0 finishes eval
            dist.barrier()

            # Non-main nodes need matching keys for broadcast
            if not is_main_process(args):
                val_metrics = _empty_val_metrics.copy()

            # Broadcast real metrics from rank 0 to all nodes
            val_metrics = broadcast_dict(val_metrics, src=0, device=f'cuda:{args.local_rank}')

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
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
        avg_comm_time = np.mean(comm_times) if comm_times else 0
        total_comm_time = sum(comm_times)
        comm_pct = (total_comm_time / total_train_time * 100) if total_train_time > 0 else 0

        logger.info("—" * 60)
        logger.info("TRAINING SUMMARY (MULTI-NODE DDP)".center(60))
        logger.info("—" * 60)
        logger.info(f"Total training time      : {format_time(total_train_time)}")
        logger.info(f"Avg epoch time           : {format_time(avg_epoch_time)}")
        logger.info(f"Min epoch time           : {format_time(min(epoch_times))}")
        logger.info(f"Max epoch time           : {format_time(max(epoch_times))}")
        logger.info(f"Total communication time : {format_time(total_comm_time)} ({comm_pct:.1f}% of total)")
        logger.info(f"Avg comm time per epoch  : {avg_comm_time:.2f}s")
        logger.info(f"Peak GPU memory          : {peak_mem:.0f} MB")
        logger.info(f"Best epoch               : {best_epoch}")
        logger.info(f"Best mean Dice           : {best_dice:.4f}")
        logger.info(f"Total epochs trained     : {args.epochs}")
        logger.info(f"Num nodes                : {args.world_size}")
        logger.info(f"Global batch size        : {global_batch_size}")
        logger.info(f"Scaling mode             : Strong")
        logger.info("—" * 60)

        if writer is not None:
            writer.add_hparams(
                {
                    'model': args.unet_arch,
                    'batch_size_per_node': args.batch_size,
                    'global_batch_size': global_batch_size,
                    'scaling_mode': 'strong',
                    'lr': args.lr,
                    'optimizer': args.optim,
                    'scheduler': args.scheduler,
                    'patch_size': args.patch_size,
                    'epochs': args.epochs,
                    'amp': args.amp,
                    'deep_supervision': args.deep_supervision,
                    'num_nodes': args.world_size,
                },
                {
                    'hparam/best_dice': best_dice,
                    'hparam/best_epoch': best_epoch,
                    'hparam/total_train_time_sec': total_train_time,
                    'hparam/total_comm_time_sec': total_comm_time,
                    'hparam/comm_pct': comm_pct,
                    'hparam/peak_gpu_memory_mb': peak_mem,
                }
            )

    # Final test — only rank 0, other nodes wait at cleanup_ddp's barrier
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
                    'total_train_time': total_train_time,
                    'total_comm_time': sum(comm_times),
                    'comm_pct': (sum(comm_times) / total_train_time * 100) if total_train_time > 0 else 0,
                    'peak_gpu_memory_mb': peak_mem,
                    'num_nodes': args.world_size,
                    'global_batch_size': global_batch_size,
                    'scaling_mode': 'strong',
                },
            }, save_path)
            logger.info(f"Saved to {save_path}")
    elif is_main_process(args):
        logger.warning("No best model found (no evaluation was run). Check eval_freq setting.")

    if is_main_process(args) and writer is not None:
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()