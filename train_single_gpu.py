import os
import time
import warnings
from copy import deepcopy

from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_train_loader, get_infer_loader
from models import get_unet
import utils.metrics as metrics
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, compute_eval_regions, initialization)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler


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


def train(args, epoch, model, train_loader, loss_fn, optimizer, scheduler,
          scaler, writer, logger):
    model.train()

    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')

    num_samples = 0
    epoch_start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    for i, (image, label, _, _) in enumerate(pbar):
        image = image.cuda()
        label = label.float().cuda()
        bsz = image.size(0)
        num_samples += bsz

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
        bce_meter.update(bce_loss.item(), bsz)
        dsc_meter.update(dsc_loss.item(), bsz)
        loss_meter.update(loss.item(), bsz)

        pbar.set_postfix(
            Loss=f"{loss_meter.val:.4f}",
            BCE=f"{bce_meter.val:.4f}",
            Dice=f"{dsc_meter.val:.4f}",
        )

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - epoch_start
    throughput = num_samples / epoch_time
    _, peak_mem, _ = get_gpu_memory_mb()

    logger.info(
        f"Train [{epoch}]:  "
        f"Loss={loss_meter.avg:.4f}  "
        f"BCE={bce_meter.avg:.4f}  "
        f"Dice={dsc_meter.avg:.4f}  "
        f"LR={optimizer.param_groups[0]['lr']:.6f}  "
        f"Time={format_time(epoch_time)}  "
        f"Throughput={throughput:.2f} samples/s  "
        f"Peak_GPU={peak_mem:.0f}MB"
    )

    if writer is not None:
        writer.add_scalar('train/bce_loss', bce_meter.avg, epoch)
        writer.add_scalar('train/dice_loss', dsc_meter.avg, epoch)
        writer.add_scalar('train/loss', loss_meter.avg, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)
        writer.add_scalar('train/throughput_samples_per_sec', throughput, epoch)
        writer.add_scalar('train/peak_gpu_memory_mb', peak_mem, epoch)
    
    return loss_meter.avg, epoch_time


@torch.no_grad()
def evaluate(args, epoch, model, infer_loader, loss_fn, writer, logger, mode='val'):
    model.eval()

    all_dice = []
    all_hd95 = []
    loss_meter = AverageMeter('Loss', ':.4f')

    eval_start = time.time()

    pbar = tqdm(infer_loader, desc=f"{mode.capitalize()} [{epoch}]", leave=True)
    for i, (image, label, _, _) in enumerate(pbar):
        image = image.cuda()
        label = label.float().cuda()
        bsz = image.size(0)

        seg_map = sliding_window_inference(
            inputs=image,
            predictor=model,
            roi_size=(args.patch_size,)*3,
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
        f"{mode.capitalize()} Summary [{epoch}]:  "
        f"Loss={loss_meter.avg:.4f}  "
        f"Dice_ET={mean_dice['Dice_ET']:.4f} (±{std_dice['Dice_ET_std']:.4f})  "
        f"Dice_TC={mean_dice['Dice_TC']:.4f} (±{std_dice['Dice_TC_std']:.4f})  "
        f"Dice_WT={mean_dice['Dice_WT']:.4f} (±{std_dice['Dice_WT_std']:.4f})  "
        f"HD95_ET={mean_hd95['HD95_ET']:.2f}  "
        f"HD95_TC={mean_hd95['HD95_TC']:.2f}  "
        f"HD95_WT={mean_hd95['HD95_WT']:.2f}  "
        f"Time={format_time(eval_time)}"
    )

    if writer is not None:
        for key, value in infer_metrics.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)
        for key, value in std_dice.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)
        writer.add_scalar(f'{mode}/loss', loss_meter.avg, epoch)
        writer.add_scalar(f'{mode}/eval_time_sec', eval_time, epoch)

    return infer_metrics


def main():
    args = parse_seg_args()
    logger, writer = initialization(args)

    logger.info("—" * 50)
    logger.info("EXPERIMENT CONFIGURATION".center(50))
    logger.info("—" * 50)
    logger.info(f"Model       : {args.unet_arch} ({args.block} block)")
    logger.info(f"Channels    : {args.channels_list}")
    logger.info(f"Patch size  : {args.patch_size}")
    logger.info(f"Batch size  : {args.batch_size}")
    logger.info(f"Epochs      : {args.epochs}")
    logger.info(f"Optimizer   : {args.optim} (lr={args.lr}, wd={args.weight_decay})")
    logger.info(f"Scheduler   : {args.scheduler}")
    logger.info(f"AMP         : {args.amp}")
    logger.info(f"Deep superv.: {args.deep_supervision}")
    logger.info(f"Seed        : {args.seed}")
    if torch.cuda.is_available():
        logger.info(f"GPU         : {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info("—" * 50)

    split = load_split(args.split_file)
    train_loader, _ = get_train_loader(args, split['train'], distributed=False)
    val_loader, _ = get_infer_loader(args, split['val'], distributed=False)
    test_loader, _ = get_infer_loader(args, split['test'], distributed=False)
    
    logger.info("—" * 50)
    logger.info("DATA SPLIT SUMMARY".center(50))
    logger.info("—" * 50)
    logger.info(f"Train      : {len(split['train'])}")
    logger.info(f"Validation : {len(split['val'])}")
    logger.info(f"Test       : {len(split['test'])}")
    logger.info("—" * 50)

    model = get_unet(args).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("—" * 50)
    logger.info("MODEL SUMMARY".center(50))
    logger.info("—" * 50)
    logger.info(f"Total parameters     : {total_params:,}")
    logger.info(f"Trainable parameters : {trainable_params:,}")
    logger.info("—" * 50)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    loss_fn = SoftDiceBCEWithLogitsLoss().cuda()

    scaler = GradScaler('cuda') if args.amp else None
    if args.amp:
        logger.info("Using AMP (Automatic Mixed Precision)")

    if args.weight_path is not None:
        state = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(state['model'])
        logger.info(f"Loaded weights from {args.weight_path}")

    torch.cuda.reset_peak_memory_stats()

    logger.info("Start training.")
    best_epoch = 0
    best_dice = 0.0
    best_model = None
    epoch_times = []
    total_train_start = time.time()

    for epoch in range(args.epochs):
        train_loss, epoch_time = train(
            args, epoch, model, train_loader, loss_fn, optimizer,
            scheduler, scaler, writer, logger
        )
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = evaluate(
                args, epoch, model, val_loader, loss_fn, writer, logger, mode='val'
            )

            mean_dice = np.mean([
                val_metrics['Dice_ET'], 
                val_metrics['Dice_TC'], 
                val_metrics['Dice_WT']
            ])
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch
                best_model = deepcopy(model.state_dict())
                logger.info(f"NEW BEST MODEL at epoch {epoch} with mean Dice {best_dice:.4f}")

        torch.cuda.empty_cache()

    total_train_time = time.time() - total_train_start

    _, peak_mem, _ = get_gpu_memory_mb()
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    
    logger.info("—" * 50)
    logger.info("TRAINING SUMMARY".center(50))
    logger.info("—" * 50)
    logger.info(f"Total training time  : {format_time(total_train_time)}")
    logger.info(f"Avg epoch time       : {format_time(avg_epoch_time)}")
    logger.info(f"Min epoch time       : {format_time(min(epoch_times))}")
    logger.info(f"Max epoch time       : {format_time(max(epoch_times))}")
    logger.info(f"Peak GPU memory      : {peak_mem:.0f} MB")
    logger.info(f"Best epoch           : {best_epoch}")
    logger.info(f"Best mean Dice       : {best_dice:.4f}")
    logger.info(f"Total epochs trained : {args.epochs}")
    logger.info("—" * 50)

    if writer is not None:
        writer.add_hparams(
            {
                'model': args.unet_arch,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'optimizer': args.optim,
                'scheduler': args.scheduler,
                'patch_size': args.patch_size,
                'epochs': args.epochs,
                'amp': args.amp,
                'deep_supervision': args.deep_supervision,
            },
            {
                'hparam/best_dice': best_dice,
                'hparam/best_epoch': best_epoch,
                'hparam/total_train_time_sec': total_train_time,
                'hparam/peak_gpu_memory_mb': peak_mem,
            }
        )

    if best_model is not None:
        logger.info(f"Testing best model from epoch {best_epoch} (mean Dice {best_dice:.4f})")
        model.load_state_dict(best_model)
        test_metrics = evaluate(
            args, best_epoch, model, test_loader, loss_fn, writer, logger, mode='test'
        )

        if args.save_model:
            save_path = os.path.join(args.exp_dir, "best_model.pth")
            torch.save({
                'model': best_model,
                'epoch': best_epoch,
                'args': vars(args),
                'metrics': {
                    'best_dice': best_dice,
                    'test_metrics': test_metrics,
                    'total_train_time': total_train_time,
                    'peak_gpu_memory_mb': peak_mem,
                },
            }, save_path)
            logger.info(f"Saved to {save_path}")
    else:
        logger.warning("No best model found (no evaluation was run). Check eval_freq setting.")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()