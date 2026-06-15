"""
extract_hd95_std.py
-------------------
Inference-only script untuk ekstrak HD95 per-kasus dari best_model.pth
dan menyimpan array per-kasus + SD ke hd95_per_case.npz.

Usage:
    python extract_hd95_std.py \
        --exp_dir /ddp/exps/YOUR_EXP_FOLDER \
        --data_root /path/to/dataset \
        --split_file /ddp/data/data_split.json

Output:
    <exp_dir>/hd95_per_case.npz dengan keys:
        - hd95_per_case : array (N, 6) — HD95 per kasus per region
        - hd95_mean     : array (6,)   — mean per region
        - hd95_std      : array (6,)   — std per region
        - region_names  : list         — ['NETC','SNFH','ET','RC','TC','WT']
        - mode          : str          — 'val' atau 'test'
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from models.unet3d import get_unet3d
from datasets.brats import BraTSDataset
from utils.metrics import MetricsCalculator


REGION_NAMES = ['NETC', 'SNFH', 'ET', 'RC', 'TC', 'WT']


def compute_eval_regions(mask):
    NETC = mask[:, 0:1]
    SNFH = mask[:, 1:2]
    ET   = mask[:, 2:3]
    RC   = mask[:, 3:4]
    TC   = NETC + ET          # Tumor Core = NETC + ET
    WT   = NETC + SNFH + ET   # Whole Tumor = NETC + SNFH + ET
    return torch.cat([NETC, SNFH, ET, RC, TC, WT], dim=1)


def hd95_per_case(output, target, spacing=(1.0, 1.0, 1.0)):
    from medpy.metric.binary import hd95 as hd95_medpy

    B, C = target.shape[:2]
    result = np.zeros((B, C), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            pred = output[b, c]
            gt   = target[b, c]
            if (not gt.sum()) and (not pred.sum()):
                result[b, c] = 0.0
            elif (not gt.sum()) and pred.sum():
                result[b, c] = 373.1287
            elif gt.sum() and (not pred.sum()):
                result[b, c] = 373.1287
            else:
                result[b, c] = hd95_medpy(pred, gt, voxelspacing=spacing)
    return result


@torch.no_grad()
def run_inference(model, dataloader, args):
    model.eval()
    all_hd95 = []

    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            image = batch[0]
            label = batch[1]
        else:
            image = batch['image']
            label = batch['label']

        image = image.cuda()
        label = label.float().cuda()

        seg_map = sliding_window_inference(
            inputs=image,
            roi_size=(args.patch_size,) * 3,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.patch_overlap,
            mode=args.sliding_window_mode,
        )

        seg_map = (torch.sigmoid(seg_map) > 0.5).float()

        seg_6   = compute_eval_regions(seg_map)
        label_6 = compute_eval_regions(label)

        seg_np   = seg_6.bool().cpu().numpy()
        label_np = label_6.bool().cpu().numpy()

        hd = hd95_per_case(seg_np, label_np)
        all_hd95.append(hd)

        print(f"  [{i+1}/{len(dataloader)}] done")

    return np.concatenate(all_hd95, axis=0) 


def main():
    parser = argparse.ArgumentParser(description='Extract HD95 per-case for SD calculation')
    parser.add_argument('--exp_dir',    type=str, required=True,
                        help='Path ke folder eksperimen, berisi best_model.pth')
    parser.add_argument('--data_root',  type=str, required=True,
                        help='Path ke root dataset')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Path ke data_split.json')
    parser.add_argument('--mode',       type=str, default='val',
                        choices=['val', 'test'],
                        help='Jalankan pada validation set atau test set (default: val)')
    parser.add_argument('--patch_size',         type=int,   default=128)
    parser.add_argument('--sw_batch_size',      type=int,   default=4)
    parser.add_argument('--patch_overlap',      type=float, default=0.5)
    parser.add_argument('--sliding_window_mode',type=str,   default='constant')
    parser.add_argument('--num_workers',        type=int,   default=4)

    parser.add_argument('--channels_list', type=int, nargs='+',
                        default=[32, 64, 128, 256, 320, 320])
    parser.add_argument('--num_classes',   type=int, default=4)
    parser.add_argument('--norm',          type=str, default='instance')
    parser.add_argument('--block',         type=str, default='plain')

    args = parser.parse_args()

    ckpt_path = os.path.join(args.exp_dir, 'best_model.pth')
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'args' in ckpt:
        train_args = ckpt['args']
        args.patch_size          = getattr(train_args, 'patch_size',          args.patch_size)
        args.sw_batch_size       = getattr(train_args, 'sw_batch_size',       args.sw_batch_size)
        args.patch_overlap       = getattr(train_args, 'patch_overlap',       args.patch_overlap)
        args.sliding_window_mode = getattr(train_args, 'sliding_window_mode', args.sliding_window_mode)
        args.channels_list       = getattr(train_args, 'channels_list',       args.channels_list)
        args.num_classes         = getattr(train_args, 'num_classes',         args.num_classes)
        args.norm                = getattr(train_args, 'norm',                args.norm)
        args.block               = getattr(train_args, 'block',               args.block)
        print("  Loaded training args from checkpoint.")

    model = get_unet3d(args).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Model loaded. Best epoch: {ckpt.get('epoch', 'unknown')}")

    with open(args.split_file) as f:
        split = json.load(f)

    split_key = args.mode  # 'val' atau 'test'
    file_list = split[split_key]

    dataset = BraTSDataset(
        data_root=args.data_root,
        file_list=file_list,
        patch_size=args.patch_size,
        mode='val',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\nRunning inference on {args.mode} set: {len(dataset)} cases")
    print(f"  patch_size:          {args.patch_size}")
    print(f"  sw_batch_size:       {args.sw_batch_size}")
    print(f"  sliding_window_mode: {args.sliding_window_mode}\n")

    all_hd95 = run_inference(model, dataloader, args)   # (N, 6)

    hd95_mean = all_hd95.mean(axis=0)   # (6,)
    hd95_std  = all_hd95.std(axis=0)    # (6,)

    print("\n=== HD95 Results ===")
    print(f"{'Region':<8} {'Mean':>8} {'Std':>8}")
    print("-" * 26)
    for i, r in enumerate(REGION_NAMES):
        print(f"{r:<8} {hd95_mean[i]:>8.2f} {hd95_std[i]:>8.2f}")

    out_path = os.path.join(args.exp_dir, 'hd95_per_case.npz')
    np.savez(
        out_path,
        hd95_per_case=all_hd95,
        hd95_mean=hd95_mean,
        hd95_std=hd95_std,
        region_names=np.array(REGION_NAMES),
        mode=np.array(args.mode),
    )
    print(f"\nSaved to: {out_path}")
    print(f"  hd95_per_case : {all_hd95.shape}  (N_cases x 6 regions)")
    print(f"  hd95_mean     : {hd95_mean.shape}")
    print(f"  hd95_std      : {hd95_std.shape}")


if __name__ == '__main__':
    main()