import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from configs import parse_seg_args
from dataset.brats2024 import load_split, get_infer_loader
from models import get_unet
import utils.metrics as metrics
from utils.misc import compute_eval_regions
from monai.inferers import sliding_window_inference


REGION_NAMES = ['NETC', 'SNFH', 'ET', 'RC', 'TC', 'WT']


@torch.no_grad()
def run_inference(model, dataloader, args):
    """
    Sliding window inference dan kumpulkan DSC per kasus.
    Returns: all_dsc — numpy array (N_cases, 6)
    """
    model.eval()
    all_dsc = []
    n = len(dataloader)

    for i, (image, label, _, _) in enumerate(dataloader):
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

        # Expand 4-channel ke 6-channel eval regions
        seg_6   = compute_eval_regions(seg_map)
        label_6 = compute_eval_regions(label)

        # DSC per kasus: returns (B, C) numpy array, B=1 karena infer_batch_size=1
        dsc = metrics.dice(seg_6, label_6)
        all_dsc.append(dsc)

        print(f"  [{i+1}/{n}] " +
              " ".join(f"{r}={dsc[0,j]:.4f}" for j, r in enumerate(REGION_NAMES)))

    return np.concatenate(all_dsc, axis=0)   # (N_cases, 6)


def main():
    # Pakai parent parser dari configs.py agar patch_size, sw_batch_size, dll
    # konsisten dengan config training
    parser = argparse.ArgumentParser(parents=[parse_seg_args()], add_help=False)

    # Args tambahan khusus script ini
    parser.add_argument('--weights_dir', type=str, required=True,
                        help='Folder berisi best_model_Xgpu.pth')
    parser.add_argument('--num_gpus',    type=int, required=True,
                        choices=[1, 2, 4],
                        help='Konfigurasi GPU checkpoint yang akan di-load')
    parser.add_argument('--eval_split',  type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluasi pada val set atau test set (default: val)')

    args = parser.parse_args()

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.weights_dir, f'best_model_{args.num_gpus}gpu.pth')
    assert os.path.exists(ckpt_path), f"Checkpoint tidak ditemukan: {ckpt_path}"

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Override args dengan args dari checkpoint untuk konsistensi penuh
    if 'args' in ckpt:
        saved = ckpt['args']
        if isinstance(saved, dict):
            from argparse import Namespace
            saved = Namespace(**saved)
        for attr in ['patch_size', 'sw_batch_size', 'patch_overlap',
                     'sliding_window_mode', 'channels_list', 'num_classes',
                     'norm', 'block', 'unet_arch', 'deep_supervision',
                     'data_root', 'split_file', 'num_workers', 'infer_batch_size']:
            if hasattr(saved, attr):
                setattr(args, attr, getattr(saved, attr))
        print(f"  Training args loaded from checkpoint.")
        print(f"  Best epoch : {ckpt.get('epoch', 'unknown')}")
    else:
        print("  WARNING: 'args' tidak ditemukan di checkpoint — pakai args CLI.")

    # ── Build model ────────────────────────────────────────────────────────────
    # Nonaktifkan deep supervision untuk inference
    args.deep_supervision = False

    model = get_unet(args).cuda()
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"  Model: {args.unet_arch}, deep_supervision=False")

    # ── Load split dan build dataloader ───────────────────────────────────────
    split = load_split(args.split_file)
    file_paths = split[args.eval_split]   # list of .npz paths untuk val atau test

    # get_infer_loader menerima (args, file_paths, distributed=False)
    # distributed=False karena ini single-GPU inference
    infer_loader, _ = get_infer_loader(args, file_paths, distributed=False)

    print(f"\nDataset    : {args.eval_split} set — {len(file_paths)} kasus")
    print(f"patch_size : {args.patch_size}")
    print(f"sw_batch   : {args.sw_batch_size}")
    print(f"sw_mode    : {args.sliding_window_mode}\n")

    # ── Inference ──────────────────────────────────────────────────────────────
    all_dsc = run_inference(model, infer_loader, args)   # (N_cases, 6)

    # ── Hitung statistik ──────────────────────────────────────────────────────
    dsc_mean = all_dsc.mean(axis=0)   # (6,)
    dsc_std  = all_dsc.std(axis=0)    # (6,)

    print("\n=== DSC Results ===")
    print(f"{'Region':<8} {'Mean':>8} {'Std':>8}")
    print("-" * 26)
    for i, r in enumerate(REGION_NAMES):
        print(f"{r:<8} {dsc_mean[i]:>8.4f} {dsc_std[i]:>8.4f}")

    # ── Simpan ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.weights_dir,
                            f'dsc_per_case_{args.num_gpus}gpu.npz')
    np.savez(
        out_path,
        dsc_per_case  = all_dsc,
        dsc_mean      = dsc_mean,
        dsc_std       = dsc_std,
        region_names  = np.array(REGION_NAMES),
        mode          = np.array(args.eval_split),
        num_gpus      = np.array(args.num_gpus),
    )
    print(f"\nSaved: {out_path}")
    print(f"  dsc_per_case shape : {all_dsc.shape}")
    print(f"  dsc_mean           : {dsc_mean}")
    print(f"  dsc_std            : {dsc_std}")


if __name__ == '__main__':
    main()