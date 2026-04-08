import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import monai.transforms as transforms
from monai.transforms import MapTransform

# ----------------------------------------------
#        Custom Transform for BraTS 2024
# ----------------------------------------------

class ConvertToMultiChannelBasedOnBrats2024Classesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            netc = (label == 1).astype(np.float32)
            snfh = (label == 2).astype(np.float32)
            et = (label == 3).astype(np.float32)
            rc = (label == 4).astype(np.float32)
            d[key] = np.stack((netc, snfh, et, rc), axis=0)
        return d
    
# ----------------------------------------------
#              Transform Pipelines
# ----------------------------------------------

def get_brats2024_base_transforms():
    base_transform = [
        ConvertToMultiChannelBasedOnBrats2024Classesd(keys='label')
    ]
    return base_transform

def get_brats2024_train_transforms(args):
    base_transform = get_brats2024_base_transforms()

    data_aug = [
        transforms.RandCropByPosNegLabeld(
            keys=['image', 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=1
        ),

        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),

        transforms.RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(keys=['image'], prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        transforms.RandAdjustContrastd(keys=['image'], prob=0.15, gamma=(0.7, 1.3)),
        
        transforms.EnsureTyped(keys=['image', 'label'])
    ]
    return transforms.Compose(base_transform + data_aug)

def get_brats2024_infer_transforms():
    base_transform = get_brats2024_base_transforms()
    infer_transform = [
        transforms.EnsureTyped(keys=['image', 'label'])
    ]
    return transforms.Compose(base_transform + infer_transform)

# ----------------------------------------------
#                   Dataset
# ----------------------------------------------

class BraTS2024Dataset(Dataset):
    def __init__(self, file_paths: list, mode: str = 'train', transform=None):
        super().__init__()
        assert mode in ['train', 'infer'], f"Unknown mode: {mode}"
        self.file_paths = file_paths
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index: int):
        file_path = self.file_paths[index]
        case_name = os.path.splitext(os.path.basename(file_path))[0]

        data = np.load(file_path)
        image = data['images'].astype(np.float32)
        label = data['seg'].astype(np.float32)

        item = self.transform({'image': image, 'label': label})

        if self.mode == 'train':
            item = item[0]

        return item['image'], item['label'], index, case_name
    
    def __len__(self):
        return len(self.file_paths)
    
# ----------------------------------------------
#                  Data Loader
# ----------------------------------------------

def load_split(split_file: str) -> dict:
    with open(split_file, 'r') as f:
        return json.load(f)
    
def get_train_loader(args, file_paths: list, distributed: bool = False):
    train_transforms = get_brats2024_train_transforms(args)
    train_dataset = BraTS2024Dataset(
        file_paths=file_paths, 
        mode='train', 
        transform=train_transforms
    )

    sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ), sampler

def get_infer_loader(args, file_paths: list, distributed: bool = False):
    infer_transform = get_brats2024_infer_transforms()
    infer_dataset = BraTS2024Dataset(
        file_paths=file_paths, 
        mode='infer', 
        transform=infer_transform
    )

    sampler = DistributedSampler(infer_dataset, shuffle=False) if distributed else None

    return DataLoader(
        infer_dataset,
        batch_size=args.infer_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ), sampler