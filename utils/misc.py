import os
import sys
import random
import logging

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------
#               Logging & Monitoring
# ------------------------------------------------

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if logger is None:
            print('\t'.join(entries))
        else:
            logger.info('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
# ------------------------------------------------
#           BraTS 2024 Evaluation Regions
# ------------------------------------------------

IDX_NETC = 0
IDX_SNFH = 1
IDX_ET = 2
IDX_RC = 3

def compute_eval_regions(pred: Tensor) -> Tensor:
    netc = pred[:, IDX_NETC]
    snfh = pred[:, IDX_SNFH]
    et = pred[:, IDX_ET]
    rc = pred[:, IDX_RC]

    tc = torch.clamp(et + netc, 0, 1)
    wt = torch.clamp(et + snfh + netc, 0, 1)

    return torch.stack([netc, snfh, et, rc, tc, wt], dim=1)

# ------------------------------------------------
#              Setup & Initialization
# ------------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def initialization(args):
    seed_everything(args.seed)
    rank = getattr(args, 'rank', 0)

    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        writer = SummaryWriter(args.exp_dir)

        logger = logging.getLogger("brats2024_ddp")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y%m%d %H:%M:%S')
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(fmt)
            fh = logging.FileHandler(os.path.join(args.exp_dir, 'log.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(ch)
            logger.addHandler(fh)

        logger.info("—" * 50)
        logger.info("DISTRIBUTED TRAINING EXPERIMENT".center(50))
        logger.info("—" * 50)
        logger.info(' '.join(sys.argv))
        logger.info(args)
    else:
        writer = None
        logger = logging.getLogger("brats2024_ddp")
        logger.setLevel(logging.WARNING)

    return logger, writer

def is_main_process(args) -> bool:
    return getattr(args, 'rank', 0) == 0