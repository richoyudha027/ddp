
import time
import numpy as np
import torch
from torch import Tensor
from medpy.metric import hd95 as hd95_medpy

# ----------------------------------------------
#               Accuracy Metrics
# ----------------------------------------------

def dice(output:Tensor, target:Tensor, eps: float=1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2,3,4)) + eps
    den = output.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


def hd95(output:Tensor, target:Tensor, spacing=None) -> np.ndarray:
    """ output and target should all be boolean tensors"""
    output = output.bool().cpu().numpy()
    target = target.bool().cpu().numpy()
    
    B, C = target.shape[:2]
    hd95 = np.zeros((B, C), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            pred, gt = output[b, c], target[b, c]

            # reward if gt all background, pred all background
            if (not gt.sum()) and (not pred.sum()):
                hd95[b, c] = 0.0
            # penalize if gt all background, pred has foreground
            elif (not gt.sum()) and (pred.sum()):
                hd95[b, c] = 373.1287
            # penalize if gt has forground, but pred has no prediction
            elif (gt.sum()) and (not pred.sum()):
                hd95[b, c] = 373.1287
            else:
                hd95[b, c] = hd95_medpy(pred, gt, voxelspacing=spacing)
    
    return hd95

# ----------------------------------------------
#             Acceleration Metrics
# ----------------------------------------------

class ThroughputMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_time = 0.0

    def update(self, batch_size: int, elapsed_time: float):
        self.total_samples += batch_size
        self.total_time += elapsed_time

    @property
    def throughput(self) -> float:
        if self.total_time > 0:
            return self.total_samples / self.total_time
        else:
            return 0.0

    def __str__(self):
        return f"Throughput: {self.throughput:.2f} samples/sec"


class EpochTimer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_times = []
        self._start_time = None

    def start(self):
        torch.cuda.synchronize()
        self._start_time = time.time()

    def stop(self) -> float:
        torch.cuda.synchronize()
        elapsed = time.time() - self._start_time
        self.epoch_times.append(elapsed)
        self._start_time = None
        return elapsed

    @property
    def last_epoch_time(self) -> float:
        return self.epoch_times[-1] if self.epoch_times else 0.0

    @property
    def total_time(self) -> float:
        return sum(self.epoch_times)

    @property
    def avg_epoch_time(self) -> float:
        if self.epoch_times:
            return np.mean(self.epoch_times)
        return 0.0
    
    def __str__(self):
        return(
            f"EpochTimer: last={self.last_epoch_time:.2f}s, "
            f"avg={self.avg_epoch_time:.2f}s, "
            f"total={self.total_time:.2f}s"
        )

def compute_speedup(time_single_gpu: float, time_multi_gpu: float) -> float:
    """
    Speedup = T1 / Tn
    """
    if time_multi_gpu <= 0:
        return 0.0
    return time_single_gpu / time_multi_gpu

def compute_scalling_efficiency(speedup: float, num_gpus: int) -> float:
    """
    Efficiency = Speedup / N
    """
    if num_gpus <= 0:
        return 0.0
    return speedup / num_gpus