import time
import numpy as np
import torch


class TimerCollector:

    def __init__(self):
        self.iter_fwd_times = []
        self.iter_bwd_times = []
        self.iter_opt_times = []
        self.iter_total_times = []

        self.epoch_times = []
        self.epoch_iter_counts = []

        self.val_times = []
        self.val_epochs = []

        self._iter_fwd_start = None
        self._iter_bwd_start = None
        self._iter_opt_start = None
        self._iter_total_start = None
        self._epoch_start = None
        self._val_start = None

        self._current_epoch_iters = {'fwd': [], 'bwd': [], 'opt': [], 'total': []}


    def start_iter(self):
        torch.cuda.synchronize()
        self._iter_total_start = time.time()

    def start_fwd(self):
        self._iter_fwd_start = time.time()

    def end_fwd(self):
        torch.cuda.synchronize()
        return time.time() - self._iter_fwd_start

    def start_bwd(self):
        self._iter_bwd_start = time.time()

    def end_bwd(self):
        torch.cuda.synchronize()
        return time.time() - self._iter_bwd_start

    def start_opt(self):
        self._iter_opt_start = time.time()

    def end_opt(self):
        torch.cuda.synchronize()
        return time.time() - self._iter_opt_start

    def end_iter(self, fwd_time, bwd_time, opt_time):
        iter_time = time.time() - self._iter_total_start
        self._current_epoch_iters['fwd'].append(fwd_time)
        self._current_epoch_iters['bwd'].append(bwd_time)
        self._current_epoch_iters['opt'].append(opt_time)
        self._current_epoch_iters['total'].append(iter_time)


    def start_epoch(self):
        self._epoch_start = time.time()
        self._current_epoch_iters = {'fwd': [], 'bwd': [], 'opt': [], 'total': []}

    def end_epoch(self):
        epoch_time = time.time() - self._epoch_start

        n_iters = len(self._current_epoch_iters['total'])
        self.iter_fwd_times.extend(self._current_epoch_iters['fwd'])
        self.iter_bwd_times.extend(self._current_epoch_iters['bwd'])
        self.iter_opt_times.extend(self._current_epoch_iters['opt'])
        self.iter_total_times.extend(self._current_epoch_iters['total'])

        self.epoch_times.append(epoch_time)
        self.epoch_iter_counts.append(n_iters)

        return epoch_time, dict(self._current_epoch_iters)


    def start_val(self):
        self._val_start = time.time()

    def end_val(self, epoch):
        val_time = time.time() - self._val_start
        self.val_times.append(val_time)
        self.val_epochs.append(epoch)
        return val_time


    def save(self, path):
        np.savez(
            path,
            iter_fwd_times=np.array(self.iter_fwd_times),
            iter_bwd_times=np.array(self.iter_bwd_times),
            iter_opt_times=np.array(self.iter_opt_times),
            iter_total_times=np.array(self.iter_total_times),
            epoch_times=np.array(self.epoch_times),
            epoch_iter_counts=np.array(self.epoch_iter_counts),
            val_times=np.array(self.val_times),
            val_epochs=np.array(self.val_epochs),
        )

    @staticmethod
    def load(path):
        return np.load(path)

    @staticmethod
    def compute_stats(times, warmup=0):
        stable = np.array(times[warmup:])
        return {
            'median_ms': float(np.median(stable) * 1000),
            'mean_ms': float(np.mean(stable) * 1000),
            'std_ms': float(np.std(stable) * 1000),
            'p95_ms': float(np.percentile(stable, 95) * 1000),
            'p99_ms': float(np.percentile(stable, 99) * 1000),
        }