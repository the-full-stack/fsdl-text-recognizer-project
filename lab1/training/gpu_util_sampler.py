"""Keras callback for measuring GPU utilization."""
import gpustat
import numpy as np
from tensorflow.keras.callbacks import Callback


class GPUUtilizationSampler(Callback):
    """
    Measure GPU utilization at the end of 1% of all batches.
    (The more frequent the measuring, the slower and less accurate this callback becomes.)

    If GPU is not present, report 0 utilization.
    """
    def __init__(self, gpu_ind):
        self.gpu_ind = gpu_ind
        self.samples = []
        super().__init__()

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.samples = []

    def on_batch_end(self, _batch, logs=None):
        if logs is None:
            logs = {}
        random_val = np.random.rand()
        if random_val > 0.99:  # pylint: disable=comparison-with-callable
            try:
                gpu_info = gpustat.GPUStatCollection.new_query()[self.gpu_ind]
                self.samples.append(gpu_info.utilization)
            except Exception:
                self.samples.append(0)
