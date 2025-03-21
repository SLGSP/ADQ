import numpy as np
from .coresetmethod import CoresetMethod

class Adaptive:
    def __init__(self, dataset, args, fraction, seed, balance=False):
        self.dataset = dataset
        self.args = args
        self.fraction = fraction
        self.seed = seed
        self.balance = balance

    def select(self):
        np.random.seed(self.seed)
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)

        # Bin sizes and importances as provided
        bin_sizes = [500, 450, 405, 364, 328, 295, 266, 239, 215, 1938]
        importances = [1, 0.985, 1.031, 1.182, 1.094, 1.176, 1.083, 1.089, 1.011, 1]

        # Check if the total bin size matches the number of samples
        if sum(bin_sizes) != num_samples:
            raise ValueError(f"Total bin size {sum(bin_sizes)} does not match number of samples {num_samples}")

        # Calculate weights
        weights = [size * importance for size, importance in zip(bin_sizes, importances)]
        total_weight = sum(weights)

        # Calculate sampling ratios
        sampling_ratios = [weight / total_weight for weight in weights]

        # Calculate samples per bin
        total_samples = int(self.fraction * num_samples)
        samples_per_bin = [round(ratio * total_samples) for ratio in sampling_ratios]

        # Collect samples
        selected_indices = []
        start_idx = 0
        for bin_size, samples in zip(bin_sizes, samples_per_bin):
            bin_indices = indices[start_idx:start_idx + bin_size]
            if len(bin_indices) == 0:
                raise ValueError(f"Bin indices are empty for bin_size {bin_size} and start_idx {start_idx}")
            selected_indices.extend(np.random.choice(bin_indices, samples, replace=False))
            start_idx += bin_size

        return {"indices": np.array(selected_indices)}