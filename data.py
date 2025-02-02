from pathlib import Path
from typing import Annotated

import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import nn, Tensor as TT

from utils import set_seed


@typed
def gen_dataset(
    dataset_size: int, n: int, chaos_ratio: float = 0.5
) -> Float[ND, "dataset_size n"]:
    chaotic_part = np.zeros((dataset_size, n))
    chaotic_part[:, 0] = np.random.rand(dataset_size)
    for i in range(1, n):
        chaotic_part[:, i] = (chaotic_part[:, i - 1] + 0.1) % 1.0  # r * x * (1 - x)
    noise = np.random.randn(dataset_size, n) + np.random.randn(dataset_size, 1)
    return chaotic_part * chaos_ratio + noise * (1 - chaos_ratio)


@typed
def visualize_data(
    n_samples: int = 5,
    seq_len: int = 100,
    chaos_ratio: float = 1.0,
    save_path: str | None = None,
    seed: int = 42,
):
    """Generate and visualize sample sequences"""
    set_seed(seed)
    data = gen_dataset(n_samples, seq_len, chaos_ratio=chaos_ratio)

    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.plot(data[i], label=f"Sample {i+1}", alpha=0.7)
    plt.title(f"Generated Sequences (chaos_ratio={chaos_ratio}, seed={seed})")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    visualize_data()
