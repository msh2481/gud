from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import typer
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from rich.progress import track
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset
from utils import set_seed


@typed
def gen_dataset(
    dataset_size: int, n: int, chaos_ratio: float = 0.5
) -> Float[TT, "dataset_size n"]:
    chaotic_part = torch.zeros((dataset_size, n))
    chaotic_part[:, 0] = torch.rand(dataset_size)
    for i in range(1, n):
        chaotic_part[:, i] = (chaotic_part[:, i - 1] + 0.1) % 1.0  # r * x * (1 - x)
    noise = torch.randn(dataset_size, n) + torch.randn(dataset_size, 1)
    return chaotic_part * chaos_ratio + noise * (1 - chaos_ratio)


class DiffusionDataset(Dataset):
    @typed
    def __init__(self, data: Float[TT, "batch seq_len"], schedule: Schedule):
        self.data = data
        self.schedule = schedule

    @typed
    def __getitem__(self, idx: int) -> tuple[
        Float[TT, "seq_len"],  # xt
        Float[TT, "seq_len"],  # signal_var
        Float[TT, "seq_len"],  # signal_ratio
        Float[TT, "seq_len"],  # noise
    ]:
        x0 = self.data[idx]
        signal_var, signal_ratio = self.schedule.sample_signal_var()
        noise = torch.randn_like(x0)
        xt = torch.sqrt(signal_var) * x0 + torch.sqrt(1 - signal_var) * noise
        return xt, signal_var, signal_ratio, noise

    def __len__(self):
        return len(self.data)


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


@typed
def visualize_dataset():
    dataset_size = 3
    seq_len = 30
    speed = 1.0
    denoise_steps = 2
    start_from = 3
    clean_data = gen_dataset(dataset_size, seq_len, 1.0)
    schedule = Schedule.make_rolling(
        seq_len, speed=speed, denoise_steps=denoise_steps, start_from=start_from
    )
    visualize_schedule(schedule)
    dataset = DiffusionDataset(clean_data, schedule)
    for _ in range(3):
        xt, signal_var, signal_ratio, noise = dataset[0]
        plt.plot(xt, label=f"xt")
        plt.plot(signal_var, label=f"signal_var")
        plt.plot(signal_ratio, label=f"signal_ratio")
        plt.plot(noise, label=f"noise")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    visualize_dataset()
