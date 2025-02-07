import hashlib
import json
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


class DataGenerator:
    @typed
    def __init__(self, **params) -> None:
        assert "length" in params, "length is required"
        assert "tolerance" in params, "tolerance is required"
        self.init_params = params
        self.data = torch.zeros(0, params["length"])

    @typed
    def inspect(self) -> None:
        for i in range(len(self)):
            logger.info(f"#{i}: {self.loss(self.data[i:i+1]).item()}")

    @classmethod
    @typed
    def load(cls, **params) -> "DataGenerator":
        instance = cls(**params)
        instance.data = instance._load()
        return instance

    @typed
    def _hash(self) -> str:
        return hashlib.md5(json.dumps(self.init_params).encode()).hexdigest()

    @typed
    def _save(self, data: Float[TT, "batch seq_len"]) -> None:
        path = f"data/{self._hash()}.pt"
        torch.save(data, path)
        logger.info(f"Data saved to {path}")

    @typed
    def _load(self) -> Float[TT, "batch seq_len"]:
        try:
            path = f"data/{self._hash()}.pt"
            result = torch.load(path)
            logger.info(f"Data loaded from {path}")
            return result
        except FileNotFoundError:
            logger.warning(f"Data not found in {path}, creating new data")
            return torch.zeros(0, self.init_params["length"])

    @typed
    def append_to_save(self) -> None:
        previous_data = self._load()
        self._save(torch.cat([previous_data, self.data], dim=0))

    @typed
    def loss(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch"]:
        raise NotImplementedError("Loss function must be implemented")

    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.randn((batch_size, self.init_params["length"]))

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        x = self.random_init(batch_size)
        x = torch.tensor(x.data, requires_grad=True)
        lr = 3e-4
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
        tolerance = self.init_params["tolerance"]
        for it in range(10**9):
            nr = 1e-5 * (2 * lr) ** 0.5
            opt.zero_grad()
            loss = self.loss(x).mean()
            loss.backward()
            opt.step()
            x.data += nr * torch.randn_like(x)
            if debug and it % 10000 == 0:
                logger.debug(f"#{it}: {loss.item()} | {x.detach().numpy()}")
            if loss < tolerance and it % 2000 == 0:
                break
        x = x.detach()
        logger.debug(f"Final (loss={loss.item()}): {x.detach().numpy()}")
        self.data = torch.cat([self.data, x], dim=0)
        if debug:
            logger.debug(
                f"Sampled {batch_size} samples. New data size: {len(self.data)}"
            )
        return x

    @typed
    def __len__(self) -> int:
        return len(self.data)


class Zigzag(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> TT:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def loss(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch"]:
        predictions = (x[:, :-1].detach() + 0.1) % 1.0
        targets = x[:, 1:]
        zigzag_loss = (predictions - targets).square().sum(dim=-1)
        start_loss = (x[:, 0] - (x[:, 0] % 1.0).detach()).square().sum(dim=-1)
        return zigzag_loss + start_loss


class LogisticMap(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def loss(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch"]:
        clauses = self.init_params["clauses"]
        loss = torch.zeros(x.shape[0])
        for sources, target in clauses:
            mean = x[:, sources].mean(dim=-1)
            prediction = 3.993 * mean * (1 - mean)
            loss = loss + (prediction - x[:, target]).square()
        return loss

    @classmethod
    @typed
    def linear(cls, n: int) -> list[tuple[list[int], int]]:
        clauses = []
        for i in range(1, n):
            clauses.append(([i - 1], i))
        return clauses

    @classmethod
    @typed
    def complicated(cls) -> list[tuple[list[int], int]]:
        clauses = [
            ([0, 1], 6),
            ([2, 3], 7),
            ([4, 5], 8),
            ([6, 7], 8),
        ]
        return clauses


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
    n_samples: int = 200,
    seq_len: int = 10,
    chaos_ratio: float = 1.0,
    save_path: str | None = None,
    seed: int = 42,
):
    """Generate and visualize sample sequences"""
    # set_seed(seed)
    generator = LogisticMap.load(
        length=9, clauses=LogisticMap.complicated(), tolerance=1e-3
    )
    generator.inspect()
    while len(generator) < n_samples:
        generator.sample(10, debug=True)
    generator.append_to_save()
    data = generator.data[:n_samples]

    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.plot(data[i], alpha=0.2, lw=1, color="k")
    plt.title(f"Generated Sequences (chaos_ratio={chaos_ratio}, seed={seed})")

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
    generator = Zigzag.load(length=seq_len, tolerance=1e-3)
    while len(generator) < dataset_size:
        generator.sample(10, debug=True)
    clean_data = generator.data[:dataset_size]
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
    visualize_data()
    # visualize_dataset()
