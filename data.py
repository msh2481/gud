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
        self.init_params["class"] = self.__class__.__name__
        # logger.debug(f"Hashing object: {self.init_params}")
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
            # logger.info(f"Data loaded from {path}")
            return result
        except FileNotFoundError:
            logger.warning(f"Data not found in {path}, creating new data")
            return torch.zeros(0, self.init_params["length"])

    @typed
    def append_to_save(self) -> None:
        previous_data = self._load()
        self._save(torch.cat([previous_data, self.data], dim=0))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        raise NotImplementedError("Losses per clause must be implemented")

    @typed
    def loss(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch"]:
        return self.losses_per_clause(x).sum(dim=-1)

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


class WhiteNoise(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> TT:
        return torch.randn((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch 1"]:
        return torch.zeros((x.shape[0], 1))

    @typed
    def sample(self, batch_size: int, debug: bool = False) -> TT:
        result = torch.randn((batch_size, self.init_params["length"]))
        self.data = torch.cat([self.data, result], dim=0)
        return result


class Zigzag(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> TT:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def loss_per_clause(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch 1"]:
        predictions = (x[:, :-1].detach() + 0.1) % 1.0
        targets = x[:, 1:]
        zigzag_loss = (predictions - targets).square().sum(dim=-1)
        start_loss = (x[:, 0] - (x[:, 0] % 1.0).detach()).square().sum(dim=-1)
        result = zigzag_loss + start_loss
        return result[:, None]


class LogisticMap(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        clauses = self.init_params["clauses"]
        if not clauses:
            clauses = self.complicated(x.shape[1])
        results = torch.zeros(x.shape[0], len(clauses))
        for i, (sources, target) in enumerate(clauses):
            mean = x[:, sources].mean(dim=-1)
            prediction = (1 - mean) * mean * 3.993
            results[:, i] = (prediction - x[:, target]).square()
        return results

    @classmethod
    @typed
    def complicated(cls, n: int) -> list[tuple[list[int], int]]:
        assert n >= 9, "n must be at least 9"
        clauses = [
            ([0, 1], n - 3),
            ([2, 3], n - 2),
            ([4, 5], n - 1),
            ([n - 3, n - 2], n - 1),
        ]
        return clauses


class LogisticMapForward(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        # print(
        #     f"x min={x.min()}, max={x.max()} mean={x.mean()} q50={x.quantile(0.5).item()} q5={x.quantile(0.05).item()} q95={x.quantile(0.95).item()}"
        # )
        length = self.init_params["length"]
        results = torch.zeros(x.shape[0], length - 1)
        for i in range(length - 1):
            prediction = (1 - x[:, i]) * x[:, i] * 3.993
            results[:, i] = (prediction - x[:, i + 1]).square()
        return results

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        length = self.init_params["length"]
        result = torch.zeros((batch_size, length))
        # Sample first element randomly
        result[:, 0] = torch.rand(batch_size)
        # Ancestrally sample the rest
        for i in range(length - 1):
            result[:, i + 1] = (1 - result[:, i]) * result[:, i] * 3.993
        self.data = torch.cat([self.data, result], dim=0)
        return result


class LogisticMapBackward(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        length = self.init_params["length"]
        results = torch.zeros(x.shape[0], length - 1)
        for i in range(length - 1):
            prediction = (1 - x[:, i + 1]) * x[:, i + 1] * 3.993
            results[:, i] = (prediction - x[:, i]).square()
        return results

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        length = self.init_params["length"]
        result = torch.zeros((batch_size, length))
        # Sample last element randomly
        result[:, -1] = torch.rand(batch_size)
        # Ancestrally sample backwards
        for i in range(length - 2, -1, -1):
            result[:, i] = (1 - result[:, i + 1]) * result[:, i + 1] * 3.993
        self.data = torch.cat([self.data, result], dim=0)
        return result


class LogisticMapPermutation(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        length = self.init_params["length"]
        permutation = self.init_params["permutation"]
        assert (
            len(permutation) == length
        ), "Permutation length must match sequence length"
        assert sorted(permutation) == list(range(length)), "Invalid permutation"

        results = torch.zeros(x.shape[0], length - 1)
        # For each position (except first in permutation), compute loss based on its predecessor
        for i in range(1, length):
            curr_idx = permutation[i]  # Current position in permutation
            prev_idx = permutation[i - 1]  # Previous position in permutation
            prediction = (1 - x[:, prev_idx]) * x[:, prev_idx] * 3.993
            results[:, i - 1] = (prediction - x[:, curr_idx]).square()
        return results

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        length = self.init_params["length"]
        permutation = self.init_params["permutation"]
        result = torch.zeros((batch_size, length))

        # Initialize first position in permutation with random values
        first_pos = permutation[0]
        result[:, first_pos] = torch.rand(batch_size)

        # Generate remaining positions in permutation order
        for i in range(1, length):
            curr_idx = permutation[i]  # Position to generate
            prev_idx = permutation[i - 1]  # Position to base it on
            result[:, curr_idx] = (
                (1 - result[:, prev_idx]) * result[:, prev_idx] * 3.993
            )

        self.data = torch.cat([self.data, result], dim=0)
        return result


class OneMinusX(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(self, x: TT) -> TT:
        clauses = self.init_params["clauses"]
        results = torch.zeros(x.shape[0], len(clauses))
        for i, (sources, target) in enumerate(clauses):
            mean = x[:, sources].mean(dim=-1)
            prediction = 1 - mean
            results[:, i] = (prediction - x[:, target]).square()
        return results

    @classmethod
    @typed
    def linear(cls, n: int) -> list[tuple[list[int], int]]:
        clauses = []
        for i in range(1, n):
            clauses.append(([i - 1], i))
        return clauses

    @classmethod
    @typed
    def complicated(cls, n: int) -> list[tuple[list[int], int]]:
        assert n >= 9, "n must be at least 9"
        clauses = [
            ([0, 1], n - 3),
            ([2, 3], n - 2),
            ([4, 5], n - 1),
            ([n - 3, n - 2], n - 1),
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
    seq_len: int = 20,
    chaos_ratio: float = 1.0,
    save_path: str | None = None,
    seed: int = 42,
):
    """Generate and visualize sample sequences"""
    # set_seed(seed)
    generator = OneMinusX.load(
        length=seq_len, clauses=OneMinusX.complicated(seq_len), tolerance=1e-3
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
    generator = Zigzag.load(length=seq_len, tolerance=1e-4)
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
