import hashlib
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import typer
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from rich.progress import track
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        raise NotImplementedError("Losses per clause must be implemented")

    @typed
    def loss(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch"]:
        # Ensure x is on the same device as self.data
        if len(self.data) > 0 and x.device != self.data.device:
            x = x.to(self.data.device)
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


class Zero(DataGenerator):
    @typed
    def random_init(self, batch_size: int) -> TT:
        return torch.zeros((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(self, x: Float[TT, "batch seq_len"]) -> Float[TT, "batch 1"]:
        return x.square().mean(dim=-1, keepdim=True)

    @typed
    def sample(self, batch_size: int, debug: bool = False) -> TT:
        result = torch.zeros((batch_size, self.init_params["length"]))
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
        for i in range(1, length):
            curr_idx = permutation[i]
            prev_idx = permutation[i - 1]
            prediction = 1 - x[:, prev_idx]
            results[:, i - 1] = (prediction - x[:, curr_idx]).square()
        return results

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        length = self.init_params["length"]
        permutation = self.init_params["permutation"]
        result = torch.zeros((batch_size, length))

        first_pos = permutation[0]
        result[:, first_pos] = torch.rand(batch_size)

        for i in range(1, length):
            curr_idx = permutation[i]
            prev_idx = permutation[i - 1]
            result[:, curr_idx] = 1 - result[:, prev_idx]

        self.data = torch.cat([self.data, result], dim=0)
        return result


class MNIST(DataGenerator):
    @typed
    def __init__(self, **params) -> None:
        self.resize_to = 14
        self.side = 10
        assert (
            params["length"] == self.side * self.side
        ), f"MNIST length must be {self.side * self.side}"
        super().__init__(**params)
        transform = transforms.Compose(
            [
                transforms.Resize(self.resize_to),
                transforms.CenterCrop(self.side),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.reshape(-1)),
            ]
        )
        self.mnist_data = datasets.MNIST(
            root="./data",
            train=self.init_params.get("train", True),
            download=True,
            transform=transform,
        )

    @typed
    def random_init(self, batch_size: int) -> Float[TT, "batch seq_len"]:
        return torch.rand((batch_size, self.init_params["length"]))

    @typed
    def losses_per_clause(
        self, x: Float[TT, "batch seq_len"]
    ) -> Float[TT, "batch n_clauses"]:
        prefix = 100
        batch_size = x.shape[0]
        reference_data = torch.stack([self.mnist_data[i][0] for i in range(prefix)])
        # Compute distances between each input and all reference samples
        distances = (x.unsqueeze(1) - reference_data.unsqueeze(0)).square().sum(dim=2)
        # Get minimum distance for each sample in the batch
        min_distances = distances.min(dim=1, keepdim=True).values
        return min_distances

    @typed
    def sample(
        self, batch_size: int, debug: bool = False
    ) -> Float[TT, "batch seq_len"]:
        # For MNIST, we'll just sample random real MNIST images
        indices = torch.randperm(len(self.mnist_data))[:batch_size]
        result = torch.stack([self.mnist_data[i][0] for i in indices])

        self.data = torch.cat([self.data, result], dim=0)
        return result


def test_mnist():
    # Create MNIST data generator
    mnist_gen = MNIST(length=100, tolerance=0.01)

    # Sample some data points
    samples = mnist_gen.sample(batch_size=5)

    # Reshape back to 28x28 and visualize
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        img = samples[i].reshape(10, 10)
        assert 0 <= img.min() <= img.max() <= 1, "Invalid pixel values"
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mnist_samples.png")
    plt.close()

    print(f"MNIST samples saved to mnist_samples.png")
    print(
        f"Sample shape: {samples.shape}, min: {samples.min().item()}, max: {samples.max().item()}"
    )


class DiffusionDataset(Dataset):
    @typed
    def __init__(self, data: Float[TT, "batch seq_len"], schedule: Schedule):
        self.data = data
        self.schedule = schedule

    @typed
    def __getitem__(self, idx: int) -> tuple[
        Float[TT, "seq_len"],  # xt
        Float[TT, "seq_len"],  # signal_var
        Float[TT, "seq_len"],  # dsnr_dt
        Float[TT, "seq_len"],  # x0
        Float[TT, "seq_len"],  # snr
        Float[TT, ""],  # timestep
    ]:
        x0 = self.data[idx]
        timestep = self.schedule.sample_time()
        signal_var = self.schedule.signal_var(timestep)
        dsnr_dt = self.schedule.dsnr_dt(timestep)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(signal_var) * x0 + torch.sqrt(1 - signal_var) * noise
        snr = self.schedule.snr(timestep)
        return xt, signal_var, dsnr_dt, x0, snr, timestep

    @typed
    def sample_distribution(
        self,
        generator: DataGenerator,
        n_samples: int = 100,
        timestep: float | None = None,
        output_path: str = "dataset_samples.png",
    ) -> tuple[Float[TT, "n_samples seq_len"], Float[TT, "n_samples"]]:
        """Generate and visualize multiple samples from the dataset.

        This method:
        1. Generates n_samples from the dataset at a specific timestep
        2. Plots all samples as line plots to visualize the distribution
        3. Creates a histogram of the first element values across all samples

        Args:
            generator: DataGenerator to calculate losses
            n_samples: Number of samples to generate
            timestep: Specific timestep to use (None for random timesteps)
            output_path: Path to save the visualization

        Returns:
            tuple: (samples, losses) - the generated samples and their losses
        """
        # Ensure we have enough samples
        n_samples = min(n_samples, len(self.data))

        # Generate samples
        samples = []
        x0_samples = []

        for i in range(n_samples):
            if timestep is None:
                # Get a clean sample
                x0 = xt = self.data[i]
            else:
                # Get a sample with specific timestep
                x0 = self.data[i]
                t = torch.tensor(timestep, dtype=torch.float64)
                signal_var = self.schedule.signal_var(t)
                noise = torch.randn_like(x0)
                xt = torch.sqrt(signal_var) * x0 + torch.sqrt(1 - signal_var) * noise

            samples.append(xt)
            x0_samples.append(x0)

        # Stack samples
        samples = torch.stack(samples)
        x0_samples = torch.stack(x0_samples)

        # Calculate losses for each sample
        losses = generator.loss(samples)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Main plot for all samples
        for i in range(n_samples):
            ax1.plot(samples[i].cpu().numpy(), alpha=0.2, color="blue", lw=0.1)

        # Add statistics to the plot
        mean_loss = losses.mean().item()
        q50 = losses.quantile(0.50).item()

        title = f"Distribution of {n_samples} Dataset Samples"
        if timestep is not None:
            title += f" (t={timestep:.3f})"
        title += f"\nMean Loss: {mean_loss:.3f}, Median: {q50:.3f}"

        ax1.set_title(title)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)

        i = 3
        first_elements = samples[:, i].cpu().numpy()
        ax2.hist(first_elements, bins=20, alpha=0.7, color="blue")
        ax2.set_title(f"Histogram of {i}th element values")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Dataset sample distribution visualization saved to {output_path}")
        return samples, losses

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    test_mnist()
