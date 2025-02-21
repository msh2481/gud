from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from utils import set_seed


def allclose(a: TT, val: float) -> bool:
    return torch.allclose(a, torch.ones_like(a) * val, atol=1e-6)


VAR_0 = 1 - 1e-4


class Schedule:
    signal_ratio: Float[TT, "n_steps seq_len"]  # aka alpha
    noise_level: Float[TT, "n_steps seq_len"]  # aka beta
    signal_var: Float[TT, "n_steps seq_len"]  # aka prod(alpha)
    noise_var: Float[TT, "n_steps seq_len"]  # aka 1 - signal_var

    @typed
    def __init__(
        self,
        signal_ratio: Float[TT, "n_steps seq_len"],
        noise_level: Float[TT, "n_steps seq_len"],
        signal_var: Float[TT, "n_steps seq_len"],
        noise_var: Float[TT, "n_steps seq_len"],
    ):
        self.signal_ratio = signal_ratio
        self.noise_level = noise_level
        self.signal_var = signal_var
        self.noise_var = noise_var

        self.assert_invariants()

    @typed
    def to(self, device: torch.device | str) -> "Schedule":
        return Schedule(
            self.signal_ratio.to(device),
            self.noise_level.to(device),
            self.signal_var.to(device),
            self.noise_var.to(device),
        )

    @typed
    def sample_signal_var(
        self,
    ) -> tuple[Float[TT, "seq_len"], Float[TT, "seq_len"]]:
        pos = torch.randint(1, self.signal_var.shape[0], ())
        return self.signal_var[pos], self.signal_ratio[pos]

    @typed
    def assert_invariants(self):
        # Shape
        assert (
            self.signal_ratio.shape
            == self.noise_level.shape
            == self.signal_var.shape
            == self.noise_var.shape
        ), "All shapes must match"
        # NaNs
        assert (
            torch.isnan(self.signal_ratio).sum() == 0
        ), "signal_ratio must not contain NaNs"
        assert (
            torch.isnan(self.noise_level).sum() == 0
        ), "noise_level must not contain NaNs"
        assert (
            torch.isnan(self.signal_var).sum() == 0
        ), "signal_var must not contain NaNs"
        assert torch.isnan(self.noise_var).sum() == 0, "noise_var must not contain NaNs"
        # Non-negativity
        assert (self.signal_ratio >= 0).all(), "signal_ratio must be non-negative"
        assert (self.noise_level >= 0).all(), "noise_level must be non-negative"
        assert (self.signal_var >= 0).all(), "signal_var must be non-negative"
        assert (self.noise_var >= 0).all(), "noise_var must be non-negative"
        # Sum to 1
        assert allclose(
            self.signal_ratio + self.noise_level, 1
        ), "signal_ratio + noise_level must be 1"
        assert allclose(
            self.signal_var + self.noise_var, 1
        ), "signal_var + noise_var must be 1"
        # x[0] is not noised
        assert allclose(self.signal_var[0, :], VAR_0), "x[0] must be signal"
        assert allclose(self.signal_ratio[0, :], VAR_0), "x[0] must be not ratioed"
        # signal_var is prod(signal_ratio)
        for i in range(1, self.signal_ratio.shape[0]):
            assert allclose(
                self.signal_var[i, :],
                self.signal_var[i - 1, :] * self.signal_ratio[i, :],
            ), "signal_var must be prod(signal_ratio)"

    @classmethod
    def from_signal_ratio(
        cls, signal_ratio: Float[TT, "n_steps seq_len"]
    ) -> "Schedule":
        signal_ratio = torch.cat(
            [torch.ones(1, signal_ratio.shape[1]) * VAR_0, signal_ratio], dim=0
        )
        noise_level = 1 - signal_ratio
        signal_var = torch.cumprod(signal_ratio, dim=0)
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def from_noise_level(cls, noise_level: Float[TT, "n_steps seq_len"]) -> "Schedule":
        noise_level = torch.cat(
            [torch.ones(1, noise_level.shape[1]) * (1 - VAR_0), noise_level], dim=0
        )
        signal_ratio = 1 - noise_level
        signal_var = torch.cumprod(signal_ratio, dim=0)
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def from_signal_var(cls, signal_var: Float[TT, "n_steps seq_len"]) -> "Schedule":
        signal_ratio = torch.ones_like(signal_var) * VAR_0
        signal_ratio[1:] = signal_var[1:] / signal_var[:-1]
        noise_level = 1 - signal_ratio
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def make_rolling(
        cls,
        seq_len: int,
        *,
        n_steps: int | None = None,
        speed: float | None = None,
        denoise_steps: int | None = None,
        window: float | None = None,
        start_from: int = 0,
        final_signal_var: float = 1e-2,
    ) -> "Schedule":
        """Create a rolling schedule for denoising.

        Two ways to specify the schedule:
        1. n_steps + window
        2. speed + denoise_steps
        """
        # Compute speed and denoise_steps based on input parameters
        if window is not None and n_steps is not None:
            speed = (seq_len + window - 1) / n_steps
            denoise_steps = max(1, int(window / speed + 0.5))
            # n_steps = int(ceil((seq_len - 1 - start_from) / speed)) + denoise_steps
        elif speed is not None and denoise_steps is not None and n_steps is None:
            # 0 step: first step denoising start_from
            # k step: first step denoising seq_len - 1
            # k + denoise_steps - 1 step: last step denoising seq_len - 1
            # k = (seq_len - 1 - start_from) / speed
            n_steps = int(ceil((seq_len - 1 - start_from) / speed)) + denoise_steps
        else:
            raise ValueError(
                "Must provide one of: (n_steps + window), (speed + denoise_steps), or (n_steps + denoise_steps)"
            )

        # Binary search to find optimal noise scale
        def get_final_signal_var(scale: float) -> float:
            betas = scale * torch.linspace(1, 10, denoise_steps).float()
            if betas.max() >= 1:
                return 0.0
            return torch.prod(1 - betas).item()

        lef, rig = 0, 1
        while rig - lef > 1e-9:
            mid = (lef + rig) / 2
            if get_final_signal_var(mid) > final_signal_var:
                lef = mid
            else:
                rig = mid

        # Create noise schedule
        noise_levels = torch.zeros((n_steps, seq_len))
        betas = ((lef + rig) / 2) * torch.linspace(1, 10, denoise_steps).float()
        logger.info(f"betas: {betas}")

        # Apply rolling noise pattern
        for pos in range(start_from, seq_len):
            start_time = max(0, int(0.5 + (seq_len - 1 - pos) / speed))
            end_time = start_time + denoise_steps
            noise_levels[start_time:end_time, pos] = betas[: end_time - start_time]

        return cls.from_noise_level(noise_levels)


@typed
def test_schedule():
    signal_ratio = torch.randn(3, 5).sigmoid()
    schedule = Schedule.from_noise_level(signal_ratio)
    print("Signal ratio:")
    print(schedule.signal_ratio)
    print("Noise level:")
    print(schedule.noise_level)
    print("Signal var:")
    print(schedule.signal_var)
    print("Noise var:")
    print(schedule.noise_var)

    for _ in range(10):
        print(schedule.sample_signal_var())


@typed
def visualize_schedule(schedule: Schedule) -> None:
    """Shows all four matrices in a 2x2 grid"""
    fig, axs = plt.subplots(2, 2)
    cmap = plt.cm.gray
    plt.subplot(2, 2, 1)
    plt.title("Signal ratio")
    plt.imshow(schedule.signal_ratio, cmap=cmap)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title("Noise level")
    plt.imshow(schedule.noise_level, cmap=cmap)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title("Signal var (log10)")
    plt.imshow(torch.log10(schedule.signal_var), cmap=cmap)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.title("Noise var (log10)")
    plt.imshow(torch.log10(schedule.noise_var), cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("schedule.png")
    plt.close()
    plt.cla()
    plt.clf()


def test_elbo():
    def elbo(n: int) -> float:
        schedule = Schedule.make_rolling(
            seq_len=20,
            speed=1e3,
            denoise_steps=n,
            final_signal_var=0.01,
            start_from=0,
        )
        pos = 0
        alpha = schedule.signal_ratio[:, pos]
        beta = schedule.noise_level[:, pos]
        pi = schedule.signal_var[:, pos]
        w = beta[1:] / (alpha[1:] * (1 - pi[1:]))
        return w.sum().item()

    for n in range(1, 100):
        print(n, elbo(n))


if __name__ == "__main__":
    test_elbo()
    exit(0)
    schedule = Schedule.make_rolling(
        seq_len=10,
        speed=0.5,
        denoise_steps=1,
        final_signal_var=0.01,
        start_from=3,
    )
    visualize_schedule(schedule)
