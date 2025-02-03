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
    def sample_signal_var(self) -> tuple[Float[TT, "seq_len"], Float[TT, "seq_len"]]:
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
        assert allclose(self.signal_var[0, :], 1), "x[0] must be signal"
        assert allclose(self.signal_ratio[0, :], 1), "x[0] must be not ratioed"
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
            [torch.ones(1, signal_ratio.shape[1]), signal_ratio], dim=0
        )
        noise_level = 1 - signal_ratio
        signal_var = torch.cumprod(signal_ratio, dim=0)
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def from_noise_level(cls, noise_level: Float[TT, "n_steps seq_len"]) -> "Schedule":
        noise_level = torch.cat(
            [torch.zeros(1, noise_level.shape[1]), noise_level], dim=0
        )
        signal_ratio = 1 - noise_level
        signal_var = torch.cumprod(signal_ratio, dim=0)
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def from_signal_var(cls, signal_var: Float[TT, "n_steps seq_len"]) -> "Schedule":
        signal_ratio = torch.ones_like(signal_var)
        signal_ratio[1:] = signal_var[1:] / signal_var[:-1]
        noise_level = 1 - signal_ratio
        noise_var = 1 - signal_var
        return cls(signal_ratio, noise_level, signal_var, noise_var)

    @classmethod
    def make_rolling(
        cls,
        seq_len: int,
        n_steps: int | None = None,
        speed: float | None = None,
        denoise_steps: int = 10,
        start_from: int = 0,
        final_signal_var: float = 1e-2,
    ) -> "Schedule":
        if (n_steps is None) == (speed is None):
            raise ValueError("Exactly one of n_steps or speed must be provided")

        # Compute schedule only for tokens that need denoising
        remaining_len = seq_len - start_from

        if n_steps is None:
            n_steps = int(remaining_len / speed + denoise_steps)
            logger.info(f"Computed n_steps: {n_steps}")
        else:
            assert n_steps > denoise_steps, "n_steps must be greater than denoise_steps"
            speed = remaining_len / (n_steps - denoise_steps)
            logger.info(f"Computed speed: {speed}")

        noise_levels = torch.zeros((n_steps, seq_len))
        # individual_ratios = torch.ones((denoise_steps,)) * final_signal_var ** (
        #     1 / denoise_steps
        # )

        lef, rig = 0, 1
        while rig - lef > 1e-9:
            mid = (lef + rig) / 2
            betas = mid * torch.arange(denoise_steps).float()
            prod_alphas = torch.prod(1 - betas)
            if prod_alphas > final_signal_var:
                lef = mid
            else:
                rig = mid
        mid = (lef + rig) / 2
        logger.info(f"Base noise level: {mid}")
        betas = mid * torch.arange(denoise_steps).float()
        for pos in range(start_from, seq_len):
            start_time = int((seq_len - pos) / speed)
            end_time = start_time + denoise_steps
            noise_levels[start_time:end_time, pos] = betas
        return cls.from_noise_level(noise_levels)


@typed
def test_schedule():
    set_seed(42)
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
    plt.show()


if __name__ == "__main__":
    schedule = Schedule.make_rolling(
        seq_len=10,
        speed=0.5,
        denoise_steps=1,
        final_signal_var=0.01,
        start_from=3,
    )
    visualize_schedule(schedule)
