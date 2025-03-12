from math import ceil
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from utils import set_seed

torch.set_default_dtype(torch.float64)


def allclose(a: TT, val: float) -> bool:
    return torch.allclose(a, torch.ones_like(a) * val, atol=1e-6)


class Schedule:
    """
    Continuous time schedule parametrized by log-SNR.

    This class represents a schedule as a function of continuous time t in [0, 1],
    rather than as discrete steps. It provides methods to compute:
    - log_snr(t): log of signal-to-noise ratio at time t
    - signal_var(t): signal variance at time t, computed as sigmoid(log_snr(t))
    - dlogsnr_dt(t): derivative of log_snr with respect to t
    """

    @typed
    def __init__(
        self,
        *,
        N: int,
        w: Float[TT, ""],
        alpha_0: float = 1 - 1e-4,
        alpha_1: float = 1e-2,
    ):
        self.N = N
        self.w = w
        to_snr = lambda x: torch.tensor(x / (1 - x), dtype=torch.float64)
        self.snr_0 = to_snr(alpha_0)
        self.snr_1 = to_snr(alpha_1)
        self.snr_mid = torch.tensor(1.0, dtype=torch.float64)
        assert (
            self.snr_0 > self.snr_mid
        ), f"snr_0 = {self.snr_0} must be greater than snr_mid = {self.snr_mid}"
        assert (
            self.snr_1 < self.snr_mid
        ), f"snr_1 = {self.snr_1} must be less than snr_mid = {self.snr_mid}"

    @typed
    def to(self, device: torch.device | str) -> "Schedule":
        self.w = self.w.to(device)
        self.snr_0 = self.snr_0.to(device)
        self.snr_mid = self.snr_mid.to(device)
        self.snr_1 = self.snr_1.to(device)
        return self

    @typed
    def raw_progress(
        self, times: Float[TT, "T"] | Float[TT, ""]
    ) -> Float[TT, "T N"] | Float[TT, "N"]:
        is_single_time = times.ndim == 0
        if is_single_time:
            times = times.unsqueeze(0)
        times = times.to(self.w.device)
        time_per_token = self.w / (self.N - 1 + self.w)
        v = (self.N - 1) / (1 - time_per_token)
        l = (
            self.N - 1 - torch.arange(self.N, dtype=torch.float64, device=self.w.device)
        ) / v
        r = l + time_per_token
        result = (times[:, None] - l) / (r - l)

        if is_single_time:
            result = result.squeeze(0)

        return result

    @typed
    def snr(
        self, times: Float[TT, "T"] | Float[TT, ""]
    ) -> Float[TT, "T N"] | Float[TT, "N"]:
        is_single_time = times.ndim == 0
        if is_single_time:
            times = times.unsqueeze(0)

        progress = self.raw_progress(times).clamp(0, 1)
        base = torch.tensor(10)
        # a = -1.774113580011895
        # b = -0.22276306695937126
        # snr = torch.pow(base, -2.0 * progress**2) * self.snr_0
        snr = (
            self.snr_0.log() + progress * (self.snr_1.log() - self.snr_0.log())
        ).exp()
        if is_single_time:
            snr = snr.squeeze(0)

        return snr

        # is_single_time = times.ndim == 0
        # if is_single_time:
        #     times = times.unsqueeze(0)

        # progress = self.raw_progress(times).clamp(0, 1)
        # exponential = (
        #     self.snr_0.log() + 2 * progress * (self.snr_mid.log() - self.snr_0.log())
        # ).exp()
        # linear = self.snr_mid + 2 * (progress - 0.5) * (self.snr_1 - self.snr_mid)
        # result = torch.where(progress < 0.5, exponential, linear)

        # if is_single_time:
        #     result = result.squeeze(0)

        # return result
        var = self.signal_var(times)
        return var / (1 - var)

    @typed
    def signal_var(
        self, times: Float[TT, "T"] | Float[TT, ""]
    ) -> Float[TT, "T N"] | Float[TT, "N"]:
        snr = self.snr(times)
        return snr / (snr + 1)

        is_single_time = times.ndim == 0
        if is_single_time:
            times = times.unsqueeze(0)

        progress = self.raw_progress(times).clamp(0, 1)
        base = torch.tensor(10)
        # a = -1.774113580011895
        # b = -0.22276306695937126
        var = torch.pow(base, -2.0 * progress**3) * (self.snr_0 / (1 + self.snr_0))
        if is_single_time:
            var = var.squeeze(0)

        return var

    @typed
    def log_snr(
        self, times: Float[TT, "T"] | Float[TT, ""]
    ) -> Float[TT, "T N"] | Float[TT, "N"]:
        return torch.log(self.snr(times))

    @typed
    def dsnr_dt(
        self, times: Float[TT, "T"] | Float[TT, ""]
    ) -> Float[TT, "T N"] | Float[TT, "N"]:
        is_single_time = times.ndim == 0
        if is_single_time:
            times = times.unsqueeze(0)

        # progress = self.raw_progress(times)
        # is_denoising = ((0 <= progress) & (progress <= 1)).to(dtype=torch.float64)
        # common = is_denoising * 2 * (self.N - 1 + self.w) / self.w
        # exponential = self.snr(times) * (self.snr_mid.log() - self.snr_0.log())
        # linear = self.snr_1 - self.snr_mid
        # result = (common * torch.where(progress < 0.5, exponential, linear)).abs()

        eps = torch.minimum((0.5 - torch.abs(times - 0.5)) / 2, torch.tensor(1e-6))
        snr_minus = self.snr(times - eps)
        snr_plus = self.snr(times + eps)
        result = -((snr_plus - snr_minus) / (2 * eps))

        if is_single_time:
            result = result.squeeze(0)

        return result

    @typed
    def sample_time(self) -> Float[TT, ""]:
        """
        Sample a random time point in [0, 1] using a precomputed low-discrepancy sequence.

        Returns:
            A time point in [0, 1]
        """
        if not hasattr(self, "_precomputed"):
            sequence_length = 2**13
            sobol_engine = torch.quasirandom.SobolEngine(
                dimension=1,
                scramble=True,
            )
            self._precomputed = sobol_engine.draw(sequence_length).squeeze()
            # Add small random offsets to break patterns
            offsets = torch.rand(sequence_length) * 0.01
            self._precomputed = (self._precomputed + offsets) % 1.0
            self._sequence_idx = 0
        sample = self._precomputed[self._sequence_idx].item()
        self._sequence_idx = (self._sequence_idx + 1) % len(self._precomputed)
        return torch.tensor(sample, dtype=torch.float64)


@typed
def visualize_schedule(schedule: Schedule, num_points: int = 1000):
    """
    Visualize a schedule by plotting signal variance and SNR over time.

    Args:
        schedule: The Schedule instance to visualize
        num_points: Number of time points to sample (default: 100)

    Saves the visualization to 'schedule.png'
    """
    times = torch.linspace(0, 1, num_points, dtype=torch.float64)

    # Calculate signal variance and SNR for each time point
    signal_var = schedule.signal_var(times)
    snr = schedule.snr(times)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot signal variance
    im1 = ax1.imshow(
        signal_var.T.cpu().numpy(),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0, 1, 0, schedule.N],
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax1.set_title("Signal Variance Over Time")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Position")
    fig.colorbar(im1, ax=ax1, label="Signal Variance")

    # Plot SNR (log scale)
    log_snr = torch.log10(snr)
    vmin = max(-2, log_snr.min().item())
    vmax = min(4, log_snr.max().item())

    im2 = ax2.imshow(
        log_snr.T.cpu().numpy(),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0, 1, 0, schedule.N],
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    ax2.set_title("Log10 SNR Over Time")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Position")
    fig.colorbar(im2, ax=ax2, label="Log10 SNR")

    # Add schedule parameters as text
    plt.figtext(
        0.5,
        0.01,
        f"N={schedule.N}, w={schedule.w.item():.1f}, "
        f"SNR range: [{schedule.snr_1.item():.2f}, {schedule.snr_0.item():.2f}]",
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("schedule.png", dpi=150)
    plt.close()


def test_dsnr_dt():
    # Numerically compare dsnr_dt(t) with finite-differences
    w = torch.tensor(1.0, dtype=torch.float64)
    schedule = Schedule(N=2, w=w)
    points = 17
    times = (torch.linspace(0, 1, points) + torch.rand(points) * 1e-6).clamp(
        0.0001, 0.9999
    )
    dsnr_dt = schedule.dsnr_dt(times)
    eps = 1e-9
    snr_minus = schedule.snr(times - eps)
    snr_plus = schedule.snr(times + eps)
    dsnr_dt_fd = -((snr_plus - snr_minus) / (2 * eps))
    print(dsnr_dt)
    print(dsnr_dt_fd)
    for i in range(len(dsnr_dt)):
        assert torch.allclose(
            dsnr_dt[i], dsnr_dt_fd[i], rtol=1e-4
        ), f"dsnr_dt[{i}] = {dsnr_dt[i]}, dsnr_dt_fd[{i}] = {dsnr_dt_fd[i]}"


if __name__ == "__main__":
    test_dsnr_dt()
