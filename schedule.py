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
        alpha_1: float = 1e-4,
    ):
        self.N = N
        self.w = w
        to_snr = lambda x: torch.tensor(x / (1 - x), dtype=torch.float64)
        self.snr_0 = to_snr(alpha_0)
        self.snr_mid = torch.tensor(1.0, dtype=torch.float64)
        self.snr_1 = to_snr(alpha_1)
        assert (
            self.snr_0 > self.snr_mid
        ), f"snr_0 = {self.snr_0} must be greater than snr_mid = {self.snr_mid}"
        assert (
            self.snr_1 < self.snr_mid
        ), f"snr_1 = {self.snr_1} must be less than snr_mid = {self.snr_mid}"
        print(self.snr_0, self.snr_1)

    @typed
    def raw_progress(self, times: Float[TT, "T"]) -> Float[TT, "T N"]:
        time_per_token = self.w / (self.N - 1 + self.w)
        v = (self.N - 1) / (1 - time_per_token)
        l = torch.arange(self.N, dtype=torch.float64) / v
        r = l + time_per_token
        return (times[:, None] - l) / (r - l)

    @typed
    def snr(self, times: Float[TT, "T"]) -> Float[TT, "T N"]:
        progress = self.raw_progress(times).clamp(0, 1)
        exponential = (
            self.snr_0.log() + progress * (self.snr_mid.log() - self.snr_0.log())
        ).exp()
        linear = self.snr_mid + (progress - 0.5) * (self.snr_1 - self.snr_mid)
        return torch.where(progress < 0.5, exponential, linear)

    @typed
    def log_snr(self, times: Float[TT, "T"]) -> Float[TT, "T N"]:
        return torch.log(self.snr(times))

    @typed
    def dsnr_dt(self, times: Float[TT, "T"]) -> Float[TT, "T N"]:
        progress = self.raw_progress(times)
        is_denoising = ((0 <= progress) & (progress <= 1)).to(dtype=torch.float64)
        common = is_denoising * (self.N - 1 + self.w) / self.w
        exponential = self.snr(times) * (self.snr_mid.log() - self.snr_0.log())
        linear = self.snr_1 - self.snr_mid
        return (common * torch.where(progress < 0.5, exponential, linear)).abs()

    @typed
    def signal_var(self, times: Float[TT, "T"]) -> Float[TT, "T N"]:
        snr = self.snr(times)
        return snr / (snr + 1)

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
