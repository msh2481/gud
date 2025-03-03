import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch

from beartype import beartype as typed
from jaxtyping import Float
from schedule import Schedule
from torch import Tensor as TT


@typed
def save(n_steps: int) -> None:
    """
    Build a Schedule with n_steps and window=1e3, then plot and save signal_var,
    log signal_var, SNR, and log SNR for all steps but only the first token.

    Args:
        n_steps: Number of steps in the schedule
    """
    # Create output directories if they don't exist
    os.makedirs("schedule_plots", exist_ok=True)
    os.makedirs("schedule_tensors", exist_ok=True)

    # Build schedule with the specified parameters
    # Using a large seq_len (e.g., 20) and window=1e3
    seq_len = 20
    window = 1e3

    # Create the schedule
    schedule = Schedule.make_rolling(
        seq_len=seq_len,
        n_steps=n_steps,
        window=window,
        start_from=0,
    )

    # Extract data for the first token only
    signal_var = schedule.signal_var[:, 0].cpu().numpy()
    noise_var = schedule.noise_var[:, 0].cpu().numpy()

    # Calculate SNR = signal_var / noise_var
    snr = signal_var / noise_var

    # Calculate log values
    log_signal_var = np.log10(signal_var)
    log_snr = np.log10(snr)

    # Save tensors as numpy arrays
    np.save(f"schedule_tensors/signal_var_n{n_steps}.npy", signal_var)
    np.save(f"schedule_tensors/log_signal_var_n{n_steps}.npy", log_signal_var)
    np.save(f"schedule_tensors/snr_n{n_steps}.npy", snr)
    np.save(f"schedule_tensors/log_snr_n{n_steps}.npy", log_snr)

    # Create plots
    # 1. signal_var
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(signal_var)), signal_var)
    plt.title(f"Signal Variance (n_steps={n_steps}, window={window})")
    plt.xlabel("Step")
    plt.ylabel("Signal Variance")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/signal_var_n{n_steps}.png", dpi=300)
    plt.close()

    # 2. log signal_var
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(signal_var)), log_signal_var)
    plt.title(f"Log10 Signal Variance (n_steps={n_steps}, window={window})")
    plt.xlabel("Step")
    plt.ylabel("Log10 Signal Variance")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/log_signal_var_n{n_steps}.png", dpi=300)
    plt.close()

    # 3. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(snr)), snr)
    plt.title(f"Signal-to-Noise Ratio (n_steps={n_steps}, window={window})")
    plt.xlabel("Step")
    plt.ylabel("SNR")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/snr_n{n_steps}.png", dpi=300)
    plt.close()

    # 4. log SNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(snr)), log_snr)
    plt.title(f"Log10 Signal-to-Noise Ratio (n_steps={n_steps}, window={window})")
    plt.xlabel("Step")
    plt.ylabel("Log10 SNR")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/log_snr_n{n_steps}.png", dpi=300)
    plt.close()

    print(
        f"Plots and tensors for n_steps={n_steps} saved to schedule_plots/ and schedule_tensors/ directories"
    )


if __name__ == "__main__":
    save(60)
    save(100)
    save(400)
