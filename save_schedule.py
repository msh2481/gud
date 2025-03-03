import os

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
    Build a Schedule with n_steps and w parameter, then plot and save signal_var,
    log signal_var, SNR, and log SNR for all steps.

    Args:
        n_steps: Number of steps in the schedule
    """
    # Create output directories if they don't exist
    os.makedirs("schedule_plots", exist_ok=True)
    os.makedirs("schedule_tensors", exist_ok=True)

    # Build schedule with the specified parameters
    w = torch.tensor(1e3, dtype=torch.float64)

    # Create the schedule
    schedule = Schedule(
        N=20,
        w=w,
    )

    # Sample time points
    times = torch.linspace(0, 1, n_steps, dtype=torch.float64)

    # Extract data for the first token only (index 0)
    signal_var = schedule.signal_var(times)[:, 0].cpu().numpy()
    snr_values = schedule.snr(times)[:, 0].cpu().numpy()
    log_snr_values = (
        schedule.log_snr(times)[:, 0].cpu() / torch.log(torch.tensor(10.0))
    ).numpy()

    # Calculate log values for signal variance
    log_signal_var = np.log10(signal_var)

    # Save tensors as numpy arrays
    np.save(f"schedule_tensors/new_signal_var_n{n_steps}.npy", signal_var)
    np.save(f"schedule_tensors/new_log_signal_var_n{n_steps}.npy", log_signal_var)
    np.save(f"schedule_tensors/new_snr_n{n_steps}.npy", snr_values)
    np.save(f"schedule_tensors/new_log_snr_n{n_steps}.npy", log_snr_values)

    # Create plots
    # 1. signal_var
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(signal_var)), signal_var)
    plt.title(f"Signal Variance (n_steps={n_steps}, w={w.item()})")
    plt.xlabel("Step")
    plt.ylabel("Signal Variance")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/new_signal_var_n{n_steps}.png", dpi=300)
    plt.close()

    # 2. log signal_var
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(signal_var)), log_signal_var)
    plt.title(f"Log10 Signal Variance (n_steps={n_steps}, w={w.item()})")
    plt.xlabel("Step")
    plt.ylabel("Log10 Signal Variance")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/new_log_signal_var_n{n_steps}.png", dpi=300)
    plt.close()

    # 3. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(snr_values)), snr_values)
    plt.title(f"Signal-to-Noise Ratio (n_steps={n_steps}, w={w.item()})")
    plt.xlabel("Step")
    plt.ylabel("SNR")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/new_snr_n{n_steps}.png", dpi=300)
    plt.close()

    # 4. log SNR
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(log_snr_values)), log_snr_values)
    plt.title(f"Log10 Signal-to-Noise Ratio (n_steps={n_steps}, w={w.item()})")
    plt.xlabel("Step")
    plt.ylabel("Log10 SNR")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"schedule_plots/new_log_snr_n{n_steps}.png", dpi=300)
    plt.close()

    print(
        f"Plots and tensors for n_steps={n_steps} saved to schedule_plots/ and schedule_tensors/ directories"
    )


if __name__ == "__main__":
    save(60)
    save(100)
    save(400)
