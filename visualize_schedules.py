import os

import torch
from schedule import Schedule, visualize_schedule

# Create output directory if it doesn't exist
os.makedirs("schedule_visualizations", exist_ok=True)


def save_schedule_visualization(schedule, filename):
    """Helper function to save schedule visualization"""
    visualize_schedule(schedule)
    os.rename("schedule.png", f"schedule_visualizations/{filename}")


def get_swapped_permutation(seq_len: int, step: int) -> list:
    """Generate permutation with swapped elements at step intervals"""
    permutation = list(range(seq_len))
    for i in range(0, len(permutation) - (step - 1), step):
        j = i + step - 1
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


seq_len = 20  # Fixed sequence length used in runner.py
steps = [2, 4, 6, 8, 10]  # Step sizes for swaps
windows = [18, 20, 22, 24, 26, 28, 30, 32, 64]  # Window sizes for UD

# Test 1: Autoregressive (AR) schedules
for step in steps:
    permutation = get_swapped_permutation(seq_len, step)
    schedule = Schedule.make_rolling(
        seq_len=seq_len,
        speed=1,  # AR uses speed=1
        denoise_steps=1,  # AR uses denoise_steps=1
        start_from=0,
        final_signal_var=0.01,
    )
    save_schedule_visualization(schedule, f"AR_step_{step}.png")

# Test 2: Pure Diffusion (D) schedules
for step in steps:
    permutation = get_swapped_permutation(seq_len, step)
    schedule = Schedule.make_rolling(
        seq_len=seq_len,
        speed=1e3,  # D uses high speed
        denoise_steps=seq_len + 2,  # D uses seq_len + 2 steps
        start_from=0,
        final_signal_var=0.01,
    )
    save_schedule_visualization(schedule, f"D_step_{step}.png")

# Test 3: Unified Diffusion (UD) schedules with window parameter
for step in steps:
    permutation = get_swapped_permutation(seq_len, step)
    for window in windows:
        schedule = Schedule.make_rolling(
            seq_len=seq_len,
            n_steps=seq_len,  # Total number of steps
            window=window,  # Window size determines denoising overlap
            start_from=0,
            final_signal_var=0.01,
        )
        save_schedule_visualization(schedule, f"UD_step_{step}_window_{window}.png")

print("Generated schedule visualizations in schedule_visualizations/ directory")
print("\nParameters tested:")
print("1. Model types: AR, D (Diffusion), UD (Unified Diffusion)")
print("2. Step sizes:", steps)
print("3. Window sizes (for UD):", windows)
print("\nNote: All visualizations use sequence length =", seq_len)
