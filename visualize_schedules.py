import os

import torch
from schedule import Schedule, visualize_schedule

# Create output directory if it doesn't exist
os.makedirs("schedule_visualizations", exist_ok=True)


def save_schedule_visualization(schedule, filename):
    """Helper function to save schedule visualization"""
    visualize_schedule(schedule)
    os.rename("schedule.png", f"schedule_visualizations/{filename}")


seq_len = 20  # Fixed sequence length used in runner.py
windows = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64, 128, 256]

schedule = Schedule.make_rolling(
    seq_len=seq_len,
    speed=1,  # AR uses speed=1
    denoise_steps=1,  # AR uses denoise_steps=1
    start_from=0,
    final_signal_var=0.01,
)
save_schedule_visualization(schedule, f"AR.png")

schedule = Schedule.make_rolling(
    seq_len=seq_len,
    speed=1e3,  # D uses high speed
    denoise_steps=seq_len,
    start_from=0,
    final_signal_var=0.01,
)
save_schedule_visualization(schedule, f"D.png")

for window in windows:
    schedule = Schedule.make_rolling(
        seq_len=seq_len,
        n_steps=seq_len,  # Total number of steps
        window=window,  # Window size determines denoising overlap
        start_from=0,
        final_signal_var=0.01,
    )
    save_schedule_visualization(schedule, f"UD_w_{window}.png")
