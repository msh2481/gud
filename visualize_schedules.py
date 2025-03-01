import os

import torch
from schedule import Schedule, visualize_schedule

# Create output directory if it doesn't exist
os.makedirs("schedule_visualizations", exist_ok=True)


def save_schedule_visualization(schedule, filename):
    """Helper function to save schedule visualization"""
    visualize_schedule(schedule)
    os.rename("schedule.png", f"schedule_visualizations/{filename}")


seq_len = 2
windows = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64, 128, 256]


for window in windows:
    w = torch.tensor(window, dtype=torch.float64)
    schedule = Schedule(
        N=seq_len,
        w=w,
    )
    save_schedule_visualization(schedule, f"UD_w_{window}.png")
