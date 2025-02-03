from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from data import DiffusionDataset, gen_dataset
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from rich.progress import track
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset
from utils import set_seed

MODEL_PATH = "denoiser.pt"
SEQ_LEN = 20
DENOISE_STEPS = 10
START_FROM = 3


class DenoiserConv(nn.Module):
    @typed
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        # Use padding=0 and manually pad on the left
        self.conv = nn.Conv1d(2, 32, kernel_size=window_size, padding=0)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    @typed
    def forward(
        self,
        noisy_seq: Float[TT, "batch seq_len"],
        signal_var: Float[TT, "batch seq_len"],
    ) -> Float[TT, "batch seq_len"]:
        """
        Args:
            noisy_seq: [batch_size, seq_len]
            signal_var: [batch_size, seq_len] - remaining signal variance per position
        Returns:
            [batch_size, seq_len] predicted noise
        """
        # Mock solution
        shifted = torch.zeros_like(noisy_seq)
        shifted[:, 1:] = noisy_seq[:, :-1]
        gt = (shifted + 0.1) % 1.0
        eps = 1e-8
        # noisy_seq = gt * sqrt(signal_var) + noise * sqrt(1 - signal_var)
        noise = (noisy_seq - gt * torch.sqrt(signal_var)) / (
            torch.sqrt(1 - signal_var) + eps
        )

        # Stack inputs as channels
        x = torch.stack([noisy_seq, signal_var], dim=1)  # [batch, 2, seq_len]
        # Add causal padding on the left
        pad_size = self.window_size - 1
        x = F.pad(x, (pad_size, 0), mode="constant", value=0)
        x = self.conv(x)
        x = self.mlp(x).squeeze(1)
        assert x.shape == noise.shape
        return x * 1e-9 + noise


@typed
def train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train model and return loss history"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for xt, signal_var, signal_ratio, noise in train_loader:
            xt, signal_var, signal_ratio, noise = (
                xt.to(device),
                signal_var.to(device),
                signal_ratio.to(device),
                noise.to(device),
            )
            pred_noise = model(xt, signal_var)
            delta_var = signal_var * (1 - signal_ratio)
            loss = ((pred_noise - noise).square() * delta_var).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    return losses


@typed
def do_sample(
    model: nn.Module,
    x_t: Float[TT, "batch seq_len"],
    schedule: Schedule,
) -> Float[TT, "batch n_steps seq_len"]:
    """Sample from the diffusion model following the schedule."""
    device = x_t.device
    n_steps = len(schedule.signal_var)
    batch_size, seq_len = x_t.shape

    # Store all intermediate steps [batch, n_steps, seq_len]
    xs = torch.zeros(batch_size, n_steps, seq_len, device=device)
    xs[:, -1] = x_t
    eps = 1e-8
    noise_clip = 5.0

    for it in reversed(range(n_steps - 1)):
        curr_var = schedule.signal_var[it + 1]
        assert not curr_var.isnan().any(), f"curr_var is nan at it={it}"
        alpha = schedule.signal_ratio[it + 1]
        beta = schedule.noise_level[it + 1]
        x_cur = xs[:, it + 1]
        assert not x_cur.isnan().any(), f"x_cur is nan at it={it}"
        pred_noise = model(x_cur, curr_var.repeat(batch_size, 1))
        max_noise = torch.max(torch.abs(pred_noise))
        if max_noise > noise_clip:
            logger.warning(
                f"Max noise at it={it} is {max_noise}, clipping to {noise_clip}"
            )
            pred_noise = pred_noise.clamp(-noise_clip, noise_clip)
        assert not pred_noise.isnan().any(), f"pred_noise is nan at it={it}"
        upscale_coef = 1 / torch.sqrt(alpha)
        assert not torch.isnan(upscale_coef).any(), f"upscale_coef is nan at it={it}"
        noise_coef = beta / (torch.sqrt(1 - curr_var) + eps)
        assert not torch.isnan(noise_coef).any(), f"noise_coef is nan at it={it}"
        # logger.info(f"beta: {beta}")
        # logger.info(f"sqrt(1 - curr_var): {torch.sqrt(1 - curr_var)}")
        # logger.info(f"noise_coef: {noise_coef}")
        assert (noise_coef <= 1).all(), f"noise_coef is greater than 1 at it={it}"
        x_new = upscale_coef * (x_cur - noise_coef * pred_noise)
        assert not torch.isnan(x_new).any(), f"x_new is nan at it={it}"

        if it < n_steps - 2:
            x_new = x_new + torch.sqrt(beta) * torch.randn_like(x_new)
            assert not torch.isnan(x_new).any(), f"x_new' is nan at it={it}"
        xs[:, it] = x_new

    return xs.flip(dims=[1])


def train_denoiser(
    output_path: str = "denoiser.pt",
    epochs: int = 100,
    batch_size: int = 32,
    dataset_size: int = 1000,
    seq_len: int = SEQ_LEN,
    chaos_ratio: float = 1.0,
    device: str | None = None,
    seed: int = 42,
):
    """Train the denoising model"""
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize model
    model = DenoiserConv(window_size=2)

    # Generate dataset
    clean_data = gen_dataset(dataset_size, seq_len, chaos_ratio=chaos_ratio)
    schedule = Schedule.make_rolling(
        seq_len,
        speed=1 / DENOISE_STEPS,
        denoise_steps=DENOISE_STEPS,
        start_from=START_FROM,
    )
    dataset = DiffusionDataset(clean_data, schedule)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    losses = train(model, train_loader, num_epochs=epochs, device=device)

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (seed={seed})")
    plt.savefig("training_loss.png")
    plt.close()

    # Save model
    model.cpu()
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")


def sample(
    model_path: str = "denoiser.pt",
    seq_len: int = SEQ_LEN,
    chaos_ratio: float = 1.0,
    schedule: Schedule | None = None,
    device: str | None = None,
    output_path: str = "denoising_animation.gif",
    seed: int = 42,
):
    """Sample from a trained model"""
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = DenoiserConv(window_size=2)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    logger.info("Loaded legacy model format")

    model.to(device).eval()

    with torch.no_grad():
        # Generate input sequence
        x0 = gen_dataset(1, seq_len, chaos_ratio=chaos_ratio)[0]
        x0 = x0.unsqueeze(0).to(device)

        # Create schedule
        schedule = Schedule.make_rolling(
            seq_len=seq_len,
            speed=1 / DENOISE_STEPS,
            denoise_steps=DENOISE_STEPS,
            start_from=START_FROM,
        )
        visualize_schedule(schedule)
        schedule = schedule.to(device)
        logger.info(f"Schedule shape: {schedule.signal_var.shape}")

        # Sample
        signal_var = schedule.signal_var[-1]
        # TODO: remove *0
        xt = (
            x0 * torch.sqrt(signal_var)
            + torch.randn_like(x0) * torch.sqrt(1 - signal_var) * 0
        )
        samples = do_sample(model, xt, schedule)
        logger.info(f"Generated samples shape: {samples.shape}")

        # Create animation
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()

        def update(frame):
            ax.clear()
            colors = plt.cm.jet(np.linspace(0, 1, len(schedule.signal_var)))
            for i in range(frame + 1):
                alpha = 1.1 ** (i - frame)
                ax.plot(samples[0, i].cpu(), color=colors[i], lw=0.5, alpha=alpha)
            ax.set_title(
                f"Denoising Steps (Step {frame + 1}/{len(schedule.signal_var)}, seed={seed})"
            )
            ax.set_ylim(-3, 3)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(schedule.signal_var),
            interval=100,
            blit=False,
        )

        anim.save(output_path, writer="pillow")
        plt.close()
        logger.info(f"Animation saved to {output_path}")


def test_model():
    model = DenoiserConv(window_size=2)
    model.load_state_dict(torch.load("denoiser.pt"))
    seed = 42
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    with torch.no_grad():
        schedule = Schedule.make_rolling(
            seq_len=SEQ_LEN,
            speed=1 / DENOISE_STEPS,
            denoise_steps=DENOISE_STEPS,
            start_from=START_FROM,
        )
        x0 = gen_dataset(1, SEQ_LEN, chaos_ratio=1.0)[0]
        x0 = x0.unsqueeze(0).to(device)
        schedule = schedule.to(device)
        for t in range(len(schedule.signal_var) - 1):
            print("\n\n")
            big_var = schedule.signal_var[t]
            small_var = schedule.signal_var[t + 1]
            ratio = schedule.signal_ratio[t + 1]
            xt = (
                x0 * torch.sqrt(small_var)
                + torch.randn_like(x0) * torch.sqrt(1 - small_var) * 0
            )
            logger.info(f"xt shape: {xt.shape} | small_var shape: {small_var.shape}")
            pred_noise = model(xt, small_var.repeat(1, 1))
            print(pred_noise.shape)
            print(pred_noise)


if __name__ == "__main__":
    # train_denoiser()
    sample()
    # test_model()
