from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset

TRAIN_MODEL = False
MODEL_PATH = "denoiser.pt"


@typed
def gen(n: int, chaos_ratio: float = 0.5) -> Float[ND, "n"]:
    logistic_map = np.zeros(n)
    logistic_map[0] = np.random.rand()
    r = 3.9993
    for i in range(1, n):
        x = logistic_map[i - 1]
        logistic_map[i] += r * x * (1 - x)
    noise = np.random.randn(n) + np.random.randn()
    return logistic_map * chaos_ratio + noise * (1 - chaos_ratio)


@typed
def gen_dataset(dataset_size: int, *args, **kwargs) -> Float[ND, "dataset_size n"]:
    return np.stack([gen(*args, **kwargs) for _ in range(dataset_size)])


@typed
def uniform_schedule(seq_len: int) -> Float[TT, "seq_len"]:
    """Returns signal variance (pi) values ~ U[0,1] for each position"""
    return t.rand(seq_len)


@typed
def make_schedule(
    seq_len: int,
    n_steps: int | None = None,
    shape: str = "linear",
    speed: float | None = None,  # Tokens per step
    denoise_steps: int = 10,  # Steps to denoise each token
    start_from: int = 0,  # Position to start denoising from
) -> Float[TT, "n_steps seq_len"]:
    """Generate a noise schedule that progressively denoises from left to right.

    Args:
        seq_len: Length of sequence
        n_steps: Total number of denoising steps (computed from speed if None)
        shape: Schedule shape ('linear' for now)
        speed: How many tokens to advance per step (computed from n_steps if None)
        denoise_steps: How many steps to spend denoising each token
        start_from: Position to start denoising from (earlier positions stay clean)

    Returns:
        Schedule of signal variances [n_steps, seq_len]
        where 0 = pure noise, 1 = clean signal
    """
    if shape != "linear":
        raise ValueError("Only 'linear' shape supported for now")

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

    schedule = t.ones(n_steps, seq_len)  # Initialize all to clean

    # Only schedule denoising for tokens after start_from
    for step in range(n_steps):
        right_pos = start_from + step * speed
        pos = t.arange(start_from, seq_len)
        dist = (pos - right_pos) / (denoise_steps * speed)
        schedule[step, start_from:] = (-dist).clamp(1e-3, 1)

    return schedule


class DenoiserConv(nn.Module):
    def __init__(self, window_size=5):
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
        # Stack inputs as channels
        x = t.stack([noisy_seq, signal_var], dim=1)  # [batch, 2, seq_len]
        # Add causal padding on the left
        pad_size = self.window_size - 1
        x = F.pad(x, (pad_size, 0), mode="constant", value=0)
        x = self.conv(x)
        x = self.mlp(x)
        return x.squeeze(1)


class DiffusionDataset(Dataset):
    @typed
    def __init__(self, clean_data: Float[TT, "batch seq_len"], schedule_fn):
        """
        Args:
            clean_data: [num_samples, seq_len] tensor of clean sequences
            schedule_fn: function(seq_len) -> [seq_len] tensor of signal variances
        """
        self.clean_data = clean_data
        self.schedule_fn = schedule_fn

    @typed
    def __getitem__(self, idx: int) -> tuple[
        Float[TT, "seq_len"],  # xt
        Float[TT, "seq_len"],  # signal_var
        Float[TT, "seq_len"],  # noise
    ]:
        x0 = self.clean_data[idx]
        signal_var = self.schedule_fn(len(x0))
        noise = t.randn_like(x0)
        xt = t.sqrt(signal_var) * x0 + t.sqrt(1 - signal_var) * noise
        return xt, signal_var, noise

    def __len__(self):
        return len(self.clean_data)


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
    opt = t.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for xt, signal_var, noise in train_loader:
            xt, signal_var, noise = (
                xt.to(device),
                signal_var.to(device),
                noise.to(device),
            )
            pred_noise = model(xt, signal_var)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    return losses


@typed
def sample(
    model: nn.Module,
    x_t: Float[TT, "batch seq_len"],
    signal_var_schedule: Float[TT, "n_steps seq_len"],
) -> Float[TT, "batch n_steps seq_len"]:
    """Sample from the diffusion model following the schedule."""
    device = x_t.device
    n_steps = len(signal_var_schedule)
    batch_size, seq_len = x_t.shape

    # Assert that signal_var_schedule[:, i] is a decreasing sequence
    assert (signal_var_schedule[0, :] <= signal_var_schedule[-1, :]).all()
    # logger.info(f"signal_var_schedule shape: {signal_var_schedule.shape}")
    # logger.info(f"signal_var_schedule: {signal_var_schedule[:5, 10:15]}")

    # Store all intermediate steps [batch, n_steps, seq_len]
    xs = t.zeros(batch_size, n_steps, seq_len, device=device)
    xs[:, 0] = x_t
    eps = 1e-8

    for it in range(n_steps - 1):
        curr_var = signal_var_schedule[it]
        nxt_var = signal_var_schedule[it + 1]
        assert (curr_var <= nxt_var).all()
        alpha = curr_var / (nxt_var + eps)
        beta = 1 - alpha
        beta = beta.clamp(0.0, 1.0)
        x_cur = xs[:, it]
        assert not x_cur.isnan().any(), f"x_cur is nan at it={it}"
        pred_noise = model(x_cur, curr_var.repeat(batch_size, 1))

        upscale_coef = 1 / t.sqrt(alpha)
        noise_coef = beta / (t.sqrt(1 - curr_var) + eps)
        x_new = upscale_coef * (x_cur - noise_coef * pred_noise)

        if it < n_steps - 2:
            x_new = x_new + t.sqrt(beta) * t.randn_like(x_new)
        xs[:, it + 1] = x_new

    return xs


def main():
    # Generate synthetic dataset
    batch_size = 32
    seq_len = 100
    dataset_size = 1000
    device = "cuda" if t.cuda.is_available() else "cpu"

    # Initialize model
    model = DenoiserConv(window_size=5)

    if TRAIN_MODEL:
        clean_data = gen_dataset(dataset_size, seq_len, chaos_ratio=0.5)
        dataset = DiffusionDataset(t.from_numpy(clean_data).float(), uniform_schedule)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train model
        losses = train(model, train_loader, num_epochs=10, device=device)

        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss")
        plt.show()

        # Save model
        model.cpu()
        t.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
    else:
        # Load model
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file {MODEL_PATH} not found. Set TRAIN_MODEL=True to train first."
            )
        model.load_state_dict(t.load(MODEL_PATH))
        logger.info(f"Model loaded from {MODEL_PATH}")

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    # Check for NaNs in model parameters
    for p in model.parameters():
        assert not p.isnan().any()

    # Visualization code continues as before...
    with t.no_grad():
        xt = gen_dataset(1, seq_len, chaos_ratio=0.5)[0]
        xt = t.from_numpy(xt).float().unsqueeze(0).to(device)
        prefix = 10

        schedule = make_schedule(
            seq_len=len(xt[0]),
            n_steps=50,
            denoise_steps=10,
            start_from=prefix,
        )
        schedule = schedule.to(device)
        # # Heatmap of schedule
        # plt.figure(figsize=(10, 5))
        # plt.imshow(schedule.cpu(), cmap="hot", aspect="auto")
        # plt.colorbar()
        # plt.show()

        samples = sample(model, xt, schedule)

        logger.info(f"Samples shape: {samples.shape}")

        # Set up the animation
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()

        def update(frame):
            ax.clear()
            colors = plt.cm.jet(np.linspace(0, 1, len(schedule)))
            for i in range(frame + 1):
                alpha = 1.1 ** (i - frame)
                ax.plot(samples[0, i].cpu(), color=colors[i], lw=0.5, alpha=alpha)
            ax.set_title(f"Denoising Steps (Step {frame + 1}/{len(schedule)})")
            ax.set_ylim(samples[-1].cpu().min() - 1, samples[-1].cpu().max() + 1)

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(schedule),
            interval=100,
            blit=False,
        )

        # Save as GIF
        anim.save("denoising_animation.gif", writer="pillow")
        plt.close()

        logger.info("Animation saved as denoising_animation.gif")


if __name__ == "__main__":
    main()
