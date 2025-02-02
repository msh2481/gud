from pathlib import Path
from typing import Annotated

import numpy as np
import torch as t
import torch.nn.functional as F
import typer
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from rich.progress import track
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset

app = typer.Typer()

TRAIN_MODEL = True
MODEL_PATH = "denoiser.pt"


@typed
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


@typed
def gen(n: int, chaos_ratio: float = 0.5) -> Float[ND, "n"]:
    logistic_map = np.zeros(n)
    logistic_map[0] = np.random.rand()
    r = 3.9993
    for i in range(1, n):
        x = logistic_map[i - 1]
        logistic_map[i] += (x + 0.1) % 1.0  # r * x * (1 - x)
    noise = np.random.randn(n) + np.random.randn()
    return logistic_map * chaos_ratio + noise * (1 - chaos_ratio)


@typed
def gen_dataset(dataset_size: int, *args, **kwargs) -> Float[ND, "dataset_size n"]:
    return np.stack([gen(*args, **kwargs) for _ in range(dataset_size)])


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
def do_sample(
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


@app.command()
def visualize_data(
    n_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 5,
    seq_len: Annotated[int, typer.Option(help="Length of each sequence")] = 100,
    chaos_ratio: Annotated[
        float, typer.Option(help="Ratio of chaos vs noise (0-1)")
    ] = 1.0,
    save_path: Annotated[str | None, typer.Option(help="Path to save the plot")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    """Generate and visualize sample sequences"""
    set_seed(seed)
    data = gen_dataset(n_samples, seq_len, chaos_ratio=chaos_ratio)

    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.plot(data[i], label=f"Sample {i+1}", alpha=0.7)
    plt.title(f"Generated Sequences (chaos_ratio={chaos_ratio}, seed={seed})")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


@app.command()
def train_denoiser(
    output_path: Annotated[
        str, typer.Option(help="Path to save the trained model")
    ] = "denoiser.pt",
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 100,
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 32,
    dataset_size: Annotated[
        int, typer.Option(help="Number of training sequences")
    ] = 1000,
    seq_len: Annotated[int, typer.Option(help="Length of each sequence")] = 100,
    chaos_ratio: Annotated[
        float, typer.Option(help="Ratio of chaos vs noise (0-1)")
    ] = 1.0,
    device: Annotated[
        str | None, typer.Option(help="Device to train on (cpu/cuda)")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    """Train the denoising model"""
    set_seed(seed)
    if device is None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize model
    model = DenoiserConv(window_size=5)

    # Generate dataset
    clean_data = gen_dataset(dataset_size, seq_len, chaos_ratio=chaos_ratio)
    dataset = DiffusionDataset(t.from_numpy(clean_data).float(), uniform_schedule)
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
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "chaos_ratio": chaos_ratio,
            "seq_len": seq_len,
            "epochs": epochs,
        },
        output_path,
    )
    logger.info(f"Model saved to {output_path}")


@app.command()
def sample(
    model_path: Annotated[
        str, typer.Option(help="Path to the trained model")
    ] = "denoiser.pt",
    seq_len: Annotated[int, typer.Option(help="Length of sequence to generate")] = 100,
    n_steps: Annotated[int, typer.Option(help="Number of denoising steps")] = 50,
    denoise_steps: Annotated[
        int, typer.Option(help="Steps to denoise each token")
    ] = 10,
    prefix: Annotated[int, typer.Option(help="Position to start denoising from")] = 10,
    chaos_ratio: Annotated[
        float, typer.Option(help="Ratio of chaos vs noise (0-1)")
    ] = 0.5,
    device: Annotated[
        str | None, typer.Option(help="Device to run on (cpu/cuda)")
    ] = None,
    output_path: Annotated[
        str, typer.Option(help="Path to save the animation")
    ] = "denoising_animation.gif",
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    """Sample from a trained model"""
    set_seed(seed)
    if device is None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = DenoiserConv(window_size=5)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found")

    checkpoint = t.load(model_path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded model trained with seed={checkpoint.get('seed')}, "
            f"chaos_ratio={checkpoint.get('chaos_ratio')}"
        )
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded legacy model format")

    model.to(device).eval()

    with t.no_grad():
        # Generate input sequence
        xt = gen_dataset(1, seq_len, chaos_ratio=chaos_ratio)[0]
        xt = t.from_numpy(xt).float().unsqueeze(0).to(device)

        # Create schedule
        schedule = make_schedule_for_sampling(
            seq_len=len(xt[0]),
            n_steps=n_steps,
            denoise_steps=denoise_steps,
            start_from=prefix,
        ).to(device)

        # Sample
        samples = do_sample(model, xt, schedule)
        logger.info(f"Generated samples shape: {samples.shape}")

        # Create animation
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()

        def update(frame):
            ax.clear()
            colors = plt.cm.jet(np.linspace(0, 1, len(schedule)))
            for i in range(frame + 1):
                alpha = 1.1 ** (i - frame)
                ax.plot(samples[0, i].cpu(), color=colors[i], lw=0.5, alpha=alpha)
            ax.set_title(
                f"Denoising Steps (Step {frame + 1}/{len(schedule)}, seed={seed})"
            )
            ax.set_ylim(samples[-1].cpu().min() - 1, samples[-1].cpu().max() + 1)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(schedule),
            interval=100,
            blit=False,
        )

        anim.save(output_path, writer="pillow")
        plt.close()
        logger.info(f"Animation saved to {output_path}")


if __name__ == "__main__":
    app()
