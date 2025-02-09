import math
from pathlib import Path
from typing import Annotated

import matplotlib.animation as animation

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from data import DataGenerator, DiffusionDataset, LogisticMap
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
DENOISE_STEPS = 1
SPEED = 1  # 4 / DENOISE_STEPS
START_FROM = 0
CAUSAL_MASK = True

D_MODEL = 64
N_HEADS = 16
N_LAYERS = 2
DROPOUT = 0.0


class Denoiser(nn.Module):
    @typed
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        # Input projection from 2 + pos features to d_model
        self.input_proj = nn.Linear(2, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        self.register_buffer("causal_mask", mask)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # [batch, seq, features]
            norm_first=True,  # Better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

        # Initialize positional encodings
        self._init_pos_encoding()

    def _init_pos_encoding(self):
        """Initialize positional encodings using sine/cosine functions"""
        position = torch.arange(SEQ_LEN).unsqueeze(1)
        div_term = torch.pi * torch.exp(
            torch.arange(0, self.pos_encoding.size(-1), 2)
            * (-math.log(1000.0) / self.pos_encoding.size(-1))
        )
        # print("Div term: ", div_term[:10].detach().cpu().numpy())
        pe = torch.zeros_like(self.pos_encoding[0])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe.unsqueeze(0))
        # Show pos_encoding as a heatmap
        plt.imshow(self.pos_encoding.detach().cpu().squeeze().numpy())
        plt.colorbar()
        plt.savefig("pos_encoding.png")
        plt.close()
        # print(f"Saved pos_encoding.png with shape {self.pos_encoding.shape}")

    @typed
    def forward(
        self,
        noisy_seq: Float[TT, "batch seq_len"],
        signal_var: Float[TT, "batch seq_len"],
        signal_ratio: Float[TT, "batch seq_len"],
    ) -> Float[TT, "batch seq_len"]:
        """
        Args:
            noisy_seq: [batch_size, seq_len]
            signal_var: [batch_size, seq_len] - remaining signal variance per position
            signal_ratio: [batch_size, seq_len] - signal ratio per position
        Returns:
            [batch_size, seq_len] predicted noise
        """
        x = torch.stack([noisy_seq, signal_var], dim=-1)  # [batch, seq_len, 2]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding
        if CAUSAL_MASK:
            x = self.transformer(x, mask=self.causal_mask)  # [batch, seq_len, d_model]
        else:
            x = self.transformer(x)  # [batch, seq_len, d_model]
        x = self.output_proj(x)  # [batch, seq_len, 1]
        x = x.squeeze(-1)  # [batch, seq_len]

        return x


@typed
def train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    eval_every: int = 10,
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
            pred_noise = model(xt, signal_var, signal_ratio)
            delta_var = signal_var * (1 / (signal_ratio + 1e-8) - 1)
            loss = ((pred_noise - noise).square() * delta_var).mean(dim=0).sum()
            # loss = (pred_noise - noise).square().sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        current_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(current_loss)
        half = len(losses) // 2
        avg_loss = sum(losses[half:]) / len(losses[half:])
        logger.info(f"Epoch {epoch}: loss = {current_loss:.4f} (avg={avg_loss:.4f})")

        if epoch % eval_every == 0:
            # Save model to `current_model.pt`
            model_path = f"current_model.pt"
            model.cpu()
            torch.save(model.state_dict(), model_path)
            model.to(device)
            evaluate(model_path, device=device)

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
        beta_cur = schedule.noise_level[it + 1]
        beta_next = schedule.noise_level[it]
        x_cur = xs[:, it + 1]
        assert not x_cur.isnan().any(), f"x_cur is nan at it={it}"
        pred_noise = model(
            x_cur, curr_var.repeat(batch_size, 1), alpha.repeat(batch_size, 1)
        )
        # max_noise = torch.max(torch.abs(pred_noise))
        # if max_noise > noise_clip:
        #     logger.warning(
        #         f"Max noise at it={it} is {max_noise}, clipping to {noise_clip}"
        #     )
        #     pred_noise = pred_noise.clamp(-noise_clip, noise_clip)
        assert not pred_noise.isnan().any(), f"pred_noise is nan at it={it}"
        upscale_coef = 1 / torch.sqrt(alpha)
        assert not torch.isnan(upscale_coef).any(), f"upscale_coef is nan at it={it}"
        noise_coef = beta_cur / (torch.sqrt(1 - curr_var) + eps)
        assert not torch.isnan(noise_coef).any(), f"noise_coef is nan at it={it}"
        # logger.info(f"beta: {beta}")
        # logger.info(f"sqrt(1 - curr_var): {torch.sqrt(1 - curr_var)}")
        # logger.info(f"noise_coef: {noise_coef}")
        assert (noise_coef <= 1).all(), f"noise_coef is greater than 1 at it={it}"
        x_new = upscale_coef * (x_cur - noise_coef * pred_noise)
        assert not torch.isnan(x_new).any(), f"x_new is nan at it={it}"

        if it < n_steps - 2:
            x_new = x_new + torch.sqrt(beta_next) * torch.randn_like(x_new)
            assert not torch.isnan(x_new).any(), f"x_new' is nan at it={it}"
        xs[:, it] = x_new

    return xs.flip(dims=[1])


def train_denoiser(
    output_path: str = "denoiser.pt",
    epochs: int = 1000,
    batch_size: int = 32,
    dataset_size: int = 200,
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
    model = Denoiser(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT
    )
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"#params = {n_parameters}")

    # Generate dataset
    generator = LogisticMap.load(
        length=SEQ_LEN, clauses=LogisticMap.complicated(SEQ_LEN), tolerance=1e-3
    )
    while len(generator) < dataset_size:
        generator.sample(10)
    clean_data = generator.data[:dataset_size]
    schedule = Schedule.make_rolling(
        seq_len,
        speed=SPEED,
        denoise_steps=DENOISE_STEPS,
        start_from=START_FROM,
    )
    visualize_schedule(schedule)
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


def load_model(model_path: str) -> tuple[nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Denoiser(
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model, device


def create_animation(
    samples: Float[TT, "batch n_steps seq_len"],
    schedule: Schedule,
    output_path: str,
    generator: DataGenerator,
) -> None:
    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()

    def update(frame):
        ax.clear()
        # colors = plt.cm.jet(np.linspace(0, 1, len(schedule.signal_var)))
        for i in range(frame + 1):
            alpha = 1.5 ** (i - frame)
            ax.plot(samples[0, i].cpu(), color="blue", lw=0.5, alpha=alpha)
        current = samples[:1, frame]
        assert current.shape == (1, SEQ_LEN)
        losses = generator.losses_per_clause(current)[0].detach().cpu().numpy()
        losses_str = " ".join(f"{loss:.3f}" for loss in losses)
        ax.set_title(
            f"Denoising Steps (Step {frame + 1}/{len(schedule.signal_var)})"
            f"\nLosses: {losses_str}"
        )
        ax.axhline(y=0, color="black", lw=0.5)
        ax.axhline(y=1, color="black", lw=0.5)
        ax.set_ylim(-1, 2)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(schedule.signal_var),
        interval=500,
        blit=False,
    )
    anim.save(output_path, writer="pillow")
    plt.close()
    logger.info(f"Animation saved to {output_path}")


def animated_sample(
    model_path: str = "denoiser.pt",
    seq_len: int = SEQ_LEN,
    output_path: str = "denoising_animation.gif",
    seed: int = 42,
):
    """Sample from a trained model"""
    set_seed(seed)
    model, device = load_model(model_path)
    generator = LogisticMap.load(
        length=SEQ_LEN, clauses=LogisticMap.complicated(SEQ_LEN), tolerance=1e-3
    )
    while len(generator) < 1:
        generator.sample(10)
    x0 = generator.data[0].unsqueeze(0).to(device)
    with torch.no_grad():
        # Create schedule
        schedule = Schedule.make_rolling(
            seq_len=seq_len,
            speed=SPEED,
            denoise_steps=DENOISE_STEPS,
            start_from=START_FROM,
        )
        schedule = schedule.to(device)
        signal_var = schedule.signal_var[-1]
        xt = x0 * torch.sqrt(signal_var) + torch.randn_like(x0) * torch.sqrt(
            1 - signal_var
        )
        samples = do_sample(model, xt, schedule)
        create_animation(samples, schedule, output_path, generator)


def evaluate(
    model_path: str = "denoiser.pt",
    n_samples: int = 1000,
    seq_len: int = SEQ_LEN,
    device: str | None = None,
):
    """Evaluate the model"""
    model, device = load_model(model_path)
    generator = LogisticMap.load(
        length=SEQ_LEN, clauses=LogisticMap.complicated(SEQ_LEN), tolerance=1e-3
    )
    schedule = Schedule.make_rolling(
        seq_len=seq_len,
        speed=SPEED,
        denoise_steps=DENOISE_STEPS,
        start_from=START_FROM,
    )
    schedule = schedule.to(device)
    signal_var = schedule.signal_var[-1]
    assert signal_var.max().item() <= 0.011
    with torch.no_grad():
        xt = torch.randn((n_samples, seq_len))
        samples = do_sample(model, xt, schedule)
        n_steps = samples.shape[1]
        for step in range(n_steps):
            selection = samples[:, step]
            losses = generator.loss(selection)
            q25 = losses.quantile(0.25)
            q50 = losses.quantile(0.50)
            q75 = losses.quantile(0.75)
            if step == 0 or step == n_steps - 1 or step == n_steps // 2:
                print(
                    f"Step {step}: {losses.mean():.3f} (q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f})"
                )


if __name__ == "__main__":
    train_denoiser()
    # evaluate()
    # animated_sampleload()
    # test_model()
