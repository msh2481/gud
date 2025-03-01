import math
from pathlib import Path
from typing import Annotated

import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from data import *
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from rich.progress import track
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset

torch.set_default_dtype(torch.float64)

# Initialize Sacred experiment
ex = Experiment("denoising_diffusion")
ex.observers.append(MongoObserver(db_name="sacred"))


@ex.config
def config():
    tags = ["gud"]

    # Model configuration
    model_config = {
        "seq_len": 20,
        "d_model": 64,
        "n_heads": 16,
        "n_layers": 4,
        "dropout": 0.0,
        "use_causal_mask": False,
    }

    # Training configuration
    train_config = {
        "output_path": "denoiser.pt",
        "epochs": 200,
        "batch_size": 16,
        "dataset_size": 2000,
        "lr": 1e-3,
        "eval_every": 20,
    }

    # Diffusion configuration
    diffusion_config = {
        "window": 1,
        "sampling_steps": 400,
    }

    generator_config = {
        "generator_class": "LogisticMapPermutation",
        "length": 20,
        "tolerance": 1e-3,
        "permutation": list(range(20)),
    }


@typed
def combine_gaussians(
    mu_prior: Float[TT, "..."],
    sigma_prior: Float[TT, "..."],
    mu_posterior: Float[TT, "..."],
    sigma_posterior: Float[TT, "..."],
) -> tuple[Float[TT, "..."], Float[TT, "..."]]:
    precision_prior = 1.0 / sigma_prior
    precision_posterior = 1.0 / sigma_posterior
    precision_combined = precision_prior + precision_posterior
    sigma_combined = 1.0 / precision_combined
    mu_combined = sigma_combined * (
        precision_prior * mu_prior + precision_posterior * mu_posterior
    )
    return mu_combined, sigma_combined


class Denoiser(nn.Module):
    @typed
    def __init__(
        self,
        N: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.pos_encoding = nn.Parameter(torch.zeros(1, N, d_model))
        self.input_proj = nn.Linear(3, d_model)

        mask = torch.triu(torch.ones(N, N), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        self.register_buffer("causal_mask", mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # [batch, seq, features]
            norm_first=True,  # Better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        # Modified to output both mu and log_sigma
        self.output_proj = nn.Linear(d_model, 2)
        self._init_pos_encoding()

    def _init_pos_encoding(self):
        """Initialize positional encodings using sine/cosine functions"""
        position = torch.arange(self.pos_encoding.size(1)).unsqueeze(1)
        div_term = torch.pi * torch.exp(
            torch.arange(0, self.pos_encoding.size(-1), 2)
            * (-math.log(1000.0) / self.pos_encoding.size(-1))
        )
        pe = torch.zeros_like(self.pos_encoding[0])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data.copy_(pe.unsqueeze(0))

    @typed
    def forward(
        self,
        noisy_seq: Float[TT, "batch seq_len"],
        signal_var: Float[TT, "batch seq_len"],
    ) -> Float[TT, "batch seq_len"]:
        x = torch.stack(
            [noisy_seq, torch.sqrt(signal_var), torch.sqrt(1 - signal_var)], dim=-1
        )  # [batch, seq_len, 3]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding
        if self.use_causal_mask:
            x = self.transformer(x, mask=self.causal_mask)  # [batch, seq_len, d_model]
        else:
            x = self.transformer(x)  # [batch, seq_len, d_model]

        outputs = self.output_proj(x)  # [batch, seq_len, 2]
        mu_likelihood = outputs[..., 0]  # [batch, seq_len]
        log_sigma_likelihood = outputs[..., 1]  # [batch, seq_len]
        sigma_likelihood = F.softplus(log_sigma_likelihood)

        # Prior: N(noisy_seq, 1 - signal_var)
        mu_prior = noisy_seq
        sigma_prior = 1 - signal_var

        mu, _ = combine_gaussians(
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
            mu_posterior=mu_likelihood,
            sigma_posterior=sigma_likelihood,
        )

        return mu


@typed
def backward_process(
    model: nn.Module,
    schedule: Schedule,
    *,
    p: Float[TT, "batch"],
    t: Float[TT, "batch"],
    x_t: Float[TT, "batch seq_len"],
) -> tuple[Float[TT, "batch seq_len"], Float[TT, "batch seq_len"]]:
    assert (p < t).all(), f"p must be less than t, got p={p}, t={t}"
    alpha_0p = schedule.signal_var(p)
    alpha_0t = schedule.signal_var(t)
    alpha_pt = alpha_0t / alpha_0p
    beta_0p = 1 - alpha_0p
    beta_0t = 1 - alpha_0t
    beta_pt = 1 - alpha_pt
    x0_hat = model(x_t, alpha_0p)
    mu = (
        beta_pt / beta_0t * alpha_0p.sqrt() * x0_hat
        + beta_0p / beta_0t * alpha_pt.sqrt() * x_t
    )
    sigma = beta_0p / beta_0t * beta_pt
    return mu, sigma


@typed
@ex.capture
def get_samples(
    model: nn.Module,
    x_1: Float[TT, "batch seq_len"],
    schedule: Schedule,
    diffusion_config: dict,
) -> Float[TT, "batch n_steps seq_len"]:
    """Sample from the diffusion model following the schedule."""
    device = x_1.device
    n_steps = diffusion_config["sampling_steps"]
    batch_size, seq_len = x_1.shape
    xs = torch.zeros(batch_size, n_steps + 1, seq_len, device=device)
    ts = torch.arange(n_steps + 1, device=device, dtype=torch.float64) / n_steps
    ts = ts.repeat(batch_size, 1)
    xs[:, n_steps] = x_1

    for it in reversed(range(n_steps)):
        x_t = xs[:, it + 1]
        p = ts[:, it]
        t = ts[:, it + 1]
        mu, sigma = backward_process(model, schedule, p=p, t=t, x_t=x_t)
        xs[:, it] = mu + sigma * torch.randn_like(x_t)
    return xs.flip(dims=[1])


@ex.capture
@typed
def setup_model(
    _run, model_config: dict, train_config: dict
) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = Denoiser(
        N=model_config["seq_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        use_causal_mask=model_config["use_causal_mask"],
    )
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"#params = {n_parameters}")
    _run.log_scalar("n_parameters", n_parameters)
    return model, device


@ex.capture
@typed
def get_dataset(
    _run,
    model_config: dict,
    train_config: dict,
    diffusion_config: dict,
    generator_config: dict,
    inference: bool = False,
) -> tuple[DataGenerator, Float[TT, "batch seq_len"], Schedule, DataLoader]:
    generator_class = globals()[generator_config["generator_class"]]
    generator = generator_class(**generator_config)
    while len(generator) < train_config["dataset_size"]:
        generator.sample(10)
    # generator.append_to_save()
    clean_data = generator.data[: train_config["dataset_size"]]
    w = torch.tensor(diffusion_config["window"], dtype=torch.float64)
    schedule = Schedule(
        N=model_config["seq_len"],
        w=w,
    )
    visualize_schedule(schedule)
    dataset = DiffusionDataset(clean_data, schedule)
    train_loader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True
    )
    return generator, clean_data, schedule, train_loader


@ex.capture
@typed
def train_batch(
    model: nn.Module,
    batch: tuple[
        Float[TT, "batch seq_len"],  # xt
        Float[TT, "batch seq_len"],  # signal_var
        Float[TT, "batch seq_len"],  # dsnr_dt
        Float[TT, "batch seq_len"],  # x0
        Float[TT, "batch"],  # timestep
    ],
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    xt, signal_var, dsnr_dt, x0, timestep = batch
    xt, signal_var, dsnr_dt, x0 = (
        xt.to(device),
        signal_var.to(device),
        dsnr_dt.to(device),
        x0.to(device),
    )
    x0_hat = model(xt, signal_var)
    x0_errors = (x0_hat - x0).square()
    losses = (dsnr_dt * x0_errors).sum(dim=-1)
    # losses = x0_errors.sum(dim=-1)
    assert losses.shape == (len(xt),), f"losses.shape = {losses.shape}"
    loss = losses.mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@ex.capture
@typed
def save_model(model: nn.Module, train_config: dict, device: torch.device) -> None:
    model_path = train_config["output_path"]
    model.cpu()
    torch.save(model.state_dict(), model_path)
    model.to(device)


@ex.capture
def train_denoiser(_run, model_config, train_config, diffusion_config):
    """Train the denoising model"""
    model, device = setup_model()
    _generator, _clean_data, _schedule, train_loader = get_dataset(inference=False)
    opt = torch.optim.Adam(
        model.parameters(), lr=train_config["lr"], betas=(0.95, 0.999)
    )
    losses = []
    for epoch in range(train_config["epochs"]):
        epoch_losses = []

        for batch in train_loader:
            loss = train_batch(model=model, batch=tuple(batch), opt=opt, device=device)
            epoch_losses.append(loss)

        # Compute epoch & avg loss
        current_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(current_loss)
        half = len(losses) // 2
        avg_loss = sum(losses[half:]) / len(losses[half:])
        logger.info(f"Epoch {epoch}: loss = {current_loss:.6f} (avg={avg_loss:.6f})")

        # Log metrics to Sacred
        _run.log_scalar("loss", current_loss, epoch)
        _run.log_scalar("avg_loss", avg_loss, epoch)

        # Save model & evaluate
        if (epoch + 1) % train_config["eval_every"] == 0:
            save_model(model=model, device=device)
            evaluate(model_path=train_config["output_path"], epoch_number=epoch)

    save_model(model=model, device=device)


@ex.capture
def load_model(
    model_path: str, diffusion_config: dict, model_config: dict
) -> tuple[nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Denoiser(
        N=model_config["seq_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        use_causal_mask=model_config["use_causal_mask"],
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found")
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model, device


@ex.capture
def create_animation(
    samples: Float[TT, "batch n_steps seq_len"],
    schedule: Schedule,
    output_path: str,
    generator: DataGenerator,
    model_config: dict,
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
        assert current.shape == (1, model_config["seq_len"])
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
        interval=1500,
        blit=False,
    )
    anim.save(output_path, writer="pillow")
    plt.close()
    logger.info(f"Animation saved to {output_path}")


@ex.capture
def animated_sample(
    diffusion_config: dict,
    train_config: dict,
    output_path: str = "denoising_animation.gif",
):
    """Sample from a trained model"""
    model, device = load_model(model_path=train_config["output_path"])
    generator, clean_data, schedule, _train_loader = get_dataset(inference=True)

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    x0 = clean_data[:1].to(device)
    assert x0.dtype == torch.float64, f"x0 should be float64, got {x0.dtype}"

    schedule = schedule.to(device)
    signal_var = schedule.signal_var[-1]
    xt = x0 * torch.sqrt(signal_var) + torch.randn_like(x0) * torch.sqrt(1 - signal_var)
    assert xt.dtype == torch.float64, f"xt should be float64, got {xt.dtype}"

    with torch.no_grad():
        samples = get_samples(model, xt, schedule)
        assert (
            samples.dtype == torch.float64
        ), f"samples should be float64, got {samples.dtype}"
        create_animation(samples, schedule, output_path, generator)


@ex.capture
def evaluate(
    _run,
    model_config: dict,
    model_path: str = "denoiser.pt",
    n_samples: int = 100,
    epoch_number: int | None = None,
):
    """Evaluate the model"""
    model, device = load_model(model_path=model_path)
    generator, clean_data, schedule, _train_loader = get_dataset(inference=True)

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    schedule = schedule.to(device)
    with torch.no_grad():
        x_1 = torch.randn((n_samples, model_config["seq_len"]), dtype=torch.float64)
        assert x_1.dtype == torch.float64, f"xt should be float64, got {x_1.dtype}"

        samples = get_samples(model, x_1, schedule)
        assert (
            samples.dtype == torch.float64
        ), f"samples should be float64, got {samples.dtype}"

        n_steps = samples.shape[1]
        q25, q50, q75 = 0, 0, 0
        for step in range(n_steps):
            selection = samples[:, step]
            losses = generator.loss(selection)
            assert (
                losses.dtype == torch.float64
            ), f"losses should be float64, got {losses.dtype}"

            q25 = losses.quantile(0.25).item()
            q50 = losses.quantile(0.50).item()
            q75 = losses.quantile(0.75).item()
            if step == 0 or step == n_steps - 1 or step == n_steps // 2:
                print(
                    f"Step {step}: {losses.mean():.3f} (q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f})"
                )
        _run.log_scalar("q25", q25, epoch_number)
        _run.log_scalar("q50", q50, epoch_number)
        _run.log_scalar("q75", q75, epoch_number)

        # Save distribution of final samples
        final_samples = samples[:, -1, :]  # [n_samples, seq_len]
        final_losses = generator.loss(final_samples)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Main plot for all samples
        for i in range(n_samples):
            ax1.plot(
                final_samples[i].cpu().numpy(), alpha=0.2, color="blue", linewidth=0.5
            )

        # Add statistics to the plot
        mean_loss = final_losses.mean().item()
        q50 = final_losses.quantile(0.50).item()

        title = f"Distribution of {n_samples} Samples"
        if epoch_number is not None:
            title += f" (Epoch {epoch_number})"
        title += f"\nMean Loss: {mean_loss:.3f}, Median: {q50:.3f}"

        ax1.set_title(title)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)

        # Create histogram of first element values
        first_elements = final_samples[:, 0].cpu().numpy()
        ax2.hist(first_elements, bins=20, alpha=0.7, color="blue")
        ax2.set_xlim(0, 1)
        ax2.set_title("Histogram of First Element Values")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("samples.png", dpi=300)
        plt.close()

        logger.info("Evaluation visualization saved to samples.png")

        return final_samples, final_losses


@ex.capture
def sample_distribution(
    _run,
    model_path: str,
    diffusion_config: dict,
    train_config: dict,
    model_config: dict,
    n_samples: int = 1000,
):
    """Generate multiple samples and visualize their distribution.

    This function:
    1. Generates n_samples from the diffusion model
    2. Plots all samples as line plots to visualize the distribution
    3. Creates a histogram of the first element values across all samples
    """
    model, device = load_model(model_path=model_path)
    generator, clean_data, schedule, _train_loader = get_dataset(inference=True)

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    schedule = schedule.to(device)

    # Generate samples
    with torch.no_grad():
        xt = torch.randn((n_samples, model_config["seq_len"]), dtype=torch.float64)
        assert xt.dtype == torch.float64, f"xt should be float64, got {xt.dtype}"

        samples = get_samples(model, xt, schedule)
        assert (
            samples.dtype == torch.float64
        ), f"samples should be float64, got {samples.dtype}"

        # Get final samples (last step of denoising)
        final_samples = samples[:, -1, :]  # [n_samples, seq_len]

        # Calculate losses for each sample
        losses = generator.loss(final_samples)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Main plot for all samples
        for i in range(n_samples):
            ax1.plot(
                final_samples[i].cpu().numpy(), alpha=0.2, color="blue", linewidth=0.5
            )

        # Add statistics to the plot
        mean_loss = losses.mean().item()
        q50 = losses.quantile(0.50).item()

        ax1.set_title(
            f"Distribution of {n_samples} Samples\nMean Loss: {mean_loss:.3f}, Median: {q50:.3f}"
        )
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)

        # Create histogram of first element values
        first_elements = final_samples[:, 0].cpu().numpy()
        ax2.hist(first_elements, bins=20, alpha=0.7, color="blue")
        ax2.set_xlim(0, 1)
        ax2.set_title("Histogram of First Element Values")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("samples.png", dpi=300)
        plt.close()

        logger.info("Sample distribution visualization saved to samples.png")
        return final_samples, losses


@ex.automain
def main():
    train_denoiser()
    # animated_sample()
    # sample_distribution(model_path="models/69900e21-5e57-45d9-914c-93a299b79f93.pt")
