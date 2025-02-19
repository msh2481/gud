import math
from pathlib import Path
from typing import Annotated

import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from data import *
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from rich.progress import track
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset
from utils import set_seed

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
        "predict_x0": True,
        "use_causal_mask": False,
    }

    # Training configuration
    train_config = {
        "output_path": "denoiser.pt",
        "epochs": 200,
        "batch_size": 32,
        "dataset_size": 2000,
        "lr": 1e-3,
        "eval_every": 10,
    }

    # Diffusion configuration
    diffusion_config = {
        "denoise_steps": 1,
        "speed": 1,
        "start_from": 0,
    }

    generator_config = {
        "generator_class": "LogisticMapPermutation",
        "length": 20,
        "tolerance": 1e-3,
        "permutation": list(range(20)),
    }


class Denoiser(nn.Module):
    @typed
    def __init__(
        self,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        predict_x0: bool = True,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.predict_x0 = predict_x0
        self.use_causal_mask = use_causal_mask
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_proj = nn.Linear(2, d_model)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
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
        self.output_proj = nn.Linear(d_model, 1)
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
        signal_ratio: Float[TT, "batch seq_len"],
    ) -> Float[TT, "batch seq_len"]:
        x = torch.stack([noisy_seq, signal_var], dim=-1)  # [batch, seq_len, 2]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding
        if self.use_causal_mask:
            x = self.transformer(x, mask=self.causal_mask)  # [batch, seq_len, d_model]
        else:
            x = self.transformer(x)  # [batch, seq_len, d_model]
        x = self.output_proj(x)  # [batch, seq_len, 1]
        x = x.squeeze(-1)  # [batch, seq_len]

        if self.predict_x0:
            eps = (noisy_seq - torch.sqrt(signal_var) * x) / torch.sqrt(
                1 - signal_var + 1e-8
            )
            return eps
        else:
            return x


@ex.capture
def get_samples(model, x_t, schedule, diffusion_config):
    """Sample from the diffusion model following the schedule."""
    device = x_t.device
    n_steps = len(schedule.signal_var)
    batch_size, seq_len = x_t.shape

    # Store all intermediate steps [batch, n_steps, seq_len]
    xs = torch.zeros(batch_size, n_steps, seq_len, device=device)
    xs[:, -1] = x_t
    eps = 1e-8

    for it in reversed(range(n_steps - 1)):
        curr_var = schedule.signal_var[it + 1]
        assert not curr_var.isnan().any(), f"curr_var is nan at it={it}"
        alpha = schedule.signal_ratio[it + 1]
        beta_cur = schedule.noise_level[it + 1]
        beta_next = schedule.noise_level[it]
        x_cur = xs[:, it + 1]
        assert not x_cur.isnan().any(), f"x_cur is nan at it={it}"
        pred_noise = model(
            x_cur,
            curr_var.repeat(batch_size, 1),
            alpha.repeat(batch_size, 1),
        )
        upscale_coef = 1 / torch.sqrt(alpha)
        noise_coef = beta_cur / (torch.sqrt(1 - curr_var) + eps)
        x_new = upscale_coef * (x_cur - noise_coef * pred_noise)

        if it < n_steps - 2:
            x_new = x_new + torch.sqrt(beta_next) * torch.randn_like(x_new)
        xs[:, it] = x_new
    return xs.flip(dims=[1])


@ex.capture
@typed
def setup_model(
    _run, model_config: dict, train_config: dict
) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = Denoiser(
        seq_len=model_config["seq_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        predict_x0=model_config["predict_x0"],
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
) -> tuple[DataGenerator, Float[TT, "batch seq_len"], Schedule, DataLoader]:
    generator_class = globals()[generator_config["generator_class"]]
    generator = generator_class(**generator_config)
    while len(generator) < train_config["dataset_size"]:
        generator.sample(10)
    # generator.append_to_save()
    clean_data = generator.data[: train_config["dataset_size"]]
    schedule = Schedule.make_rolling(
        seq_len=model_config["seq_len"],
        speed=diffusion_config["speed"],
        denoise_steps=diffusion_config["denoise_steps"],
        start_from=diffusion_config["start_from"],
    )
    visualize_schedule(schedule)
    dataset = DiffusionDataset(clean_data, schedule)
    train_loader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True
    )
    return generator, clean_data, schedule, train_loader


@typed
def train_batch(
    model: nn.Module,
    batch: tuple[
        Float[TT, "batch seq_len"],  # xt
        Float[TT, "batch seq_len"],  # signal_var
        Float[TT, "batch seq_len"],  # signal_ratio
        Float[TT, "batch seq_len"],  # true_noise
    ],
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    xt, signal_var, signal_ratio, true_noise = batch
    xt, signal_var, signal_ratio, true_noise = (
        xt.to(device),
        signal_var.to(device),
        signal_ratio.to(device),
        true_noise.to(device),
    )
    pred_noise = model(xt, signal_var, signal_ratio)
    delta_var = signal_var * (1 / (signal_ratio + 1e-8) - 1)
    loss = ((pred_noise - true_noise).square() * delta_var).mean(dim=0).sum()
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
    param_vector = torch.cat([p.data.view(-1) for p in model.parameters()]).detach()
    _generator, _clean_data, _schedule, train_loader = get_dataset()
    opt = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    losses = []
    for epoch in range(train_config["epochs"]):
        epoch_losses = [
            train_batch(model=model, batch=tuple(batch), opt=opt, device=device)
            for batch in train_loader
        ]
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
        if epoch % train_config["eval_every"] == 0:
            save_model(model=model, device=device)
            evaluate(model_path=train_config["output_path"], epoch_number=epoch)
    save_model(model=model, device=device)


@ex.capture
def load_model(
    model_path: str, diffusion_config: dict, model_config: dict
) -> tuple[nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Denoiser(
        seq_len=model_config["seq_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        predict_x0=model_config["predict_x0"],
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
    generator, clean_data, schedule, _train_loader = get_dataset()
    x0 = clean_data[:1].to(device)
    schedule = schedule.to(device)
    signal_var = schedule.signal_var[-1]
    xt = x0 * torch.sqrt(signal_var) + torch.randn_like(x0) * torch.sqrt(1 - signal_var)
    with torch.no_grad():
        samples = get_samples(model, xt, schedule)
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
    generator, clean_data, schedule, _train_loader = get_dataset()
    schedule = schedule.to(device)
    signal_var = schedule.signal_var[-1]
    assert (
        signal_var.max().item() <= 0.011
    ), f"not noised enough: signal_var.max() = {signal_var.max().item()}"
    with torch.no_grad():
        xt = torch.randn((n_samples, model_config["seq_len"]))
        samples = get_samples(model, xt, schedule)
        n_steps = samples.shape[1]
        q25, q50, q75 = 0, 0, 0
        for step in range(n_steps):
            selection = samples[:, step]
            losses = generator.loss(selection)
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


@ex.automain
def main():
    train_denoiser()
    # evaluate()
    # animated_sample()
    # test_model()
