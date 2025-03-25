import math
from pathlib import Path

import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype as typed
from beartype.typing import Any
from data import *
import neptune
from jaxtyping import Float, Int
from loguru import logger
from matplotlib import pyplot as plt
from neptune.integrations.sacred import NeptuneObserver
from neptune.types import File
from numpy import ndarray as ND
from rich.progress import track
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from schedule import Schedule, visualize_schedule
from torch import nn, Tensor as TT
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid


run = neptune.init_run(
    project="mlxa/GUD",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
)
torch.set_default_dtype(torch.float64)

# Initialize Sacred experiment
ex = Experiment("denoising_diffusion")
mongo_url = "mongodb://0.tcp.ngrok.io:10368/sacred"
ex.observers.append(MongoObserver.create(url=mongo_url))
ex.observers.append(NeptuneObserver(run=run))


@ex.config
def config():
    tags = ["gud"]
    length = 20

    # Model configuration
    model_config = {
        "seq_len": length,
        "d_model": 64,
        "n_heads": 16,
        "n_layers": 4,
        "dropout": 0.0,
        "use_causal_mask": False,
        "mlp": False,
    }

    # Training configuration
    train_config = {
        "output_path": "denoiser.pt",
        "epochs": 1000,
        "batch_size": 16,
        "dataset_size": 2000,
        "lr": 2e-3,
        "eval_every": 50,
        "eval_samples": 100,
        "lr_schedule": "step",  # Options: "constant", "cosine", "linear", "step"
        "lr_warmup_epochs": 10,
        "lr_min_factor": 0.01,
        "lr_step_size": 30,  # For step schedule: epochs per step
        "lr_gamma": 0.1**0.1,  # For step schedule: multiplicative factor
        "loss_type": "mask_dsnr",  # Options: "simple", "vlb", "mask_dsnr"
        "ema_decay": 0.999,
    }

    # Diffusion configuration
    diffusion_config = {
        "window": 1,
        "sampling_steps": 1000,
    }

    generator_config = {
        "generator_class": "LogisticMapPermutation",
        "length": length,
        "tolerance": 1e-3,
        "permutation": list(range(length)),
    }


@typed
def combine_gaussians(
    mu_prior: Float[TT, "..."],
    sigma_prior: Float[TT, "..."],
    mu_likelihood: Float[TT, "..."],
    sigma_likelihood: Float[TT, "..."],
) -> tuple[Float[TT, "..."], Float[TT, "..."]]:
    precision_prior = 1.0 / sigma_prior
    precision_likelihood = 1.0 / sigma_likelihood
    precision_combined = precision_prior + precision_likelihood
    sigma_combined = 1.0 / precision_combined
    mu_combined = sigma_combined * (
        precision_prior * mu_prior + precision_likelihood * mu_likelihood
    )
    return mu_combined, sigma_combined


class EMA:
    @typed
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @typed
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Denoiser(nn.Module):
    @typed
    def __init__(
        self,
        N: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.0,
        use_causal_mask: bool = False,
        mlp: bool = False,
    ):
        super().__init__()
        if mlp:
            raise NotImplementedError("MLP not implemented")
        else:
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
    ) -> Float[TT, "batch seq_len"]:
        dv = noisy_seq.device
        x = torch.stack(
            [
                noisy_seq,
                torch.sqrt(signal_var).to(device=dv),
                torch.sqrt(1 - signal_var).to(device=dv),
            ],
            dim=-1,
        )  # [batch, seq_len, 3]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding
        if self.use_causal_mask:
            x = self.transformer(x, mask=self.causal_mask)  # [batch, seq_len, d_model]
        else:
            x = self.transformer(x)  # [batch, seq_len, d_model]
        return self.output_proj(x).squeeze(-1)

    @typed
    def get_score(
        self,
        noisy_seq: Float[TT, "batch seq_len"],
        signal_var: Float[TT, "batch seq_len"],
    ) -> Float[TT, "batch seq_len"]:
        x0_hat = self(noisy_seq, signal_var)
        diff = noisy_seq - signal_var.sqrt() * x0_hat
        return -(1 / (1 - signal_var + 1e-10)) * diff


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
        xs[:, it] = mu + sigma.sqrt() * torch.randn_like(x_t)
    return xs.flip(dims=[1])


@typed
def velocity_field(
    model: Denoiser,
    schedule: Schedule,
    x_t: Float[TT, "batch seq_len"],
    t: Float[TT, "batch"],
) -> Float[TT, "batch seq_len"]:
    signal_var = schedule.signal_var(t)
    assert not signal_var.isnan().any(), f"signal_var is nan"
    beta = schedule.beta(t)
    assert not beta.isnan().any(), f"beta is nan"
    score = model.get_score(x_t, signal_var)
    assert not score.isnan().any(), f"score is nan"
    return -0.5 * beta * (x_t + score)


@typed
def hutchinson_trace(
    model: Denoiser,
    schedule: Schedule,
    x_t: Float[TT, "batch seq_len"],
    t: Float[TT, "batch"],
) -> Float[TT, "batch"]:
    batch_size, seq_len = x_t.shape
    with torch.set_grad_enabled(True):
        x_t.requires_grad_(True)
        v = velocity_field(model, schedule, x_t, t)
        # Spherial uniform random vector
        z = torch.randn_like(x_t)
        z /= z.norm(dim=1, keepdim=True)
        # Compute vector-Jacobian product via autograd
        vjp = torch.autograd.grad(
            outputs=v,
            inputs=x_t,
            grad_outputs=z,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        assert vjp.shape == (batch_size, seq_len)
        # Hutchinson trace estimate
        trace_estimate = (vjp * z).sum(dim=1)
        assert trace_estimate.shape == (batch_size,)
        return trace_estimate


@typed
@ex.capture
def ode_sampling(
    model: Denoiser,
    x_1: Float[TT, "batch seq_len"],
    schedule: Schedule,
    diffusion_config: dict,
    compute_likelihood: bool = True,
) -> (
    tuple[Float[TT, "batch n_steps seq_len"], Float[TT, "batch"]]
    | Float[TT, "batch n_steps seq_len"]
):
    """
    ODE:
    `dx/dt = f(t) * xt - 1/2 * g^2(t) * model.get_score(xt, t)`
    where
    `f(t) = schedule.drift_term(t)`
    `g^2(t) = schedule.diffusion_term(t)`
    """
    device = x_1.device
    n_steps = diffusion_config["sampling_steps"]
    batch_size, seq_len = x_1.shape
    xs = torch.zeros(batch_size, n_steps + 1, seq_len, device=device)
    ts = torch.arange(n_steps + 1, device=device, dtype=torch.float64) / n_steps
    ts = ts.repeat(batch_size, 1)
    xs[:, n_steps] = x_1
    # start with log PDF of multivariate standard normal at x_1
    likelihood = (
        -0.5 * (seq_len * torch.log(torch.tensor(2 * torch.pi)) + (x_1**2).sum(dim=1))
    ).to(
        device=device,
        dtype=torch.float64,
    )

    for it in reversed(range(n_steps)):
        x_t = xs[:, it + 1]
        t = ts[:, it]
        dt = ts[:, it + 1] - t
        v = velocity_field(model, schedule, x_t, t)
        xs[:, it] = x_t + v * dt[:, None]
        if compute_likelihood:
            trace = hutchinson_trace(model, schedule, x_t, t)
            addition = trace * dt
            if it % 20 == 0:
                logger.debug(f"{it}: {addition.mean():.6f}")
            if not addition.isnan().any():
                likelihood += addition
            else:
                logger.warning(f"likelihood is nan at {it}")
    if compute_likelihood:
        logger.info(f"likelihood: {likelihood.mean():.6f}")
        return xs.flip(dims=[1]), likelihood
    else:
        return xs.flip(dims=[1])


@ex.capture
@typed
def setup_model(_run, model_config: dict, train_config: dict) -> tuple[nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = Denoiser(
        N=model_config["seq_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"],
        use_causal_mask=model_config["use_causal_mask"],
        mlp=model_config["mlp"],
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
    device: torch.device | str | None = None,
) -> tuple[DataGenerator, Float[TT, "batch seq_len"], Schedule, DataLoader]:
    generator_class = globals()[generator_config["generator_class"]]
    generator = generator_class(**generator_config)
    while len(generator) < train_config["dataset_size"]:
        generator.sample(10)
    # generator.append_to_save()
    clean_data = generator.data[: train_config["dataset_size"]]
    w = torch.tensor(diffusion_config["window"], dtype=torch.float64)
    if device is not None:
        w = w.to(device)
        clean_data = clean_data.to(device)
    schedule = Schedule(
        N=model_config["seq_len"],
        w=w,
    )
    if device is not None:
        schedule = schedule.to(device)
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
        Float[TT, "batch seq_len"],  # snr
        Float[TT, "seq_len"],  # timestep
    ],
    opt: torch.optim.Optimizer,
    device: torch.device | str,
    train_config: dict,
) -> float:
    xt, signal_var, dsnr_dt, x0, snr, timestep = batch
    xt, signal_var, dsnr_dt, x0, snr = (
        xt.to(device),
        signal_var.to(device),
        dsnr_dt.to(device),
        x0.to(device),
        snr.to(device),
    )
    assert not signal_var.isnan().any(), f"signal_var is nan"
    x0_hat = model(xt, signal_var)
    assert not x0_hat.isnan().any(), f"x0_hat is nan"
    x0_errors = (x0_hat - x0).square()

    assert (
        dsnr_dt.shape == x0_errors.shape
    ), f"dsnr_dt.shape = {dsnr_dt.shape}, x0_errors.shape = {x0_errors.shape}"
    loss_type = train_config["loss_type"]
    if loss_type == "simple":
        losses = x0_errors.sum(dim=-1)
    elif loss_type == "vlb":
        losses = (dsnr_dt * x0_errors).sum(dim=-1)
    elif loss_type == "mask_dsnr":
        nonzero_mask = dsnr_dt.abs() > 1e-18
        losses = (nonzero_mask * x0_errors).sum(dim=-1)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    assert losses.shape == (len(xt),), f"losses.shape = {losses.shape}"
    loss = losses.mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Use VLB for logging
    loss = (dsnr_dt * x0_errors).sum(dim=-1).mean()
    return loss.item()


@ex.capture
@typed
def save_model(
    model: nn.Module,
    train_config: dict,
    device: torch.device | str,
    is_ema: bool = False,
) -> None:
    model_path = train_config["output_path"]
    if is_ema:
        # Save EMA model with different filename
        model_path = model_path.replace(".pt", "_ema.pt")
    model.cpu()
    torch.save(model.state_dict(), model_path)
    model.to(device)


@ex.capture
def train_denoiser(_run, model_config, train_config, diffusion_config):
    """Train the denoising model"""
    model, device = setup_model()
    _generator, _clean_data, _schedule, train_loader = get_dataset(
        inference=False, device=device
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_config["lr"],  # betas=(0.95, 0.999)
    )

    # Initialize EMA model tracker
    ema = EMA(model, decay=train_config["ema_decay"])

    scheduler = setup_lr_scheduler(optimizer=opt)

    losses = []
    for epoch in range(train_config["epochs"]):
        epoch_losses = []

        for batch in train_loader:
            loss = train_batch(model=model, batch=tuple(batch), opt=opt, device=device)
            epoch_losses.append(loss)
            # Update EMA parameters after each batch
            ema.update()

        # Compute epoch & avg loss
        current_loss = np.mean(epoch_losses)
        losses.append(current_loss)
        half = len(losses) // 2
        avg_loss = sum(losses[half:]) / len(losses[half:])

        # Get current learning rate
        current_lr = opt.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}: loss = {current_loss:.6f} (avg={avg_loss:.6f}, lr={current_lr:.6f})"
        )

        # Log metrics to Sacred
        _run.log_scalar("loss", current_loss, epoch)
        _run.log_scalar("avg_loss", avg_loss, epoch)
        _run.log_scalar("lr", current_lr, epoch)

        # Step the learning rate scheduler
        scheduler.step()

        # Save model & evaluate
        if (epoch + 1) % train_config["eval_every"] == 0:
            # Use EMA model for evaluation
            ema.apply_shadow()
            save_model(model=model, device=device, is_ema=True)
            evaluate(
                model_path=train_config["output_path"], epoch_number=epoch, use_ema=True
            )
            ema.restore()

    # Save both the regular model and EMA model
    save_model(model=model, device=device)
    ema.apply_shadow()
    save_model(model=model, device=device, is_ema=True)
    ema.restore()


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
    diffusion_config: dict,
    show_mnist: bool = False,
) -> None:
    n_steps = diffusion_config["sampling_steps"]

    if show_mnist:
        # Create animation for MNIST-like 2D data
        fig = plt.figure(figsize=(8, 8))

        def update(frame):
            plt.clf()
            current = samples[: min(9, samples.shape[0]), frame]
            # Reshape to 10x10 images if the sequence length is 100
            images = current.reshape(-1, 1, 10, 10)
            grid = make_grid(images, nrow=3)
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f"Denoising Steps (Step {frame + 1}/{n_steps})")
            plt.axis("off")

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_steps,
            interval=50,
            blit=False,
        )
    else:
        # Original 1D animation code
        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()

        def update(frame):
            ax.clear()
            for i in range(frame + 1):
                alpha = 1.5 ** (i - frame)
                ax.plot(samples[0, i].cpu(), color="blue", lw=0.5, alpha=alpha)
            current = samples[:1, frame]
            assert current.shape == (1, model_config["seq_len"])
            losses = generator.losses_per_clause(current)[0].detach().cpu().numpy()
            losses_str = " ".join(f"{loss:.3f}" for loss in losses)
            ax.set_title(
                f"Denoising Steps (Step {frame + 1}/{n_steps})"
                f"\nLosses: {losses_str}"
            )
            ax.axhline(y=0, color="black", lw=0.5)
            ax.axhline(y=1, color="black", lw=0.5)
            ax.set_ylim(-1, 2)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_steps,
            interval=40,
            blit=False,
        )

    anim.save(output_path, writer="pillow")
    plt.close()
    logger.info(f"Animation saved to {output_path}")


@ex.capture
def animated_sample(
    model_path: str,
    diffusion_config: dict,
    train_config: dict,
    output_path: str = "denoising_animation.gif",
    show_mnist: bool = False,
    use_ode: bool = False,
    use_ema: bool = True,
):
    """Sample from a trained model"""
    if use_ema and not model_path.endswith("_ema.pt"):
        model_path = model_path.replace(".pt", "_ema.pt")

    model, device = load_model(model_path=model_path)
    generator, clean_data, schedule, _train_loader = get_dataset(
        inference=True, device=device
    )

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    x0 = clean_data[:1].to(device)
    assert x0.dtype == torch.float64, f"x0 should be float64, got {x0.dtype}"

    schedule = schedule.to(device)
    assert isinstance(
        schedule, Schedule
    ), f"schedule should be a Schedule, got {type(schedule)}"
    times = torch.ones((1,), device=device)
    signal_var = schedule.signal_var(times)
    xt = x0 * torch.sqrt(signal_var) + torch.randn_like(x0) * torch.sqrt(1 - signal_var)
    assert xt.dtype == torch.float64, f"xt should be float64, got {xt.dtype}"

    with torch.no_grad():
        if use_ode:
            samples, _likelihood = ode_sampling(model, xt, schedule)
        else:
            samples = get_samples(model, xt, schedule)
        assert (
            samples.dtype == torch.float64
        ), f"samples should be float64, got {samples.dtype}"
        create_animation(
            samples, schedule, output_path, generator, show_mnist=show_mnist
        )


@typed
def show_1d_samples(
    final_samples: Float[TT, "batch seq_len"],
    final_losses: Float[TT, "batch"],
    n_samples: int,
    epoch_number: int | None = None,
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    for i in range(n_samples):
        ax1.plot(final_samples[i].cpu().numpy(), alpha=0.2, color="blue", linewidth=0.5)
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
    first_elements = final_samples[:, 0].cpu().numpy()
    ax2.hist(first_elements, bins=20, alpha=0.7, color="blue")
    ax2.set_xlim(0, 1)
    ax2.set_title("Histogram of First Element Values")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("samples.png", dpi=300)
    plt.close()
    run["samples"].append(File("samples.png"))


@typed
def show_mnist_samples(
    final_samples: Float[TT, "batch seq_len"],
    final_losses: Float[TT, "batch"],
    n_samples: int,
    epoch_number: int | None = None,
    mid_samples: Float[TT, "batch seq_len"] | None = None,
):
    # Convert final_samples to numpy and reshape to 10x10 images
    # Then use torchvision.utils.make_grid to create a grid of the images
    # Grid should be 3x3
    # Save the grid to a file
    images = final_samples[:9].reshape(-1, 1, 10, 10)
    grid = make_grid(images, nrow=3)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Final Samples")
    plt.savefig("samples.png", dpi=300)
    plt.close()

    # Upload to Neptune
    run["samples"].append(File("samples.png"))

    # If mid_samples are provided, create a similar visualization
    if mid_samples is not None:
        mid_images = mid_samples[:9].reshape(-1, 1, 10, 10)
        mid_grid = make_grid(mid_images, nrow=3)
        plt.figure(figsize=(8, 8))
        plt.imshow(mid_grid.permute(1, 2, 0))
        plt.title("Half-Denoised Samples")
        plt.savefig("partial_samples.png", dpi=300)
        plt.close()

        # Upload to Neptune
        run["partial_samples"].append(File("partial_samples.png"))


@ex.capture
def evaluate(
    _run,
    model_config: dict,
    train_config: dict,
    model_path: str = "denoiser.pt",
    epoch_number: int | None = None,
    use_ema: bool = False,
    show_mnist: bool = False,
):
    """Evaluate the model"""
    n_samples = train_config["eval_samples"]

    # If evaluating with EMA model directly and not during training
    if use_ema and not model_path.endswith("_ema.pt"):
        model_path = model_path.replace(".pt", "_ema.pt")

    model, device = load_model(model_path=model_path)
    generator, clean_data, schedule, _train_loader = get_dataset(
        inference=True, device=device
    )

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    schedule = schedule.to(device)
    with torch.no_grad():
        x_1 = torch.randn(
            (n_samples, model_config["seq_len"]), dtype=torch.float64, device=device
        )
        assert x_1.dtype == torch.float64, f"xt should be float64, got {x_1.dtype}"

        samples, loglikelihood = ode_sampling(
            model, x_1, schedule, compute_likelihood=True
        )
        nll = -loglikelihood
        mean_nll = nll.mean().item()
        median_nll = nll.median().item()
        q25_nll = nll.quantile(0.25).item()
        q75_nll = nll.quantile(0.75).item()
        if epoch_number is not None:
            _run.log_scalar("NLL_mean", mean_nll, epoch_number)
            _run.log_scalar("NLL_median", median_nll, epoch_number)
            _run.log_scalar("NLL_q25", q25_nll, epoch_number)
            _run.log_scalar("NLL_q75", q75_nll, epoch_number)

        logger.info(
            f"NLL stats: mean={mean_nll:.3f}, median={median_nll:.3f}, q25={q25_nll:.3f}, q75={q75_nll:.3f}"
        )

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

            if step == 0 or step == n_steps - 1 or step == n_steps // 2:
                q25 = losses.quantile(0.25).item()
                q50 = losses.quantile(0.50).item()
                q75 = losses.quantile(0.75).item()
                print(
                    f"Step {step}: {losses.mean():.3f} (q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f})"
                )
        _run.log_scalar("q25", q25, epoch_number)
        _run.log_scalar("q50", q50, epoch_number)
        _run.log_scalar("q75", q75, epoch_number)

        # Save distribution of final samples
        final_samples = samples[:, -1, :].cpu()  # [n_samples, seq_len]
        final_losses = generator.loss(final_samples)

        # Get mid-denoised samples (from middle step)
        mid_step = n_steps // 2
        mid_samples = samples[:, mid_step, :].cpu()  # [n_samples, seq_len]

        if show_mnist:
            show_mnist_samples(
                final_samples, final_losses, n_samples, epoch_number, mid_samples
            )
        else:
            show_1d_samples(final_samples, final_losses, n_samples, epoch_number)

        return final_samples, final_losses


@ex.capture
def sample_distribution(
    _run,
    model_path: str,
    diffusion_config: dict,
    train_config: dict,
    model_config: dict,
    n_samples: int = 100,
    use_ema: bool = True,
):
    """Generate multiple samples and visualize their distribution.

    This function:
    1. Generates n_samples from the diffusion model
    2. Plots all samples as line plots to visualize the distribution
    3. Creates a histogram of the first element values across all samples
    """
    if use_ema and not model_path.endswith("_ema.pt"):
        model_path = model_path.replace(".pt", "_ema.pt")

    model, device = load_model(model_path=model_path)
    generator, clean_data, schedule, _train_loader = get_dataset(
        inference=True, device=device
    )

    # Assert that tensors are float64
    assert (
        clean_data.dtype == torch.float64
    ), f"clean_data should be float64, got {clean_data.dtype}"

    schedule = schedule.to(device)

    # Generate samples
    with torch.no_grad():
        xt = torch.randn(
            (n_samples, model_config["seq_len"]), dtype=torch.float64, device=device
        )
        assert xt.dtype == torch.float64, f"xt should be float64, got {xt.dtype}"

        # Use ODE sampling with likelihood calculation
        samples, likelihood = ode_sampling(model, xt, schedule, compute_likelihood=True)

        # Log likelihood statistics
        mean_likelihood = likelihood.mean().item()
        median_likelihood = likelihood.median().item()
        q25_likelihood = likelihood.quantile(0.25).item()
        q75_likelihood = likelihood.quantile(0.75).item()

        logger.info(
            f"Distribution sample likelihood stats: mean={mean_likelihood:.3f}, median={median_likelihood:.3f}"
        )

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
            f"Distribution of {n_samples} Samples\nMean Loss: {mean_loss:.3f}, Median: {q50:.3f}\n"
            f"Mean Likelihood: {mean_likelihood:.3f}"
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


@ex.capture
@typed
def setup_lr_scheduler(optimizer: torch.optim.Optimizer, train_config: dict) -> Any:
    """Set up learning rate scheduler based on configuration.

    Returns:
        scheduler: The PyTorch scheduler
    """
    epochs = train_config["epochs"]
    lr = train_config["lr"]
    schedule_type = train_config["lr_schedule"]
    warmup_epochs = train_config["lr_warmup_epochs"]
    min_factor = train_config["lr_min_factor"]

    # Pre-compute the full learning rate schedule for logging
    lr_values = torch.zeros(epochs)

    if schedule_type == "constant":
        # Constant learning rate after warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0,
        )

        # Fill lr_values
        for epoch in range(epochs):
            if epoch < warmup_epochs:
                lr_values[epoch] = lr * (epoch / warmup_epochs)
            else:
                lr_values[epoch] = lr

    elif schedule_type == "cosine":
        # Cosine annealing with warmup
        if warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    # Warmup phase
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-8,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    # Cosine annealing phase
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * min_factor
                    ),
                ]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * min_factor
            )
    elif schedule_type == "linear":
        if warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-8,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=min_factor,
                        total_iters=epochs - warmup_epochs,
                    ),
                ]
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=min_factor, total_iters=epochs
            )
    elif schedule_type == "step":
        step_size = train_config["lr_step_size"]
        gamma = train_config["lr_gamma"]
        if warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-8,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=step_size, gamma=gamma
                    ),
                ]
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
    else:
        raise ValueError(f"Unknown learning rate schedule: {schedule_type}")

    return scheduler


@ex.automain
def main():
    train_denoiser()
