import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def uniform_schedule(seq_len: int) -> t.Tensor:
    """Returns signal variance (pi) values ~ U[0,1] for each position"""
    return t.rand(seq_len)


class DenoiserConv(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(2, 32, kernel_size=window_size, padding="same")
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, noisy_seq: t.Tensor, signal_var: t.Tensor) -> t.Tensor:
        """
        Args:
            noisy_seq: [batch_size, seq_len]
            signal_var: [batch_size, seq_len] - remaining signal variance per position
        Returns:
            [batch_size, seq_len] predicted noise
        """
        # Stack noisy_seq and signal_var as channels
        x = t.stack([noisy_seq, signal_var], dim=1)  # [batch, 2, seq_len]
        x = self.conv(x)
        x = self.mlp(x)
        return x.squeeze(1)


class DiffusionDataset(Dataset):
    def __init__(self, clean_data: t.Tensor, schedule_fn):
        """
        Args:
            clean_data: [num_samples, seq_len] tensor of clean sequences
            schedule_fn: function(seq_len) -> [seq_len] tensor of signal variances
        """
        self.clean_data = clean_data
        self.schedule_fn = schedule_fn

    def __getitem__(self, idx):
        x0 = self.clean_data[idx]  # [seq_len]
        signal_var = self.schedule_fn(len(x0))
        noise = t.randn_like(x0)
        xt = t.sqrt(signal_var) * x0 + t.sqrt(1 - signal_var) * noise
        return xt, signal_var, noise

    def __len__(self):
        return len(self.clean_data)


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


def main():
    # Generate synthetic dataset
    batch_size = 32
    seq_len = 100
    dataset_size = 1000
    device = "cuda" if t.cuda.is_available() else "cpu"

    clean_data = gen_dataset(dataset_size, seq_len, chaos_ratio=0.5)
    dataset = DiffusionDataset(t.from_numpy(clean_data).float(), uniform_schedule)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    model = DenoiserConv(window_size=5)
    losses = train(model, train_loader, num_epochs=10, device=device)

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()

    # Plot example predictions
    model.eval()
    with t.no_grad():
        xt, signal_var, noise = next(iter(train_loader))
        xt, signal_var = xt.to(device), signal_var.to(device)
        pred_noise = model(xt, signal_var)

        plt.figure(figsize=(15, 5))
        idx = 0  # Plot first sequence in batch
        plt.plot(xt[idx].cpu(), label="Noisy")
        plt.plot(noise[idx].cpu(), label="True Noise")
        plt.plot(pred_noise[idx].cpu(), label="Predicted Noise")
        plt.legend()
        plt.title("Example Prediction")
        plt.show()


if __name__ == "__main__":
    main()
