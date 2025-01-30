import numpy as np
import torch as t
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray as ND
from torch import Tensor as T


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


def main():
    batch_size = 10
    n = 100
    data = gen_dataset(batch_size, n, chaos_ratio=0.0)
    logger.info(f"data shape: {data.shape}")
    plt.plot(data.T, "k-", alpha=1 / batch_size, lw=1)
    plt.show()


if __name__ == "__main__":
    main()
