import random
import uuid
from pprint import pprint

from beartype import beartype as typed
from beartype.typing import Literal
from loguru import logger
from main import ex


@typed
def get_config(
    kind: Literal["AR", "D", "UD"],
    direction: Literal["forward", "backward", "shuffled", "slightly_shuffled", "swaps"],
    step: int | None = None,
    window: float | int | None = None,
):
    if direction == "forward":
        permutation = list(range(20))
    elif direction == "backward":
        permutation = list(range(20))[::-1]
    elif direction == "shuffled":
        permutation = [
            8,
            18,
            7,
            17,
            16,
            11,
            12,
            5,
            4,
            6,
            3,
            0,
            2,
            14,
            15,
            1,
            9,
            19,
            13,
            10,
        ]
    elif direction == "slightly_shuffled":
        permutation = [
            2,
            1,
            3,
            0,
            6,
            5,
            4,
            7,
            11,
            8,
            9,
            10,
            12,
            15,
            14,
            13,
            19,
            16,
            18,
            17,
        ]
    elif direction == "swaps":
        assert step is not None, "step must be provided for swaps"
        permutation = list(range(20))
        for i in range(0, len(permutation) - (step - 1), step):
            j = i + step - 1
            permutation[i], permutation[j] = permutation[j], permutation[i]
    if kind == "AR":
        denoise_steps = 1
        speed = 1
    elif kind == "D":
        denoise_steps = len(permutation) + 2
        speed = 1e3
    elif kind == "UD":
        N = len(permutation)
        T = N
        speed = int((N + window) / T)
        denoise_steps = max(1, int(window / speed + 0.5))

    output_path = f"models/{uuid.uuid4()}.pt"
    config_updates = {
        "diffusion_config": {
            "denoise_steps": denoise_steps,
            "speed": speed,
        },
        "train_config": {
            "output_path": output_path,
        },
        "model_config": {
            "seq_len": len(permutation),
        },
        "generator_config": {
            "length": len(permutation),
            "permutation": permutation,
        },
    }
    return config_updates


@typed
def run(
    kind: Literal["AR", "D", "UD"],
    direction: Literal["forward", "backward", "shuffled", "slightly_shuffled", "swaps"],
    step: int | None = None,
    window: float | int | None = None,
    comment: str = "",
):
    config_updates = get_config(kind, direction, step, window)
    denoise_steps = config_updates["diffusion_config"]["denoise_steps"]
    speed = config_updates["diffusion_config"]["speed"]
    ex.run(
        config_updates=config_updates,
        meta_info={
            "comment": f"k={kind} d={direction} n={denoise_steps} s={speed:.4f} w={window} step={step} | {comment}"
        },
    )


name = "slope-1"

# run(kind="AR", direction="backward", comment=f"{name}")

for rep in range(10):
    for step in [2, 4, 6, 8, 10]:
        # run(kind="AR", direction="swaps", step=step, comment=f"{name} #{rep}")
        for w in [18, 20, 22, 24, 26, 28, 30, 32, 64]:
            run(
                kind="UD",
                direction="swaps",
                step=step,
                window=w,
                comment=f"{name} #{rep}",
            )
        # run(kind="D", direction="swaps", step=step, comment=f"{name} #{rep}")


""" 
model_config = {
    "seq_len": 20,
    "n_heads": 16,
    "n_layers": 4,
    "predict_x0": True,
}
train_config = {
    "epochs": 100,
    "batch_size": 32,
    "dataset_size": 2000,
    "lr": 1e-3,
}
diffusion_config = {
    "denoise_steps": 1,
    "speed": 1,
    "generator_class": "LogisticMapPermutation",
}
generator_config = {
    "length": 20,
    "permutation": list(range(20)),
}
"""
