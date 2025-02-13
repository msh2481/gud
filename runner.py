import random
from pprint import pprint

from beartype import beartype as typed
from beartype.typing import Literal
from main import ex


@typed
def get_config(
    kind: Literal["AR", "D", "GUD"],
    direction: Literal["forward", "backward", "shuffled", "slightly_shuffled"],
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

    denoise_steps = None
    speed = None
    if kind == "AR":
        denoise_steps = 1
        speed = 1
    elif kind == "D":
        denoise_steps = len(permutation) + 2
        speed = 1e3
    elif kind == "GUD":
        denoise_steps = 5
        speed = 1 / denoise_steps
    config_updates = {
        "diffusion_config": {
            "denoise_steps": denoise_steps,
            "speed": 1 / denoise_steps,
        },
        "train_config": {},
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
    kind: Literal["AR", "D", "GUD"],
    direction: Literal["forward", "backward", "shuffled", "slightly_shuffled"],
    comment: str,
):
    config_updates = get_config(kind, direction)
    denoise_steps = config_updates["diffusion_config"]["denoise_steps"]
    speed = config_updates["diffusion_config"]["speed"]
    ex.run(
        config_updates=config_updates,
        meta_info={
            "comment": f"k={kind} d={direction} n={denoise_steps} s={speed} | {comment}"
        },
    )


cfg = get_config(kind="AR", direction="forward")
del cfg["generator_config"]["permutation"]
# cfg["generator_config"]["generator_class"] = "WhiteNoise"
cfg["generator_config"]["generator_class"] = "LogisticMapForward"
cfg["train_config"]["lr"] = 1e-9
pprint(cfg)
ex.run(
    config_updates=cfg,
    meta_info={"comment": "testing AR forward"},
)

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
