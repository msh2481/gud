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
    n_steps: int | None = None,
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

    if n_steps is None:
        raise ValueError("n_steps must be provided")
    if kind == "AR":
        window = 1
    elif kind == "D":
        window = 1
    elif kind == "UD":
        assert window is not None, "window must be provided for UD"

    output_path = f"models/{uuid.uuid4()}.pt"
    config_updates = {
        "diffusion_config": {
            "window": window,
            "n_steps": n_steps,
        },
        "train_config": {
            "output_path": output_path,
        },
        "model_config": {
            "seq_len": len(permutation),
        },
        "generator_config": {
            "generator_class": "LogisticMapPermutation",
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
    n_steps: int | None = None,
    comment: str = "",
):
    if n_steps is None:
        n_steps = 60
    config_updates = get_config(
        kind=kind,
        direction=direction,
        step=step,
        window=window,
        n_steps=n_steps,
    )
    ex.run(
        config_updates=config_updates,
        meta_info={
            "comment": f"k={kind} d={direction} w={window} T={n_steps} step={step} | {comment}"
        },
    )


name = "slope-4"

for rep in range(10):
    for step in [1, 2, 4, 8]:
        run(kind="AR", direction="swaps", step=step, comment=f"{name} #{rep}")
        w_candidates = [step, step + 1, step + 2, step + 3, step + 4] + [
            12,
            14,
            16,
            20,
            24,
            28,
            32,
            64,
            128,
        ]
        for w in sorted(list(set(w_candidates))):
            run(
                kind="UD",
                direction="swaps",
                step=step,
                window=w,
                comment=f"{name} #{rep}",
            )
        run(kind="D", direction="swaps", step=step, comment=f"{name} #{rep}")
