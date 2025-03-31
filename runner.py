import os
import random
import uuid
from pprint import pprint

import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from loguru import logger
from main import ex


@typed
def get_config(
    kind: Literal["AR", "D", "UD"],
    direction: Literal["forward", "backward", "shuffled", "block_shuffle", "swaps"],
    step: int | None = None,
    window: float | int | None = None,
    sampling_steps: int | None = None,
    length: int | None = None,
    loss_type: Literal["simple", "vlb", "mask_dsnr"] = "simple",
    generator_class: Literal[
        "LogisticMapPermutation",
        "LogisticMapForward",
        "LogisticMapBackward",
        "MNIST",
        "Stochastic",
    ] = "LogisticMapPermutation",
):
    assert length is not None, "length must be provided"
    if direction == "forward":
        permutation = list(range(length))
    elif direction == "backward":
        permutation = list(range(length))[::-1]
    elif direction == "swaps":
        assert step is not None, "step must be provided for swaps"
        permutation = list(range(length))
        for i in range(0, len(permutation) - (step - 1), step):
            j = i + step - 1
            permutation[i], permutation[j] = permutation[j], permutation[i]
    elif direction == "block_shuffle":
        assert step is not None, "step must be provided for block shuffle"
        if length % step != 0:
            logger.warning(
                f"length {length} is not divisible by step {step}, padding with zeros"
            )
        # generate a single random permutation of size `step` and apply it to every consecutive block of size `step`
        np.random.seed(42)
        block_permutation = np.random.permutation(step)
        # ensure that number of inversions is about half
        while True:
            inversions = sum(
                (block_permutation[i] > block_permutation[j])
                for i in range(step)
                for j in range(i + 1, step)
            )
            max_inv = step * (step - 1) // 2
            l = int(np.ceil(0.4 * max_inv))
            r = max(l + 1, int(0.6 * max_inv))
            print(f"Inversions: {inversions} (l={l}, r={r}, max_inv={max_inv})")
            if l <= inversions <= r:
                break
            block_permutation = np.random.permutation(step)
        result = np.arange(length)
        for i in range(0, length, step):
            current_block = result[i : i + step].copy()
            result[i : i + step] = current_block[block_permutation]
        permutation = result.tolist()
        logger.info(f"Permutation: {permutation}")

    if kind == "AR":
        window = 1
    elif kind == "D":
        window = 256
    elif kind == "UD":
        assert window is not None, "window must be provided for UD"

    if generator_class == "MNIST":
        length = 100

    # ensure that models/ directory exists
    os.makedirs("models", exist_ok=True)
    output_path = f"models/{uuid.uuid4()}.pt"
    config_updates = {
        "diffusion_config": {
            "window": window,
            "sampling_steps": sampling_steps,
        },
        "train_config": {
            "output_path": output_path,
            "loss_type": loss_type,
        },
        "model_config": {
            "seq_len": length,
        },
        "generator_config": {
            "generator_class": generator_class,
            "length": length,
            "permutation": permutation,
        },
    }
    return config_updates


@typed
def run(
    kind: Literal["AR", "D", "UD"],
    direction: Literal["forward", "backward", "shuffled", "block_shuffle", "swaps"],
    step: int | None = None,
    length: int | None = None,
    window: float | int | None = None,
    sampling_steps: int | None = None,
    comment: str = "",
    loss_type: Literal["simple", "vlb", "mask_dsnr"] = "mask_dsnr",
    generator_class: Literal[
        "LogisticMapPermutation",
        "LogisticMapForward",
        "LogisticMapBackward",
        "MNIST",
        "Stochastic",
    ] = "LogisticMapPermutation",
):
    assert sampling_steps is not None, "sampling_steps must be provided"

    config_updates = get_config(
        kind=kind,
        direction=direction,
        step=step,
        length=length,
        window=window,
        sampling_steps=sampling_steps,
        loss_type=loss_type,
        generator_class=generator_class,
    )
    comment = f"w={window} step={step} l={loss_type} | {comment}"
    config_updates["comment"] = comment
    updates_items = [f"{k}={repr(v)}" for k, v in config_updates.items()]
    updates_str = " ".join([repr(item) for item in updates_items])
    cli_command = f"python main.py with {updates_str} -c '{comment}'"
    logger.info(f"Running: {cli_command}")
    os.system(cli_command)


name = "eps-2"

for rep in range(10):
    for step in [1, 2, 4, 8, 12, 24]:
        ws = [1, 2, 4, 8, 12, 16, 24, 32, 64, 128]
        for w in ws:
            run(
                kind="UD",
                direction="block_shuffle",
                step=step,
                length=48,
                window=w,
                sampling_steps=500,
                generator_class="Stochastic",
                comment=f"{name} #{rep}",
            )
