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
    sampling_steps: int | None = None,
    length: int = 100,
    loss_type: Literal["simple", "vlb", "mask_dsnr"] = "simple",
    generator_class: Literal[
        "LogisticMapPermutation", "LogisticMapForward", "LogisticMapBackward", "MNIST"
    ] = "LogisticMapPermutation",
):
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

    if kind == "AR":
        window = 1
    elif kind == "D":
        window = 256
    elif kind == "UD":
        assert window is not None, "window must be provided for UD"

    if generator_class == "MNIST":
        length = 100

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
    direction: Literal["forward", "backward", "shuffled", "slightly_shuffled", "swaps"],
    step: int | None = None,
    window: float | int | None = None,
    sampling_steps: int | None = None,
    comment: str = "",
    loss_type: Literal["simple", "vlb", "mask_dsnr"] = "mask_dsnr",
    generator_class: Literal[
        "LogisticMapPermutation", "LogisticMapForward", "LogisticMapBackward", "MNIST"
    ] = "LogisticMapPermutation",
):
    if sampling_steps is None:
        sampling_steps = 1000

    config_updates = get_config(
        kind=kind,
        direction=direction,
        step=step,
        window=window,
        sampling_steps=sampling_steps,
        loss_type=loss_type,
        generator_class=generator_class,
    )
    ex.run(
        config_updates=config_updates,
        meta_info={
            "comment": f"k={kind} d={direction} w={window} step={step} l={loss_type} | {comment}"
        },
    )


name = "mnist-1"

run(
    kind="D",
    direction="forward",
    generator_class="MNIST",
    sampling_steps=784,
    comment=f"{name}",
)

# for rep in range(10):
#     for step in [1, 2, 4, 8, 12]:
#         run(kind="AR", direction="swaps", step=step, comment=f"{name} #{rep}")
#         w_candidates = [step, step + 1, step + 2, step + 3, step + 4] + [
#             2,
#             4,
#             8,
#             12,
#             16,
#             24,
#             32,
#             128,
#         ]
#         for w in sorted(list(set(w_candidates))):
#             run(
#                 kind="UD",
#                 direction="swaps",
#                 step=step,
#                 window=w,
#                 comment=f"{name} #{rep}",
#             )
#         run(kind="D", direction="swaps", step=step, comment=f"{name} #{rep}")
