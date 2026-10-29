"""AXIS Franka (DROID 8-D) input and output transforms for pi0.5.

state[8] = 7 arm joint positions (rad) + gripper closedness in [0, 1] (the DROID 8-D layout).
action[8] = 7 arm joint velocity commands + gripper closedness, the layout the AXIS benchmark
executes. Images arrive repacked to base_0_rgb / left_wrist_0_rgb by the data config's repack
step.

State and actions are NOT padded here. The pipeline order is
    data_transforms.inputs -> Normalize -> model_transforms.inputs
and `PadStatesAndActions` lives in model_transforms, after `Normalize`; padding here would feed the
quantile normaliser 24 dims of zeros that map to a constant -1.0.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms as _transforms


def _to_hwc_uint8(img) -> np.ndarray:
    """LeRobot v3.0 yields images as torch CHW float[0,1]; openpi wants numpy HWC uint8."""
    arr = np.asarray(img)  # torch CPU tensor -> numpy via __array__
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    """Crop the WIDER axis to the shorter one, centred. A no-op on an already-square frame.

    A 16:9 frame (640x360, both cameras of the AXIS corpora) resized straight to the model's
    224x224 input would be squashed into 1:1; cropping it to its centre square keeps the
    proportions of what the camera sees.

    CROPPING IS DECLARED, NEVER INFERRED -- the caller passes `center_crop`; a transform that
    silently activated on frame shape would change what a corpus means without anything in the
    config recording it.
    """
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return arr[top : top + side, left : left + side]


@dataclasses.dataclass(frozen=True)
class AxisFrankaInputs(_transforms.DataTransformFn):
    """Repack AXIS-Franka rows into the pi0.5 input layout.

    Deliberately does NOT pad state/actions to the model action dim -- see the module
    docstring. `PadStatesAndActions` in `model_transforms` does that, after `Normalize`.
    """

    # Crop both cameras to their centre square before the downstream resize to 224: declared for
    # 16:9 frames, a no-op on square ones. Off by default, so a config crops only when it says so.
    center_crop: bool = False

    def __call__(self, data: dict) -> dict:
        # Pass 8-D state through unpadded: Normalize runs next, and padding here would
        # feed it 24 dims of zeros that quantile-normalize to -1.0 (see module docstring).
        state = np.asarray(data["state"], dtype=np.float32)
        # WIDTH IS A CONTRACT: the droid8 layout is 8-D. An unconverted corpus has a 9-D state
        # (7 joint angles + 2 finger widths); it would train happily on absolute joint positions
        # and slice the output to [:8], keeping one finger width -- nothing errors, the loss falls,
        # and the policy is meaningless.
        if state.shape[-1] != 8:
            raise ValueError(
                f"state is {state.shape[-1]}-D; the AXIS Franka layout is 8-D (DROID: 7 joint "
                f"positions + gripper closedness). A 9-D state is the raw corpus layout (7 joint "
                f"angles + 2 finger widths): train on the droid8 conversion of the corpus."
            )
        base = _to_hwc_uint8(data["base_0_rgb"])
        raw_wrist = data.get("left_wrist_0_rgb")
        has_wrist = raw_wrist is not None
        wrist = _to_hwc_uint8(raw_wrist) if has_wrist else np.zeros_like(base)
        if self.center_crop:
            # BOTH cameras, identically. The wrist is 16:9 too, and cropping only one would
            # give the two views different effective fields of view.
            base = _center_crop_square(base)
            wrist = _center_crop_square(wrist)
        # openpi wants separate `image` / `image_mask` dicts with the model's 3 camera
        # slots; pad the unused right wrist with zeros (masked off for the pi0.5 flow).
        out = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            # Unpadded, for the same reason as `state`. AxisFrankaOutputs slices the
            # model's 32-D output back to 8 on the way out.
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class AxisFrankaOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :8]}


QUALITY_MARKER = "\nQuality: "


@dataclasses.dataclass(frozen=True)
class ServeQualityTag(_transforms.DataTransformFn):
    """Serve-time quality tag: append "\\nQuality: {q}" to the prompt (the conditional branch).

    The unconditional branch of a guided policy is the same transform with `drop_all=True`, which
    leaves the prompt bare, so the two guidance inputs differ in the prompt tag and nothing else. A
    sample without a prompt is passed through unchanged (both branches then agree); a prompt that
    already carries a tag is refused, since tagging it again would condition on a string no run
    trained on.
    """

    q: int = 5
    drop_all: bool = False

    def __post_init__(self) -> None:
        if not 1 <= int(self.q) <= 5:
            raise ValueError(f"quality tag {self.q} is not a bin in [1, 5]")

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data or self.drop_all:
            return data
        prompt = str(data["prompt"])
        if QUALITY_MARKER in prompt:
            raise ValueError(f"prompt {prompt[:120]!r} already carries a quality tag")
        return {**data, "prompt": f"{prompt}{QUALITY_MARKER}{int(self.q)}"}


# The benchmark protocol's optional handshake keys (qvla.policy_server/1). The client sends raw
# frames, 8-D state and reads 8 action columns; these say what this server does with them.
# STATE_ORDER is axis.bench.protocol.STATE_ORDER, copied: this line imports nothing from AXIS-QVLA.
STATE_ORDER = ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper")
ACTION_FPS = 15
ACTION_DIM_OUT = 8


def serve_contract(*, image_center_crop: bool, quality_tag: int | None, action_units: str = "rad_s") -> dict:
    """The optional flat handshake keys an AXIS Franka server publishes, from its own config.
    `image_geometry` is the transform this server applies to each raw frame it receives:
    `center_crop224` is a centre square crop then a resize to 224 (`AxisFrankaInputs(center_crop=True)`
    then `ResizeImages(224, 224)`), `letterbox224` the resize-with-pad alone; actions are in the
    config's action units (`rad_s`, or `droid_normalized_joint_velocity_command` for the FR3 runs) at
    15 Hz with gripper closedness in column 8; `prompt_suffix` is what the
    server appends to every prompt."""
    if not isinstance(image_center_crop, bool):
        raise TypeError(f"image_center_crop is {image_center_crop!r}; the geometry is declared, so it must be a bool")
    geometry = "center_crop224" if image_center_crop else "letterbox224"
    return {
        "image_geometry": {"base_0_rgb": geometry, "left_wrist_0_rgb": geometry},
        "state_order": list(STATE_ORDER),
        "action_units": action_units,
        "action_fps": ACTION_FPS,
        "action_dim_out": ACTION_DIM_OUT,
        "prompt_suffix": "" if quality_tag is None else f"{QUALITY_MARKER}{int(quality_tag)}",
    }
