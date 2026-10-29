"""Photometric augmentation of rendered frames for the real-world line's stage-1 pretraining.

Colour jitter (brightness and contrast 0.8-1.2, saturation 0.7-1.3, hue +-12/256), Gaussian blur (radius
0.1-1.0), posterize to 6 bits, sharpness x2 and grayscale, each with its own probability, applied
identically to both cameras of a sample. It runs on the raw corpus frames before the repack, so it is a
training transform only. A square third-person frame (a real, centre-cropped recording) passes untouched.
Each data-loader process draws from its own generator, seeded from fresh entropy when first used.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps

from openpi import transforms as _transforms

BASE_KEY = "observation.images.third_person"
WRIST_KEY = "observation.images.wrist"
_rng: np.random.Generator | None = None


def _generator() -> np.random.Generator:
    global _rng  # noqa: PLW0603
    if _rng is None:
        _rng = np.random.default_rng()
    return _rng


def _hwc(a: np.ndarray) -> tuple[np.ndarray, bool]:
    chw = a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3)
    return (np.moveaxis(a, 0, -1) if chw else a), chw


@dataclasses.dataclass(frozen=True)
class SimImageAug(_transforms.DataTransformFn):
    p_jitter: float = 0.5
    p_blur: float = 0.2
    p_posterize: float = 0.1
    p_sharpness: float = 0.2
    p_grayscale: float = 0.05

    def _draw(self, rng: np.random.Generator) -> dict:
        d = {}
        if rng.random() < self.p_jitter:
            d["jitter"] = (
                rng.uniform(0.8, 1.2),
                rng.uniform(0.8, 1.2),
                rng.uniform(0.7, 1.3),
                int(rng.integers(-12, 13)),
            )
        if rng.random() < self.p_blur:
            d["blur"] = float(rng.uniform(0.1, 1.0))
        d["posterize"] = rng.random() < self.p_posterize
        d["sharpness"] = rng.random() < self.p_sharpness
        d["grayscale"] = rng.random() < self.p_grayscale
        return d

    @staticmethod
    def _apply(img: np.ndarray, d: dict) -> np.ndarray:
        h, chw = _hwc(np.asarray(img))
        floating = h.dtype != np.uint8
        pil = Image.fromarray((np.clip(h, 0.0, 1.0) * 255.0).round().astype(np.uint8) if floating else h)
        if "jitter" in d:
            b, c, s, hue = d["jitter"]
            pil = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(pil).enhance(b)).enhance(c)).enhance(
                s
            )
            hsv = np.array(pil.convert("HSV"), dtype=np.int16)
            hsv[..., 0] = (hsv[..., 0] + hue) % 256
            pil = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        if "blur" in d:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=d["blur"]))
        if d["posterize"]:
            pil = ImageOps.posterize(pil, 6)
        if d["sharpness"]:
            pil = ImageEnhance.Sharpness(pil).enhance(2.0)
        if d["grayscale"]:
            pil = pil.convert("L").convert("RGB")
        out = np.asarray(pil, dtype=np.uint8)
        out = (out.astype(np.float32) / 255.0).astype(h.dtype) if floating else out
        return np.moveaxis(out, -1, 0) if chw else out

    def __call__(self, data: dict) -> dict:
        base, _ = _hwc(np.asarray(data[BASE_KEY]))
        if base.shape[0] == base.shape[1]:
            return data
        d = self._draw(_generator())
        return {**data, **{k: self._apply(data[k], d) for k in (BASE_KEY, WRIST_KEY) if k in data}}
