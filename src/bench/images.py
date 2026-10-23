"""Task-owned canvases and affine geometry shared by evaluation and reports."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Canvas:
    image: Image.Image
    valid: np.ndarray
    to_original: np.ndarray
    resized_size: tuple[int, int]
    offset: tuple[int, int]


def fit_pad_geometry(size, side):
    width, height = size
    scale = min(side / width, side / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (side - resized[0]) // 2, (side - resized[1]) // 2
    matrix = np.array([[side / resized[0], 0, -left / resized[0]], [0, side / resized[1], -top / resized[1]], [0, 0, 1]], dtype=np.float32)
    return resized, (left, top), matrix


def make_canvas(image, task):
    image = image.convert("RGB")
    width, height = image.size
    side = task["canvas"]
    if task["transform"] == "resize-center-crop":
        short = task["resize_short"]
        # Pin floor rounding, matching torchvision's integer resize convention.
        resized = (short, int(height * short / width)) if width <= height else (int(width * short / height), short)
        left, top = round((resized[0] - side) / 2), round((resized[1] - side) / 2)
        canvas = image.resize(resized, Image.Resampling.BICUBIC).crop((left, top, left + side, top + side))
        matrix = np.array([[side / resized[0], 0, left / resized[0]], [0, side / resized[1], top / resized[1]], [0, 0, 1]], dtype=np.float32)
        return Canvas(canvas, np.ones((side, side), dtype=bool), matrix, resized, (-left, -top))
    resized, (left, top), matrix = fit_pad_geometry(image.size, side)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(image.resize(resized, Image.Resampling.BICUBIC), (left, top))
    valid = np.zeros((side, side), dtype=bool)
    valid[top:top + resized[1], left:left + resized[0]] = True
    return Canvas(canvas, valid, matrix, resized, (left, top))


def mask_canvas(mask, canvas):
    return fit_pad_mask(mask, canvas.image.width, canvas.resized_size, canvas.offset)


def fit_pad_mask(mask, side, resized_size, offset):
    result = np.full((side, side), 255, dtype=np.int64)
    resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize(resized_size, Image.Resampling.NEAREST), dtype=np.int64)
    left, top = offset
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result


def original_to_canvas(points, original_size, canvas):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2) / np.asarray(original_size)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ np.linalg.inv(canvas.to_original).T)[:, :2] * canvas.image.width
