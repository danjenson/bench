"""Task-owned canvases and affine geometry shared by evaluation and reports."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Canvas:
    image: Image.Image
    valid: np.ndarray
    to_original: np.ndarray
    resized_size: tuple[int, int]
    offset: tuple[int, int]


def resize_short_size(size, short):
    width, height = size
    return (int(short * width / height + 0.5), short) if width >= height else (short, int(short * height / width + 0.5))


def window_boxes(height, width, side, stride):
    for y in range(math.ceil(max(height - side, 0) / stride) + 1):
        bottom = min(y * stride + side, height)
        top = max(0, bottom - side)
        for x in range(math.ceil(max(width - side, 0) / stride) + 1):
            right = min(x * stride + side, width)
            yield top, bottom, max(0, right - side), right


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
    if task["transform"] == "resize-short-sliding-window":
        resized = resize_short_size(image.size, side)
        canvas = image.resize(resized, Image.Resampling.BILINEAR)
        return Canvas(canvas, np.ones((resized[1], resized[0]), dtype=bool), np.eye(3, dtype=np.float32), resized, (0, 0))
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
    if canvas.image.width != canvas.image.height:
        return np.array(Image.fromarray(mask.astype(np.uint8)).resize(canvas.resized_size, Image.Resampling.NEAREST), dtype=np.int64)
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
