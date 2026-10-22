from __future__ import annotations

import numpy as np


def rgb_to_yiq(img_rgb: np.ndarray) -> np.ndarray:
    img_rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0
    transform = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.311],
        ],
        dtype=np.float32,
    )
    return np.tensordot(img_rgb, transform, axes=([2], [1]))


def yiq_to_rgb(img_yiq: np.ndarray) -> np.ndarray:
    inverse_transform = np.array(
        [
            [1.0, 0.956, 0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106, 1.703],
        ],
        dtype=np.float32,
    )
    img_yiq = np.asarray(img_yiq, dtype=np.float32)
    img_rgb = np.tensordot(img_yiq, inverse_transform, axes=([2], [1]))
    img_rgb = np.clip(img_rgb, 0, 1)
    return (img_rgb * 255).astype(np.uint8)
