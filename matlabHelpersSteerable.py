from __future__ import annotations

from math import factorial

import cv2
import numpy as np
from scipy.signal import firwin

try:
    import pyfftw.interfaces.numpy_fft as fftw
except ImportError:  # pragma: no cover - depends on local environment
    fftw = np.fft


PYFFTW_THREADS = 8
_EPSILON = np.finfo(np.float32).eps


def _fft(values: np.ndarray, axis: int | None = None, threads: int = PYFFTW_THREADS) -> np.ndarray:
    try:
        if axis is None:
            return fftw.fft(values, threads=threads)
        return fftw.fft(values, axis=axis, threads=threads)
    except TypeError:
        if axis is None:
            return fftw.fft(values)
        return fftw.fft(values, axis=axis)


def _ifft(values: np.ndarray, axis: int | None = None, threads: int = PYFFTW_THREADS) -> np.ndarray:
    try:
        if axis is None:
            return fftw.ifft(values, threads=threads)
        return fftw.ifft(values, axis=axis, threads=threads)
    except TypeError:
        if axis is None:
            return fftw.ifft(values)
        return fftw.ifft(values, axis=axis)


def maxSCFpyrHt(h: int, w: int) -> int:
    return int(np.floor(np.log2(min(h, w))) - 2)


def getPolarGrid(dimension: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    center = (
        np.ceil((dimension[0] + 0.5) / 2).astype(int),
        np.ceil((dimension[1] + 0.5) / 2).astype(int),
    )
    y = (np.arange(1, dimension[0] + 1) - center[0]) / (dimension[0] / 2)
    x = (np.arange(1, dimension[1] + 1) - center[1]) / (dimension[1] / 2)
    xramp, yramp = np.meshgrid(x, y)
    angle = np.arctan2(yramp, xramp)
    rad = np.sqrt(xramp**2 + yramp**2)
    rad[center[0] - 1, center[1] - 1] = rad[center[0] - 1, max(center[1] - 2, 0)]
    return angle, rad


def getRadialMaskPair(
    r: float,
    rad: np.ndarray,
    twidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    log_rad = np.log2(rad) - np.log2(r)
    himask = np.clip(log_rad, -twidth, 0)
    himask = np.abs(np.cos(himask * np.pi / (2 * twidth)))
    lomask = np.sqrt(1 - himask**2)
    return himask, lomask


def getAngleMask(b: int, orientations: int, angle: np.ndarray) -> np.ndarray:
    order = orientations - 1
    const = (
        (2 ** (2 * order))
        * (factorial(order) ** 2)
        / (orientations * factorial(2 * order))
    )
    angle_shifted = np.mod(np.pi + angle - np.pi * (b - 1) / orientations, 2 * np.pi) - np.pi
    return (
        2
        * np.sqrt(const)
        * np.cos(angle_shifted) ** order
        * (np.abs(angle_shifted) < (np.pi / 2))
    )


def getFilters(
    dimension: tuple[int, int],
    r_vals: np.ndarray,
    orientations: int,
    twidth: float = 1.0,
) -> list[np.ndarray]:
    angle, rad = getPolarGrid(dimension)
    angle_masks = [
        getAngleMask(j, orientations, angle) * 0.5
        for j in range(1, orientations + 1)
    ]

    filters: list[np.ndarray] = []
    himask, lomask_prev = getRadialMaskPair(r_vals[0], rad, twidth)
    filters.append(himask.astype(np.float32))

    for k in range(1, len(r_vals)):
        himask, lomask = getRadialMaskPair(r_vals[k], rad, twidth)
        rad_mask = himask * lomask_prev
        filters.extend((rad_mask * mask).astype(np.float32) for mask in angle_masks)
        lomask_prev = lomask

    filters.append(lomask_prev.astype(np.float32))
    return filters


def getIDXfromFilter(filter_kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    above_zero = filter_kernel > 1e-10
    dim1 = np.logical_or(np.sum(above_zero, axis=1) > 0, (np.sum(above_zero, axis=1) > 0)[::-1])
    dim2 = np.logical_or(np.sum(above_zero, axis=0) > 0, (np.sum(above_zero, axis=0) > 0)[::-1])

    idx1_all = np.arange(filter_kernel.shape[0])
    idx2_all = np.arange(filter_kernel.shape[1])
    idx1 = idx1_all[dim1]
    idx2 = idx2_all[dim2]
    return np.arange(idx1.min(), idx1.max() + 1), np.arange(idx2.min(), idx2.max() + 1)


def getFilterIDX(
    filters: list[np.ndarray],
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    cropped_filters = []
    filter_indices = []

    for filter_kernel in filters:
        rows_idx, cols_idx = getIDXfromFilter(filter_kernel)
        filter_indices.append((rows_idx, cols_idx))
        cropped_filters.append(filter_kernel[np.ix_(rows_idx, cols_idx)].astype(np.float32))

    return cropped_filters, filter_indices


def FIRWindowBP(delta: np.ndarray, fl: float, fh: float) -> np.ndarray:
    rows, _cols, frames = delta.shape
    output = delta.astype(np.float32)
    coefficients = firwin(frames, [fl * 2, fh * 2], pass_zero=False)
    transfer_function = _fft(np.fft.ifftshift(coefficients), threads=PYFFTW_THREADS)
    transfer_function = transfer_function.reshape(1, 1, -1)

    batches = 20
    batch_size = int(np.ceil(rows / batches))
    for batch_index in range(batches):
        idx_start = batch_index * batch_size
        idx_end = min((batch_index + 1) * batch_size, rows)
        if idx_start >= idx_end:
            continue
        batch = slice(idx_start, idx_end)
        frequency_domain = _fft(delta[batch, :, :], axis=2, threads=PYFFTW_THREADS)
        frequency_domain *= transfer_function
        output[batch, :, :] = np.real(
            _ifft(frequency_domain, axis=2, threads=PYFFTW_THREADS)
        ).astype(np.float32)

    return output


def fast_gaussian_blur_wrap(img: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    pad = kernel_size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="wrap")
    blurred = cv2.GaussianBlur(
        padded,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    return blurred[pad:-pad, pad:-pad]


def AmplitudeWeightedBlur(
    in_array: np.ndarray,
    weight: np.ndarray,
    sigma: float,
) -> np.ndarray:
    in_array = in_array.astype(np.float32, copy=False)
    weight = weight.astype(np.float32, copy=False)

    if sigma == 0:
        return in_array

    weight = weight + _EPSILON
    weighted_input = in_array * weight
    blurred_input = fast_gaussian_blur_wrap(weighted_input, sigma)
    blurred_weight = fast_gaussian_blur_wrap(weight, sigma)
    return (blurred_input / blurred_weight).astype(np.float32)
