from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import h5py
import numpy as np

from matlabHelpersSteerable import (
    AmplitudeWeightedBlur,
    FIRWindowBP,
    getFilterIDX,
    getFilters,
    maxSCFpyrHt,
)
from utils import rgb_to_yiq, yiq_to_rgb

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw

    pyfftw.interfaces.cache.enable()
except ImportError:  # pragma: no cover - depends on local environment
    pyfftw = None
    fftw = np.fft


class SteerableMotionAmplifier:
    def __init__(self, pyfftw_threads: int = 8) -> None:
        self.pyfftw_threads = pyfftw_threads

    def amplify(
        self,
        amplification_factor: float,
        low_cutoff: float,
        high_cutoff: float,
        sampling_rate: float,
        video_path: str,
        output_dir: str,
        pyr_type: str = "halfOctave",
        sigma: int = 3,
        attenuate_other_frequencies: bool = False,
        scale: float = 1.0,
        update_progress: Optional[Callable[[float], None]] = None,
        start_frame: int = 0,
        end_frame: int = -1,
        freq_spectrum: bool = False,
    ) -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)
        frames, fps = self._read_frames(video_path, scale, start_frame, end_frame)
        if not frames:
            raise RuntimeError("Cannot read frames from video file.")

        frame_height, frame_width = frames[0].shape[:2]
        channel_count = frames[0].shape[2]
        frame_count = len(frames)

        filters = self._build_filters(frame_height, frame_width, pyr_type)
        cropped_filters, filter_indices = getFilterIDX(filters)
        num_levels = len(filters)

        write_tag = os.path.splitext(os.path.basename(video_path))[0]
        out_name = (
            f"{write_tag}-band{low_cutoff:.2f}-{high_cutoff:.2f}Hz-"
            f"sr{sampling_rate}Hz-alpha{amplification_factor}-"
            f"mp{int(attenuate_other_frequencies)}-sigma{sigma}-"
            f"{pyr_type}-scale{scale:.2f}"
        )

        video_fft = np.zeros((frame_height, frame_width, frame_count), dtype=np.complex64)
        for frame_index, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            luma = rgb_to_yiq(frame_rgb)[:, :, 0]
            video_fft[:, :, frame_index] = np.fft.fftshift(
                self._fft2(luma).astype(np.complex64)
            )

        magnified_luma_fft = np.zeros_like(video_fft)
        spectrum_file, spectrum_datasets = self._open_spectrum_file(
            output_dir,
            out_name,
            num_levels,
            freq_spectrum,
        )

        try:
            for level in range(1, num_levels - 1):
                if update_progress is not None:
                    update_progress(level / max(num_levels - 1, 1))

                reference = self._build_level(
                    video_fft[:, :, 0],
                    level,
                    cropped_filters,
                    filter_indices,
                )
                reference_phase = reference / (np.abs(reference) + np.finfo(np.float32).eps)
                reference_angle = np.angle(reference)

                level_responses = [
                    self._build_level(video_fft[:, :, frame_index], level, cropped_filters, filter_indices)
                    for frame_index in range(frame_count)
                ]
                response_angles = np.transpose(np.angle(np.asarray(level_responses)), (1, 2, 0))
                phase_delta = (
                    np.mod(np.pi + response_angles - reference_angle[:, :, np.newaxis], 2 * np.pi)
                    - np.pi
                ).astype(np.float32)
                phase_delta = FIRWindowBP(
                    phase_delta,
                    low_cutoff / sampling_rate,
                    high_cutoff / sampling_rate,
                )

                for frame_index, original_level in enumerate(level_responses):
                    phase_of_frame = phase_delta[:, :, frame_index]
                    if sigma != 0:
                        phase_of_frame = AmplitudeWeightedBlur(
                            phase_of_frame,
                            np.abs(original_level) + np.finfo(np.float32).eps,
                            sigma,
                        )

                    phase_of_frame = amplification_factor * phase_of_frame
                    if attenuate_other_frequencies:
                        transform_source = np.abs(original_level) * reference_phase
                    else:
                        transform_source = original_level

                    transformed = np.exp(1j * phase_of_frame) * transform_source
                    reconstructed_level = self._reconstruct_level(
                        transformed,
                        level,
                        cropped_filters,
                    )

                    if spectrum_datasets is not None:
                        amplitude = np.real(reconstructed_level).astype(np.float32)
                        dataset = spectrum_datasets[level - 1]
                        dataset.resize((dataset.shape[0] + 1,))
                        dataset[-1] = amplitude.flatten()
                        dataset.attrs[f"shape_{dataset.shape[0] - 1}"] = amplitude.shape

                    rows, cols = self._mesh_indices(filter_indices[level])
                    magnified_luma_fft[rows, cols, frame_index] += reconstructed_level

            residual_level = num_levels - 1
            rows, cols = self._mesh_indices(filter_indices[residual_level])
            for frame_index in range(frame_count):
                lowpass_frame = video_fft[rows, cols, frame_index] * (cropped_filters[-1] ** 2)
                magnified_luma_fft[rows, cols, frame_index] += lowpass_frame
        finally:
            if spectrum_file is not None:
                spectrum_file.close()

        self._write_output_video(
            frames,
            magnified_luma_fft,
            os.path.join(output_dir, out_name + ".avi"),
            fps,
            channel_count,
        )

        if freq_spectrum:
            return os.path.join(output_dir, out_name + "_spek.h5")
        return None

    def _build_filters(
        self,
        frame_height: int,
        frame_width: int,
        pyr_type: str,
    ) -> list[np.ndarray]:
        max_height = maxSCFpyrHt(frame_height, frame_width)
        if pyr_type == "octave":
            r_vals = 2.0 ** np.arange(0, -max_height - 1, -1)
            return getFilters((frame_height, frame_width), r_vals, 2)
        if pyr_type == "halfOctave":
            r_vals = 2.0 ** np.arange(0, -max_height - 0.5, -0.5)
            return getFilters((frame_height, frame_width), r_vals, 8, twidth=0.75)
        raise ValueError("pyr_type must be 'halfOctave' or 'octave'.")

    def _read_frames(
        self,
        video_path: str,
        scale: float,
        start_frame: int,
        end_frame: int,
    ) -> tuple[list[np.ndarray], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: list[np.ndarray] = []
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if end_frame == -1 or start_frame <= frame_index <= end_frame:
                frames.append(
                    cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                )
            frame_index += 1
        cap.release()
        return frames, fps

    def _build_level(
        self,
        image_dft: np.ndarray,
        level: int,
        cropped_filters: list[np.ndarray],
        filter_indices: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        rows, cols = filter_indices[level]
        filtered = cropped_filters[level] * image_dft[np.ix_(rows, cols)]
        return self._ifft2(np.fft.ifftshift(filtered))

    def _reconstruct_level(
        self,
        image_dft: np.ndarray,
        level: int,
        cropped_filters: list[np.ndarray],
    ) -> np.ndarray:
        return 2 * (
            cropped_filters[level]
            * np.fft.fftshift(self._fft2(image_dft))
        )

    def _fft2(self, values: np.ndarray) -> np.ndarray:
        try:
            return fftw.fft2(values, threads=self.pyfftw_threads)
        except TypeError:
            return fftw.fft2(values)

    def _ifft2(self, values: np.ndarray) -> np.ndarray:
        try:
            return fftw.ifft2(values, threads=self.pyfftw_threads)
        except TypeError:
            return fftw.ifft2(values)

    def _open_spectrum_file(
        self,
        output_dir: str,
        out_name: str,
        num_levels: int,
        enabled: bool,
    ) -> tuple[Optional[h5py.File], Optional[list[h5py.Dataset]]]:
        if not enabled:
            return None, None

        h5_file = h5py.File(os.path.join(output_dir, out_name + "_spek.h5"), "w")
        datasets = []
        data_type = h5py.special_dtype(vlen=np.dtype("float32"))
        for level in range(num_levels - 2):
            group = h5_file.create_group(f"level_{level}")
            datasets.append(
                group.create_dataset(
                    "amplitudes",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=data_type,
                    chunks=True,
                )
            )
        return h5_file, datasets

    @staticmethod
    def _mesh_indices(
        indices: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(indices[0]).ravel().astype(int)
        cols = np.asarray(indices[1]).ravel().astype(int)
        return np.meshgrid(rows, cols, indexing="ij")

    def _write_output_video(
        self,
        frames: list[np.ndarray],
        magnified_luma_fft: np.ndarray,
        output_path: str,
        fps: float,
        channel_count: int,
    ) -> None:
        frame_height, frame_width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (frame_width, frame_height),
        )

        for frame_index, frame in enumerate(frames):
            magnified_luma = np.real(
                self._ifft2(np.fft.ifftshift(magnified_luma_fft[:, :, frame_index]))
            )
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = rgb_to_yiq(frame_rgb)
            output_frame = np.zeros((frame_height, frame_width, channel_count), dtype=np.float32)
            output_frame[:, :, 0] = magnified_luma
            output_frame[:, :, 1:] = original_frame[:, :, 1:]
            output_rgb = yiq_to_rgb(output_frame)
            writer.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

        writer.release()
