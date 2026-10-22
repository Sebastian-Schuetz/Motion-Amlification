from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol

from phaseAmplify import RieszMotionAmplifier
from steerable_amplifier import SteerableMotionAmplifier


ProgressCallback = Optional[Callable[[float], None]]


class AlgorithmName(str, Enum):
    RIESZ = "Riesz"
    STEERABLE = "Steerable"


@dataclass(frozen=True)
class BaseAmplificationConfig:
    video_path: str
    output_dir: str
    sampling_rate: float
    low_cutoff: float
    high_cutoff: float
    amplification_factor: float
    scale: float
    start_frame: int
    end_frame: int
    freq_spectrum: bool


@dataclass(frozen=True)
class RieszConfig(BaseAmplificationConfig):
    levels: int
    gaussian_kernel_sd: int


@dataclass(frozen=True)
class SteerableConfig(BaseAmplificationConfig):
    pyramid_type: str
    sigma: int
    attenuate_other_frequencies: bool
    pyfftw_threads: int


class MotionAmplifier(Protocol):
    def amplify(
        self,
        config: BaseAmplificationConfig,
        update_progress: ProgressCallback = None,
    ) -> Optional[str]:
        ...


class RieszAlgorithm:
    def __init__(self) -> None:
        self._amplifier = RieszMotionAmplifier()

    def amplify(
        self,
        config: BaseAmplificationConfig,
        update_progress: ProgressCallback = None,
    ) -> Optional[str]:
        if not isinstance(config, RieszConfig):
            raise TypeError("RieszAlgorithm requires RieszConfig.")
        return self._amplifier.amplify(
            config.amplification_factor,
            config.low_cutoff,
            config.high_cutoff,
            config.sampling_rate,
            config.video_path,
            config.output_dir,
            config.levels,
            config.gaussian_kernel_sd,
            config.scale,
            update_progress,
            config.start_frame,
            config.end_frame,
            config.freq_spectrum,
        )


class SteerableAlgorithm:
    def amplify(
        self,
        config: BaseAmplificationConfig,
        update_progress: ProgressCallback = None,
    ) -> Optional[str]:
        if not isinstance(config, SteerableConfig):
            raise TypeError("SteerableAlgorithm requires SteerableConfig.")
        amplifier = SteerableMotionAmplifier(pyfftw_threads=config.pyfftw_threads)
        return amplifier.amplify(
            config.amplification_factor,
            config.low_cutoff,
            config.high_cutoff,
            config.sampling_rate,
            config.video_path,
            config.output_dir,
            config.pyramid_type,
            config.sigma,
            config.attenuate_other_frequencies,
            config.scale,
            update_progress,
            config.start_frame,
            config.end_frame,
            config.freq_spectrum,
        )


def create_algorithm(name: AlgorithmName) -> MotionAmplifier:
    if name is AlgorithmName.RIESZ:
        return RieszAlgorithm()
    if name is AlgorithmName.STEERABLE:
        return SteerableAlgorithm()
    raise ValueError(f"Unsupported algorithm: {name}")
