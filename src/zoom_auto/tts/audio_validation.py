"""Shared audio validation utilities for voice sample quality checks.

Used by both the CLI recording tool and web upload endpoints to ensure
voice samples meet quality requirements for TTS cloning.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Target audio format for TTS
TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"

# Quality thresholds
MIN_SNR_DB = 20.0
MAX_PEAK_AMPLITUDE = 0.99  # Clipping threshold (normalized -1.0 to 1.0)
MIN_RMS_AMPLITUDE = 0.01  # Minimum volume (too quiet)
MAX_RMS_AMPLITUDE = 0.9  # Maximum RMS (too loud / distorted)
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0


@dataclass
class AudioQualityReport:
    """Report on audio sample quality metrics.

    Attributes:
        is_valid: Whether the sample passes all quality checks.
        snr_db: Signal-to-noise ratio in decibels.
        peak_amplitude: Maximum absolute amplitude (0.0 to 1.0).
        rms_amplitude: Root mean square amplitude.
        duration_seconds: Duration of the audio in seconds.
        sample_rate: Sample rate of the audio.
        channels: Number of audio channels.
        has_clipping: Whether clipping was detected.
        issues: List of quality issues found.
    """

    is_valid: bool
    snr_db: float
    peak_amplitude: float
    rms_amplitude: float
    duration_seconds: float
    sample_rate: int
    channels: int
    has_clipping: bool
    issues: list[str]


def validate_audio_file(path: Path) -> AudioQualityReport:
    """Validate an audio file for TTS voice cloning quality.

    Reads the audio file and checks SNR, clipping, volume consistency,
    duration, and format.

    Args:
        path: Path to the audio file (WAV, FLAC, etc. readable by soundfile).

    Returns:
        AudioQualityReport with all quality metrics and issues.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, sample_rate = sf.read(path, dtype="float64")
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {e}") from e

    # Handle multi-channel: take the first channel for analysis
    if data.ndim > 1:
        channels = data.shape[1]
        data_mono = data[:, 0]
    else:
        channels = 1
        data_mono = data

    return validate_audio_data(data_mono, sample_rate, channels)


def validate_audio_data(
    data: np.ndarray,
    sample_rate: int,
    channels: int = 1,
) -> AudioQualityReport:
    """Validate raw audio data for TTS voice cloning quality.

    Args:
        data: Audio data as a 1D numpy array (mono, float64, normalized to -1.0..1.0).
        sample_rate: Sample rate of the audio.
        channels: Number of channels in the original audio.

    Returns:
        AudioQualityReport with all quality metrics and issues.
    """
    issues: list[str] = []

    # Duration
    duration = len(data) / sample_rate

    if duration < MIN_DURATION_SECONDS:
        issues.append(f"Too short: {duration:.1f}s (min {MIN_DURATION_SECONDS}s)")
    if duration > MAX_DURATION_SECONDS:
        issues.append(f"Too long: {duration:.1f}s (max {MAX_DURATION_SECONDS}s)")

    # Peak amplitude (clipping detection)
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    has_clipping = peak >= MAX_PEAK_AMPLITUDE

    if has_clipping:
        # Count clipping samples for severity
        clip_count = int(np.sum(np.abs(data) >= MAX_PEAK_AMPLITUDE))
        issues.append(f"Clipping detected: {clip_count} samples at max amplitude")

    # RMS amplitude (volume check)
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0

    if rms < MIN_RMS_AMPLITUDE:
        issues.append(f"Too quiet: RMS {rms:.4f} (min {MIN_RMS_AMPLITUDE})")
    if rms > MAX_RMS_AMPLITUDE:
        issues.append(f"Too loud: RMS {rms:.4f} (max {MAX_RMS_AMPLITUDE})")

    # SNR estimation
    # Simple approach: assume noise is the quietest 10% of the signal
    snr_db = _estimate_snr(data)

    if snr_db < MIN_SNR_DB:
        issues.append(f"Low SNR: {snr_db:.1f}dB (min {MIN_SNR_DB}dB)")

    is_valid = len(issues) == 0

    return AudioQualityReport(
        is_valid=is_valid,
        snr_db=snr_db,
        peak_amplitude=peak,
        rms_amplitude=rms,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        has_clipping=has_clipping,
        issues=issues,
    )


def _estimate_snr(data: np.ndarray) -> float:
    """Estimate signal-to-noise ratio from audio data.

    Uses a two-stage approach:
    1. Split audio into short frames and compute per-frame energy.
    2. Separate "active" frames (speech/signal) from "silent" frames
       using an energy threshold (the lower 20th percentile of frame energies).
    3. If no clear silence is found (e.g., continuous tone), estimate noise
       from the spectral residual after removing dominant frequencies.

    For voice samples destined for TTS cloning, an SNR > 20dB indicates
    acceptably clean recording conditions.

    Args:
        data: 1D audio data (float64, normalized to -1.0..1.0).

    Returns:
        Estimated SNR in decibels. Returns 0.0 for empty/silent audio.
    """
    if len(data) == 0:
        return 0.0

    signal_power = float(np.mean(data**2))
    if signal_power < 1e-10:
        return 0.0

    # Frame-based energy analysis
    frame_size = 1024
    num_frames = len(data) // frame_size

    if num_frames < 2:
        # Too short for frame analysis — use spectral method
        return _spectral_snr(data)

    # Compute energy per frame
    frame_energies = np.array(
        [np.mean(data[i * frame_size : (i + 1) * frame_size] ** 2) for i in range(num_frames)]
    )

    # Sort frame energies
    sorted_energies = np.sort(frame_energies)

    # Use the 20th percentile as noise floor estimate
    noise_cutoff = max(1, num_frames // 5)
    noise_energy = float(np.mean(sorted_energies[:noise_cutoff]))

    # Signal energy from the louder frames
    signal_energy = float(np.mean(sorted_energies[noise_cutoff:]))

    if signal_energy < 1e-10:
        return 0.0

    # Check if noise and signal energies are very close (e.g., continuous tone).
    # In that case the frame-based method is unreliable, fall back to spectral.
    if noise_energy > 0 and signal_energy / max(noise_energy, 1e-10) < 2.0:
        return _spectral_snr(data)

    noise_energy = max(noise_energy, 1e-10)
    return float(10 * np.log10(signal_energy / noise_energy))


def _spectral_snr(data: np.ndarray) -> float:
    """Estimate SNR using spectral analysis.

    Computes the FFT, identifies dominant signal components (top bins by
    magnitude), and treats the remaining spectrum as noise.

    Args:
        data: 1D audio data (float64, normalized).

    Returns:
        Estimated SNR in decibels.
    """
    signal_power = float(np.mean(data**2))
    if signal_power < 1e-10:
        return 0.0

    # Compute FFT magnitudes
    n = len(data)
    fft_mags = np.abs(np.fft.rfft(data))
    power_spectrum = fft_mags**2

    # Sort bins by power
    sorted_powers = np.sort(power_spectrum)[::-1]
    total_power = float(np.sum(power_spectrum))

    if total_power < 1e-10:
        return 0.0

    # Find how many bins account for 90% of the total power (signal)
    cumsum = np.cumsum(sorted_powers)
    signal_bin_count = int(np.searchsorted(cumsum, 0.9 * total_power)) + 1

    # Signal power = top bins, noise power = remaining bins
    signal_pwr = float(np.sum(sorted_powers[:signal_bin_count]))
    noise_pwr = float(np.sum(sorted_powers[signal_bin_count:]))
    noise_pwr = max(noise_pwr, 1e-10)

    return float(10 * np.log10(signal_pwr / noise_pwr))


def convert_to_target_format(
    input_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> Path:
    """Convert an audio file to the target WAV format using ffmpeg.

    Converts to WAV 16-bit PCM at the target sample rate and channel count.

    Args:
        input_path: Path to the input audio file (any format ffmpeg supports).
        output_path: Path where the converted WAV file will be written.
        sample_rate: Target sample rate (default 22050).
        channels: Target channel count (default 1 = mono).

    Returns:
        Path to the converted file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If ffmpeg conversion fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg: sudo apt install ffmpeg"
        ) from None

    return output_path


def normalize_audio(data: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio data to a target RMS level.

    Args:
        data: 1D audio data (float64).
        target_rms: Target RMS amplitude (default 0.1).

    Returns:
        Normalized audio data.
    """
    current_rms = float(np.sqrt(np.mean(data**2)))
    if current_rms < 1e-10:
        return data

    scale = target_rms / current_rms
    normalized = data * scale

    # Prevent clipping
    peak = float(np.max(np.abs(normalized)))
    if peak > 0.95:
        normalized = normalized * (0.95 / peak)

    return normalized
