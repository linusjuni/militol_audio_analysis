"""Audio-related helper utilities shared across training and inference."""

from __future__ import annotations

import torch


def ensure_sample_rate(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample ``waveform`` to ``target_sr`` when necessary.

    Args:
        waveform: Audio tensor shaped (channels, samples) or (1, samples).
        orig_sr: Sample rate the waveform was recorded at.
        target_sr: Desired sample rate for downstream processing.

    Returns:
        Waveform tensor at ``target_sr``.
    """
    if orig_sr == target_sr:
        return waveform

    import torchaudio

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


__all__ = ["ensure_sample_rate"]
