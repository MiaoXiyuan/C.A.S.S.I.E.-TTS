"""F0 shaping for Cassie inference.

The goal is low, stable pitch: optional global pitch shift plus flattening /
limiting / quantizing the F0 contour.  WORLD (pyworld) is used so the spectral
envelope (formants) stays mostly intact while only F0 is changed.
"""

from __future__ import annotations

import numpy as np

try:
    import pyworld as pw
except Exception:  # pragma: no cover - surfaced as a clear runtime error
    pw = None

F0_FLOOR = 50.0
F0_CEIL = 500.0
MODES = {"off", "flat", "limited", "quantized"}


def _as_mono(audio: np.ndarray) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float32)


def process_f0(
    audio: np.ndarray,
    sr: int = 24000,
    *,
    semitones: float = 0.0,
    mode: str = "off",
    limit: float = 1.0,
    step: float = 1.5,
) -> np.ndarray:
    """Return audio with shifted/flattened F0, preserving duration.

    modes:
      off       only global pitch shift (semitones)
      flat      all voiced frames at the median F0 (× global shift)
      limited   F0 within ±limit semitones of the median, then global shift
      quantized F0 quantized to `step` semitone steps around the median
    """
    y = _as_mono(audio)
    semitones = float(semitones or 0.0)
    mode = (mode or "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown f0 mode: {mode}")
    if abs(semitones) < 1e-6 and mode == "off":
        return y
    if pw is None:
        raise RuntimeError("pyworld is required for F0 processing; pip install pyworld")
    if y.size < int(0.1 * sr):
        return y

    x = y.astype(np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL)
    voiced = f0 > 0
    if not np.any(voiced):
        return y

    sp = pw.cheaptrick(x, f0, t, sr, f0_floor=F0_FLOOR, fft_size=2048)
    ap = pw.d4c(x, f0, t, sr, fft_size=2048)
    center = float(np.median(f0[voiced]))
    shift = 2.0 ** (semitones / 12.0)
    new_f0 = f0.copy()

    if mode == "flat":
        new_f0[voiced] = center * shift
    elif mode == "limited":
        lim = max(0.0, float(limit))
        semi = 12.0 * np.log2(f0[voiced] / center)
        semi = np.clip(semi, -lim, lim)
        new_f0[voiced] = center * (2.0 ** (semi / 12.0)) * shift
    elif mode == "quantized":
        step = max(0.1, float(step))
        semi = 12.0 * np.log2(f0[voiced] / center)
        semi = np.round(semi / step) * step
        new_f0[voiced] = center * (2.0 ** (semi / 12.0)) * shift
    else:
        new_f0[voiced] = f0[voiced] * shift

    new_f0[voiced] = np.clip(new_f0[voiced], F0_FLOOR, F0_CEIL)
    out = pw.synthesize(new_f0, sp, ap, sr).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = (out / peak) * 0.99
    return np.ascontiguousarray(out, dtype=np.float32)
