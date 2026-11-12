"""
gpu_backend.py — Transparent GPU/CPU backend for FilmDev engines.

Uses PyTorch (CUDA) if available, falls back to NumPy silently.
All functions accept and return float32 numpy arrays — GPU tensors
are only used internally so the rest of the codebase is unchanged.

Usage:
    from core.gpu_backend import backend
    result_np = backend.multiply(image_np, multipliers)
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ── Detect backend ────────────────────────────────────────────────────────────

def _detect():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
            logger.info("GPU backend: PyTorch CUDA — %s (%d MB VRAM)", name, vram)
            return "cuda", torch
    except ImportError:
        pass
    logger.info("GPU backend: NumPy CPU")
    return "cpu", None

_BACKEND, _torch = _detect()


def is_gpu_available() -> bool:
    return _BACKEND == "cuda"


def device_name() -> str:
    if _BACKEND == "cuda" and _torch:
        return _torch.cuda.get_device_properties(0).name
    return "CPU (NumPy)"


# ── Helper: numpy ↔ tensor ────────────────────────────────────────────────────

def _to_tensor(arr: np.ndarray):
    """numpy float32 → CUDA tensor (non-blocking)."""
    return _torch.from_numpy(arr).to("cuda", non_blocking=True)


def _to_numpy(t) -> np.ndarray:
    """CUDA tensor → numpy float32."""
    return t.cpu().numpy()


# ── Core operations (all take/return numpy float32 [0,1]) ────────────────────

def apply_wb(image: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """
    White balance: multiply each channel by its multiplier, clip to [0, 1].
    multipliers: shape (3,) — [mult_R, mult_G, mult_B]
    """
    if _BACKEND == "cuda":
        t = _to_tensor(image)                          # (H,W,3) on GPU
        m = _torch.tensor(multipliers, dtype=_torch.float32, device="cuda")
        result = _torch.clamp(t * m, 0.0, 1.0)
        return _to_numpy(result)
    else:
        return np.clip(image * multipliers, 0.0, 1.0).astype(np.float32)


def invert_transmittance(image: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Physically correct inversion: result = 1 / image, per-channel normalised.
    WB MUST be applied before calling this.
    """
    if _BACKEND == "cuda":
        t = _to_tensor(image).double()                 # float64 for precision
        t = _torch.clamp(t, epsilon, 1.0)
        inv = 1.0 / t                                  # (H,W,3)
        # Per-channel normalise
        lo = inv.flatten(0, 1).min(dim=0).values       # (3,)
        hi = inv.flatten(0, 1).max(dim=0).values       # (3,)
        span = hi - lo
        span = _torch.where(span < 1e-10, _torch.ones_like(span), span)
        result = (inv - lo) / span
        return _to_numpy(result.float())
    else:
        src = np.clip(image, epsilon, 1.0).astype(np.float64)
        inv = 1.0 / src
        result = np.empty_like(inv)
        for c in range(3):
            ch = inv[:, :, c]
            lo, hi = ch.min(), ch.max()
            span = hi - lo
            result[:, :, c] = 0.0 if span < 1e-10 else (ch - lo) / span
        return result.astype(np.float32)


def invert_linear(image: np.ndarray) -> np.ndarray:
    """Simple 1 - x inversion."""
    if _BACKEND == "cuda":
        t = _to_tensor(image)
        return _to_numpy(1.0 - _torch.clamp(t, 0.0, 1.0))
    else:
        return (1.0 - np.clip(image, 0.0, 1.0)).astype(np.float32)


def invert_log(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Log inversion: -ln(x + ε), normalised."""
    if _BACKEND == "cuda":
        t = _to_tensor(image).double()
        t = _torch.clamp(t, 0.0, 1.0)
        inv = -_torch.log(t + epsilon)
        lo = inv.min(); hi = inv.max(); span = hi - lo
        if span < 1e-10:
            return np.zeros_like(image)
        result = (inv - lo) / span
        return _to_numpy(result.float())
    else:
        src = np.clip(image, 0.0, 1.0).astype(np.float64)
        inv = -np.log(src + epsilon)
        lo, hi = inv.min(), inv.max()
        span = hi - lo
        if span < 1e-10:
            return np.zeros_like(image, dtype=np.float32)
        return ((inv - lo) / span).astype(np.float32)


def auto_levels_percentile(
    image: np.ndarray,
    bp_pct: float = 0.5,
    wp_pct: float = 99.5,
) -> np.ndarray:
    """
    Stretch levels: bp_pct → 0, wp_pct → 1, clip.
    Operates per-channel so colour balance is preserved.
    """
    if _BACKEND == "cuda":
        t = _to_tensor(image)
        result = _torch.empty_like(t)
        MAX_QUANTILE_SAMPLES = 2_000_000  # torch.quantile limit
        for c in range(3):
            ch = t[:, :, c].flatten()
            # Subsample if too large for torch.quantile (limit ~16M but safe at 2M)
            if ch.numel() > MAX_QUANTILE_SAMPLES:
                idx = _torch.randperm(ch.numel(), device='cuda')[:MAX_QUANTILE_SAMPLES]
                ch_sample = ch[idx]
            else:
                ch_sample = ch
            bp = _torch.quantile(ch_sample, bp_pct / 100.0)
            wp = _torch.quantile(ch_sample, wp_pct / 100.0)
            span = wp - bp
            if span < 1e-10:
                result[:, :, c] = t[:, :, c]
            else:
                result[:, :, c] = _torch.clamp((t[:, :, c] - bp) / span, 0.0, 1.0)
        return _to_numpy(result)
    else:
        result = np.empty_like(image)
        for c in range(3):
            ch = image[:, :, c]
            bp = np.percentile(ch, bp_pct)
            wp = np.percentile(ch, wp_pct)
            span = wp - bp
            if span < 1e-10:
                result[:, :, c] = ch
            else:
                result[:, :, c] = np.clip((ch - bp) / span, 0.0, 1.0)
        return result.astype(np.float32)


def auto_levels_iterative(
    image: np.ndarray,
    threshold: float = 0.001,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Iterative: pull in BP/WP until no pure-black or pure-white pixels remain.
    Runs on CPU (iterative loops don't benefit much from GPU).
    """
    arr = image.copy().astype(np.float32)
    bp = np.zeros(3, dtype=np.float32)
    wp = np.ones(3, dtype=np.float32)

    for _ in range(max_iter):
        changed = False
        for c in range(3):
            ch = arr[:, :, c]
            if (ch >= (wp[c] - 1e-4)).sum() > 0 and wp[c] > bp[c] + threshold:
                wp[c] -= threshold; changed = True
            if (ch <= (bp[c] + 1e-4)).sum() > 0 and bp[c] < wp[c] - threshold:
                bp[c] += threshold; changed = True
        if not changed:
            break

    result = np.empty_like(arr)
    for c in range(3):
        span = wp[c] - bp[c]
        result[:, :, c] = np.clip((arr[:, :, c] - bp[c]) / span, 0.0, 1.0) if span > 1e-10 else arr[:, :, c]
    return result


# ── Convenience: full pipeline in one GPU pass ───────────────────────────────

def process_pipeline(
    image: np.ndarray,
    wb_multipliers: Optional[np.ndarray],
    invert: bool = True,
    inversion_method: str = "transmittance",
    bp_pct: float = 0.5,
    wp_pct: float = 99.5,
) -> np.ndarray:
    """
    Full pipeline in correct order:
      1. WB (if multipliers provided)
      2. Invert (if neg mode)
      3. Auto levels (percentile)

    All GPU ops stay on-device between steps — only one CPU↔GPU transfer total.
    """
    if _BACKEND == "cuda" and wb_multipliers is not None:
        # Do everything in one GPU session
        t = _to_tensor(image)
        m = _torch.tensor(wb_multipliers, dtype=_torch.float32, device="cuda")
        t = _torch.clamp(t * m, 0.0, 1.0)             # step 1: WB

        if invert:
            if inversion_method == "transmittance":
                t = t.double()
                t = _torch.clamp(t, 1e-4, 1.0)
                t = 1.0 / t
                lo = t.flatten(0,1).min(dim=0).values
                hi = t.flatten(0,1).max(dim=0).values
                span = _torch.where((hi-lo) < 1e-10, _torch.ones_like(hi), hi-lo)
                t = (t - lo) / span
                t = t.float()
            elif inversion_method == "log":
                t = t.double()
                t = -_torch.log(_torch.clamp(t, 0.0, 1.0) + 1e-6)
                lo = t.min(); hi = t.max()
                t = (t - lo) / (hi - lo + 1e-10)
                t = t.float()
            else:  # linear
                t = 1.0 - _torch.clamp(t, 0.0, 1.0)  # step 2: invert

        # step 3: per-channel levels (subsample for large images)
        MAX_Q = 2_000_000
        for c in range(3):
            ch = t[:, :, c].flatten()
            samp = ch[_torch.randperm(ch.numel(), device='cuda')[:MAX_Q]] if ch.numel() > MAX_Q else ch
            bp = _torch.quantile(samp, bp_pct / 100.0)
            wp_v = _torch.quantile(samp, wp_pct / 100.0)
            span = wp_v - bp
            if span > 1e-10:
                t[:, :, c] = _torch.clamp((t[:, :, c] - bp) / span, 0.0, 1.0)

        return _to_numpy(t)

    else:
        # CPU fallback, same order
        result = image.copy()
        if wb_multipliers is not None:
            result = apply_wb(result, wb_multipliers)
        if invert:
            if inversion_method == "transmittance":
                result = invert_transmittance(result)
            elif inversion_method == "log":
                result = invert_log(result)
            else:
                result = invert_linear(result)
        result = auto_levels_percentile(result, bp_pct, wp_pct)
        return result
