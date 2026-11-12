"""
invert_engine.py — Film negative inversion.

Three methods:
  1. Transmittance (CORRECT, default): inverted = 1 / image  (then normalised)
     Film records density = log10(1/transmittance), so 1/transmittance recovers
     the original scene exposure linearly. This is what Negative Lab Pro and
     professional inversion tools use.
  2. Linear (simple):   inverted = 1.0 - image
     Fast, ok for slides/positives or quick preview, but tonally incorrect for
     colour negatives (compresses shadows, clips highlights).
  3. Log inversion:     inverted = -log(image + epsilon), then normalised
     Similar to transmittance in effect but less mathematically rigorous.

All expect and return float32 arrays in [0, 1].

Order of operations (negative workflow) — ALWAYS:
  1. Load RAW → float32
  2. Apply WB correction (neutralises the orange mask) ← MUST come first
  3. Invert (this module)
  4. Auto levels (levels_engine) to fill tonal range
  5. Export
"""

import logging

import numpy as np
from core.gpu_backend import (
    invert_transmittance as _gpu_transmittance,
    invert_linear as _gpu_linear,
    invert_log as _gpu_log,
    is_gpu_available,
    device_name,
)

logger = logging.getLogger(__name__)

# Epsilon prevents log(0)
LOG_EPSILON = 1e-6


class InvertEngine:
    """Invert a float32 image from negative to positive."""

    # ------------------------------------------------------------------
    # Transmittance inversion (correct, default)
    # ------------------------------------------------------------------

    def invert_transmittance(
        self,
        image: np.ndarray,
        epsilon: float = 1e-4,
        _use_gpu: bool = True,
    ) -> np.ndarray:
        """
        Physically correct inversion: result = 1 / image (transmittance method).

        Film records density = log10(1/transmittance). Dividing 1 by the
        measured transmittance recovers scene exposure linearly, preserving
        highlight and shadow detail correctly across all three channels.

        Steps:
          1. Clamp to [epsilon, 1] to avoid divide-by-zero
          2. Compute 1/image per channel
          3. Normalise to [0, 1]

        Args:
            image:   float32 (H, W, 3), WB-corrected, values in [0, 1].
            epsilon: Floor to prevent 1/0 (default 1e-4 ≈ stops at density ~4).

        Returns:
            float32 (H, W, 3), values in [0, 1].
        """
        src = np.clip(image, epsilon, 1.0).astype(np.float64)
        inverted = 1.0 / src

        # Normalise per-channel so each channel fills [0, 1] independently.
        # This mirrors what NLP and Silverfast do: per-channel density normalisation.
        result = np.empty_like(inverted)
        for c in range(inverted.shape[2]):
            ch = inverted[:, :, c]
            lo, hi = ch.min(), ch.max()
            span = hi - lo
            if span < 1e-10:
                result[:, :, c] = 0.0
            else:
                result[:, :, c] = (ch - lo) / span

        if _use_gpu and is_gpu_available():
            logger.debug("Transmittance inversion via GPU (%s)", device_name())
            return _gpu_transmittance(image, epsilon)

        logger.debug(
            "Transmittance inversion (CPU): in=[%.4f,%.4f]",
            float(image.min()), float(image.max()),
        )
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Linear inversion (simple fallback)
    # ------------------------------------------------------------------

    def invert_linear(self, image: np.ndarray, _use_gpu: bool = True) -> np.ndarray:
        """
        Simple arithmetic inversion: result = 1.0 - image.

        Tonally incorrect for colour negatives but fast.
        Use transmittance method for proper results.

        Args:
            image: float32 (H, W, 3), values in [0, 1].

        Returns:
            float32 (H, W, 3), values in [0, 1].
        """
        if _use_gpu and is_gpu_available():
            return _gpu_linear(image)
        result = 1.0 - np.clip(image, 0.0, 1.0)
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Log inversion
    # ------------------------------------------------------------------

    def invert_log(
        self,
        image: np.ndarray,
        epsilon: float = LOG_EPSILON,
        base: str = "natural",
        _use_gpu: bool = True,
    ) -> np.ndarray:
        """
        Logarithmic inversion for improved tonal compression.

        The negative film has a roughly logarithmic density response.
        Inverting in log space better maps density to perceptual brightness.

        Algorithm:
          1. log_image = log(image + epsilon)           ← log of scene density
          2. inverted  = -log_image                     ← flip (dense = dark)
          3. Normalise to [0, 1]                        ← fit to display range

        Args:
            image:   float32 (H, W, 3), values in [0, 1].
            epsilon: Small offset to prevent log(0).
            base:    "natural" (ln), "log2", or "log10".

        Returns:
            float32 (H, W, 3), normalised to [0, 1].
        """
        src = np.clip(image, 0.0, 1.0).astype(np.float64)  # use float64 for precision

        if base == "log2":
            log_fn = np.log2
        elif base == "log10":
            log_fn = np.log10
        else:
            log_fn = np.log  # natural log

        # Compute log and negate
        inverted = -log_fn(src + epsilon)

        # Normalise per-image to [0, 1]
        inv_min = inverted.min()
        inv_max = inverted.max()
        span = inv_max - inv_min

        if span < 1e-10:
            logger.warning("Log inversion: near-zero span — returning flat image")
            return np.zeros_like(image, dtype=np.float32)

        result = (inverted - inv_min) / span
        if _use_gpu and is_gpu_available():
            logger.debug("Log inversion via GPU (%s)", device_name())
            return _gpu_log(image, epsilon)

        logger.debug(
            "Log inversion (%s CPU): in=[%.4f,%.4f] log=[%.4f,%.4f]",
            base, float(src.min()), float(src.max()), float(inv_min), float(inv_max),
        )
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def invert(self, image: np.ndarray, method: str = "transmittance", **kwargs) -> np.ndarray:
        """
        Unified inversion entry point.

        Args:
            image:  float32 (H, W, 3).
            method: "transmittance" (default, correct), "linear", or "log".
            **kwargs: Forwarded to the chosen method.

        Returns:
            float32 (H, W, 3).
        """
        if method == "transmittance":
            return self.invert_transmittance(image, **kwargs)
        elif method == "log":
            return self.invert_log(image, **kwargs)
        elif method == "linear":
            return self.invert_linear(image)
        else:
            raise ValueError(f"Unknown inversion method: {method!r}. Use 'transmittance', 'linear', or 'log'.")
