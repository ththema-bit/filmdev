"""
levels_engine.py — Auto Black-Point / White-Point stretching.

Two methods:
  1. Percentile  (fast, deterministic, recommended default)
  2. Iterative   (matches no clipped pixels)
"""

import logging
from typing import Tuple

import numpy as np
from core.gpu_backend import (
    auto_levels_percentile as _gpu_percentile,
    auto_levels_iterative as _gpu_iterative,
    is_gpu_available,
)

logger = logging.getLogger(__name__)


class LevelsEngine:
    """Auto levels: stretch tonal range so no pixels clip."""

    # ------------------------------------------------------------------
    # Method 1: Percentile (recommended default)
    # ------------------------------------------------------------------

    def auto_levels_percentile(
        self,
        image: np.ndarray,
        bp_percentile: float = 0.5,
        wp_percentile: float = 99.5,
        per_channel: bool = False,
    ) -> np.ndarray:
        """
        Stretch levels using percentile-based black and white points.

        Args:
            image:          float32 (H, W, 3) in [0, 1] (any range technically OK).
            bp_percentile:  Percentile for black point  (default 0.5%).
            wp_percentile:  Percentile for white point  (default 99.5%).
            per_channel:    If True, compute BP/WP independently per channel.
                            If False (default), compute on luminance then apply equally.

        Returns:
            float32 (H, W, 3) stretched to [0, 1].
        """
        # GPU fast path: per-channel percentile on CUDA
        if is_gpu_available():
            return _gpu_percentile(image, bp_percentile, wp_percentile)

        result = image.astype(np.float32).copy()

        if per_channel:
            for c in range(3):
                ch = result[:, :, c]
                bp = float(np.percentile(ch, bp_percentile))
                wp = float(np.percentile(ch, wp_percentile))
                span = wp - bp
                if span < 1e-8:
                    logger.warning("Channel %d has near-zero span — skipping stretch", c)
                    continue
                result[:, :, c] = np.clip((ch - bp) / span, 0.0, 1.0)
            logger.debug("Levels (percentile per-channel): bp=%.4f%% wp=%.4f%%", bp_percentile, wp_percentile)
        else:
            # Use luminance-weighted combined view
            flat = result.reshape(-1, 3)
            bp = float(np.percentile(flat, bp_percentile))
            wp = float(np.percentile(flat, wp_percentile))
            span = wp - bp
            if span < 1e-8:
                logger.warning("Image has near-zero tonal span — skipping levels stretch")
                return result
            result = np.clip((result - bp) / span, 0.0, 1.0).astype(np.float32)
            logger.debug(
                "Levels (percentile global): bp=%.4f (%.1f%%) wp=%.4f (%.1f%%)",
                bp, bp_percentile, wp, wp_percentile,
            )

        return result

    # ------------------------------------------------------------------
    # Method 2: Iterative
    # ------------------------------------------------------------------

    def auto_levels_iterative(
        self,
        image: np.ndarray,
        step: float = 0.001,
        white_threshold: float = 0.999,
        black_threshold: float = 0.001,
        max_iterations: int = 500,
        convergence_threshold: float = 1e-5,
    ) -> np.ndarray:
        """
        Iteratively shrink white-point and raise black-point until no pixels clip.

        Algorithm:
          - Start wp=1.0, bp=0.0
          - While pixels exist above white_threshold: wp -= step
          - While pixels exist below black_threshold: bp += step
          - Alternate passes until neither condition holds or max_iterations reached
          - Stop early if white-pixel count stops decreasing (local minimum)

        Args:
            image:               float32 (H, W, 3).
            step:                Adjustment step per iteration (default 0.001).
            white_threshold:     Pixel value considered "clipped white" (default 0.999).
            black_threshold:     Pixel value considered "clipped black" (default 0.001).
            max_iterations:      Safety cap.
            convergence_threshold: Stop if wp-bp change < this value per full pass.

        Returns:
            float32 (H, W, 3) stretched to [0, 1].
        """
        arr = image.astype(np.float32).copy()
        wp = 1.0
        bp = 0.0

        prev_white_count = None

        for iteration in range(max_iterations):
            # Normalise with current bp/wp
            span = wp - bp
            if span < 1e-8:
                logger.warning("Iterative levels: wp-bp span too small at iteration %d — stopping", iteration)
                break

            normalised = np.clip((arr - bp) / span, 0.0, 1.0)
            white_count = int(np.sum(normalised >= white_threshold))
            black_count = int(np.sum(normalised <= black_threshold))

            changed = False

            # Shrink white point if clipped whites exist
            if white_count > 0:
                # Check for convergence: white count not improving
                if prev_white_count is not None and white_count >= prev_white_count:
                    logger.debug(
                        "Iterative levels: white count not decreasing (%d) at iter %d — stopping WP",
                        white_count, iteration,
                    )
                else:
                    new_wp = wp - step
                    if new_wp > bp + 1e-8:
                        wp = new_wp
                        changed = True
                prev_white_count = white_count

            # Raise black point if clipped blacks exist
            if black_count > 0:
                new_bp = bp + step
                if new_bp < wp - 1e-8:
                    bp = new_bp
                    changed = True

            if not changed:
                logger.debug("Iterative levels converged at iteration %d: bp=%.4f wp=%.4f", iteration, bp, wp)
                break

        # Final stretch with solved bp/wp
        span = wp - bp
        if span < 1e-8:
            span = 1.0  # fallback — avoid div by zero
        result = np.clip((arr - bp) / span, 0.0, 1.0).astype(np.float32)

        logger.debug("Iterative levels final: bp=%.4f wp=%.4f iters=%d", bp, wp, iteration + 1)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_bp_wp_percentile(
        self,
        image: np.ndarray,
        bp_percentile: float = 0.5,
        wp_percentile: float = 99.5,
    ) -> Tuple[float, float]:
        """Return (bp, wp) values without applying them."""
        flat = image.reshape(-1)
        bp = float(np.percentile(flat, bp_percentile))
        wp = float(np.percentile(flat, wp_percentile))
        return bp, wp

    def apply_levels(
        self,
        image: np.ndarray,
        bp: float,
        wp: float,
    ) -> np.ndarray:
        """Apply explicit bp/wp values to an image."""
        span = wp - bp
        if span < 1e-8:
            return image.copy()
        result = np.clip((image.astype(np.float32) - bp) / span, 0.0, 1.0)
        return result.astype(np.float32)
