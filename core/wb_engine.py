"""
wb_engine.py — White Balance sampling and correction engine.

Workflow:
  1. User clicks on the film mask (border area with orange cast) in the NEGATIVE.
  2. sample_point() samples a small neighbourhood (median per channel).
  3. compute_multipliers() derives per-channel multipliers to neutralise the cast.
     Method: green_ref (default) -- G is the anchor (mult_G = 1.0),
             R and B are scaled to match G.  Matches Aaron Buchler's guide and
             Adobe Camera Raw / Negative Lab Pro internals.
  4. apply() applies the multipliers to the NEGATIVE image BEFORE inversion.

Pipeline order (CRITICAL -- must be enforced by caller):
  RAW -> WB (on negative, before inversion) -> Invert -> Levels -> Export
"""

import logging
from typing import Optional, Tuple

import numpy as np
from core.gpu_backend import apply_wb as _gpu_apply_wb, is_gpu_available

logger = logging.getLogger(__name__)


class WBEngine:
    """White balance sample-and-correct engine."""

    def __init__(self, sample_radius: int = 7):
        """
        Args:
            sample_radius: Half-size of the sampling window in pixels.
                           e.g. radius=7 → 15×15 patch.
        """
        self.sample_radius = sample_radius
        self.last_sample: Optional[np.ndarray] = None
        self.last_multipliers: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_point(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample a patch around (x, y) and return the median RGB values.

        Args:
            image: float32 array, shape (H, W, 3), range [0, 1].
            x: Column index (pixel X).
            y: Row index (pixel Y).
            radius: Override instance radius for this call.

        Returns:
            1-D float32 array [R_median, G_median, B_median].
        """
        r = radius if radius is not None else self.sample_radius
        H, W = image.shape[:2]

        # Clamp patch to image bounds
        y0 = max(0, y - r)
        y1 = min(H, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(W, x + r + 1)

        patch = image[y0:y1, x0:x1, :]  # shape (ph, pw, 3)

        if patch.size == 0:
            logger.warning("Empty patch at (%d, %d) — returning grey", x, y)
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Median per channel (robust against dust / scratches in the mask)
        sample = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)
        logger.debug("WB sample at (%d,%d) r=%d: R=%.4f G=%.4f B=%.4f", x, y, r, *sample)
        self.last_sample = sample
        return sample

    # ------------------------------------------------------------------
    # Multiplier computation
    # ------------------------------------------------------------------

    def compute_multipliers(
        self,
        sample: np.ndarray,
        target: Optional[float] = None,
        method: str = "green_ref",
    ) -> np.ndarray:
        """
        Compute per-channel multiplicative gains to neutralise the sample.

        The sample represents the film base / mask colour (orange film mask).
        We want that region to appear neutral grey (equal R, G, B) after WB.

        Two methods are available:

        ``green_ref`` (default, recommended for colour negatives):
            Green is used as the anchor channel (mult_G = 1.0). R and B are
            scaled to match G.  This is the method described by Aaron Buchler
            (https://abpy.github.io/2023/08/20/color-neg.html) and matches
            what Adobe Camera Raw / Lightroom / Negative Lab Pro do internally.
            G has the highest sensor sensitivity and is least affected by the
            orange mask, making it the most stable reference channel.

        ``mean`` (legacy):
            All three channels are scaled to their mean value.  Kept for
            compatibility but changes the G channel which can cause downstream
            colour-balance drift.

        Args:
            sample: 1-D float32 [R, G, B] from sample_point().
            target: Desired grey level (only used by ``mean`` method).
            method: ``"green_ref"`` (default) or ``"mean"``.

        Returns:
            1-D float32 array [mult_R, mult_G, mult_B], clamped to [0.1, 4.0].
        """
        sample = np.asarray(sample, dtype=np.float32)

        # Protect against near-zero values (pure black mask → division by zero)
        eps = 1e-6
        sample_safe = np.maximum(sample, eps)

        if method == "green_ref":
            # G channel is the anchor — it stays unchanged (mult_G = 1.0).
            # R and B are multiplied so their response equals G.
            # After WB: R_out = R * (G/R) = G, B_out = B * (G/B) = G  → neutral.
            g = float(sample_safe[1])  # green reference
            mults = np.array(
                [g / float(sample_safe[0]),   # mult_R = G_sample / R_sample
                 1.0,                          # mult_G = 1.0  (anchor)
                 g / float(sample_safe[2])],   # mult_B = G_sample / B_sample
                dtype=np.float32,
            )
        else:  # method == "mean" (legacy)
            if target is None:
                target = float(np.mean(sample_safe))
            if target < eps:
                target = eps
            mults = np.array(
                [target / float(sample_safe[0]),
                 target / float(sample_safe[1]),
                 target / float(sample_safe[2])],
                dtype=np.float32,
            )

        # Clamp to a safe range — very saturated masks (e.g. mult_B=5.0 from a
        # strongly orange mask) would cause blue clipping after inversion.
        # Range [0.1, 4.0] covers all practical film stocks.
        mults = np.clip(mults, 0.1, 4.0).astype(np.float32)

        logger.debug(
            "WB mults [%s]: R=%.4f G=%.4f B=%.4f", method, *mults
        )
        self.last_multipliers = mults
        return mults

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
        multipliers: np.ndarray,
    ) -> np.ndarray:
        """
        Apply per-channel multipliers to an image.

        Args:
            image: float32 array (H, W, 3), range [0, 1].
            multipliers: 1-D float32 [mult_R, mult_G, mult_B].

        Returns:
            float32 array (H, W, 3), values clipped to [0, 1].
        """
        mults = np.asarray(multipliers, dtype=np.float32)
        if is_gpu_available():
            return _gpu_apply_wb(image, mults)
        result = image.copy()
        result[:, :, 0] *= mults[0]
        result[:, :, 1] *= mults[1]
        result[:, :, 2] *= mults[2]
        np.clip(result, 0.0, 1.0, out=result)
        return result

    # ------------------------------------------------------------------
    # Convenience one-shot
    # ------------------------------------------------------------------

    def correct_from_click(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample → compute → apply in one call.

        Returns:
            (corrected_image, sample_rgb, multipliers)
        """
        sample = self.sample_point(image, x, y, radius)
        mults = self.compute_multipliers(sample)
        corrected = self.apply(image, mults)
        return corrected, sample, mults

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def get_last_multipliers(self) -> Optional[np.ndarray]:
        return self.last_multipliers

    def get_last_sample(self) -> Optional[np.ndarray]:
        return self.last_sample



