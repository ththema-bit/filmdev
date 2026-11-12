"""
filmstock_profiles.py — Filmstock profile database for FilmDev.

Each profile encodes the colorimetric characteristics of a specific film:
  - mask_ref:    Expected orange-mask RGB triplet (float, 0-1) for WB override
  - gamma:       Tonal gamma to apply post-inversion (1.0 = neutral)
  - saturation:  Saturation multiplier (0.0 = B&W, 1.0 = unchanged, >1.0 = boost)
  - wb_tint:     Per-channel colour balance tweak [R, G, B] applied before curve
  - curve:       Tone curve type: "linear" | "gentle_s" | "strong_s"
  - description: Human-readable description

Usage (in pipeline, after WB + inversion + levels):
    from core.filmstock_profiles import apply_profile
    result = apply_profile(inverted_arr, "kodak_gold_200")

Stufe 3 hook:
    from core.filmstock_profiles import detect_filmtype
    hint = detect_filmtype(mask_rgb)   # → "kodak_gold_200" / "fuji_superia_400" / "auto"
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile database
# ---------------------------------------------------------------------------

FILMSTOCK_PROFILES: Dict[str, dict] = {
    "auto": {
        "label": "Auto (no profile)",
        "type": None,
        "mask_ref": None,
        "gamma": 1.0,
        "saturation": 1.0,
        "wb_tint": [1.0, 1.0, 1.0],
        "curve": "linear",
        "description": "No film profile — pure math inversion only",
    },
    "kodak_gold_200": {
        "label": "Kodak Gold 200",
        "type": "C-41",
        "mask_ref": [0.820, 0.515, 0.185],
        "gamma": 1.18,
        "saturation": 1.06,
        "wb_tint": [1.02, 1.0, 0.96],
        "curve": "gentle_s",
        "description": "Kodak Gold 200 — warm, affordable, everyday film",
    },
    "kodak_colorplus_200": {
        "label": "Kodak ColorPlus 200",
        "type": "C-41",
        "mask_ref": [0.830, 0.510, 0.175],
        "gamma": 1.20,
        "saturation": 1.08,
        "wb_tint": [1.03, 1.0, 0.95],
        "curve": "gentle_s",
        "description": "Kodak ColorPlus 200 — budget, punchy colours",
    },
    "kodak_ultramax_400": {
        "label": "Kodak Ultramax 400",
        "type": "C-41",
        "mask_ref": [0.800, 0.520, 0.195],
        "gamma": 1.15,
        "saturation": 1.05,
        "wb_tint": [1.01, 1.0, 0.97],
        "curve": "gentle_s",
        "description": "Kodak Ultramax 400 — fast, versatile, slightly green",
    },
    "kodak_portra_400": {
        "label": "Kodak Portra 400",
        "type": "C-41",
        "mask_ref": [0.790, 0.505, 0.200],
        "gamma": 1.12,
        "saturation": 0.98,
        "wb_tint": [1.01, 1.0, 0.98],
        "curve": "gentle_s",
        "description": "Kodak Portra 400 — skin-tone optimised, low saturation",
    },
    "kodak_portra_160": {
        "label": "Kodak Portra 160",
        "type": "C-41",
        "mask_ref": [0.785, 0.510, 0.205],
        "gamma": 1.10,
        "saturation": 0.96,
        "wb_tint": [1.0, 1.0, 0.99],
        "curve": "gentle_s",
        "description": "Kodak Portra 160 — finest grain Portra, neutral",
    },
    "kodak_ektar_100": {
        "label": "Kodak Ektar 100",
        "type": "C-41",
        "mask_ref": [0.775, 0.515, 0.210],
        "gamma": 1.22,
        "saturation": 1.15,
        "wb_tint": [0.99, 1.0, 1.01],
        "curve": "gentle_s",
        "description": "Kodak Ektar 100 — hyper-saturated, cool, finest grain",
    },
    "fuji_superia_400": {
        "label": "Fujifilm Superia 400",
        "type": "C-41",
        "mask_ref": [0.760, 0.540, 0.240],
        "gamma": 1.14,
        "saturation": 1.08,
        "wb_tint": [0.97, 1.0, 1.04],
        "curve": "gentle_s",
        "description": "Fujifilm Superia 400 — cool, green bias, punchy",
    },
    "fuji_superia_200": {
        "label": "Fujifilm Superia 200",
        "type": "C-41",
        "mask_ref": [0.755, 0.535, 0.235],
        "gamma": 1.12,
        "saturation": 1.06,
        "wb_tint": [0.98, 1.0, 1.03],
        "curve": "gentle_s",
        "description": "Fujifilm Superia 200 — slightly cooler than 400",
    },
    "fuji_pro_400h": {
        "label": "Fujifilm Pro 400H",
        "type": "C-41",
        "mask_ref": [0.740, 0.545, 0.255],
        "gamma": 1.08,
        "saturation": 0.95,
        "wb_tint": [0.96, 1.0, 1.05],
        "curve": "gentle_s",
        "description": "Fujifilm Pro 400H — pastel, very cool, portrait film",
    },
    "fuji_velvia_50": {
        "label": "Fujifilm Velvia 50 (Slide)",
        "type": "E-6",
        "mask_ref": None,
        "gamma": 1.35,
        "saturation": 1.25,
        "wb_tint": [1.0, 1.0, 1.0],
        "curve": "strong_s",
        "description": "Fujifilm Velvia 50 — hyper vivid slide film, no orange mask",
    },
    "ilford_hp5": {
        "label": "Ilford HP5 Plus (B&W)",
        "type": "B&W",
        "mask_ref": None,
        "gamma": 1.20,
        "saturation": 0.0,
        "wb_tint": [1.0, 1.0, 1.0],
        "curve": "gentle_s",
        "description": "Ilford HP5 Plus — classic B&W, wide latitude",
    },
    "ilford_delta_400": {
        "label": "Ilford Delta 400 (B&W)",
        "type": "B&W",
        "mask_ref": None,
        "gamma": 1.18,
        "saturation": 0.0,
        "wb_tint": [1.0, 1.0, 1.0],
        "curve": "gentle_s",
        "description": "Ilford Delta 400 — T-grain, fine detail B&W",
    },
    "kodak_tmax_400": {
        "label": "Kodak T-Max 400 (B&W)",
        "type": "B&W",
        "mask_ref": None,
        "gamma": 1.22,
        "saturation": 0.0,
        "wb_tint": [1.0, 1.0, 1.0],
        "curve": "gentle_s",
        "description": "Kodak T-Max 400 — very fine grain B&W",
    },
}


# ---------------------------------------------------------------------------
# Tone curves
# ---------------------------------------------------------------------------

def _build_lut(curve_name: str) -> np.ndarray:
    """Build a 256-entry LUT for the given curve type.

    All LUTs map float32 [0,1] via an integer index (x * 255 → index).
    The LUT itself stores float32 values in [0, 1].
    """
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)

    if curve_name == "linear":
        lut = x

    elif curve_name == "gentle_s":
        # Smooth cubic S-curve parameterised by shadow/highlight factors
        # and a midtone contrast boost.
        #
        # Method: apply a soft-knee S shape using a cubic polynomial that:
        #   - passes through (0,0) and (1,1)
        #   - pulls shadows slightly darker (below midpoint)
        #   - pushes highlights slightly brighter (above midpoint)
        #   - increases contrast in the mid quarter
        #
        # shadows_factor=0.95  → shadow toe pulled down 5%
        # highlights_factor=1.05 → highlight shoulder lifted 5%
        # midtone_contrast=1.12 → gamma-like lift in midtones
        shadows_factor = 0.95
        highlights_factor = 1.05
        midtone_contrast = 1.12

        # Base: slight gamma push for midtones
        mid = np.power(np.clip(x, 1e-8, 1.0), 1.0 / midtone_contrast)

        # Shadow knee: soften shadows below 0.25
        shadow_mask = x < 0.25
        mid = np.where(shadow_mask,
                       mid * (shadows_factor + (1.0 - shadows_factor) * (x / 0.25)),
                       mid)

        # Highlight shoulder: brighten above 0.75
        highlight_mask = x > 0.75
        blend = (x - 0.75) / 0.25
        mid = np.where(highlight_mask,
                       mid + (1.0 - mid) * blend * (highlights_factor - 1.0),
                       mid)

        lut = np.clip(mid, 0.0, 1.0)

    elif curve_name == "strong_s":
        # Stronger S-curve for slide film (Velvia-style contrast punch)
        # shadows_factor=0.88, highlights_factor=1.08, midtone_contrast=1.25
        shadows_factor = 0.88
        highlights_factor = 1.08
        midtone_contrast = 1.25

        mid = np.power(np.clip(x, 1e-8, 1.0), 1.0 / midtone_contrast)

        shadow_mask = x < 0.25
        mid = np.where(shadow_mask,
                       mid * (shadows_factor + (1.0 - shadows_factor) * (x / 0.25)),
                       mid)

        highlight_mask = x > 0.75
        blend = (x - 0.75) / 0.25
        mid = np.where(highlight_mask,
                       mid + (1.0 - mid) * blend * (highlights_factor - 1.0),
                       mid)

        lut = np.clip(mid, 0.0, 1.0)

    else:
        logger.warning("Unknown curve type %r — using linear", curve_name)
        lut = x

    return lut.astype(np.float32)


# Pre-build all LUTs once at import time
_CURVE_LUTS: Dict[str, np.ndarray] = {
    name: _build_lut(name)
    for name in ("linear", "gentle_s", "strong_s")
}


def apply_curve(arr: np.ndarray, curve_name: str) -> np.ndarray:
    """Apply a named tone curve to a float32 image via LUT interpolation.

    Args:
        arr:        float32 (H, W, 3) in [0, 1].
        curve_name: "linear" | "gentle_s" | "strong_s"

    Returns:
        float32 (H, W, 3) in [0, 1].
    """
    if curve_name == "linear":
        return arr  # fast-path: no copy needed

    lut = _CURVE_LUTS.get(curve_name, _CURVE_LUTS["linear"])

    # Map pixel values [0,1] → LUT indices [0,255], interpolate
    clamped = np.clip(arr, 0.0, 1.0)
    idx_float = clamped * 255.0
    idx_lo = np.floor(idx_float).astype(np.int32)
    idx_hi = np.minimum(idx_lo + 1, 255)
    frac = (idx_float - idx_lo).astype(np.float32)

    result = lut[idx_lo] * (1.0 - frac) + lut[idx_hi] * frac
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Profile application
# ---------------------------------------------------------------------------

def apply_profile(arr: np.ndarray, profile_key: str) -> np.ndarray:
    """Apply filmstock profile adjustments to an already-inverted, WB-corrected image.

    Call order in pipeline (after invert + levels):
        result = apply_profile(result, filmstock_key)

    Steps applied (in order):
      1. wb_tint  — per-channel colour balance offset
      2. gamma    — gamma curve (arr ** (1/gamma))
      3. saturation — luminance-preserving desaturate/saturate
      4. curve    — tone curve (S-curve or linear)

    Args:
        arr:         float32 (H, W, 3) in [0, 1], post-inversion.
        profile_key: Key in FILMSTOCK_PROFILES.

    Returns:
        float32 (H, W, 3) in [0, 1].
    """
    profile = FILMSTOCK_PROFILES.get(profile_key)
    if profile is None:
        logger.warning("Unknown filmstock profile %r — using auto", profile_key)
        profile = FILMSTOCK_PROFILES["auto"]

    if profile["type"] is None:
        # "auto" profile — no adjustments
        return arr

    result = np.clip(arr, 0.0, 1.0).astype(np.float32)

    # ---- Step 1: wb_tint (per-channel multiplicative balance offset) ----
    tint = profile.get("wb_tint", [1.0, 1.0, 1.0])
    if tint != [1.0, 1.0, 1.0]:
        result[:, :, 0] = np.clip(result[:, :, 0] * tint[0], 0.0, 1.0)
        result[:, :, 1] = np.clip(result[:, :, 1] * tint[1], 0.0, 1.0)
        result[:, :, 2] = np.clip(result[:, :, 2] * tint[2], 0.0, 1.0)
        logger.debug("Profile %r: wb_tint applied %s", profile_key, tint)

    # ---- Step 2: gamma correction ----
    gamma = profile.get("gamma", 1.0)
    if gamma != 1.0:
        # arr ** (1/gamma) — standard gamma encode
        # Clamp to avoid 0**negative = inf
        result = np.power(np.clip(result, 1e-8, 1.0), 1.0 / gamma).astype(np.float32)
        logger.debug("Profile %r: gamma=%.3f applied", profile_key, gamma)

    # ---- Step 3: saturation (luminance-preserving) ----
    saturation = profile.get("saturation", 1.0)
    if saturation != 1.0:
        # Rec.709 luminance coefficients
        lum = (0.2126 * result[:, :, 0:1]
               + 0.7152 * result[:, :, 1:2]
               + 0.0722 * result[:, :, 2:3])
        result = np.clip(lum + saturation * (result - lum), 0.0, 1.0).astype(np.float32)
        logger.debug("Profile %r: saturation=%.3f applied", profile_key, saturation)

    # ---- Step 4: tone curve ----
    curve = profile.get("curve", "linear")
    result = apply_curve(result, curve)

    logger.debug("Profile %r applied: gamma=%.2f sat=%.2f curve=%s", profile_key, gamma, saturation, curve)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_profile_labels() -> Dict[str, str]:
    """Return {profile_key: display_label} for populating a dropdown.

    Order: "auto" first, then C-41, then E-6, then B&W.
    """
    order = ["auto"]
    c41  = [k for k, v in FILMSTOCK_PROFILES.items() if v.get("type") == "C-41"]
    e6   = [k for k, v in FILMSTOCK_PROFILES.items() if v.get("type") == "E-6"]
    bw   = [k for k, v in FILMSTOCK_PROFILES.items() if v.get("type") == "B&W"]
    order += c41 + e6 + bw
    return {k: FILMSTOCK_PROFILES[k]["label"] for k in order}


def get_profile_keys_by_label() -> Dict[str, str]:
    """Return {display_label: profile_key} — inverse of get_profile_labels()."""
    return {v["label"]: k for k, v in FILMSTOCK_PROFILES.items()}


def detect_filmtype(mask_rgb: List[float]) -> str:
    """Heuristic filmstock hint from mask sample RGB (Stufe 3 data-collection hook).

    This is a coarse first-guess, NOT a hard override. The user's dropdown
    selection always takes precedence.

    Logic:
      1. If R ≈ G ≈ B (max deviation < 0.08) → E-6/B&W → return "auto"
      2. R/B ratio > 4.0 → Kodak-warm → "kodak_gold_200"
      3. R/B ratio < 3.5 → Fuji-cool  → "fuji_superia_400"
      4. Otherwise       → generic warm → "kodak_gold_200"

    Args:
        mask_rgb: [R, G, B] float values from WB sample (e.g. [0.82, 0.51, 0.19])

    Returns:
        A profile key string (suitable for FILMSTOCK_PROFILES lookup).
    """
    if not mask_rgb or len(mask_rgb) < 3:
        return "auto"

    r, g, b = float(mask_rgb[0]), float(mask_rgb[1]), float(mask_rgb[2])
    b = max(b, 1e-6)  # avoid division by zero

    # Check neutrality (E-6/B&W have near-neutral base)
    values = [r, g, b]
    deviation = max(values) - min(values)
    if deviation < 0.08:
        logger.debug("detect_filmtype: near-neutral mask (dev=%.3f) → auto", deviation)
        return "auto"

    ratio = r / b
    if ratio > 4.0:
        logger.debug("detect_filmtype: R/B=%.2f → Kodak-warm hint", ratio)
        return "kodak_gold_200"
    elif ratio < 3.5:
        logger.debug("detect_filmtype: R/B=%.2f → Fuji-cool hint", ratio)
        return "fuji_superia_400"
    else:
        logger.debug("detect_filmtype: R/B=%.2f → generic warm hint", ratio)
        return "kodak_gold_200"
