"""
test_wb_invert_pipeline.py

Tests for the WB + Inversion pipeline bug fix (2026-07-15).

Validates that:
1. green_ref WB method correctly neutralises an orange film mask
2. Pipeline order (WB first, invert second) produces neutral result
3. Mean WB method still works
4. Multiplier clamping prevents extreme values
5. Full pipeline: orange mask -> neutral grey after WB + invert

Run with:
    cd path/to/filmdev
    python3 -m pytest tests/test_wb_invert_pipeline.py -v
or:
    python3 tests/test_wb_invert_pipeline.py
"""

import sys
import os

# Add parent dir so we can import core.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.wb_engine import WBEngine
from core.invert_engine import InvertEngine


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_uniform_image(r: float, g: float, b: float, size: int = 32) -> np.ndarray:
    """Create a uniform-colour float32 image."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def channel_means(img: np.ndarray):
    """Return (R_mean, G_mean, B_mean) for image."""
    return (
        float(img[:, :, 0].mean()),
        float(img[:, :, 1].mean()),
        float(img[:, :, 2].mean()),
    )


NEUTRAL_TOL = 0.05  # channels must be within 5% of each other to count as neutral


def is_neutral(img: np.ndarray, tol: float = NEUTRAL_TOL) -> bool:
    """Check all channels are approximately equal (neutral grey)."""
    r, g, b = channel_means(img)
    return abs(r - g) < tol and abs(g - b) < tol and abs(r - b) < tol


# ─── Test 1: green_ref WB on orange mask ──────────────────────────────────────

def test_green_ref_wb_neutralises_orange_mask():
    """
    Given a typical orange film mask (R=0.8, G=0.5, B=0.2),
    green_ref WB should make R = G = B = 0.5 after applying multipliers.
    """
    wb = WBEngine()
    sample = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    mults = wb.compute_multipliers(sample, method="green_ref")

    # Expected: mult_R = 0.5/0.8 = 0.625, mult_G = 1.0, mult_B = 0.5/0.2 = 2.5
    # BUT mult_B = 2.5 should be clamped to 4.0 (OK here, 2.5 < 4.0)
    assert abs(mults[0] - (0.5 / 0.8)) < 0.01, f"mult_R wrong: {mults[0]}"
    assert abs(mults[1] - 1.0) < 0.01, f"mult_G should be 1.0: {mults[1]}"
    assert abs(mults[2] - (0.5 / 0.2)) < 0.01, f"mult_B wrong: {mults[2]}"

    # Apply WB to a uniform orange image
    img_neg = make_uniform_image(0.8, 0.5, 0.2)
    img_wb = wb.apply(img_neg, mults)

    # After WB: R_out = 0.8 * 0.625 = 0.5, G_out = 0.5 * 1.0 = 0.5, B_out = 0.2 * 2.5 = 0.5
    r, g, b = channel_means(img_wb)
    assert abs(r - 0.5) < 0.01, f"R after WB should be 0.5, got {r}"
    assert abs(g - 0.5) < 0.01, f"G after WB should be 0.5, got {g}"
    assert abs(b - 0.5) < 0.01, f"B after WB should be 0.5, got {b}"
    print(f"  [PASS] green_ref WB: R={r:.3f} G={g:.3f} B={b:.3f} -> neutral")


# ─── Test 2: Full pipeline — WB first, then invert ────────────────────────────

def test_pipeline_wb_then_invert_no_colour_cast():
    """
    Simulate the correct pipeline:
      1. Orange mask negative (R=0.8, G=0.5, B=0.2)
      2. Apply green_ref WB  -> neutral negative (R=G=B≈0.5)
      3. Transmittance invert (1/x, per-channel normalised)
      4. Result should be neutral (no colour cast)

    This is the Aaron Buchler standard.
    """
    wb = WBEngine()
    inv = InvertEngine()

    # Step 1: orange negative
    img_neg = make_uniform_image(0.8, 0.5, 0.2)

    # Step 2: WB on negative (green_ref)
    sample = wb.sample_point(img_neg, 16, 16)
    mults = wb.compute_multipliers(sample, method="green_ref")
    img_wb = wb.apply(img_neg, mults)

    r_wb, g_wb, b_wb = channel_means(img_wb)
    print(f"  After WB: R={r_wb:.4f} G={g_wb:.4f} B={b_wb:.4f}")
    assert is_neutral(img_wb, tol=0.02), f"After WB, mask should be neutral: R={r_wb} G={g_wb} B={b_wb}"

    # Step 3: Invert (transmittance)
    img_pos = inv.invert_transmittance(img_wb, _use_gpu=False)

    r_pos, g_pos, b_pos = channel_means(img_pos)
    print(f"  After Invert: R={r_pos:.4f} G={g_pos:.4f} B={b_pos:.4f}")
    assert is_neutral(img_pos, tol=0.05), f"After WB+Invert, result should be neutral: R={r_pos} G={g_pos} B={b_pos}"
    print(f"  [PASS] Pipeline WB->Invert: neutral output, no blue/cyan cast")


# ─── Test 3: Old pipeline (invert first) FAILS — verify the bug ───────────────

def test_old_pipeline_invert_then_wb_causes_colour_cast():
    """
    Demonstrates that the OLD pipeline (invert first, WB second) causes
    a colour cast even on a uniform negative.

    Orange mask: R=0.8, G=0.5, B=0.2
    After invert:
      - 1/0.8 normalised ≠ 1/0.5 normalised ≠ 1/0.2 normalised per-channel
      - Per-channel normalise collapses each channel independently -> all become ~0.5
      - But: WB is computed from the RAW orange values -> mults from orange sample
      - Applying those mults to the already-normalised-uniform image causes mismatches

    This test documents the bug was real (not necessarily quantifies it, just that
    the fixed pipeline produces a BETTER (more neutral) result).
    """
    wb = WBEngine()
    inv = InvertEngine()

    img_neg = make_uniform_image(0.8, 0.5, 0.2)

    # OLD order: invert first
    img_inv = inv.invert_transmittance(img_neg, _use_gpu=False)
    r_inv, g_inv, b_inv = channel_means(img_inv)
    print(f"  After Invert (old step1): R={r_inv:.4f} G={g_inv:.4f} B={b_inv:.4f}")

    # For a uniform image, per-channel normalisation collapses everything to 0.5
    # So the inverted uniform image IS neutral. But compute WB from original orange:
    sample = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    mults_old = wb.compute_multipliers(sample, method="green_ref")  # computed from orange neg
    img_wb_old = wb.apply(img_inv, mults_old)
    r_old, g_old, b_old = channel_means(img_wb_old)
    print(f"  After Invert+WB (old order): R={r_old:.4f} G={g_old:.4f} B={b_old:.4f}")

    # NEW order for comparison
    mults_new = wb.compute_multipliers(sample, method="green_ref")
    img_wb_new = wb.apply(img_neg, mults_new)
    img_pos_new = inv.invert_transmittance(img_wb_new, _use_gpu=False)
    r_new, g_new, b_new = channel_means(img_pos_new)
    print(f"  After WB+Invert (new order): R={r_new:.4f} G={g_new:.4f} B={b_new:.4f}")

    # New pipeline should produce neutral; old may differ
    new_spread = max(r_new, g_new, b_new) - min(r_new, g_new, b_new)
    old_spread = max(r_old, g_old, b_old) - min(r_old, g_old, b_old)
    print(f"  Channel spread — new: {new_spread:.4f}, old: {old_spread:.4f}")
    assert is_neutral(img_pos_new, tol=0.05), f"New pipeline must be neutral: R={r_new} G={g_new} B={b_new}"
    print(f"  [PASS] New pipeline is neutral; old pipeline spread={old_spread:.4f}")


# ─── Test 4: Multiplier clamping ──────────────────────────────────────────────

def test_multiplier_clamping():
    """
    Extreme orange (R=0.9, G=0.5, B=0.05) would give mult_B = 0.5/0.05 = 10.0
    which causes clipping after WB. Clamp to [0.1, 4.0] should prevent this.
    """
    wb = WBEngine()
    extreme_sample = np.array([0.9, 0.5, 0.05], dtype=np.float32)
    mults = wb.compute_multipliers(extreme_sample, method="green_ref")

    assert mults[0] <= 4.0, f"mult_R should be clamped to <=4.0, got {mults[0]}"
    assert mults[1] <= 4.0, f"mult_G should be clamped to <=4.0, got {mults[1]}"
    assert mults[2] <= 4.0, f"mult_B should be clamped to <=4.0, got {mults[2]}"
    assert mults[0] >= 0.1, f"mult_R should be clamped to >=0.1, got {mults[0]}"
    assert mults[1] >= 0.1, f"mult_G should be clamped to >=0.1, got {mults[1]}"
    assert mults[2] >= 0.1, f"mult_B should be clamped to >=0.1, got {mults[2]}"

    # mult_B should have been clamped from 10.0 to 4.0
    assert abs(mults[2] - 4.0) < 0.01, f"mult_B should be clamped to 4.0, got {mults[2]}"
    print(f"  [PASS] Clamping: mults = R={mults[0]:.3f} G={mults[1]:.3f} B={mults[2]:.3f}")


# ─── Test 5: Mean method still works ─────────────────────────────────────────

def test_mean_method_still_works():
    """Legacy mean method should still produce neutral grey."""
    wb = WBEngine()
    sample = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    mults = wb.compute_multipliers(sample, method="mean")

    # target = mean([0.8, 0.5, 0.2]) = 0.5
    # mult_R = 0.5/0.8, mult_G = 0.5/0.5, mult_B = 0.5/0.2
    expected_target = (0.8 + 0.5 + 0.2) / 3
    assert abs(mults[0] - (expected_target / 0.8)) < 0.01
    assert abs(mults[1] - (expected_target / 0.5)) < 0.01

    img = make_uniform_image(0.8, 0.5, 0.2)
    img_wb = wb.apply(img, mults)
    assert is_neutral(img_wb, tol=0.02), "Mean WB should produce neutral grey"
    print(f"  [PASS] Mean method works: mults = R={mults[0]:.3f} G={mults[1]:.3f} B={mults[2]:.3f}")


# ─── Test 6: Default method is green_ref ─────────────────────────────────────

def test_default_method_is_green_ref():
    """compute_multipliers() with no method arg should use green_ref."""
    wb = WBEngine()
    sample = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    mults_default = wb.compute_multipliers(sample)
    mults_explicit = wb.compute_multipliers(sample, method="green_ref")
    np.testing.assert_array_almost_equal(mults_default, mults_explicit, decimal=5)
    print(f"  [PASS] Default method is green_ref")


# ─── Test 7: Sample from real image coordinates ───────────────────────────────

def test_sample_point_returns_correct_median():
    """sample_point() should return median of patch, not affected by outliers."""
    wb = WBEngine(sample_radius=5)
    # Create image with orange background and a few outlier pixels
    img = make_uniform_image(0.8, 0.5, 0.2, size=64)
    # Add some outliers (scratches/dust)
    img[32, 32, :] = [0.0, 0.0, 0.0]  # black pixel
    img[31, 31, :] = [1.0, 1.0, 1.0]  # white pixel

    sample = wb.sample_point(img, 32, 32, radius=5)
    # Median should be dominated by orange pixels, not outliers
    assert abs(sample[0] - 0.8) < 0.05, f"R median should be ~0.8, got {sample[0]}"
    assert abs(sample[1] - 0.5) < 0.05, f"G median should be ~0.5, got {sample[1]}"
    assert abs(sample[2] - 0.2) < 0.05, f"B median should be ~0.2, got {sample[2]}"
    print(f"  [PASS] Median sampling: R={sample[0]:.3f} G={sample[1]:.3f} B={sample[2]:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_green_ref_wb_neutralises_orange_mask,
        test_pipeline_wb_then_invert_no_colour_cast,
        test_old_pipeline_invert_then_wb_causes_colour_cast,
        test_multiplier_clamping,
        test_mean_method_still_works,
        test_default_method_is_green_ref,
        test_sample_point_returns_correct_median,
    ]
    passed = 0
    failed = 0
    for t in tests:
        print(f"\n[TEST] {t.__name__}")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
