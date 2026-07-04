"""Tests for mbes_tools.wc_sample_bound (per-ping decode upper sample bound)."""
import numpy as np
import pytest

from mbes_tools.wc_sample_bound import SampleBoundSpec, sample_upper_bound

# c=1500, fs=750 -> samp_per_m = 2*fs/c = 1 sample per metre of one-way range,
# so range-sample index == one-way range in metres (handy for the arithmetic).
C, FS = 1500.0, 750.0
PAD = SampleBoundSpec().pad_samples  # 2


def test_none_when_no_active_filter():
    assert sample_upper_bound(None, [0.0], [0], C, FS) is None
    assert sample_upper_bound(SampleBoundSpec(), [0.0], [0], C, FS) is None


def test_clean_water_bounds_at_min_detection():
    det = [80, 50, 60, 0]  # min positive detection = 50; the 0 (no bottom) ignored
    spec = SampleBoundSpec(clean_water=True)
    assert sample_upper_bound(spec, [0.0, 10, 20, 30], det, C, FS) == 50 + PAD


def test_clean_water_percentile_and_guard():
    det = [30, 80, 82, 84, 86]
    # percentile ignores the shallow outlier -> larger R_min than the strict 30.
    p = sample_upper_bound(SampleBoundSpec(clean_water=True, msr_percentile=25),
                           [0.0] * 5, det, C, FS)
    assert p > 30 + PAD
    # guard (metres -> samples at 1 sample/m) pulls the cutoff inward.
    g = sample_upper_bound(SampleBoundSpec(clean_water=True, msr_guard_m=10.0),
                           [0.0] * 5, det, C, FS)
    assert g == 30 - 10 + PAD


def test_depth_ceiling_uses_outermost_beam():
    # k <= depth / cosθ (samp_per_m=1); outermost beam (60°, cos .5) gives the max.
    import math
    spec = SampleBoundSpec(max_depth_m=100.0)
    got = sample_upper_bound(spec, [0.0, 60.0], [0, 0], C, FS)
    assert got == math.ceil(100.0 / np.cos(np.radians(60.0))) + PAD  # ~200 + pad


def test_depth_band_hi_and_max_depth_take_the_tighter():
    # depth_band hi=50 is tighter than max_depth_m=100 -> 50 wins.
    spec = SampleBoundSpec(max_depth_m=100.0, depth_band=(10.0, 50.0))
    got = sample_upper_bound(spec, [0.0], [0], C, FS)
    assert got == 50 + PAD


def test_and_composition_takes_the_min():
    # clean-water (min det 50) AND depth ceiling (~200) -> 50 binds.
    spec = SampleBoundSpec(clean_water=True, max_depth_m=200.0)
    got = sample_upper_bound(spec, [0.0, 60.0], [50, 90], C, FS)
    assert got == 50 + PAD


def test_altitude_band_bounds_near_deepest_bottom():
    # keep hab in [20, hi]; deepest kept sample per beam at hab=20 -> k = det - 20/cosθ.
    det = [100, 120]
    spec = SampleBoundSpec(altitude_band=(20.0, 400.0))
    got = sample_upper_bound(spec, [0.0, 0.0], det, C, FS)
    assert got == (120 - 20) + PAD  # deepest beam det=120, minus 20 m, +pad


def test_bound_is_conservative_never_below_a_kept_sample():
    # A depth ceiling must not cut below the deepest in-band sample of any beam.
    det = [0, 0]
    spec = SampleBoundSpec(depth_band=(0.0, 300.0))
    k = sample_upper_bound(spec, [0.0, 45.0], det, C, FS)
    # 45° beam keeps depth<=300 -> k up to 300/cos45 ≈ 424; bound must be >= that.
    assert k >= 300.0 / np.cos(np.radians(45.0))
