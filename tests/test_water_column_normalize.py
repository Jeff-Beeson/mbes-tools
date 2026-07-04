"""Tests for mbes_tools.water_column_normalize (empirical de-trending).

Synthetic frames with a *known* injected range- and beam-angle-dependent gain;
the normalizer must flatten both trends over the open water while leaving a real
off-background target standing out. No data files, no matplotlib.
"""
import warnings

import numpy as np
import pytest

from mbes_tools import water_column_normalize as wn
from mbes_tools.wc_diagnostics import WCFrame


def _profiles(norm):
    """Per-range and per-beam medians of a masked grid (all-NaN slices silenced)."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(norm, axis=0), np.nanmedian(norm, axis=1)


def make_frame(amp_db, angles_deg, detected_samples, c=1500.0, fs=750.0):
    amp = np.asarray(amp_db, float)
    return WCFrame(
        amp_db=amp,
        phase_deg=None,
        angles_deg=np.asarray(angles_deg, float),
        detected_samples=np.asarray(detected_samples, int),
        sound_speed_m_s=c,
        sample_freq_hz=fs,
        phase_flag=0,
        sector_freqs_hz=[],
        label="synthetic",
    )


def _synthetic_frame(nb=40, width=200, bottom=180, base=-50.0,
                     range_slope=-0.06, angle_slope=0.15, seed=0, noise=0.0):
    """Water field = base + range_trend(k) + angle_trend(beam) [+ noise], with a
    seafloor at ``bottom`` and near-field below sample 5 blanked out."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(-60.0, 60.0, nb)
    k = np.arange(width)
    range_trend = range_slope * k                    # TVG-like ramp vs range
    angle_trend = angle_slope * np.abs(angles)       # beam-pattern-like vs angle
    amp = base + range_trend[None, :] + angle_trend[:, None]
    if noise:
        amp = amp + rng.normal(0.0, noise, amp.shape)
    # Blank the near field and everything at/after the seafloor (as a real ping).
    amp[:, :5] = np.nan
    for b in range(nb):
        amp[b, bottom:] = np.nan
    det = np.full(nb, bottom, dtype=int)
    return make_frame(amp, angles, det), range_trend, angle_trend


def _water_region(frame, guard=5):
    """Boolean [beam,width] of the finite open-water samples (for assertions)."""
    amp = frame.amp_db
    k = np.arange(amp.shape[1])
    det = frame.detected_samples
    mask = np.isfinite(amp) & (k[None, :] >= guard) & (k[None, :] < (det - guard)[:, None])
    return mask


def test_normalization_flattens_range_and_angle_trends():
    frame, _, _ = _synthetic_frame()
    out = wn.normalize_frame(frame, method="empirical", n_iter=4)
    mask = _water_region(frame)
    norm = np.where(mask, out.amp_db, np.nan)

    # Per-range spread (across beams) and per-beam spread (across ranges) of the
    # water background must collapse: an additive row+col model is fully removable.
    range_profile, angle_profile = _profiles(norm)
    range_profile = range_profile[np.isfinite(range_profile)]
    angle_profile = angle_profile[np.isfinite(angle_profile)]

    # Before: the injected trends give a large spread; after: near-flat.
    assert np.std(range_profile) < 0.05
    assert np.std(angle_profile) < 0.05

    # And the flattened water sits near the reported reference level.
    assert np.nanmedian(norm) == pytest.approx(out_ref := wn.detrend_amplitude(
        frame.amp_db, _water_region(frame), n_iter=4).reference_db, abs=0.5)


def test_normalization_preserves_a_real_target():
    frame, _, _ = _synthetic_frame()
    # Inject a bright midwater target well above the local background.
    frame.amp_db[20, 90] = -10.0
    bg_before = np.nanmedian(frame.amp_db)
    out = wn.normalize_frame(frame, method="empirical", n_iter=4)
    # The target must still stand well clear of the flattened background.
    assert out.amp_db[20, 90] - np.nanmedian(out.amp_db) > 20.0


def test_normalize_none_is_identity():
    frame, _, _ = _synthetic_frame()
    out = wn.normalize_frame(frame, method="none")
    assert out is frame


def test_normalize_rejects_bad_method():
    frame, _, _ = _synthetic_frame()
    with pytest.raises(ValueError, match="normalize method"):
        wn.normalize_frame(frame, method="sv")  # deferred path, not yet available


def test_normalize_preserves_frame_fields():
    frame, _, _ = _synthetic_frame()
    out = wn.normalize_frame(frame, method="empirical")
    assert out.amp_db.shape == frame.amp_db.shape
    assert np.array_equal(out.angles_deg, frame.angles_deg)
    assert np.array_equal(out.detected_samples, frame.detected_samples)
    assert out.sound_speed_m_s == frame.sound_speed_m_s
    assert out.sample_freq_hz == frame.sample_freq_hz
    assert out.label == frame.label


def test_detrend_handles_empty_water_mask():
    frame, _, _ = _synthetic_frame()
    empty = np.zeros(frame.amp_db.shape, dtype=bool)
    res = wn.detrend_amplitude(frame.amp_db, empty, n_iter=3)
    # No water to fit -> zero effects, reference 0; amplitude passes through.
    assert res.reference_db == 0.0
    assert np.allclose(res.range_effect_db, 0.0)
    assert np.allclose(res.angle_effect_db, 0.0)
    np.testing.assert_array_equal(
        np.nan_to_num(res.normalized_db, nan=-999),
        np.nan_to_num(frame.amp_db, nan=-999),
    )


def test_frame_normalizer_factory():
    assert wn.frame_normalizer(None) is None
    assert wn.frame_normalizer("none") is None
    fn = wn.frame_normalizer("empirical", n_iter=2)
    frame, _, _ = _synthetic_frame()
    out = fn(frame)
    assert out.amp_db.shape == frame.amp_db.shape


def test_normalization_robust_with_noise():
    frame, _, _ = _synthetic_frame(noise=1.0, seed=5)
    out = wn.normalize_frame(frame, method="empirical", n_iter=4)
    mask = _water_region(frame)
    norm = np.where(mask, out.amp_db, np.nan)
    range_profile, angle_profile = _profiles(norm)
    # With noise the trend removal is approximate but still collapses the spread
    # (injected trends span ~10-18 dB; residual medians should be well under 1 dB).
    assert np.nanstd(range_profile) < 0.6
    assert np.nanstd(angle_profile) < 0.6
