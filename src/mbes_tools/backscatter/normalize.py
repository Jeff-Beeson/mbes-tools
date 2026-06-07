"""Lambertian sector/angle backscatter normalization (headless compute).

These functions implement the first-pass Lambertian balancing that the
interactive balancer GUI (``mbes_tools.scripts.sector_balancer_gui``) drives,
factored out so they can be used without Tk. The model assumes flat-seafloor
backscatter follows a Lambertian angular shape::

    intensity(theta) ~= intercept + 10 * log10(cos(theta))

where ``theta`` is the absolute beam angle from nadir. Fitting an intercept to
the observed angular curve gives a reference; the per-sector offset that aligns
each sector's curve to that reference is the correction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Kongsberg depth-mode labels after the +1 shift applied when loading a
# generate-table CSV (raw KMALL depthMode 1 -> 2 Shallow, etc.). These are
# vessel-generic for EM-series systems.
DEPTH_MODE_LABELS = {
    2: "Shallow",
    3: "Medium",
    4: "Deep",
    5: "Deeper",
    6: "Very Deep",
    7: "Extra Deep",
    8: "Extreme Deep",
}

# Depth modes that operate as a single swath (only fan index 0 is meaningful).
SINGLE_SWATH_MODES = {6, 7, 8}

# Minimum number of valid samples required to fit a Lambertian intercept.
MIN_LAMBERT_SAMPLES = 5


def _lambert_valid_mask(
    abs_angle_deg: np.ndarray,
    values_db: np.ndarray,
    cos_theta: np.ndarray,
    min_angle_deg: float,
    max_angle_deg: float,
) -> np.ndarray:
    """Boolean mask of samples usable for a Lambertian fit."""
    return (
        np.isfinite(values_db)
        & np.isfinite(cos_theta)
        & (cos_theta > 0.05)
        & (values_db > -32767)
        & (abs_angle_deg >= min_angle_deg)
        & (abs_angle_deg <= max_angle_deg)
    )


def fit_lambert_intercept(
    angles_deg: np.ndarray,
    values_db: np.ndarray,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 75.0,
) -> Optional[float]:
    """Fit the Lambertian intercept (in dB) for one angular backscatter curve.

    The intercept is the median of ``values_db - 10*log10(cos(theta))`` over the
    valid angular window. Returns ``None`` when fewer than
    ``MIN_LAMBERT_SAMPLES`` valid samples are available.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    values_db = np.asarray(values_db, dtype=float)
    abs_angle = np.abs(angles_deg)
    cos_theta = np.cos(np.deg2rad(abs_angle))

    valid = _lambert_valid_mask(abs_angle, values_db, cos_theta, min_angle_deg, max_angle_deg)
    if np.count_nonzero(valid) < MIN_LAMBERT_SAMPLES:
        return None

    lambert_shape = 10.0 * np.log10(cos_theta[valid])
    return float(np.nanmedian(values_db[valid] - lambert_shape))


def lambert_reference_curve(angles_deg: np.ndarray, intercept: float) -> np.ndarray:
    """Evaluate the Lambertian reference curve at the given angles.

    Returns dB values, with NaN where ``cos(theta)`` is too small to be valid.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    cos_theta = np.cos(np.deg2rad(np.abs(angles_deg)))
    valid = cos_theta > 0.05
    reference = np.full_like(angles_deg, np.nan, dtype=float)
    reference[valid] = intercept + 10.0 * np.log10(cos_theta[valid])
    return reference


def solve_sector_shifts(
    angles_deg: np.ndarray,
    values_db: np.ndarray,
    sectors: np.ndarray,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 75.0,
    max_shift_db: float = 15.0,
) -> Dict[int, float]:
    """Solve the per-sector dB shift that aligns each sector to the Lambertian fit.

    Args:
        angles_deg: Beam angle per sample.
        values_db: (Already-adjusted) intensity per sample, in dB.
        sectors: TX sector number per sample.
        min_angle_deg / max_angle_deg: Angular window used for fitting.
        max_shift_db: Shifts are clipped to +/- this magnitude.

    Returns:
        Mapping of sector number -> additive correction in dB. Sectors with
        too few valid samples are omitted.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    values_db = np.asarray(values_db, dtype=float)
    sectors = np.asarray(sectors, dtype=int)

    abs_angle = np.abs(angles_deg)
    cos_theta = np.cos(np.deg2rad(abs_angle))
    valid = _lambert_valid_mask(abs_angle, values_db, cos_theta, min_angle_deg, max_angle_deg)
    if np.count_nonzero(valid) < MIN_LAMBERT_SAMPLES:
        return {}

    lambert_shape = 10.0 * np.log10(cos_theta[valid])
    intercept = float(np.nanmedian(values_db[valid] - lambert_shape))

    shifts: Dict[int, float] = {}
    for sector in sorted(np.unique(sectors[valid])):
        sector_i = int(sector)
        sector_mask = valid & (sectors == sector_i)
        if np.count_nonzero(sector_mask) < 2:
            continue
        sector_target = intercept + 10.0 * np.log10(cos_theta[sector_mask])
        residual = values_db[sector_mask] - sector_target
        shift = float(np.clip(-np.nanmedian(residual), -max_shift_db, max_shift_db))
        shifts[sector_i] = shift

    return shifts


def valid_fans_for_mode(mode: int, unique_fans: np.ndarray) -> np.ndarray:
    """Return the receive-fan indices that are meaningful for a depth mode."""
    if int(mode) in SINGLE_SWATH_MODES:
        return np.array([0])
    return np.asarray(unique_fans)
