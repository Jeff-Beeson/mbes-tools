"""Empirical normalization of water-column amplitude (D1 products).

Water-column amplitude as logged is dominated by **acquisition gain**, not by the
water itself: a strong range-dependent trend (the applied TVG plus absorption and
spreading) and a beam-angle-dependent trend (beam pattern, array shading,
port/starboard asymmetry). Those swamp the real, comparatively weak scattering
structure (midwater layers, plumes, wakes).

This module removes both trends **empirically** — no instrument-model assumptions
and nothing beyond what the datagram already gives — over the *open water* only
(near-field and the seafloor return excluded, reusing
:func:`mbes_tools.water_column.water_column_water_mask`). It is a two-way additive
**median polish** in the dB domain: alternately subtract the per-range
(across-beam) median and the per-beam-angle (across-range) median a few times, so
the range (TVG/absorption) trend and the angle (beam-pattern) trend are each
peeled off. The fitted trends are then removed from the *whole* grid and the
frame's water level is re-anchored to a common reference (the frame's water
median), so real off-background returns — plumes, the seafloor — stand out while
the flat water background reads the same across range, angle, and ping.

This is a **relative** normalization (a de-biased dB amplitude), not a calibrated
volume-scattering strength (``Sv``): source level and beam solid angle are absent
from ``#MWC``/``k``, so absolute calibration is impossible from these files alone.
That (a semi-physical relative-``Sv`` path using the recorded TVG law + a supplied
absorption) is a deliberate follow-up; see ``docs/WATER_COLUMN_HANDOFF.md``.

Core is numpy + stdlib; it is reused by ``mbes-wc-mosaic``, ``mbes-wc-grid`` and
``mbes-wc-viewer`` via ``--normalize empirical``.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mbes_tools.water_column import water_column_water_mask
from mbes_tools.wc_diagnostics import WCFrame, range_axis

NORMALIZE_METHODS = ("none", "empirical", "sv")

# Volume-scattering (Sv) spreading law: 20·log10(R). The seafloor/reflectivity
# default the instrument applies is 30·log10(R) (Kongsberg TVGfunctionApplied),
# so the Sv re-expression adds (20 − X)·log10(R).
_SV_TVG = 20.0


@dataclass
class NormalizeResult:
    """Diagnostics from :func:`detrend_amplitude` (for review / before-after panels)."""

    normalized_db: np.ndarray      # [beam, width], de-trended amplitude
    range_effect_db: np.ndarray    # [width], fitted per-range (TVG/absorption) trend
    angle_effect_db: np.ndarray    # [beam], fitted per-beam-angle (beam-pattern) trend
    reference_db: float            # common water level the background is anchored to
    water_mask: np.ndarray         # [beam, width] samples used to fit the trends
    n_iter: int


def detrend_amplitude(
    amp_db: np.ndarray,
    water_mask: np.ndarray,
    *,
    n_iter: int = 3,
) -> NormalizeResult:
    """Two-way median-polish de-trend of a ``[beam, width]`` dB amplitude grid.

    ``water_mask`` marks the open-water samples the trends are *fitted* from
    (near-field and seafloor excluded). The fitted per-range and per-beam trends
    are then subtracted from **every** sample and the water level is re-anchored
    to the frame's water median, so ``normalized_db`` is comparable across range,
    angle and pings. Ranges or beams with no water samples get a zero effect
    (nothing to estimate), so they pass through only re-anchored by the reference.
    """
    amp = np.asarray(amp_db, dtype=float)
    masked = np.where(water_mask, amp, np.nan)

    nb, width = amp.shape
    col_eff = np.zeros(width)     # per-range (TVG/absorption)
    row_eff = np.zeros(nb)        # per-beam-angle (beam pattern)
    resid = masked.copy()

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        reference = float(np.nanmedian(masked)) if np.isfinite(masked).any() else 0.0
        for _ in range(max(1, n_iter)):
            cm = np.nanmedian(resid, axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            col_eff += cm
            resid = resid - cm[None, :]
            rm = np.nanmedian(resid, axis=1)
            rm = np.where(np.isfinite(rm), rm, 0.0)
            row_eff += rm
            resid = resid - rm[:, None]

    normalized = amp - col_eff[None, :] - row_eff[:, None] + reference
    return NormalizeResult(
        normalized_db=normalized,
        range_effect_db=col_eff,
        angle_effect_db=row_eff,
        reference_db=reference,
        water_mask=water_mask,
        n_iter=max(1, n_iter),
    )


def sv_relative(
    frame: WCFrame, *, absorption_db_km: float = 0.0, target_tvg: float = _SV_TVG
) -> WCFrame:
    """Re-express the frame's amplitude as **relative volume-scattering ``Sv``**.

    The logged amplitude has the instrument's ``X·log10(R)`` TVG applied
    (``X`` = ``TVGfunctionApplied``, default 30 — a seafloor/reflectivity law) plus
    a constant offset ``OFS`` (``TVGoffset_dB``). Volume scattering wants the
    ``20·log10(R)`` law, so per one-way slant range ``R`` (from ``c·k/(2·fs)``):

        Sv_rel = amp_dB − OFS + (target_tvg − X)·log10(R) + 2·α·R

    ``α`` is the absorption coefficient (``absorption_db_km`` → dB/m); it is **not**
    in ``#MWC``/``k`` (only on the ``.all``/``.wcd`` runtime datagram), so it is
    supplied explicitly and defaults to 0. This is a **relative** quantity: the
    source level and equivalent beam solid angle needed for absolute ``Sv`` are not
    in these datagrams, so an unknown constant remains — only the range dependence
    is corrected. Raises if the frame lacks the applied-TVG constants (e.g. a
    synthetic frame).
    """
    X, ofs = frame.tvg_function_x, frame.tvg_offset_db
    if X is None or ofs is None:
        raise ValueError(
            "Sv normalization needs the applied-TVG constants (TVGfunctionApplied, "
            "TVGoffset_dB) from the #MWC/k datagram, which this frame does not carry."
        )
    r = range_axis(frame.width, frame.sound_speed_m_s, frame.sample_freq_hz)  # one-way R [width]
    alpha_db_m = absorption_db_km / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.where(r > 0, np.log10(np.where(r > 0, r, 1.0)), np.nan)
    corr = (target_tvg - float(X)) * logr + 2.0 * alpha_db_m * r  # [width]
    sv = frame.amp_db - float(ofs) + corr[None, :]
    return dataclasses.replace(frame, amp_db=sv)


def normalize_frame(
    frame: WCFrame,
    *,
    method: str = "empirical",
    n_iter: int = 3,
    surface_guard_samples: int = 5,
    min_range_m: float = 0.0,
    bottom_guard_samples: int = 5,
    bottom_guard_m: Optional[float] = None,
    absorption_db_km: float = 0.0,
) -> WCFrame:
    """Return a copy of ``frame`` with acquisition gain removed from ``amp_db``.

    ``method="none"`` returns the frame unchanged. ``method="empirical"`` builds
    the open-water mask (:func:`water_column_water_mask`, honouring the same
    near-field / seafloor guards as the anomaly detector) and de-trends via
    :func:`detrend_amplitude`. ``method="sv"`` re-expresses the amplitude as
    relative volume-scattering ``Sv`` (:func:`sv_relative`, using the recorded
    applied-TVG constants + ``absorption_db_km``). All other frame fields (angles,
    bottom detection, sound speed, sample rate, phase, label) are preserved, so the
    normalized frame drops straight into gridding / georeferencing / the viewer.
    """
    if method not in NORMALIZE_METHODS:
        raise ValueError(f"normalize method must be one of {NORMALIZE_METHODS}, got {method!r}")
    if method == "none":
        return frame
    if method == "sv":
        return sv_relative(frame, absorption_db_km=absorption_db_km)

    guard = bottom_guard_samples
    if bottom_guard_m is not None:
        c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
        guard = max(0, int(round(2.0 * fs * bottom_guard_m / c)))
    mask = water_column_water_mask(
        frame,
        surface_guard_samples=surface_guard_samples,
        min_range_m=min_range_m,
        bottom_guard_samples=guard,
    )
    res = detrend_amplitude(frame.amp_db, mask, n_iter=n_iter)
    return dataclasses.replace(frame, amp_db=res.normalized_db)


def frame_normalizer(method: Optional[str], **kwargs):
    """Return a ``frame -> frame`` callable, or ``None`` when no normalization.

    Convenience for the CLIs/builders: ``frame_normalizer("empirical")`` gives a
    function to apply to each decoded frame; ``None``/``"none"`` gives ``None`` so
    callers can skip the call entirely.
    """
    if not method or method == "none":
        return None
    return lambda frame: normalize_frame(frame, method=method, **kwargs)
