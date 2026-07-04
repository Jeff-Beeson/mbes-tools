"""Per-ping upper sample bound from the active water-column filters.

The mosaic/grid/viewer discard most of each ping late: ``--clean-water`` /
``--max-depth-m`` / ``--depth-band`` / ``--altitude-band`` keep only part of the
range. Because the sample index maps to range monotonically
(``R = c·k/(2·fs)``, see :func:`mbes_tools.wc_diagnostics.range_axis`), each filter
implies an **upper sample index** ``k_max`` beyond which no sample survives — and
that bound is computable from the beam headers (pointing angle, detected-bottom
sample) plus the per-ping ``c``/``fs``, *before* the amplitude samples are turned
into a grid. Capping the decoded/gridded width at ``k_max`` (a tail trim) shrinks
the dominant ``padded_grid`` work while leaving the geometry of every retained
sample unchanged, so the exact downstream filters still produce identical output.

The bound is a **conservative superset**: it is the tightest upper ``k`` that keeps
every sample any active filter could keep (filters compose with AND → the overall
max kept ``k`` is the ``min`` of each filter's swath-wide max kept ``k``), padded a
couple of samples for rounding safety. Front trimming (a *lower* bound, the main
win for the depth/altitude bands) is intentionally not done here — it would need a
sample offset threaded through the range geometry.

Leaf module: numpy + stdlib only, no imports from the readers or ``water_column``
(so those can import it without a cycle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Beams point within ~±75° of vertical; clamp cosθ away from 0 so a grazing beam
# can't blow up the depth→sample conversion.
_COS_MIN = math.cos(math.radians(85.0))


@dataclass(frozen=True)
class SampleBoundSpec:
    """The active filter values that can bound a ping's upper sample index.

    Mirrors the CLI filters directly, so callers build it straight from the
    parameters they already thread. ``pad_samples`` widens the computed bound so
    the trim never drops a sample the exact downstream filter would keep.
    """

    max_depth_m: Optional[float] = None
    depth_band: Optional[Tuple[float, float]] = None
    altitude_band: Optional[Tuple[float, float]] = None
    clean_water: bool = False
    msr_guard_m: float = 0.0
    msr_percentile: float = 0.0
    pad_samples: int = 2

    def bounds_tail(self) -> bool:
        """True if any active filter can cap the upper sample index."""
        return (
            self.max_depth_m is not None
            or self.depth_band is not None
            or self.altitude_band is not None
            or self.clean_water
        )


def sample_upper_bound(
    spec: Optional[SampleBoundSpec],
    angles_deg,
    detected_samples,
    sound_speed_m_s: float,
    sample_freq_hz: float,
) -> Optional[int]:
    """Conservative upper sample index ``k_max`` for this ping, or ``None``.

    ``None`` means "don't trim" (no active filter bounds the tail). Otherwise every
    sample any active filter keeps has index ``< k_max`` (padded), so capping the
    grid at ``k_max`` is bit-identical once the exact filters run downstream.
    """
    if spec is None or not spec.bounds_tail():
        return None
    if sound_speed_m_s <= 0 or sample_freq_hz <= 0:
        return None

    angles = np.asarray(angles_deg, dtype=float)
    det = np.asarray(detected_samples)
    cos = np.clip(np.cos(np.radians(angles)), _COS_MIN, 1.0)
    samp_per_m = 2.0 * sample_freq_hz / sound_speed_m_s   # k per metre of one-way range
    has_bottom = det > 0

    bounds = []

    # Absolute-depth ceiling: keep Z <= D  ->  k <= 2·fs·D/(c·cosθ). The swath-wide
    # max is the outermost beam (smallest cosθ). max_depth_m and depth_band's upper
    # edge both cap depth; take the tighter.
    depth_ceiling = None
    if spec.max_depth_m is not None:
        depth_ceiling = spec.max_depth_m
    if spec.depth_band is not None:
        hi = spec.depth_band[1]
        depth_ceiling = hi if depth_ceiling is None else min(depth_ceiling, hi)
    if depth_ceiling is not None:
        bounds.append(float(np.max(depth_ceiling * samp_per_m / cos)))

    # Height-above-seafloor band: the deepest kept sample per beam is at hab = lo
    # -> k = det − lo·(samp_per_m/cosθ). No-bottom beams are excluded by the band.
    if spec.altitude_band is not None:
        lo = spec.altitude_band[0]
        if has_bottom.any():
            kb = det[has_bottom] - lo * (samp_per_m / cos[has_bottom])
            bounds.append(float(np.max(kb)))
        else:
            bounds.append(0.0)  # band needs a bottom; nothing survives

    # Clean water (minimum-slant-range): keep k < R_min (min bottom sample, or a low
    # percentile), minus a guard. A ping with no bottom keeps everything -> no bound.
    if spec.clean_water and has_bottom.any():
        v = det[has_bottom]
        rmin = float(v.min()) if spec.msr_percentile <= 0 else float(np.percentile(v, spec.msr_percentile))
        guard = int(round(samp_per_m * spec.msr_guard_m)) if spec.msr_guard_m else 0
        bounds.append(rmin - guard)

    if not bounds:
        return None
    k_max = int(math.ceil(min(bounds))) + int(spec.pad_samples)
    return max(k_max, 0)
