"""Per-beam reduction of seabed-image samples to one backscatter value (Source B).

The backscatter pipeline has two per-beam intensity sources:

- **Source A** — the sonar's per-beam reflectivity (``reflectivity2_dB`` in .kmall
  #MRZ, ``XBeam.reflectivity_db`` in .all XYZ88). One value per beam already.
- **Source B** — the per-beam **seabed-image** (sidescan) sample array
  (``SIsample_desidB`` in .kmall, ``YBeam.samples`` in .all), int16 in units of
  0.1 dB. This module reduces that array to one dB value per beam with a
  selectable statistic, over a selectable sample window.

Statistics (the ``beam_stat`` registry): ``mean``, ``median``, ``std``, ``var``,
``mode``, ``trimmed_mean``, ``min``, ``max``, ``range``, ``count`` (valid-sample
count), and ``p<NN>`` percentiles (e.g. ``p10``, ``p90``). ``std``/``var``/
``range`` double as within-beam **texture** features.

Window (``si_window``): ``None`` reduces the whole beam; an integer half-width
``w`` reduces only samples within ``[centre - w, centre + w]`` around the
bottom-detection (centre) sample.

All reducers are pure numpy and operate on a 1-D dB array, so they are reusable
by both the .kmall and .all front-ends and easy to unit test.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Seabed-image samples are stored as int16 deci-dB.
DESIDB_TO_DB = 0.1
# Samples at or below this are Kongsberg no-data sentinels (matches apply.py).
NODATA_THRESHOLD = -32767


def window_bounds(
    n_samples: int, centre: Optional[int], half_width: Optional[int]
) -> Tuple[int, int]:
    """Return ``(start, stop)`` sample indices for the reduction window.

    ``half_width is None`` selects the whole beam. Otherwise the window is
    ``[centre - half_width, centre + half_width]`` (inclusive), clamped to the
    beam. If ``centre`` is missing or out of range, the beam midpoint is used.
    """
    if half_width is None or n_samples <= 0:
        return 0, n_samples
    if centre is None or not (0 <= centre < n_samples):
        centre = n_samples // 2
    start = max(0, centre - half_width)
    stop = min(n_samples, centre + half_width + 1)
    return start, stop


def clean_db(samples_desidb: Sequence[int]) -> np.ndarray:
    """Drop no-data sentinels and scale int16 deci-dB samples to a dB float array."""
    arr = np.asarray(samples_desidb, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = arr[arr > NODATA_THRESHOLD]
    return arr * DESIDB_TO_DB


# --- Reducers: callable(values_db: np.ndarray) -> float --------------------


def _mode_db(values_db: np.ndarray) -> float:
    """Most common sample value, to 0.1 dB resolution (the native sample step)."""
    rounded = np.round(values_db, 1)
    vals, counts = np.unique(rounded, return_counts=True)
    return float(vals[int(np.argmax(counts))])


def _trimmed_mean(values_db: np.ndarray, proportion: float = 0.1) -> float:
    if values_db.size == 0:
        return float("nan")
    ordered = np.sort(values_db)
    k = int(np.floor(values_db.size * proportion))
    core = ordered[k : values_db.size - k] if values_db.size - 2 * k > 0 else ordered
    return float(np.mean(core))


_REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
    "std": lambda v: float(np.std(v, ddof=0)),
    "var": lambda v: float(np.var(v, ddof=0)),
    "min": lambda v: float(np.min(v)),
    "max": lambda v: float(np.max(v)),
    "range": lambda v: float(np.max(v) - np.min(v)),
    "count": lambda v: float(v.size),
    "mode": _mode_db,
    "trimmed_mean": _trimmed_mean,
}


def get_reducer(name: str) -> Callable[[np.ndarray], float]:
    """Return the reducer callable for a beam-stat name.

    Supports the fixed registry plus dynamic ``p<NN>`` percentiles
    (e.g. ``p90`` -> 90th percentile).
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name.startswith("p") and name[1:].isdigit():
        pct = float(name[1:])
        if 0 <= pct <= 100:
            return lambda v: float(np.percentile(v, pct))
    raise ValueError(
        f"Unknown beam-stat {name!r}. Known: {sorted(_REGISTRY)} or p<0-100>."
    )


def available_stats() -> List[str]:
    """Names of the fixed reducers (percentiles ``p<NN>`` are also accepted)."""
    return sorted(_REGISTRY)


def reduce_beam(
    samples_desidb: Sequence[int],
    centre_sample: Optional[int],
    half_width: Optional[int],
    stat: str,
) -> float:
    """Reduce one beam's seabed-image samples to a single dB value.

    Returns NaN when the windowed beam has no valid samples.
    """
    n = len(samples_desidb)
    start, stop = window_bounds(n, centre_sample, half_width)
    values_db = clean_db(samples_desidb[start:stop])
    if values_db.size == 0:
        return float("nan")
    return get_reducer(stat)(values_db)


def reduce_beam_multi(
    samples_desidb: Sequence[int],
    centre_sample: Optional[int],
    half_width: Optional[int],
    stats: Sequence[str],
) -> Dict[str, float]:
    """Reduce one beam with several statistics in a single read (window once)."""
    n = len(samples_desidb)
    start, stop = window_bounds(n, centre_sample, half_width)
    values_db = clean_db(samples_desidb[start:stop])
    if values_db.size == 0:
        return {s: float("nan") for s in stats}
    return {s: get_reducer(s)(values_db) for s in stats}
