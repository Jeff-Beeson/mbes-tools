"""Vessel-frame water-column products: gridded echograms + midwater anomalies.

This is the first **product** layer on top of the validated water-column readers
(:mod:`mbes_tools.kmwcd` ``#MWC`` and :mod:`mbes_tools.wcd` ``k``). It turns a
parsed ping into things you can look at and reason about, without yet needing
position/attitude (that is the geo-referenced Slice 2). Everything here is in the
**vessel frame** — across-track / depth relative to the transducer.

Three pieces, all numpy + stdlib (matplotlib is lazy, only for the plot panels):

1. **`.wcd` ping reassembly** (:func:`reassemble_wcd_pings`). Most real ``.wcd``
   pings are split across several ``k`` datagrams (``num_datagram`` > 1, beams
   partitioned by ``datagram_num``). A full swath needs all fragments of one
   ``counter`` concatenated in order first. ``#MWC`` is not fragmented this way,
   so it has no reassembly step (dual-swath fans are separate datagrams kept
   separate here).

2. **Cartesian gridding** (:func:`grid_frame` → :class:`WaterColumnGrid`). Each
   amplitude sample has an ``(across_track_m, depth_m)`` position from the same
   ``r = c·k/(2·fs)`` slant-range geometry the diagnostics use
   (:func:`mbes_tools.wc_diagnostics.wedge_coords`, angle **positive = port**).
   Those scattered samples are binned onto a regular grid — ``reduce="mean"``
   averages in the **linear-intensity domain** (``10**(dB/10)``) before
   converting back to dB, the physically appropriate aggregation for incoherent
   echo integration; ``reduce="max"`` is a peak-hold useful for thin targets.

3. **TVG-residual midwater / plume anomaly pass** (:func:`detect_anomalies` →
   :class:`WaterColumnAnomalies`). Subtract a per-range background (the across-
   beam median amplitude at each range sample, computed over **water only** —
   excluding each beam's seafloor and a guard band below it). What is left is the
   TVG/absorption-flattened residual; coherent off-bottom returns (e.g. Samoa
   hydrothermal plumes) show up as positive residual outliers, flagged by a
   robust (median + N·MAD) threshold above a minimum range.

Validated against the committed water-column fixtures and the full Atlantis EM122
``.wcd`` (762 pings, mostly 2 fragments each); see ``docs/BUILD_STATUS.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools.kmwcd import iter_mwc_datagrams
from mbes_tools.wc_diagnostics import (
    WCFrame,
    frame_from_mwc,
    frame_from_wcd,
    range_axis,
    wedge_coords,
)
from mbes_tools.wcd import WaterColumnDatagram, iter_water_column_datagrams


# ---------------------------------------------------------------------------
# 1. .wcd ping reassembly (by ``counter``).
# ---------------------------------------------------------------------------


def merge_wcd_fragments(fragments: Sequence[WaterColumnDatagram]) -> WaterColumnDatagram:
    """Concatenate the ``k`` datagram *fragments* of one ping into a full ping.

    Fragments are ordered by ``datagram_num`` and their beam lists concatenated.
    Per-ping metadata (sound speed, sample frequency, tx sectors, …) is taken
    from the first fragment; ``num_beams_this_datagram`` on the returned
    datagram is set to the merged beam count and ``datagram_num``/``num_datagram``
    to ``1`` to reflect that it is now a single reassembled ping.
    """
    if not fragments:
        raise ValueError("merge_wcd_fragments needs at least one fragment")
    ordered = sorted(fragments, key=lambda d: d.datagram_num)
    beams: List = []
    for frag in ordered:
        beams.extend(frag.beams)
    first = ordered[0]
    return dataclasses.replace(
        first,
        num_datagram=1,
        datagram_num=1,
        num_beams_this_datagram=len(beams),
        beams=beams,
    )


def reassemble_wcd_pings(
    datagrams: Iterable[WaterColumnDatagram],
    *,
    allow_incomplete: bool = False,
) -> Iterator[WaterColumnDatagram]:
    """Reassemble fragmented ``k`` Water Column datagrams into full pings.

    Groups by ``counter`` and emits a single merged :class:`WaterColumnDatagram`
    per ping once all ``num_datagram`` fragments have arrived (fragments for a
    ping are contiguous in real files, so memory stays bounded). Pings that never
    complete (e.g. a truncated tail) are dropped unless ``allow_incomplete`` is
    set, in which case the partial ping is emitted at end-of-stream.

    A non-fragmented file (``num_datagram == 1`` throughout, like the committed
    3-ping fixture) passes through one ping per datagram.
    """
    pending: "OrderedDict[int, List[WaterColumnDatagram]]" = OrderedDict()
    for dgm in datagrams:
        bucket = pending.setdefault(dgm.counter, [])
        bucket.append(dgm)
        if len(bucket) >= dgm.num_datagram:
            yield merge_wcd_fragments(bucket)
            del pending[dgm.counter]
    if allow_incomplete:
        for bucket in pending.values():
            yield merge_wcd_fragments(bucket)


def reassembled_wcd_frames(
    path: Path, *, allow_incomplete: bool = False
) -> Iterator[WCFrame]:
    """Yield a :class:`WCFrame` per reassembled full ping of a ``.wcd`` file."""
    pings = reassemble_wcd_pings(
        iter_water_column_datagrams(Path(path)), allow_incomplete=allow_incomplete
    )
    for ping in pings:
        yield frame_from_wcd(ping)


def mwc_frames(path: Path) -> Iterator[WCFrame]:
    """Yield a :class:`WCFrame` per ``#MWC`` datagram of a ``.kmwcd``/``.kmall``."""
    for dgm in iter_mwc_datagrams(Path(path)):
        yield frame_from_mwc(dgm)


# ---------------------------------------------------------------------------
# 2. Cartesian (across_track_m, depth_m) gridding.
# ---------------------------------------------------------------------------


@dataclass
class WaterColumnGrid:
    """A vessel-frame ``(across_track_m, depth_m)`` amplitude grid.

    ``amplitude_db`` is ``[n_depth, n_across]`` (row 0 = shallowest), NaN where no
    sample fell in the cell. ``counts`` is the per-cell sample tally. Bin edges
    are monotonic; ``across_centers`` / ``depth_centers`` give cell midpoints.
    """

    amplitude_db: np.ndarray
    counts: np.ndarray
    across_edges: np.ndarray
    depth_edges: np.ndarray
    reduce: str
    sound_speed_m_s: float
    sample_freq_hz: float
    label: str

    @property
    def across_centers(self) -> np.ndarray:
        return 0.5 * (self.across_edges[:-1] + self.across_edges[1:])

    @property
    def depth_centers(self) -> np.ndarray:
        return 0.5 * (self.depth_edges[:-1] + self.depth_edges[1:])

    @property
    def cell_size_m(self) -> Tuple[float, float]:
        """``(across, depth)`` cell size in metres (from the first interval)."""
        dx = float(self.across_edges[1] - self.across_edges[0])
        dz = float(self.depth_edges[1] - self.depth_edges[0])
        return dx, dz


def _edges(lo: float, hi: float, res: float, max_cells: int) -> np.ndarray:
    """Regular edges spanning ``[lo, hi]`` at step ``res`` snapped to the grid.

    ``res`` is enlarged if necessary so the axis stays within ``max_cells`` cells.
    """
    lo_s = np.floor(lo / res) * res
    hi_s = np.ceil(hi / res) * res
    span = max(hi_s - lo_s, res)
    n = int(np.ceil(span / res))
    if n > max_cells:
        res = span / max_cells
        n = max_cells
    return lo_s + res * np.arange(n + 1)


def grid_frame(
    frame: WCFrame,
    *,
    across_res_m: Optional[float] = None,
    depth_res_m: Optional[float] = None,
    reduce: str = "mean",
    max_depth_m: Optional[float] = None,
    max_across_m: Optional[float] = None,
    max_cells: int = 2000,
) -> WaterColumnGrid:
    """Bin a ping's amplitude samples onto an ``(across_track_m, depth_m)`` grid.

    Sample positions come from :func:`wedge_coords`
    (``X = r·sinθ`` across-track +port, ``Z = r·cosθ`` depth,
    ``r = c·k/(2·fs)``). Default cell size is the one-way range resolution
    ``c/(2·fs)`` on both axes (so the grid roughly matches native sampling),
    enlarged per axis only if it would exceed ``max_cells`` cells.

    ``reduce``:

    * ``"mean"`` — average in the **linear-intensity domain** (``10**(dB/10)``)
      then convert back to dB (incoherent echo-integration mean).
    * ``"max"`` — peak amplitude (dB) per cell (peak-hold; good for thin targets).

    ``max_depth_m`` / ``max_across_m`` clip the gridded extent (samples beyond are
    dropped) — handy to cut the noisy outer/long-range fringe.
    """
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")

    c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
    X, Z = wedge_coords(frame.angles_deg, frame.width, c, fs)
    amp = frame.amp_db

    finite = np.isfinite(amp)
    if max_depth_m is not None:
        finite &= Z <= max_depth_m
    if max_across_m is not None:
        finite &= np.abs(X) <= max_across_m

    res = c / (2.0 * fs)
    ax_res = across_res_m if across_res_m is not None else res
    dz_res = depth_res_m if depth_res_m is not None else res

    if not finite.any():
        # Degenerate (no samples): emit a 1x1 empty grid rather than crash.
        empty = np.full((1, 1), np.nan)
        return WaterColumnGrid(
            amplitude_db=empty,
            counts=np.zeros((1, 1), int),
            across_edges=np.array([0.0, ax_res]),
            depth_edges=np.array([0.0, dz_res]),
            reduce=reduce,
            sound_speed_m_s=c,
            sample_freq_hz=fs,
            label=frame.label,
        )

    xs, zs, as_ = X[finite], Z[finite], amp[finite]
    across_edges = _edges(float(xs.min()), float(xs.max()), ax_res, max_cells)
    depth_edges = _edges(min(0.0, float(zs.min())), float(zs.max()), dz_res, max_cells)
    n_across = len(across_edges) - 1
    n_depth = len(depth_edges) - 1

    ix = np.clip(np.digitize(xs, across_edges) - 1, 0, n_across - 1)
    iz = np.clip(np.digitize(zs, depth_edges) - 1, 0, n_depth - 1)
    flat = iz * n_across + ix

    counts = np.bincount(flat, minlength=n_depth * n_across).reshape(n_depth, n_across)

    if reduce == "mean":
        lin = np.power(10.0, as_ / 10.0)
        sum_lin = np.bincount(flat, weights=lin, minlength=n_depth * n_across)
        sum_lin = sum_lin.reshape(n_depth, n_across)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_lin = np.where(counts > 0, sum_lin / counts, np.nan)
            amplitude_db = 10.0 * np.log10(mean_lin)
    else:  # max
        grid = np.full(n_depth * n_across, -np.inf)
        np.maximum.at(grid, flat, as_)
        grid = grid.reshape(n_depth, n_across)
        amplitude_db = np.where(np.isfinite(grid), grid, np.nan)

    return WaterColumnGrid(
        amplitude_db=amplitude_db,
        counts=counts,
        across_edges=across_edges,
        depth_edges=depth_edges,
        reduce=reduce,
        sound_speed_m_s=c,
        sample_freq_hz=fs,
        label=frame.label,
    )


# ---------------------------------------------------------------------------
# 3. TVG-residual midwater / plume anomaly detection.
# ---------------------------------------------------------------------------


@dataclass
class WaterColumnAnomalies:
    """Result of the TVG-residual midwater anomaly pass for one ping.

    All ``[beam, sample]`` arrays share the frame's grid. ``residual_db`` is
    amplitude minus the per-range background; ``mask`` flags water samples whose
    residual exceeds ``threshold_db``; ``water_mask`` marks the samples that
    counted as open water (above the seafloor + guard, beyond ``min_range_m``).
    ``across_m`` / ``depth_m`` are the vessel-frame positions of every sample
    (for plotting / reporting the flagged points).
    """

    residual_db: np.ndarray
    background_db: np.ndarray
    mask: np.ndarray
    water_mask: np.ndarray
    across_m: np.ndarray
    depth_m: np.ndarray
    range_m: np.ndarray
    threshold_db: float
    label: str

    @property
    def n_flagged(self) -> int:
        return int(self.mask.sum())


def water_column_water_mask(
    frame: WCFrame,
    *,
    surface_guard_samples: int = 5,
    min_range_m: float = 0.0,
    bottom_guard_samples: int = 5,
) -> np.ndarray:
    """Boolean ``[beam, sample]`` mask of open-water samples.

    A sample counts as water when it is finite, past the near-transducer block
    (sample index ``>= surface_guard_samples`` **and** range ``>= min_range_m``),
    and — for beams that detected a bottom (``detected_range_samples > 0``) —
    strictly above ``detected_range - bottom_guard_samples`` (the guard keeps the
    seafloor return and its near sidelobe out of the background and the anomaly
    search). Beams with no detected bottom (swath edges) contribute their whole
    finite column.

    ``surface_guard_samples`` is sample-based so it scales across systems (5
    samples ≈ 0.12 m on a 30 kHz EM2040, ≈ 30 m on a decimated 127 Hz EM124);
    ``min_range_m`` is an optional absolute floor on top of it.
    """
    amp = frame.amp_db
    nb, width = amp.shape
    r = range_axis(width, frame.sound_speed_m_s, frame.sample_freq_hz)
    sample_idx = np.arange(width)

    mask = (
        np.isfinite(amp)
        & (sample_idx[None, :] >= surface_guard_samples)
        & (r[None, :] >= min_range_m)
    )
    det = frame.detected_samples
    bottom_limit = det - bottom_guard_samples
    has_bottom = det > 0
    # For beams with a bottom, keep only samples above the guarded bottom.
    above_bottom = sample_idx[None, :] < bottom_limit[:, None]
    mask &= np.where(has_bottom[:, None], above_bottom, True)
    return mask


def minimum_slant_range_sample(detected_samples: np.ndarray) -> Optional[int]:
    """Sample index of the ping's **shortest** bottom detection (``R_min``).

    Across a ping the seafloor is detected at different slant ranges; the shortest
    (the nadir range, for a flat bottom) is where the bottom's sidelobe energy
    arrives in *every* beam. Returns that sample index, or ``None`` when no beam
    detected a bottom (no reference to define the clean zone).
    """
    valid = detected_samples[detected_samples > 0]
    return int(valid.min()) if valid.size else None


def apply_min_slant_range(frame: WCFrame, *, guard_m: float = 0.0) -> WCFrame:
    """Return a copy of ``frame`` keeping only the bottom-sidelobe-free water column.

    Samples at or beyond the ping's minimum bottom-detect slant range ``R_min``
    (:func:`minimum_slant_range_sample`) are set to ``NaN`` — this is the standard
    **minimum-slant-range** ("clean water column") filter: beyond ``R_min`` every
    beam may carry the nadir seafloor's sidelobe, so only the near-range arc below
    ``R_min`` is trustworthy. Because slant range is monotonic in sample index and
    the range axis is common to all beams, the cut is a single column index applied
    to every beam. ``guard_m`` pulls the cutoff inward by that many metres (via the
    per-ping range resolution) to stay clear of the sidelobe onset. A ping with no
    bottom detection anywhere is returned unchanged (no ``R_min`` reference).
    """
    cutoff = minimum_slant_range_sample(frame.detected_samples)
    if cutoff is None:
        return frame
    if guard_m:
        c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
        cutoff -= max(0, int(round(2.0 * fs * guard_m / c)))
    cutoff = max(cutoff, 0)
    amp = frame.amp_db.copy()
    amp[:, cutoff:] = np.nan
    return dataclasses.replace(frame, amp_db=amp)


def detect_anomalies(
    frame: WCFrame,
    *,
    threshold_db: Optional[float] = None,
    n_mad: float = 6.0,
    surface_guard_samples: int = 5,
    min_range_m: float = 0.0,
    bottom_guard_samples: int = 5,
    bottom_guard_m: Optional[float] = None,
    background_stat: str = "median",
) -> WaterColumnAnomalies:
    """Flag midwater / plume anomalies by per-range TVG-residual.

    Steps:

    1. Build the open-water mask (:func:`water_column_water_mask`).
    2. ``background_db[k]`` = across-beam ``background_stat`` (``"median"`` or
       ``"mean"``) of amplitude at range sample ``k``, over water samples only —
       this is the range-dependent TVG/absorption + ambient-scattering trend.
    3. ``residual_db = amplitude − background``.
    4. Threshold: if ``threshold_db`` is given, use it; otherwise a robust
       ``median + n_mad · (1.4826·MAD)`` of the water residuals.
    5. ``mask`` = water samples whose residual exceeds the threshold.

    The near-field is excluded by ``surface_guard_samples`` (sample-based, scales
    across systems) plus an optional absolute ``min_range_m`` floor. The seafloor
    is excluded by ``bottom_guard_samples`` (or ``bottom_guard_m``, which
    overrides it via the per-ping range resolution).
    """
    if background_stat not in ("median", "mean"):
        raise ValueError(f"background_stat must be 'median' or 'mean', got {background_stat!r}")

    c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
    amp = frame.amp_db
    width = amp.shape[1]
    if bottom_guard_m is not None:
        bottom_guard_samples = max(0, int(round(2.0 * fs * bottom_guard_m / c)))

    water = water_column_water_mask(
        frame,
        surface_guard_samples=surface_guard_samples,
        min_range_m=min_range_m,
        bottom_guard_samples=bottom_guard_samples,
    )

    masked = np.where(water, amp, np.nan)
    # Columns past every beam's bottom are all-NaN; nanmedian/nanmean warn on
    # those slices and return NaN (the intended "no background here") — silence it.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if background_stat == "median":
            background = np.nanmedian(masked, axis=0)
        else:
            background = np.nanmean(masked, axis=0)
    # Columns with no water samples have NaN background -> their residual is NaN
    # and they cannot be flagged (NaN comparisons are False).
    residual = amp - background[None, :]

    water_res = residual[water]
    water_res = water_res[np.isfinite(water_res)]
    if threshold_db is None:
        if water_res.size:
            med = float(np.median(water_res))
            mad = float(np.median(np.abs(water_res - med)))
            threshold = med + n_mad * 1.4826 * mad
        else:
            threshold = np.inf
    else:
        threshold = float(threshold_db)

    with np.errstate(invalid="ignore"):
        mask = water & (residual > threshold)

    X, Z = wedge_coords(frame.angles_deg, width, c, fs)
    r = range_axis(width, c, fs)
    range_m = np.broadcast_to(r[None, :], amp.shape)

    return WaterColumnAnomalies(
        residual_db=residual,
        background_db=background,
        mask=mask,
        water_mask=water,
        across_m=X,
        depth_m=Z,
        range_m=range_m,
        threshold_db=float(threshold),
        label=frame.label,
    )


def summarize_anomalies(anom: WaterColumnAnomalies) -> Dict[str, object]:
    """One-ping summary dict of an anomaly result (JSON/print friendly).

    Includes the flagged count, the water-cell fraction flagged, the threshold,
    and the strongest anomaly's residual + vessel-frame location.
    """
    n_water = int(anom.water_mask.sum())
    n_flag = anom.n_flagged
    out: Dict[str, object] = {
        "label": anom.label,
        "threshold_db": round(anom.threshold_db, 3),
        "n_water_cells": n_water,
        "n_flagged": n_flag,
        "flagged_fraction": round(n_flag / n_water, 6) if n_water else 0.0,
    }
    if n_flag:
        res = np.where(anom.mask, anom.residual_db, -np.inf)
        bi, si = np.unravel_index(int(np.argmax(res)), res.shape)
        depths = anom.depth_m[anom.mask]
        out.update(
            max_residual_db=round(float(anom.residual_db[bi, si]), 3),
            max_at_beam=int(bi),
            max_at_across_m=round(float(anom.across_m[bi, si]), 2),
            max_at_depth_m=round(float(anom.depth_m[bi, si]), 2),
            max_at_range_m=round(float(anom.range_m[bi, si]), 2),
            flagged_depth_min_m=round(float(depths.min()), 2),
            flagged_depth_max_m=round(float(depths.max()), 2),
        )
    return out


# ---------------------------------------------------------------------------
# Plot panels (lazy matplotlib). Each returns the saved path.
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def panel_grid(grid: WaterColumnGrid, out: Path, stem: str) -> Path:
    """Vessel-frame gridded echogram (across-track vs depth)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 8))
    masked = np.ma.masked_invalid(grid.amplitude_db)
    mesh = ax.pcolormesh(
        grid.across_edges, grid.depth_edges, masked, cmap="viridis", shading="flat"
    )
    ax.invert_yaxis()
    ax.set_aspect("equal", "datalim")
    dx, dz = grid.cell_size_m
    ax.set(
        xlabel="across-track (m, +port)",
        ylabel="depth (m)",
        title=f"{grid.label} — gridded echogram ({grid.reduce}, {dx:.2g}×{dz:.2g} m cells)",
    )
    plt.colorbar(mesh, ax=ax, label="amplitude (dB)", shrink=0.8)
    fig.tight_layout()
    p = out / f"wc_grid_{stem}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def panel_anomalies(
    frame: WCFrame, anom: WaterColumnAnomalies, out: Path, stem: str
) -> Path:
    """1x3: amplitude echogram + flagged anomalies, residual echogram, background
    profile with threshold."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    amp = frame.amp_db

    im0 = ax[0].imshow(
        amp.T, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest"
    )
    ax[0].plot(np.arange(amp.shape[0]), frame.detected_samples, color="red", lw=0.8,
               label="detected bottom")
    bset, sset = np.nonzero(anom.mask)
    if bset.size:
        ax[0].scatter(bset, sset, s=10, facecolors="none", edgecolors="magenta",
                      lw=0.6, label=f"anomaly ({bset.size})")
    ax[0].set(xlabel="beam #", ylabel="sample # (range →)",
              title=f"{frame.label} — amplitude + flagged anomalies")
    ax[0].legend(loc="upper right", fontsize=7)
    plt.colorbar(im0, ax=ax[0], label="amplitude (dB)", shrink=0.8)

    res = np.where(anom.water_mask, anom.residual_db, np.nan)
    vmax = np.nanmax(np.abs(res)) if np.isfinite(res).any() else 1.0
    im1 = ax[1].imshow(res.T, aspect="auto", origin="upper", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax[1].set(xlabel="beam #", ylabel="sample # (range →)",
              title="TVG residual (water only, dB)")
    plt.colorbar(im1, ax=ax[1], label="residual (dB)", shrink=0.8)

    r = range_axis(frame.width, frame.sound_speed_m_s, frame.sample_freq_hz)
    bg = anom.background_db
    fin = np.isfinite(bg)
    ax[2].plot(bg[fin], r[fin], lw=1.0, label="per-range background")
    ax[2].axvline(0.0, color="gray", lw=0.6)
    ax[2].invert_yaxis()
    ax[2].set(xlabel="background amplitude (dB)", ylabel="one-way range (m)",
              title=f"background vs range  (thr {anom.threshold_db:.1f} dB, "
                    f"{anom.n_flagged} flagged)")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    p = out / f"wc_anomaly_{stem}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate(
    output_dir,
    mwc_files: Sequence = (),
    wcd_files: Sequence = (),
    *,
    reduce: str = "mean",
    threshold_db: Optional[float] = None,
    min_range_m: float = 0.0,
    max_depth_m: Optional[float] = None,
    normalize: Optional[str] = None,
) -> List[Path]:
    """Grid + anomaly panels for the first ping of each supplied file.

    ``.wcd`` files are reassembled by ``counter`` first, so the first ping is a
    full swath. ``normalize="empirical"`` de-trends each ping's amplitude (per-range
    + per-beam-angle acquisition gain) before gridding. Returns the saved PNG
    paths; a failing panel is reported but does not abort the rest. Prints the
    per-ping anomaly summary.
    """
    # Lazy import avoids a circular import (water_column_normalize imports this).
    from mbes_tools.water_column_normalize import frame_normalizer

    normalizer = frame_normalizer(normalize)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []

    def run(fn, *a, **k):
        try:
            p = fn(*a, **k)
            if p:
                made.append(p)
                print("OK  ", p.name)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not abort the rest
            print("FAIL", getattr(fn, "__name__", fn), "->", type(exc).__name__, str(exc)[:120])

    def handle(frame: WCFrame, stem: str):
        if normalizer is not None:
            frame = normalizer(frame)
        grid = grid_frame(frame, reduce=reduce, max_depth_m=max_depth_m)
        run(panel_grid, grid, out, stem)
        anom = detect_anomalies(frame, threshold_db=threshold_db, min_range_m=min_range_m)
        run(panel_anomalies, frame, anom, out, stem)
        print("    anomalies:", summarize_anomalies(anom))

    for f in mwc_files:
        stem = Path(f).stem
        try:
            frame = next(mwc_frames(f))
        except StopIteration:
            print("FAIL read", f, "-> no #MWC datagrams")
            continue
        except Exception as exc:  # noqa: BLE001
            print("FAIL read", f, "->", type(exc).__name__, str(exc)[:120])
            continue
        handle(frame, stem)

    for f in wcd_files:
        stem = Path(f).stem
        try:
            frame = next(reassembled_wcd_frames(f))
        except StopIteration:
            print("FAIL read", f, "-> no k datagrams")
            continue
        except Exception as exc:  # noqa: BLE001
            print("FAIL read", f, "->", type(exc).__name__, str(exc)[:120])
            continue
        handle(frame, stem)

    return made


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Vessel-frame water-column products: gridded (across/depth) echogram "
                    "+ TVG-residual midwater anomaly detection from real .kmwcd/.wcd files."
    )
    ap.add_argument("-o", "--output", default="mbes_wc_grid", help="Output directory for PNGs.")
    ap.add_argument("--mwc", nargs="+", default=[],
                    help="One or more #MWC-bearing files (.kmwcd or .kmall).")
    ap.add_argument("--wcd", nargs="+", default=[],
                    help="One or more .wcd files (reassembled by counter).")
    ap.add_argument("--reduce", choices=["mean", "max"], default="mean",
                    help="Cell aggregation: intensity-mean (default) or peak-hold max.")
    ap.add_argument("--threshold-db", type=float, default=None,
                    help="Fixed anomaly residual threshold (dB); default = robust median+6·MAD.")
    ap.add_argument("--min-range-m", type=float, default=0.0,
                    help="Optional absolute near-range floor for the anomaly pass "
                         "(on top of the sample-based near-field guard).")
    ap.add_argument("--max-depth-m", type=float, default=None,
                    help="Clip the gridded echogram below this depth.")
    ap.add_argument("--normalize", choices=["none", "empirical"], default="none",
                    help="Remove per-range + per-beam-angle acquisition gain (median polish over "
                         "the open water) before gridding. Relative dB, not calibrated Sv.")
    args = ap.parse_args(argv)
    if not args.mwc and not args.wcd:
        ap.error("supply at least one --mwc or --wcd file")
    made = generate(
        args.output, mwc_files=args.mwc, wcd_files=args.wcd,
        reduce=args.reduce, threshold_db=args.threshold_db,
        min_range_m=args.min_range_m, max_depth_m=args.max_depth_m,
        normalize=args.normalize,
    )
    print(f"\nWrote {len(made)} panel(s) to {args.output}")


if __name__ == "__main__":
    main()
