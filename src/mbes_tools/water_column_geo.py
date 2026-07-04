"""Geo-referenced water-column products (D1 products, Slice 2).

Slice 1 (:mod:`mbes_tools.water_column`) gridded a ping in the **vessel frame**
(across-track / depth). This slice puts the returns at real **geographic**
coordinates by composing that wedge geometry with the platform's
position + heading (and, optionally, install lever arms) — the D3 navigation /
attitude / installation path — then accumulates many pings into one plan-view
**mosaic**. This is the Samoa-relevant deliverable (plume footprint mapping,
midwater context over ground).

Pieces (core numpy + stdlib; matplotlib lazy; pyproj optional):

1. **`NavTrack`** — time-indexed platform position + heading with linear position
   interpolation and *circular* heading interpolation. Built from a `.kmall`
   (``#SKM`` gives position + true heading; falls back to ``#SPO`` position +
   course-over-ground as a heading proxy) or a `.all` (``P`` position + heading).
   **Water-column files are frequently nav-poor** — a `.kmwcd` often lacks
   ``#SKM`` (no true heading) and a bare `.wcd` has no ``P`` at all — so
   :func:`resolve_nav_track` does not assume the WC file is self-contained: it
   prefers an explicit companion, then the same-stem ``.kmall``/``.all`` sibling
   (Kongsberg logs them side by side, and that is where position + true heading
   live), then the WC file's own nav. A ping whose time the chosen track does not
   span is **skipped** (not silently clamped to a far-away endpoint).

2. **`georeference_frame`** — turn a Slice-1 `WCFrame` + a ping time into
   `GeoSamples`: every amplitude sample placed at ``(easting_m, northing_m,
   depth_m)``. Across-track (positive = port) becomes a starboard/forward vessel
   offset (plus the transducer lever arm), rotated by heading into east/north,
   added to the platform position. Coordinates are either a **local ENU metric
   frame** anchored at a reference lon/lat (pure numpy, the default in the base
   env) or **true UTM** via pyproj when available; the auto-UTM EPSG is always
   resolved (via :mod:`mbes_tools.projection`) for provenance.

3. **`GeoMosaic`** — a streaming plan-view accumulator binning samples onto a
   fixed ``cell_m`` east/north grid, reducing by peak-hold (``max``) or
   intensity-mean over an optional ``depth_band`` (absolute depth, e.g. a
   midwater band for plume mapping) and/or ``altitude_band`` (height above the
   detected seafloor, e.g. a near-bottom layer that tracks the terrain).
   ``finalize()`` returns a dense grid + edges + CRS.

Verified against the committed EM124 ``.kmwcd`` + its ``.kmall`` companion and
the full TN447 pair (matched ``.kmwcd``/``.kmall``, where ``#SKM`` nav spans the
``#MWC`` ping times); see ``docs/BUILD_STATUS.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools.install_params import InstallationParameters
from mbes_tools.projection import resolve_target_crs
from mbes_tools.water_column_normalize import frame_normalizer
from mbes_tools.wc_diagnostics import WCFrame, frame_from_mwc, frame_from_wcd, wedge_coords

# WGS84 semi-major axis — used by the local ENU (equirectangular) fallback.
_WGS84_A = 6378137.0


# ---------------------------------------------------------------------------
# 1. NavTrack — time-indexed position + heading.
# ---------------------------------------------------------------------------


@dataclass
class NavTrack:
    """Platform navigation as sorted ``(time, lat, lon)`` + ``(time, heading)``.

    Times are in whatever clock the source uses — absolute Unix seconds for
    `.kmall`, seconds-since-midnight for `.all`/`.wcd`; the water-column ping
    times are read from the same clock, so interpolation is self-consistent per
    format (do not mix formats in one track). ``position_source`` /
    ``heading_source`` record provenance (e.g. ``"#SKM"`` vs ``"#SPO COG"``).
    """

    t_pos: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    t_hdg: np.ndarray
    hdg_deg: np.ndarray
    position_source: str
    heading_source: str
    # Optional attitude (roll/pitch/heave) — present from #SKM / .all ``A``, else None.
    t_att: Optional[np.ndarray] = None
    roll_deg: Optional[np.ndarray] = None
    pitch_deg: Optional[np.ndarray] = None
    heave_m: Optional[np.ndarray] = None
    attitude_source: Optional[str] = None

    @classmethod
    def from_lists(
        cls, t_pos, lat, lon, t_hdg, hdg_deg, position_source, heading_source,
        t_att=None, roll_deg=None, pitch_deg=None, heave_m=None, attitude_source=None,
    ):
        if not t_pos or not t_hdg:
            raise ValueError("NavTrack needs at least one position and one heading sample")
        tp = np.asarray(t_pos, float)
        op = np.argsort(tp, kind="stable")
        th = np.asarray(t_hdg, float)
        oh = np.argsort(th, kind="stable")
        ta = ro = pi = he = None
        if t_att is not None and len(t_att):
            ta = np.asarray(t_att, float)
            oa = np.argsort(ta, kind="stable")
            ta = ta[oa]
            ro = np.asarray(roll_deg, float)[oa]
            pi = np.asarray(pitch_deg, float)[oa]
            he = np.asarray(heave_m, float)[oa]
        return cls(
            t_pos=tp[op],
            lat=np.asarray(lat, float)[op],
            lon=np.asarray(lon, float)[op],
            t_hdg=th[oh],
            hdg_deg=np.asarray(hdg_deg, float)[oh],
            position_source=position_source,
            heading_source=heading_source,
            t_att=ta, roll_deg=ro, pitch_deg=pi, heave_m=he,
            attitude_source=attitude_source if ta is not None else None,
        )

    def position_at(self, t) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolated ``(lat, lon)`` at time(s) ``t`` (clamped past the ends)."""
        lat = np.interp(t, self.t_pos, self.lat)
        lon = np.interp(t, self.t_pos, self.lon)
        return lat, lon

    def heading_at(self, t) -> np.ndarray:
        """Interpolated heading (deg, 0–360) at ``t``, wrap-safe via sin/cos."""
        rad = np.deg2rad(self.hdg_deg)
        s = np.interp(t, self.t_hdg, np.sin(rad))
        c = np.interp(t, self.t_hdg, np.cos(rad))
        return np.rad2deg(np.arctan2(s, c)) % 360.0

    @property
    def has_attitude(self) -> bool:
        return self.t_att is not None and self.t_att.size > 0

    def attitude_at(self, t) -> Tuple[float, float, float]:
        """Interpolated ``(roll_deg, pitch_deg, heave_m)`` at scalar time ``t``.

        Returns ``(0, 0, 0)`` when the track carries no attitude (e.g. built from
        ``#SPO`` course-over-ground or a ``P``-only ``.all``), so callers can
        apply it unconditionally. Roll/pitch/heave are small, non-wrapping
        quantities, so plain linear interpolation is appropriate.
        """
        if not self.has_attitude:
            return 0.0, 0.0, 0.0
        return (
            float(np.interp(t, self.t_att, self.roll_deg)),
            float(np.interp(t, self.t_att, self.pitch_deg)),
            float(np.interp(t, self.t_att, self.heave_m)),
        )

    def covers(self, t) -> bool:
        """True when ``t`` lies within the position time span (no extrapolation)."""
        return bool(self.t_pos.min() <= t <= self.t_pos.max())

    @property
    def time_span(self) -> Tuple[float, float]:
        return float(self.t_pos.min()), float(self.t_pos.max())


def nav_track_from_kmall(paths: Sequence[Path]) -> NavTrack:
    """Build a :class:`NavTrack` from `.kmall`/`.kmwcd` navigation datagrams.

    Prefers ``#SKM`` (KMbinary: position **and** true heading). If no ``#SKM`` is
    present (common in a `.kmwcd` water-column file), falls back to ``#SPO``
    position with **course-over-ground as a heading proxy** — valid for a vessel
    making way, and the only heading available from ``#SPO`` alone.
    """
    from mbes_tools.kmall import iter_skm_datagrams, iter_spo_datagrams

    tp: List[float] = []
    la: List[float] = []
    lo: List[float] = []
    th: List[float] = []
    hd: List[float] = []
    ro: List[float] = []
    pi: List[float] = []
    he: List[float] = []
    for p in paths:
        for d in iter_skm_datagrams(Path(p), on_error="skip"):
            for s in d.samples:
                if -90.0 <= s.latitude_deg <= 90.0 and -180.0 <= s.longitude_deg <= 180.0:
                    t = s.time_sec + s.time_nanosec * 1e-9
                    tp.append(t); la.append(s.latitude_deg); lo.append(s.longitude_deg)
                    th.append(t); hd.append(s.heading_deg)
                    ro.append(s.roll_deg); pi.append(s.pitch_deg); he.append(s.heave_m)
    if tp:
        # #SKM KMbinary carries roll/pitch/heave too — attach the attitude track.
        return NavTrack.from_lists(
            tp, la, lo, th, hd, "#SKM", "#SKM heading",
            t_att=list(tp), roll_deg=ro, pitch_deg=pi, heave_m=he, attitude_source="#SKM",
        )

    for p in paths:
        for d in iter_spo_datagrams(Path(p), on_error="skip"):
            if not d.is_available:
                continue
            t = d.time_s + d.time_ns * 1e-9
            tp.append(t); la.append(d.latitude_deg); lo.append(d.longitude_deg)
            th.append(t); hd.append(d.course_over_ground_deg)
    return NavTrack.from_lists(tp, la, lo, th, hd, "#SPO", "#SPO course-over-ground")


def nav_track_from_all(paths: Sequence[Path]) -> NavTrack:
    """Build a :class:`NavTrack` from `.all`/`.wcd` ``P`` position datagrams.

    ``P`` carries lat/lon **and** true heading, so it supplies both tracks. Times
    use the absolute ``date + time`` header clock (:func:`_all_header_time`). If
    ``A`` attitude datagrams are present, roll/pitch/heave are attached too.
    """
    from mbes_tools.all import iter_position_datagrams

    tp: List[float] = []
    la: List[float] = []
    lo: List[float] = []
    hd: List[float] = []
    for p in paths:
        for d in iter_position_datagrams(Path(p)):
            if -90.0 <= d.latitude_deg <= 90.0 and -180.0 <= d.longitude_deg <= 180.0:
                t = _all_header_time(d.header)
                tp.append(t); la.append(d.latitude_deg); lo.append(d.longitude_deg)
                hd.append(d.heading_deg)

    ta: List[float] = []
    ro: List[float] = []
    pi: List[float] = []
    he: List[float] = []
    try:
        from mbes_tools.all import iter_attitude_datagrams
        for p in paths:
            for d in iter_attitude_datagrams(Path(p)):
                base = _all_header_time(d.header)
                for s in d.samples:
                    ta.append(base + s.time_ms / 1000.0)
                    ro.append(s.roll_deg); pi.append(s.pitch_deg); he.append(s.heave_m)
    except Exception:  # noqa: BLE001 - attitude is an optional refinement
        ta = []

    return NavTrack.from_lists(
        tp, la, lo, list(tp), hd, "P position", "P heading",
        t_att=ta or None, roll_deg=ro, pitch_deg=pi, heave_m=he,
        attitude_source="A" if ta else None,
    )


def build_nav_track(paths: Sequence[Path]) -> NavTrack:
    """Dispatch to the `.kmall` or `.all` nav builder by file extension."""
    paths = [Path(p) for p in paths]
    ext = paths[0].suffix.lower()
    if ext in (".kmall", ".kmwcd"):
        return nav_track_from_kmall(paths)
    if ext in (".all", ".wcd"):
        return nav_track_from_all(paths)
    raise ValueError(f"cannot build a nav track from {ext!r} files")


# ---------------------------------------------------------------------------
# Installation loading (optional lever arm).
# ---------------------------------------------------------------------------


def load_installation(paths: Sequence[Path]) -> Optional[InstallationParameters]:
    """Return the first installation parameters found in the file(s), or None.

    Reads the `.kmall` ``#IIP`` or `.all` ``I`` datagram (both expose a
    structured :class:`InstallationParameters`). Best-effort — returns ``None``
    if none is present or parsing fails (the lever arm is a small correction, so
    the geo product still works without it).
    """
    for p in paths:
        p = Path(p)
        try:
            if p.suffix.lower() in (".kmall", ".kmwcd"):
                from mbes_tools.kmall import iter_iip_datagrams
                for d in iter_iip_datagrams(p, on_error="skip"):
                    return d.parameters
            else:
                from mbes_tools.all import iter_installation_datagrams
                for d in iter_installation_datagrams(p):
                    return d.parameters
        except Exception:  # noqa: BLE001 - install is optional
            continue
    return None


def _resolve_group(install: InstallationParameters) -> Optional[str]:
    """Pick the RX transducer group name across `.kmall` / `.all` schemes."""
    for g in ("TRAI_RX1", "TRAI_TX1"):
        if g in install.sections:
            return g
    if any(k.startswith("S1") for k in install.params):
        return "S1"
    return None


# ---------------------------------------------------------------------------
# 2. Georeferencing a frame.
# ---------------------------------------------------------------------------


def _pyproj_available() -> bool:
    return importlib.util.find_spec("pyproj") is not None


def _vessel_en(lon, lat, anchor_lonlat, epsg: int, use_pyproj: bool) -> Tuple[float, float]:
    """Platform ``(easting, northing)`` metres in the chosen metric frame.

    ``use_pyproj`` -> true projected coordinates in ``epsg``; otherwise a local
    equirectangular (ENU) frame anchored at ``anchor_lonlat``.
    """
    if use_pyproj:
        from pyproj import Transformer
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        e, n = tr.transform(lon, lat)
        return float(e), float(n)
    lon0, lat0 = anchor_lonlat
    e = _WGS84_A * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    n = _WGS84_A * math.radians(lat - lat0)
    return float(e), float(n)


def _dcm(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Body→NED direction-cosine matrix ``Rz(H)·Ry(pitch)·Rx(roll)``.

    Body axes are ``(forward, starboard, down)``; the result maps a body vector to
    ``(north, east, down)``. With pitch=roll=0 this reduces to a pure heading yaw.
    """
    h, p, r = math.radians(heading_deg), math.radians(pitch_deg), math.radians(roll_deg)
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


@dataclass
class GeoSamples:
    """Flattened geo-referenced amplitude samples for one ping (finite only)."""

    easting_m: np.ndarray
    northing_m: np.ndarray
    depth_m: np.ndarray
    amplitude_db: np.ndarray
    crs_label: str
    projector: str
    vessel_lon: float
    vessel_lat: float
    heading_deg: float
    label: str
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    heave_m: float = 0.0
    epsg: Optional[int] = None      # real projected EPSG in "utm" mode; None for local ENU
    height_above_seafloor_m: Optional[np.ndarray] = None  # per sample; NaN where no bottom

    def __len__(self) -> int:
        return int(self.easting_m.size)


def georeference_frame(
    frame: WCFrame,
    ping_time: float,
    nav: NavTrack,
    *,
    anchor_lonlat: Optional[Tuple[float, float]] = None,
    install: Optional[InstallationParameters] = None,
    tx_group: Optional[str] = None,
    target_crs="auto",
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
) -> GeoSamples:
    """Place a ping's amplitude samples at ``(easting_m, northing_m, depth_m)``.

    Geometry: across-track ``X`` (positive = port) and depth ``Z`` come from the
    Slice-1 wedge (``r = c·k/(2·fs)``). Each sample is a body-frame vector
    ``(forward 0, starboard −X, down Z)``; the transducer **lever arm**
    ``(fx, fy, fz)`` is a second body vector. Both are rotated into
    ``(north, east, down)`` and added to the platform position.

    Attitude (``apply_attitude``, from ``nav.attitude_at`` — ``0`` when the track
    has none): the lever arm is always rotated by the **full** pose
    ``Rz(H)·Ry(pitch)·Rx(roll)``, and **heave** is added to depth. The beam
    samples use heading-only rotation when ``stabilized_beams=True`` (the default
    — Kongsberg ``#MWC``/``k`` ``beamPointAngReVertical`` is already
    roll/pitch-stabilized at receive, verified empirically: the shoalest beam
    stays at nadir independent of vessel roll, so re-rotating would
    double-correct). Set ``stabilized_beams=False`` for raw array-relative angles.

    ``projector``: ``"local"`` = ENU metres about ``anchor_lonlat`` (defaults to
    the ping's own position); ``"utm"`` = true UTM via pyproj; ``"auto"`` = UTM
    if pyproj is importable, else local. The auto-UTM EPSG is always resolved for
    the ``crs_label``.
    """
    lat0, lon0 = (float(v) for v in nav.position_at(ping_time))
    heading = float(nav.heading_at(ping_time))
    roll, pitch, heave = nav.attitude_at(ping_time) if apply_attitude else (0.0, 0.0, 0.0)
    if anchor_lonlat is None:
        anchor_lonlat = (lon0, lat0)

    crs = resolve_target_crs(target_crs, anchor_lonlat[0], anchor_lonlat[1])
    epsg = int(crs.split(":")[1]) if crs.upper().startswith("EPSG:") else 4326
    use_pyproj = projector == "utm" or (projector == "auto" and _pyproj_available())
    if projector == "utm" and not _pyproj_available():
        raise RuntimeError("projector='utm' requires pyproj, which is not installed")

    e0, n0 = _vessel_en(lon0, lat0, anchor_lonlat, epsg, use_pyproj)

    lever = install.transducer_offsets(tx_group or _resolve_group(install)) if install else None
    fx, fy, fz = lever if lever else (0.0, 0.0, 0.0)

    c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
    X, Z = wedge_coords(frame.angles_deg, frame.width, c, fs)
    finite = np.isfinite(frame.amp_db)
    if max_depth_m is not None:
        finite &= Z <= max_depth_m
    xs, zs, amp = X[finite], Z[finite], frame.amp_db[finite]

    # Per-sample height above the detected seafloor (for altitude bands). The
    # seafloor slant range is r_b = c·det/(2·fs); its depth Zb = cos(angle)·r_b,
    # so height = Zb − Z (positive = above bottom). Beams with no bottom detection
    # (det == 0, e.g. swath edges) get NaN and drop out of any altitude product.
    det = frame.detected_samples
    zb = np.cos(np.radians(frame.angles_deg)) * (c * det / (2.0 * fs))   # [beam]
    hab_grid = zb[:, None] - Z                                           # [beam, width]
    hab_grid[det == 0, :] = np.nan
    habs = hab_grid[finite]

    # Lever arm: rigid in the body frame -> rotated by the full vessel pose.
    dcm_full = _dcm(heading, pitch, roll)
    lev_n, lev_e, lev_d = dcm_full @ np.array([fx, fy, fz])

    # Beam samples: body vector (forward 0, starboard -X, down Z). Beams are
    # already vertical-stabilized, so rotate by heading only unless told otherwise.
    dcm_beam = dcm_full if not stabilized_beams else _dcm(heading, 0.0, 0.0)
    beam_body = np.vstack([np.zeros_like(xs), -xs, zs])   # (3, N)
    bn, be, bd = dcm_beam @ beam_body

    north = n0 + lev_n + bn
    east = e0 + lev_e + be
    depth = lev_d + bd + heave

    return GeoSamples(
        easting_m=east,
        northing_m=north,
        depth_m=depth,
        amplitude_db=amp,
        crs_label=crs if use_pyproj else f"local-ENU@{anchor_lonlat[1]:.5f},{anchor_lonlat[0]:.5f} ({crs} zone)",
        projector="utm" if use_pyproj else "local",
        vessel_lon=lon0,
        vessel_lat=lat0,
        heading_deg=heading,
        label=frame.label,
        roll_deg=roll,
        pitch_deg=pitch,
        heave_m=heave,
        epsg=epsg if use_pyproj else None,
        height_above_seafloor_m=habs,
    )


# ---------------------------------------------------------------------------
# 3. GeoMosaic — streaming plan-view accumulator.
# ---------------------------------------------------------------------------


@dataclass
class GeoMosaicResult:
    """Finalized plan-view mosaic: a dense amplitude grid in projected metres."""

    amplitude_db: np.ndarray      # [n_north, n_east], NaN where empty
    counts: np.ndarray
    east_edges: np.ndarray
    north_edges: np.ndarray
    cell_m: float
    reduce: str
    crs_label: str
    n_pings: int
    depth_band: Optional[Tuple[float, float]]
    n_uncovered: int = 0          # pings dropped because nav didn't span their time
    epsg: Optional[int] = None    # real projected EPSG (utm mode); None for local ENU
    altitude_band: Optional[Tuple[float, float]] = None  # metres above seafloor

    @property
    def east_centers(self) -> np.ndarray:
        return 0.5 * (self.east_edges[:-1] + self.east_edges[1:])

    @property
    def north_centers(self) -> np.ndarray:
        return 0.5 * (self.north_edges[:-1] + self.north_edges[1:])


def _ping_cell_partials(
    gs: GeoSamples,
    cell_m: float,
    reduce: str,
    depth_band: Optional[Tuple[float, float]],
    altitude_band: Optional[Tuple[float, float]],
):
    """Reduce one ping's samples to its unique cells (the per-ping map step).

    Returns ``(ie, jn, val, cnt)`` where ``ie/jn`` are int64 cell indices and,
    per unique cell: for ``max`` ``val`` is the peak dB and ``cnt`` is ``None``;
    for ``mean`` ``val`` is the summed linear intensity and ``cnt`` the sample
    count. Empty arrays when no sample passes the band filters. This is the shared
    kernel behind :meth:`GeoMosaic.add` and the multiprocessing worker, so serial
    and parallel builds produce identical per-ping partials.
    """
    e, n, d, a = gs.easting_m, gs.northing_m, gs.depth_m, gs.amplitude_db
    keep = None
    if depth_band is not None:
        lo, hi = depth_band
        keep = (d >= lo) & (d <= hi)
    if altitude_band is not None:
        hab = gs.height_above_seafloor_m
        if hab is None:
            raise ValueError(
                "altitude_band requires GeoSamples.height_above_seafloor_m "
                "(produced by georeference_frame); got None"
            )
        lo, hi = altitude_band
        # NaN (no bottom detected) fails the comparison, so those samples drop.
        akeep = np.isfinite(hab) & (hab >= lo) & (hab <= hi)
        keep = akeep if keep is None else (keep & akeep)
    if keep is not None:
        e, n, a = e[keep], n[keep], a[keep]

    empty = np.empty(0, dtype=np.int64)
    if e.size == 0:
        return empty, empty, np.empty(0), (None if reduce == "max" else empty)

    ie = np.floor(e / cell_m).astype(np.int64)
    jn = np.floor(n / cell_m).astype(np.int64)
    # Encode the cell as a single int64 key so the per-ping reduce uses a 1-D
    # unique + C-level bincount / reduceat instead of a 2-D lexsort (much faster).
    ie_min = int(ie.min()); jn_min = int(jn.min())
    stride = int(jn.max()) - jn_min + 1
    key = (ie - ie_min) * stride + (jn - jn_min)

    if reduce == "max":
        order = np.argsort(key, kind="stable")
        skey = key[order]
        uniq_key, first = np.unique(skey, return_index=True)
        best = np.maximum.reduceat(a[order], first)
        return uniq_key // stride + ie_min, uniq_key % stride + jn_min, best, None
    lin = np.power(10.0, a / 10.0)
    uniq_key, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    sums = np.bincount(inv, weights=lin, minlength=uniq_key.size)
    counts = np.bincount(inv, minlength=uniq_key.size).astype(np.int64)
    return uniq_key // stride + ie_min, uniq_key % stride + jn_min, sums, counts


class GeoMosaic:
    """Accumulate geo-referenced samples into a fixed-cell plan-view grid.

    Streaming and memory-proportional-to-occupied-cells. Each ping is reduced to
    its unique cells with numpy (:func:`_ping_cell_partials`) and **buffered** as
    sparse ``(iE, iN, value[, count])`` rows in arrival order; a single global
    vectorized reduce (:meth:`finalize`) collapses the buffer to the dense grid.
    Buffers are compacted once they exceed ``compact_rows`` so memory stays
    proportional to occupied cells, not to the ping count. There is **no
    per-ping Python loop** (the old dict-merge) and **no per-cell Python loop** in
    finalize — both were the accumulation hot spots.

    ``reduce="max"`` is peak-hold (good for thin plume/target returns);
    ``reduce="mean"`` averages in the linear-intensity domain — accumulated
    strictly in arrival order (buffering + compaction preserve it) so the result
    is bit-for-bit independent of chunking. ``depth_band`` keeps only samples with
    ``lo <= depth <= hi`` (absolute depth); ``altitude_band`` keeps samples by
    height **above the detected seafloor** (``lo <= Zb − Z <= hi``), which follows
    the terrain; beams with no bottom detection are dropped. The two bands compose.
    """

    def __init__(
        self,
        cell_m: float,
        *,
        reduce: str = "max",
        depth_band: Optional[Tuple[float, float]] = None,
        altitude_band: Optional[Tuple[float, float]] = None,
        compact_rows: int = 4_000_000,
    ):
        if reduce not in ("max", "mean"):
            raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
        if cell_m <= 0:
            raise ValueError("cell_m must be positive")
        self.cell_m = float(cell_m)
        self.reduce = reduce
        self.depth_band = depth_band
        self.altitude_band = altitude_band
        self.compact_rows = int(compact_rows)
        self.crs_label: Optional[str] = None
        self.epsg: Optional[int] = None
        self.n_pings = 0
        # Buffered per-ping unique-cell partials, in arrival order.
        self._ie: List[np.ndarray] = []
        self._jn: List[np.ndarray] = []
        self._val: List[np.ndarray] = []           # max: peak dB; mean: sum linear
        self._cnt: List[np.ndarray] = []           # mean only: sample counts
        self._buffered = 0                          # rows currently buffered

    def _append(self, ie, jn, val, cnt) -> None:
        if ie.size == 0:
            return
        self._ie.append(ie)
        self._jn.append(jn)
        self._val.append(val)
        if self.reduce == "mean":
            self._cnt.append(cnt)
        self._buffered += int(ie.size)
        if self._buffered > self.compact_rows:
            self._compact()

    def _reduce_rows(self, ie, jn, val, cnt):
        """Global reduce of concatenated rows to one entry per unique cell.

        Cells are encoded as a single int64 key (a 1-D ``unique`` is far cheaper
        than a 2-D ``unique``/lexsort), then reduced with C-level primitives:
        ``bincount`` for the mean sum/count and a sort + ``maximum.reduceat`` for
        the peak. The mean accumulates in the given row order — so callers that
        preserve arrival order get an order-stable (bit-identical) result.
        """
        ie_min = int(ie.min()); jn_min = int(jn.min())
        stride = int(jn.max()) - jn_min + 1                 # rows per column
        key = (ie - ie_min).astype(np.int64) * stride + (jn - jn_min)

        if self.reduce == "max":
            order = np.argsort(key, kind="stable")
            skey = key[order]
            uniq_key, first = np.unique(skey, return_index=True)
            gmax = np.maximum.reduceat(val[order], first)
            rie = uniq_key // stride + ie_min
            rjn = uniq_key % stride + jn_min
            return rie, rjn, gmax, None

        uniq_key, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        s = np.bincount(inv, weights=val, minlength=uniq_key.size)
        c = np.bincount(inv, weights=cnt, minlength=uniq_key.size).astype(np.int64)
        rie = uniq_key // stride + ie_min
        rjn = uniq_key % stride + jn_min
        return rie, rjn, s, c

    def _compact(self) -> None:
        """Collapse the buffer to one row per cell (order-preserving)."""
        if not self._ie:
            return
        ie = np.concatenate(self._ie)
        jn = np.concatenate(self._jn)
        val = np.concatenate(self._val)
        cnt = np.concatenate(self._cnt) if self.reduce == "mean" else None
        rie, rjn, rval, rcnt = self._reduce_rows(ie, jn, val, cnt)
        self._ie, self._jn, self._val = [rie], [rjn], [rval]
        self._cnt = [rcnt] if self.reduce == "mean" else []
        self._buffered = int(rie.size)

    def add(self, gs: GeoSamples) -> int:
        """Add one ping's :class:`GeoSamples`; return the cells this ping touched."""
        self.n_pings += 1
        if self.crs_label is None:
            self.crs_label = gs.crs_label
            self.epsg = gs.epsg
        ie, jn, val, cnt = _ping_cell_partials(
            gs, self.cell_m, self.reduce, self.depth_band, self.altitude_band
        )
        self._append(ie, jn, val, cnt)
        return int(ie.size)

    def add_partial(
        self, ie, jn, val, cnt=None, *, n_pings: int = 0,
        crs_label: Optional[str] = None, epsg: Optional[int] = None,
    ) -> None:
        """Append pre-reduced per-ping rows (e.g. from a worker) in arrival order.

        The rows must be the unchanged per-ping partials in ping order (not
        cross-ping reduced), so the global reduce stays order-stable and the
        parallel build matches the serial one bit-for-bit.
        """
        if self.crs_label is None and crs_label is not None:
            self.crs_label = crs_label
            self.epsg = epsg
        self.n_pings += int(n_pings)
        self._append(np.asarray(ie, np.int64), np.asarray(jn, np.int64),
                     np.asarray(val, float),
                     None if cnt is None else np.asarray(cnt, np.int64))

    def add_frame(self, frame: WCFrame, ping_time: float, nav: NavTrack, **georef_kw) -> int:
        """Georeference a frame (:func:`georeference_frame`) and add it."""
        gs = georeference_frame(frame, ping_time, nav, **georef_kw)
        return self.add(gs)

    def finalize(self) -> GeoMosaicResult:
        """Rasterize the accumulated cells to a dense :class:`GeoMosaicResult`."""
        if not self._ie:
            return GeoMosaicResult(
                amplitude_db=np.full((1, 1), np.nan),
                counts=np.zeros((1, 1), int),
                east_edges=np.array([0.0, self.cell_m]),
                north_edges=np.array([0.0, self.cell_m]),
                cell_m=self.cell_m,
                reduce=self.reduce,
                crs_label=self.crs_label or "unknown",
                n_pings=self.n_pings,
                depth_band=self.depth_band,
                epsg=self.epsg,
                altitude_band=self.altitude_band,
            )

        ie = np.concatenate(self._ie)
        jn = np.concatenate(self._jn)
        val = np.concatenate(self._val)
        cnt = np.concatenate(self._cnt) if self.reduce == "mean" else None
        cie, cjn, cval, ccnt = self._reduce_rows(ie, jn, val, cnt)

        ie_min, ie_max = int(cie.min()), int(cie.max())
        jn_min, jn_max = int(cjn.min()), int(cjn.max())
        n_east = ie_max - ie_min + 1
        n_north = jn_max - jn_min + 1

        amp = np.full((n_north, n_east), np.nan)
        cnt_grid = np.zeros((n_north, n_east), int)
        rows = cjn - jn_min
        cols = cie - ie_min
        if self.reduce == "max":
            amp[rows, cols] = cval
            cnt_grid[rows, cols] = 1
        else:
            amp[rows, cols] = 10.0 * np.log10(cval / ccnt)
            cnt_grid[rows, cols] = ccnt

        east_edges = (ie_min + np.arange(n_east + 1)) * self.cell_m
        north_edges = (jn_min + np.arange(n_north + 1)) * self.cell_m
        return GeoMosaicResult(
            amplitude_db=amp,
            counts=cnt_grid,
            east_edges=east_edges,
            north_edges=north_edges,
            cell_m=self.cell_m,
            reduce=self.reduce,
            crs_label=self.crs_label or "unknown",
            n_pings=self.n_pings,
            depth_band=self.depth_band,
            epsg=self.epsg,
            altitude_band=self.altitude_band,
        )


# ---------------------------------------------------------------------------
# Plot panel (lazy matplotlib).
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel_mosaic(result: GeoMosaicResult, out: Path, stem: str) -> Path:
    """Plan-view geo-referenced amplitude mosaic (easting/northing metres)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 10))
    masked = np.ma.masked_invalid(result.amplitude_db)
    mesh = ax.pcolormesh(
        result.east_edges, result.north_edges, masked, cmap="viridis", shading="flat"
    )
    ax.set_aspect("equal", "datalim")
    band = f", depth {result.depth_band[0]:g}–{result.depth_band[1]:g} m" if result.depth_band else ""
    if result.altitude_band:
        band += f", {result.altitude_band[0]:g}–{result.altitude_band[1]:g} m above seafloor"
    ax.set(
        xlabel="easting (m)",
        ylabel="northing (m)",
        title=f"Water-column mosaic — {result.n_pings} pings, {result.reduce}, "
              f"{result.cell_m:g} m cells{band}\n{result.crs_label}",
    )
    plt.colorbar(mesh, ax=ax, label="amplitude (dB)", shrink=0.8)
    fig.tight_layout()
    p = out / f"wc_mosaic_{stem}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Raster export: GeoTIFF (rasterio, optional) + ESRI ASCII Grid (numpy-only).
# ---------------------------------------------------------------------------


def _mosaic_wkt(epsg: Optional[int]) -> Optional[str]:
    """WKT for a projected EPSG via pyproj, or ``None`` (pyproj missing / local frame)."""
    if not epsg:
        return None
    try:
        from pyproj import CRS
    except ImportError:
        return None
    return CRS.from_epsg(epsg).to_wkt()


def export_ascii_grid(result: GeoMosaicResult, path) -> Path:
    """Write the mosaic as an ESRI ASCII Grid (``.asc``); numpy-only, always available.

    Amplitude (dB) is written north-up (row 0 = northernmost), ``NaN`` ->
    ``NODATA_value``. A companion ``.prj`` (WKT) is written when the mosaic has a
    real projected EPSG (``projector="utm"``) and pyproj is importable; in the
    local-ENU frame the grid is still valid *relative* metres but carries no
    georeferencing, so no ``.prj`` is emitted.
    """
    path = Path(path).with_suffix(".asc")
    grid = np.flipud(result.amplitude_db)          # row 0 = north (ESRI is north-up)
    nodata = -9999.0
    body = np.where(np.isfinite(grid), grid, nodata)
    header = (
        f"ncols        {result.amplitude_db.shape[1]}\n"
        f"nrows        {result.amplitude_db.shape[0]}\n"
        f"xllcorner    {float(result.east_edges[0]):.6f}\n"
        f"yllcorner    {float(result.north_edges[0]):.6f}\n"
        f"cellsize     {result.cell_m:.6f}\n"
        f"NODATA_value {nodata:g}\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, body, fmt="%.3f")
    wkt = _mosaic_wkt(result.epsg)
    if wkt:
        path.with_suffix(".prj").write_text(wkt)
    return path


def export_geotiff(result: GeoMosaicResult, path) -> Path:
    """Write the mosaic amplitude as a single-band float32 GeoTIFF (needs rasterio).

    Requires a real projected CRS — run the mosaic with ``--projector utm`` (needs
    pyproj) so ``result.epsg`` is set; the local-ENU frame has no CRS to embed (use
    :func:`export_ascii_grid` there). rasterio is imported lazily — install it via
    ``pip install 'mbes-tools[geo]'``.
    """
    path = Path(path).with_suffix(".tif")
    if not result.epsg:
        raise RuntimeError(
            "GeoTIFF needs a projected CRS but the mosaic is in a local ENU frame "
            "(epsg=None). Re-run with --projector utm (requires pyproj), or use the "
            ".asc export."
        )
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise RuntimeError(
            "GeoTIFF export requires rasterio (pip install 'mbes-tools[geo]'); "
            "use the .asc export for a dependency-free raster."
        ) from exc

    data = np.flipud(result.amplitude_db).astype(np.float32)   # north-up (row 0 = north)
    transform = from_origin(
        float(result.east_edges[0]), float(result.north_edges[-1]),
        result.cell_m, result.cell_m,
    )
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=CRS.from_epsg(result.epsg),
        transform=transform, nodata=float("nan"),
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "water-column amplitude (dB)")
    return path


def _write_rasters(result: GeoMosaicResult, out: Path, stem: str, *, geotiff: bool, asc: bool) -> List[Path]:
    """Write the requested raster export(s) alongside the PNG; errors are non-fatal."""
    made: List[Path] = []
    base = out / f"wc_mosaic_{stem}"
    if asc:
        try:
            made.append(export_ascii_grid(result, base))
        except Exception as exc:  # noqa: BLE001
            print("FAIL asc", stem, "->", type(exc).__name__, str(exc)[:160])
    if geotiff:
        try:
            made.append(export_geotiff(result, base))
        except Exception as exc:  # noqa: BLE001
            print("FAIL geotiff", stem, "->", type(exc).__name__, str(exc)[:160])
    return made


# ---------------------------------------------------------------------------
# High-level driver + CLI.
# ---------------------------------------------------------------------------


def _mwc_time(dgm) -> float:
    return dgm.time_s + dgm.time_ns * 1e-9


def _all_header_time(header) -> float:
    """Absolute time (seconds) for a `.all`/`.wcd` datagram header.

    Combines ``record_date`` (YYYYMMDD) and ``record_time_ms`` (ms since
    midnight) into a monotonic day-ordinal + seconds value, so a ``k`` water-
    column time and a companion ``P`` position time share one clock even across
    midnight. Falls back to bare seconds-since-midnight if the date is unusable —
    still self-consistent within a single day, which both sources share.
    """
    import datetime

    secs = header.record_time_ms / 1000.0
    try:
        d = int(header.record_date)
        ordinal = datetime.date(d // 10000, (d // 100) % 100, d % 100).toordinal()
    except (ValueError, TypeError):
        return secs
    return ordinal * 86400.0 + secs


def _companion_nav_paths(wc_path: Path) -> List[Path]:
    """Same-stem sibling carrying the full sensor stream, if one exists.

    Kongsberg commonly logs the water-column file next to the paired
    full-datagram file (position + attitude + install): ``.kmwcd`` → ``.kmall``,
    ``.wcd`` → ``.all``. Water-column files themselves are frequently nav-poor
    (no ``#SKM`` true heading, sometimes no position at all — a bare ``.wcd`` has
    no ``P`` datagram), so the companion is where georeferencing nav really lives.
    """
    wc_path = Path(wc_path)
    mate = {".kmwcd": ".kmall", ".wcd": ".all"}.get(wc_path.suffix.lower())
    if not mate:
        return []
    cand = wc_path.with_suffix(mate)
    return [cand] if cand.exists() and cand != wc_path else []


def _try_build_nav_track(paths: Sequence[Path]) -> Optional[NavTrack]:
    """Build a nav track, returning ``None`` when the source has no nav samples."""
    try:
        return build_nav_track(list(paths))
    except ValueError:
        return None


def resolve_nav_track(
    wc_path: Path,
    nav_paths: Optional[Sequence[Path]] = None,
    *,
    auto_companion: bool = True,
) -> NavTrack:
    """Resolve the :class:`NavTrack` for georeferencing a water-column file.

    Resolution order (do **not** assume the WC file is self-contained):

    1. explicit ``nav_paths`` — full user control;
    2. the same-stem companion (``.kmall``/``.all``) — richer stream, prefers
       ``#SKM`` true heading over ``#SPO`` course-over-ground;
    3. the WC file's own nav (``#SPO`` / ``P``), if any;
    4. otherwise :class:`ValueError` telling the caller to supply a companion.
    """
    if nav_paths:
        return build_nav_track(list(nav_paths))
    if auto_companion:
        comp = _companion_nav_paths(wc_path)
        if comp:
            nav = _try_build_nav_track(comp)
            if nav is not None:
                return nav
    own = _try_build_nav_track([wc_path])
    if own is not None:
        return own
    raise ValueError(
        f"no navigation found in {Path(wc_path).name} and no usable companion "
        f"nav file; pass a paired .kmall/.all via nav_paths (CLI: --nav)"
    )


def _resolve_nav_and_install(path, nav, nav_paths, install, install_paths, auto_companion):
    """Fill in nav + install for a WC file (see :func:`resolve_nav_track`)."""
    if nav is None:
        nav = resolve_nav_track(path, nav_paths, auto_companion=auto_companion)
    if install is None:
        sources = [*(install_paths or []), path]
        if auto_companion:
            sources += _companion_nav_paths(path)
        install = load_installation(sources)
    return nav, install


def _warn_uncovered(path, kind, n_uncovered, nav):
    lo, hi = nav.time_span
    import warnings
    warnings.warn(
        f"{Path(path).name}: {n_uncovered} {kind} ping(s) skipped — nav track "
        f"{nav.position_source} spans [{lo:.0f},{hi:.0f}] but did not cover their "
        f"time(s); check that the nav file matches this water-column file.",
        stacklevel=3,
    )


def _accumulate_into(
    mosaic: "GeoMosaic",
    anchor_ref: List[Optional[Tuple[float, float]]],
    items,
    time_fn,
    frame_fn,
    nav,
    *,
    install,
    projector: str,
    max_depth_m: Optional[float],
    on_uncovered: str,
    coverage_tol_s: float,
    limit: Optional[int],
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
) -> int:
    """Georeference one file's pings into ``mosaic``; return the skipped count.

    ``time_fn(item)`` is evaluated first so an out-of-coverage ping can be
    skipped *before* the (more expensive) ``frame_fn(item)`` decode.
    ``anchor_ref`` is a 1-element list holding the shared ``(lon, lat)`` local-ENU
    anchor — ``[None]`` until the first kept ping sets it — so a composite mosaic
    keeps one anchor (hence one CRS) across many files.
    """
    from mbes_tools.water_column import apply_min_slant_range

    lo, hi = nav.time_span
    normalizer = frame_normalizer(normalize)
    n_uncovered = 0
    for i, item in enumerate(items):
        if limit is not None and i >= limit:
            break
        t = time_fn(item)
        if not (lo - coverage_tol_s <= t <= hi + coverage_tol_s):
            n_uncovered += 1
            if on_uncovered == "skip":
                continue
        frame = frame_fn(item)
        if clean_water:
            frame = apply_min_slant_range(frame, guard_m=msr_guard_m)
        if normalizer is not None:
            frame = normalizer(frame)
        if anchor_ref[0] is None:
            lat0, lon0 = (float(v) for v in nav.position_at(t))
            anchor_ref[0] = (lon0, lat0)
        mosaic.add_frame(
            frame, t, nav, anchor_lonlat=anchor_ref[0], install=install,
            projector=projector, max_depth_m=max_depth_m,
            apply_attitude=apply_attitude, stabilized_beams=stabilized_beams,
        )
    return n_uncovered


def _accumulate_mosaic(
    path,
    items,
    time_fn,
    frame_fn,
    nav,
    install,
    *,
    kind: str,
    cell_m: float,
    reduce: str,
    depth_band: Optional[Tuple[float, float]],
    projector: str,
    max_depth_m: Optional[float],
    on_uncovered: str,
    coverage_tol_s: float,
    limit: Optional[int],
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    altitude_band: Optional[Tuple[float, float]] = None,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
) -> GeoMosaicResult:
    """Single-file wrapper over :func:`_accumulate_into` (own mosaic + anchor)."""
    if on_uncovered not in ("skip", "clamp"):
        raise ValueError(f"on_uncovered must be 'skip' or 'clamp', got {on_uncovered!r}")
    mosaic = GeoMosaic(cell_m, reduce=reduce, depth_band=depth_band, altitude_band=altitude_band)
    n_uncovered = _accumulate_into(
        mosaic, [None], items, time_fn, frame_fn, nav, install=install,
        projector=projector, max_depth_m=max_depth_m, on_uncovered=on_uncovered,
        coverage_tol_s=coverage_tol_s, limit=limit,
        apply_attitude=apply_attitude, stabilized_beams=stabilized_beams, normalize=normalize,
        clean_water=clean_water, msr_guard_m=msr_guard_m,
    )
    if n_uncovered and on_uncovered == "skip":
        _warn_uncovered(path, kind, n_uncovered, nav)
    result = mosaic.finalize()
    result.n_uncovered = n_uncovered
    return result


def build_mosaic_from_kmall(
    path: Path,
    *,
    nav: Optional[NavTrack] = None,
    nav_paths: Optional[Sequence[Path]] = None,
    install: Optional[InstallationParameters] = None,
    install_paths: Optional[Sequence[Path]] = None,
    auto_companion: bool = True,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    altitude_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    on_uncovered: str = "skip",
    coverage_tol_s: float = 2.0,
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    limit: Optional[int] = None,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
) -> GeoMosaicResult:
    """Accumulate a mosaic from every ``#MWC`` ping of a `.kmall`/`.kmwcd` file.

    Nav is resolved via :func:`resolve_nav_track` (explicit ``nav`` object >
    ``nav_paths`` > same-stem companion > the WC file's own nav) — the WC file is
    **not** assumed to carry position/heading. Install (lever arm) is read from
    ``install_paths``, the WC file, then the companion — the first that parses. A
    shared anchor (the first ping's position) fixes the local ENU frame.

    ``on_uncovered`` guards against a nav track that does not span the ping times
    (e.g. a companion file from a different interval, or a clock offset):
    ``"skip"`` (default) drops such pings — since ``position_at`` would otherwise
    **silently clamp** them to a track endpoint, placing them tens of km away —
    and records the count in ``GeoMosaicResult.n_uncovered``; ``"clamp"`` keeps
    the old clamp-to-nearest behavior. ``coverage_tol_s`` admits small clip-edge
    gaps at the track ends.
    """
    from mbes_tools.kmwcd import iter_mwc_datagrams

    path = Path(path)
    nav, install = _resolve_nav_and_install(
        path, nav, nav_paths, install, install_paths, auto_companion
    )
    return _accumulate_mosaic(
        path, iter_mwc_datagrams(path), _mwc_time, frame_from_mwc, nav, install,
        kind="#MWC", cell_m=cell_m, reduce=reduce, depth_band=depth_band,
        altitude_band=altitude_band, normalize=normalize,
        clean_water=clean_water, msr_guard_m=msr_guard_m,
        projector=projector, max_depth_m=max_depth_m, on_uncovered=on_uncovered,
        coverage_tol_s=coverage_tol_s, limit=limit,
        apply_attitude=apply_attitude, stabilized_beams=stabilized_beams,
    )


def build_mosaic_from_wcd(
    path: Path,
    *,
    nav: Optional[NavTrack] = None,
    nav_paths: Optional[Sequence[Path]] = None,
    install: Optional[InstallationParameters] = None,
    install_paths: Optional[Sequence[Path]] = None,
    auto_companion: bool = True,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    altitude_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    on_uncovered: str = "skip",
    coverage_tol_s: float = 2.0,
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    limit: Optional[int] = None,
    allow_incomplete: bool = False,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
) -> GeoMosaicResult:
    """Accumulate a mosaic from a `.wcd`/`.all` ``k`` Water Column file.

    Same as :func:`build_mosaic_from_kmall` but for the ``.all`` family: the
    (often fragmented) ``k`` datagrams are reassembled by ``counter`` into full
    pings (:func:`mbes_tools.water_column.reassemble_wcd_pings`), and both the
    ping time and the ``P`` nav time use the same absolute ``date + time``
    clock (:func:`_all_header_time`). A bare `.wcd` has **no position of its
    own** — nav almost always comes from the companion `.all` (auto-discovered),
    which is exactly why the WC file must not be assumed self-contained.
    """
    from mbes_tools.wcd import iter_water_column_datagrams
    from mbes_tools.water_column import reassemble_wcd_pings

    path = Path(path)
    nav, install = _resolve_nav_and_install(
        path, nav, nav_paths, install, install_paths, auto_companion
    )
    pings = reassemble_wcd_pings(
        iter_water_column_datagrams(path), allow_incomplete=allow_incomplete
    )
    return _accumulate_mosaic(
        path, pings, lambda p: _all_header_time(p.header), frame_from_wcd, nav, install,
        kind="k", cell_m=cell_m, reduce=reduce, depth_band=depth_band,
        altitude_band=altitude_band, normalize=normalize,
        clean_water=clean_water, msr_guard_m=msr_guard_m,
        projector=projector, max_depth_m=max_depth_m, on_uncovered=on_uncovered,
        coverage_tol_s=coverage_tol_s, limit=limit,
        apply_attitude=apply_attitude, stabilized_beams=stabilized_beams,
    )


def build_mosaic(path: Path, **kwargs) -> GeoMosaicResult:
    """Dispatch to the `.kmall`/`.kmwcd` or `.wcd`/`.all` mosaic builder by extension.

    ``allow_incomplete`` (``.wcd`` only) is ignored for the `.kmall` path.
    """
    ext = Path(path).suffix.lower()
    if ext in (".kmall", ".kmwcd"):
        kwargs.pop("allow_incomplete", None)
        return build_mosaic_from_kmall(path, **kwargs)
    if ext in (".wcd", ".all"):
        return build_mosaic_from_wcd(path, **kwargs)
    raise ValueError(f"cannot build a water-column mosaic from {ext!r} files")


def _file_ping_source(path: Path, *, allow_incomplete: bool = False):
    """``(kind, items, time_fn, frame_fn)`` for a WC file, selected by extension."""
    ext = Path(path).suffix.lower()
    if ext in (".kmall", ".kmwcd"):
        from mbes_tools.kmwcd import iter_mwc_datagrams
        return "#MWC", iter_mwc_datagrams(Path(path)), _mwc_time, frame_from_mwc
    if ext in (".wcd", ".all"):
        from mbes_tools.wcd import iter_water_column_datagrams
        from mbes_tools.water_column import reassemble_wcd_pings
        pings = reassemble_wcd_pings(
            iter_water_column_datagrams(Path(path)), allow_incomplete=allow_incomplete
        )
        return "k", pings, lambda p: _all_header_time(p.header), frame_from_wcd
    raise ValueError(f"cannot read water-column pings from {ext!r} files")


def _first_kept_anchor(paths, cfg) -> Optional[Tuple[float, float]]:
    """The ``(lon, lat)`` of the first ping any file would decode, in file order.

    Mirrors :func:`_accumulate_into`'s lazy anchor: the first covered ping (or,
    under ``on_uncovered="clamp"``, the first ping) of the first nav-resolvable
    file. Fixing it up front lets every parallel worker share one grid/CRS and
    match the serial build bit-for-bit.
    """
    for p in paths:
        p = Path(p)
        try:
            nav, _ = _resolve_nav_and_install(
                p, None, cfg["nav_paths"], None, cfg["install_paths"], cfg["auto_companion"]
            )
        except ValueError:
            continue
        _, items, time_fn, _ = _file_ping_source(p, allow_incomplete=cfg["allow_incomplete"])
        lo, hi = nav.time_span
        for i, item in enumerate(items):
            if cfg["limit"] is not None and i >= cfg["limit"]:
                break
            t = time_fn(item)
            covered = lo - cfg["coverage_tol_s"] <= t <= hi + cfg["coverage_tol_s"]
            if not covered and cfg["on_uncovered"] == "skip":
                continue
            lat0, lon0 = (float(v) for v in nav.position_at(t))
            return (lon0, lat0)
    return None


def _mosaic_worker(args):
    """Process one file into its per-ping unique-cell rows (picklable, top-level).

    Uses the shared ``cfg['anchor']`` so cell indices align across workers, and
    returns the per-ping partials **concatenated in ping order, not reduced across
    pings** — the main process does the single global reduce, so parallel and
    serial builds are bit-identical (see :meth:`GeoMosaic.add_partial`).
    """
    path, cfg = args
    path = Path(path)
    try:
        nav, install = _resolve_nav_and_install(
            path, None, cfg["nav_paths"], None, cfg["install_paths"], cfg["auto_companion"]
        )
    except ValueError:
        return None  # no navigation -> file skipped (not fatal)
    kind, items, time_fn, frame_fn = _file_ping_source(path, allow_incomplete=cfg["allow_incomplete"])
    lo, hi = nav.time_span
    reduce = cfg["reduce"]
    normalizer = frame_normalizer(cfg.get("normalize"))
    clean_water = cfg.get("clean_water", False)
    msr_guard_m = cfg.get("msr_guard_m", 0.0)
    if clean_water:
        from mbes_tools.water_column import apply_min_slant_range
    ies: List[np.ndarray] = []
    jns: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    cnts: List[np.ndarray] = []
    n_unc = 0
    n_pings = 0
    crs_label = None
    epsg = None
    for i, item in enumerate(items):
        if cfg["limit"] is not None and i >= cfg["limit"]:
            break
        t = time_fn(item)
        if not (lo - cfg["coverage_tol_s"] <= t <= hi + cfg["coverage_tol_s"]):
            n_unc += 1
            if cfg["on_uncovered"] == "skip":
                continue
        frame = frame_fn(item)
        if clean_water:
            frame = apply_min_slant_range(frame, guard_m=msr_guard_m)
        if normalizer is not None:
            frame = normalizer(frame)
        gs = georeference_frame(
            frame, t, nav, anchor_lonlat=cfg["anchor"], install=install,
            projector=cfg["projector"], max_depth_m=cfg["max_depth_m"],
            apply_attitude=cfg["apply_attitude"], stabilized_beams=cfg["stabilized_beams"],
        )
        if crs_label is None:
            crs_label, epsg = gs.crs_label, gs.epsg
        ie, jn, val, cnt = _ping_cell_partials(
            gs, cfg["cell_m"], reduce, cfg["depth_band"], cfg["altitude_band"]
        )
        n_pings += 1
        if ie.size:
            ies.append(ie); jns.append(jn); vals.append(val)
            if reduce == "mean":
                cnts.append(cnt)
    if ies:
        ie = np.concatenate(ies); jn = np.concatenate(jns); val = np.concatenate(vals)
        cnt = np.concatenate(cnts) if reduce == "mean" else None
    else:
        ie = np.empty(0, np.int64); jn = np.empty(0, np.int64); val = np.empty(0)
        cnt = np.empty(0, np.int64) if reduce == "mean" else None
    return {
        "ie": ie, "jn": jn, "val": val, "cnt": cnt, "n_unc": n_unc, "n_pings": n_pings,
        "crs_label": crs_label, "epsg": epsg, "kind": kind,
        "nav_span": nav.time_span, "nav_source": nav.position_source,
    }


def build_composite_mosaic(
    paths: Sequence[Path],
    *,
    nav_paths: Optional[Sequence[Path]] = None,
    install_paths: Optional[Sequence[Path]] = None,
    auto_companion: bool = True,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    altitude_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    on_uncovered: str = "skip",
    coverage_tol_s: float = 2.0,
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    limit: Optional[int] = None,
    allow_incomplete: bool = False,
    workers: int = 1,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
    verbose: bool = False,
) -> GeoMosaicResult:
    """Accumulate **many** water-column files into one shared plan-view mosaic.

    Every file's pings go into a single :class:`GeoMosaic` sharing one anchor
    (the first kept ping of the first file), so adjacent survey lines compose
    into one coverage map — filling the along-track gaps a single line leaves.
    Files may mix ``.kmwcd``/``.kmall`` and ``.wcd``/``.all`` (each resolves its
    own companion nav). A file with no resolvable nav is skipped, not fatal.

    ``workers`` > 1 georeferences the files across a process pool (one file per
    task): the shared anchor is fixed up front (:func:`_first_kept_anchor`) and
    each worker returns its per-ping partials, merged **in file order** — so the
    result is bit-for-bit identical to the serial (``workers=1``) build for both
    ``max`` and ``mean``. The default ``workers=1`` keeps the original streaming
    path (and its per-file uncovered warnings).

    The returned :class:`GeoMosaicResult` covers the union extent;
    ``n_pings``/``n_uncovered`` are summed across the contributing files.
    """
    if on_uncovered not in ("skip", "clamp"):
        raise ValueError(f"on_uncovered must be 'skip' or 'clamp', got {on_uncovered!r}")
    mosaic = GeoMosaic(cell_m, reduce=reduce, depth_band=depth_band, altitude_band=altitude_band)

    if workers and workers > 1:
        return _build_composite_parallel(
            paths, mosaic, workers=workers, nav_paths=nav_paths, install_paths=install_paths,
            auto_companion=auto_companion, projector=projector, max_depth_m=max_depth_m,
            on_uncovered=on_uncovered, coverage_tol_s=coverage_tol_s,
            apply_attitude=apply_attitude, stabilized_beams=stabilized_beams,
            limit=limit, allow_incomplete=allow_incomplete, normalize=normalize,
            clean_water=clean_water, msr_guard_m=msr_guard_m, verbose=verbose,
        )

    anchor_ref: List[Optional[Tuple[float, float]]] = [None]
    total_uncovered = 0
    for p in paths:
        p = Path(p)
        try:
            nav, install = _resolve_nav_and_install(
                p, None, nav_paths, None, install_paths, auto_companion
            )
        except ValueError as exc:
            if verbose:
                print("SKIP", p.name, "-> no navigation:", str(exc)[:160])
            continue
        kind, items, time_fn, frame_fn = _file_ping_source(p, allow_incomplete=allow_incomplete)
        n_unc = _accumulate_into(
            mosaic, anchor_ref, items, time_fn, frame_fn, nav, install=install,
            projector=projector, max_depth_m=max_depth_m, on_uncovered=on_uncovered,
            coverage_tol_s=coverage_tol_s, limit=limit,
            apply_attitude=apply_attitude, stabilized_beams=stabilized_beams, normalize=normalize,
            clean_water=clean_water, msr_guard_m=msr_guard_m,
        )
        total_uncovered += n_unc
        if verbose:
            status = f"skipped {n_unc} uncovered" if n_unc else "all pings covered"
            print(f"  + {p.name}: nav={nav.position_source}, {status}")
        if n_unc and on_uncovered == "skip":
            _warn_uncovered(p, kind, n_unc, nav)
    result = mosaic.finalize()
    result.n_uncovered = total_uncovered
    return result


def _build_composite_parallel(
    paths, mosaic, *, workers, nav_paths, install_paths, auto_companion, projector,
    max_depth_m, on_uncovered, coverage_tol_s, apply_attitude, stabilized_beams,
    limit, allow_incomplete, normalize, clean_water, msr_guard_m, verbose,
) -> GeoMosaicResult:
    """Process-pool file-parallel composite; merges partials in file order."""
    import warnings
    from concurrent.futures import ProcessPoolExecutor

    cfg = dict(
        nav_paths=list(nav_paths) if nav_paths else None,
        install_paths=list(install_paths) if install_paths else None,
        auto_companion=auto_companion, cell_m=mosaic.cell_m, reduce=mosaic.reduce,
        depth_band=mosaic.depth_band, altitude_band=mosaic.altitude_band,
        projector=projector, max_depth_m=max_depth_m, on_uncovered=on_uncovered,
        coverage_tol_s=coverage_tol_s, apply_attitude=apply_attitude,
        stabilized_beams=stabilized_beams, limit=limit, allow_incomplete=allow_incomplete,
        normalize=normalize, clean_water=clean_water, msr_guard_m=msr_guard_m,
    )
    anchor = _first_kept_anchor(paths, cfg)
    if anchor is None:  # no nav-resolvable file with a decodable ping
        return mosaic.finalize()
    cfg["anchor"] = anchor

    total_uncovered = 0
    tasks = [(str(p), cfg) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # ex.map preserves input (file) order -> order-stable mean.
        for p, res in zip(paths, ex.map(_mosaic_worker, tasks)):
            if res is None:
                if verbose:
                    print("SKIP", Path(p).name, "-> no navigation")
                continue
            mosaic.add_partial(
                res["ie"], res["jn"], res["val"], res["cnt"],
                n_pings=res["n_pings"], crs_label=res["crs_label"], epsg=res["epsg"],
            )
            total_uncovered += res["n_unc"]
            if verbose:
                status = f"skipped {res['n_unc']} uncovered" if res["n_unc"] else "all pings covered"
                print(f"  + {Path(p).name}: nav={res['nav_source']}, {status}")
            if res["n_unc"] and on_uncovered == "skip":
                lo, hi = res["nav_span"]
                warnings.warn(
                    f"{Path(p).name}: {res['n_unc']} {res['kind']} ping(s) skipped — "
                    f"nav track {res['nav_source']} spans [{lo:.0f},{hi:.0f}] but did not "
                    f"cover their time(s); check that the nav file matches this file.",
                    stacklevel=2,
                )
    result = mosaic.finalize()
    result.n_uncovered = total_uncovered
    return result


def generate(
    output_dir,
    mwc_files: Sequence = (),
    *,
    nav_paths: Optional[Sequence[Path]] = None,
    install_paths: Optional[Sequence[Path]] = None,
    auto_companion: bool = True,
    combine: bool = False,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    altitude_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    on_uncovered: str = "skip",
    apply_attitude: bool = True,
    stabilized_beams: bool = True,
    limit: Optional[int] = None,
    write_geotiff: bool = False,
    write_asc: bool = False,
    workers: int = 1,
    normalize: Optional[str] = None,
    clean_water: bool = False,
    msr_guard_m: float = 0.0,
) -> List[Path]:
    """Build + render mosaic panel(s) from water-column files. Returns output paths.

    Accepts both families (dispatched by extension): `.kmwcd`/`.kmall` (``#MWC``)
    and `.wcd`/`.all` (``k``). ``nav_paths`` (if given) supplies the navigation
    for every file; otherwise each file's nav is resolved independently
    (companion sibling, then in-file). With ``combine=True`` all files accumulate
    into **one** composite mosaic (one panel); otherwise one panel per file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if combine:
        files = [Path(f) for f in mwc_files]
        print(f"Combining {len(files)} file(s) into one composite mosaic...")
        result = build_composite_mosaic(
            files, nav_paths=nav_paths, install_paths=install_paths,
            auto_companion=auto_companion, cell_m=cell_m, reduce=reduce,
            depth_band=depth_band, altitude_band=altitude_band,
            projector=projector, max_depth_m=max_depth_m,
            on_uncovered=on_uncovered, apply_attitude=apply_attitude,
            stabilized_beams=stabilized_beams, limit=limit, workers=workers,
            normalize=normalize, clean_water=clean_water, msr_guard_m=msr_guard_m, verbose=True,
        )
        uncov = f", {result.n_uncovered} uncovered-skipped" if result.n_uncovered else ""
        print(f"OK   composite: {result.n_pings} pings, grid {result.amplitude_db.shape}, "
              f"{int(result.counts.sum())} samples{uncov}, {result.crs_label}")
        stem = f"composite_{len(files)}files"
        made: List[Path] = []
        try:
            made.append(panel_mosaic(result, out, stem))
        except Exception as exc:  # noqa: BLE001
            print("FAIL panel", stem, "->", type(exc).__name__, str(exc)[:120])
        made += _write_rasters(result, out, stem, geotiff=write_geotiff, asc=write_asc)
        return made

    made: List[Path] = []
    for f in mwc_files:
        stem = Path(f).stem
        try:
            nav = resolve_nav_track(Path(f), nav_paths, auto_companion=auto_companion)
        except ValueError as exc:
            print("SKIP", f, "-> no navigation:", str(exc)[:200])
            continue
        try:
            result = build_mosaic(
                Path(f), nav=nav, install_paths=install_paths, auto_companion=auto_companion,
                cell_m=cell_m, reduce=reduce, depth_band=depth_band,
                altitude_band=altitude_band,
                projector=projector, max_depth_m=max_depth_m,
                on_uncovered=on_uncovered, apply_attitude=apply_attitude,
                stabilized_beams=stabilized_beams, limit=limit, normalize=normalize,
                clean_water=clean_water, msr_guard_m=msr_guard_m,
            )
        except Exception as exc:  # noqa: BLE001
            print("FAIL", f, "->", type(exc).__name__, str(exc)[:160])
            continue
        uncov = f", {result.n_uncovered} uncovered-skipped" if result.n_uncovered else ""
        print(f"OK   {stem}: {result.n_pings} pings, grid {result.amplitude_db.shape}, "
              f"{int(result.counts.sum())} samples, nav={nav.position_source}/{nav.heading_source}{uncov}, "
              f"{result.crs_label}")
        try:
            made.append(panel_mosaic(result, out, stem))
        except Exception as exc:  # noqa: BLE001
            print("FAIL panel", stem, "->", type(exc).__name__, str(exc)[:120])
        made += _write_rasters(result, out, stem, geotiff=write_geotiff, asc=write_asc)
    return made


def _parse_band(spec: Optional[str]) -> Optional[Tuple[float, float]]:
    if not spec:
        return None
    lo, _, hi = spec.partition(":")
    return (float(lo), float(hi))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Geo-referenced water-column mosaic: accumulate water-column pings "
                    "(.kmwcd/.kmall #MWC or .wcd/.all k) into a plan-view "
                    "(easting/northing) amplitude grid using companion nav + install."
    )
    ap.add_argument("-o", "--output", default="mbes_wc_mosaic",
                    help="Output directory for the mosaic PNG (+ any --geotiff/--asc rasters).")
    ap.add_argument("--mwc", nargs="+", required=True,
                    help="One or more water-column files (.kmwcd/.kmall or .wcd/.all).")
    ap.add_argument("--nav", nargs="+", default=None, metavar="FILE",
                    help="Companion file(s) to build the nav track from (.kmall/.all — prefers "
                         "#SKM true heading). Overrides in-file nav; use when the WC file lacks "
                         "position/heading. If omitted, a same-stem .kmall/.all sibling is "
                         "auto-discovered, then the WC file's own #SPO/P is tried.")
    ap.add_argument("--install", nargs="+", default=None, metavar="FILE",
                    help="File(s) to read install params (lever arms) from; default = the WC file "
                         "or its companion.")
    ap.add_argument("--no-auto-companion", action="store_true",
                    help="Do not auto-use a same-stem .kmall/.all sibling for nav/install.")
    ap.add_argument("--combine", action="store_true",
                    help="Accumulate ALL --mwc files into one composite mosaic (one panel) "
                         "instead of one panel per file — e.g. adjacent survey lines into a "
                         "single coverage map.")
    ap.add_argument("--cell-m", type=float, default=25.0, help="Mosaic cell size (metres).")
    ap.add_argument("--reduce", choices=["max", "mean"], default="max",
                    help="Cell aggregation over depth: peak-hold max (default) or intensity-mean.")
    ap.add_argument("--depth-band", default=None, metavar="LO:HI",
                    help="Keep only samples with depth in [LO,HI] m (e.g. 200:2000 for a midwater band).")
    ap.add_argument("--altitude-band", default=None, metavar="LO:HI",
                    help="Keep only samples whose height ABOVE the detected seafloor is in [LO,HI] m "
                         "(e.g. 20:200 for a near-bottom layer that tracks the terrain). Beams with no "
                         "bottom detection are excluded. Composes with --depth-band (both must hold).")
    ap.add_argument("--max-depth-m", type=float, default=None, help="Drop samples deeper than this.")
    ap.add_argument("--projector", choices=["auto", "utm", "local"], default="auto",
                    help="Coordinate frame: auto (UTM if pyproj else local ENU), utm, or local.")
    ap.add_argument("--on-uncovered", choices=["skip", "clamp"], default="skip",
                    help="Pings whose time the nav track does not span: skip them (default, "
                         "avoids clamping to a far-away endpoint) or clamp to the nearest fix.")
    ap.add_argument("--no-attitude", action="store_true",
                    help="Ignore #SKM/A roll-pitch-heave (heading-only). Default applies them: "
                         "heave -> depth and the lever arm rotated by the full vessel pose.")
    ap.add_argument("--unstabilized-beams", action="store_true",
                    help="Rotate the beam fan by roll/pitch too. ONLY for raw array-relative "
                         "angles — Kongsberg #MWC/k beamPointAngReVertical is already "
                         "roll/pitch-stabilized, so this double-corrects normal data.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N pings per file.")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="With --combine, georeference the files across N processes (one file "
                         "per task; default 1 = serial). The result is identical to serial.")
    ap.add_argument("--normalize", choices=["none", "empirical"], default="none",
                    help="Remove acquisition gain before gridding: 'empirical' de-trends each "
                         "ping's amplitude over the open water (per-range TVG/absorption + "
                         "per-beam-angle beam-pattern, via median polish), so real midwater/plume "
                         "structure stands out. Relative dB, not calibrated Sv. Default none.")
    ap.add_argument("--clean-water", action="store_true",
                    help="Keep only the bottom-sidelobe-free water column: drop samples at/beyond "
                         "each ping's minimum bottom-detect slant range (the nadir range), where "
                         "the seafloor sidelobe contaminates every beam. Composes with "
                         "--depth-band/--altitude-band (all must hold).")
    ap.add_argument("--msr-guard-m", type=float, default=0.0, metavar="M",
                    help="With --clean-water, pull the minimum-slant-range cutoff inward by M "
                         "metres to stay clear of the sidelobe onset (default 0 = exact nadir range).")
    ap.add_argument("--geotiff", action="store_true",
                    help="Also write a georeferenced GeoTIFF (needs --projector utm + rasterio; "
                         "pip install 'mbes-tools[geo]').")
    ap.add_argument("--asc", action="store_true",
                    help="Also write an ESRI ASCII Grid (.asc + .prj sidecar); numpy-only, works "
                         "in the base env and is GDAL-convertible to GeoTIFF.")
    args = ap.parse_args(argv)

    made = generate(
        args.output, mwc_files=args.mwc, nav_paths=args.nav, install_paths=args.install,
        auto_companion=not args.no_auto_companion, combine=args.combine,
        cell_m=args.cell_m, reduce=args.reduce,
        depth_band=_parse_band(args.depth_band),
        altitude_band=_parse_band(args.altitude_band), projector=args.projector,
        max_depth_m=args.max_depth_m, on_uncovered=args.on_uncovered,
        apply_attitude=not args.no_attitude, stabilized_beams=not args.unstabilized_beams,
        limit=args.limit, write_geotiff=args.geotiff, write_asc=args.asc,
        workers=args.workers, normalize=args.normalize,
        clean_water=args.clean_water, msr_guard_m=args.msr_guard_m,
    )
    print(f"\nWrote {len(made)} mosaic output(s) to {args.output}")


if __name__ == "__main__":
    main()
