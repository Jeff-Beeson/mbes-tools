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
   intensity-mean over an optional ``depth_band`` (e.g. a midwater band for
   plume mapping). ``finalize()`` returns a dense grid + edges + CRS.

Verified against the committed EM124 ``.kmwcd`` (self-contained: ``#MWC`` +
``#SPO`` + ``#IIP``) and the full 374-ping TN447 ``.kmwcd``; see
``docs/BUILD_STATUS.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools.install_params import InstallationParameters
from mbes_tools.projection import resolve_target_crs
from mbes_tools.wc_diagnostics import WCFrame, frame_from_mwc, wedge_coords

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

    @classmethod
    def from_lists(cls, t_pos, lat, lon, t_hdg, hdg_deg, position_source, heading_source):
        if not t_pos or not t_hdg:
            raise ValueError("NavTrack needs at least one position and one heading sample")
        tp = np.asarray(t_pos, float)
        op = np.argsort(tp, kind="stable")
        th = np.asarray(t_hdg, float)
        oh = np.argsort(th, kind="stable")
        return cls(
            t_pos=tp[op],
            lat=np.asarray(lat, float)[op],
            lon=np.asarray(lon, float)[op],
            t_hdg=th[oh],
            hdg_deg=np.asarray(hdg_deg, float)[oh],
            position_source=position_source,
            heading_source=heading_source,
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
    for p in paths:
        for d in iter_skm_datagrams(Path(p), on_error="skip"):
            for s in d.samples:
                if -90.0 <= s.latitude_deg <= 90.0 and -180.0 <= s.longitude_deg <= 180.0:
                    t = s.time_sec + s.time_nanosec * 1e-9
                    tp.append(t); la.append(s.latitude_deg); lo.append(s.longitude_deg)
                    th.append(t); hd.append(s.heading_deg)
    if tp:
        return NavTrack.from_lists(tp, la, lo, th, hd, "#SKM", "#SKM heading")

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
    are seconds-since-midnight from the datagram header (single-day assumption).
    """
    from mbes_tools.all import iter_position_datagrams

    tp: List[float] = []
    la: List[float] = []
    lo: List[float] = []
    hd: List[float] = []
    for p in paths:
        for d in iter_position_datagrams(Path(p)):
            if -90.0 <= d.latitude_deg <= 90.0 and -180.0 <= d.longitude_deg <= 180.0:
                t = d.header.time_seconds
                tp.append(t); la.append(d.latitude_deg); lo.append(d.longitude_deg)
                hd.append(d.heading_deg)
    return NavTrack.from_lists(tp, la, lo, list(tp), hd, "P position", "P heading")


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
) -> GeoSamples:
    """Place a ping's amplitude samples at ``(easting_m, northing_m, depth_m)``.

    Geometry: across-track ``X`` (positive = port) and depth ``Z`` come from the
    Slice-1 wedge (``r = c·k/(2·fs)``). In the vessel frame a sample sits at
    forward ``x_v`` (0 + lever-arm X), starboard ``y_v`` (``-X`` + lever-arm Y);
    rotating by heading ``H`` gives east ``x_v·sinH + y_v·cosH`` and north
    ``x_v·cosH − y_v·sinH``, added to the platform position. Depth is below the
    transducer (plus the lever-arm Z when ``install`` is given).

    ``projector``: ``"local"`` = ENU metres about ``anchor_lonlat`` (defaults to
    the ping's own position); ``"utm"`` = true UTM via pyproj; ``"auto"`` = UTM
    if pyproj is importable, else local. The auto-UTM EPSG is always resolved for
    the ``crs_label``.
    """
    lat0, lon0 = (float(v) for v in nav.position_at(ping_time))
    heading = float(nav.heading_at(ping_time))
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
    x_v = fx + np.zeros_like(xs)      # forward (along-track ~ 0 for the WC fan)
    y_v = fy - xs                     # starboard = -across(+port)
    depth = fz + zs

    h = math.radians(heading)
    east = e0 + (x_v * math.sin(h) + y_v * math.cos(h))
    north = n0 + (x_v * math.cos(h) - y_v * math.sin(h))

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

    @property
    def east_centers(self) -> np.ndarray:
        return 0.5 * (self.east_edges[:-1] + self.east_edges[1:])

    @property
    def north_centers(self) -> np.ndarray:
        return 0.5 * (self.north_edges[:-1] + self.north_edges[1:])


class GeoMosaic:
    """Accumulate geo-referenced samples into a fixed-cell plan-view grid.

    Streaming and memory-proportional-to-occupied-cells: samples are reduced
    into a sparse ``{(iE, iN): value}`` dict as they arrive, then rasterized to a
    dense grid by :meth:`finalize`. ``reduce="max"`` is peak-hold (good for thin
    plume/target returns); ``reduce="mean"`` averages in the linear-intensity
    domain. ``depth_band`` keeps only samples with ``lo <= depth <= hi`` — set a
    midwater band to map plumes without the seafloor swamping the peak-hold.
    """

    def __init__(
        self,
        cell_m: float,
        *,
        reduce: str = "max",
        depth_band: Optional[Tuple[float, float]] = None,
    ):
        if reduce not in ("max", "mean"):
            raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
        if cell_m <= 0:
            raise ValueError("cell_m must be positive")
        self.cell_m = float(cell_m)
        self.reduce = reduce
        self.depth_band = depth_band
        self.crs_label: Optional[str] = None
        self.n_pings = 0
        # For max: cell -> best dB. For mean: cell -> [sum_linear, count].
        self._cells: Dict[Tuple[int, int], object] = {}

    def add(self, gs: GeoSamples) -> int:
        """Add one ping's :class:`GeoSamples`; return the number of cells touched.

        The ping's samples are reduced to their unique cells with numpy first
        (many samples share a cell), then those far fewer cells are merged into
        the running global store — so the per-sample cost stays vectorized.
        """
        self.n_pings += 1
        if self.crs_label is None:
            self.crs_label = gs.crs_label

        e, n, d, a = gs.easting_m, gs.northing_m, gs.depth_m, gs.amplitude_db
        if self.depth_band is not None:
            lo, hi = self.depth_band
            keep = (d >= lo) & (d <= hi)
            e, n, a = e[keep], n[keep], a[keep]
        if e.size == 0:
            return len(self._cells)

        ie = np.floor(e / self.cell_m).astype(np.int64)
        jn = np.floor(n / self.cell_m).astype(np.int64)
        uniq, inv = np.unique(np.column_stack([ie, jn]), axis=0, return_inverse=True)
        inv = inv.ravel()
        cells = self._cells

        if self.reduce == "max":
            best = np.full(uniq.shape[0], -np.inf)
            np.maximum.at(best, inv, a)
            for (k_e, k_n), val in zip(map(tuple, uniq.tolist()), best.tolist()):
                cur = cells.get((k_e, k_n))
                if cur is None or val > cur:
                    cells[(k_e, k_n)] = val
        else:
            lin = np.power(10.0, a / 10.0)
            sums = np.zeros(uniq.shape[0])
            np.add.at(sums, inv, lin)
            counts = np.bincount(inv, minlength=uniq.shape[0])
            for (k_e, k_n), s, ct in zip(map(tuple, uniq.tolist()), sums.tolist(), counts.tolist()):
                cur = cells.get((k_e, k_n))
                if cur is None:
                    cells[(k_e, k_n)] = [s, ct]
                else:
                    cur[0] += s
                    cur[1] += ct
        return len(cells)

    def add_frame(self, frame: WCFrame, ping_time: float, nav: NavTrack, **georef_kw) -> int:
        """Georeference a frame (:func:`georeference_frame`) and add it."""
        gs = georeference_frame(frame, ping_time, nav, **georef_kw)
        return self.add(gs)

    def finalize(self) -> GeoMosaicResult:
        """Rasterize the accumulated cells to a dense :class:`GeoMosaicResult`."""
        if not self._cells:
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
            )

        ies = np.fromiter((k[0] for k in self._cells), dtype=np.int64)
        jns = np.fromiter((k[1] for k in self._cells), dtype=np.int64)
        ie_min, ie_max = int(ies.min()), int(ies.max())
        jn_min, jn_max = int(jns.min()), int(jns.max())
        n_east = ie_max - ie_min + 1
        n_north = jn_max - jn_min + 1

        amp = np.full((n_north, n_east), np.nan)
        cnt = np.zeros((n_north, n_east), int)
        for (k_e, k_n), v in self._cells.items():
            col = k_e - ie_min
            row = k_n - jn_min
            if self.reduce == "max":
                amp[row, col] = v
                cnt[row, col] = 1
            else:
                s, c = v
                amp[row, col] = 10.0 * math.log10(s / c)
                cnt[row, col] = c

        east_edges = (ie_min + np.arange(n_east + 1)) * self.cell_m
        north_edges = (jn_min + np.arange(n_north + 1)) * self.cell_m
        return GeoMosaicResult(
            amplitude_db=amp,
            counts=cnt,
            east_edges=east_edges,
            north_edges=north_edges,
            cell_m=self.cell_m,
            reduce=self.reduce,
            crs_label=self.crs_label or "unknown",
            n_pings=self.n_pings,
            depth_band=self.depth_band,
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
# High-level driver + CLI.
# ---------------------------------------------------------------------------


def _mwc_time(dgm) -> float:
    return dgm.time_s + dgm.time_ns * 1e-9


def build_mosaic_from_kmall(
    path: Path,
    *,
    nav: Optional[NavTrack] = None,
    install: Optional[InstallationParameters] = None,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    limit: Optional[int] = None,
) -> GeoMosaicResult:
    """Accumulate a mosaic from every ``#MWC`` ping of a `.kmall`/`.kmwcd` file.

    Nav + install default to those read from the same file. A shared anchor (the
    first ping's position) fixes the local ENU frame across pings.
    """
    from mbes_tools.kmwcd import iter_mwc_datagrams

    path = Path(path)
    if nav is None:
        nav = nav_track_from_kmall([path])
    if install is None:
        install = load_installation([path])

    mosaic = GeoMosaic(cell_m, reduce=reduce, depth_band=depth_band)
    anchor: Optional[Tuple[float, float]] = None

    for i, dgm in enumerate(iter_mwc_datagrams(path)):
        if limit is not None and i >= limit:
            break
        t = _mwc_time(dgm)
        frame = frame_from_mwc(dgm)
        if anchor is None:
            lat0, lon0 = (float(v) for v in nav.position_at(t))
            anchor = (lon0, lat0)
        mosaic.add_frame(
            frame, t, nav, anchor_lonlat=anchor, install=install,
            projector=projector, max_depth_m=max_depth_m,
        )
    return mosaic.finalize()


def generate(
    output_dir,
    mwc_files: Sequence = (),
    *,
    cell_m: float = 25.0,
    reduce: str = "max",
    depth_band: Optional[Tuple[float, float]] = None,
    projector: str = "auto",
    max_depth_m: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Path]:
    """Build + render a mosaic panel per `.kmall`/`.kmwcd` file. Returns PNG paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    for f in mwc_files:
        stem = Path(f).stem
        try:
            result = build_mosaic_from_kmall(
                Path(f), cell_m=cell_m, reduce=reduce, depth_band=depth_band,
                projector=projector, max_depth_m=max_depth_m, limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            print("FAIL", f, "->", type(exc).__name__, str(exc)[:160])
            continue
        print(f"OK   {stem}: {result.n_pings} pings, grid {result.amplitude_db.shape}, "
              f"{int(result.counts.sum())} samples, {result.crs_label}")
        try:
            made.append(panel_mosaic(result, out, stem))
        except Exception as exc:  # noqa: BLE001
            print("FAIL panel", stem, "->", type(exc).__name__, str(exc)[:120])
    return made


def _parse_band(spec: Optional[str]) -> Optional[Tuple[float, float]]:
    if not spec:
        return None
    lo, _, hi = spec.partition(":")
    return (float(lo), float(hi))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Geo-referenced water-column mosaic: accumulate #MWC pings into a "
                    "plan-view (easting/northing) amplitude grid using in-file nav + install."
    )
    ap.add_argument("-o", "--output", default="mbes_wc_mosaic", help="Output directory for PNGs.")
    ap.add_argument("--mwc", nargs="+", required=True,
                    help="One or more #MWC-bearing files (.kmwcd or .kmall) with in-file #SPO/#SKM nav.")
    ap.add_argument("--cell-m", type=float, default=25.0, help="Mosaic cell size (metres).")
    ap.add_argument("--reduce", choices=["max", "mean"], default="max",
                    help="Cell aggregation over depth: peak-hold max (default) or intensity-mean.")
    ap.add_argument("--depth-band", default=None, metavar="LO:HI",
                    help="Keep only samples with depth in [LO,HI] m (e.g. 200:2000 for a midwater band).")
    ap.add_argument("--max-depth-m", type=float, default=None, help="Drop samples deeper than this.")
    ap.add_argument("--projector", choices=["auto", "utm", "local"], default="auto",
                    help="Coordinate frame: auto (UTM if pyproj else local ENU), utm, or local.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N pings per file.")
    args = ap.parse_args(argv)

    made = generate(
        args.output, mwc_files=args.mwc, cell_m=args.cell_m, reduce=args.reduce,
        depth_band=_parse_band(args.depth_band), projector=args.projector,
        max_depth_m=args.max_depth_m, limit=args.limit,
    )
    print(f"\nWrote {len(made)} mosaic panel(s) to {args.output}")


if __name__ == "__main__":
    main()
