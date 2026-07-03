"""Interactive water-column viewer — linked along-track stack + wedge fan.

The first *interactive* water-column product (the `wc_diagnostics` /
`water_column` / `water_column_geo` renderers all emit static PNGs). One window
with two linked panels for QC-ing a whole water-column file:

* **top** — an along-track *depth stack* of the **whole file**: each ping's fan
  collapsed to a vertical amplitude-vs-depth column and laid side by side
  (x = ping index along the track, y = depth). A movable cursor marks the
  selected ping.
* **bottom** — the selected ping's **navigation/attitude-corrected wedge fan**
  (across-track x depth), amplitude-coloured, with the bottom detection overlaid.

Interactive controls (in :meth:`WaterColumnViewer.show`):

* a **whole-file amplitude histogram** with a ``RangeSlider`` that sets a colour
  scale **shared by both panels**, and a **clamp/cut** toggle — out-of-range
  samples either show the end colours or are cut out (rendered transparent);
* **drag a horizontal band across the fan** to choose the across-track swath that
  feeds the along-track stack, which rebuilds live (double-click or ``r`` resets);
* a **cursor lat/lon readout** — over the fan, the geographic position of the
  cursor's across-track point for the current ping; over the stack, the nadir
  (vessel) position of the ping under the cursor.

Geometry, navigation and attitude are reused verbatim from
:mod:`mbes_tools.water_column_geo` so the viewer agrees with the mosaic product:
the Kongsberg beam pointing angles are already roll/pitch **stabilized at
receive**, so the fan is left level (re-rotating would double-correct) and the
correction that is applied is **heave** (added to depth) plus the transducer
depth offset from the installation lever arm. Vessel roll/pitch/heave are
reported in the panel title.

matplotlib is imported lazily (the project's ``gui`` extra); the data model and
the static renderer run headless (Agg), and only :meth:`WaterColumnViewer.show`
needs an interactive backend.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools.water_column_geo import (
    NavTrack,
    _file_ping_source,
    _resolve_group,
    _resolve_nav_and_install,
    _vessel_en,
)

_WGS84_A = 6378137.0


# ---------------------------------------------------------------------------
# Per-ping decimated fan + navigation/attitude.
# ---------------------------------------------------------------------------


@dataclass
class PingView:
    """One ping's decimated fan plus the nav/attitude needed to place it.

    ``amp`` is a ``[beam, sample]`` grid decimated in the range direction to
    ``sample_idx`` (absolute sample numbers), stored as float16 to keep a whole
    file in memory. Depth is ``cos(angle)*range + heave + transducer_depth`` and
    across-track is ``sin(angle)*range`` (positive = port) — the beams are left
    receive-stabilized (see the module docstring).
    """

    index: int
    time: float
    label: str
    lat: float
    lon: float
    heading_deg: float
    roll_deg: float
    pitch_deg: float
    heave_m: float
    transducer_depth_m: float
    sound_speed_m_s: float
    sample_freq_hz: float
    angles_deg: np.ndarray          # (W,)
    detected_samples: np.ndarray    # (W,) absolute sample number, 0 = no detect
    sample_idx: np.ndarray          # (S,) absolute sample numbers kept
    amp: np.ndarray                 # (W, S) float16 dB, NaN-padded
    along_track_m: float = 0.0      # filled in after the pass

    def _range(self) -> np.ndarray:
        return self.sound_speed_m_s * self.sample_idx / (2.0 * self.sample_freq_hz)

    def fan_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finite ``(across_m[+port], depth_m, amplitude_db)`` scatter points."""
        r = self._range()
        th = np.deg2rad(self.angles_deg)
        across = np.sin(th)[:, None] * r[None, :]
        depth = np.cos(th)[:, None] * r[None, :] + self.heave_m + self.transducer_depth_m
        amp = self.amp.astype(np.float32)
        f = np.isfinite(amp)
        return across[f], depth[f], amp[f]

    def bottom_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Detected-bottom ``(across_m, depth_m)`` polyline (detected beams only)."""
        good = self.detected_samples > 0
        r = self.sound_speed_m_s * self.detected_samples[good] / (2.0 * self.sample_freq_hz)
        th = np.deg2rad(self.angles_deg[good])
        return np.sin(th) * r, np.cos(th) * r + self.heave_m + self.transducer_depth_m

    def max_bottom_depth(self) -> float:
        """Deepest detected-bottom depth (0 when the ping has no detections)."""
        good = self.detected_samples > 0
        if not good.any():
            return 0.0
        r = self.sound_speed_m_s * self.detected_samples[good] / (2.0 * self.sample_freq_hz)
        d = np.cos(np.deg2rad(self.angles_deg[good])) * r + self.heave_m + self.transducer_depth_m
        return float(np.nanmax(d))

    def across_to_lonlat(self, across_m: float) -> Tuple[float, float]:
        """``(lon, lat)`` of the across-track point at ``across_m`` (+ port).

        The fan is a zero-along-track curtain, so the point lies directly abeam of
        the vessel: an equirectangular offset from the ping position rotated by
        heading — the same body→ENU rotation ``_dcm(H,0,0) @ [0, -across, ·]`` used
        to georeference the beams (north ``= across·sinH``, east ``= -across·cosH``).
        """
        h = math.radians(self.heading_deg)
        dn = across_m * math.sin(h)
        de = -across_m * math.cos(h)
        lat = self.lat + math.degrees(dn / _WGS84_A)
        lon = self.lon + math.degrees(de / (_WGS84_A * math.cos(math.radians(self.lat))))
        return lon, lat

    def depth_column(
        self,
        depth_edges: np.ndarray,
        mode: str,
        nadir_halfangle_deg: float,
        across_window: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Collapse the fan to a 1-D amplitude-vs-depth column on ``depth_edges``.

        ``mode`` sets the reduce: ``"swath-mean"`` averages in the linear-intensity
        domain, everything else peak-holds. ``across_window=(lo, hi)`` restricts
        the collapse to samples whose across-track (``sin(angle)·range``, + port)
        falls in the band — the interactive swath selection; when ``None`` the
        ``"nadir"`` mode instead keeps only beams within ``nadir_halfangle_deg`` of
        vertical (``"swath-*"`` keeps the whole swath).
        """
        r = self._range()
        th = np.deg2rad(self.angles_deg)
        depth = np.cos(th)[:, None] * r[None, :] + self.heave_m + self.transducer_depth_m
        amp = self.amp.astype(np.float32)
        if across_window is not None:
            lo, hi = across_window
            across = np.sin(th)[:, None] * r[None, :]
            amp = np.where((across >= lo) & (across <= hi), amp, np.nan)
        elif mode == "nadir":
            sel = np.abs(self.angles_deg) <= nadir_halfangle_deg
            if not sel.any():
                sel = np.zeros(self.angles_deg.size, bool)
                sel[int(np.argmin(np.abs(self.angles_deg)))] = True
            depth, amp = depth[sel], amp[sel]

        zf, af = depth.ravel(), amp.ravel()
        finite = np.isfinite(af) & np.isfinite(zf)
        zf, af = zf[finite], af[finite]
        n = depth_edges.size - 1
        col = np.full(n, np.nan, np.float32)
        if zf.size == 0:
            return col
        idx = np.digitize(zf, depth_edges) - 1
        inb = (idx >= 0) & (idx < n)
        idx, af = idx[inb], af[inb]
        if idx.size == 0:
            return col
        if mode == "swath-mean":
            lin = np.power(10.0, af / 10.0)
            sums = np.bincount(idx, weights=lin, minlength=n)
            cnts = np.bincount(idx, minlength=n)
            nz = cnts > 0
            col[nz] = 10.0 * np.log10(sums[nz] / cnts[nz])
        else:  # swath-max / nadir -> peak-hold
            peak = np.full(n, -np.inf, np.float32)
            np.maximum.at(peak, idx, af)
            hit = np.isfinite(peak) & (peak > -np.inf)
            col[hit] = peak[hit]
        return col


# ---------------------------------------------------------------------------
# Whole-file model.
# ---------------------------------------------------------------------------


@dataclass
class WaterColumnFileView:
    """A whole water-column file reduced to a linked stack + per-ping fans."""

    path: Path
    pings: List[PingView]
    depth_edges: np.ndarray
    stack: np.ndarray               # (n_depth, n_pings) dB, NaN where empty
    stack_mode: str
    nav_position_source: str
    nav_heading_source: str
    nav_attitude_source: Optional[str]
    n_total: int
    n_uncovered: int
    stride: int
    max_depth_m: float
    nadir_halfangle_deg: float = 3.0
    along_track_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Bounded whole-file subsample of finite fan amplitudes for the histogram.
    amp_sample: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    @property
    def n_pings(self) -> int:
        return len(self.pings)

    def rebuild_stack(
        self,
        mode: Optional[str] = None,
        across_window: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Recompute the along-track stack for a new mode / across-track band.

        Reuses the in-memory decimated fans (no file re-read). Updates and returns
        ``self.stack``.
        """
        if mode is not None:
            self.stack_mode = mode
        self.stack = np.column_stack([
            p.depth_column(self.depth_edges, self.stack_mode, self.nadir_halfangle_deg,
                           across_window=across_window)
            for p in self.pings
        ])
        return self.stack

    @classmethod
    def from_file(
        cls,
        path,
        *,
        nav: Optional[NavTrack] = None,
        install=None,
        nav_paths: Optional[Sequence[Path]] = None,
        install_paths: Optional[Sequence[Path]] = None,
        auto_companion: bool = True,
        max_pings: int = 500,
        fan_samples: int = 600,
        max_depth_m: Optional[float] = None,
        stack_mode: str = "swath-max",
        nadir_halfangle_deg: float = 3.0,
        allow_incomplete: bool = False,
        on_uncovered: str = "skip",
        coverage_tol_s: float = 2.0,
    ) -> "WaterColumnFileView":
        if on_uncovered not in ("skip", "clamp"):
            raise ValueError("on_uncovered must be 'skip' or 'clamp'")
        path = Path(path)
        nav, install = _resolve_nav_and_install(
            path, nav, nav_paths, install, install_paths, auto_companion
        )
        tdepth = 0.0
        if install is not None:
            lever = install.transducer_offsets(_resolve_group(install))
            if lever:
                tdepth = float(lever[2])

        kind, items, time_fn, frame_fn = _file_ping_source(
            path, allow_incomplete=allow_incomplete
        )
        lo, hi = nav.time_span

        kept: List[PingView] = []
        stride = 1
        seen = 0
        n_uncovered = 0
        n_total = 0
        for raw in items:
            n_total += 1
            if seen % stride == 0:
                t = float(time_fn(raw))
                if not (lo - coverage_tol_s <= t <= hi + coverage_tol_s):
                    n_uncovered += 1
                    if on_uncovered == "skip":
                        seen += 1
                        continue
                    # clamp: process anyway (position_at clamps to a track end).
                pv = cls._build_ping(
                    raw, frame_fn, len(kept), t, nav, tdepth, fan_samples
                )
                kept.append(pv)
                if len(kept) > max_pings:
                    # Adaptive halving keeps memory bounded at ~max_pings while
                    # leaving an evenly-strided subset of the whole file.
                    kept = kept[::2]
                    for i, p in enumerate(kept):
                        p.index = i
                    stride *= 2
            seen += 1

        if not kept:
            raise ValueError(
                f"{path.name}: no water-column pings were covered by the nav track "
                f"{nav.position_source} (span [{lo:.0f},{hi:.0f}], {n_uncovered} "
                f"uncovered / {n_total} total)"
            )

        # Along-track distance (cumulative horizontal metres) in a local ENU frame.
        anchor = (kept[0].lon, kept[0].lat)
        en = np.array(
            [_vessel_en(p.lon, p.lat, anchor, 4326, False) for p in kept], float
        )
        step = np.r_[0.0, np.hypot(np.diff(en[:, 0]), np.diff(en[:, 1]))]
        along = np.cumsum(step)
        for p, a in zip(kept, along):
            p.along_track_m = float(a)

        # Common depth axis, then collapse every ping onto it.
        if max_depth_m is None:
            bottoms = np.array([p.max_bottom_depth() for p in kept], float)
            bottoms = bottoms[bottoms > 0]
            if bottoms.size:
                max_depth_m = float(np.percentile(bottoms, 99) * 1.10)
            else:
                max_depth_m = float(
                    max(p.sound_speed_m_s * p.sample_idx.max() / (2 * p.sample_freq_hz)
                        for p in kept)
                )
        depth_edges = np.linspace(0.0, max_depth_m, 601)
        stack = np.column_stack(
            [p.depth_column(depth_edges, stack_mode, nadir_halfangle_deg) for p in kept]
        )

        # Bounded whole-file amplitude subsample (fan-domain) for the histogram.
        cap, per = 200_000, max(1, 200_000 // len(kept))
        parts = []
        for p in kept:
            a = p.amp.astype(np.float32).ravel()
            a = a[np.isfinite(a)]
            if a.size > per:
                a = a[:: max(1, a.size // per)][:per]
            parts.append(a)
        amp_sample = np.concatenate(parts) if parts else np.zeros(0, np.float32)

        return cls(
            path=path,
            pings=kept,
            depth_edges=depth_edges,
            stack=stack,
            stack_mode=stack_mode,
            nav_position_source=nav.position_source,
            nav_heading_source=nav.heading_source,
            nav_attitude_source=nav.attitude_source,
            n_total=n_total,
            n_uncovered=n_uncovered,
            stride=stride,
            max_depth_m=max_depth_m,
            nadir_halfangle_deg=nadir_halfangle_deg,
            along_track_m=along,
            amp_sample=amp_sample,
        )

    @staticmethod
    def _build_ping(raw, frame_fn, index, t, nav: NavTrack, tdepth, fan_samples) -> PingView:
        frame = frame_fn(raw)
        lat, lon = (float(v) for v in nav.position_at(t))
        heading = float(nav.heading_at(t))
        roll, pitch, heave = nav.attitude_at(t)
        ns = frame.amp_db.shape[1]
        samp_stride = max(1, int(math.ceil(ns / max(1, fan_samples))))
        sample_idx = np.arange(ns)[::samp_stride]
        amp = frame.amp_db[:, ::samp_stride].astype(np.float16)
        return PingView(
            index=index,
            time=t,
            label=frame.label,
            lat=lat,
            lon=lon,
            heading_deg=heading,
            roll_deg=roll,
            pitch_deg=pitch,
            heave_m=heave,
            transducer_depth_m=tdepth,
            sound_speed_m_s=frame.sound_speed_m_s,
            sample_freq_hz=frame.sample_freq_hz,
            angles_deg=np.asarray(frame.angles_deg, float),
            detected_samples=np.asarray(frame.detected_samples, int),
            sample_idx=sample_idx,
            amp=amp,
        )


# ---------------------------------------------------------------------------
# Rendering (lazy matplotlib; static render is headless-safe).
# ---------------------------------------------------------------------------


def _plt(interactive: bool = False):
    import matplotlib

    if not interactive:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


class WaterColumnViewer:
    """Two linked panels over a :class:`WaterColumnFileView`.

    Build it, then either :meth:`render_static` (headless PNG of one ping) or
    :meth:`show` (interactive: click the stack or use the arrow keys to scrub the
    fan). Construction is backend-agnostic; only :meth:`show` needs a GUI backend.
    """

    def __init__(
        self,
        view: WaterColumnFileView,
        plt,
        cmap: str = "viridis",
        *,
        clim: Optional[Tuple[float, float]] = None,
        clip_mode: str = "clamp",
        across_window: Optional[Tuple[float, float]] = None,
    ):
        import matplotlib

        self.view = view
        self.plt = plt
        self.current = 0
        # Cap the fan scatter so interactive redraws (scrub / slider) stay snappy;
        # a dense wedge needs far fewer than the full decimated grid to read well.
        self._max_fan_points = 60_000
        self._clip_mode = "cut" if clip_mode == "cut" else "clamp"
        self._across_window = across_window
        # ``_cmap`` is derived from this base with under/over colours that encode
        # the clip mode (shared by the stack imshow and the fan scatter).
        self._base_cmap = matplotlib.colormaps[cmap]
        # Whole-file amplitude population for the histogram + initial clim.
        pop = (view.amp_sample if view.amp_sample.size
               else np.ma.masked_invalid(view.stack).compressed())
        self._amp_pop = np.asarray(pop, float)
        if clim is not None:
            self._clim = (float(clim[0]), float(clim[1]))
        elif self._amp_pop.size:
            self._clim = (float(np.percentile(self._amp_pop, 2)),
                          float(np.percentile(self._amp_pop, 98)))
        else:
            self._clim = (0.0, 1.0)
        # Interactive widgets (created in show()).
        self._clim_slider = self._clip_radio = self._span = None
        if across_window is not None:
            view.rebuild_stack(across_window=across_window)
        self._apply_clip_to_cmap()
        self._build_figure()

    def _apply_clip_to_cmap(self) -> None:
        """Rebuild ``_cmap`` with under/over per the clip mode (bad = transparent)."""
        transparent = (0, 0, 0, 0)
        if self._clip_mode == "cut":
            under = over = transparent
        else:  # clamp -> out-of-range shows the end colours
            under, over = self._base_cmap(0.0), self._base_cmap(1.0)
        self._cmap = self._base_cmap.with_extremes(under=under, over=over, bad=transparent)

    def _build_figure(self) -> None:
        plt = self.plt
        v = self.view
        self.fig = plt.figure(figsize=(15, 9))
        # Main panels + shared colorbar on the left; a control column is reserved
        # on the right (widgets added in show(); the histogram is drawn always).
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 0.035], height_ratios=[1.0, 1.15],
            left=0.06, right=0.72, top=0.91, bottom=0.11, hspace=0.30, wspace=0.02,
        )
        self.ax_stack = self.fig.add_subplot(gs[0, 0])
        self.ax_fan = self.fig.add_subplot(gs[1, 0])
        self.cax = self.fig.add_subplot(gs[:, 1])

        depth_lo, depth_hi = v.depth_edges[0], v.depth_edges[-1]
        vmin, vmax = self._clim
        self.im = self.ax_stack.imshow(
            np.ma.masked_invalid(v.stack),
            aspect="auto",
            origin="upper",
            extent=[-0.5, v.n_pings - 0.5, depth_hi, depth_lo],
            cmap=self._cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self.ax_stack.set(
            xlabel="ping index along track  →",
            ylabel="depth (m)",
            title=self._stack_title(),
        )
        self.fig.colorbar(self.im, cax=self.cax, label="amplitude (dB)")
        self.cursor = self.ax_stack.axvline(self.current, color="red", lw=1.2, alpha=0.9)

        self.ax_hist = self.fig.add_axes([0.78, 0.60, 0.19, 0.30])
        self._draw_histogram()
        self.sc = None
        self._draw_fan()
        self._update_suptitle()
        # Lat/lon readout goes through the (cheap) toolbar coordinate display — no
        # per-motion canvas redraw (that would lock up the heavy fan scatter).
        self.ax_fan.format_coord = lambda x, y: self._status_over_fan(x, y)
        self.ax_stack.format_coord = lambda x, y: self._status_over_stack(round(x), y)
        self.status = self.fig.text(
            0.06, 0.02,
            "drag the fan = swath band · double-click / r = reset · ←/→ scrub · "
            "hover = lat/lon (toolbar)",
            fontsize=8, color="0.35",
        )

    def _draw_histogram(self) -> None:
        self.ax_hist.clear()
        if self._amp_pop.size:
            self.ax_hist.hist(self._amp_pop, bins=80, color="0.6")
        vmin, vmax = self._clim
        self._hist_lo = self.ax_hist.axvline(vmin, color="red", lw=1.2)
        self._hist_hi = self.ax_hist.axvline(vmax, color="red", lw=1.2)
        self.ax_hist.set_title("amplitude histogram (whole file)", fontsize=9)
        self.ax_hist.set_xlabel("dB", fontsize=8)
        self.ax_hist.set_yticks([])
        self.ax_hist.tick_params(labelsize=7)

    def _stack_title(self) -> str:
        v = self.view
        km = v.along_track_m[-1] / 1000.0 if v.along_track_m.size else 0.0
        extra = f" · stride {v.stride}" if v.stride > 1 else ""
        if self._across_window is not None:
            lo, hi = self._across_window
            swath = f" · swath [{lo:.0f},{hi:.0f}] m"
        else:
            swath = ""
        return (
            f"along-track depth stack ({v.stack_mode}) — {v.n_pings} pings, "
            f"{km:.1f} km{extra}{swath}"
        )

    def _draw_fan(self) -> None:
        p = self.view.pings[self.current]
        self.ax_fan.clear()
        across, depth, amp = p.fan_points()
        if amp.size > self._max_fan_points:  # decimate for a snappy redraw
            step = int(np.ceil(amp.size / self._max_fan_points))
            across, depth, amp = across[::step], depth[::step], amp[::step]
        vmin, vmax = self._clim
        self.sc = None
        if amp.size:
            self.sc = self.ax_fan.scatter(
                across, depth, c=amp, s=1.5, cmap=self._cmap, vmin=vmin, vmax=vmax,
            )
        bx, bz = p.bottom_line()
        if bx.size:
            order = np.argsort(bx)
            self.ax_fan.plot(bx[order], bz[order], color="red", lw=1.0, label="bottom detect")
            self.ax_fan.legend(loc="lower center", fontsize=8)
        if self._across_window is not None:
            lo, hi = self._across_window
            self.ax_fan.axvspan(lo, hi, color="red", alpha=0.10)
        self.ax_fan.set_aspect("equal", "datalim")
        # Equal aspect governs the limits; just point depth downward.
        if not self.ax_fan.yaxis_inverted():
            self.ax_fan.invert_yaxis()
        self.ax_fan.set(
            xlabel="across-track (m, + port)",
            ylabel="depth (m)",
            title=self._fan_title(p),
        )

    def _fan_title(self, p: PingView) -> str:
        att = (
            f"roll {p.roll_deg:+.2f}°  pitch {p.pitch_deg:+.2f}°  heave {p.heave_m:+.2f} m"
            if self.view.nav_attitude_source
            else "no attitude (heading-only nav)"
        )
        return (
            f"ping {p.index}/{self.view.n_pings - 1} — {p.label}\n"
            f"heading {p.heading_deg:.1f}°   {att}   (beams receive-stabilized)"
        )

    def _update_suptitle(self) -> None:
        v = self.view
        att = v.nav_attitude_source or "none"
        skip = f", {v.n_uncovered} uncovered/skipped" if v.n_uncovered else ""
        self.fig.suptitle(
            f"{v.path.name}   —   nav: pos {v.nav_position_source} / hdg "
            f"{v.nav_heading_source} / att {att}   ({v.n_total} pings{skip})",
            fontsize=10,
        )

    def select(self, index: int) -> None:
        """Show ping ``index`` (clamped) in the fan panel and move the cursor."""
        index = int(np.clip(index, 0, self.view.n_pings - 1))
        if index == self.current:
            return
        self.current = index
        self.cursor.set_xdata([index, index])
        self._draw_fan()
        self.fig.canvas.draw_idle()

    # -- colour scale + clip -----------------------------------------------

    def _apply_clim(self, vmin: float, vmax: float) -> None:
        """Set the shared colour limits on both panels + the histogram guides."""
        if vmax <= vmin:
            return
        self._clim = (float(vmin), float(vmax))
        self.im.set_clim(vmin, vmax)
        if self.sc is not None:
            self.sc.set_clim(vmin, vmax)
        self._hist_lo.set_xdata([vmin, vmin])
        self._hist_hi.set_xdata([vmax, vmax])
        self.fig.canvas.draw_idle()

    def _set_clip(self, label) -> None:
        """Switch out-of-range handling: 'clamp' end colours vs 'cut' transparent."""
        self._clip_mode = "cut" if str(label).startswith("cut") else "clamp"
        self._apply_clip_to_cmap()
        self.im.set_cmap(self._cmap)
        if self.sc is not None:
            self.sc.set_cmap(self._cmap)
        self.fig.canvas.draw_idle()

    # -- swath selection ---------------------------------------------------

    def _on_swath(self, x0: float, x1: float) -> None:
        """Rebuild the stack from the dragged across-track band."""
        if abs(x1 - x0) < 1.0:  # a click, not a drag
            return
        self._across_window = (float(min(x0, x1)), float(max(x0, x1)))
        self._rebuild_and_refresh()

    def _reset_swath(self) -> None:
        if self._across_window is None:
            return
        self._across_window = None
        self._rebuild_and_refresh()

    def _rebuild_and_refresh(self) -> None:
        self.view.rebuild_stack(across_window=self._across_window)
        self.im.set_data(np.ma.masked_invalid(self.view.stack))
        self.ax_stack.set_title(self._stack_title())
        self._draw_fan()  # refresh the band overlay
        self.fig.canvas.draw_idle()

    # -- cursor lat/lon readout --------------------------------------------

    def _status_over_fan(self, across_m: float, depth_m: float) -> str:
        lon, lat = self.view.pings[self.current].across_to_lonlat(across_m)
        return (f"fan    across {across_m:+8.0f} m   depth {depth_m:6.0f} m   "
                f"lat {lat:+.5f}  lon {lon:+.5f}")

    def _status_over_stack(self, index: int, depth_m: float) -> str:
        i = int(np.clip(index, 0, self.view.n_pings - 1))
        p = self.view.pings[i]
        return (f"stack  ping {i:4d}   nadir lat {p.lat:+.5f}  lon {p.lon:+.5f}   "
                f"depth {depth_m:6.0f} m")

    def render_static(self, out: Path, index: int = 0) -> Path:
        """Headless PNG of the linked panels at ``index`` (for review/tests)."""
        self.select(index) if index != self.current else self._draw_fan()
        self.cursor.set_xdata([self.current, self.current])
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=110)
        return out

    # -- interactive event wiring ------------------------------------------

    def _on_click(self, event) -> None:
        if getattr(event, "dblclick", False):
            self._reset_swath()
            return
        if event.inaxes is self.ax_stack and event.xdata is not None:
            self.select(round(event.xdata))

    def _on_key(self, event) -> None:
        step = {"right": 1, "left": -1, "pageup": 10, "pagedown": -10}
        if event.key in step:
            self.select(self.current + step[event.key])
        elif event.key == "home":
            self.select(0)
        elif event.key == "end":
            self.select(self.view.n_pings - 1)
        elif event.key == "r":
            self._reset_swath()

    def _build_controls(self) -> None:
        """Create the interactive widgets (needs an interactive backend)."""
        from matplotlib.widgets import RadioButtons, RangeSlider, SpanSelector

        if self._amp_pop.size:
            amin, amax = float(np.min(self._amp_pop)), float(np.max(self._amp_pop))
        else:
            amin, amax = 0.0, 1.0
        if amax <= amin:
            amax = amin + 1.0
        vinit = (max(amin, self._clim[0]), min(amax, self._clim[1]))
        self._ax_slider = self.fig.add_axes([0.78, 0.54, 0.19, 0.03])
        self._clim_slider = RangeSlider(self._ax_slider, "clim", amin, amax, valinit=vinit)
        self._clim_slider.on_changed(lambda val: self._apply_clim(val[0], val[1]))

        self._ax_radio = self.fig.add_axes([0.78, 0.34, 0.19, 0.14])
        labels = ("clamp (end colours)", "cut (transparent)")
        self._clip_radio = RadioButtons(
            self._ax_radio, labels, active=1 if self._clip_mode == "cut" else 0
        )
        self._clip_radio.on_clicked(self._set_clip)

        # Drag horizontally across the fan to pick the swath band for the stack.
        self._span = SpanSelector(
            self.ax_fan, self._on_swath, "horizontal", useblit=True,
            props=dict(alpha=0.2, facecolor="red"),
        )

    def _raise_window(self) -> None:
        """Best-effort: bring the window on-screen and to the front.

        WSLg (and some window managers) can open a new window off-screen or behind
        others — the taskbar shows an icon but nothing appears. Nudge it to a
        visible position and briefly mark it top-most. Backend-agnostic and fully
        guarded so it is a no-op on headless / unknown backends.
        """
        try:
            win = self.fig.canvas.manager.window
        except Exception:
            return
        try:
            if hasattr(win, "wm_attributes"):  # Tk
                win.deiconify()
                win.geometry("+60+60")  # force on-screen (position only, keeps size)
                win.lift()
                win.wm_attributes("-topmost", True)
                win.after(600, lambda: win.wm_attributes("-topmost", False))
            elif hasattr(win, "raise_"):  # Qt
                win.showNormal(); win.raise_(); win.activateWindow()
        except Exception:
            pass

    def show(self) -> None:
        """Launch the interactive window (needs a GUI backend)."""
        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._raise_window()
        self.plt.show()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="mbes-wc-viewer",
        description="Interactive water-column viewer: whole-file along-track depth "
        "stack linked to a nav/attitude-corrected wedge fan per ping.",
    )
    ap.add_argument("path", type=Path, help="water-column file (.kmwcd/.kmall/.wcd/.all)")
    ap.add_argument("--nav", type=Path, action="append", metavar="FILE",
                    help="explicit nav file(s) (.kmall/.all); repeatable")
    ap.add_argument("--install", type=Path, action="append", metavar="FILE",
                    help="explicit installation-parameter file(s); repeatable")
    ap.add_argument("--no-auto-companion", action="store_true",
                    help="do not auto-discover a same-stem .kmall/.all companion")
    ap.add_argument("--max-pings", type=int, default=500,
                    help="cap displayed pings (adaptive stride above this; default 500)")
    ap.add_argument("--fan-samples", type=int, default=600,
                    help="range samples kept per ping for the fan (default 600)")
    ap.add_argument("--max-depth-m", type=float, default=None,
                    help="depth-axis limit (default: from detected bottoms)")
    ap.add_argument("--stack", choices=["swath-max", "swath-mean", "nadir"],
                    default="swath-max", help="how each ping is collapsed to a column")
    ap.add_argument("--nadir-halfangle-deg", type=float, default=3.0,
                    help="half-angle for --stack nadir (default 3°)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="keep truncated tail pings when reassembling .wcd/.all")
    ap.add_argument("--on-uncovered", choices=["skip", "clamp"], default="skip",
                    help="pings outside the nav time span: skip (default) or clamp")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--clim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="initial colour-scale min/max dB (default: 2nd–98th percentile)")
    ap.add_argument("--clip", choices=["clamp", "cut"], default="clamp",
                    help="out-of-range samples: clamp to end colours (default) or cut (transparent)")
    ap.add_argument("--swath", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="initial across-track band (m, +port) feeding the stack")
    ap.add_argument("--save", type=int, metavar="PING", default=None,
                    help="headless: render this ping's linked panels to --out and exit")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output PNG for --save (default alongside the input)")
    args = ap.parse_args(argv)

    view = WaterColumnFileView.from_file(
        args.path,
        nav_paths=args.nav,
        install_paths=args.install,
        auto_companion=not args.no_auto_companion,
        max_pings=args.max_pings,
        fan_samples=args.fan_samples,
        max_depth_m=args.max_depth_m,
        stack_mode=args.stack,
        nadir_halfangle_deg=args.nadir_halfangle_deg,
        allow_incomplete=args.allow_incomplete,
        on_uncovered=args.on_uncovered,
    )
    print(
        f"{args.path.name}: {view.n_pings} pings "
        f"(of {view.n_total}, {view.n_uncovered} uncovered, stride {view.stride}); "
        f"nav pos={view.nav_position_source} hdg={view.nav_heading_source} "
        f"att={view.nav_attitude_source or 'none'}; "
        f"depth 0–{view.max_depth_m:.0f} m; stack={view.stack_mode}"
    )
    if view.n_uncovered:
        warnings.warn(
            f"{args.path.name}: {view.n_uncovered} ping(s) outside the nav time "
            f"span were skipped — check the nav file matches this water-column file."
        )

    vkw = dict(
        cmap=args.cmap,
        clim=tuple(args.clim) if args.clim else None,
        clip_mode=args.clip,
        across_window=tuple(args.swath) if args.swath else None,
    )
    if args.save is not None:
        plt = _plt(interactive=False)
        viewer = WaterColumnViewer(view, plt, **vkw)
        out = args.out or args.path.with_name(f"wc_viewer_{args.path.stem}_ping{args.save}.png")
        p = viewer.render_static(out, args.save)
        print(f"wrote {p}")
        return

    plt = _plt(interactive=True)
    viewer = WaterColumnViewer(view, plt, **vkw)
    viewer.show()


if __name__ == "__main__":  # pragma: no cover
    main()
