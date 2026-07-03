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

    def depth_column(
        self, depth_edges: np.ndarray, mode: str, nadir_halfangle_deg: float
    ) -> np.ndarray:
        """Collapse the fan to a 1-D amplitude-vs-depth column on ``depth_edges``.

        ``mode``: ``"swath-max"`` peak-holds every beam per depth bin (any
        midwater target in the swath shows), ``"swath-mean"`` averages in the
        linear-intensity domain, ``"nadir"`` uses only beams within
        ``nadir_halfangle_deg`` of vertical (a near-vertical section).
        """
        r = self._range()
        th = np.deg2rad(self.angles_deg)
        depth = np.cos(th)[:, None] * r[None, :] + self.heave_m + self.transducer_depth_m
        amp = self.amp.astype(np.float32)
        if mode == "nadir":
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
    along_track_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_pings(self) -> int:
        return len(self.pings)

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
            along_track_m=along,
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

    def __init__(self, view: WaterColumnFileView, plt, cmap: str = "viridis"):
        self.view = view
        self.plt = plt
        self.cmap = cmap
        self.current = 0
        self._build_figure()

    def _build_figure(self) -> None:
        plt = self.plt
        v = self.view
        self.fig = plt.figure(figsize=(14, 9), layout="constrained")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
        self.ax_stack = self.fig.add_subplot(gs[0])
        self.ax_fan = self.fig.add_subplot(gs[1])

        depth_lo, depth_hi = v.depth_edges[0], v.depth_edges[-1]
        stack = np.ma.masked_invalid(v.stack)
        finite = stack.compressed()
        vlo, vhi = (
            (float(np.percentile(finite, 2)), float(np.percentile(finite, 99)))
            if finite.size
            else (0.0, 1.0)
        )
        self.im = self.ax_stack.imshow(
            stack,
            aspect="auto",
            origin="upper",
            extent=[-0.5, v.n_pings - 0.5, depth_hi, depth_lo],
            cmap=self.cmap,
            vmin=vlo,
            vmax=vhi,
            interpolation="nearest",
        )
        self.ax_stack.set(
            xlabel="ping index along track  →",
            ylabel="depth (m)",
            title=self._stack_title(),
        )
        self.plt.colorbar(self.im, ax=self.ax_stack, label="amplitude (dB)", shrink=0.85)
        self.cursor = self.ax_stack.axvline(self.current, color="red", lw=1.2, alpha=0.9)

        self._fan_vlo, self._fan_vhi = vlo, vhi
        self._draw_fan()
        self._update_suptitle()

    def _stack_title(self) -> str:
        v = self.view
        km = v.along_track_m[-1] / 1000.0 if v.along_track_m.size else 0.0
        extra = f" · stride {v.stride}" if v.stride > 1 else ""
        return (
            f"along-track depth stack ({v.stack_mode}) — {v.n_pings} pings, "
            f"{km:.1f} km{extra}"
        )

    def _draw_fan(self) -> None:
        p = self.view.pings[self.current]
        self.ax_fan.clear()
        across, depth, amp = p.fan_points()
        if amp.size:
            self.sc = self.ax_fan.scatter(
                across, depth, c=amp, s=1.5, cmap=self.cmap,
                vmin=self._fan_vlo, vmax=self._fan_vhi,
            )
        bx, bz = p.bottom_line()
        if bx.size:
            order = np.argsort(bx)
            self.ax_fan.plot(bx[order], bz[order], color="red", lw=1.0, label="bottom detect")
            self.ax_fan.legend(loc="lower center", fontsize=8)
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

    def show(self) -> None:
        """Launch the interactive window (needs a GUI backend)."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
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

    if args.save is not None:
        plt = _plt(interactive=False)
        viewer = WaterColumnViewer(view, plt, cmap=args.cmap)
        out = args.out or args.path.with_name(f"wc_viewer_{args.path.stem}_ping{args.save}.png")
        p = viewer.render_static(out, args.save)
        print(f"wrote {p}")
        return

    plt = _plt(interactive=True)
    viewer = WaterColumnViewer(view, plt, cmap=args.cmap)
    viewer.show()


if __name__ == "__main__":  # pragma: no cover
    main()
