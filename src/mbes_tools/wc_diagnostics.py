"""Visual diagnostics for the water-column readers — a reviewable, repeatable suite.

Renders a spread of PNG plots from **real** Kongsberg water-column files so a
reviewer can eyeball that the ``#MWC`` (`.kmwcd`) and ``k`` (`.wcd`) parsers
decode real geometry, amplitude, and phase correctly:

* **echogram** — beam x sample amplitude (dB) with the per-beam detected-bottom
  overlay; the bright seafloor return should sit on the detection line.
* **geo-referenced wedge** — beam angle + range -> across-track/depth; the
  bottom should form a coherent across-track profile (proves angle sign +
  ``c·sample/2·fs`` range scaling).
* **nadir profile** — one beam's amplitude vs range, spiking at the detected
  range.
* **bottom-detect vs amplitude-peak** — should fall on the 1:1 line (the
  amplitude array and the detection field are mutually aligned).
* **phase panels** — for ``phase_flag`` 1 (int8, 180/128 deg) and 2 (int16,
  0.01 deg): phase echogram + histogram, which should span the full +/-180 deg.
* **sector centre frequencies** — a cross-file sanity bar (EM122/124 ~12 kHz,
  EM2040 ~200-400 kHz).

matplotlib is imported lazily (the project's ``gui`` extra), so importing this
module and its pure helpers (:func:`phase_scale_deg`, :func:`padded_grid`,
:func:`wedge_coords`, :func:`peak_sample_indices`, :func:`range_axis`) works in
the base environment; only the plotting functions need matplotlib.

CLI::

    mbes-wc-diagnostics -o plots/ \
        --mwc tests/fixtures/sample_tn447_em124.kmwcd \
              tests/fixtures/sample_em2040_wc_phase1.kmall \
        --wcd tests/fixtures/sample_atlantis_em122.wcd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools.kmwcd import iter_mwc_datagrams
from mbes_tools.wcd import iter_water_column_datagrams


# ---------------------------------------------------------------------------
# Numeric helpers (no matplotlib; unit-tested).
# ---------------------------------------------------------------------------


def phase_scale_deg(phase_flag: int) -> float:
    """Degrees per stored phase unit for a #MWC ``phase_flag``.

    ``0`` -> 0.0 (no phase), ``1`` -> 180/128 (int8, low resolution),
    ``2`` -> 0.01 (int16, high resolution). Both non-zero scales map the
    stored integer range onto +/-180 deg.
    """
    if phase_flag == 0:
        return 0.0
    if phase_flag == 1:
        return 180.0 / 128.0
    if phase_flag == 2:
        return 0.01
    raise ValueError(f"unknown phase_flag {phase_flag!r}")


def padded_grid(
    starts: Sequence[int],
    nsamps: Sequence[int],
    value_lists: Sequence[Sequence[int]],
    scale: float = 1.0,
) -> np.ndarray:
    """Build a ``[beam, width]`` array of per-beam sample values, NaN-padded.

    Each beam's ``value_lists[i]`` is placed at absolute sample positions
    ``starts[i] : starts[i] + nsamps[i]`` and multiplied by ``scale``;
    positions outside a beam's samples are NaN. ``width`` is
    ``max(start + nsamp)``.
    """
    nb = len(value_lists)
    width = int(max((s + n for s, n in zip(starts, nsamps)), default=0))
    grid = np.full((nb, width), np.nan)
    for i, vals in enumerate(value_lists):
        s, n = int(starts[i]), int(nsamps[i])
        if n:
            grid[i, s:s + n] = np.asarray(vals, dtype=float) * scale
    return grid


def range_axis(width: int, sound_speed_m_s: float, sample_freq_hz: float) -> np.ndarray:
    """One-way slant range (m) for absolute sample numbers ``0..width-1``."""
    k = np.arange(width)
    return sound_speed_m_s * k / (2.0 * sample_freq_hz)


def wedge_coords(
    angles_deg: np.ndarray, width: int, sound_speed_m_s: float, sample_freq_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """across-track ``X[beam, k]`` and depth ``Z[beam, k]`` for each sample.

    Uses ``range = c·k / (2·fs)`` and the per-beam pointing angle
    (positive = port): ``X = range·sin(angle)``, ``Z = range·cos(angle)``.
    """
    r = range_axis(width, sound_speed_m_s, sample_freq_hz)
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    X = np.sin(th)[:, None] * r[None, :]
    Z = np.cos(th)[:, None] * r[None, :]
    return X, Z


def peak_sample_indices(amp_grid: np.ndarray) -> np.ndarray:
    """Per-beam absolute sample index of maximum amplitude (NaN-safe).

    Beams with no finite samples return 0.
    """
    out = np.zeros(amp_grid.shape[0], dtype=int)
    for i, row in enumerate(amp_grid):
        if np.isfinite(row).any():
            out[i] = int(np.nanargmax(row))
    return out


# ---------------------------------------------------------------------------
# Frame: a reader-agnostic bundle of the arrays a panel needs.
# ---------------------------------------------------------------------------


@dataclass
class WCFrame:
    """Decoded water-column ping arrays, ready to plot.

    ``amp_db`` and ``phase_deg`` are ``[beam, width]`` NaN-padded grids.
    """

    amp_db: np.ndarray
    phase_deg: Optional[np.ndarray]
    angles_deg: np.ndarray
    detected_samples: np.ndarray
    sound_speed_m_s: float
    sample_freq_hz: float
    phase_flag: int
    sector_freqs_hz: List[float]
    label: str

    @property
    def width(self) -> int:
        return self.amp_db.shape[1]


def frame_from_mwc(dgm, label: Optional[str] = None) -> WCFrame:
    """Build a :class:`WCFrame` from a parsed ``#MWC`` datagram."""
    starts = [b.start_range_sample_num for b in dgm.beams]
    nsamps = [b.num_sample_data for b in dgm.beams]
    amp = padded_grid(starts, nsamps, [b.sample_amplitudes_0p5_db for b in dgm.beams], 0.5)
    phase = None
    if dgm.phase_flag in (1, 2):
        phase = padded_grid(
            starts, nsamps, [b.phase_samples for b in dgm.beams], phase_scale_deg(dgm.phase_flag)
        )
    return WCFrame(
        amp_db=amp,
        phase_deg=phase,
        angles_deg=np.array([b.beam_point_angle_re_vertical_deg for b in dgm.beams], float),
        detected_samples=np.array([b.detected_range_samples for b in dgm.beams], int),
        sound_speed_m_s=float(dgm.sound_velocity_m_s),
        sample_freq_hz=float(dgm.sample_freq_hz),
        phase_flag=int(dgm.phase_flag),
        sector_freqs_hz=[s.centre_freq_hz for s in dgm.tx_sectors],
        label=label or f"EM{dgm.echo_sounder_id} #MWC ping {dgm.ping_cnt} (phaseFlag {dgm.phase_flag})",
    )


def frame_from_wcd(dgm, label: Optional[str] = None) -> WCFrame:
    """Build a :class:`WCFrame` from a parsed ``k`` Water Column datagram."""
    starts = [b.start_range_sample for b in dgm.beams]
    nsamps = [b.number_of_samples for b in dgm.beams]
    amp = padded_grid(starts, nsamps, [b.samples for b in dgm.beams], 0.5)
    return WCFrame(
        amp_db=amp,
        phase_deg=None,
        angles_deg=np.array([b.pointing_angle_deg for b in dgm.beams], float),
        detected_samples=np.array([b.detected_range_samples for b in dgm.beams], int),
        sound_speed_m_s=float(dgm.sound_speed_m_s),
        sample_freq_hz=float(dgm.sample_frequency_hz),
        phase_flag=0,
        sector_freqs_hz=[s.centre_frequency_hz for s in dgm.tx_sectors],
        label=label or f"EM .wcd k ping {dgm.counter} ({dgm.num_tx_sectors} sectors)",
    )


def first_mwc_frame(path) -> WCFrame:
    """Frame for the first ``#MWC`` datagram of a .kmwcd/.kmall file."""
    return frame_from_mwc(next(iter_mwc_datagrams(Path(path))))


def first_wcd_frame(path) -> WCFrame:
    """Frame for the first ``k`` datagram of a .wcd file."""
    return frame_from_wcd(next(iter_water_column_datagrams(Path(path))))


# ---------------------------------------------------------------------------
# Plot panels (lazy matplotlib). Each returns the saved path.
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _stride_for(width: int, target: int = 700) -> int:
    return max(1, width // target)


def panel_echogram_wedge(frame: WCFrame, out: Path, stem: str) -> Path:
    """2x2: amplitude echogram (+detected overlay), geo wedge, nadir profile,
    bottom-detect vs amplitude-peak alignment."""
    plt = _plt()
    amp, det, ang = frame.amp_db, frame.detected_samples, frame.angles_deg
    c, fs = frame.sound_speed_m_s, frame.sample_freq_hz
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    im = ax[0, 0].imshow(amp.T, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest")
    ax[0, 0].plot(np.arange(amp.shape[0]), det, color="red", lw=0.8, label="detected bottom (sample)")
    ax[0, 0].set(xlabel="beam #", ylabel="sample # (range →)", title=f"{frame.label} — amplitude echogram")
    ax[0, 0].legend(loc="upper right", fontsize=7)
    plt.colorbar(im, ax=ax[0, 0], label="amplitude (dB)", shrink=0.8)

    # geo-referenced wedge (column-decimated for a snappy scatter).
    st = _stride_for(frame.width)
    X, Z = wedge_coords(ang, frame.width, c, fs)
    Xs, Zs, As = X[:, ::st], Z[:, ::st], amp[:, ::st]
    m = np.isfinite(As)
    pc = ax[0, 1].scatter(Xs[m], Zs[m], c=As[m], s=0.5, cmap="viridis")
    R = c * det / (2.0 * fs)
    th = np.deg2rad(ang)
    good = det > 0
    ax[0, 1].plot(np.sin(th[good]) * R[good], np.cos(th[good]) * R[good], color="red", lw=1.0, label="bottom detect")
    ax[0, 1].invert_yaxis(); ax[0, 1].set_aspect("equal", "datalim")
    ax[0, 1].set(xlabel="across-track (m, +port)", ylabel="depth (m)", title="geo-referenced swath wedge")
    ax[0, 1].legend(loc="lower center", fontsize=7)
    plt.colorbar(pc, ax=ax[0, 1], label="amplitude (dB)", shrink=0.8)

    # nadir beam amplitude profile from the grid row.
    nadir = int(np.argmin(np.abs(ang)))
    r = range_axis(frame.width, c, fs)
    row = amp[nadir]
    fin = np.isfinite(row)
    ax[1, 0].plot(r[fin], row[fin], lw=0.7)
    dr_m = c * det[nadir] / (2.0 * fs)
    if det[nadir] > 0:
        ax[1, 0].axvline(dr_m, color="red", ls="--", label=f"detected range {dr_m:.0f} m")
        ax[1, 0].legend(fontsize=8)
    ax[1, 0].set(xlabel="one-way range (m)", ylabel="amplitude (dB)",
                 title=f"nadir beam #{nadir} (angle {ang[nadir]:.2f}°) profile")

    # bottom-detect vs amplitude-peak alignment.
    peak = peak_sample_indices(amp)
    g = det > 0
    ax[1, 1].scatter(det[g], peak[g], s=5, alpha=0.4)
    hi = int(max(det.max(initial=0), peak.max(initial=0)))
    ax[1, 1].plot([0, hi], [0, hi], "r--", lw=1, label="1:1")
    ax[1, 1].set(xlabel="detected_range (sample)", ylabel="amplitude-peak (sample)",
                 title="bottom-detect vs amplitude-peak alignment")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); p = out / f"wc_echogram_{stem}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_phase(frame: WCFrame, out: Path, stem: str) -> Path:
    """1x3: amplitude echogram, phase echogram (deg), phase histogram (+/-180)."""
    if frame.phase_deg is None:
        raise ValueError("panel_phase requires a frame with phase data (phase_flag 1 or 2)")
    plt = _plt()
    amp, phase = frame.amp_db, frame.phase_deg
    res = "180/128°·unit (int8)" if frame.phase_flag == 1 else "0.01°·unit (int16)"
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    im0 = ax[0].imshow(amp.T, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest")
    ax[0].plot(np.arange(amp.shape[0]), frame.detected_samples, color="red", lw=0.8)
    ax[0].set(xlabel="beam #", ylabel="sample # (range →)", title=f"{frame.label} — amplitude")
    plt.colorbar(im0, ax=ax[0], label="amplitude (dB)", shrink=0.8)

    im1 = ax[1].imshow(phase.T, aspect="auto", origin="upper", cmap="twilight",
                       vmin=-180, vmax=180, interpolation="nearest")
    ax[1].set(xlabel="beam #", ylabel="sample # (range →)", title=f"phase echogram ({res})")
    plt.colorbar(im1, ax=ax[1], label="phase (deg)", shrink=0.8)

    allph = phase[np.isfinite(phase)]
    ax[2].hist(allph, bins=150, color="slateblue")
    ax[2].set(xlabel="phase (deg)", ylabel="count",
              title=f"phase distribution  span [{allph.min():.0f}, {allph.max():.0f}]°")

    fig.tight_layout(); p = out / f"wc_phase_{stem}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_sector_frequencies(specs: List[Tuple[str, Sequence[float]]], out: Path) -> Path:
    """Cross-file sector centre-frequency sanity (log axis; ~12 kHz band shaded)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(specs)), 5))
    for i, (name, fl) in enumerate(specs):
        khz = np.array(fl, float) / 1000.0
        ax.scatter([i] * len(khz), khz, s=40, label=f"{name} ({len(khz)} sectors)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(specs))); ax.set_xticklabels([n for n, _ in specs], rotation=15, ha="right")
    ax.axhspan(11, 14, color="gray", alpha=0.15)
    ax.set(ylabel="sector centre frequency (kHz, log)",
           title="Sector centre frequencies — low-freq deep ~12 kHz vs EM2040 high-freq")
    ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "wc_sector_frequencies.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate(output_dir, mwc_files: Sequence = (), wcd_files: Sequence = ()) -> List[Path]:
    """Run the panels for each supplied water-column file; return saved PNGs.

    For each ``#MWC`` file: an echogram/wedge panel, plus a phase panel when the
    ping carries phase (``phase_flag`` 1 or 2). For each ``.wcd`` file: an
    echogram/wedge panel. A final cross-file sector-frequency panel is added
    when any file was read. A failing panel is reported but does not abort.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    freq_specs: List[Tuple[str, Sequence[float]]] = []

    def run(fn, *a, **k):
        try:
            p = fn(*a, **k)
            if p:
                made.append(p); print("OK  ", p.name)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not abort the rest
            print("FAIL", getattr(fn, "__name__", fn), "->", type(exc).__name__, str(exc)[:120])

    for f in mwc_files:
        stem = Path(f).stem
        try:
            frame = first_mwc_frame(f)
        except Exception as exc:  # noqa: BLE001
            print("FAIL read", f, "->", type(exc).__name__, str(exc)[:120]); continue
        run(panel_echogram_wedge, frame, out, stem)
        if frame.phase_deg is not None:
            run(panel_phase, frame, out, stem)
        freq_specs.append((stem, frame.sector_freqs_hz))

    for f in wcd_files:
        stem = Path(f).stem
        try:
            frame = first_wcd_frame(f)
        except Exception as exc:  # noqa: BLE001
            print("FAIL read", f, "->", type(exc).__name__, str(exc)[:120]); continue
        run(panel_echogram_wedge, frame, out, stem)
        freq_specs.append((stem, frame.sector_freqs_hz))

    if freq_specs:
        run(panel_sector_frequencies, freq_specs, out)
    return made


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate water-column (.kmwcd #MWC / .wcd k) review diagnostic plots from real files."
    )
    ap.add_argument("-o", "--output", default="mbes_wc_diagnostics", help="Output directory for PNGs.")
    ap.add_argument("--mwc", nargs="+", default=[],
                    help="One or more #MWC-bearing files (.kmwcd or .kmall). Phase panels are added "
                         "automatically when a ping carries phaseFlag 1/2.")
    ap.add_argument("--wcd", nargs="+", default=[], help="One or more .wcd files.")
    args = ap.parse_args(argv)
    if not args.mwc and not args.wcd:
        ap.error("supply at least one --mwc or --wcd file")
    made = generate(args.output, mwc_files=args.mwc, wcd_files=args.wcd)
    print(f"\nWrote {len(made)} panel(s) to {args.output}")


if __name__ == "__main__":
    main()
