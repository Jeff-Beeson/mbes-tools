"""Visual diagnostics for the backscatter pipeline — a reviewable, repeatable suite.

Generates a spread of PNG plots from **real** Kongsberg files so a reviewer can
eyeball that parsing, the angular response (ARC), the configurable backscatter
source/reducer, the depth-mode keying, the projection, and the robustness path
all behave correctly. It is path-agnostic: pass the files/dirs to plot; panels
whose inputs are not supplied are skipped.

matplotlib is imported lazily (it is the project's ``gui`` extra), so importing
this module and its pure helpers works in the base environment; only the
plotting functions need matplotlib. The numeric helpers
(:func:`all_arc_rows`, :func:`kmall_arc_rows`, :func:`fold_port_starboard`,
:func:`port_starboard_symmetry`) are dependency-light and unit-tested.

CLI::

    mbes-diagnostics --output plots/ \
        --all EQUINOX.all --all-multisector BRASILIS.all \
        --kmall TN447.kmall --kmall-survey-dir MONTEREY/ \
        --manifest docs/verification_manifest.json --pyall-dir /path/to/pyall
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mbes_tools import beam_stat as bs
from mbes_tools.backscatter import normalize
from mbes_tools.backscatter.all_table import accumulate_all_file
from mbes_tools.backscatter.table import Agg, accumulate_file, build_rows
from mbes_tools.depth_modes import kmall_depth_mode_label
from mbes_tools.projection import utm_epsg_from_lonlat

# Default "no QC / no grids / no flat-filter" knobs for ARC building.
_QC = dict(
    min_ping_valid_fraction=None, min_ping_valid_soundings=None,
    min_ping_median_intensity_db=None, max_ping_intensity_std_db=None,
    max_port_starboard_diff_db=None, min_port_starboard_soundings=25,
    min_ping_angle_coverage_deg=None,
)
_FLAT = dict(
    flat_filter=False, flat_radius_m=50.0, flat_min_neighbors=50,
    flat_max_slope_deg=3.0, flat_max_roughness_m=1.0, flat_max_bs_std_db=None,
)
_GRID = dict(
    geometry_filter=None, spatial_projector=None, slope_sampler=None,
    bathy_sd_sampler=None, slope_max_deg=None, bathy_sd_max_m=None,
)


# ---------------------------------------------------------------------------
# Numeric helpers (no matplotlib; unit-tested).
# ---------------------------------------------------------------------------


def all_arc_rows(
    path, bs_source: str = "reflectivity", beam_stats: Optional[Sequence[str]] = None,
    si_window: Optional[int] = None, min_soundings: int = 25,
) -> List[dict]:
    """Build a sector/angle ARC table from one .all file (see backscatter.table)."""
    agg: Dict = defaultdict(Agg); raw: Dict = defaultdict(set)
    pg: Dict = defaultdict(int); pf: Dict = defaultdict(int)
    extra = list(beam_stats) if (bs_source == "seabed_image" and beam_stats) else []
    ea = {s: defaultdict(Agg) for s in extra}
    accumulate_all_file(
        Path(path), agg, raw, pg, pf, reflectivity_source="xyz88", angle_bin_size=1.0,
        valid_only=True, min_depth=None, max_depth=None, ping_filter_stats=defaultdict(int),
        bs_source=bs_source, beam_stats=list(beam_stats) if beam_stats else None,
        si_window=si_window, extra_agg=ea, extra_stats=extra, on_error="skip",
        error_log=[], **_GRID, **_FLAT, **_QC,
    )
    return build_rows(
        agg, raw, pg, pf, min_soundings=min_soundings, reference_group="mode_fan",
        reference_stat="median", max_abs_correction=None, flat_filter_used=False,
        extra_agg=ea, extra_stats=extra,
    )


def kmall_arc_rows(paths, min_soundings: int = 40) -> List[dict]:
    """Build an ARC table from one or more .kmall files (reflectivity2_dB)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    agg: Dict = defaultdict(Agg); raw: Dict = defaultdict(set)
    pg: Dict = defaultdict(int); pf: Dict = defaultdict(int)
    for p in paths:
        accumulate_file(
            kmall_file=Path(p), agg=agg, raw_depth_modes=raw, pre_geometry_counts=pg,
            pre_flat_counts=pf, reflectivity_field="reflectivity2_dB", angle_bin_size=1.0,
            normalize_manual_depth_modes=True, valid_only=True, min_depth=None, max_depth=None,
            ping_filter_stats=defaultdict(int), on_error="skip", error_log=[], **_GRID, **_FLAT, **_QC,
        )
    return build_rows(
        agg, raw, pg, pf, min_soundings=min_soundings, reference_group="mode_fan",
        reference_stat="median", max_abs_correction=None, flat_filter_used=False,
    )


def arc_xy(rows: List[dict], col: str = "avgIntensity_dB"):
    """Return (angles, values, sectors) arrays sorted by beam angle."""
    rows = sorted(rows, key=lambda r: r["beam_angle_deg"])
    return (
        np.array([r["beam_angle_deg"] for r in rows], float),
        np.array([r[col] for r in rows], float),
        np.array([r["sector"] for r in rows], int),
    )


def mode_mean_curve(rows: List[dict], depth_mode: int):
    """Mean ARC for one depth mode, averaged across sectors per angle (clean line)."""
    by = defaultdict(list)
    for r in rows:
        if r["depthMode"] == depth_mode:
            by[r["beam_angle_deg"]].append(r["avgIntensity_dB"])
    ang = np.array(sorted(by))
    return ang, np.array([np.mean(by[a]) for a in ang])


def fold_port_starboard(rows: List[dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Mean BS vs |angle| for port (angle>0) and starboard (angle<0) halves."""
    by = {"port": defaultdict(list), "stbd": defaultdict(list)}
    for r in rows:
        a = r["beam_angle_deg"]
        if a == 0:
            continue
        by["port" if a > 0 else "stbd"][abs(a)].append(r["avgIntensity_dB"])
    out = {}
    for side in ("port", "stbd"):
        ang = np.array(sorted(by[side]))
        out[side] = (ang, np.array([np.mean(by[side][a]) for a in ang]))
    return out


def port_starboard_symmetry(rows: List[dict]) -> float:
    """Mean |port − starboard| backscatter over overlapping |angle| bins (dB).

    A balanced system is near 0; a large value flags a port/starboard imbalance.
    Returns NaN if the halves do not overlap.
    """
    g = fold_port_starboard(rows)
    pmap = dict(zip(*g["port"])); smap = dict(zip(*g["stbd"]))
    common = sorted(set(pmap) & set(smap))
    if not common:
        return float("nan")
    return float(np.mean([abs(pmap[a] - smap[a]) for a in common]))


# ---------------------------------------------------------------------------
# Plot panels (lazy matplotlib). Each returns the saved path or None.
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel_arc(path, out: Path, fmt: str, title: str, stem: str = "arc") -> Path:
    plt = _plt()
    rows = all_arc_rows(path) if fmt == "all" else kmall_arc_rows(path)
    a, v, s = arc_xy(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sc = ax[0].scatter(a, v, c=s, cmap="tab10", s=14)
    if fmt == "all":
        inter = normalize.fit_lambert_intercept(a, v)
        if inter is not None:
            ax[0].plot(a, normalize.lambert_reference_curve(a, inter), "k--", lw=1.3, label="Lambertian fit")
            ax[0].legend()
    ax[0].set(title=f"{title} — raw ARC", xlabel="beam angle (deg, +=port)", ylabel="avg backscatter (dB)")
    plt.colorbar(sc, ax=ax[0], label="tx sector")
    shifts = normalize.solve_sector_shifts(a, v, s)
    vbal = np.array([vi + shifts.get(int(si), 0.0) for vi, si in zip(v, s)])
    ax[1].scatter(a, v, s=8, c="0.7", label="raw")
    ax[1].scatter(a, vbal, s=10, c="C0", label="sector-balanced")
    ax[1].set(title="after Lambertian sector balancing", xlabel="beam angle (deg)", ylabel="dB"); ax[1].legend()
    fig.tight_layout(); p = out / f"{stem}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_source_a_vs_b(path, out: Path) -> Path:
    plt = _plt()
    ra = all_arc_rows(path); rb = all_arc_rows(path, bs_source="seabed_image", beam_stats=["mean"])
    A = {(r["sector"], r["beam_angle_deg"]): r["avgIntensity_dB"] for r in ra}
    B = {(r["sector"], r["beam_angle_deg"]): r["avgIntensity_dB"] for r in rb}
    common = sorted(set(A) & set(B), key=lambda k: k[1])
    aa = np.array([k[1] for k in common]); av = np.array([A[k] for k in common]); bv = np.array([B[k] for k in common])
    corr = np.corrcoef(av, bv)[0, 1] if len(av) > 1 else float("nan")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(aa, av, "C0.-", ms=4, label="Source A (reflectivity)")
    ax[0].plot(aa, bv, "C1.-", ms=4, label="Source B (seabed-image mean)")
    ax[0].set(title="Source A vs Source B angular response", xlabel="beam angle (deg)", ylabel="dB"); ax[0].legend()
    ax[1].scatter(av, bv, s=10, c="C2"); lim = [min(av.min(), bv.min()), max(av.max(), bv.max())]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set(title=f"A vs B per bin (corr={corr:.2f})", xlabel="Source A (dB)", ylabel="Source B (dB)")
    fig.tight_layout(); p = out / "source_a_vs_b.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_multistat(path, out: Path) -> Path:
    plt = _plt()
    rows = sorted(all_arc_rows(path, bs_source="seabed_image", beam_stats=["mean", "median", "p90", "std"]),
                  key=lambda r: r["beam_angle_deg"])
    a = np.array([r["beam_angle_deg"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5)); ax2 = ax.twinx()
    for col, c, lab in [("avgIntensity_mean_dB", "C0", "mean"), ("avgIntensity_median_dB", "C1", "median"), ("avgIntensity_p90_dB", "C2", "p90")]:
        ax.plot(a, [r[col] for r in rows], ".-", ms=3, color=c, label=lab)
    ax2.plot(a, [r["avgIntensity_std_dB"] for r in rows], "C3.", ms=3, alpha=0.6, label="std (texture)")
    ax.set(title="Multi-stat (one read pass) + within-beam std texture", xlabel="beam angle (deg)", ylabel="backscatter (dB)")
    ax2.set_ylabel("within-beam std (dB)", color="C3"); ax.legend(loc="lower center")
    fig.tight_layout(); p = out / "multistat_texture.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_si_window(path, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    for win, c, lab in [(None, "C0", "full beam"), (10, "C1", "+/-10 samples"), (3, "C3", "+/-3 samples")]:
        a, v, _ = arc_xy(all_arc_rows(path, bs_source="seabed_image", beam_stats=["mean"], si_window=win))
        ax.plot(a, v, ".-", ms=4, color=c, label=f"si-window {lab}")
    ax.set(title="--si-window: Source B mean over full beam vs windows around centre sample",
           xlabel="beam angle (deg)", ylabel="avg backscatter (dB)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = out / "si_window.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_depthmode_split(paths, out: Path, label: str) -> Path:
    plt = _plt()
    rows = kmall_arc_rows(paths, min_soundings=40)
    modes = sorted(set(r["depthMode"] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in modes:
        ang, val = mode_mean_curve(rows, m)
        ax.plot(ang, val, ".-", ms=4, label=f"depthMode {m} ({kmall_depth_mode_label(m)})")
    ax.set(title=f"{label} — ARC split by depth mode", xlabel="beam angle (deg, +=port)",
           ylabel="avg backscatter reflectivity2 (dB)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = out / "depthmode_split.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_port_starboard(specs: List[Tuple[str, str, str]], out: Path) -> Path:
    """specs: list of (path, fmt, name)."""
    plt = _plt()
    fig, ax = plt.subplots(1, len(specs), figsize=(6 * len(specs), 4.5), squeeze=False)
    for axi, (path, fmt, name) in zip(ax[0], specs):
        rows = all_arc_rows(path) if fmt == "all" else kmall_arc_rows(path)
        g = fold_port_starboard(rows); sym = port_starboard_symmetry(rows)
        axi.plot(*g["port"], "C0.-", ms=4, label="port (angle>0)")
        axi.plot(*g["stbd"], "C1.-", ms=4, label="starboard (angle<0)")
        axi.set(title=f"{name}\nport/stbd symmetry: mean|Δ|={sym:.2f} dB",
                xlabel="|beam angle| (deg)", ylabel="avg backscatter (dB)"); axi.legend(); axi.grid(alpha=0.3)
    fig.suptitle("QC: port vs starboard ARC should overlap for a balanced system")
    fig.tight_layout(); p = out / "port_starboard_symmetry.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_fan_geometry(path, out: Path) -> Path:
    plt = _plt()
    from mbes_tools.all import iter_depth_datagrams
    x = next(iter_depth_datagrams(Path(path)))
    y = np.array([b.across_track_m for b in x.beams]); z = np.array([b.depth_m for b in x.beams])
    valid = np.array([b.is_valid for b in x.beams])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y[valid], z[valid], s=10, c="C0", label=f"valid ({valid.sum()})")
    ax.scatter(y[~valid], z[~valid], s=14, c="C3", marker="x", label=f"invalid ({(~valid).sum()})")
    ax.invert_yaxis()
    ax.set(title=f"ping {x.counter}: across-track swath fan (num_valid_detections={x.num_valid_detections})",
           xlabel="across-track (m, +=starboard)", ylabel="depth (m)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = out / "fan_geometry.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_autoutm(manifest, out: Path) -> Path:
    plt = _plt()
    import json
    rows = [r for r in json.load(open(manifest)) if r.get("latitude") is not None]
    lon = np.array([r["longitude"] for r in rows]); lat = np.array([r["latitude"] for r in rows])
    epsg = np.array([utm_epsg_from_lonlat(lo, la) for lo, la in zip(lon, lat)])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(lon, lat, c=epsg, cmap="turbo", s=70, edgecolor="k")
    slon, slat = -171.8, -13.8
    ax.scatter([slon], [slat], marker="*", s=320, c="red", edgecolor="k", zorder=5)
    ax.annotate(f"SAMOA EPSG:{utm_epsg_from_lonlat(slon, slat)} (2S)", (slon, slat), color="red",
                fontsize=8, xytext=(6, 6), textcoords="offset points")
    for x in range(-180, 181, 6):
        ax.axvline(x, color="0.92", lw=0.4, zorder=0)
    ax.set(title="auto UTM/UPS EPSG from position (no baked-in zone)", xlabel="longitude",
           ylabel="latitude", xlim=(-185, 185), ylim=(-90, 90)); ax.axhline(0, color="0.5", lw=0.5)
    fig.tight_layout(); p = out / "autoutm_map.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_robustness(path, out: Path) -> Path:
    plt = _plt()
    import struct
    from mbes_tools.all import iter_datagrams
    clean = all_arc_rows(path)
    raw = bytearray(Path(path).read_bytes())
    recs = list(iter_datagrams(Path(path)))
    n = 0
    for rec in recs:
        if rec.header.type_of_datagram == "X" and n < 5:
            o = rec.offset; raw[o:o + 4] = struct.pack("<L", 999_999_999); n += 1
    blob = bytes(raw[:5000]) + b"\xde\xad\xbe\xef" * 40 + bytes(raw[5000:])
    cf = out / "_corrupt.all"; cf.write_bytes(blob)
    log: list = []
    nrec = sum(1 for _ in iter_datagrams(cf, on_error="skip", error_log=log))
    corr = all_arc_rows(str(cf), min_soundings=10)
    cf.unlink()
    ac, vc, _ = arc_xy(clean); ax2, vc2, _ = arc_xy(corr)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ac, vc, "C0.-", ms=4, label=f"clean ({len(recs)} datagrams)")
    ax[0].plot(ax2, vc2, "C3x", ms=5, label=f"corrupted + on_error=skip ({nrec} read)")
    ax[0].set(title="ARC recovered after injected corruption", xlabel="beam angle (deg)", ylabel="dB"); ax[0].legend(fontsize=8)
    ax[1].bar(["clean", "corrupt\n(skip)"], [len(recs), nrec], color=["C0", "C3"])
    ax[1].set(title="datagrams read (resync recovers the survey)", ylabel="datagrams")
    fig.tight_layout(); p = out / "robustness_recovery.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def panel_parity(path, pyall_dir, out: Path) -> Optional[Path]:
    """beam_stat mean vs MBES_ARC (sum/len/10) per Y beam, cross-read with pyall."""
    import sys
    sys.path.insert(0, str(pyall_dir))
    try:
        from pyall import ALLReader
    except Exception:
        return None
    plt = _plt()
    r = ALLReader(str(path)); mine = []; theirs = []
    while r.moreData() and len(mine) < 4000:
        t, d = r.readDatagram()
        if t == "Y":
            d.read()
            for b in d.beams:
                if len(b.samples) == 0:
                    continue
                theirs.append(sum(b.samples) / len(b.samples) / 10.0)
                mine.append(bs.reduce_beam(b.samples, None, None, "mean"))
    r.close()
    mine = np.array(mine); theirs = np.array(theirs); resid = mine - theirs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(theirs, mine, s=6, alpha=0.4); lim = [theirs.min(), theirs.max()]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(title=f"beam_stat mean vs MBES_ARC (n={len(mine)})", xlabel="MBES_ARC sum/len/10 (dB)", ylabel="beam_stat mean (dB)")
    ax[1].hist(resid, bins=40); ax[1].set(title=f"residual (max|Δ|={np.max(np.abs(resid)):.1e} dB)", xlabel="mine − MBES_ARC (dB)")
    fig.tight_layout(); p = out / "parity_mbesarc.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate(output_dir, all_file=None, all_multisector=None, kmall_file=None,
             kmall_survey_dir=None, manifest=None, pyall_dir=None) -> List[Path]:
    """Run whichever panels have their inputs; return saved PNG paths."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []

    def run(fn, *a, **k):
        try:
            p = fn(*a, **k)
            if p:
                made.append(p); print("OK  ", p.name)
        except Exception as exc:  # noqa: BLE001 - a failing panel must not abort the rest
            print("FAIL", getattr(fn, "__name__", fn), "->", type(exc).__name__, str(exc)[:120])

    if all_file:
        run(panel_arc, all_file, out, "all", "EM .all", stem="arc_all")
        run(panel_source_a_vs_b, all_file, out)
        run(panel_multistat, all_file, out)
        run(panel_si_window, all_file, out)
        run(panel_fan_geometry, all_file, out)
        run(panel_robustness, all_file, out)
        if pyall_dir:
            run(panel_parity, all_multisector or all_file, pyall_dir, out)
    if kmall_file:
        run(panel_arc, kmall_file, out, "kmall", "EM .kmall", stem="arc_kmall")
    if kmall_survey_dir:
        files = sorted(glob.glob(str(Path(kmall_survey_dir) / "*.kmall")))[:6]
        if files:
            run(panel_depthmode_split, files, out, f"{Path(kmall_survey_dir).name} ({len(files)} files)")
    ps = [(p, "all", n) for p, n in [(all_file, "EM .all"), (all_multisector, "EM .all (multi-sector)")] if p]
    if ps:
        run(panel_port_starboard, ps, out)
    if manifest:
        run(panel_autoutm, manifest, out)
    return made


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate backscatter review diagnostic plots from real files.")
    ap.add_argument("-o", "--output", default="mbes_diagnostics", help="Output directory for PNGs.")
    ap.add_argument("--all", dest="all_file", help="An EM .all file (drives most .all panels).")
    ap.add_argument("--all-multisector", help="A multi-sector EM .all file (port/stbd + parity).")
    ap.add_argument("--kmall", dest="kmall_file", help="An EM .kmall file (kmall ARC panel).")
    ap.add_argument("--kmall-survey-dir", help="A dir of .kmall files (depth-mode split).")
    ap.add_argument("--manifest", help="verification_manifest.json (auto-UTM map).")
    ap.add_argument("--pyall-dir", help="Dir containing pyall.py (MBES_ARC parity panel).")
    args = ap.parse_args(argv)
    made = generate(
        args.output, all_file=args.all_file, all_multisector=args.all_multisector,
        kmall_file=args.kmall_file, kmall_survey_dir=args.kmall_survey_dir,
        manifest=args.manifest, pyall_dir=args.pyall_dir,
    )
    print(f"\nWrote {len(made)} panel(s) to {args.output}")


if __name__ == "__main__":
    main()
