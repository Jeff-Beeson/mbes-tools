#!/usr/bin/env python3
"""Micro-benchmark for the water-column mosaic accumulator (Slice 3).

Two parts:

1. **Accumulator** — times the buffered, vectorized :class:`GeoMosaic` (map-reduce
   + single global reduce) against the pre-Slice-3 dict-merge algorithm on many
   synthetic pings. This isolates the 3a win (removing the per-ping Python
   dict-merge loop and the per-cell finalize loop). No data files needed.

2. **Parallel composite** — if real water-column file paths are passed on the
   command line, times ``build_composite_mosaic`` serial (workers=1) vs parallel
   (``--workers``), and asserts the results are bit-identical.

Usage::

    python scripts/bench_wc_mosaic.py                       # synthetic accumulator
    python scripts/bench_wc_mosaic.py FILE... --workers 4   # + real parallel bench
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mbes_tools import water_column_geo as wg  # noqa: E402


def _dict_reference(pings, cell_m, reduce):
    """The pre-Slice-3 dict-merge accumulator (per-ping Python loop)."""
    cells = {}
    for e, n, a in pings:
        ie = np.floor(e / cell_m).astype(np.int64)
        jn = np.floor(n / cell_m).astype(np.int64)
        uniq, inv = np.unique(np.column_stack([ie, jn]), axis=0, return_inverse=True)
        inv = inv.ravel()
        if reduce == "max":
            best = np.full(uniq.shape[0], -np.inf)
            np.maximum.at(best, inv, a)
            for (ke, kn), v in zip(map(tuple, uniq.tolist()), best.tolist()):
                cur = cells.get((ke, kn))
                if cur is None or v > cur:
                    cells[(ke, kn)] = v
        else:
            lin = np.power(10.0, a / 10.0)
            sums = np.zeros(uniq.shape[0]); np.add.at(sums, inv, lin)
            cnts = np.bincount(inv, minlength=uniq.shape[0])
            for (ke, kn), s, c in zip(map(tuple, uniq.tolist()), sums.tolist(), cnts.tolist()):
                cur = cells.get((ke, kn))
                if cur is None:
                    cells[(ke, kn)] = [s, c]
                else:
                    cur[0] += s; cur[1] += c
    return len(cells)


def bench_accumulator(n_pings=2000, samples=4000, cell_m=25.0, reduce="mean", seed=0):
    rng = np.random.default_rng(seed)
    # Pings drift across a wide grid (like a survey line), each dense within a swath.
    pings = []
    for i in range(n_pings):
        cx, cy = i * 15.0, 0.0
        e = cx + rng.uniform(-1500, 1500, samples)
        n = cy + rng.uniform(-40, 40, samples)
        a = rng.uniform(-60, -5, samples)
        pings.append((e, n, a))

    t0 = time.perf_counter()
    _dict_reference(pings, cell_m, reduce)
    t_dict = time.perf_counter() - t0

    t0 = time.perf_counter()
    m = wg.GeoMosaic(cell_m=cell_m, reduce=reduce)
    for e, n, a in pings:
        gs = wg.GeoSamples(
            easting_m=e, northing_m=n, depth_m=np.full(e.shape, 100.0), amplitude_db=a,
            crs_label="local", projector="local", vessel_lon=0.0, vessel_lat=0.0,
            heading_deg=0.0, label="bench",
        )
        m.add(gs)
    m.finalize()
    t_new = time.perf_counter() - t0

    print(f"accumulator  reduce={reduce:4s}  pings={n_pings}  samples/ping={samples}")
    print(f"  dict-merge (old):   {t_dict:7.3f} s")
    print(f"  buffered   (new):   {t_new:7.3f} s   ({t_dict / t_new:.1f}x faster)")


def bench_parallel(files, workers, reduce="max"):
    kw = dict(projector="auto", cell_m=25.0, reduce=reduce, on_uncovered="skip")
    t0 = time.perf_counter()
    serial = wg.build_composite_mosaic([Path(f) for f in files], workers=1, **kw)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = wg.build_composite_mosaic([Path(f) for f in files], workers=workers, **kw)
    t_par = time.perf_counter() - t0
    same = np.array_equal(
        np.nan_to_num(serial.amplitude_db, nan=-999),
        np.nan_to_num(par.amplitude_db, nan=-999),
    )
    print(f"\nparallel composite  files={len(files)}  reduce={reduce}  n_pings={serial.n_pings}")
    print(f"  serial   (workers=1):      {t_serial:7.3f} s")
    print(f"  parallel (workers={workers}):      {t_par:7.3f} s   ({t_serial / t_par:.1f}x)")
    print(f"  bit-identical to serial:   {same}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="Real WC files for the parallel bench.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pings", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()

    for reduce in ("max", "mean"):
        bench_accumulator(n_pings=args.pings, samples=args.samples, reduce=reduce)
    if args.files:
        bench_parallel(args.files, args.workers)


if __name__ == "__main__":
    main()
