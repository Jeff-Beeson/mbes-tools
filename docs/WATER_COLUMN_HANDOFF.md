# mbes-tools — water-column (D1) handoff

A design/planning handoff for the water-column work: current state, the technical
facts and conventions a product designer must respect, and the open questions for
the next phase. For the full per-capability status see `docs/BUILD_STATUS.md`
(canonical), the API view in `docs/IMPLEMENTATION_SUMMARY.md`, the spec in
`docs/UPGRADE_PLAN.md`, and the real-data corpus in `docs/VERIFICATION_DATA.md`.

**As of:** 2026-06-29 · **Version:** 0.7.0 · **PR:** #8
(`capability-d1/water-column-validation` → `main`). `pytest -q` = 142 passed, 2 skipped.

## What D1 delivered

1. **Validated the water-column readers against real data.** `mbes_tools.kmwcd`
   (`#MWC`, `.kmwcd`) and `mbes_tools.wcd` (`k`, `.wcd`) were previously spec-only;
   `kmwcd` phase handling was untested. Both are now confirmed by **exact
   datagram-size byte reconciliation** — predict each datagram's size from the
   struct sizes + the per-beam block formula
   `numBytesPerBeamEntry + Ns·(1 + phaseSize)` and match the declared
   `numBytesDgm`. A wrong struct size or phase element size (int8 vs int16) cannot
   reconcile across hundreds of variable-length beams, so this pins the layout —
   including the `phase_flag` 1/2 sample sizes — against reality.

   | Reader | Real file | Result |
   |---|---|---|
   | `kmwcd` `#MWC` | EM124 `.kmwcd` — TN447, R/V Thompson, abyssal | 374/374 reconcile; `phase_flag` 0, `dgm_version` 2 (16-byte beam entry) |
   | `kmwcd` `#MWC` | EM2040 ASV LowResPhase `.kmall` | 40/40; `phase_flag` 1 (int8), phase ±128 ≈ ±180° |
   | `kmwcd` `#MWC` | EM2040 ASV HiResPhase `.kmall` | 4/4; `phase_flag` 2 (int16), phase ±18000 ≈ ±180° |
   | `wcd` `k` | EM122 `.wcd` — R/V Atlantis | 1407/1407 reconcile (constant +1 spare byte, no drift); fragmented pings handled |

2. **`mbes_tools.wc_diagnostics` (`mbes-wc-diagnostics`)** — review suite that makes
   the validation reproducible: amplitude echogram + detected-bottom overlay,
   geo-referenced swath wedge, nadir profile, bottom-detect/amplitude-peak 1:1
   alignment, phase echogram + histogram (`phase_flag` 1/2), sector-frequency
   sanity. matplotlib is lazy (`gui` extra); numeric helpers are unit-tested; the
   render test is gated on matplotlib over the committed fixtures.

   ```
   mbes-wc-diagnostics -o plots/ \
     --mwc tests/fixtures/sample_tn447_em124.kmwcd \
           tests/fixtures/sample_em2040_wc_phase1.kmall \
     --wcd tests/fixtures/sample_atlantis_em122.wcd
   ```

## API/CLI to build products on

- Readers: `kmwcd.iter_mwc_datagrams(path)` → `MWCDatagram` (beams, tx_sectors,
  `phase_flag`, `sample_freq_hz`, `sound_velocity_m_s`);
  `wcd.iter_water_column_datagrams(path)` → `WaterColumnDatagram`. The `#MWC`
  reader works on `.kmwcd` **and** `.kmall` (identical framing).
- `wc_diagnostics` reusable numeric helpers (numpy-only, no matplotlib):
  `phase_scale_deg`, `padded_grid(starts, nsamps, vals, scale)` → `[beam, width]`
  NaN-padded grid, `range_axis(width, c, fs)`, `wedge_coords(angles, width, c, fs)`
  → `(X across, Z depth)`, `peak_sample_indices`; plus the `WCFrame` bundle and
  `frame_from_mwc` / `frame_from_wcd` adapters.
- Geometry the helpers encode (reuse for products): one-way slant range
  `r = c·k / (2·fs)`; `X = r·sin(angle)`, `Z = r·cos(angle)`, **angle positive = port**.

## Technical facts the design must respect

- **Phase:** flag 1 = int8 @ 180/128°·unit; flag 2 = int16 @ 0.01°·unit; both span
  ±180°. Incoherent (≈uniform) in open water; flag-2 concentrates near 0° (coherent).
- **Sample-rate decimation:** deep CW systems decimate WC `sampleFreq` hard
  (EM122 ~67 Hz, EM124 ~127 Hz → ~6–11 m range bins) vs EM2040 ~30 kHz.
  Physically correct; compute range from the **per-ping** `fs`, never assume.
- **`.wcd` fragmentation:** most EM122 pings span multiple `k` datagrams
  (`num_datagram` > 1). **Reassemble by `counter`** before forming a full swath
  (the committed fixture has 3 complete single-datagram pings; full files fragment).
- **`#MWC` dual swath:** EM124 deep mode emits `rx_fans_per_ping` up to 2
  (`rx_fan_index`); a full ping = both fans.
- **No bottom detection** at swath edges → `detected_range_samples == 0`
  (treat as "no bottom").
- **Georeferencing gap:** `#MWC`/`k` give beam angle + range + heave only — **not**
  vessel position/attitude per sample. True geo-referenced mosaics need
  position/attitude (`#SPO`/`#SKM`; `.all` `P`/attitude) → pulls in **D3**
  (attitude/installation parsers) + `mbes_tools.projection` (auto-UTM;
  Samoa → EPSG:32702 / UTM 2S).

## Data / fixtures

- Committed (lean): `sample_tn447_em124.kmwcd` (623K, EM124 abyssal, phase 0),
  `sample_em2040_wc_phase1.kmall` (384K, EM2040 phase-1 int8),
  `sample_atlantis_em122.wcd` (152K, EM122, 3 pings).
- Full real data stays external. EM124 `.kmwcd`: TN447 `EM124.Data` (15 files,
  abyssal Samoa-ship match). Phase-enabled `#MWC`: valschmidt
  `kmall-master/data/*Phase_subset.kmall` (HiRes = flag 2, LowRes = flag 1).
  EM122 `.wcd`: Atlantis. The large `phase_flag`-2 test is gated on
  `MBES_MWC_PHASE2_FILE`.

## Recommended next products (design targets)

1. **Geo-referenced echogram grids / mosaics** — bin (across, depth) or
   (lon, lat, depth) amplitude using `projection`; needs `.wcd` ping reassembly +
   (for true geo) position/attitude → consider sequencing **D3 with/before** full
   mosaics. A **vessel-frame (across/depth) product needs nothing new** and is the
   cheap first win.
2. **Midwater / plume anomaly detection** — TVG-residual / background-subtracted
   amplitude above the bottom; flag coherent off-bottom returns (Samoa hydrothermal
   plumes). Validated phase supports interferometric water-column targets.
3. **Bottom-detection QC** — compare `#MWC` `detected_range` vs `#MRZ` bathy;
   the alignment check already exists as a diagnostic primitive.

## Conventions / constraints (keep)

- Core paths **numpy + stdlib only**; scipy/pandas/pyproj/matplotlib lazy-imported
  (matplotlib = `gui` extra). Base env has numpy + dotenv only.
- One capability per branch + PR (merge commits, not squash); bump version in
  `pyproject.toml` **and** `src/mbes_tools/__init__.py`; update CHANGELOG/README/docs.
- Heavy data out of the repo; integration tests gated and skip when data absent;
  verify each capability against real data and state what was used.
- Backward compatible: `samoa_cm_tools` imports `mbes_tools` unchanged.

## Open questions for design

- Vessel-frame echogram product first (no new deps), or straight to geo-mosaic
  (requires D3 attitude/position)?
- Plume detection: rule-based TVG-residual first, or hold for a labeled dataset?
- Pooled (sample-count-weighted) vs per-beam aggregation for water column, mirroring
  the documented backscatter ARC weighting difference?
