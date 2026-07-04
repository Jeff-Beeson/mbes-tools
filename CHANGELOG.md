# Changelog

All notable changes to `mbes-tools` are documented here. The project follows
[semantic versioning](https://semver.org/) (currently 0.x — minor versions may
add features; the public API is kept backward compatible where practical).

## [0.15.0] - 2026-07-03

### Changed (water-column mosaic — accumulator performance)
- **`GeoMosaic` rewritten as a vectorized map-reduce accumulator.** The old
  per-ping Python **dict-merge** loop and the per-cell Python **finalize** loop
  (the two accumulation hot spots) are gone. Each ping is reduced to its unique
  cells and **buffered** as sparse `(iE, iN, value[, count])` rows; a single
  global reduce rasterizes to the dense grid. Both the per-ping and global
  reductions encode the cell as one int64 key and use C-level primitives
  (`bincount` for the mean sum/count, sort + `maximum.reduceat` for the peak)
  instead of a 2-D `unique`/lexsort. **~6–10× faster** accumulation on a
  synthetic 2000-ping × 4000-sample line (see `scripts/bench_wc_mosaic.py`);
  output is **bit-identical** to the previous implementation (a regression test
  asserts this cell-by-cell for both `max` and `mean`). Buffers compact once they
  exceed `compact_rows` so memory stays proportional to occupied cells.

### Added (water-column mosaic — parallel composite)
- **`mbes-wc-mosaic --combine --workers N`** georeferences the files across a
  process pool (one file per task). The shared grid anchor is fixed up front and
  each worker returns its per-ping partials, merged **in file order**, so the
  result is **bit-for-bit identical to the serial build** for both `max` and
  `mean` (verified on the real EM124 fixture). Default `workers=1` keeps the
  original streaming path. Best for many-file surveys; small inputs are dominated
  by process-startup overhead.
- `scripts/bench_wc_mosaic.py` — micro-benchmark for the accumulator (new vs old)
  and, given real files, the serial-vs-parallel composite (with a bit-identity
  check).

## [0.14.0] - 2026-07-03

### Added (water-column mosaic — height-above-seafloor band)
- **`mbes-wc-mosaic --altitude-band LO:HI`** — keep only samples whose height
  **above the detected seafloor** is in `[LO, HI]` metres, e.g.
  `--altitude-band 20:200` for a near-bottom scattering layer that follows the
  terrain rather than a flat depth slice. Complements the existing
  `--depth-band` (absolute depth); the two **compose** (a sample must satisfy
  both).
- `georeference_frame` now computes per-sample **height above seafloor** from the
  per-beam bottom detection already carried on the frame
  (`WCFrame.detected_samples`): seafloor depth `Zb = cos(angle)·c·det/(2·fs)`,
  height `= Zb − Z`. Beams with no bottom detection (`det == 0`, e.g. swath
  edges) yield `NaN` and are excluded from any altitude product. Exposed as
  `GeoSamples.height_above_seafloor_m`; `GeoMosaic`/`GeoMosaicResult` gain an
  `altitude_band`.
- **Verified end-to-end** on the committed EM124 `.kmwcd` (~2900 m abyssal):
  `--altitude-band 0:300` keeps 262/395 near-bottom cells and `1000:3000` a
  different 236/395 higher in the water column — each a valid seafloor-relative
  subset. New unit tests cover the HAB geometry (nadir + oblique beams), the
  no-bottom NaN exclusion, the mosaic filter, depth+altitude composition, and the
  missing-HAB guard.

## [0.13.0] - 2026-07-03

### Added (water-column mosaic — georeferenced raster export)
- **`mbes-wc-mosaic` can now write the plan-view mosaic as a raster**, not just a
  PNG panel:
  - **`--geotiff`** — a single-band float32 **GeoTIFF** (amplitude dB, `NaN`
    nodata, north-up) with the real projected CRS embedded. Requires
    `--projector utm` (so a true EPSG is resolved) + rasterio
    (`pip install 'mbes-tools[geo]'`); `export_geotiff` raises a clear error in the
    local-ENU frame or when rasterio is absent.
  - **`--asc`** — an **ESRI ASCII Grid** (`.asc`) plus a `.prj` WKT sidecar when a
    projected EPSG is available. Pure numpy + stdlib, so it works in the base env
    and is GDAL-convertible to GeoTIFF.
- **EPSG is now threaded through** `GeoSamples` → `GeoMosaic` → `GeoMosaicResult`
  (`epsg` = the real projected code in `utm` mode, `None` for local ENU), so the
  writers embed the correct CRS instead of re-parsing the `crs_label` string.
- New optional extra **`geo = ["pyproj", "rasterio"]`**; both are lazy-imported so
  the core stays numpy + stdlib.
- **Verified end-to-end** on the committed EM124 `.kmwcd`: `--projector utm
  --geotiff --asc` writes a UTM-52N GeoTIFF (EPSG:32652, 25 m cells) whose
  filled-cell centroid round-trips to 126.974°E / 6.904°N — the fixture's true
  position — plus a matching `.asc`/`.prj`. New unit tests cover north-up
  orientation, the ASCII header, EPSG threading, the projected-CRS guard, and a
  gated GeoTIFF round-trip (rasterio).

## [0.12.0] - 2026-07-02

### Added (water-column viewer — interactive display controls)
- **`mbes-wc-viewer` gained operator controls** on top of the linked
  stack + fan (all live; also settable up front from the CLI):
  - **Shared colour scale + whole-file amplitude histogram** — a `RangeSlider`
    beneath a histogram of the file's amplitudes sets one min/max shared by both
    panels (`--clim LO HI`); the histogram shows the distribution with the
    current min/max as red guide lines.
  - **Clamp/cut toggle** (`--clip {clamp,cut}`) — out-of-range samples either show
    the colormap end colours (clamp, default) or are cut out and rendered
    transparent (via the colormap's under/over), so weak water column can be
    dropped to isolate the seafloor and strong scatterers.
  - **Drag-a-band swath selection** — a `SpanSelector` on the fan: drag an
    across-track band and the along-track stack rebuilds from only that band
    (`PingView.depth_column(..., across_window=...)` +
    `WaterColumnFileView.rebuild_stack`), shown as a shaded overlay on the fan;
    double-click or `r` resets to the full swath. `--swath LO HI` sets it up front.
  - **Cursor lat/lon readout** — a status line driven by `motion_notify_event`:
    over the fan, the geographic position of the cursor's across-track point for
    the current ping (`PingView.across_to_lonlat`, equirectangular from the ping
    position + heading); over the stack, the nadir (vessel) position of the ping
    under the cursor.
- The interactive widgets live only in `WaterColumnViewer.show()`, so the
  headless `render_static` path and its tests are unaffected; every control's
  effect is a small method (`_apply_clim` / `_set_clip` / `_on_swath` /
  `_status_over_*`) unit-tested on an Agg viewer plus pure-geometry tests for
  `across_to_lonlat` and the `across_window` collapse. Verified on real EM304
  H14070 `.kmwcd` (clim + cut isolate the seafloor/midwater; a ±400 m swath band
  gives a clean near-nadir section). `232 passed, 2 skipped`.

### Fixed (water-column viewer)
- **Window freeze on mouse move** — the lat/lon readout was wired to
  `motion_notify_event` and called `draw_idle()` on every move, forcing a full
  redraw of the ~200k-point fan on each event (which, together with the blitting
  `SpanSelector`, locked the window up). Moved the readout to matplotlib's
  built-in `Axes.format_coord` (toolbar coordinate area, no per-motion redraw)
  and capped the fan scatter at ~60k points (display-only decimation) so
  ping-scrub and slider redraws stay snappy.
- **Off-screen / hidden window on WSLg** — `show()` now nudges the window to a
  visible position and briefly raises it to the front (backend-agnostic, guarded,
  a no-op headless).

## [0.11.0] - 2026-07-02

### Added (Capability D1 products — interactive water-column viewer)
- **`mbes_tools.wc_viewer`** (console script **`mbes-wc-viewer`**) — the first
  *interactive* water-column product (the earlier renderers all emit static
  PNGs). One window, two linked panels over a whole file:
  - **top** — an along-track **depth stack of the whole file**: every ping's fan
    collapsed to an amplitude-vs-depth column and laid side by side (x = ping
    index along track, y = depth), with a movable ping cursor. Collapse modes:
    `swath-max` (peak-hold across all beams — surfaces any midwater target),
    `swath-mean` (linear-intensity average), or `nadir` (a clean near-vertical
    section from beams within a half-angle of vertical).
  - **bottom** — the selected ping's **navigation/attitude-corrected wedge fan**
    (across-track x depth), amplitude-coloured, with the bottom detection
    overlaid; roll/pitch/heave are shown in the title.
  - **linked & interactive** — click the stack or use ←/→ (and PgUp/PgDn,
    Home/End) to scrub which ping the fan shows.
  - Navigation, attitude and geometry are reused verbatim from
    `water_column_geo` (`resolve_nav_track` companion discovery, the coverage
    guard with `--on-uncovered skip|clamp`, `NavTrack.attitude_at`): the fan is
    left **receive-stabilized** (heave + transducer depth added to depth, beams
    not re-rotated — the Slice-3 physics), so the viewer agrees with the mosaic.
  - Bounded memory: a whole file is decimated to `--max-pings` (adaptive stride)
    x `--fan-samples`. `--save PING` renders the linked panels headless (Agg) for
    review/tests; the interactive window needs a GUI backend (TkAgg/Qt).
- **Verified against real data:** full EM304 H14070 `.kmwcd` (`0012_*`, S221
  Cascadia): 370/372 pings (2 lead pings correctly coverage-skipped), companion
  `#SKM` nav + attitude auto-discovered, the stack traces the ~1900 m seafloor
  (shoaling along track) with a midwater scattering layer above it, and the fan
  shows the symmetric ±2.8 km wedge with the bottom-detect overlay. Geometry and
  the fan->column reductions are unit-tested; the model + static render are
  gated over the committed EM124 `.kmwcd` / EM122 `.wcd` fixtures.

## [0.10.0] - 2026-07-02

### Added (Capability D1 products — Slices 2–3: geo-referenced mosaics + attitude)
- **`mbes_tools.water_column_geo`** (console script **`mbes-wc-mosaic`**) — true
  geo-referenced plan-view water-column products (no new required deps; numpy +
  stdlib, matplotlib + pyproj lazy/optional):
  - **`NavTrack`** (linear position interp, circular sin/cos heading interp) with
    optional roll/pitch/heave, built by `nav_track_from_kmall` (prefers `#SKM`
    true heading, falls back to `#SPO` course-over-ground) / `nav_track_from_all`
    (`P`, with `A` attitude).
  - **`georeference_frame`** — places each sample at `(easting, northing, depth)`
    reusing the Slice-1 wedge; lever arm from `install.transducer_offsets`,
    heading rotation, local ENU metres or true UTM (pyproj), auto-UTM EPSG always
    resolved for provenance.
  - **`GeoMosaic`** — streaming sparse-cell accumulator (`max` peak-hold or
    `mean` linear-intensity) with an optional `depth_band` (midwater/plume) ->
    dense grid.
  - **Nav-source robustness** — `resolve_nav_track` does not assume the WC file
    carries usable nav: explicit `--nav` > same-stem `.kmall`/`.all` companion
    (prefers `#SKM`) > the WC file's own `#SPO`/`P` > a clear error; a coverage
    guard (`--on-uncovered skip`, default) drops pings outside the nav span
    instead of clamping them to a track endpoint.
  - **`.wcd`/`.all` path** and a multi-line **`--combine`** composite (many files
    into one shared-anchor, one-CRS mosaic); midwater/plume product via
    `--depth-band LO:HI --reduce mean`.
  - **Vessel attitude (Slice 3)** — the lever arm is rotated by the full pose
    `Rz(H)·Ry(pitch)·Rx(roll)` and heave added to depth, but the **beam fan is
    left roll/pitch-stabilized** (Kongsberg `beamPointAngReVertical` is already
    stabilized at receive — re-rotating double-corrects; `--unstabilized-beams`
    opts out). Verified on real EM124/EM122/EM304 data (matched `.kmwcd`/`.kmall`
    pairs, `.wcd` reassembly, TN447 + H14070).

## [0.9.0] - 2026-06-30

### Added (Capability D1 products — Slice 1: vessel-frame water column)
- **`mbes_tools.water_column`** — the first *product* layer on the validated
  water-column readers (no new dependencies; core numpy + stdlib, matplotlib
  lazy). Three pieces:
  - **`.wcd` ping reassembly** — `reassemble_wcd_pings` / `merge_wcd_fragments`
    concatenate the multiple `k` datagrams of a fragmented ping (`num_datagram`
    > 1, beams partitioned by `datagram_num`) back into one full swath, grouped
    by `counter`; `reassembled_wcd_frames(path)` yields a `WCFrame` per full
    ping. Streaming (bounded memory); truncated tail pings are dropped unless
    `allow_incomplete=True`.
  - **Cartesian gridding** — `grid_frame(frame) -> WaterColumnGrid`: bins each
    amplitude sample onto a regular `(across_track_m, depth_m)` grid using the
    existing `wc_diagnostics` wedge geometry (`r = c·k/(2·fs)`, angle positive =
    port). `reduce="mean"` averages in the **linear-intensity domain**
    (`10**(dB/10)` → back to dB, the correct incoherent echo-integration mean);
    `reduce="max"` is a peak-hold. Default cell = one-way range resolution,
    capped at `max_cells`; `max_depth_m` / `max_across_m` clip the extent.
  - **TVG-residual midwater/plume anomaly pass** — `detect_anomalies(frame) ->
    WaterColumnAnomalies`: subtract a per-range background (across-beam median of
    **water-only** samples — near-field and seafloor + guard band excluded) to
    flatten the TVG/absorption trend, then flag positive residual outliers with a
    robust `median + n_mad·(1.4826·MAD)` threshold (or a fixed `threshold_db`).
    `summarize_anomalies` returns a JSON-friendly per-ping summary.
- **`mbes-wc-grid`** console script — renders a gridded-echogram panel and an
  amplitude/residual/background anomaly panel per file (reassembling `.wcd` by
  `counter` first) and prints the anomaly summary. matplotlib lazy (`gui` extra).
- **Verified against real data:** `.wcd` reassembly on the full Atlantis EM122
  file — **762/762 pings** reconstruct exactly (`sum(num_beams_this_datagram) ==
  num_beams_ping` for every ping, mostly 2 fragments; beam numbers monotonic);
  the committed 3-ping clip passes through unchanged. Grid + anomaly produce sane
  vessel-frame geometry on the committed EM124 `.kmwcd` (symmetric ±30° swath to
  ~6.7 km, `mean`/`max` consistent), EM2040 phase-1 `.kmall` (~9 m, 2.4 cm
  cells), and reassembled EM122 `.wcd`; anomaly defaults yield non-zero water on
  both deep and shallow regimes. Numeric helpers unit-tested; render test gated
  on matplotlib over the committed fixtures. Review PNGs:
  `~/mbes_review_plots/wc_d1_products/`.

## [0.8.0] - 2026-06-29

### Added (Capability D3 — attitude & installation parsers)
- **`.all` attitude (`A`, 65) and installation (`I`, 73).** `parse_attitude` /
  `iter_attitude_datagrams` decode the per-entry roll/pitch/heave/heading time
  series; `parse_installation` / `iter_installation_datagrams` decode the
  install string. New dataclasses `AttitudeSample`, `AttitudeDatagram`,
  `InstallationDatagram`.
- **`.kmall` attitude (`#SKM`), installation/runtime (`#IIP`/`#IOP`), and
  navigation (`#SPO`/`#CPO`).** `parse_skm_datagram` / `iter_skm_datagrams`
  decode the KMbinary attitude/velocity/acceleration samples (advancing by
  `numBytesPerSample` so versions are tolerated); `parse_kmall_params_datagram`
  / `iter_iip_datagrams` / `iter_iop_datagrams` decode the parameter text;
  `parse_position_datagram` / `iter_spo_datagrams` / `iter_cpo_datagrams` decode
  the position sensor output (lat/lon, SOG/COG, ellipsoid height, fix quality,
  raw NMEA) — the `.kmall` analogue of the `.all` `P` datagram (`#SPO` referred
  to the vessel reference point, `#CPO` to the antenna at water level). New
  dataclasses `KMBinarySample`, `SKMDatagram`, `KmallParamsDatagram`,
  `KmallPositionDatagram`. All share the existing `on_error="skip"` resync.
- **`mbes_tools.install_params`** — structures the Kongsberg install/runtime
  text (transducer **lever arms** `X/Y/Z`, **mount angles** `R/P/H`,
  **waterline**, EM model, serial) for downstream re-georeferencing / ARC
  refinement, across both schemes: `.all` flat `KEY=VALUE` (`S{n}` groups) and
  `.kmall #IIP` nested `SECTION:sub=val;sub=val` (`TRAI_TX1`/`TRAI_RX1` groups).
  Supersedes the catalog's regex EM-model scrape.
- **Verified against committed fixtures by byte reconciliation + field sanity:**
  EM124 `#SKM` (102 KMbinary samples/datagram, 132 B each: `20 + infoPart +
  N·perSample + 4 == numBytesDgm`); EM124 `#IIP` → EM124 / SN 10055 / waterline
  0.74 m / TX (4.221, 0.914, 6.225) m, RX (8.558, 1.517, 6.225) m; EM302 `.all`
  `A` (102 samples, descriptor) and `I` → waterline −2.07 m, S1/S2 lever arms +
  mount angles. `#SPO`/`#CPO` reconcile exactly and the decoded binary lat/lon
  matches the embedded NMEA `GGA` (EM124 6.565°N/126.760°E; EM304 Monterey
  36.467°N/−122.608°W). No new fixtures required — all these datagrams were
  already present in `sample_tn447_em124.kmall` / `sample_nautilus.all`.

### Not yet
- `.all` network attitude (`n`, 110) — the variable-length raw-input attitude
  variant — is not parsed yet; `A` (65) covers attitude. Follow-up if needed.

## [0.7.0] - 2026-06-29

### Added (Capability D1 — water-column reader validation)
- **Validated `mbes_tools.kmwcd` (`#MWC`) and `mbes_tools.wcd` (`k`) against
  real water-column data.** Both readers were previously written from spec with
  synthetic tests only; `kmwcd` phase-sample handling (`phase_flag` 1/2) was
  explicitly untested. They are now confirmed against real bytes by exact
  datagram-size reconciliation (predicted size from the struct sizes + per-beam
  block formula `numBytesPerBeamEntry + numSampleData*(1 + phaseSize)` ==
  declared `numBytesDgm`), which pins the layout — including the `phase_flag`
  1/2 sample element sizes — against reality:
  - EM124 `.kmwcd` (R/V Thompson, cruise TN447, abyssal): 374/374 `#MWC`,
    `phase_flag` 0, `dgm_version` 2 (16-byte beam entry with the
    high-resolution detection field).
  - EM2040 ASV `.kmall`, `phase_flag` 1 (int8, 180/128 deg): 40/40 `#MWC`;
    real phase range ±128 ≈ ±180 deg.
  - EM2040 ASV `.kmall`, `phase_flag` 2 (int16, 0.01 deg): 4/4 `#MWC`; real
    phase range ±18000 ≈ ±180 deg.
  - EM122 `.wcd` (R/V Atlantis): 1407/1407 `k` datagrams reconcile (no drift,
    modulo the single Kongsberg spare byte before the footer); fragmented pings
    (`num_datagram` > 1) modelled correctly.
- **New committed fixtures** (`tests/fixtures/`): `sample_tn447_em124.kmwcd`
  (EM124 abyssal, `phase_flag` 0), `sample_em2040_wc_phase1.kmall` (EM2040 ASV,
  `phase_flag` 1), `sample_atlantis_em122.wcd` (EM122, 3 pings). The
  high-resolution `phase_flag` 2 file (~2.9 MB/`#MWC`) is too large to commit;
  its test is gated on the external file (`MBES_MWC_PHASE2_FILE`).
- **New tests:** real-fixture integration tests for both readers, a synthetic
  `phase_flag` 2 round-trip, and a gated real `phase_flag` 2 test.
- **`mbes_tools.wc_diagnostics` (`mbes-wc-diagnostics`):** a reproducible
  water-column visual review suite (the `mbes_tools.diagnostics` analogue for
  `#MWC`/`k`). Renders, from real files, an amplitude **echogram** with the
  per-beam detected-bottom overlay, a **geo-referenced swath wedge**
  (across-track/depth from beam angle + range), a **nadir amplitude profile**,
  a **bottom-detect vs amplitude-peak** alignment check, **phase echogram +
  histogram** for `phase_flag` 1/2, and a cross-file **sector-frequency**
  sanity panel. matplotlib is lazy (the `gui` extra); the numeric helpers
  (`phase_scale_deg`, `padded_grid`, `wedge_coords`, `range_axis`,
  `peak_sample_indices`) are dependency-light and unit-tested, with a render
  test gated on matplotlib that uses the committed fixtures.

### Fixed
- `kmwcd.py`: corrected a misleading code comment that stated the RX beam header
  was 16 bytes (v1) / 20 bytes (v2); the actual struct sizes are 12 / 16 bytes
  (already correct in code and asserted by `test_struct_sizes`).

## [0.6.0] - 2026-06-29

### Added
- `mbes_tools.diagnostics` (`mbes-diagnostics`): a path-agnostic visual review
  suite that renders backscatter diagnostics from real files — ARC + Lambertian
  balancing, Source A vs Source B, multi-stat + within-beam texture, `--si-window`
  comparison, depth-mode split, port/starboard symmetry, swath-fan geometry +
  validity, auto-UTM map, MBES_ARC beam-level parity, and corruption-recovery.
  matplotlib is imported lazily (the `gui` extra); the numeric helpers
  (`all_arc_rows`, `kmall_arc_rows`, `arc_xy`, `mode_mean_curve`,
  `fold_port_starboard`, `port_starboard_symmetry`) are dependency-light and
  unit-tested. A gated end-to-end test renders panels when `MBES_TEST_DATA_ROOT`
  and matplotlib are available.

## [0.5.0] - 2026-06-29

### Added (Capability C — generalization for other geographies & instruments)
- `mbes_tools.projection`: geography-agnostic target-CRS resolution —
  `utm_epsg_from_lonlat` / `resolve_target_crs` pick the right UTM zone (or polar
  UPS) from a position, safe across hemispheres, the antimeridian, and high
  latitudes. No survey-specific zone is baked into library logic.
  `SpatialProjector.from_spec(spec, lon, lat)` resolves `auto` → UTM/UPS, or an
  explicit EPSG / PROJ string.
- **Robustness — tolerant readers.** `mbes_tools.all.iter_datagrams` and
  `mbes_tools.kmall.iter_mrz_datagrams` take `on_error="skip"` (+ optional
  `error_log`), resynchronizing past a corrupt/truncated datagram to the next
  valid one instead of raising. `mbes-bs-table --on-error skip` (default) carries
  this through the pipeline: per-file failures are caught, corrupt datagrams are
  counted and skipped, and a robustness summary is printed — one bad ping or
  file never aborts a survey. gzip/zip containers are reported, not crashed.
- **Depth-mode maps consolidated** into `mbes_tools.depth_modes`
  (`kmall_raw_to_calib`, `kmall_depth_mode_label` alongside the .all maps);
  `backscatter.apply.depth_mode_raw_to_calib` now delegates there.

### Verified
- Auto-UTM matches every position in the verification manifest and the Samoa
  area (−171.8, −13.8) → EPSG:32702 (UTM 2S). Tolerant readers recover
  downstream datagrams past injected corruption on real EM2040 `.all` and EM124
  `.kmall`; the pipeline completes on a gzip'd `.all`. Cross-model parse sweep
  clean across EM122/124/302/304/2040 in both formats.

## [0.4.0] - 2026-06-29

### Added (Capability B — configurable backscatter source + per-beam reducer)
- `mbes_tools.beam_stat`: a pluggable numpy registry that reduces a beam's
  seabed-image sample array to one dB value — `mean`, `median`, `std`, `var`,
  `mode`, `trimmed_mean`, `min`, `max`, `range`, `count`, and `p<NN>`
  percentiles — over a selectable sample window (`window_bounds`) around the
  bottom-detection (centre) sample. `std`/`var`/`range` double as within-beam
  texture features.
- **Source B** (seabed-image samples reduced per beam) wired into **both**
  front-ends alongside the existing **Source A** (per-beam reflectivity):
  `--bs-source reflectivity|seabed_image`, `--beam-stat` (list), `--si-window`.
  .kmall reduces `SIsample_desidB`; .all reduces the matched `Y` datagram
  (paired with `N`/`X` by ping counter). `kmall.MRZSounding.si_centre_sample`
  is now parsed.
- **Multi-stat single-read pass**: `SoundingRecord.intensity_by_stat` plus
  `extra_agg`/`extra_stats` plumbing let several reducers be compared in one
  read; each appears as `avgIntensity_<stat>_dB` / `stdIntensity_<stat>_dB`
  columns in the table CSV.

### Verified
- **Beam-level MBES_ARC parity**: across 2,160 real EM302 `Y` beams,
  `beam_stat` `mean` reproduces MBES_ARC's per-beam ARC value (`sum/len/10`) to
  ~1e-14 dB. On full EM2040 (170,682 soundings) Source A and Source B correlate
  at 0.85 with a 0.18 dB median offset.

## [0.3.0] - 2026-06-29

### Added (Capability A — .all backscatter parity)
- `mbes_tools.all`: parsers for **datagram 78 `N`** (raw range and angle —
  per-beam pointing angle + transmit sector + reflectivity) and **datagram 82
  `R`** (runtime — depth/ping mode). `XBeam.is_valid` flags real detections.
  `iter_datagrams(types=...)` can skip bodies of unwanted datagram types.
  Byte layouts verified against the pyall reference on real EM2040 data.
- `mbes_tools.depth_modes`: one documented table mapping each (model, format)
  depth/ping-mode encoding onto a canonical id + label, including the
  EM2040/EM2045 case where the .all runtime mode byte selects **frequency**
  (200/300/400 kHz) rather than a depth band.
- `mbes_tools.backscatter.all_table`: `process_all_ping` / `accumulate_all_file`
  join per-ping `X`+`N`+`R`+`P` into the existing `SoundingRecord`, reusing
  `aggregate_records` / ping QC / `build_rows` / `normalize` unchanged, so
  EM2040 `.all` and EM124 `.kmall` share one backscatter pipeline.
- `.all` write-back path in `mbes_tools.backscatter.apply` (`process_one_all`)
  patches `Y` seabed-image samples by sector correction (pairs `Y`↔`N` by ping
  counter for sectors, depth mode from `R`).
- `mbes-bs-table` and `mbes-bs-apply` now **dispatch by file extension**
  (`--format auto|kmall|all`); `--reflectivity-source xyz88|rawrange78` for `.all`.
- Committed fixtures: `sample_equinox_em2040.all` (EM2040, Samoa AUV-matched) and
  `sample_tn447_em124.kmall` (EM124, Samoa ship-matched). `docs/DEPTH_MODES.md`.

### Changed
- Ping-level QC extracted to `backscatter.table.apply_ping_qc`, shared by the
  .kmall and .all front-ends (behavior for .kmall unchanged).

## [0.2.0] - 2026-06-28

### Added
- `mbes_tools.catalog`: a path-agnostic tool to inventory Kongsberg
  `.all`/`.wcd`/`.kmall`/`.kmwcd` files into a verification manifest (CSV/JSON).
  Records path, format, MB-System format id, EM model, vessel (heuristic),
  latitude/longitude, UTM zone, depth regime, datagram-type census,
  seabed-image presence, and per-file errors. Envelope-only scanning keeps it
  cheap on large files; `--per-dir-limit` samples representative files; gzip/zip
  containers are detected and reported instead of crashing. Console script
  `mbes-catalog`.
- `docs/VERIFICATION_DATA.md` and `docs/verification_manifest.csv`/`.json`:
  the real verification corpus catalog for the upgrade plan (§2), covering
  EM124, EM2040, EM304, EM302, EM122, EM710 across both hemispheres and all four
  Kongsberg file formats.

## [0.1.0]

### Added
- Initial scaffolding lifted from prior project work: `kmall`, `all`, `wcd`,
  `kmwcd`, `imb` readers; `mbsystem` MB-System wrappers; `backscatter`
  sector/angle normalization (`table`, `normalize`, `apply`) with console
  scripts `mbes-bs-table`, `mbes-bs-apply`, `mbes-bs-gui`.
