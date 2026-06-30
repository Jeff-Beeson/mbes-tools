# Changelog

All notable changes to `mbes-tools` are documented here. The project follows
[semantic versioning](https://semver.org/) (currently 0.x — minor versions may
add features; the public API is kept backward compatible where practical).

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
