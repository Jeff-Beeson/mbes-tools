# mbes-tools — implementation summary (what the library now provides)

A reference for what landed from the `docs/UPGRADE_PLAN.md` work, for planning
downstream work (e.g. `samoa_cm_tools`). For the resume/handoff view (conventions,
gaps, Capability D direction, next-session checklist) see `docs/BUILD_STATUS.md`;
for the test corpus see `docs/VERIFICATION_DATA.md`; for mode maps see
`docs/DEPTH_MODES.md`.

**Status:** UPGRADE_PLAN Capabilities **A, B, C done & on `main`** (v0.6.0);
**D1 water-column reader validation done** (v0.7.0, branch
`capability-d1/water-column-validation`); **D2/D3** = backlog; **Samoa
acceptance** pending data download.
**Tests:** `python -m pytest -q` → 142 passed, 2 skipped (2 gated on
`MBES_TEST_DATA_ROOT` + matplotlib; a 3rd water-column test runs when the large
external `phase_flag`-2 file is present).
**Dependency contract:** core paths are **numpy + stdlib only**; scipy / pandas /
pyproj / matplotlib are lazy-imported optional extras. `samoa_cm_tools` imports
`mbes_tools` **unchanged** (backward compatible).

## What was built, by capability

### Catalog (UPGRADE_PLAN §2) — verification data
- **Where:** `src/mbes_tools/catalog.py`; console `mbes-catalog`; outputs
  `docs/verification_manifest.{csv,json}` + `docs/VERIFICATION_DATA.md`.
- **Does:** path-agnostic inventory of `.all/.wcd/.kmall/.kmwcd` → per-file format,
  EM model, vessel, lat/lon, UTM zone, depth regime, datagram census,
  seabed-image flag, errors. Envelope-only scan; `--per-dir-limit`; gzip/zip
  detection.
- **Coverage (real):** EM124 ×2 (Samoa ship match), EM2040 in `.all` and `.kmall`
  (Samoa AUV match), EM304/302/122/710; UTM 9N–55N, both hemispheres,
  shallow→abyssal, all 4 formats.

### Capability A — `.all` backscatter parity (one pipeline for `.all` + `.kmall`)
- **Where:** `src/mbes_tools/all.py`, `src/mbes_tools/depth_modes.py`,
  `src/mbes_tools/backscatter/all_table.py`, `backscatter/apply.py`,
  `backscatter/table.py`.
- **Does:**
  - Datagram-**78 `N`** parser (`parse_raw_range_angle`,
    `iter_raw_range_angle_datagrams`): per-beam pointing angle + transmit sector +
    reflectivity. Datagram-**82 `R`** parser (`parse_runtime`,
    `iter_runtime_datagrams`): depth/ping mode. `XBeam.is_valid`.
    `iter_datagrams(types=…)` to skip unwanted datagram bodies.
  - `process_all_ping(...)` / `accumulate_all_file(...)`: join per-ping
    `X`+`N`+`R`+`P` into the existing `SoundingRecord`, reusing
    `aggregate_records` / ping-QC / `build_rows` / `normalize` unchanged.
  - `.all` write-back (`process_one_all`) patches `Y` seabed-image samples by
    sector correction.
  - `mbes-bs-table` / `mbes-bs-apply` dispatch by extension: `--format
    auto|kmall|all`; `.all` adds `--reflectivity-source xyz88|rawrange78`.
- **Depth modes:** `depth_modes.py` documents per-(model,format) maps —
  **EM2040/EM2045 `.all` runtime mode = frequency (200/300/400 kHz)**, not depth;
  general models = depth band (low 3 bits); `.all` ladder has no "Deeper" step
  (documented divergence from `.kmall`).
- **Verified:** N/R **byte-exact vs pyall** (EM2040); `XBeam.is_valid` count ==
  file `num_valid_detections`; `R` mode=1 → 300 kHz (matches filename); full
  EM2040 (456 pings) + EM302 (1,924 pings) ARCs sane; `.all` apply shifts only
  the corrected sector.

### Capability B — configurable backscatter source + per-beam reducer
- **Where:** `src/mbes_tools/beam_stat.py`, wired into `backscatter/table.py` and
  `all_table.py`; `kmall.py` now parses `SIcentreSample`.
- **Does:**
  - **Source A** = per-beam reflectivity (existing). **Source B** = per-beam
    seabed-image samples (`SIsample_desidB` / `.all` `Y`) reduced per beam.
  - `beam_stat` registry: `mean median std var mode trimmed_mean min max range
    count` + `p<NN>` percentiles; window around the bottom-detection (centre)
    sample. `std/var/range` double as within-beam **texture** features.
  - CLI: `--bs-source reflectivity|seabed_image`, `--beam-stat <list>`,
    `--si-window <half-width>`.
  - Multi-stat single read: `SoundingRecord.intensity_by_stat` → extra
    `avgIntensity_<stat>_dB` columns.
- **Verified:** **beam-level MBES_ARC parity exact** (`beam_stat` mean == MBES_ARC
  `sum/len/10` to ~1e-14 dB over 2,160 EM302 beams); Source A↔B correlate 0.85.

### Capability C — generalization (geography / instruments / robustness)
- **Where:** `src/mbes_tools/projection.py`, tolerant readers in `all.py`/`kmall.py`,
  `--on-error` in `backscatter/table.py`, depth-mode consolidation in `depth_modes.py`.
- **Does:**
  - **Projection:** `utm_epsg_from_lonlat`, `resolve_target_crs` — configurable
    EPSG **or** `auto` UTM/UPS from position; hemisphere/antimeridian/polar-safe;
    **no baked-in zone**. `SpatialProjector.from_spec(...)`.
  - **Robustness:** `iter_datagrams` / `iter_mrz_datagrams` accept
    `on_error="skip"` with bounded resync (+ `error_log`); `mbes-bs-table/-apply`
    default `--on-error skip`, per-file try/except, robustness summary → one bad
    ping/file never aborts a survey. gzip/zip reported, not crashed.
  - **Depth-mode maps** consolidated (`kmall_raw_to_calib`,
    `kmall_depth_mode_label`); `apply` delegates there.
- **Verified:** auto-UTM matches every manifest position and **Samoa
  (−171.8, −13.8) → EPSG:32702 (UTM 2S)**; resync recovers data past injected
  corruption; pipeline completes on a gzip'd `.all`; cross-model parse sweep
  clean (EM122/124/302/304/2040, both formats).

### Diagnostics — visual review suite
- **Where:** `src/mbes_tools/diagnostics.py`; console `mbes-diagnostics`.
- **Does:** renders, from real files (path-agnostic, matplotlib lazy): ARC +
  Lambertian balancing, Source A-vs-B, multi-stat + texture, `--si-window`,
  depth-mode split, port/starboard symmetry, swath-fan geometry + validity,
  auto-UTM map, MBES_ARC beam-level parity, corruption-recovery. Pure numeric
  helpers unit-tested; gated end-to-end render test.
- **QC finding to carry forward:** the EM2040 (Equinox) shows a real **~2.9 dB
  port/starboard imbalance** (EM302 ~1.4 dB).

## New public API & CLI (quick reference)
- **Modules added:** `mbes_tools.catalog`, `.depth_modes`, `.beam_stat`,
  `.projection`, `.diagnostics`, `.backscatter.all_table`; **D1:**
  `.wc_diagnostics` (water-column review suite; `.kmwcd`/`.wcd` readers
  validated).
- **Console scripts:** `mbes-bs-table`, `mbes-bs-apply` (both `--format`-dispatched),
  `mbes-bs-gui`, `mbes-catalog`, `mbes-diagnostics`, `mbes-wc-diagnostics`.
- **New table/apply flags:** `--format`, `--reflectivity-source`, `--bs-source`,
  `--beam-stat`, `--si-window`, `--on-error`.

## Committed fixtures
`tests/fixtures/`: `sample_equinox_em2040.all` (EM2040, Samoa AUV match),
`sample_tn447_em124.kmall` (EM124, Samoa ship match), pre-existing
`sample_nautilus.all` (EM302) and `sample_dpdk027.kmall` (EM304), plus the D1
water-column clips `sample_tn447_em124.kmwcd` (EM124 `#MWC`, `phase_flag` 0),
`sample_em2040_wc_phase1.kmall` (EM2040 `#MWC`, `phase_flag` 1 int8 phase), and
`sample_atlantis_em122.wcd` (EM122 `k`). Cut with
`tests/fixtures/clip_datagrams.py`. Full surveys (and the large `phase_flag`-2
`#MWC` file) stay external under `MBES_TEST_DATA_ROOT` / `MBES_MWC_PHASE2_FILE`.

## Conventions / decisions
- One capability per branch + PR (stacked, merge commits not squash); verify each
  against real data; heavy data out of the repo; integration tests gated on
  `MBES_TEST_DATA_ROOT` and skip when unset.
- Beam-angle sign: **positive = port** in both `.all` and `.kmall` — they compose.
- Aggregation averages **per-beam** values (equal weight per beam, consistent with
  Source A); MBES_ARC pools by sample count — beam reduction is identical,
  downstream weighting differs by design (documented).

## Known gaps / caveats
- **Samoa acceptance** pending download (re-run identical `table→normalize→apply`
  for ship EM124 `.kmall` + AUV EM2040 `.all`; auto-UTM already resolves to 2S/3S;
  document the normalization decision per system).
- **MBES_ARC aggregate `.mat` parity** not committed (scipy not in base env) —
  parity proven at the beam level instead.
- ~~`kmwcd.py` not yet validated~~ **DONE (D1, v0.7.0):** `kmwcd` (`#MWC`,
  `phase_flag` 0/1/2) and `wcd` (`k`) validated against real EM124/EM2040/EM122
  data by exact byte reconciliation; fixtures committed. Remaining water-column
  work is **products** (echograms / midwater detection), not reader validation.
- One literal EPSG remains: the optional geometry-mask CLI default `EPSG:32610`
  (must match the user's mask file; not survey projection).
- Archived gzip'd `.all` exist in the corpus — gunzip first.

## Remaining — Capability D (backlog, demand-driven)
- **D1:** ✅ reader validation done (v0.7.0) — `.wcd`/`.kmwcd` validated against
  real fixtures; first product `mbes_tools.wc_diagnostics`
  (`mbes-wc-diagnostics`) renders echogram / geo-wedge / phase review panels.
  Next: geo-referenced echogram **grids/mosaics** + plume/midwater detection.
  Samoa-relevant.
- **D2 (highest interoperability leverage):** native **GSF** reader feeding the
  same `SoundingRecord` pipeline; parity oracle = MB-System (format 121).
- **D3:** attitude (`A`/`n`/`#SKM`) + installation (`I`/`#IIP`/`#IOP`) parsers →
  precise re-georeferencing, grazing-angle ARC refinement, AUV motion correction,
  provenance.
