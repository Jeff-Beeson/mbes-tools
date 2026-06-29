# mbes-tools Upgrade Plan — top-level Claude Code task

**Mission.** Elevate **`mbes-tools`** into the robust, general, well-tested MBES foundation
for the Samoa critical-minerals project **and future surveys in other geographies and
instruments**. `samoa_cm_tools` (and later projects) depend on it. Every capability is
**verified against real data**, not just synthetic.

> Execute this **inside the `mbes-tools` repo** (its own git), on the Mac Studio, via Claude
> Code CLI. This file is the spec; copy it to `mbes-tools/docs/UPGRADE_PLAN.md` when you start.
> The Samoa-specific build (`samoa_cm_tools`) consumes the result — see
> `samoa_cm_tools/PLAN.md`, whose Phase 0 now depends on this work landing first.

---

## 0. Operating rules
- **Branch + PR per capability.** Tests green before merge; update `README.md` + a
  `CHANGELOG.md`; bump version (semver, currently v0).
- **Backward compatible.** `samoa_cm_tools` already imports `mbes_tools`; don't break the
  public API without a deprecation note.
- **Generalize, don't hard-code Samoa.** Configurable CRS, multi-EM-model, multi-geography.
  If a choice is survey-specific, it's a parameter, not a constant.
- **Heavy data stays out of the repo.** Commit only tiny clipped fixtures (made with
  `tests/fixtures/clip_datagrams.py`). Integration tests that need full files read a
  `MBES_TEST_DATA_ROOT` env var and **skip** cleanly when it's unset.

---

## 1. Capabilities to add (priority order)

### A. `.all` backscatter parity (so EM124 `.kmall` and EM2040 `.all` share one pipeline)
- Add to `mbes_tools.all`: a **datagram-78 `N`** parser (per-beam BeamPointingAngle +
  TransmitSectorNumber) and a **datagram-82 `R`** runtime parser (DepthMode) — both already
  on `all.py`'s TODO. Reference `MBES_ARC`/pyall for byte layouts.
- Add to `mbes_tools.backscatter`: `process_all_ping(...)` (analog of `process_mrz_datagram`)
  joining per ping `X`(depth/pos/reflectivity) + `N`(angle/sector) + `R`(mode) + `P`(nav)
  → the existing `SoundingRecord`; reuse `aggregate_records`/`qc`/`normalize` unchanged.
- Add an `.all` `apply` (write-back) path; make CLI (`mbes-bs-table`/`-apply`) dispatch by
  file extension. Map `.all` DepthMode IDs into the same calibration-mode space as `.kmall`.

### B. Configurable backscatter source + per-beam reducer (both formats)
- **Source A — per-beam reflectivity** (`reflectivity1/2_db` / `XBeam.reflectivity_db`).
- **Source B — seabed-image samples reduced per beam** (`SIsample_desidB` / `YBeam.samples`,
  ×0.1 dB) via a selectable statistic: **mean, median, std, mode, trimmed-mean, percentile,
  min/max/range, valid-count**, over a configurable **sample window** (full beam vs a window
  around the bottom detection / `centre_sample_number`).
- Build a numpy **`beam_stat`** reducer (pluggable registry); wire a
  `bs_source`/`beam_stat`/`si_window` selector into **both** front-ends; add an optional
  **multi-stat single-read pass** (`SoundingRecord.intensity_by_stat`) so several reducers can
  be compared without re-reading raw files. CLI: `--bs-source`, `--beam-stat` (list),
  `--si-window`. (A reference `beam_stat.py` can be lifted from
  `samoa_cm_tools/prototypes/` if generated.) **std/range double as within-beam texture
  features** worth carrying.

### C. Generalization for other geographies & instruments
- **EM-model coverage.** Make the readers robust across Kongsberg models the corpus contains
  (EM122/124/302/304/712/2040/2045…); confirm datagram-field variants per model.
- **CRS / geography-agnostic projection.** Configurable target EPSG **or** auto UTM-zone-from-
  position; safe across hemispheres, the antimeridian, and high latitudes. No baked-in zone
  (Samoa is UTM 2S/3S; other surveys differ).
- **Depth-mode normalization maps** per model/format (kmall manual ≥100 vs `.all` runtime IDs)
  kept in one documented table.
- **Robustness.** Skip unsupported/corrupt datagrams gracefully; stream large files; clear,
  actionable errors; never crash a survey on one bad ping.

### D. Backlog (note now, schedule later)
GSF support; water-column parity/products; attitude/installation parsers — add as project
needs pull them.

---

## 2. Verification data — assemble with Claude Code on the Mac (deferred by design)
**First Claude Code task before coding capability tests:** discover and catalog the real
verification corpus from the user's databases on the Mac. Do **not** hard-code paths here.
- Build a small **manifest** (CSV/JSON): path, format, EM model, vessel, geography, depth
  regime, datagram types present (from `mbinfo`), and whether it has seabed-image (`Y`/SI)
  data. Aim for coverage: **at least one EM124 and one EM2040** (Samoa-matched models) plus
  diversity in model, depth, and geography.
- **Samoa data is the primary target but gated on the download finishing.** Verify against the
  other datasets first so progress isn't blocked; add the Samoa `.kmall` (ship EM124) and
  `.all` (AUV EM2040) and re-run acceptance once they've landed.
- **Fixtures:** from a few representative files, cut tiny **committable** clips with
  `clip_datagrams.py` (one per model/format) for the repo's unit + integration tests; keep
  full surveys external under `MBES_TEST_DATA_ROOT`.
- *(A diverse multi-model corpus is known to exist in the user's folders — EM124, EM2040,
  EM122, multiple vessels — so the data to catalog is real; Claude Code resolves the actual
  database locations and the completed Samoa paths at run time.)*

---

## 3. Test strategy
- **Unit (synthetic / numpy):** reducers, angle binning, Lambertian normalization, depth-mode
  maps. No data dependency; always run.
- **Integration (clipped real fixtures):** for each committed fixture (per model & format),
  `parse → table → normalize → apply` runs clean with sanity asserts (angle coverage, BS
  ranges, sector/mode counts, non-empty output).
- **Cross-model / cross-geography (full files via `MBES_TEST_DATA_ROOT`):** no crashes;
  outputs sane across surveys; `.all` vs `.kmall` consistency where comparable.
- **Parity / regression:** compare the new `.all` ARC pipeline against `MBES_ARC`'s known
  outputs (e.g., a Langseth `.all`) as ground truth; snapshot the angular-response tables and
  diff on change.
- **Samoa acceptance (when data ready):** EM124 and EM2040 run the **identical** raw
  `table → normalize → apply`; both backscatter sources/reducers work on both; document the
  normalization decision per system.

---

## 4. Definition of done
- **Per capability:** branch merged; unit + integration tests green on **≥2 models including a
  Samoa-matched one**; README/CHANGELOG updated; public API documented.
- **Overall:** both `.kmall` and `.all` run the same configurable backscatter
  source/reducer/`table`/`normalize`/`apply`; verified on real multi-geography data; Samoa
  verified once available; `samoa_cm_tools` imports the upgraded library **unchanged**.

## 5. Sequencing
This upgrade is the **prerequisite / top priority**. `samoa_cm_tools` work (coregistration,
ARC component rasters, detection) resumes on top of the upgraded library; its `PLAN.md`
Phase 0 = "depend on the upgraded `mbes-tools`."

## 6. Driving with Claude Code CLI
1. `cd mbes-tools`; `claude`; `/init` (let it extend the repo's own `CLAUDE.md`).
2. *"Read docs/UPGRADE_PLAN.md. Start with the verification-data catalog task in §2 against my
   databases, then Capability A. One capability per branch/PR; keep tests green."*
3. One capability per PR; for each, report the diff, what real data it was verified on, and the
   tests added. Re-run Samoa acceptance when that data completes.
