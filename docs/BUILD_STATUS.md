# mbes-tools — build status & continuation notes

**Snapshot:** 2026-06-29. This is the handoff/status doc for the
`docs/UPGRADE_PLAN.md` work. Capabilities **A, B, C** of the plan are
implemented, verified against real data, and **merged to `main`** (v0.6.0);
**D** is the backlog described below.

> Read this together with `docs/UPGRADE_PLAN.md` (the spec),
> `docs/VERIFICATION_DATA.md` (the real-data corpus + how to regenerate the
> manifest), and `docs/DEPTH_MODES.md` (the depth/ping-mode maps).

---

## 1. What merged (all on `main`)

The work landed as a stack of per-capability PRs (merged in order with merge
commits; all feature branches deleted). For history, `git log --oneline --merges`.

| PR | Capability | Contents |
|----|-----------|----------|
| #1 | catalog (§2) | `mbes_tools.catalog`, manifest, `docs/` |
| #7 | A | `.all` backscatter parity (N/R parsers, `process_all_ping`, `.all` apply, CLI dispatch) — re-opened from the auto-closed #2 |
| #3 | B | configurable bs-source + `beam_stat` reducer (Source B), multi-stat |
| #4 | C | projection/auto-UTM, tolerant readers, depth-mode consolidation |
| #6 | diagnostics | `mbes_tools.diagnostics` visual review suite (`mbes-diagnostics`) |
| #5 | docs | this document |

> Note for future stacked-PR merges: deleting a PR's *base* branch auto-**closes**
> the child PR (GitHub does not retarget on base-deletion). Retarget children to
> `main` first (`gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f base=main` —
> `gh pr edit --base` was blocked here by a Projects-classic GraphQL error), then
> merge. Use merge commits, not squash, so the chain stays consistent.

**Resume a session:**
```bash
cd ~/code/mbes-tools
git checkout main && git pull
python -m pytest -q                                # expect 131 passed, 2 skipped
```
Start the next capability by branching off `main`. Keep one capability per
branch/PR; keep `pytest -q` green;
bump the version (`pyproject.toml` **and** `src/mbes_tools/__init__.py`) and add
a `CHANGELOG.md` entry per capability.

---

## 2. What is done

### Catalog (§2) — PR #1, v0.2.0
- `mbes_tools.catalog` (`mbes-catalog`): path-agnostic inventory of
  `.all/.wcd/.kmall/.kmwcd` → manifest (path, format, EM model, vessel, lat/lon,
  UTM zone, depth regime, datagram census, seabed-image flag, errors).
  Envelope-only scan; `--per-dir-limit`; gzip/zip detection.
- `docs/verification_manifest.csv|json` + `docs/VERIFICATION_DATA.md`.
- **Coverage:** EM124 ×2 (Samoa ship match), EM2040 in `.all` **and** `.kmall`
  (Samoa AUV match), EM304/302/122/710; UTM 9N–55N, both hemispheres,
  shallow→abyssal, all four file formats.

### Capability A — PR #2, v0.3.0 — `.all` backscatter parity
- `mbes_tools.all`: datagram **78 `N`** (raw range/angle: per-beam pointing angle
  + tx sector + reflectivity) and **82 `R`** (runtime: depth/ping mode) parsers;
  `XBeam.is_valid`; `iter_datagrams(types=...)`.
- `mbes_tools.depth_modes`: per-(model, format) mode maps; EM2040/2045 `.all`
  runtime mode = **frequency** (200/300/400 kHz), not depth band.
- `mbes_tools.backscatter.all_table.process_all_ping` / `accumulate_all_file`:
  join per-ping `X`+`N`+`R`+`P` into the existing `SoundingRecord`, reusing
  `aggregate_records` / ping QC / `build_rows` / `normalize` unchanged.
- `.all` apply write-back (`process_one_all`, patches `Y` samples).
- `mbes-bs-table` / `mbes-bs-apply` **dispatch by extension** (`--format`).
- **Verified:** N/R byte-exact vs pyall on EM2040; `XBeam.is_valid` count ==
  file's `num_valid_detections`; full EM2040 (456 pings) + EM302 (1924 pings)
  tables sane; `.all` apply shifts only the corrected sector.

### Capability B — PR #3, v0.4.0 — bs-source + per-beam reducer
- `mbes_tools.beam_stat`: numpy reducer registry (`mean median std var mode
  trimmed_mean min max range count` + `p<NN>`), windowed around the centre
  sample. `std/var/range` = texture features.
- **Source A** (per-beam reflectivity) and **Source B** (seabed-image reduced
  per beam) in both front-ends: `--bs-source`, `--beam-stat` (list),
  `--si-window`. `.kmall` reduces `SIsample_desidB` (now parses
  `SIcentreSample`); `.all` reduces the matched `Y` datagram.
- Multi-stat single read: `SoundingRecord.intensity_by_stat` →
  `avgIntensity_<stat>_dB` columns.
- **Verified:** beam-level MBES_ARC parity exact (`beam_stat` mean == MBES_ARC
  `sum/len/10` to ~1e-14 dB over 2,160 EM302 beams); full EM2040 Source A↔B
  corr 0.85.

### Capability C — PR #4, v0.5.0 — generalization
- `mbes_tools.projection`: `utm_epsg_from_lonlat` / `resolve_target_crs` —
  auto UTM/UPS from position, hemisphere/antimeridian/polar safe; no baked-in
  zone. `SpatialProjector.from_spec`.
- Tolerant readers: `on_error="skip"` + bounded resync in `all.iter_datagrams`
  and `kmall.iter_mrz_datagrams` (+ `error_log`). `mbes-bs-table/-apply` default
  `--on-error skip`, per-file try/except, robustness summary.
- Depth-mode maps consolidated in `depth_modes` (`kmall_raw_to_calib`,
  `kmall_depth_mode_label`); `apply` delegates.
- **Verified:** auto-UTM matches every manifest position + Samoa→EPSG:32702 (2S);
  resync recovers past injected corruption; pipeline completes on a gzip'd `.all`;
  cross-model parse sweep clean (EM122/124/302/304/2040, both formats).

---

## 3. Build / test status
- **`python -m pytest -q` → 125 passed, 1 skipped** at the tip (PR #4).
- The skip is the gated cross-model test; run it with
  `MBES_TEST_DATA_ROOT=<dir> python -m pytest -k cross_model`.
- **Environment:** base env has **numpy + python-dotenv only**; scipy / pandas /
  pyproj / matplotlib are **NOT installed**. Core code paths are numpy + stdlib;
  heavy deps are lazy-imported (geometry mask/projection→pyproj, flat filter/grid
  sampling→scipy, apply CSV→pandas, GUI→matplotlib). Keep it that way so unit
  tests run without the extras.

---

## 4. Conventions / rules of the road (follow these)
- **One capability per branch + PR**, stacked; tests green before opening; bump
  version in both places; update `CHANGELOG.md` + `README.md`.
- **Verify against real data** and state what was used (the corpus is in
  `docs/VERIFICATION_DATA.md`; don't hard-code dataset paths in code).
- **Heavy data stays out of the repo**; commit only tiny clips via
  `tests/fixtures/clip_datagrams.py`. Integration tests on full files read
  `MBES_TEST_DATA_ROOT` and **skip** when unset.
- **Backward compatible:** `samoa_cm_tools` imports `mbes_tools` unchanged; new
  reader/pipeline params are keyword-only with defaults that preserve old
  behavior (Source A default, `on_error="raise"` default in the library).
- **Beam-angle sign:** both `.all` (N) and `.kmall` (#MRZ) use **positive = port**
  — they compose directly; do not flip.
- **Depth-mode space:** all maps live in `mbes_tools.depth_modes`. `.all` general
  ladder has no "Deeper" step, so a `.all` "Very Deep" ≠ `.kmall` "Very Deep"
  integer (documented in `docs/DEPTH_MODES.md`); EM2040 `.all` mode = frequency.
- **Aggregation weighting:** the table averages **per-beam** values (equal weight
  per beam, consistent with Source A). MBES_ARC pools by sample count — beam-level
  reduction is identical, downstream weighting differs by design.

---

## 5. Known gaps / caveats (carry forward)
- **Samoa acceptance is pending the data download.** Once the ship EM124 `.kmall`
  and AUV EM2040 `.all` land: add them to the manifest, cut fixtures if useful,
  and run the identical `table → normalize → apply` for both; auto-UTM already
  resolves Samoa to UTM 2S/3S. Document the normalization decision per system.
- **MBES_ARC aggregate `.mat` parity not committed** — scipy isn't in the base
  env, so parity was proven at the **beam level** (exact) instead of loading the
  QCorr `_r78_data.mat`. If a full aggregate snapshot test is wanted, add scipy
  to the dev extra and compare per-(mode, angle) ARC, accounting for the
  per-beam-vs-pooled weighting difference (or add a `pooled` aggregation option).
- **Archived gzip'd `.all`** exist in the corpus (e.g. one Nautilus file);
  readers reject them and the catalog/pipeline report them — gunzip first.
- **Geometry-mask CLI default CRS is `EPSG:32610`** (Monterey). It must match the
  user's mask file; it is the one remaining literal EPSG and is a *mask* default,
  not survey projection (which is position-derived via `mbes_tools.projection`).
- `kmwcd.py` is **not yet validated against a real `.kmwcd` fixture** (see D).

---

## 6. Capability D — direction & next steps (backlog)

D is demand-driven (`UPGRADE_PLAN §1.D`: "add as project needs pull them").
Readers already **skip** all D datagram types gracefully, so D is purely
additive. Recommended order and concrete first steps below.

### D1 (recommended first — lowest effort, data in hand): validate water column
- **Goal:** finish/validate the existing `wcd` (`.wcd`, `k` datagram) and
  `kmwcd` (`.kmwcd`, `#MWC`) readers against **real fixtures**, then reach
  "parity" with the backscatter path.
- **Data in hand:** TN447 has 15 `.kmwcd` (EM124) under `MBES_TEST_DATA_ROOT`;
  the corpus has `.wcd` (Atlantis EM122). Cut tiny fixtures with
  `clip_datagrams.py --target-type '#MWC'` / `--target-type k`.
- **First steps:**
  1. Cut `sample_tn447_em124.kmwcd` + a `.wcd` clip; add fixture tests asserting
     `#MWC`/`k` beam counts, sample-array lengths, phase-flag handling.
  2. Validate `kmwcd` phase-sample handling (`phase_flag` 1/2) against the real
     file (its docstring flags this as unvalidated).
  3. Then **products** (separate PR): geo-referenced echograms / beam-fan
     amplitude grids using `mbes_tools.projection`; optional midwater/plume
     anomaly detection. Reuse sector/beam-angle handling from the backscatter path.
- **Why / trigger:** Samoa hydrothermal-plume / midwater detection and bottom-
  detection QC. Validation is unblocked **now**.

### D2 (highest leverage for the general mission): GSF support
- **Goal:** native `mbes_tools.gsf` reader so processed data (CARIS/Qimera/
  archives) flows through the same `SoundingRecord` pipeline.
- **First steps:**
  1. Reader for GSF records: swath bathy ping (depth/across/along/beam angle),
     plus backscatter — "BRB intensity" / per-beam imagery (snippets) → Source B
     via `beam_stat`; per-beam amplitude → Source A.
  2. Add a `process_gsf_ping` analog of `process_all_ping`; wire `--format gsf`
     into the table/apply dispatch.
  3. Cross-check against `mbsystem.py` (MB-System reads GSF as format 121) on a
     real GSF file as the parity oracle.
- **Why / trigger:** many surveys/archives ship only GSF; needed to compare
  against CARIS/Qimera products and to support "other geographies/instruments".

### D3: attitude / installation parsers
- **Goal:** native parsers for attitude (`.all` `A`/`n`; `.kmall` `#SKM`) and
  installation (`.all` `I` + the already-parsed `R`; `.kmall` `#IIP`/`#IOP`).
- **First steps:**
  1. Add dataclasses + parsers alongside the existing `P/X/Y/N/R` (and #MRZ)
     parsers; expose `iter_*` helpers.
  2. Structure `#IIP`/`I` install text (lever arms, mount angles, serials, SVP
     source) — supersedes the catalog's regex EM-model scrape.
  3. Consumers (later): precise re-georeferencing + grazing-angle ARC refinement
     (with `projection`), AUV motion correction, provenance in the manifest.
- **Why / trigger:** re-georeferencing, ray-bending/grazing-angle ARC, AUV
  (Samoa EM2040) motion correction, multi-sensor fusion, GSF export.

### Suggested sequencing
1. **D1 water-column validation** (unblocked, small, Samoa-relevant).
2. **D2 GSF** (broadest interoperability payoff).
3. **D3 attitude/installation** (enables precise re-georeferencing; pulls in
   when ARC refinement or GSF export needs it).

### Next-session checklist
- [ ] Confirm PR stack #1→#4 still open / merge state; rebase if any merged.
- [ ] Pick a D item; branch off the current tip (or `main` if merged).
- [ ] Cut any needed fixtures with `clip_datagrams.py`; keep full data external.
- [ ] Implement reader/parser → pipeline wiring → CLI; keep core numpy+stdlib.
- [ ] Add unit (synthetic) + integration (fixture) + gated full-file tests.
- [ ] Verify against the real corpus; record what was used.
- [ ] Bump version, update CHANGELOG/README, open a stacked PR.

---

## 7. Reference pointers (machine-local; not committed)
- Reference readers cross-checked during A/B: pyall at
  `/mnt/d/Cowork_OS/_WSL_Staging/projects/_modules/pyAllConditioner-master/pyall.py`;
  valschmidt KMALL at `/mnt/d/Cowork_OS/_WSL_Staging/projects/kmall-master/KMALL/kmall.py`.
- MBES_ARC reference ARC + QCorr `.mat`/`.Backscatter` outputs under
  `/mnt/d/Cowork_OS/_WSL_Staging/projects/MBES_ARC/` (EM2040 Equinox QCorr files
  are the parity reference).
- Full corpus locations and per-file gotchas: `docs/VERIFICATION_DATA.md`.
