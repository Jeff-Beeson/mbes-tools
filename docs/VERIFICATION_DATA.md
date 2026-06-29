# Verification-data catalog (UPGRADE_PLAN §2)

This documents the real Kongsberg multibeam corpus used to verify `mbes-tools`
capabilities against actual data (not just synthetic fixtures), and how to
regenerate the manifest. Heavy data stays **out of the repo**; only the small
manifest and tiny clipped fixtures are committed.

## Regenerating the manifest

The catalog tool is path-agnostic — pass the roots (or individual files) to
scan. It walks `.all`/`.wcd`/`.kmall`/`.kmwcd` by datagram envelope (bodies are
not read for the type census, so it is cheap even on large files), decodes a few
datagrams per file for model/geography/depth, and writes CSV/JSON.

```bash
# Scan whole roots, sampling a few files per directory:
python -m mbes_tools.catalog ~/MGL1701 ~/RR1413 ~/projects ~/code /mnt/d \
    --per-dir-limit 3 -o docs/verification_manifest.csv --json docs/verification_manifest.json

# Or catalog a curated set of representative files (what produced the committed manifest):
python -m mbes_tools.catalog <file1> <file2> ... --per-dir-limit 0
```

Notes:
- `--per-dir-limit N` keeps at most N files per directory **and** extension, so a
  survey of hundreds of near-identical lines yields a small representative
  manifest. `0` keeps everything.
- `--max-scan-bytes` caps the datagram-type census on very large files; the
  `scan_truncated` column flags when the cap was hit (model/geography are still
  captured because those datagrams sit near the start of the file).
- One unreadable/corrupt/compressed file never aborts the run — the reason lands
  in the `error` column.

## Committed manifest

`docs/verification_manifest.csv` / `.json` — generated from a curated
representative set (smallest file per dataset). Columns: `path, format,
mbsystem_format_id, em_model, vessel, latitude, longitude, utm_zone,
depth_regime, median_depth_m, datagram_types, has_seabed_image,
n_datagrams_scanned, scan_truncated, error`.

### Coverage summary (representative set)

| EM model | Format(s)        | Example dataset / vessel              | Depth regime | UTM (geography)         |
|----------|------------------|---------------------------------------|--------------|-------------------------|
| EM124    | .kmall, .kmwcd   | TN447 / R/V Thomas G. Thompson        | abyssal      | 52N (W Pacific)         |
| EM124    | .kmall           | FKt230303 / R/V Falkor (too)          | abyssal      | 21N (Atlantic)          |
| EM2040   | .all             | Equinox_2040_300kHz / Fugro Equinox   | shelf/slope  | 49N                     |
| EM2040   | .all             | FA2806 / NOAA (kluster sample)        | shallow      | 10N                     |
| EM2040   | .kmall           | kmall-master HiRes/LowRes subset      | shallow      | 9N                      |
| EM2040   | .kmall           | ASVBEN (autonomous surface vehicle)   | shallow      | 17N                     |
| EM304    | .kmall           | DPDK027 / R/V David Packard (MBARI)   | deep/abyssal | 10N (Monterey)          |
| EM302    | .all             | Fugro Brasilis / Searcher / FK180824  | shelf→deep   | 15N / 24S / 10N         |
| EM302    | .all (gzip'd)    | E/V Nautilus (MBES_ARC test data)     | —            | — (see caveat)          |
| EM122    | .all             | MGL1701 / R/V Marcus G. Langseth      | deep         | 18S                     |
| EM122    | .all             | RR1413 / R/V Roger Revelle            | deep         | 55N                     |
| EM122    | .wcd             | Atlantis (water column)               | —            | —                       |
| EM710    | .all             | PS01                                  | shallow      | 31N                     |

**Plan §2 requirement (≥1 EM124 and ≥1 EM2040) is exceeded:** two EM124 datasets
(both Samoa-matched ship sonar) and four EM2040 datasets across **both** `.all`
and `.kmall` (Samoa-matched AUV sonar). Geographic diversity spans UTM zones
9N–55N in both hemispheres; depth regimes span shallow → abyssal; all four
Kongsberg file formats are represented.

### Full corpus (approximate file counts found under the roots)

- `~/MGL1701/.../MB/em122` — ~200 `.all`, EM122, R/V Langseth (cruise MGL1701)
- `~/RR1413/rawmultibeam` — ~200 `.all`, EM122, R/V Revelle (cruise RR1413)
- `~/projects/Monterey_Canyon_MBES/DPDK027/...` — ~198 `.kmall`, EM304, R/V David Packard
- `/mnt/d/temp/TN447.../EM124.Data` — 15 `.kmall` + 15 `.kmwcd`, EM124, R/V Thompson
- `/mnt/d/.../MBES_ARC/test_data` — EM302 (Nautilus, Falkor) and EM2040 (Equinox) `.all`
- `/mnt/d/.../kmall-master`, `Bard_ChatGPT3` — EM2040 + EM124 `.kmall` samples
- `/mnt/d/.../OCS_Hydro/kluster` — EM2040 `.all` (FA2806)
- `/mnt/d/temp/PS01` — EM710 `.all`

## Caveats / real-world notes

- **Gzip'd `.all`:** `MBES_ARC/test_data/Nautilus_Data/0000_20160415_011924_Nautilus.all`
  is gzip-compressed despite the `.all` extension. The reader rejects it cleanly;
  the catalog flags it (`gzip-compressed ...`). Decompress before use, or use a
  different EM302 line (Brasilis/Searcher/FK180824 parse fine).
- **Vessel names** are a best-effort guess from the filename's trailing token,
  with a small alias table. Files whose name has no vessel token (e.g. TN447
  `0184_...kmall`) show a blank vessel — the dataset/cruise is known from the
  directory (TN447 = R/V Thomas G. Thompson; FKt230303 = R/V Falkor (too)).
- **Water-column files** (`.wcd`, `.kmwcd`) carry no seabed-image / sounding
  datagrams, so `has_seabed_image=0` and geography/depth are blank — expected.

## `.all` datagram types relevant to backscatter (Capability A)

The representative `.all` files carry the datagram set the `.all` backscatter
pipeline needs:

- `P` (80) position · `X` (88) XYZ depth · `Y` (83) seabed image ·
  `N` (78) raw range & angle · `R` (82) runtime parameters.

e.g. the Equinox EM2040 line reports `N:456;P:184;R:21;X:456;Y:456` — full
per-ping `X`+`N`+`R`+`Y`+`P` coverage for parity work.

## Samoa (deferred)

The Samoa critical-minerals data (ship EM124 `.kmall`, AUV EM2040 `.all`) is the
primary acceptance target but is **gated on its download finishing**. Verify
against the datasets above first; add the Samoa paths to the manifest and re-run
acceptance once they land. Full survey files are read via `MBES_TEST_DATA_ROOT`
and integration tests skip cleanly when it is unset (see UPGRADE_PLAN §0).
