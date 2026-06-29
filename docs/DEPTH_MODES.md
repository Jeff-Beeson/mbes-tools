# Depth-mode normalization maps (one documented table)

Backscatter angular response is grouped by **depth mode** (a.k.a. ping mode).
The encoding differs by file format and EM model, so `mbes_tools.depth_modes`
is the single place that maps each encoding onto a canonical id + label. This
page documents those maps; the code is the source of truth.

## Canonical ladder (follows the KMALL auto-mode numbering)

| id | label        |
|----|--------------|
| 0  | Very Shallow |
| 1  | Shallow      |
| 2  | Medium       |
| 3  | Deep         |
| 4  | Deeper       |
| 5  | Very Deep    |
| 6  | Extra Deep   |
| 7  | Extreme Deep |

Calibration-file mode number (used by `backscatter.apply`) = `id + 1`
(Shallow = 2, Medium = 3, …). KMALL manual modes carry a +100 offset that is
normalized away before reaching here (`kmall.normalize_depth_mode`).

## `.kmall` (#MRZ pingInfo `depthMode`)

Stored directly as the canonical id above (after manual +100 removal).

## `.all` (`R` runtime `mode` byte)

The `mode` byte is bit-encoded and **model-specific**.

### General EM models (EM122 / EM124 / EM300 / EM302 / EM304 / EM710 / EM712 …)

Depth band from the low 3 bits (`mode & 0x07`). Note there is **no "Deeper"
step** — so a `.all` "Very Deep" (id 4) is not the same integer as a `.kmall`
"Very Deep" (id 5). This is intentional and documented rather than silently
coerced; within one survey/sonar it is self-consistent.

| `mode & 0x07` | id | label        |
|---------------|----|--------------|
| 0             | 0  | Very Shallow |
| 1             | 1  | Shallow      |
| 2             | 2  | Medium       |
| 3             | 3  | Deep         |
| 4, 6          | 4  | Very Deep    |
| 5, 7          | 5  | Extra Deep   |

### EM2040 / EM2045 (frequency, not depth)

The low bits select the operating **frequency**. The id space is distinct from
the depth ladder; the label disambiguates.

| bits        | id | label  |
|-------------|----|--------|
| (default)   | 0  | 200kHz |
| bit0        | 1  | 300kHz |
| bit1        | 2  | 400kHz |

Verified on real data: `0002_..._Equinox_2040_300kHz.all` reports `mode = 1`,
which decodes to **300 kHz**, matching the filename and the pyall reference.

## Why this matters for parity

EM2040 (`.all`, Samoa AUV) and EM124 (`.kmall`, Samoa ship) flow through the
same `table → normalize → apply` code. They do not need identical depth-mode
integers across models — they need the **same code path** and a documented,
model-aware mode key. That is what this module provides.
