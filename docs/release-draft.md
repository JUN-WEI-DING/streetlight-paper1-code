# EIAR reproducibility release — author-review draft

Status: prepared for author review; repository remains private. This document
is a release draft; the data-sharing basis and attribution are described below.
The reviewed candidate is identified by its exact Git commit in
`dist/release-review.md`; a public release tag has not been created.

## Release description

This independent repository contains the implementation and study configuration
for the dispatch-aware PV–battery streetlight analysis, a synthetic example,
and reproducibility entry points. Original code and documentation use MIT;
third-party data retain their source-specific terms.

The local companion data archive contains 160 files: processed municipal PAR,
AEF and power inputs, population weights, capacity and sensitivity results, and
13 numerical figure tables. The capacity panel has 99,000 rows (4,500 designs
across 22 municipalities). Raw public-source files and licensed LCA process
databases are not distributed in this archive. NASA weather and county geometry
are obtained separately using the supplied commands.

Quick replay reuses capacity and scenario intermediates to recompute selection
and downstream results. Full analysis reruns the documented suite from processed
inputs. Neither route is a verified reconstruction from all original observations
or a licensed SimaPro process-model rebuild. Commands and validation scope are
in [data-bundle.md](data-bundle.md).

## Review files

All generated files are under the existing ignored `dist/` directory:

- `paper1-code.tar.gz`: source snapshot from the candidate Git commit.
- `paper1-data-review.tar.gz`: complete local review data package; publication
  awaits the author-approved release.
- Each archive has its own `.sha256` checksum file.
- `release-review.md`: exact candidate commit, data-source commit, archive sizes,
  checksums and validation summary for the author's review.

Extract the source archive, install with `uv sync --locked --extra pv-benchmark`,
and follow the quick/full commands in `docs/data-bundle.md`. Both routes require
the complete data package plus separately fetched weather and geometry. The
synthetic example runs without study observations.

## PAR sharing basis

The project adopts the [JAXA Terms of Use of Research
Data](https://earth.jaxa.jp/en/data/policy/index.html) as the basis for sharing
the processed municipal PAR tables and derived research results, with attribution
and processing disclosure. This is the authors' policy interpretation and does
not represent individual JAXA approval. A separate permission request is not a
release prerequisite under this adopted approach.

Retain both municipal PAR and imputation-flag tables. Credit the original PAR
product to JAXA's P-Tree System and identify the authors' municipal aggregation
and missing-value treatment at 10-minute resolution. Include the policy link
and the notice in `THIRD_PARTY_NOTICES.md` with the data archive. Raw public-source
products remain available from their providers and are not bundled. Data terms
remain separate from the code's MIT license.

## Publication handoff

After the author reviews these files, select a
release tag for the reviewed commit and publish only the approved assets. Update
private/local wording in the README and replace manuscript, SI and R2C14 access
placeholders with the actual accessible URLs and fixed version. Verify access
without authentication. Keep data licensing separate from the code's MIT license.
Graphical-abstract revision and final submission proofing are separate remaining
editorial tasks; this release draft does not mark them complete.
