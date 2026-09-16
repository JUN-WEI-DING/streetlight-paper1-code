# EIAR reproducibility release — author-review draft

Status: prepared for author review; repository remains private. This document
is a release draft; the data-sharing basis and attribution are described below.
The reviewed candidate is identified by its exact Git commit in
`dist/release-review.md`; a public release tag has not been created.

## Proposed fixed release

- Repository: `JUN-WEI-DING/streetlight-paper1-code`.
- Proposed tag: `v1.0.0` (research reproduction snapshot; Python package remains `0.1.0`).
- Title: `Paper 1 reproducibility materials v1.0.0`.
- Target: exact code commit and asset SHA-256 values in `dist/release-review.md`.
- Upload exactly four assets: `paper1-code.tar.gz`, `paper1-code.tar.gz.sha256`,
  `paper1-data-review.tar.gz`, `paper1-data-review.tar.gz.sha256`.
- Keep the current asset names so documented replay commands remain valid.
- `release-review.md` is the local author handoff, not an upload asset.

Planned URLs (not yet published):

- Release: https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/tag/v1.0.0
- Data: https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/download/v1.0.0/paper1-data-review.tar.gz
- Code: https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/download/v1.0.0/paper1-code.tar.gz

## Release description

This independent repository contains the implementation and study configuration
for the dispatch-aware PV–battery streetlight analysis, a synthetic example,
and reproducibility entry points. Original code and documentation use MIT;
third-party data retain their source-specific terms.

The local companion data archive contains 160 scientific/provenance files, plus
`bundle.json` and a third-party notice: processed municipal PAR,
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

## Proposed GitHub release text

The following text is prepared for publication after author approval:

> Reproducibility materials for *A dispatch-aware framework for resource-efficient
> distributed electrification: A case study of photovoltaic-battery streetlights*.
>
> The source archive contains the implementation, study configuration, synthetic
> example and replay commands. The companion data archive contains processed
> municipal inputs, the 99,000-row capacity panel, sensitivity results and 13
> numerical figure tables. Download both archives and their SHA-256 files.
>
> Quick replay reuses analysis intermediates; the full route reruns the documented
> analyses from processed inputs. Follow `docs/data-bundle.md` in the source
> archive to stage the data and separately obtain NASA weather and county
> boundaries. Raw satellite products and licensed LCA process databases are not
> included; this is not a complete raw-observation or SimaPro reconstruction.
>
> Original code uses MIT. Data retain their source-specific terms. Processed PAR
> carries JAXA/P-Tree attribution and the project's adopted JAXA research-data
> policy basis. See `THIRD_PARTY_NOTICES.md` and `bundle.json` for provenance.

## Publication handoff

After author approval, publish the reviewed commit as `v1.0.0` with the four
listed assets and verify anonymous downloads and checksums. Any source change
after this review requires refreshed source-archive checksums and a new target
commit before tagging. Update publication-status wording as appropriate and
replace manuscript, SI and R2C14 access
placeholders with the actual accessible URLs and fixed version. Verify access
without authentication. Keep data licensing separate from the code's MIT license.
Graphical-abstract revision and final submission proofing are separate remaining
editorial tasks; this release draft does not mark them complete.
