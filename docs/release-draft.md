# EIAR reproducibility release — author-review draft

Status: prepared for author review; repository remains private. This document
is a release draft, not a declaration that data redistribution is authorized.
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
- `paper1-data-review.tar.gz`: complete local review data package; not for upload
  until the data-sharing scope below is resolved.
- Each archive has its own `.sha256` checksum file.
- `release-review.md`: exact candidate commit, data-source commit, archive sizes,
  checksums and validation summary for the author's review.

Extract the source archive, install with `uv sync --locked --extra pv-benchmark`,
and follow the quick/full commands in `docs/data-bundle.md`. Both routes require
the complete data package plus separately fetched weather and geometry. The
synthetic example runs without study observations.

## PAR sharing decision

Official pages checked on 2026-09-16:

- [P-Tree registration](https://www.eorc.jaxa.jp/ptree/registration_top.html)
  permits registered access, states third-party redistribution restrictions,
  and asks users to contact the secretariat before publicly releasing research
  results. It retains non-profit-use conditions for pre-February-2026 data.
- [P-Tree terms, Sections 6 and 10](https://www.eorc.jaxa.jp/ptree/terms.html)
  distinguish JAXA geophysical products from JMA standard data and specify
  attribution when reporting research.
- [JAXA site policy](https://global.jaxa.jp/policy.html) gives general use and
  attribution conditions. It does not expressly settle this municipal-table case.

The remaining question is permission to share our processed tables and derived
results, not whether readers can register to obtain raw PAR. The pages alone do
not establish permission or a categorical prohibition for these derivatives.
Keep the complete review archive local pending clarification. Removing only the
two PAR tables would not automatically clear all other PAR-derived results.

If municipal PAR cannot be supplied, both documented replay routes lose their
PV/calibration/dispatch input and cannot reproduce the full analysis as written.
Readers could inspect supplied results and code, but would need independently
prepared matching PAR to run these routes. Do not describe such a reduced archive
as a complete reproduction package. No reduced archive has been substituted.

### Inquiry draft — not sent

Recipient: P-Tree Secretariat, using the contact listed on its official page.
Subject: Sharing processed municipal PAR time series and derived research results

Dear P-Tree Secretariat,

We are preparing reproducibility materials for a non-commercial research paper
on photovoltaic–battery streetlights in Taiwan using 2024 Himawari-derived PAR
obtained through P-Tree. We would like to clarify the permitted scope of public
sharing before releasing the materials.

The proposed package would contain a processed PAR table at 10-minute resolution
for 22 municipalities and an accompanying imputation-flag table. It would not
contain the original satellite files or gridded product. We also propose to share
model outputs, including PV–battery capacity-search results, sensitivity summaries
and numerical figure tables derived in part from the PAR inputs.

Could you please confirm whether these processed municipal tables and derived
model results may be made publicly downloadable with a research-code repository?
If permitted, please specify the required attribution, use conditions or notices.
If the municipal time series cannot be shared, please clarify whether the derived
model-output tables may be released while readers obtain PAR through individual
P-Tree registration.

We can provide a description of the processing method and product details if
needed. Thank you for your guidance.

[Corresponding author name and affiliation]

## Publication handoff

After the author reviews these files and the sharing scope is resolved, select a
release tag for the reviewed commit and publish only the approved assets. Update
private/local wording in the README and replace manuscript, SI and R2C14 access
placeholders with the actual accessible URLs and fixed version. Verify access
without authentication. Keep data licensing separate from the code's MIT license.
Graphical-abstract revision and final submission proofing are separate remaining
editorial tasks; this release draft does not mark them complete.
