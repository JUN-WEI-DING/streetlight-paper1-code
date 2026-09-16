# Paper 1 reproducibility release v1.0.0

[Public release and downloads](https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/tag/v1.0.0)

Fixed code commit: `e42c221b07a66b2d9d181792837c5e488e777de2`.
Data-source commit: `f3854c3ce52cdec348691024059cd98a267c2104`.
The research release is v1.0.0; the Python package version is 0.1.0.

## Download assets

- `paper1-code.tar.gz`: 92 source files from the fixed code commit.
- `paper1-data-review.tar.gz`: 160 scientific/provenance payloads, `bundle.json`
  and a third-party notice.
- Each archive has its own `.sha256` file; verify before extraction.

The filenames and archive contents preserve the author-reviewed snapshot.
Private/local status wording inside that snapshot describes preparation; the
release and all four attachments are now public. The default branch documentation
reflects publication. Use the tagged code or source archive for reproducibility.

The data archive retains municipal PAR and imputation flags, AEF and power inputs,
population weights, the 99,000-row capacity panel, sensitivity results and 13
numerical figure tables. Three provenance JSON files replace six author-machine
paths with portable labels and preserve their original hashes. Scientific values
are unchanged. Raw public-source products and licensed LCA process databases are
not bundled. Follow [the bundle guide](data-bundle.md) for quick replay or the
analysis-suite rerun and separate weather/boundary downloads.

## Source terms

Original code and documentation use MIT; data retain source-specific terms.
Processed PAR carries JAXA/P-Tree attribution and processing descriptions under
the project's adopted [JAXA research-data policy](https://earth.jaxa.jp/en/data/policy/index.html)
interpretation, not individual JAXA approval. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Verification

GitHub-reported SHA-256 digests for all four attachments match the reviewed
local files. Anonymous access was checked with full downloads of the source
archive and checksum files, and a matching prefix download of the data archive.
The slow full data re-download was stopped; its full-file integrity check uses
the GitHub-reported digest. The tag points to the reviewed code commit.
Twelve focused tests and both archive-staging routes passed before publication.
Publication does not add a fresh numerical analysis run or establish complete
raw-observation or licensed SimaPro reconstruction.
