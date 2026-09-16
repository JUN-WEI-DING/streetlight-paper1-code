# Third-party materials

The MIT grant covers original code and documentation, not external data,
commercial databases, or dependency packages. Each Python dependency retains
its own license; installed versions are recorded in `uv.lock`.

No Himawari observations, Taipower time series, administrative boundary files,
municipal population tables, weather cache, or SimaPro/ecoinvent process exports
are tracked in this source repository. The companion data bundle in public release v1.0.0
contains processed input snapshots, intermediates and reference results. Raw public-source files are obtained from their providers.
The project uses the JAXA research-data policy as the basis for sharing processed
municipal PAR, as described below.
See [data origins and publication status](docs/data-bundle.md).
Obtain external inputs under their applicable terms. The NASA POWER weather adapter
only downloads when explicitly requested with its fetch option.

Study settings include published emission-factor assumptions and calibrated
aggregate hardware coefficients used by the research. These are numerical model
parameters, not redistributed process databases or a license to SimaPro or
ecoinvent. The Python hardware calculation cannot reconstruct the underlying
licensed process model. Source comments identify origins where available; paths
to private supporting evidence do not indicate that those files are included.

The region-to-city JSON is the study's allocation configuration, not a boundary
map or an observational dataset. Artificial inputs in the example and tests are
explicitly synthetic and carry no observational provenance.

## Processed PAR attribution and terms

The original PAR research product was supplied by the P-Tree System, Japan
Aerospace Exploration Agency (JAXA). The authors aggregated it to municipal
time series at 10-minute resolution and prepared missing-value treatment and
accompanying imputation flags. These tables are author-processed derivatives,
not original JAXA gridded files or JAXA-endorsed products. The preprocessing
entry points and input schemas are described in `docs/reproduction.md`.

The project's adopted sharing basis is the [JAXA Terms of Use of Research
Data](https://earth.jaxa.jp/en/data/policy/index.html), including source attribution
when distributing derivatives. This is the authors' policy interpretation,
not an individual approval from JAXA. Retain this attribution, the processing
description and the policy link when reusing the tables. The code's MIT license
does not replace source-data terms. Raw products are obtained separately
through P-Tree under its access terms.
