# Third-party materials

The MIT grant covers original code and documentation, not external data,
commercial databases, or dependency packages. Each Python dependency retains
its own license; installed versions are recorded in `uv.lock`.

No Himawari observations, Taipower time series, administrative boundary files,
municipal population tables, weather cache, or SimaPro/ecoinvent process exports
are tracked in this source repository. A separate local author-review bundle
contains processed input snapshots and intermediates; it is not cleared for
public distribution. See [data origins and publication status](docs/data-bundle.md).
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
