import hashlib
import json
from zipfile import ZipFile

import pytest
import fetch_bundle_boundaries as boundaries


def test_checks_all_components_before_writing(tmp_path):
    rows = []
    archive = tmp_path / 'official.zip'
    with ZipFile(archive, 'w') as z:
        for suffix in ('.cpg', '.dbf', '.prj', '.shp', '.shx'):
            name = 'COUNTY_MOI_1140318' + suffix
            raw = suffix.encode()
            z.writestr(name, raw)
            rows.append({'path': 'data/geo/county_boundaries' + suffix, 'member': name,
                         'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = tmp_path / 'bundle.json'
    correct = rows[-1]['sha256']
    rows[-1]['sha256'] = 'wrong'
    manifest.write_text(json.dumps({'external_boundaries': rows}))
    with pytest.raises(ValueError, match='version differs'):
        boundaries.fetch(tmp_path, archive=archive)
    assert not (tmp_path / 'data').exists()
    rows[-1]['sha256'] = correct
    manifest.write_text(json.dumps({'external_boundaries': rows}))
    boundaries.fetch(tmp_path, archive=archive)
    boundaries.fetch(tmp_path, check_only=True)
    assert len(list((tmp_path / 'data/geo').iterdir())) == 5
