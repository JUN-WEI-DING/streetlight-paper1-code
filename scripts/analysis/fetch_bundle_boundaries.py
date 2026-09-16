"""Retrieve the official 1140318 county boundaries omitted from the bundle.

NLSC, Ministry of the Interior, Taiwan; Open Government Data License 1.0.
Source: https://data.gov.tw/dataset/7442 ; license: https://data.gov.tw/license
The five original components must match the study's SHA-256 hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
URL = ('https://www.tgos.tw/tgos/VirtualDir/Product/'
       '1cd4f4c9-6b01-4cf9-bf6c-23a73aa17d24/'
       + quote('直轄市、縣(市)界線1140318.zip'))


def fetch(root, *, check_only=False, archive=None):
    rows = json.loads((root / 'bundle.json').read_text()).get('external_boundaries')
    suffixes = {'.cpg', '.dbf', '.prj', '.shp', '.shx'}
    if not rows or len(rows) != 5 or {Path(r['path']).suffix for r in rows} != suffixes:
        raise ValueError('Bundle must identify all five boundary components')
    for row in rows:
        relative = Path(row['path'])
        if relative.as_posix() != 'data/geo/county_boundaries' + relative.suffix:
            raise ValueError('Invalid boundary destination')
    missing = [r for r in rows if not (root / r['path']).exists()]
    if missing and check_only:
        raise FileNotFoundError('Missing county boundaries; run fetch_bundle_boundaries.py')
    contents = {}
    if missing:
        if archive is not None:
            raw = archive.read_bytes()
        else:
            with urlopen(URL, timeout=60) as response:
                raw = response.read()
        with ZipFile(io.BytesIO(raw)) as z:
            for row in missing:
                matches = [n for n in z.namelist() if Path(n).name == row['member']]
                if len(matches) != 1:
                    raise ValueError('Official archive lacks the expected unique boundary component')
                contents[row['path']] = z.read(matches[0])
    # Check every component before writing any; do not mix geometry versions.
    for row in rows:
        path = root / row['path']
        data = path.read_bytes() if path.exists() else contents[row['path']]
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Boundary version differs from the study: {path.name}')
    for relative, data in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps({'validated_boundary_files': len(rows), 'retrieved_files': len(contents)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--archive', type=Path, help='Official ZIP downloaded manually if direct access is unavailable')
    args = parser.parse_args()
    fetch(ROOT, check_only=args.check_only, archive=args.archive)


if __name__ == '__main__':
    main()
