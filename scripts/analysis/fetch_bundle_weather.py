"""Fetch omitted public NASA POWER inputs and verify their scientific values.

Run after reproduce_from_bundle.py --stage-only. No author cache is needed.
Differences in API metadata are allowed; changed meteorological values are not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from paper1_pv_benchmark import parse_weather, weather_url

ROOT = Path(__file__).resolve().parents[2]


def validate(payload, expected):
    parse_weather(payload)
    encoded = json.dumps(payload['properties']['parameter'], sort_keys=True,
                         separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError('NASA meteorological values differ from the study snapshot; '
                         'do not claim exact reproduction with this response')


def fetch(root, *, check_only=False):
    manifest = json.loads((root / 'bundle.json').read_text())
    rows = manifest.get('external_weather')
    if not rows:
        raise ValueError('Bundle lacks external_weather; use the current data bundle')
    downloaded = 0
    for row in rows:
        relative = Path(row['path'])
        if (relative.parent.as_posix() != 'outputs/final_runs/paper1_pv_benchmark_inputs'
                or relative.name != f"met_{row['latitude']:.3f}_{row['longitude']:.3f}.json"):
            raise ValueError('Invalid meteorology destination')
        path = root / relative
        if path.exists():
            raw = path.read_bytes()
        elif check_only:
            raise FileNotFoundError(path)
        else:
            with urlopen(weather_url(row['latitude'], row['longitude']), timeout=60) as response:
                raw = response.read()
        validate(json.loads(raw), row['parameter_sha256'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            downloaded += 1
        print(f'Validated {path.name}', flush=True)
    print(json.dumps({'validated_cells': len(rows), 'downloaded_cells': downloaded}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate local inputs without network access')
    args = parser.parse_args()
    fetch(ROOT, check_only=args.check_only)


if __name__ == '__main__':
    main()
