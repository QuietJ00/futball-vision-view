"""Download the exact public fixture files and verify their recorded SHA-256."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(clips, manifest, data):
    index = json.loads((manifest.parent / 'sources.json').read_text())
    data.mkdir(parents=True, exist_ok=True)
    filenames = dict.fromkeys(name for clip in clips
                              for name in [*clip['sources'], *([clip['sensor']] if 'sensor' in clip else [])])
    for name in filenames:
        entry, destination = index[name], data / name
        if destination.exists() and sha256(destination) == entry['sha256']:
            continue
        print(f'Downloading {name}', flush=True)
        request = Request(entry['url'], headers={'User-Agent': 'FutbolVision benchmark/1.0'})
        with tempfile.TemporaryDirectory(dir=data) as directory:
            staged = Path(directory) / 'download'
            with urlopen(request, timeout=60) as response, staged.open('wb') as out:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > entry['bytes']:
                        raise ValueError(f'{name}: download larger than recorded fixture')
                    out.write(block)
            if count != entry['bytes'] or sha256(staged) != entry['sha256']:
                raise ValueError(f'{name}: download does not match recorded fixture')
            os.replace(staged, destination)
