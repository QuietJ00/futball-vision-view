"""Reproducible pitch and speed evaluation on external match footage."""
import argparse
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def validate_fixed_mount_fixtures(manifest):
    if manifest.get('suite') != 'fixed_mount_v2':
        return
    for clip in manifest['clips']:
        if clip.get('camera_mount') not in ('documented_fixed', 'fixed_viewpoint_consistent'):
            raise ValueError(f"{clip['id']}: a reviewed fixed camera viewpoint is required")
        if clip.get('dimensions') != 'published' or not all(
            clip.get(key) for key in ('venue_name', 'dimensions_source', 'footage_source', 'camera_evidence')
        ):
            raise ValueError(f"{clip['id']}: provide the stadium and sources for its dimensions and camera")
        pitch = clip.get('pitch', {})
        if not all(isinstance(pitch.get(key), (int, float)) and math.isfinite(pitch[key]) and pitch[key] > 0
                   for key in ('length', 'width')):
            raise ValueError(f"{clip['id']}: provide positive pitch length and width in metres")


def snapshot(revision, directory):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Each engine needs a full Git commit SHA')
    destination = directory / revision
    if not destination.exists():
        directory.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(['git', 'archive', revision], cwd=ROOT)
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            staged = Path(tmp) / revision
            staged.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
                contents.extractall(staged, filter='data')
            try:
                staged.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fetch', 'prepare', 'solve', 'score'))
    parser.add_argument('--manifest', type=Path, default=HERE / 'fixtures.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'work/external_benchmark/data')
    parser.add_argument('--cache', type=Path, default=ROOT / 'work/external_benchmark/cache')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/external_benchmark')
    parser.add_argument('--clip', action='append')
    parser.add_argument('--engine', choices=('pivotanchor', 'turflens'))
    parser.add_argument('--strict', action='store_true', help='exit nonzero when a score gate fails')
    args = parser.parse_args()
    if args.stage == 'solve' and not args.engine:
        parser.error('solve requires --engine')
    if args.strict and args.stage != 'score':
        parser.error('--strict applies to score')
    manifest = json.loads(args.manifest.read_text())
    validate_fixed_mount_fixtures(manifest)
    clips = [clip for clip in manifest['clips'] if not args.clip or clip['id'] in args.clip]
    if args.clip and set(args.clip) != {clip['id'] for clip in clips}:
        parser.error('unknown clip')
    revision = manifest['engines'][args.engine] if args.stage == 'solve' else None
    code = snapshot(revision, args.cache / 'engines') if revision else ROOT
    sys.path.insert(0, str(code.resolve()))
    if args.stage == 'fetch':
        from fetch import fetch

        fetch(clips, args.manifest, args.data)
    elif args.stage == 'prepare':
        from prepare import prepare

        for clip in clips:
            print(f"Preparing {clip['id']}", flush=True)
            prepare(clip, args.data, args.cache / clip['id'])
    elif args.stage == 'solve':
        from solve import solve

        for clip in clips:
            solve(clip, args.cache / clip['id'], args.output, args.engine, revision)
    else:
        from score import score

        passed = score(clips, args.manifest.parent, args.data, args.cache, args.output,
                       manifest['engines'], manifest['policy'])
        if args.strict and not passed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
