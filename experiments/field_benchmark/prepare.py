"""Decode each fixture and compute inputs shared by both calibration engines."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

from autohl import camera, detect, video


def fingerprint(paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def prepare(clip, data, cache):
    sources = [data / name for name in clip['sources']]
    cache.mkdir(parents=True, exist_ok=True)
    key = fingerprint([*sources, Path(detect.MODEL), Path(detect.__file__),
                       Path(camera.__file__), Path(video.__file__)],
                      {k: clip.get(k) for k in ('start', 'duration', 'segment_seconds')})
    manifest = cache / 'input.json'
    if manifest.exists() and json.loads(manifest.read_text())['key'] == key:
        return
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        tmp = Path(directory)
        decoded = []
        for i, source in enumerate(sources):
            target = tmp / f'{i}.npy'
            start = clip.get('start', 0) if len(sources) == 1 else 0
            seconds = clip.get('segment_seconds', clip['duration'])
            input_file = source
            if source.suffix == '.h264':
                input_file = tmp / f'{i}.mp4'
                subprocess.run([video.FF, '-loglevel', 'error', '-r', str(video.FPS), '-i',
                                str(source), '-c:v', 'copy', str(input_file)], check=True)
            count = video.decode(str(input_file), str(target), start, start + seconds)
            if abs(count - round(seconds * video.FPS)) > 1:
                raise ValueError(f'{source.name}: unexpected duration ({count} frames)')
            if source.suffix == '.h264' and count != round(seconds * video.FPS):
                raise ValueError(f'{source.name}: timestamped segments need exact frame counts')
            decoded.append(video.load(target))
        count = sum(len(part) for part in decoded)
        frames = np.lib.format.open_memmap(tmp / 'frames.npy', mode='w+', dtype=np.uint8,
                                          shape=(count, video.H, video.W, 3))
        offset = 0
        for part in decoded:
            frames[offset:offset + len(part)] = part
            offset += len(part)
        frames.flush()
        cumulative, inliers, sky = camera.solve(frames)
        np.savez(tmp / 'camera.npz', cumulative=cumulative, inliers=inliers, sky=sky)
        detect.save(tmp / 'dets.npz', detect.run(frames))
        for name in ('frames.npy', 'camera.npz', 'dets.npz'):
            os.replace(tmp / name, cache / name)
    payload = {'key': key, 'frames': count, 'fps': video.FPS, 'resolution': [video.W, video.H]}
    tmp_manifest = manifest.with_suffix('.tmp')
    tmp_manifest.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(tmp_manifest, manifest)
