"""Run a frozen calibration engine with only shared video inputs and pitch dimensions."""
import json
import os
from time import perf_counter

import cv2
import numpy as np

from autohl import detect


def solve(clip, cache, output, engine, revision):
    key = json.loads((cache / 'input.json').read_text())['key']
    target = output / f"{clip['id']}__{engine}"
    meta = clip['pitch']
    if target.with_suffix('.json').exists() and target.with_suffix('.npz').exists():
        previous = json.loads(target.with_suffix('.json').read_text())
        if (previous.get('input_key'), previous.get('revision'), previous.get('pitch'), previous.get('error')) == (key, revision, meta, None):
            print(f'{target.name}: using saved prediction', flush=True)
            return
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    frames = np.load(cache / 'frames.npy', mmap_mode='r')
    boxes = detect.load(cache / 'dets.npz')
    with np.load(cache / 'camera.npz') as c:
        cumulative, inliers, sky = c['cumulative'], c['inliers'], int(c['sky'])
    n = len(frames)
    matrices = np.full((n, 3, 3), np.nan)
    valid = np.zeros(n, bool)
    segments = np.zeros(n, int)
    k1, focal, reason, failure = 0., 1., None, None
    started = perf_counter()
    try:
        if engine == 'pivotanchor':
            from autohl import auto_pitch

            r = auto_pitch.calibrate(frames, boxes, cumulative, inliers, meta, sky,
                                     lens_offset_m=.12, venue=None, reference_bank=None)
            reason = r.reason
            if r.calibration is not None:
                valid = r.supported.copy()
                if valid.any():
                    matrices[valid] = [r.calibration.matrices[k] for k in np.flatnonzero(valid)]
        else:
            from autohl import field_geometry, pitch

            os.environ['AUTOHL_CALIBRATOR'] = 'camera'
            r = field_geometry.calibrate(frames, cumulative, meta, sky,
                                         inliers=inliers, boxes=boxes)
            if r is None:
                reason = 'No calibration'
            else:
                c = pitch.Calib(r, cumulative)
                valid = np.array([c.supports(k) for k in range(n)])
                matrices[valid] = c.matrix[valid]
                segments, k1, focal = c.segments, c.k1, c.focal
        if valid.any() and (not np.isfinite(matrices[valid]).all()
                            or np.any(np.abs(np.linalg.det(matrices[valid])) < 1e-12)):
            raise ValueError('Supported frame has invalid projection matrix')
    except Exception as exc:
        valid[:] = False
        failure = f'{type(exc).__name__}: {exc}'
    report = dict(engine=engine, revision=revision, input_key=key, pitch=meta, reason=reason, error=failure,
                  supported=int(valid.sum()), frames=n, elapsed_s=round(perf_counter()-started, 3))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), matrices=matrices, valid=valid, segments=segments,
                        k1=k1, focal=focal)
    target.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
