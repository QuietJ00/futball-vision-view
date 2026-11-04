"""Run algorithms on synthetic clips and write one row per clip and method.

    python -m experiments.hybrid_pitch.benchmark --dataset synthetic --models A B C D E --seed 42
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import cv2
import numpy as np

from autohl import camfit, speed
from . import methods, synthetic_data as sd

RESULTS = Path(__file__).parent/'results'


def scenarios():
    out = []
    # Representative grid of camera placements, oracle lines.
    for h in (2., 5., 10.):
        for side in (5., 15., 30.):
            for pan in (15., 45., 90.):
                out.append(sd.Scene(f'grid_h{h:g}_s{side:g}_p{pan:g}', height=h, side=side, pan=pan, tags=('grid',)))
    # The same placements with an imperfect detector.
    for h in (2., 5., 10.):
        for side in (5., 15., 30.):
            out.append(sd.Scene(f'noisy_h{h:g}_s{side:g}', mode='noisy', height=h, side=side, noise=3.,
                                missing=.2, wrong=.1, tags=('noisy',)))
    for noise in (0., 1., 3., 5.):
        out.append(sd.Scene(f'noise_{noise:g}px', noise=noise, tags=('noise',)))
    for hidden in (0., .25, .5, .75):
        out.append(sd.Scene(f'occlusion_{hidden:g}', occlusion=hidden, tags=('occlusion',)))
    for r in (0., .02, .05, .3):
        out.append(sd.Scene(f'lens_offset_{r:g}m', offset=r, tags=('tripod',)))
    out += [sd.Scene('only_centre_circle', lines='centre_circle', tags=('sparse',)),
            sd.Scene('only_box_part', lines='box_part', tags=('sparse',)),
            sd.Scene('only_touch_halfway', lines='touch_halfway', tags=('sparse',)),
            sd.Scene('heavy_occlusion_noisy', mode='noisy', occlusion=.6, noise=3., missing=.2, wrong=.1, tags=('stress',)),
            sd.Scene('large_pan', pan=90., tags=('stress',)),
            sd.Scene('camera_moves_3m', drift=3., tripod=False, tags=('moving',)),
            sd.Scene('camera_moves_10m', drift=10., tripod=False, tags=('moving',)),
            sd.Scene('lens_distortion', k1=-.1, tags=('distortion',)),
            sd.Scene('strong_distortion', k1=-.2, tags=('distortion',))]
    for wrong in (.05, .1, .2, .4):
        out.append(sd.Scene(f'wrong_class_{wrong:g}', mode='noisy', wrong=wrong, tags=('classes',)))
    return out


def _errors(calibration, clip):
    keys = clip['keys']
    held, near, far, reprojected, position, rotation = [], [], [], [], [], []
    for k in keys:
        world, uv = sd.holdout(clip, k)
        if not len(world):
            continue
        found = calibration.to_pitch(k, uv)
        e = np.linalg.norm(np.nan_to_num(found, nan=1e4)-world, axis=1)
        p = clip['cameras'][k]
        distance = np.linalg.norm(world-p[4:6], axis=1)
        held.append(e)
        near.append(e[distance <= np.median(distance)])
        far.append(e[distance > np.median(distance)])
        reprojected.append(np.linalg.norm(np.nan_to_num(calibration.to_img(k, world), nan=1e4)-uv, axis=1))
        try:
            _, w, centre = camfit.decompose(calibration.img_from_pitch(k))
            R = cv2.Rodrigues(w)[0]
            truth = cv2.Rodrigues(cv2.Rodrigues(sd.fs._rotation(*p[:3]))[0])[0]
            position.append(min(np.linalg.norm(centre-np.r_[p[4:6], -p[6]]),
                                np.linalg.norm(centre*[-1, -1, 1]+[100, 0, 0]-np.r_[p[4:6], -p[6]])))
            rotation.append(np.degrees(np.arccos(np.clip((np.trace(R.T @ truth)-1)/2, -1, 1))))
        except (cv2.error, np.linalg.LinAlgError, TypeError):  # diagnostics only: a view with no pitch in it
            pass
    if not held:
        return {}
    held = np.concatenate(held)
    return {'hold_median_m': float(np.median(held)), 'hold_p95_m': float(np.percentile(held, 95)),
            'hold_rmse_m': float(np.sqrt(np.mean(np.minimum(held, 1e3)**2))),
            'near_median_m': float(np.median(np.concatenate(near))),
            'far_median_m': float(np.median(np.concatenate(far))),
            'reproject_px': float(np.median(np.concatenate(reprojected))),
            'camera_position_m': float(np.median(position)) if position else None,
            'camera_rotation_deg': float(np.median(rotation)) if rotation else None}


def _longest_run(pixels):
    seen = np.isfinite(pixels).all(1)
    best, start = (0, 0), None
    for i, s in enumerate(np.r_[seen, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i-start > best[1]-best[0]:
                best = start, i
            start = None
    return best


def _speed(calibration, clip):
    players = clip['players']
    if players is None:
        return {}
    out = {}
    lo, hi = _longest_run(players['runner']['pixels'])
    if hi-lo >= 45:
        result = speed.analyse(calibration, players['runner']['pixels'], lo, hi-1)
        world = players['runner']['world'][lo:hi]
        true_distance = np.linalg.norm(np.diff(world, axis=0), axis=1).sum()
        true_speed = true_distance/((hi-1-lo)/sd.FPS)*3.6
        out.update(distance_error_m=abs(result['run']-true_distance),
                   mean_speed_error_kmh=abs(result['avg']-true_speed),
                   top_speed_error_kmh=abs(result['top']-true_speed))
    lo, hi = _longest_run(players['still']['pixels'])
    if hi-lo >= 45:
        out['still_fake_speed_kmh'] = speed.analyse(calibration, players['still']['pixels'], lo, hi-1)['top']
    return out


def run_one(job):
    scene, seed, name = job
    cv2.setNumThreads(1)
    clip = sd.make_clip(scene, seed)
    row = {'scene': scene.name, 'mode': scene.mode, 'tags': '+'.join(scene.tags), 'seed': seed, 'method': name,
           'height': scene.height, 'side': scene.side, 'pan': scene.pan, 'tripod_true': scene.tripod}
    started = time.perf_counter()
    try:
        calibration = methods.METHODS[name](clip)
        row['error'] = ''
    except Exception as e:  # recorded as a failure of the method, never hidden
        calibration, row['error'] = None, f'{type(e).__name__}: {e}'[:120]
    row['seconds'] = time.perf_counter()-started
    row['status'] = 'ok' if calibration is not None else 'failed'
    row['notes'] = getattr(calibration, 'notes', '')
    if calibration is not None:
        try:
            row.update(_errors(calibration, clip))
            row.update(_speed(calibration, clip))
        except Exception as e:  # counted as a failure of the method, and kept in the table
            row.update(status='failed', error=f'evaluation {type(e).__name__}: {e}'[:120])
    return row


def environment(models, seed, repeats):
    version = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return {'models': models, 'first_seed': seed, 'repeats': repeats, 'code': version or 'uncommitted',
            'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
            'machine': platform.machine(), 'cpus': os.cpu_count(), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', choices=['synthetic'])
    parser.add_argument('--models', nargs='+', default=['TRUTH', 'A', 'B', 'C', 'D0', 'D', 'E', 'AUTO'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)-2))
    parser.add_argument('--smoke', action='store_true', help='a few scenes, for checking the code')
    args = parser.parse_args()
    models = sorted(set(args.models)|{'TRUTH'}, key=list(methods.METHODS).index)
    scenes = scenarios()
    if args.smoke:
        scenes = scenes[::9]
    jobs = [(scene, args.seed+i, m) for scene in scenes for i in range(args.repeats) for m in models]
    print(f'{len(scenes)} scenes, {len(jobs)} runs', flush=True)
    RESULTS.mkdir(exist_ok=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=4))
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(RESULTS/'comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    info = environment(models, args.seed, args.repeats)
    info.update(scenes=len(scenes), runs=len(rows), wall_seconds=time.time()-started)
    (RESULTS/'run_info.json').write_text(json.dumps(info, indent=1))
    print(f'done in {time.time()-started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
