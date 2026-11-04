"""Compare calibrations on the real clips available locally.

No clip has measured player positions, so true metres and km/h stay NOT_VERIFIED. What is measured
here: hand-read paint no method saw, how static turf moves (it should not), speed agreement between
methods, and how many frames get coordinates. A low paint error does not prove low position error.

    python -m experiments.hybrid_pitch.real_eval
"""

import json
import os
from pathlib import Path
import sys
import time

os.environ['AUTOHL_CALIBRATOR'] = 'camera'
import cv2
import numpy as np

from autohl import camera, camfit, detect, field_geometry, field_solver as fs, hybrid_calib as hc, pipeline, pitch, semantic_pitch, speed, video
from autohl.camera import warp

HERE = Path(__file__).parent
RESULTS = HERE/'results'
HOLDOUT = json.loads((HERE/'holdout_paint.json').read_text())
META = pitch.venues()['noksi']
WEIGHTS = os.environ.get('AUTOHL_LINE_WEIGHTS')
STAND = ['qoqn-1309-field', 'ye-0920-goal', '1103-181629-kmh0']
SPEED_WINDOWS = {'qoqn-1309-field': [(120, 145), (225, 275)], 'ye-0920-goal': [(225, 270), (227, 257)]}


class Fixed:
    """Per-frame matrices stored by an earlier hand calibration."""

    def __init__(self, stack):
        self.stack, self.meta = stack, META
        self.valid = np.isfinite(stack).all(axis=(1, 2))

    def supports(self, a, b=None):
        return bool(self.valid[a:(a if b is None else b)+1].all())

    def img_from_pitch(self, k):
        return self.stack[k]

    def to_pitch(self, k, p):
        return warp(p, np.linalg.inv(self.stack[k]))


class Fitted:
    # Calibrations raise on frames they cannot serve; treat those as unsupported.
    def __init__(self, calibration, frames):
        self.c, self.valid = calibration, np.zeros(frames, bool)
        for k in range(frames):
            try:
                calibration.img_from_pitch(k)
                self.valid[k] = True
            except (TypeError, ValueError, cv2.error):
                pass

    def supports(self, a, b=None):
        return bool(self.valid[a:(a if b is None else b)+1].all())

    def to_pitch(self, k, p):
        return self.c.to_pitch(k, p)

    def img_from_pitch(self, k):
        return self.c.img_from_pitch(k)


def paint_error(c, name):
    out = {}
    segments = pitch.model_segments(META)
    for frame, lines in HOLDOUT[name]['lines'].items():
        frame = int(frame)
        if not c.supports(frame):
            out[f'f{frame}'] = None
            continue
        for segment, points in lines.items():
            kind, *geometry = segments[int(segment)]
            model = np.linspace(geometry[0], geometry[1], 6000) if kind == 'line' else geometry[0]
            found = c.to_pitch(frame, np.array(points, float))
            out[f'f{frame}_line{segment}'] = float(np.median(np.linalg.norm(found[:, None]-model[None], axis=2).min(1)))
    for frame, points in HOLDOUT[name]['circle'].items():
        frame = int(frame)
        if c.supports(frame):
            radius = np.linalg.norm(c.to_pitch(frame, np.array(points, float))-[META['half_x'], 0], axis=1)
            out[f'f{frame}_circle'] = float(np.median(np.abs(radius-9.15)))
        else:
            out[f'f{frame}_circle'] = None
    return out


def static_speed(c, F, D, sky, window=30, step=15):
    """Turf does not move: any speed measured on it is instrument error."""
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    grey = {}

    def g(k):
        if k not in grey:
            grey[k] = cv2.cvtColor(F[k], cv2.COLOR_BGR2GRAY)
        return grey[k]

    values = []
    for a in range(0, len(F)-window, step):
        if not all(c.supports(k) for k in range(a, a+window+1)):
            continue
        mask = np.zeros_like(g(a))
        mask[pitch.turf_mask(F[a], sky) > 0] = 255
        for x1, y1, x2, y2, *_ in D[a].astype(int):
            mask[max(0, y1-12):y2+12, max(0, x1-12):x2+12] = 0
        points = cv2.goodFeaturesToTrack(g(a), 300, .01, 14, mask=mask, blockSize=7)
        if points is None or len(points) < 40:
            continue
        current = points.astype(np.float32)
        path, ok = [current.reshape(-1, 2).copy()], np.ones(len(points), bool)
        for k in range(a+1, a+window+1):
            moved, s, _ = cv2.calcOpticalFlowPyrLK(g(k-1), g(k), current, None, **lk)
            back, s2, _ = cv2.calcOpticalFlowPyrLK(g(k), g(k-1), moved, None, **lk)
            ok &= s.ravel().astype(bool) & s2.ravel().astype(bool) & (np.linalg.norm(current-back, axis=2).ravel() < .5)
            xy = moved.reshape(-1, 2)
            for x1, y1, x2, y2, *_ in D[k]:
                ok &= ~((xy[:, 0] > x1-6) & (xy[:, 0] < x2+6) & (xy[:, 1] > y1-6) & (xy[:, 1] < y2+6))
            ok &= (xy[:, 0] > 4) & (xy[:, 0] < 1916) & (xy[:, 1] > sky) & (xy[:, 1] < 1076)
            current = moved
            path.append(xy.copy())
        path = np.array(path)[:, ok]
        if path.shape[1] < 25:
            continue
        H, inliers = cv2.findHomography(path[0], path[-1], cv2.RANSAC, 2.5)
        if H is None:
            continue
        path = path[:, inliers.ravel() > 0]
        ground = np.stack([c.to_pitch(a+i, path[i]) for i in range(window+1)])
        keep = (ground[0, :, 0] > -3) & (ground[0, :, 0] < 103) & (np.abs(ground[0, :, 1]) < 33)
        ground = ground[:, keep]
        if ground.shape[1] < 15:
            continue
        seen = np.ones(window+1, bool)
        values.append(np.median([speed.measure_window(ground[:, i], seen, 0, window)['speed_kmh']
                                 for i in range(ground.shape[1])]))
    if not values:
        return None
    return {'median_kmh': float(np.median(values)), 'worst_kmh': float(np.max(values)), 'windows': len(values)}


def track_of(name, j):
    if name == 'ye-0920-goal':
        z = np.load(j.f('full_track_calibrations.npz'))
        return z['foot'], z['observed']
    if name == 'qoqn-1309-field':
        data = json.load(open('outputs/qoqn_1313_sprint_measurements.json'))
        foot, seen = np.full((len(j.frames()), 2), np.nan), np.zeros(len(j.frames()), bool)
        for p in data['points']:
            foot[p['frame']], seen[p['frame']] = p['image_xy_px'], p['observed']
        return foot, seen
    return None, None


def speeds(c, name, foot, seen):
    out = {}
    for a, b in SPEED_WINDOWS.get(name, ()):
        if foot is None or not c.supports(a, b):
            out[f'{a}-{b}'] = None
            continue
        ground = np.full_like(foot, np.nan)
        for k in range(a, b+1):
            if np.isfinite(foot[k]).all():
                ground[k] = c.to_pitch(k, foot[k:k+1])[0]
        try:
            out[f'{a}-{b}'] = float(speed.measure_window(ground, seen, a, b)['speed_kmh'])
        except ValueError:
            out[f'{a}-{b}'] = None
    return out


def auto_labels(c, F, D, keys):
    """Name each paint stroke by the model line a solved camera puts it on, as C/D/E input."""
    features = hc._features(META)
    out = {}
    for k in keys:
        strokes, _, _ = fs._strokes(F[k], D[k], False)
        points = fs._sample(strokes, 8)
        lines = []
        if len(points):
            ground = c.to_pitch(k, points)
            distance = np.stack([tree.query(ground)[0] for tree, *_ in features], 1)
            nearest, close = distance.argmin(1), distance.min(1) < 1.
            for segment in np.unique(nearest[close]):
                group = points[close & (nearest == segment)]
                if len(group) >= 6 and segment != 2:
                    lines.append({'segment': int(segment), 'points': group, 'confidence': 1.})
        edge = fs._far_edge(F[k], D[k])
        if len(edge):
            lines.append({'segment': 2, 'points': fs._sample(edge, 16), 'confidence': 1.})
        out[k] = lines
    return out


def wrapped(calibration, frames):
    return None if calibration is None else Fitted(calibration, frames)


def measure(c, label, name, F, D, sky, foot, seen, started):
    row = {'method': label, 'seconds': time.time()-started, 'status': 'ok' if c is not None else 'failed'}
    if c is not None:
        row['coverage'] = float(np.mean(c.valid))
        row['paint_error_m'] = paint_error(c, name) if name in HOLDOUT else {}
        row['static_turf'] = static_speed(c, F, D, sky)
        row['speed_kmh'] = speeds(c, name, foot, seen)
    print(json.dumps({k: v for k, v in row.items()}, ensure_ascii=False)[:400], flush=True)
    return row


def ai_lines(model, F, keys):
    return {k: [{'segment': semantic_pitch.SEGMENT_OF[l['class_id']], 'points': np.linspace(*np.array(l['points']), 12),
                 'confidence': l['confidence']} for l in model.detect(F[k], k)['lines']] for k in keys}


def stand_clip(name, model):
    j = pipeline.Job(name)
    F, D = j.frames(), j.dets()
    cum, sky = j.cam()
    inliers = j.camera_inliers()
    foot, seen = track_of(name, j)
    n = len(F)
    keys = list(range(0, n, 30))
    rows = []

    t = time.time()
    if name == 'ye-0920-goal':
        z = np.load(j.f('goal_calibration_variants.npz'))
        hand = Fixed(z['270_box_and_boundaries'])
    else:
        C = j.calib()
        hand = Fixed(np.stack([C.img_from_pitch(k) if C.supports(k) else np.full((3, 3), np.nan) for k in range(n)]))
    rows.append(measure(hand, 'A_hand', name, F, D, sky, foot, seen, t))

    t = time.time()
    cal = pitch.auto(F, cum, 'noksi', sky, inliers=inliers, log=lambda *_: None)
    rows.append(measure(Fitted(pitch.Calib(cal, cum), n) if cal else None, 'A_reference_photos', name, F, D, sky, foot, seen, t))
    rows[-1]['note'] = 'includes a reference photo from the 14:48 clip itself' if name == '1103-181629-kmh0' else ''

    t = time.time()
    try:
        fitted = camfit.fit(F, cum, hand if hand.valid.all() else _Filled(hand, cum), sky, dets=D, step=30, log=lambda *_: None, free_dims=False)[0]
        rows.append(measure(Fitted(fitted, n), 'B_camfit', name, F, D, sky, foot, seen, t))
    except Exception as e:
        rows.append({'method': 'B_camfit', 'status': 'failed', 'error': repr(e)[:160], 'seconds': time.time()-t})

    if model is not None:
        t = time.time()
        seen_ai = ai_lines(model, F, keys)
        counts = [len(v) for v in seen_ai.values()]
        print(name, 'AI lines per key', counts, flush=True)
        c = hc.fit_homographies(seen_ai, cum, META)
        rows.append({**measure(wrapped(c, n), 'C_ai_lines', name, F, D, sky, foot, seen, t), 'ai_lines_per_key': counts})
        t = time.time()
        c = hc.fit_shared(seen_ai, cum, META)
        rows.append({**measure(wrapped(c, n), 'D_ai_lines', name, F, D, sky, foot, seen, t), 'ai_lines_per_key': counts})

    t = time.time()
    messages = []
    cal = field_geometry.calibrate(F, cum, META, sky, inliers=inliers, boxes=D, log=messages.append)
    solved = pitch.Calib(cal, cum) if cal else None
    row = measure(solved, 'S_geometry_solver', name, F, D, sky, foot, seen, t)
    row['log'] = [m for m in messages if '카메라 위치 (' in m or '믿을' in m]
    rows.append(row)

    if solved is not None:
        valid_keys = [k for k in keys if solved.supports(k)]
        labelled = auto_labels(solved, F, D, valid_keys)
        t = time.time()
        rows.append(measure(wrapped(hc.fit_homographies(labelled, cum, META), n), 'C_auto_labels', name, F, D, sky, foot, seen, t))
        t = time.time()
        shared = hc.fit_shared(labelled, cum, META)
        rows.append(measure(wrapped(shared, n), 'D_auto_labels', name, F, D, sky, foot, seen, t))
        t = time.time()
        rows.append(measure(wrapped(hc.fit_tripod(labelled, cum, META, shared) if shared else None, n), 'E_auto_labels', name, F, D, sky, foot, seen, t))
    return rows


class _Filled:
    # camfit needs a starting matrix on every frame: fill gaps from the nearest calibrated frame.
    def __init__(self, hand, cum):
        self.hand, self.cum, self.meta = hand, cum, META
        self.good = np.flatnonzero(hand.valid)

    def img_from_pitch(self, k):
        s = int(self.good[np.abs(self.good-k).argmin()])
        return pitch.chain(self.cum, s, k, self.hand.stack[s])


def low_clip(label, F, cum, inliers, sky, D):
    messages = []
    started = time.time()
    cal = field_geometry.calibrate(F, cum, META, sky, inliers=inliers, boxes=D, log=messages.append)
    return {'clip': label, 'accepted': cal is not None, 'seconds': time.time()-started,
            'log': [m for m in messages if '카메라 위치 (' in m or '믿을' in m]}


def low_clips():
    out = []
    root = Path('work/zcz_20241008')
    for n in 'abc':
        F = video.load(root/f'{n}.npy')
        c, d = np.load(root/f'{n}_cam.npz'), np.load(root/f'{n}_det.npz')
        out.append(low_clip(f'zczL3Gikw2E-{n}', F, c['cum'], c['inl'], int(c['sky']), np.split(d['boxes'], np.cumsum(d['counts'])[:-1])))
    root = Path('work/venue_20241008')
    for s in (5, 20, 41, 52):
        F = video.load(root/f'frames_{s}.npy')
        c = np.load(root/f'camera_{s}.npz')
        n = len(F)
        wanted = sorted({int(k) for w in np.linspace(30, n-31, int(np.clip(n/45, 1, 9))).astype(int)
                         for k in np.linspace(w-30, w+30, 5).astype(int)})
        found = detect.run([F[k] for k in wanted], log=lambda *_: None)
        boxes = [np.empty((0, 5)) for _ in range(n)]
        for k, b in zip(wanted, found):
            boxes[k] = b
        out.append(low_clip(f'5zbSRX1Axcs-{s}s', F, c['cum'], c['inliers'], int(c['sky']), boxes))
    return out


def main():
    cv2.setNumThreads(4)
    model = None
    if WEIGHTS:
        model = semantic_pitch.SemanticLines(WEIGHTS)
    only = sys.argv[1:]
    saved = RESULTS/'real.json'
    # A partial run replaces only the clips it was asked for.
    result = json.loads(saved.read_text()) if only and saved.exists() else {'stand': {}, 'low': [], 'weights': bool(WEIGHTS)}
    for name in STAND:
        if not only or name in only:
            print('==', name, flush=True)
            result['stand'][name] = stand_clip(name, model)
            (RESULTS/'real.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    if not only or 'low' in only:
        result['low'] = low_clips()
    (RESULTS/'real.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
