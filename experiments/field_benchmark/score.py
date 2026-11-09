"""Score saved predictions against references which never enter a calibration engine."""
import csv
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from autohl import camera, detect, pitch, speed, track
from sensor import aligned, frame_times, read_tags

ENGINES = ('pivotanchor', 'turflens')
LINE_LIMITS_PX = (10, 20, 40)


class Projection:
    def __init__(self, result):
        self.matrices = result['matrices']
        self.valid = result['valid']
        self.segments = result['segments']
        self.k1, self.focal = float(result['k1']), float(result['focal'])
        self.inverse = np.full_like(self.matrices, np.nan)
        self.inverse[self.valid] = np.linalg.inv(self.matrices[self.valid])

    def supports(self, k):
        return bool(self.valid[k])

    def to_pitch(self, k, points):
        return camera.warp(camera.undistort(points, self.k1, self.focal), self.inverse[k])

    def to_img(self, k, points):
        return camera.distort(camera.warp(points, self.matrices[k]), self.k1, self.focal)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(values.max()), n=len(values))


def paint_scores(C, meta, marks):
    lengths = meta['length']
    lines = pitch.model_segments(meta)
    samples = [np.linspace(shape[0], shape[1], 2400) if kind == 'line' else shape[0]
               for kind, *shape in lines]
    attempts = []
    # Resolve the field's unavoidable 180-degree symmetry once per clip, never frame by frame.
    for rotate in (False, True):
        frames, landmarks = {}, {}
        for frame, annotations in marks.items():
            k = int(frame)
            if not C.supports(k):
                continue
            errors = []
            for line, points in annotations.get('lines', {}).items():
                world = samples[int(line)]
                if rotate:
                    world = [lengths, 0] - world
                image = C.to_img(k, world)
                image = image[np.isfinite(image).all(axis=1)]
                if len(image):
                    errors.extend(cKDTree(image).query(points)[0])
            if errors:
                frames[frame] = errors
            keypoints = annotations.get('landmarks', [])
            if keypoints:
                xy = np.array([p['world'] for p in keypoints])
                image = np.array([p['pixel'] for p in keypoints])
                world = C.to_pitch(k, image)
                if rotate:
                    world = [lengths, 0] - world
                landmarks[frame] = np.linalg.norm(world - xy, axis=1)
        pixels = [e for values in frames.values() for e in values]
        meters = [e for values in landmarks.values() for e in values]
        objective = np.median(pixels) if pixels else np.inf
        attempts.append((objective, rotate, frames, pixels, meters))
    _, rotate, frames, pixels, meters = min(attempts, key=lambda a: a[:2])
    by_frame = {f: distribution(errors) for f, errors in frames.items()}
    return {'labelled_frames': len(marks), 'scored_frames': len(frames),
            'line_error_px': distribution(pixels), 'landmark_error_m': distribution(meters),
            'within_px': {x: sum(f['p95'] <= x for f in by_frame.values()) for x in LINE_LIMITS_PX},
            'rotated_180': rotate if pixels else None, 'per_frame': by_frame}


def grass_pairs(frames, boxes, fps):
    """Independent SIFT links on static grass, sampled 1 s apart with players masked."""
    sift = cv2.SIFT_create(2400, contrastThreshold=.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = {}
    for a in range(0, len(frames) - fps, max(1, fps//2)):
        b = a + fps
        for k in (a, b):
            if k not in features:
                half = cv2.resize(frames[k], (960, 540), interpolation=cv2.INTER_AREA)
                h, s, _ = cv2.split(cv2.cvtColor(half, cv2.COLOR_BGR2HSV))
                mask = cv2.morphologyEx(np.uint8((h >= 25) & (h <= 95) & (s > 35))*255,
                                        cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
                for x0, y0, x1, y1, _ in np.asarray(boxes[k]/2, dtype=int):
                    mask[max(0, y0-4):y1+5, max(0, x0-4):x1+5] = 0
                features[k] = sift.detectAndCompute(cv2.cvtColor(half, cv2.COLOR_BGR2GRAY), mask)
        ka, da = features[a]
        kb, db = features[b]
        if da is None or db is None or min(len(da), len(db)) < 25:
            continue
        matches = [m[0] for m in matcher.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < .75*m[1].distance]
        if len(matches) < 25:
            continue
        p = np.float32([ka[m.queryIdx].pt for m in matches]) * 2
        q = np.float32([kb[m.trainIdx].pt for m in matches]) * 2
        _, inliers = cv2.findHomography(p, q, cv2.USAC_MAGSAC, 2.)
        if inliers is not None and inliers.sum() >= 25:
            keep = inliers.ravel().astype(bool)
            if cv2.contourArea(cv2.convexHull(p[keep])) > 12000:
                yield a, b, p[keep], q[keep]


def motion_scores(C, pairs):
    scores = {}
    for a, b, p, q in pairs:
        if not (C.valid[a:b+1].all() and C.segments[a] == C.segments[b]):
            continue
        error = np.linalg.norm(C.to_img(b, C.to_pitch(a, p)) - q, axis=1)
        if np.isfinite(error).all():
            scores[f'{a}/{b}'] = float(np.median(error))
    return {'eligible_pairs': len(pairs), 'error_px': distribution(list(scores.values())), 'per_pair': scores}


def manual_control(labels, count):
    points = labels.get('manual_control_frame_0')
    if not points:
        return None
    xy = np.array([p['world'] for p in points])
    image = np.array([p['pixel'] for p in points])
    H, _ = cv2.findHomography(xy, image, 0)
    return Projection(dict(matrices=np.repeat(H[None], count, axis=0), valid=np.ones(count, bool),
                           segments=np.zeros(count, int), k1=0., focal=1.))


def track_targets(clip, labels, cache, data, frames, boxes, fps):
    targets = labels.get('targets', [])
    if not targets:
        return []
    with np.load(cache / 'camera.npz') as data_cache:
        cumulative = data_cache['cumulative']
    features = track.features(frames, boxes)
    times = frame_times(clip, fps, len(frames)) + labels.get('sensor_clock_offset_s', 0.)
    tags = read_tags(data / clip['sensor'], times)
    control = manual_control(labels, len(frames))
    result = []
    for target in targets:
        if target['tag'] not in tags:
            raise ValueError(f"{clip['id']}: annotated sensor tag absent")
        first, last = target['frames']
        aligned_tag = aligned(tags[target['tag']], times)
        reference = aligned_tag[:, 2].copy()
        reference[:first] = reference[last+1:] = np.nan
        anchors = {int(k): v for k, v in target['anchors'].items()}
        try:
            path = track.track(frames, boxes, cumulative, anchors, FE=features, log=lambda *_: None)
            foot = path['foot']
            observed = (path['src'] >= 1) & ~path['risk']
            check = {k: float(np.linalg.norm(foot[int(k)] - point))
                     if np.isfinite(foot[int(k)]).all() else None
                     for k, point in target.get('check_feet', {}).items()}
            sensor_check = {}
            if control is not None:
                for k, point in target.get('check_feet', {}).items():
                    x, y = aligned_tag[int(k), :2]
                    xy = control.to_pitch(int(k), [point])[0]
                    sensor_check[k] = float(np.linalg.norm(xy - [x, clip['pitch']['width']/2-y]))
            sensor_check = {k: (v if np.isfinite(v) else None) for k, v in sensor_check.items()}
            audited = (len(check) >= 2 and all(x is not None and x <= target['check_max_px']
                                               for x in check.values()) and len(sensor_check) >= 2
                       and all(x is not None and x <= 3. for x in sensor_check.values()))
            result.append((target, reference, foot, observed, check, sensor_check, audited, None))
        except ValueError as exc:
            result.append((target, reference, None, None, {}, {}, False, str(exc)))
    return result


def velocity_scores(C, targets, fps, writer, clip_id, engine):
    rows, errors = [], {}
    for target, ref, foot, observed, check, sensor_check, audited, failure in targets:
        first, last = target['frames']
        estimated = np.full(len(ref), np.nan)
        if audited and np.any(C.valid[first:last+1] & observed[first:last+1]):
            estimated = speed.player_speed(C, foot, observed, first, last, fps=fps)['kmh']
        eligible = np.isfinite(ref)
        paired = eligible & np.isfinite(estimated)
        residual = estimated[paired] - ref[paired]
        if len(residual):
            error = {'mae': float(np.mean(np.abs(residual))), 'rmse': float(np.sqrt(np.mean(residual**2))),
                     'bias': float(np.mean(residual)), 'absolute': distribution(np.abs(residual))}
            # Speed reported in the app is a short sustained peak. Use the same window and support.
            true, tk = speed.sustained_top(np.where(paired, ref, np.nan))
            pred, pk = speed.sustained_top(np.where(paired, estimated, np.nan))
            peak = dict(sensor_kmh=true, estimate_kmh=pred, timing_error_s=(pk-tk)/fps) if tk is not None else None
        else:
            error, peak = None, None
        name = target['id']
        errors.update({f'{name}/{k}': float(estimated[k]-ref[k]) for k in np.flatnonzero(paired)})
        rows.append(dict(id=name, eligible=int(eligible.sum()), scored=int(paired.sum()),
                         identity_audit_pass=audited, identity_check_error_px=check,
                         sensor_check_error_m=sensor_check, tracking_error=failure,
                         error_kmh=error, sustained_peak_same_support=peak))
        for k in np.flatnonzero(eligible):
            writer.writerow([clip_id, engine, name, k, k/fps, ref[k], estimated[k] if paired[k] else ''])
    return rows, errors


def common_errors(a, b):
    overlap = a.keys() & b.keys()
    return {engine: distribution([abs(values[k]) for k in sorted(overlap)])
            for engine, values in zip(ENGINES, (a, b))}


def gates(clip, row, policy):
    checks = {'no_crash': row['error'] is None}
    if clip['class'].startswith('negative_'):
        checks['no_false_accept'] = row['supported'] == 0 and checks['no_crash']
    if clip['class'].startswith('elevated_'):
        checks['field_coverage'] = row['supported']/row['frames'] >= policy['elevated_frame_coverage']
        paint = row['paint']
        good = sum(frame['p95'] <= policy['line_p95_px'] for frame in paint['per_frame'].values())
        checks['field_accuracy_yield'] = (paint['labelled_frames'] > 0 and
            good/paint['labelled_frames'] >= policy['elevated_good_labelled_share'])
    if 'sensor' in clip:
        checks['speed'] = bool(row['speed']) and all(
            t['identity_audit_pass'] and t['eligible'] and t['scored']/t['eligible'] >= policy['speed_sample_coverage']
            and t['error_kmh']['mae'] <= policy['speed_mae_kmh'] for t in row['speed'])
    return checks


def score(clips, labels_dir, data, cache, output, revisions, policy):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    labelled = json.loads((labels_dir / 'reference.json').read_text())
    result = {}
    with (output / 'speed_trace.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'engine', 'target', 'frame', 'elapsed_s', 'sensor_kmh', 'estimate_kmh'])
        for clip in clips:
            name = clip['id']
            directory = cache / name
            inputs = json.loads((directory / 'input.json').read_text())
            labels = labelled.get(name, {})
            frames = np.load(directory / 'frames.npy', mmap_mode='r')
            boxes = detect.load(directory / 'dets.npz')
            pairs = list(grass_pairs(frames, boxes, inputs['fps'])) if labels.get('frames') else []
            targets = track_targets(clip, labels, directory, data, frames, boxes, inputs['fps'])
            results, motions, velocities = {}, [], []
            for engine in ENGINES:
                path = output / f'{name}__{engine}'
                row = json.loads(path.with_suffix('.json').read_text())
                if (row['input_key'], row['revision'], row.get('pitch', clip['pitch'])) != (
                        inputs['key'], revisions[engine], clip['pitch']):
                    raise ValueError(f'{name} {engine}: stale prediction')
                with np.load(path.with_suffix('.npz')) as prediction:
                    calibration = Projection(prediction)
                row['paint'] = paint_scores(calibration, clip['pitch'], labels.get('frames', {}))
                row['motion'] = motion_scores(calibration, pairs)
                row['speed'], errors = velocity_scores(calibration, targets, inputs['fps'], writer, name, engine)
                row['false_accept_frames'] = row['supported'] if clip['class'].startswith('negative_') else None
                row['gates'] = gates(clip, row, policy)
                results[engine] = row
                motions.append(row['motion']['per_pair'])
                velocities.append(errors)
            result[name] = dict(venue=clip['venue'], match=clip['match'], camera_class=clip['class'],
                                pitch_dimensions=clip['dimensions'], engines=results,
                                common_motion_px=common_errors(*motions), common_speed_kmh=common_errors(*velocities))
            control = manual_control(labels, len(frames))
            if control is not None:
                control_speed, _ = velocity_scores(control, targets, inputs['fps'], writer, name, 'manual_control')
                result[name]['manual_control'] = dict(speed=control_speed,
                                                       paint=paint_scores(control, clip['pitch'], labels['frames']))
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    passed = True
    for name, clip in result.items():
        for engine, row in clip['engines'].items():
            failed = [check for check, ok in row['gates'].items() if not ok]
            passed &= not failed
            status = f"FAIL {', '.join(failed)}" if failed else 'PASS'
            if not failed and len(row['gates']) == 1:
                status = 'DIAGNOSTIC (no crash)'
            print(f"{name:22} {engine:11} {row['supported']:3}/{row['frames']:<3} {status}")
    return passed
