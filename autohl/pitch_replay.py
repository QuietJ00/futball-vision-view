"""Offline source + fixed-map replay. Run with python -m autohl.pitch_replay --help."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from . import auto_pitch, camera, detect, fixed_pitch, pipeline, pitch, speed, track, video


def _venue(args, config):
    name = args.venue or config.get('venue')
    if args.length is not None or args.width is not None:
        if args.venue or not (args.length and args.width):
            raise ValueError('Supply both --length and --width, or select a --venue.')
        meta = {'length': args.length, 'width': args.width}
    else:
        venues = pitch.venues()
        if name not in venues:
            raise ValueError('Select a --venue or enter --length and --width in metres.')
        meta = dict(venues[name])
    meta.update({key: value for key, value in vars(args).items()
                 if key in (*pitch.STD_DIMS, 'half_x') and value is not None})
    l, w = meta['length'], meta['width']
    d = pitch.dims(meta)
    values = [l, w, meta.get('half_x', l/2), *d.values()]
    if (not np.isfinite(values).all() or min(values) <= 0
            or l <= 2*d['box_depth'] or w < 2*d['box_half']
            or d['ga_depth'] >= d['box_depth'] or d['ga_half'] >= d['box_half']
            or not 0 < d['box_depth'] - d['spot'] < 9.15
            or not 9.15 < meta.get('half_x', l/2) < l-9.15):
        raise ValueError('Pitch measurements do not fit the template. Check the dimensions in metres.')
    return meta


def _marks(config, key):
    return {int(k): {int(i): points for i, points in lines.items()}
            for k, lines in config.get(key, {}).items()}


def _ground(calibration, frame, pixels):
    xy = calibration.to_pitch(frame, pixels)
    depth_step = calibration.to_pitch(frame, pixels + [0, 1])
    metres_per_px = np.linalg.norm(xy - depth_step, axis=1)
    limit = [calibration.meta['length'], calibration.meta['width']/2]
    keep = (np.isfinite(xy).all(axis=1) & (xy >= [0, -limit[1]]).all(axis=1)
            & (xy <= limit).all(axis=1) & (metres_per_px <= .75))
    xy[~keep] = np.nan
    return xy


def _replay_data(calibration, valid, boxes, target, first, last, source_offset):
    n = len(valid)
    xy = np.full((n, 2), np.nan)
    observed = np.zeros(n, dtype=bool)
    if target is not None:
        for k in np.flatnonzero(valid & (target['src'] > 0) & ~target['risk']):
            xy[k] = _ground(calibration, k, target['foot'][k:k+1])[0]
        observed = np.isfinite(xy).all(axis=1)

    records = []
    for k in range(first, last+1):
        players = []
        if valid[k]:
            detected = boxes[k]
            detected = detected[(detected[:, 0] > 2) & (detected[:, 2] < video.W-2)
                                & (detected[:, 3] < video.H-2)]
            feet = np.c_[(detected[:, 0]+detected[:, 2])/2, detected[:, 3]]
            for pixel, ground, box in zip(feet, _ground(calibration, k, feet), detected):
                if np.isfinite(ground).all():
                    players.append({'foot_px': pixel.tolist(), 'xy_m': ground.tolist(),
                                    'confidence': float(box[4])})

        sample = {'frame': k, 'source_time_s': source_offset+k/video.FPS,
                  'field_supported': bool(valid[k]), 'detections': players, 'target': None}
        if observed[k]:
            sample['target'] = {'foot_px': target['foot'][k].tolist(), 'xy_m': xy[k].tolist(),
                                'speed_kmh': None}
            a, b = k-video.FPS//2, k+video.FPS//2
            if first <= a and b <= last and valid[a:b+1].all():
                try:
                    measurement = speed.measure_window(xy, observed, a, b)
                except ValueError:
                    pass
                else:
                    sample['target'].update(measurement, speed_frames=[a, b])
        records.append(sample)
    return records


def _draw_polyline(image, points, color, thickness=1):
    visible = np.isfinite(points).all(axis=1) & (np.abs(points) < 20000).all(axis=1)
    indices = np.flatnonzero(visible)
    for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1):
        if len(run) >= 2:
            cv2.polylines(image, [np.rint(points[run]).astype(np.int32)], False,
                          color, thickness, cv2.LINE_AA)


def _render(frames, calibration, records, path, show_unverified_overlay=False, *, meta=None):
    meta = calibration.meta if calibration is not None else meta
    lines = pitch.model_lines(meta)
    l, w = meta['length'], meta['width']
    scale = min(640/l, 410/w)
    corner = np.array([345-l*scale/2, 290])
    map_xy = lambda p: np.asarray(p)*scale+corner
    base = np.full((600, 690, 3), (27, 27, 19), np.uint8)
    lo, hi = [tuple(np.rint(map_xy(p)).astype(int)) for p in ((0, -w/2), (l, w/2))]
    cv2.rectangle(base, lo, hi, (52, 100, 49), -1)
    for line in lines:
        _draw_polyline(base, map_xy(line), (221, 238, 223), 2)

    def label(img, text, x, y, color=(235, 239, 230)):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .68, color, 1, cv2.LINE_AA)

    def dot(img, p, color, size):
        cv2.circle(img, tuple(np.rint(p).astype(int)), size, color, -1, cv2.LINE_AA)

    label(base, f'FIXED PITCH MAP  |  {l:g} x {w:g} m', 24, 37)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.mp4')
    cmd = [video.FF, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', '1650x600', '-r', str(video.FPS), '-i', '-', '-an', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp)]
    trail = []
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            with encoder.stdin:
                for sample in records:
                    k = sample['frame']
                    good = sample['field_supported']
                    source = cv2.resize(frames[k], (960, 540), interpolation=cv2.INTER_AREA)
                    scene = np.full((600, 960, 3), (27, 27, 19), np.uint8)
                    scene[55:595] = source
                    field = base.copy()
                    label(scene, f'SOURCE  |  {video.fmt_time(sample["source_time_s"])}', 24, 37)
                    tint = (147, 254, 97) if good else (30, 154, 255)
                    if calibration is not None and (good or show_unverified_overlay):
                        for line in lines:
                            _draw_polyline(scene[55:595], calibration.to_img(k, line)*.5, tint)
                    for person in sample['detections']:
                        dot(field, map_xy(person['xy_m']), (242, 224, 171), 5)
                    selected = sample['target']
                    if selected:
                        trail.append(selected['xy_m'])
                        _draw_polyline(field, map_xy(trail[-60:]), (70, 240, 251), 2)
                        dot(field, map_xy(selected['xy_m']), (70, 240, 251), 8)
                        dot(scene[55:595], np.asarray(selected['foot_px'])*.5, (70, 240, 251), 5)
                    else:
                        trail.clear()
                    if not good:
                        label(field, 'CALIBRATION UNVERIFIED  |  coordinates withheld', 24, 560, tint)
                    else:
                        label(field, 'Blue: person detections   Yellow: tracked player', 24, 544)
                        kmh = selected['speed_kmh'] if selected else None
                        if kmh is not None:
                            label(field, f'1 s speed estimate: {kmh:.1f} km/h', 24, 574, (70, 240, 251))
                    encoder.stdin.write(np.hstack((scene, field)).tobytes())
            if encoder.wait() != 0:
                raise RuntimeError('FFmpeg could not encode the replay.')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--job', help='Existing job with frames, camera motion and detections')
    source.add_argument('--source', type=Path, help='Local clip, at most 90 seconds after trimming')
    parser.add_argument('--start', type=video.parse_time)
    parser.add_argument('--end', type=video.parse_time)
    parser.add_argument('--marks', type=Path, help='Manual reference JSON. Omit for automatic calibration.')
    parser.add_argument('--reference-bank', type=Path, help='Optional approved venue reference views')
    parser.add_argument('--clock-offset', type=video.parse_time,
                        help='Timestamp of the first decoded frame in the original video')
    parser.add_argument('--player', type=int, nargs=3, metavar=('FRAME', 'X', 'Y'), action='append',
                        help='Optional player click for an identity and speed; repeat to disambiguate.')
    parser.add_argument('--output', type=Path, required=True, help='Output MP4 path, with JSON sidecar')
    parser.add_argument('--venue')
    for name in ('length', 'width', 'half_x', *pitch.STD_DIMS):
        parser.add_argument('--'+name.replace('_', '-'), type=float)
    parser.add_argument('--lens-offset-m', type=float, default=0.,
                        help='Signed distance from pivot toward the front of the lens, in metres')
    parser.add_argument('--max-p90-px', type=float, default=10.)
    parser.add_argument('--max-check-gap-s', type=float, default=2.5)
    parser.add_argument('--show-unverified-overlay', action='store_true',
                        help='Draw unverified source projections in orange for debugging')
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


def run(args):
    config = json.loads(args.marks.read_text(encoding='utf-8')) if args.marks else {}
    if (args.output.suffix != '.mp4' or (args.source and args.output.resolve() == args.source.resolve())
            or (args.marks and args.output.with_suffix('.json').resolve() == args.marks.resolve())
            or (args.marks and args.reference_bank)
            or (args.job and (args.start is not None or args.end is not None))
            or not all(np.isfinite(x) and x > 0 for x in [args.max_p90_px, args.max_check_gap_s])):
        raise ValueError('Check the output path, source trim, and validation thresholds.')
    meta = _venue(args, config)
    marks, holdout = _marks(config, 'fit'), _marks(config, 'holdout')
    fit_edges = {int(k): points for k, points in config.get('fit_far_edges', {}).items()}
    check_edges = {int(k): points for k, points in config.get('holdout_far_edges', {}).items()}
    if args.marks and (not holdout or marks.keys() & holdout.keys()):
        raise ValueError('Provide holdout marks on frames excluded from fitting.')
    if marks.keys() & check_edges.keys():
        raise ValueError('Holdout surface edges must be on frames excluded from fitting.')
    with ExitStack() as stack:
        if args.job:
            job = pipeline.Job(args.job)
            frames, boxes = job.frames(), job.dets()
            cumulative, sky = job.cam()
            inliers = job.camera_inliers()
        else:
            directory = Path(stack.enter_context(TemporaryDirectory(prefix='pitch-replay-')))
            decoded = str(directory/'frames.npy')
            video.decode(str(args.source), decoded, args.start, args.end)
            frames = video.load(decoded)
            cumulative, inliers, sky = camera.solve(frames)
            boxes = detect.run(frames)

        first, last = config.get('render_frames', [0, len(frames)-1])
        if not 0 <= first <= last < len(frames):
            raise ValueError('render_frames falls outside the decoded clip.')
        if (len(boxes) != len(frames) or cumulative.shape != (len(frames), 3, 3)
                or (inliers is not None and len(inliers) != len(frames))):
            raise ValueError('The cached motion and detections must match the decoded frames.')
        max_gap = round(args.max_check_gap_s*video.FPS)
        reason = None
        if args.marks:
            calibration = fixed_pitch.fit(cumulative, inliers, meta, marks, args.lens_offset_m, fit_edges,
                                          images=frames, boxes=boxes)
            errors = fixed_pitch.errors(calibration, holdout)
            edge_errors = fixed_pitch.far_edge_errors(calibration, check_edges)
            valid = fixed_pitch.supported_frames(calibration, errors, args.max_p90_px, max_gap, edge_errors)
            fit_errors = fixed_pitch.errors(calibration, marks)
        else:
            result = auto_pitch.calibrate(frames, boxes, cumulative, inliers, meta, sky,
                                          args.lens_offset_m, args.max_p90_px, max_gap,
                                          venue=meta.get('id'), reference_bank=args.reference_bank)
            calibration, valid = result.calibration, result.supported
            fit_errors, errors, edge_errors = result.fit_error, result.paint_checks, result.edge_checks
            reason = result.reason
        clicks = {int(k): p for k, p in config.get('player_clicks', {}).items()}
        clicks.update({k: (x, y) for k, x, y in (args.player or [])})
        if any(not 0 <= k < len(frames) for k in clicks):
            raise ValueError('A player click is outside the decoded clip.')
        selected = track.track(frames, boxes, cumulative, clicks) if clicks and valid[first:last+1].any() else None
        source_offset = args.clock_offset
        if source_offset is None:
            source_offset = config.get('source_time_at_frame_zero_s', args.start or 0)
        records = _replay_data(calibration, valid, boxes, selected, first, last, source_offset)
        camera_info = None if calibration is None else {
            'reference_frame': int(calibration.reference),
            'pivot_x_y_height_m': calibration.pivot.tolist(),
            'lens_offset_m': calibration.lens_offset_m, 'focal_px': calibration.focal_px,
        }
        report = {'pitch': meta, 'fps': video.FPS,
                  'mode': 'manual' if args.marks else 'automatic', 'camera': camera_info, 'reason': reason,
                  'max_p90_px': args.max_p90_px, 'max_check_gap_frames': max_gap,
                  'fit_error': fit_errors, 'holdout_error': errors,
                  'far_touchline_above_surface_p90_px': edge_errors,
                  'pitch_to_image': calibration.matrices[first:last+1].tolist() if calibration else None,
                  'supported_frames': int(valid[first:last+1].sum()), 'frames': records}
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False)
        _render(frames, calibration, records, args.output, args.show_unverified_overlay, meta=meta)
        args.output.with_suffix('.json').write_text(payload, encoding='utf-8')
        print(f'{report["supported_frames"]}/{len(records)} frames passed independent paint checks.')
        if reason:
            print(reason)


if __name__ == '__main__':
    main()
