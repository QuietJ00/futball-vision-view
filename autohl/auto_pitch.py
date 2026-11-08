"""Automatic labelled paint observations for the fixed-tripod replay."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from . import camera, field_geometry, field_solver, fixed_pitch, hybrid_calib, pitch


@dataclass
class Result:
    calibration: fixed_pitch.Calibration | None
    supported: np.ndarray
    fit_error: dict
    paint_checks: dict
    edge_checks: dict
    reason: str | None = None


PAINT_FEATURES = (0, 3, 4, 5, 6, 7, 11, 12, 13, 17, 18, 19)


def _active_x_bounds(image):
    col_max = image.max(axis=(0, 2))
    active = np.flatnonzero(col_max > 8)
    if len(active) < image.shape[1] * 0.5:
        return 0, image.shape[1]
    return int(active[0]), int(active[-1]) + 1


def _ridge(image, boxes, sky=0):
    x0, x1 = _active_x_bounds(image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grey -= .30*hsv[..., 1]
    ridge = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    coloured = ((hsv[..., 0] < 35) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 65)
    coloured = cv2.dilate(coloured.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ridge[coloured] = 0
    ridge[:max(16, sky), :] = 0
    ridge[-16:, :] = 0
    ridge[:, :16] = 0
    ridge[:, -16:] = 0
    if x0 > 0:
        ridge[:, :x0+16] = 0
    if x1 < image.shape[1]:
        ridge[:, x1-16:] = 0
    for x, y, x2, y2, *_ in boxes.astype(int):
        ridge[max(0, y-4):y2+5, max(0, x-4):x2+5] = 0
    return ridge


def _consistent(points, shifts, sample_indices, is_line, normal=None, total_samples=200):
    """Discard ridges that cannot describe one continuous painted stroke."""
    if is_line:
        n = len(points)
        if n < 2:
            return np.zeros(n, bool)
        i, j = np.triu_indices(n, 1)
        d = points[j] - points[i]
        length = np.linalg.norm(d, axis=1)
        valid = length >= 25.
        if not valid.any():
            valid = length > 1e-6
        i, j, d, length = i[valid], j[valid], d[valid], length[valid]
        normals = np.c_[-d[:, 1], d[:, 0]] / length[:, None]
        err = np.abs((points[None, :, :] - points[i, None, :]) @ normals[:, :, None])[..., 0]
        scores = np.maximum(0., 1. - (err / 2.5)**2).sum(axis=1)
        mask = err[int(np.argmax(scores))] <= 2.5
        if mask.sum() >= 3:
            pts_in = points[mask]
            mean = pts_in.mean(axis=0)
            _, _, vt = np.linalg.svd(pts_in - mean)
            n_vec = np.array([-vt[0, 1], vt[0, 0]])
            mask = np.abs((points - mean) @ n_vec) <= 2.6
        return mask
    theta = 2.*np.pi*sample_indices / max(1, total_samples-1)
    if normal is not None and len(normal) == len(shifts):
        design = np.c_[np.ones(len(theta)), normal[:, 0], normal[:, 1], np.cos(theta), np.sin(theta)]
    else:
        u = (sample_indices-sample_indices.mean()) / max(1, np.ptp(sample_indices))
        design = np.c_[np.ones(len(u)), u, u*u]
    weights = np.ones(len(shifts))
    for _ in range(4):
        r = np.sqrt(weights)
        beta = np.linalg.lstsq(design*r[:, None], shifts*r, rcond=None)[0]
        error = np.abs(design @ beta - shifts)
        weights = np.minimum(1., 2.5/np.maximum(error, 1e-6))
    return error <= 4.5


def _read_strokes(image, boxes, H, lines, sky, radius, ground=None):
    """Read locally coherent white ridges, named by the starting pitch geometry."""
    ridge = _ridge(image, boxes, sky=sky)
    height, width = ridge.shape
    found = {}
    near = _near_touchline(ridge, H, lines[1])
    if near is not None:
        found[1] = near
    H_inv = np.linalg.inv(H)
    ground_fit = None
    if ground is not None and len(ground) >= 2 and np.ptp(ground[:, 0]) > 50:
        ground_fit = np.polyfit(ground[:, 0], ground[:, 1], 1)
    projected_lines = {idx: camera.warp(lines[idx], H).astype(float) for idx in PAINT_FEATURES}
    for index in PAINT_FEATURES:
        r_cur = radius
        offsets = np.arange(-r_cur, r_cur+1, dtype=float)
        line = lines[index]
        pixels = projected_lines[index]
        tangent = np.gradient(pixels, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        r2 = (pixels[:, 0] - width/2)**2 + (pixels[:, 1] - height/2)**2
        corner_distorted = (pixels[:, 1] > .72*height) & (r2 > (.33*width)**2)
        in_bounds = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] > max(64, r_cur+4))
                     & (pixels[:, 0] < width-max(64, r_cur+4)) & (pixels[:, 1] > max(18, sky))
                     & (pixels[:, 1] < height-max(24, r_cur+2)) & (norm > .4) & (~corner_distorted))
        if in_bounds.sum() < 8:
            continue
        w0 = camera.warp(pixels, H_inv)
        w1 = camera.warp(pixels + [0., 1.], H_inv)
        mpp_v = np.linalg.norm(w1 - w0, axis=1)
        if not np.isfinite(mpp_v[in_bounds]).any() or np.nanmedian(mpp_v[in_bounds]) > .425:
            continue
        take = in_bounds & np.isfinite(mpp_v) & (mpp_v <= .45)
        if take.sum() < 8:
            continue
        idx_take = np.flatnonzero(take)
        p, v = pixels[take], tangent[take]/norm[take, None]
        normal = np.c_[-v[:, 1], v[:, 0]]
        other_pts = [pts[np.isfinite(pts).all(axis=1)] for idx, pts in projected_lines.items() if idx != index]
        if other_pts:
            other_all = np.vstack(other_pts)
            if len(other_all):
                d2 = ((p[:, None, :] - other_all[None, :, :])**2).sum(axis=2).min(axis=1)
                cross_mask = d2 < 10.**2
            else:
                cross_mask = np.zeros(len(p), bool)
        else:
            other_all = np.empty((0, 2))
            cross_mask = np.zeros(len(p), bool)
        samples = p[:, None]+normal[:, None]*offsets[None, :, None]
        profile = cv2.remap(ridge, samples[..., 0].astype(np.float32),
                            samples[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        # True strokes produce a coherent ridge along the projected feature.
        along = cv2.GaussianBlur(profile, (3, 9), 0)
        best = (along-.18*np.abs(offsets)).argmax(axis=1)
        strength = along[np.arange(len(p)), best]
        delta = offsets[best]
        coherence = np.abs(delta - cv2.medianBlur(delta.astype(np.float32).reshape(-1, 1), 5)[:, 0])
        contrast = along.max(axis=1) - along.mean(axis=1)
        xy = samples[np.arange(len(p)), best]
        if len(other_all):
            d2_xy = ((xy[:, None, :] - other_all[None, :, :])**2).sum(axis=2).min(axis=1)
            stolen = (np.abs(delta) >= 7) & ((d2_xy < 6.**2) | ((index >= 18) & (d2_xy < (delta**2))))
        else:
            stolen = np.zeros(len(p), bool)
        keep = (strength >= 8.0) & (contrast >= 3.5) & (coherence <= 3.0) & (np.abs(delta) < r_cur-1) & (~cross_mask) & (~stolen)
        if ground_fit is not None:
            bottom = ground_fit[0]*xy[:, 0] + ground_fit[1]
            keep &= xy[:, 1] >= bottom-2
        if keep.sum() < max(7, .33*len(p)):
            continue
        k_idx = np.flatnonzero(keep)
        points = samples[k_idx, best[keep]]
        d_keep = delta[keep]
        s_idx = idx_take[keep]
        n_keep = normal[keep]
        agree = _consistent(points, d_keep, s_idx, index < 17, normal=n_keep, total_samples=len(line))
        min_span = 60 if index < 17 else 75
        if agree.sum() < max(7, .55*len(points)) or np.ptp(points[agree], axis=0).max() < min_span:
            continue
        points = points[agree]
        found[index] = points[::max(1, len(points)//24)]
    return found


def _near_touchline(ridge, H, model_line):
    """Find the continuous near paint, including when it is clipped by the lower image edge."""
    endpoints = camera.warp(model_line[[0, -1]], H)
    a, b = endpoints[np.argsort(endpoints[:, 0])]
    if b[0]-a[0] < 100:
        return None
    slope = (b[1]-a[1])/(b[0]-a[0])
    if abs(slope) > .55:
        return None
    height, width = ridge.shape
    x = np.arange(width)
    expected = a[1]+slope*(x-a[0])
    r2 = (x - width/2)**2 + (expected - height/2)**2
    corner_distorted = (expected > .72*height) & (r2 > (.31*width)**2)
    visible = (x >= max(64, a[0]-20)) & (x <= min(width-64, b[0]+20)) & (~corner_distorted)
    if not visible.any():
        return None
    strip = (np.abs(np.arange(height)[:, None]-expected) < 65.) & visible
    mask = np.uint8((ridge > 31) & strip)*255
    detected = cv2.HoughLinesP(mask, 1, np.pi/720, 45, minLineLength=160, maxLineGap=24)
    if detected is None:
        return None
    best = None
    for x1, y1, x2, y2 in detected.reshape(-1, 4):
        if x2 == x1:
            continue
        angle = (y2-y1)/(x2-x1)
        if abs(angle-slope) > .055:
            continue
        sy = y1+angle*(x-x1)
        cols = np.flatnonzero(visible & (sy >= 4) & (sy < height-4))[::6]
        if len(cols) < 25:
            continue
        rows = np.rint(sy[cols]).astype(int)
        band = rows[:, None]+np.arange(-3, 4)
        values = ridge[band, cols[:, None]]
        peaks = values.max(axis=1)
        use = peaks > 31
        if use.sum() < max(25, .55*len(cols)) or np.median(peaks[use]) < 37:
            continue
        p = np.c_[cols[use], band[use, values[use].argmax(axis=1)]]
        rank = float(np.minimum(peaks[use], 85).sum())
        if best is None or rank > best[0]:
            best = rank, p
    return best[1][::max(1, len(best[1])//24)] if best is not None else None


def _edge(image, boxes, H, width_m, length_m=100.):
    """Find a long coherent red-track boundary in a broad band around the far side."""
    normal = np.linalg.solve(H.T, [0., 1., width_m/2])
    if abs(normal[1]) < 1e-9*np.linalg.norm(normal[:2]):
        return None
    height, width = image.shape[:2]
    b, g, r = (channel.astype(np.float32) for channel in cv2.split(image))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.blur(np.float32((r-g > 14) & (r-b > 9)), (29, 1)) >= .6
    turf_below = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 90)
                  & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40) & (hsv[..., 2] <= 225))
    H_inv = np.linalg.inv(H)
    slope = -normal[0]/normal[1]
    x_lo = max(24, int(.073*width))
    x_hi = min(width-24, int(.906*width))
    points = []
    for x in range(x_lo, x_hi, 24):
        y = slope*x-normal[2]/normal[1]
        if not (24 <= y < height-30):
            continue
        wp = camera.warp(np.array([[float(x), float(y)]]), H_inv)[0]
        if not (np.isfinite(wp).all() and 12. <= wp[0] <= length_m-12.):
            continue
        lo, hi = max(0, int(y)-55), min(height, int(y)+55)
        if hi-lo < 8:
            continue
        rows = np.flatnonzero(red[lo:hi, x])+lo
        for group in np.split(rows, np.flatnonzero(np.diff(rows) > 2)+1):
            if 4 <= len(group) <= 38 and group[-1] < hi-1:
                yc = int(group[-1])
                patch = turf_below[min(height, yc+6):min(height, yc+22), max(0, x-10):min(width, x+11)]
                if patch.size and patch.mean() >= .55:
                    points.append((x, yc))
    if not points:
        return None
    p = np.asarray(points)
    for x, y, x2, y2, *_ in boxes:
        p = p[~((p[:, 0] >= x-7) & (p[:, 0] <= x2+7)
                & (p[:, 1] >= y-7) & (p[:, 1] <= y2+7))]
    if len(p) < 10:
        return None
    # Distinct long strips win over banners or the edge of a player's kit.
    i, j = np.triu_indices(len(p), 1)
    dx = p[j, 0]-p[i, 0]
    slopes = (p[j, 1]-p[i, 1])/np.maximum(dx, 1)
    use = (dx >= 320) & (np.abs(slopes-slope) <= .048)
    if not use.any():
        return None
    slopes = slopes[use]
    intercepts = p[i[use], 1]-slopes*p[i[use], 0]
    diff = np.abs(slopes[:, None]*p[:, 0]+intercepts[:, None]-p[:, 1])
    best = diff[np.argmax((diff <= 3.0).sum(axis=1))]
    found = p[best <= 3.0]
    if len(found) < 10 or np.ptp(found[:, 0]) < 350:
        return None
    return found[::max(1, len(found)//20)]


def _diverse(lines):
    along = set(lines) & set(field_geometry.ALONG_LINES)
    across = set(lines) & set(field_geometry.ACROSS_LINES)
    return bool(across and (along or 17 in lines))


def build_bank(images, reviewed, destination, step=40):
    """Save approved reference views and projections for later shots at this venue."""
    matrices = reviewed['pitch_to_image']
    records = reviewed['frames']
    if not matrices or len(matrices) != len(records):
        raise ValueError('A reference bank needs a reviewed replay with frame projections.')
    views = [(r['frame'], H) for r, H in zip(records, matrices) if r['field_supported']]
    if len(views) < 2:
        raise ValueError('The reviewed clip has too few supported frames.')
    chosen = [views[0]]
    for item in views[1:]:
        if item[0]-chosen[-1][0] >= step:
            chosen.append(item)
    if chosen[-1][0] != views[-1][0]:
        chosen.append(views[-1])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    refs = []
    for frame, H in chosen:
        name = f'frame_{frame:05d}.jpg'
        if not cv2.imwrite(str(destination/name), images[frame], [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise OSError(f'Could not write reference view {name}.')
        refs.append({'image': name, 'pitch_to_image': H})
    (destination/'manifest.json').write_text(json.dumps(
        {'pitch': reviewed['pitch'], 'references': refs}, ensure_ascii=False), encoding='utf-8')
    return len(refs)


def _scene_feats(image, sky, boxes=None):
    if pitch._sift is None:
        pitch._sift = cv2.SIFT_create(4000, contrastThreshold=.02)
    mask = camera.plane_mask(image, sky)
    if boxes is not None and len(boxes):
        for x, y, x2, y2, *_ in np.asarray(boxes, dtype=int):
            mask[max(0, y-8):y2+9, max(0, x-8):x2+9] = 0
    grey = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)
    return pitch._sift.detectAndCompute(grey, mask)


def _bank_seeds(images, indices, bank, meta, sky, cumulative, boxes=None):
    if not bank:
        return {}
    root = Path(bank)
    manifest = root / 'manifest.json'
    if not manifest.exists():
        return {}
    index = json.loads(manifest.read_text(encoding='utf-8'))
    if any(index['pitch'][key] != meta[key] for key in ('length', 'width')):
        raise ValueError('Reference bank pitch dimensions do not match this job.')
    refs = []
    for ref in index['references']:
        image = cv2.imread(str(root/ref['image']))
        if image is None:
            raise ValueError(f'Missing reference image {ref["image"]}.')
        refs.append((pitch._feats(image, 0), _scene_feats(image, 0), np.asarray(ref['pitch_to_image'])))
    candidates = {}
    for k in indices[::24]:
        target_turf = pitch._feats(images[k], sky)
        target_scene = _scene_feats(images[k], sky, boxes[k] if boxes is not None else None)
        matches = []
        for turf_f, scene_f, H in refs:
            for feat_ref, feat_tgt in ((turf_f, target_turf), (scene_f, target_scene)):
                M, count = pitch._match_features(feat_ref, feat_tgt)
                if M is not None:
                    P = M @ H
                    score = pitch.line_score(images[k], P, meta, sky)
                    if score >= .39:
                        matches.append((score, count, P))
        if matches:
            _, _, candidates[k] = max(matches, key=lambda m: (m[0], m[1]))
    # A single unrelated photo match is insufficient evidence for reference transfer.
    trusted = {}
    grid = pitch.GRID * [meta['length']/100, meta['width']/60]
    for k, H in candidates.items():
        if any(j != k and abs(j-k) <= 75
               and pitch._gap(pitch.chain(cumulative, k, j, H), P, sky, grid) <= 35
               for j, P in candidates.items()):
            trusted[k] = H
    return trusted


def _collect_observations(images, boxes, training, checking, pose_fn, lines, meta, sky, radius, allowed_lines=None):
    fit, held, fit_edge, held_edge, initial = {}, {}, {}, {}, {}
    length_m = meta.get('length', 100.)
    for k in sorted(set(training) | set(checking)):
        H = pose_fn(k)
        ground = _edge(images[k], boxes[k], H, meta['width'], length_m=length_m)
        observed = _read_strokes(images[k], boxes[k], H, lines, sky, radius=radius, ground=ground)
        if allowed_lines is not None:
            observed = {l: pts for l, pts in observed.items() if l in allowed_lines}
        if k in training:
            if _diverse(observed):
                fit[k], initial[k] = observed, H
                if ground is not None:
                    fit_edge[k] = ground
        else:
            if _diverse(observed):
                held[k] = observed
            if ground is not None:
                held_edge[k] = ground
    return fit, held, fit_edge, held_edge, initial


def calibrate(images, boxes, cumulative, inliers, meta, sky, lens_offset_m=0.,
              threshold_px=10., max_gap_frames=45, log=print, venue=None, reference_bank=None):
    """Fit a continuous shot without target-clip clicks. Return a refusal on weak evidence."""
    segments = camera.motion_segments(inliers, len(images))
    cumulative = camera.repair_cumulative(cumulative)
    if images is not None and len(images) == len(cumulative):
        cumulative = camera.refine_turf_motion(images, boxes, cumulative, sky)
        cumulative = camera.repair_cumulative(cumulative)

    def fail(message):
        return Result(None, np.zeros(len(images), bool), {}, {}, {}, message)

    if not len(images) or np.count_nonzero(np.bincount(segments) >= 2) != 1:
        return fail('Use one continuous camera shot for automatic calibration.')
    indices = np.flatnonzero(segments == np.argmax(np.bincount(segments)))
    seeds = _bank_seeds(images, indices, reference_bank, meta, sky, cumulative, boxes=boxes) if reference_bank else {}
    if seeds:
        log(f'검증된 구장 기준 화면과 일치한 프레임 {len(seeds)}개')
    else:
        seeds = field_solver.solve(images, cumulative, meta, boxes, indices, log=log)
    if not seeds and venue in pitch.venues():
        reference = pitch.auto(images, cumulative, venue, sky, log=log, boxes=boxes, inliers=inliers)
        if reference:
            seeds = {k: H for k, H, *_ in reference['bases'] if segments[k] == segments[indices[0]]}

    is_broadcast = False
    if not seeds:
        from . import broadcast_pitch
        try:
            cum_broadcast, _ = camera.turf_motion(images, boxes, cumulative, smooth_grass=True)
            seeds = broadcast_pitch.solve(images, cum_broadcast, meta, boxes, indices, log=log)
            if seeds:
                is_broadcast = True
                cumulative = cum_broadcast
        except (ValueError, np.linalg.LinAlgError) as error:
            log(f'중계 카메라 분석 실패: {error}')
            seeds = {}

    if not seeds:
        return fail('시작 카메라 없음: 일치하는 경기장 선을 찾지 못함')

    if is_broadcast:
        matrices = np.zeros((len(images), 3, 3))
        keys = sorted(seeds)
        for k in indices:
            j = np.searchsorted(keys, k)
            if j == 0:
                H = pitch.chain(cumulative, keys[0], k, seeds[keys[0]])
            elif j >= len(keys):
                H = pitch.chain(cumulative, keys[-1], k, seeds[keys[-1]])
            else:
                a, b = keys[j-1], keys[j]
                t = (k - a) / (b - a)
                Ha = pitch.chain(cumulative, a, k, seeds[a])
                Hb = pitch.chain(cumulative, b, k, seeds[b])
                H = pitch.blend(Ha, Hb, t)
            matrices[k] = H / H[2, 2]

        valid = np.ones(len(images), bool)

        res = {'bases': [(k, seeds[k], 0, 1.0) for k in keys],
               'mode': 'camera', 'meta': meta, 'valid': valid,
               'segments': segments, 'matrix': matrices,
               'paint_score': np.ones(len(images)), 'score': 1.0,
               'carried': np.zeros(len(images), bool), 'long_lens': True}

        calib_obj = pitch.Calib(res, cumulative)
        try:
            fitted = hybrid_calib.refine_long_lens(calib_obj, images, cumulative, meta, boxes)
        except Exception as error:
            log(f'중계 카메라 정밀 보정 실패: {error}')
            fitted = None

        if fitted is not None:
            for k in np.flatnonzero(valid):
                try:
                    matrices[k] = fitted.img_from_pitch(k)
                except Exception:
                    valid[k] = False
            focal = float(fitted.focal if hasattr(fitted, 'focal') and fitted.focal else 5000.)
        else:
            focal = 5000.

        on_pitch_share = fixed_pitch.player_on_pitch(matrices, boxes, meta, margin=8.0)
        log(f'선수 경기장 내 비율: {on_pitch_share:.1%}')
        if on_pitch_share < 0.60:
            valid[:] = False
            return Result(None, valid, {}, {}, {}, f'선수가 경기장 밖에 놓여요 ({on_pitch_share:.0%})')

        ref = min(keys)
        from . import camfit
        focal_val, _, pivot = camfit.decompose(matrices[ref])
        c = fixed_pitch.Calibration(meta, ref, pivot, lens_offset_m, focal_val, matrices, segments == segments[ref])
        return Result(c, valid, {}, {}, {}, None if valid.any() else '중계 카메라 검증 실패')

    lines = pitch.model_lines(meta)
    start, end = map(int, indices[[0, -1]])
    # Interleaved frames provide separate image evidence for the quality check.
    training = set(map(int, np.linspace(start, end, min(16, max(3, len(indices)//24)), dtype=int)))
    checking = sorted((set(range(start+6, end, 12)) | {start, end}) - training)

    def seed_pose(k):
        a = min(seeds, key=lambda i: abs(i-k))
        return pitch.chain(cumulative, a, k, seeds[a])

    fit, held, fit_edge, held_edge, initial = _collect_observations(
        images, boxes, training, checking, seed_pose, lines, meta, sky, radius=24)
    if len(fit) < 2 or not held:
        return fail('Too few independently visible pitch markings.')
    starts = {k: initial[k] for k in (min(initial), max(initial), max(fit, key=lambda k: len(fit[k])))}
    try:
        c = fixed_pitch.fit(cumulative, inliers, meta, fit, lens_offset_m, fit_edge,
                            seed_homographies=starts)
        # Re-read strokes tightly around the first-pass trajectory to eliminate initial seed drift.
        fit2, _, fit_edge2, _, initial2 = _collect_observations(
            images, boxes, training, (), lambda k: c.matrices[k], lines, meta, sky, radius=22)
        if len(fit2) >= 2:
            starts2 = {k: initial2[k] for k in (min(initial2), max(initial2), max(fit2, key=lambda k: len(fit2[k])))}
            c = fixed_pitch.fit(cumulative, inliers, meta, fit2, lens_offset_m, fit_edge2,
                                images=images, boxes=boxes, seed_homographies=starts2)
            fit, fit_edge = fit2, fit_edge2
        fit_lines = {l for obs in fit.values() for l in obs}
        _, held2, _, held_edge2, _ = _collect_observations(
            images, boxes, (), checking, lambda k: c.matrices[k], lines, meta, sky, radius=22,
            allowed_lines=fit_lines)
        if held2:
            held, held_edge = held2, held_edge2
    except (ValueError, np.linalg.LinAlgError) as e:
        return fail(str(e))
    log(f'구장선 자동 보정 {len(fit)}개 프레임 · 별도 검사 {len(held)}개 프레임')
    on_pitch_share = fixed_pitch.player_on_pitch(c.matrices, boxes, meta)
    log(f'선수 경기장 내 비율: {on_pitch_share:.1%}')
    if on_pitch_share < 0.85:
        valid = np.zeros(len(images), bool)
        return Result(c, valid, fixed_pitch.errors(c, fit), fixed_pitch.errors(c, held), {},
                      f'선수가 경기장 밖에 놓여요 ({on_pitch_share:.0%})')
    paint = fixed_pitch.errors(c, held)
    edges = fixed_pitch.far_edge_errors(c, held_edge)
    valid = fixed_pitch.supported_frames(c, paint, threshold_px, max_gap_frames, edges, boxes=boxes)
    return Result(c, valid, fixed_pitch.errors(c, fit), paint, edges,
                  None if valid.any() else 'Automatic paint and surface checks rejected the projection.')
