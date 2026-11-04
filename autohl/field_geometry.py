"""Map a clip onto the pitch: automatic key frames or user marks, checked against paint."""

from functools import lru_cache
from itertools import combinations
import os

import cv2
import numpy as np

from . import field_solver, hybrid_calib, pitch
from .camera import distort, undistort, warp

MANUAL_MIN_PAINT = .19
MIN_AXIS_PAINT = .24
CONFLICT_MARGIN = .04
MAX_TURF_SLIP = 1.  # pixels per frame a refined camera may move the lines over the turf
ALONG_LINES = [1, 2, 5, 7, 8, 10, 11, 13, 14, 16]
ACROSS_LINES = [0, 3, 4, 6, 9, 12, 15]


@lru_cache(maxsize=1)
def _kernels():
    return pitch._line_kernels(L=25, off=3, n=32)


def _evidence(image, boxes):
    # At half resolution, a small directional filter retains white paint on both bright
    # and dark turf while suppressing broad reflections. Yellow training lines are excluded.
    small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey -= .35 * hsv[..., 1]
    top = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    turf = pitch.turf_mask(small)
    # On illuminated pitches the basic green mask can join the pitch to a dark green fence.
    # Recover a separate connected component of well-lit grass when it covers enough turf.
    brightness = hsv[..., 2]
    if turf.any():
        cutoff = max(70, .82 * np.median(brightness[turf != 0]))
        bright = turf & (np.uint8(brightness >= cutoff) * 255)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
        if n > 1:
            k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            main = np.uint8(labels == k) * 255
            if main.sum() > .65 * turf.sum():
                turf = main
    turf = cv2.dilate(turf, np.ones((21, 21), np.uint8)) > 0
    yellow = (hsv[..., 0] > 13) & (hsv[..., 0] < 40) & (hsv[..., 1] > 80)
    top[~turf | yellow] = 0
    for x1, y1, x2, y2, *_ in boxes:
        a, b, c, d = np.rint(np.array([x1, y1, x2, y2]) / 2).astype(int)
        top[max(0, b-2):d+3, max(0, a-2):c+3] = 0
    ridge = np.maximum.reduce([cv2.filter2D(top, -1, k) for k in _kernels()])
    return ridge, turf


def _observed_lines(ridge, turf):
    # Search horizontal and oblique strokes at multiple strengths. Reserve capacity for far
    # paint, which can be shorter and weaker than the near touchline by a large factor.
    rows = np.flatnonzero(turf.mean(axis=1) > .55)
    if not len(rows):
        return np.empty((0, 3))
    far_y = rows[0] + 65
    hough = []
    for threshold, minimum in ((19, 42), (12, 36)):
        mask = np.uint8((ridge >= threshold) & turf) * 255
        lines = cv2.HoughLinesP(mask, 1, np.pi/540, 30, minLineLength=minimum, maxLineGap=11)
        if lines is not None:
            hough.extend(lines.reshape(-1, 2, 2))
    ranked = []
    for a, b in hough:
        v = b - a
        length = np.linalg.norm(v)
        points = np.rint(np.linspace(a, b, 24)).astype(int)
        strength = np.mean(np.clip(ridge[points[:, 1], points[:, 0]], 0, 40))
        normal = np.array([-v[1], v[0]], dtype=float) / length
        normal *= 1 if normal[1] > 0 or (normal[1] == 0 and normal[0] > 0) else -1
        # Vertical fence posts can be strong above the grass but are rarely valid paint.
        midpoint = (a+b)/2
        if midpoint[1] < rows[0] - 10:
            continue
        ranked.append((length**.7 * strength, np.r_[normal, -normal @ a], midpoint))
    kept = []
    for _, line, mid in sorted(ranked, key=lambda r: -r[0]):
        if any(abs(line[:2] @ other[:2]) > .998 and abs(line[:2] @ p + line[2]) < 4
               and abs(other[:2] @ mid + other[2]) < 4 for other, p in kept):
            continue
        kept.append((line, mid))
    near = [line for line, p in kept if p[1] > far_y][:12]
    far = [line for line, p in kept if p[1] <= far_y][:18]
    return np.asarray(near + far).reshape(-1, 3)


def _rectangles(meta):
    d = pitch.dims(meta)
    length, w = meta['length'], meta['width']/2
    x = [0, d['ga_depth'], d['box_depth'], meta.get('half_x', length/2),
         length-d['box_depth'], length-d['ga_depth'], length]
    y = [-w, -d['box_half'], -d['ga_half'], d['ga_half'], d['box_half'], w]
    return np.asarray([(a, b, c, e) for a, b in combinations(sorted(set(x)), 2)
                       for c, e in combinations(sorted(set(y)), 2) if b-a >= 4 and e-c >= 4])


def _hypotheses(lines, meta):
    """Use two pairs of pitch lines to propose homographies, with pinhole camera constraints."""
    if len(lines) < 4:
        return np.empty((0, 3, 3))
    pairs = np.array(list(combinations(range(len(lines)), 2)))
    a, b = np.triu_indices(len(pairs), 1)
    ii, jj = pairs[a], pairs[b]
    distinct = (ii[:, :, None] != jj[:, None, :]).all(axis=(1, 2))
    ii, jj = ii[distinct], jj[distinct]
    corners = np.cross(lines[ii][:, :, None], lines[jj][:, None, :])
    den = corners[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = corners[..., :2] / den[..., None]
    good = np.isfinite(xy).all(axis=(1, 2, 3)) & (np.abs(xy-[480, 270]).max(axis=(1, 2, 3)) < 1250)
    xy = xy[good]
    if not len(xy):
        return np.empty((0, 3, 3))
    dx = xy[:, 1] - xy[:, 0]
    dy = xy[:, :, 1] - xy[:, :, 0]
    good = (np.linalg.norm(dx, axis=-1).min(1) > 20) & (np.linalg.norm(dy, axis=-1).min(1) > 15)
    good &= (dx[:, 0] * dx[:, 1]).sum(1) > 0
    good &= (dy[:, 0] * dy[:, 1]).sum(1) > 0
    xy = xy[good]
    if not len(xy):
        return np.empty((0, 3, 3))
    # Each vanishing direction can correspond to either pitch axis. Resolve mirror symmetry
    # in the local shot by putting +X to the right and +Y toward the near touchline.
    xy = np.concatenate((xy, xy.transpose(0, 2, 1, 3)))
    flip_x = xy[:, 1, :, 0].mean(1) < xy[:, 0, :, 0].mean(1)
    xy[flip_x] = xy[flip_x, ::-1]
    flip_y = xy[:, :, 1, 1].mean(1) < xy[:, :, 0, 1].mean(1)
    xy[flip_y] = xy[flip_y, :, ::-1]
    unit = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    unit_to_img = np.array([cv2.getPerspectiveTransform(unit, q.astype(np.float32)) for q in xy.reshape(-1, 4, 2)])
    # Orthogonality of the ground directions yields focal length for a centered phone camera.
    # A broad range also covers digital zoom and moderate principal point shifts.
    v = unit_to_img[:, :, :2].copy()
    v[:, :2] -= np.array([480, 270])[None, :, None] * v[:, 2:3]
    with np.errstate(divide='ignore', invalid='ignore'):
        focal2 = -(v[:, :2, 0] * v[:, :2, 1]).sum(1) / (v[:, 2, 0]*v[:, 2, 1])
    good = (focal2 > 260**2) & (focal2 < 4200**2)
    # Near-parallel vanishing lines do not constrain focal length reliably.
    near_infinity = np.min(np.abs(v[:, 2]) / np.maximum(np.linalg.norm(v[:, :2], axis=1), 1), axis=1) < .00012
    good |= near_infinity
    v, focal2, unit_to_img = v[good], focal2[good], unit_to_img[good]
    focal2 = np.where((focal2 > 260**2) & (focal2 < 4200**2), focal2, 750**2)
    v[:, :2] /= np.sqrt(focal2)[:, None, None]
    norms = np.linalg.norm(v, axis=1)
    ratio = norms[:, 0] / np.maximum(norms[:, 1], 1e-10)
    rects = _rectangles(meta)
    widths = rects[:, 1] - rects[:, 0]
    depths = rects[:, 3] - rects[:, 2]
    match = np.abs(np.log(np.maximum(ratio, 1e-9)[:, None] / (widths/depths))) < .22
    i, j = np.nonzero(match)
    H = unit_to_img[i].copy()
    H[:, :, 0] /= widths[j, None]
    H[:, :, 1] /= depths[j, None]
    H[:, :, 2] -= H[:, :, 0]*rects[j, 0, None] + H[:, :, 1]*rects[j, 2, None]
    f = np.sqrt(focal2[i])
    K = H.copy()
    K[:, :2] -= np.array([480, 270])[None, :, None] * K[:, 2:3]
    K[:, :2] /= f[:, None, None]
    x, y, t = K.transpose(2, 0, 1)
    norm = np.linalg.norm(x, axis=1)
    normal = np.cross(x, y)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    height = np.abs((normal*t).sum(1)) / np.maximum(norm, 1e-12)
    camera_y = -(y*t).sum(1) / np.maximum((y*y).sum(1), 1e-12)
    good = ((height > 1) & (height < 45) & (camera_y > meta['width']/2 - 6)
            & (np.abs(np.linalg.det(H)) > 1e-9))
    return H[good]


def _occupancy(H, meta, turf):
    # A collapsed or off-screen field can otherwise fit several short white lines.
    y, x = np.mgrid[0:540:18, 0:960:24]
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)[turf[y, x]]
    if len(pixels) < 24:
        return np.zeros(len(H))
    world = np.linalg.inv(H) @ pixels.T
    z = world[:, 2]
    p = world[:, :2] / np.maximum(z[:, None], 1e-8)
    half = meta['width']/2
    inside = (z > 0) & (p[:, 0] >= -2) & (p[:, 0] <= meta['length']+2) & (np.abs(p[:, 1]) <= half+2)
    return inside.mean(1)


def _template(meta, density):
    lines = pitch.model_lines(meta)
    pts = np.stack([line[np.linspace(0, len(line)-1, density).astype(int)] for line in lines])
    return np.concatenate((pts, np.ones((*pts.shape[:2], 1))), axis=-1).astype(np.float32)


def _scores(H, template, evidence, turf):
    shape = template.shape[:2]
    proj = H.astype(np.float32) @ template.reshape(-1, 3).T
    z = proj[:, 2]
    uv = proj[:, :2] / np.maximum(z[:, None], 1e-8)
    x = np.clip(np.rint(uv[:, 0]), -1, 960).astype(np.int32)
    y = np.clip(np.rint(uv[:, 1]), -1, 540).astype(np.int32)
    visible = (z > 1e-8) & (x >= 0) & (x < 960) & (y >= 0) & (y < 540)
    x, y = x.clip(0, 959), y.clip(0, 539)
    score = np.where(visible, evidence[y, x], 0).reshape(-1, *shape)
    visible = visible.reshape(-1, *shape)
    per_line = score.sum(2) / np.maximum(visible.sum(2), 1)
    samples = visible.sum(2)
    endpoints = uv.reshape(-1, 2, *shape)
    extent = np.linalg.norm(np.clip(endpoints[..., 0], -1000, 2000) -
                            np.clip(endpoints[..., -1], -1000, 2000), axis=1)
    extent[:, -3:] = 100
    hit = (per_line >= .22) & (samples >= max(3, shape[1]//4)) & (extent*samples/shape[1] > 26)
    # Both coordinate directions and more than the four lines defining the proposal must
    # appear. A penalty for painted lines outside turf limits implausible matches.
    total = np.maximum(visible.sum(axis=(1, 2)), 1)
    outside = (visible.reshape(len(H), -1) & ~turf[y, x]).sum(1) / total
    value = score.sum(axis=(1, 2)) / total - .7*outside
    value += .025 * np.minimum(hit.sum(1), 9)
    independent = np.minimum(hit[:, ALONG_LINES].sum(1), hit[:, ACROSS_LINES].sum(1))
    value -= .24 * (independent < 2)
    return value, hit.sum(1), independent


def propose(image, meta, boxes=(), diagnostics=None):
    ridge, turf = _evidence(image, boxes)
    lines = _observed_lines(ridge, turf)
    H = _hypotheses(lines, meta)
    if diagnostics is not None:
        diagnostics.update(lines=lines, proposals=len(H))
    if not len(H):
        return []
    # Rank at coarse resolution, then evaluate the strongest candidates against denser paint.
    signal = np.clip((ridge-5)/23, 0, 1)
    background = cv2.blur(signal, (17, 17))
    paint = cv2.dilate(signal, np.ones((5, 5), np.uint8)) - .95*background
    template = _template(meta, 12)
    rank = np.concatenate([_scores(chunk, template, paint, turf)[0]
                           for chunk in np.array_split(H, max(1, len(H)//256))])
    n = min(1800, len(rank))
    H = H[np.argpartition(rank, -n)[-n:]]
    paint = cv2.dilate(signal, np.ones((3, 3), np.uint8)) - .95*background
    score, support, directions = _scores(H, _template(meta, 32), paint, turf)
    occupied = _occupancy(H, meta, turf)
    score += .38 * occupied
    order = np.argsort(-score)
    return [(np.diag([2., 2., 1.]) @ H[i], float(score[i]), int(support[i]))
            for i in order[:180] if directions[i] >= 2 and support[i] >= 5 and occupied[i] >= .55]

def _automatic_anchor(image, meta, boxes):
    candidates = propose(image, meta, boxes)
    if not candidates:
        return None
    H, score, support = candidates[0]
    # The broad search can always fit clutter. Accept only a well-supported geometry
    # that clearly beats the nearest genuinely different field placement.
    if score < .86 or support < 8 or abs(H[2, 2]) < 1e-9:
        return None
    grid = pitch.GRID * [meta['length']/100, meta['width']/60]
    for alternative, other_score, _ in candidates[1:]:
        if score-other_score >= .035:
            break
        if max(pitch._gap(H, alternative, 0, grid),
               pitch._gap(alternative, H, 0, grid)) > 30:
            return None
    H /= H[2, 2]
    refined = pitch.refine(image, H, meta, iters=20)
    if (pitch._gap(H, refined, 0, grid) < 15
            and pitch.line_score(image, refined, meta) > pitch.line_score(image, H, meta)):
        H = refined
    if pitch.line_score(image, H, meta) < .65:
        return None
    return H


def _fit_points(meta, frame, marks):
    world = pitch.landmarks(meta['length'], meta['width'], meta)
    try:
        names, x, y = zip(*marks)
        src = np.float32([world[name] for name in names])
        dst = np.float32(np.column_stack((x, y)))
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError('경기장 기준점의 이름과 위치를 다시 확인해 주세요.') from exc
    if (len(set(names)) < 4 or not np.isfinite(dst).all()
            or not ((dst >= 0) & (dst < [1920, 1080])).all()
            or cv2.contourArea(cv2.convexHull(dst)) < 1800):
        raise ValueError('화면에 넓게 퍼진 서로 다른 기준점을 4개 이상 찍어 주세요.')
    H = pitch.from_points(frame, list(zip(names, dst)), meta)
    error = np.linalg.norm(warp(src, H)-dst, axis=1)
    if (not np.isfinite(error).all() or abs(np.linalg.det(H)) < 1e-9
            or np.count_nonzero(error <= 9) < max(4, (len(error)+1)//2)):
        raise ValueError('기준점들이 같은 경기장 배치에 맞지 않아요. 표시 위치를 확인해 주세요.')
    return H


def fit_strokes(meta, strokes, initial):
    """Refit an initial homography to samples on named template strokes.

    Keys are indices in pitch.model_segments. Samples need only be on the named paint,
    so the annotator can use any visible part of a line or circle.
    """
    from scipy.optimize import least_squares

    features = pitch.model_segments(meta)
    observations = [(features[int(key)], np.asarray(pts, float))
                    for key, pts in strokes.items() if len(pts) >= 2]
    if len(observations) < 4:
        raise ValueError('서로 다른 페인트선에서 관측점을 표시해 주세요.')

    def residual(params):
        H = np.r_[params, 1.].reshape(3, 3)
        errors = []
        for (kind, *geometry), points in observations:
            p = warp(geometry if kind == 'line' else geometry[0], H)
            if kind == 'line':
                direction = p[1]-p[0]
                normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-8)
                errors.append((points-p[0]) @ normal)
            else:
                a, b = p[:-1], p[1:]
                delta = b-a
                q = points[:, None]-a
                t = np.clip((q*delta).sum(-1)/np.maximum((delta*delta).sum(-1), 1e-8), 0, 1)
                errors.append(np.linalg.norm(q-t[..., None]*delta, axis=-1).min(1))
        return np.concatenate(errors)

    initial = initial / initial[2, 2]
    result = least_squares(residual, initial.ravel()[:8], loss='soft_l1', f_scale=2,
                           x_scale='jac', diff_step=1e-4, max_nfev=130)
    return np.r_[result.x, 1.].reshape(3, 3)


def _paint_check(image, H, lines, sky):
    distance = cv2.distanceTransform(255-pitch.white_mask(image, sky), cv2.DIST_L2, 3)
    best = [0., 0.]
    hits = total = 0
    for i, line in enumerate(lines):
        projected = warp(line, H)
        projected = projected[pitch._visible(projected, sky)]
        if len(projected) < 5:
            continue
        x, y = np.rint(projected).astype(int).clip([0, 0], [1919, 1079]).T
        hit = int((distance[y, x] < 4.5).sum())
        hits += hit
        total += len(projected)
        if i < 17 and np.linalg.norm(projected[-1]-projected[0]) >= 100:
            axis = int(i in ACROSS_LINES)
            best[axis] = max(best[axis], hit/len(projected))
    return hits/max(total, 1), min(best)


def _project_map(F, CUM, meta, segments, anchors, sky, mode, log):
    # Validate paint periodically; adjacent frames receive coordinates only while motion is
    # continuous. Use a single homography (and its inverse) for each accepted frame.
    N = len(F)
    valid = np.zeros(N, dtype=bool)
    matrices = np.full((N, 3, 3), np.nan)
    quality = np.full(N, np.nan)
    grid = pitch.GRID * [meta['length']/100, meta['width']/60]
    lines = pitch.model_lines(meta)
    check_frames = set(range(0, N, 10)) | {N-1} | set(anchors)
    changes = np.flatnonzero(np.diff(segments))+1
    check_frames.update(changes)
    check_frames.update(changes-1)
    checks = {}
    for k in sorted(check_frames):
        same = [a for a in anchors if segments[k] == segments[a]]
        if not same:
            continue
        previous = max((a for a in same if a <= k), default=None)
        following = min((a for a in same if a >= k), default=None)
        sources = list(dict.fromkeys(a for a in (previous, following) if a is not None))
        proposals = [pitch.chain(CUM, a, k, anchors[a]) for a in sources]
        H = proposals[0]
        if len(proposals) == 2 and mode == 'camera':
            # Both keys are trusted and agree to within field_solver.MAX_KEY_GAP. Camera motion
            # alone drifts over a long pan, so ease from one to the other across the whole
            # stretch: the correction is too slow to see, and the lines stay on the turf.
            t = (k-sources[0])/(sources[1]-sources[0])
            H = pitch.blend(proposals[0], proposals[1], t)
            H = H/H[2, 2]
            checks[k] = (H, _paint_check(F[k], H, lines, sky)[0])
            log(f'페인트선 정합 확인 {k+1}/{N}')
            continue
        if len(proposals) == 2:
            H2 = proposals[1]
            gap = max(pitch._gap(H, H2, sky, grid), pitch._gap(H2, H, sky, grid))
            if gap <= 16:
                t = (k-sources[0])/(sources[1]-sources[0])
                H = pitch.blend(H, H2, t)
                H = H/H[2, 2]
            else:
                # Opposite parts of a pan may expose different paint. Select a conflicting
                # projection only if the current image gives it independent support.
                evidence = [_paint_check(F[k], P, lines, sky) for P in proposals]
                ranked = sorted(range(2), key=lambda i: evidence[i][0]+.15*evidence[i][1], reverse=True)
                winner, loser = ranked
                s, axis = evidence[winner]
                if axis < MIN_AXIS_PAINT or s-evidence[loser][0] < CONFLICT_MARGIN:
                    continue
                H = proposals[winner]
        score, axis = _paint_check(F[k], H, lines, sky)
        minimum = .46 if mode == 'geometry' else MANUAL_MIN_PAINT  # 'camera' and 'manual' share it
        if (score < minimum or axis < MIN_AXIS_PAINT) and k not in anchors:
            continue
        # Refinement has a small trust region, as distant parallel markings can otherwise
        # lure the result off its validated world geometry.
        # A solved camera is already held to the whole clip; a per-frame fit would only add jitter.
        if k not in anchors and mode != 'camera':
            refined = pitch.refine(F[k], H, meta, sky, iters=12)
            if pitch._gap(H, refined, sky, grid) <= 7:
                new_score, new_axis = _paint_check(F[k], refined, lines, sky)
                if new_score > score + .025 and new_axis >= MIN_AXIS_PAINT:
                    H, score = refined, new_score
        checks[k] = (H, score)
        log(f'페인트선 정합 확인 {k+1}/{N}')

    for k, (H, score) in checks.items():
        matrices[k], valid[k], quality[k] = H, True, score
    ordered = sorted(checks)
    for a, b in zip(ordered, ordered[1:]):
        if segments[a] != segments[b] or b-a > 10:
            continue
        H1, s1 = checks[a]
        H2, s2 = checks[b]
        mid = (a+b)//2
        p, q = pitch.chain(CUM, a, mid, H1), pitch.chain(CUM, b, mid, H2)
        if max(pitch._gap(p, q, sky, grid), pitch._gap(q, p, sky, grid)) > 12:
            continue
        for k in range(a+1, b):
            t = (k-a)/(b-a)
            matrices[k] = pitch.blend(pitch.chain(CUM, a, k, H1), pitch.chain(CUM, b, k, H2), t)
            valid[k], quality[k] = True, min(s1, s2)
    indices = np.flatnonzero(valid)
    bad = indices[~np.isfinite(matrices[indices]).all(axis=(1, 2))
                  | (np.abs(np.linalg.det(matrices[indices])) < 1e-9)]
    valid[bad], quality[bad], matrices[bad] = False, np.nan, np.nan
    adjacent = valid[:-1] & valid[1:] & (segments[:-1] == segments[1:])
    isolated = valid & ~np.r_[False, adjacent] & ~np.r_[adjacent, False]
    isolated[list(anchors)] = False
    valid[isolated], quality[isolated], matrices[isolated] = False, np.nan, np.nan
    if not valid.any():
        return None
    return {'bases': [(k, H, 0, checks[k][1]) for k, H in sorted(anchors.items()) if valid[k]],
            'mode': mode, 'meta': meta, 'valid': valid, 'segments': segments,
            'matrix': matrices, 'paint_score': quality, 'score': float(np.nanmean(quality))}


def turf_slip(matrices, valid, cumulative, k1=0., focal=1.):
    """Worst pixels per frame that pitch lines move over the turf. The ground does not move, so
    between frames a line must go exactly where the camera motion carries the turf under it."""
    y, x = np.mgrid[640:1040:40, 60:1880:60]
    grid = np.c_[x.ravel(), y.ravel()].astype(float)
    straight = undistort(grid, k1, focal)
    worst = 0.
    for k in np.flatnonzero(valid[:-1] & valid[1:]):
        ground = warp(straight, np.linalg.inv(matrices[k]))
        lines = distort(warp(ground, matrices[k+1]), k1, focal)
        turf = warp(grid, cumulative[k+1] @ np.linalg.inv(cumulative[k]))
        worst = max(worst, float(np.nanmedian(np.linalg.norm(lines-turf, axis=1))))
    return worst


def _with_one_camera(result, F, cumulative, meta, boxes, log):
    # Replace frame-by-frame matrices with one camera and lens. Keep the first answer if the fit
    # fails: it is coarser at the far side of the frame, but it is checked and usable.
    fitted = hybrid_calib.refine_camera(pitch.Calib(result, cumulative), F, cumulative, meta, boxes)
    if fitted is None:
        log('카메라 정밀 보정을 건너뛰었어요.')
        return result
    matrices = result['matrix'].copy()
    for k in np.flatnonzero(result['valid']):
        H = fitted.img_from_pitch(int(k))
        matrices[k] = H/H[2, 2]
    if turf_slip(matrices, result['valid'], cumulative, fitted.k1, fitted.focal) > MAX_TURF_SLIP:
        log('카메라 정밀 보정이 선을 흔들어서 쓰지 않았어요.')
        return result
    log(f'렌즈 왜곡 {fitted.k1:+.2f}, 초점거리 {fitted.focal:.0f} px')
    return {**result, 'matrix': matrices, 'k1': fitted.k1, 'focal': fitted.focal}


def solver_enabled():
    # Experimental: off unless asked for, until it is validated beyond the clips it was tuned on.
    return os.environ.get('AUTOHL_CALIBRATOR') == 'camera'


def calibrate(F, cumulative, meta, sky, *, inliers=None, boxes=None, marks=None,
              seed_homographies=None, log=print):
    """Solve the camera automatically, or from user-identified painted landmarks when given."""
    from .camera import motion_segments

    segments = motion_segments(inliers, len(F))
    anchors = dict(seed_homographies or {})
    for frame, points in (marks or {}).items():
        k = int(frame)
        if not 0 <= k < len(F):
            raise ValueError('영상 안의 프레임을 선택해 주세요.')
        anchors[k] = _fit_points(meta, k, points)
    mode = 'manual' if anchors else 'geometry'
    if not anchors and solver_enabled():
        for segment in np.unique(segments):
            frames = np.flatnonzero(segments == segment)
            if len(frames) >= 2:
                anchors.update(field_solver.solve(F, cumulative, meta, boxes, frames, log=log) or {})
        mode = 'camera' if anchors else mode
    if not anchors:
        # Try several views on each connected shot. Failure is explicit and cannot become
        # a geometry result merely because a single candidate scored above a threshold.
        for segment in np.unique(segments):
            frames = np.flatnonzero(segments == segment)
            if len(frames) < 2:
                continue
            count = min(8, max(2, 1+len(frames)//45), len(frames))
            found = {}
            for k in frames[np.linspace(0, len(frames)-1, count, dtype=int)]:
                H = _automatic_anchor(F[k], meta, boxes[k] if boxes is not None else ())
                log(f'영상에서 구장선 찾는 중 {k+1}/{len(F)}')
                if H is not None:
                    found[k] = H
            grid = pitch.GRID * [meta['length']/100, meta['width']/60]
            for a, b in combinations(found, 2):
                p = pitch.chain(cumulative, a, b, found[a])
                q = found[b]
                if max(pitch._gap(p, q, sky, grid), pitch._gap(q, p, sky, grid)) <= 16:
                    anchors[a], anchors[b] = found[a], found[b]
    if not anchors:
        return None
    result = _project_map(F, cumulative, meta, segments, anchors, sky, mode, log)
    return _with_one_camera(result, F, cumulative, meta, boxes, log) if mode == 'camera' and result else result
