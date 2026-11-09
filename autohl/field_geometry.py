"""Map a clip onto the pitch: automatic key frames or user marks, checked against paint."""

from functools import lru_cache
from itertools import combinations
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import broadcast_pitch, field_solver, hybrid_calib, pitch
from .camera import distort, pinhole_motion, shot_cuts, turf_motion, undistort, warp

MANUAL_MIN_PAINT = .19
MIN_AXIS_PAINT = .24
CONFLICT_MARGIN = .04
MIN_LONG_LENS_FRAMES = 30
LONG_LENS_SLIP = 5.
MAX_LONG_LENS_SOLVES = 4
MAX_GROUND_REACH = 300  # frames a ground-followed path may run from a key, about 15 px of drift
MAX_TURF_SLIP = 1.  # pixels per frame a refined camera may move the lines over the turf
MAX_LOOSE_SHARE = .3  # frames that may slip before the whole fit is dropped
TURF_SLIP_SHARE = .03  # or this share of the turf's own motion in that frame
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


def _followed_from_key(k, key, followed):
    """The ground itself carried the camera from a solved key to this frame, step by step."""
    first, last = sorted((k, key))
    return followed is not None and last-first <= MAX_GROUND_REACH and bool(followed[first+1:last+1].all())


def _project_map(F, CUM, meta, segments, anchors, sky, mode, log, followed=None):
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
    carried = np.zeros(N, dtype=bool)
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
        # Little paint in view is no evidence against a camera the ground has carried from a key.
        if (score < minimum or axis < MIN_AXIS_PAINT) and k not in anchors:
            if not _followed_from_key(k, sources[0], followed):
                continue
            carried[k] = True
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
        # Two solved views of one camera ease into each other; elsewhere a gap means a broken path.
        if mode != 'camera' and max(pitch._gap(p, q, sky, grid), pitch._gap(q, p, sky, grid)) > 12:
            continue
        for k in range(a+1, b):
            t = (k-a)/(b-a)
            matrices[k] = pitch.blend(pitch.chain(CUM, a, k, H1), pitch.chain(CUM, b, k, H2), t)
            valid[k], quality[k] = True, min(s1, s2)
            carried[k] = carried[a] or carried[b]
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
            'matrix': matrices, 'paint_score': quality, 'score': float(np.nanmean(quality)),
            'carried': carried}


def _slips(matrices, valid, cumulative, k1=0., focal=1.):
    """Slip of pitch lines over the turf per frame, as a share of what is allowed. The ground does
    not move, so a line must go where the camera motion carries the turf. A fast pan may slip a
    little more, as the same small scale error moves more pixels, and the eye cannot follow it."""
    y, x = np.mgrid[640:1040:40, 60:1880:60]
    grid = np.c_[x.ravel(), y.ravel()].astype(float)
    straight = undistort(grid, k1, focal)
    found = {}
    for k in np.flatnonzero(valid[:-1] & valid[1:]):
        ground = warp(straight, np.linalg.inv(matrices[k]))
        lines = distort(warp(ground, matrices[k+1]), k1, focal)
        turf = warp(grid, cumulative[k+1] @ np.linalg.inv(cumulative[k]))
        slip = float(np.nanmedian(np.linalg.norm(lines-turf, axis=1)))
        moved = float(np.nanmedian(np.linalg.norm(turf-grid, axis=1)))
        found[int(k)] = slip/max(MAX_TURF_SLIP, TURF_SLIP_SHARE*moved)
    return found


def turf_slip(matrices, valid, cumulative, k1=0., focal=1.):
    """Worst slip of pitch lines over the turf, as a share of what is allowed."""
    return max(_slips(matrices, valid, cumulative, k1, focal).values(), default=0.)


def _with_one_camera(result, F, cumulative, meta, boxes, log):
    # Replace frame-by-frame matrices with one camera and lens. Keep the first answer if the fit
    # fails: it is coarser at the far side of the frame, but it is checked and usable.
    if result.get('centre_shifted'):
        # The camera is already one camera, solved with its optical centre off the frame centre.
        # The refit pins the centre, cannot hold it, and falls back to free matrices per frame
        # that slip over the turf (Alfheim: grass slip 1.6 -> 15 px, box line 5.5 -> 7.3 px).
        log('광학 중심이 프레임 중앙에서 벗어난 카메라라 정밀 보정을 건너뛰었어요.')
        return result
    carried = result.get('carried', np.zeros(len(F), dtype=bool))
    painted = result['valid'] & ~carried
    refine = hybrid_calib.refine_long_lens if result.get('long_lens') else hybrid_calib.refine_camera
    fitted = refine(pitch.Calib({**result, 'valid': painted}, cumulative), F, cumulative, meta, boxes)
    if fitted is None:
        log('카메라 정밀 보정을 건너뛰었어요.')
        return result
    # Past the first and last fitted key the fit only extrapolates, so those frames join the path.
    span = np.zeros(len(F), dtype=bool)
    span[int(np.min(fitted.keys)):int(np.max(fitted.keys))+1] = True
    painted &= span
    matrices = result['matrix'].copy()
    for k in np.flatnonzero(painted):
        H = fitted.img_from_pitch(int(k))
        matrices[k] = H/H[2, 2]
    focal = fitted.focal or 1.
    # Slip is only meaningful along one stretch, and a zoom fit eases between keys a little more.
    whole = painted & np.r_[result['segments'][:-1] == result['segments'][1:], True]
    limit = LONG_LENS_SLIP if result.get('long_lens') else 1
    slips = _slips(matrices, whole, cumulative, fitted.k1, focal)
    loose = [k for k, slip in slips.items() if slip > limit]
    if len(loose) > MAX_LOOSE_SHARE*max(whole.sum(), 1):
        log('카메라 정밀 보정이 선을 흔들어서 쓰지 않았어요.')
        return result
    # A few frames can slip where one fitted key is bad. They and the frames with too little paint
    # to fit stay on the ground path, from the nearest fitted frame.
    for k in loose:
        painted[[k, k+1]] = False
    straight = pinhole_motion(cumulative, fitted.k1, focal)
    source = np.flatnonzero(painted)
    if len(source):
        for k in np.flatnonzero(result['valid'] & ~painted):
            nearest = int(source[np.argmin(np.abs(source-k))])
            H = pitch.chain(straight, nearest, k, matrices[nearest])
            matrices[k] = H/H[2, 2]
    log(f'렌즈 왜곡 {fitted.k1:+.2f}, 초점거리 {focal:.0f} px {fitted.notes}')
    return {**result, 'matrix': matrices, 'k1': fitted.k1, 'focal': focal}


def _keys_on_paint(F, path, meta, boxes, anchors, segments, inside=False, step=15):
    """Add keys wherever paint can correct the path, working outward from the solved keys.

    The path carries a key well but not perfectly, and the frames before the first solved key can
    be far from it. Between solved keys this is optional, as it helps some clips name more paint
    and adds noise in others. Only rotation and zoom change; the camera stays where it was solved.
    """
    model = field_solver._model_tree(meta)
    model['meta'] = meta
    keys = dict(anchors)
    for segment in {segments[k] for k in anchors}:
        frames = np.flatnonzero(segments == segment)
        ordered = sorted(k for k in anchors if segments[k] == segment)
        for start, walk in ((ordered[0], frames[frames < ordered[0]][::-step]),
                            (ordered[0] if inside else ordered[-1],
                             frames[frames > (ordered[0] if inside else ordered[-1])][::step])):
            last = start
            for k in map(int, walk):
                if k in anchors:
                    last = k
                    continue
                H = pitch.chain(path, last, k, keys[last])
                points = broadcast_pitch.paint_points(field_solver._read_window(F, path, boxes, [k], k), widest=5.)
                if len(points) < 40:
                    continue
                settled = broadcast_pitch.settle(H, points, model)
                if settled is not None:
                    keys[k], last = settled, k
        _steady_keys(keys, path, segments, segment)
    return keys


def _steady_keys(keys, path, segments, segment, reach=3):
    """Remove key-to-key noise. On a perfect path every key gives the same pitch-to-first-frame
    map, so its median over neighbouring keys keeps real drift and drops what paint added."""
    frames = sorted(k for k in keys if segments[k] == segment)
    maps = np.array([np.linalg.inv(path[k]) @ keys[k] for k in frames])
    maps /= maps[:, 2:, 2:]
    for i, k in enumerate(frames):
        keys[k] = path[k] @ np.median(maps[max(0, i-reach):i+reach+1], axis=0)


def _paint_gap(result, F, path, boxes, meta, step=15):
    """Median pixels from the pitch lines in view to long thin paint. Lower is better. Only long
    paint counts, as turf sparkle lies close to any line and would flatter a wrong camera."""
    calib = pitch.Calib(result, path)
    features = pitch.model_lines(meta)
    gaps = []
    for k in np.flatnonzero(result['valid'])[::step]:
        _, lawn, loose = field_solver._strokes(np.asarray(F[k]), boxes[k], False)
        y, x = np.nonzero(field_solver.line_paint(loose, lawn, shortest=40.))
        if len(x) < 30:
            continue
        paint = cKDTree(np.c_[x, y]*2.)
        for line in features:
            image = calib.to_img(int(k), line)
            image = image[np.isfinite(image).all(1) & (image[:, 0] > 20) & (image[:, 0] < 1900)
                          & (image[:, 1] > 0) & (image[:, 1] < 1060)]
            if len(image) >= 15:
                near = paint.query(image)[0]
                near = near[near < 40]
                if len(near) >= 10:
                    gaps.append(float(np.median(near)))
    return float(np.median(gaps)) if gaps else np.inf


def _best_on_paint(F, path, meta, boxes, solved, segments, sky, followed, log):
    """Try each way of keying the clip and keep what lies closest to the paint. Which is best
    differs by clip, as it depends on how much paint each part of the pan shows."""
    outer = _keys_on_paint(F, path, meta, boxes, solved, segments)
    every = _keys_on_paint(F, path, meta, boxes, solved, segments, inside=True)
    best = None
    for keys in (solved, outer, every):
        result = _project_map(F, path, meta, segments, keys, sky, 'camera', log, followed)
        if not result:
            continue
        result = _with_one_camera(result, F, path, meta, boxes, log)
        gap = _paint_gap(result, F, path, boxes, meta)
        log(f'선 일치 {gap:.1f} px, 렌즈 {result.get("k1", 0.):+.3f}')
        if best is None or gap < best[0]:
            best = gap, result
    return best[1] if best else None


def _from_long_lens(F, cumulative, meta, boxes, inliers, log):
    """Keys for high tele footage, on the ground path, as its feature path is rarely usable.

    A solve holds only as far as the path does, so what it leaves is solved on its own, as a
    separate stretch that shares no keys with the last. The last value says whether every stretch
    was solved with the optical centre off the frame centre.
    """
    path, followed = turf_motion(F, boxes, cumulative, smooth_grass=True)
    segments = np.zeros(len(F), dtype=int)
    anchors, shifted = {}, []
    waiting = np.split(np.arange(len(F)), shot_cuts(F))
    for number, shot in enumerate(waiting):
        segments[shot] = number
    for _ in range(MAX_LONG_LENS_SOLVES+len(waiting)):
        if not waiting:
            break
        frames = waiting.pop(0)
        if len(frames) < MIN_LONG_LENS_FRAMES:
            continue
        try:
            keys = broadcast_pitch.solve(F, path, meta, boxes, frames, log=log) or {}
        except (ValueError, np.linalg.LinAlgError) as error:
            log(f'중계 카메라를 풀지 못했어요: {error}')
            keys = {}
        if not keys:
            continue
        first, last = min(keys), max(keys)
        anchors.update(keys)
        shifted.append(bool(getattr(keys, 'centre_shifted', False)))
        segments[frames[frames >= first]] = segments.max()+1
        for rest in (frames[frames < first], frames[frames > last]):
            if len(rest):
                waiting.append(rest)
                segments[rest] = segments.max()+1
    return anchors, path, segments, followed, bool(shifted) and all(shifted)


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
    followed, long_lens, solved, centre_shifted = None, False, None, False
    if not anchors and solver_enabled():
        for segment in np.unique(segments):
            frames = np.flatnonzero(segments == segment)
            if len(frames) >= 2:
                try:
                    anchors.update(field_solver.solve(F, cumulative, meta, boxes, frames, log=log) or {})
                except (ValueError, np.linalg.LinAlgError) as error:
                    # Search on a view it cannot model (a ground plane seen edge-on) is a refusal.
                    log(f'카메라를 풀지 못했어요: {error}')
        if not anchors and boxes is not None:
            anchors, cumulative, segments, followed, centre_shifted = _from_long_lens(F, cumulative, meta, boxes,
                                                                                      inliers, log)
            long_lens = bool(anchors)
        mode = 'camera' if anchors else mode
        if anchors and boxes is not None and followed is None:
            # The solver ties its keys by pure rotation, which a ground-plane path is not, so only
            # the frames between and beyond the keys follow the ground.
            cumulative, followed = turf_motion(F, boxes, cumulative)
            solved = anchors
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
    if solved is not None:
        return _best_on_paint(F, cumulative, meta, boxes, solved, segments, sky, followed, log)
    result = _project_map(F, cumulative, meta, segments, anchors, sky, mode, log, followed)
    if result:
        result['long_lens'] = long_lens
        result['centre_shifted'] = centre_shifted
    return _with_one_camera(result, F, cumulative, meta, boxes, log) if mode == 'camera' and result else result
