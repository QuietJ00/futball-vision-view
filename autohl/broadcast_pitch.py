"""Find a long-lens camera from paint, for footage from high in a stand or a broadcast gantry.

The stand solver needs four lines that form a pitch rectangle, and a tele view rarely shows one.
Here two lines are enough: with the camera position and focal length assumed, two named lines fix
the rotation exactly, and the angle between them must match, which rejects most assignments. The
survivors are ranked on all the paint in view. Nothing depends on the venue except its dimensions.
"""
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import camfit
from . import field_solver as fs
from . import pitch
from .camera import warp

MIN_FOCAL, MAX_FOCAL = 1200., 14000.
MIN_CAMERA_M, MAX_CAMERA_M = 6., 45.
SETBACKS_M = (8., 14., 20., 28., 38., 52., 70.)   # behind the touchline
HEIGHTS_M = (7., 10., 13., 17., 22., 29., 38.)
ALONG_M = (-30., -18., -9., -3., 3., 9., 18., 30.)   # from the halfway line
_RATIO = (8000. / 1500.) ** (1. / 13.)
FOCALS = np.geomspace(1500., 8000., 14)
ANGLE_TOLERANCE = .1                     # share of the angle between the two planes
MAX_ROLL = np.radians(4.)
IMAGE_LINES = 12
PRESELECT = 250
PAINT_REACH_PX = 8.
KEYS_PER_SHOT = 8
VIEWS = 5
SAMPLES = 14
MAX_PEOPLE_ERROR = .25    # median |log(predicted height / box height)|; tele boxes are looser
MIN_FEET_ON_PITCH = .6    # ball boys, stewards and the bench stand off it
MIN_PAINT_SHARE = .40
TELE_LINE_WIDEST = 14.     # half-size pixels: a long lens shows paint as a wide band
MAX_SETTLE_PX = 45.        # further than this, paint has pulled the camera onto the wrong line
MIN_PAINTED = .4
MIN_PEOPLE = 12          # whole people across the sampled frames of a shot


def _positions(meta):
    return np.array([(meta['length']/2 + a, meta['width']/2 + s, -h) for a, s, h in product(ALONG_M, SETBACKS_M, HEIGHTS_M)])


def _model(meta):
    features = pitch.model_segments(meta)
    use = [i for i, f in enumerate(features) if f[0] == 'line']
    a = np.array([np.r_[features[i][1], 0.] for i in use])
    b = np.array([np.r_[features[i][2], 0.] for i in use])
    return use, a, b


def _image_planes(lines, focal):
    """Plane normals through the camera for image lines, in camera axes, one row per focal."""
    a = np.c_[lines[:, :2]-fs.CENTRE, np.ones(len(lines))]
    b = np.c_[lines[:, 2:]-fs.CENTRE, np.ones(len(lines))]
    n = []
    for f in focal:
        s = np.array([1/f, 1/f, 1.])
        c = np.cross(a*s, b*s)
        n.append(c/np.linalg.norm(c, axis=1, keepdims=True))
    return np.array(n)


def _triad(u, v):
    x = u/np.linalg.norm(u, axis=-1, keepdims=True)
    z = np.cross(u, v)
    z = z/np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    return np.stack((x, np.cross(z, x), z), -1)


def candidates(lines, meta):
    """Homographies and focal lengths for every pair of image lines named as a pair of model lines."""
    centres = _positions(meta)
    _, a, b = _model(meta)
    world = np.cross(a[None]-centres[:, None], b[None]-centres[:, None])
    world /= np.linalg.norm(world, axis=2, keepdims=True)
    planes = _image_planes(lines, FOCALS)
    i, j = np.triu_indices(len(lines), 1)
    image = planes[:, i], planes[:, j]
    m, n = np.nonzero(~np.eye(len(a), dtype=bool))
    found_H, found_f, found_c = [], [], []
    seen_cos = np.sum(image[0]*image[1], 2)
    for c, centre in enumerate(centres):
        model_cos = np.sum(world[c, m]*world[c, n], 1)
        angle = np.arccos(np.clip(np.abs(model_cos), 0, 1))
        limit = np.maximum(ANGLE_TOLERANCE*angle*np.sin(angle), 1e-5)
        for s1, s2 in ((1, 1), (1, -1)):
            gap = np.abs(s1*s2*seen_cos[:, :, None]-model_cos[None, None])
            fi, pi, qi = np.nonzero(gap < limit[None, None])
            if not len(fi):
                continue
            for flip in (1., -1.):
                u_c = flip*s1*image[0][fi, pi]
                v_c = flip*s2*image[1][fi, pi]
                R = _triad(u_c, v_c) @ _triad(world[c, m[qi]], world[c, n[qi]]).transpose(0, 2, 1)
                # Level camera, looking down at the ground, the pitch ahead of it.
                ok = (np.abs(np.arcsin(np.clip(R[:, 0, 2], -1, 1))) < MAX_ROLL) & (R[:, 1, 2] > .9) & (R[:, 2, 2] > .02)
                if ok.any():
                    R = R[ok]
                    f = FOCALS[fi[ok]]
                    t = -R @ centre
                    K = np.zeros((len(f), 3, 3))
                    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2] = f, f, 1.
                    K[:, :2, 2] = fs.CENTRE
                    H = K @ np.stack((R[:, :, 0], R[:, :, 1], t), 2)
                    found_H.append(H/H[:, 2:, 2:])
                    found_f.append(f)
    if not found_H:
        return np.empty((0, 3, 3)), np.empty(0)
    H, focal = np.concatenate(found_H), np.concatenate(found_f)
    good = np.isfinite(H).all((1, 2)) & (np.abs(np.linalg.det(H)) > 1e-12)
    return H[good], focal[good]


def long_lines(strokes, count=IMAGE_LINES, tolerance=3.):
    """Join stroke pieces into whole image lines, longest evidence first."""
    left = np.array(strokes, float)
    lines, weights = [], []
    while len(left) and len(lines) < count:
        length = np.linalg.norm(left[:, 2:]-left[:, :2], axis=1)
        seed = left[np.argmax(length)]
        for _ in range(3):
            direction = (seed[2:]-seed[:2])/max(np.linalg.norm(seed[2:]-seed[:2]), 1e-9)
            normal = np.array([-direction[1], direction[0]])
            ends = np.abs((left.reshape(-1, 2)-seed[:2]) @ normal).reshape(-1, 2).max(1)
            member = ends < tolerance
            if not member.any():
                break
            points = left[member].reshape(-1, 2)
            mean = points.mean(0)
            axis = np.linalg.svd(points-mean)[2][0]
            along = (points-mean) @ axis
            seed = np.r_[mean+axis*along.min(), mean+axis*along.max()]
        if not member.any():
            left = left[1:]
            continue
        lines.append(seed)
        weights.append(float(length[member].sum()))
        left = left[~member]
    order = np.argsort(-np.array(weights))
    return np.array(lines).reshape(-1, 4)[order], np.array(weights)[order]


def paint_points(window, limit=3000, widest=TELE_LINE_WIDEST):
    """Paint pixels of the whole window in its base view, including curves that make no stroke."""
    found = []
    for to_base, lawn, loose in window['frames'].values():
        y, x = np.nonzero(fs.line_paint(loose, lawn, shortest=16., widest=widest))
        if len(x):
            found.append(warp(np.c_[x, y]*2., to_base))
    if not found:
        return np.empty((0, 2))
    points = np.vstack(found)
    return points[::max(1, len(points)//limit)]


def _fit(p, points, model, free=True):
    """Tie paint to its nearest model line, tightening as the camera settles."""
    meta = model['meta']
    low = np.array([-10., -1.5, -.2, np.log(MIN_FOCAL), -80., meta['width']/2+3., MIN_CAMERA_M])
    high = np.array([10., 1.5, .2, np.log(MAX_FOCAL), meta['length']+80., meta['width']/2+120., MAX_CAMERA_M])
    count = 7 if free else 4
    p = np.clip(p, low+1e-6, high-1e-6)
    for limit in (4., 2.5, 1.5, 1.):
        near, lines = fs._associate(p, points, model, limit)
        if len(near) < 12:
            break
        fitted = least_squares(lambda v: fs._line_error(np.r_[v, p[count:]], near, lines), p[:count], loss='soft_l1',
                               f_scale=3., max_nfev=20, x_scale=[.02, .02, .02, .05, .5, .5, .2][:count],
                               bounds=(low[:count], high[:count]))
        p = np.r_[fitted.x, p[count:]]
    return p


def _explained(p, points, model):
    near, _ = fs._associate(p, points, model, 1.)
    return len(near)/max(len(points), 1)


def _painted(p, points, model):
    """Share of the model in view that has paint on it. A camera at the wrong scale can lay most
    paint on some line, yet its circle and box then fall where the turf is bare."""
    H = fs._homography(p)
    image = warp(model['points'], H)
    inside = (image[:, 0] > 0) & (image[:, 1] > 0) & (image[:, 0] < points[:, 0].max()) & (image[:, 1] < 1080)
    if inside.sum() < 40:
        return 0.
    return float(np.mean(cKDTree(points).query(image[inside])[0] < PAINT_REACH_PX))


def search(window, model, meta, keep=8):
    """Refined cameras for one window, best first, from its paint and people."""
    strokes, feet, seen = window['strokes'], window['feet'], window['seen']
    if len(strokes) < 2:
        return []
    lines, _ = long_lines(strokes)
    if len(lines) < 2:
        return []
    H, focal = candidates(lines, meta)
    if not len(H):
        return []
    points = paint_points(window)
    if len(points) < 40:
        return []
    template = fs._soft_template(meta, sigma=2.)
    rank = fs._paint_rank(H, points[::4], template)
    best = np.argsort(-rank)[:PRESELECT]
    H, focal = H[best], focal[best]
    R, centre = fs._poses(H, focal)
    settled = []
    for i in range(len(H)):
        p = _fit(fs._parameters(R[i], centre[i], focal[i]), points, model, free=False)
        settled.append((_explained(p, points, model), i, p))
    settled.sort(key=lambda s: -s[0])
    found, used = [], []
    for share, _, p in settled[:40]:
        if any(np.abs(p[:4]-u[:4]).max() < .005 for u in used):
            continue
        used.append(p)
        p = _fit(p, points, model)
        r = fs._rate(p, window, model, meta)
        r['share'] = _explained(p, points, model)
        found.append(r)
        if len(found) >= keep:
            break
    return sorted(found, key=lambda r: -r['share']*float(fs._people_weight(r['people'], r['on_pitch'])))


def settle(H, points, model):
    """H with its rotation and zoom corrected onto paint, or None if that explains no more of it."""
    focal, rotation, centre = camfit.decompose(H)
    if not np.isfinite([focal, *centre]).all():
        return None
    before = fs._parameters(camfit.rodrigues(rotation), centre, focal)
    after = fs._refine_angles(before, points, model)
    if _explained(after, points, model) < _explained(before, points, model):
        return None
    settled = fs._homography(after)
    grid = np.stack(np.meshgrid(np.arange(200., 1800., 200.), np.arange(500., 1050., 100.)), -1).reshape(-1, 2)
    moved = np.linalg.norm(warp(warp(grid, np.linalg.inv(H)), settled)-grid, axis=1)
    return settled if np.median(moved) <= MAX_SETTLE_PX else None


def _enough_people(feet):
    """The people check says nothing without people, and a close-up has one or two."""
    return len(feet) >= MIN_PEOPLE


def _carried(p, path, frame, base):
    """A camera solved at one frame, restated at the base frame along the camera path."""
    focal, rotation, centre = camfit.decompose(pitch.chain(path, frame, base, fs._homography(p)))
    if not np.isfinite([focal, *centre]).all() or not MIN_FOCAL/2 < focal < MAX_FOCAL*2:
        return None
    return fs._parameters(camfit.rodrigues(rotation), centre, focal)


def solve(F, path, meta, boxes, frames, log=print):
    """Pitch-to-image homographies at key frames of one shot, or None if none is credible."""
    frames = np.asarray(frames)
    base = int(frames[len(frames)//2])
    sample = frames[np.linspace(0, len(frames)-1, min(len(frames), SAMPLES)).astype(int)]
    window = fs._read_window(F, path, boxes, sample, base)
    model = fs._model_tree(meta)
    model['meta'] = meta
    points = paint_points(window)
    if len(points) < 40 or not _enough_people(window['feet']):
        return None
    found = search(window, model, meta)
    # A view with few lines can still solve alone when the joined one does not, so each solved
    # view is carried to the base and judged on all the paint.
    reach = min(8, (len(frames)-1)//2)
    for k in np.linspace(frames[reach], frames[-1-reach], VIEWS).astype(int):
        view = fs._read_window(F, path, boxes, np.unique([k-reach, k, k+reach]), int(k))
        for r in search(view, model, meta, keep=4):
            if abs(r['people']) > MAX_PEOPLE_ERROR:
                continue
            p = _carried(r['p'], path, int(k), base)
            if p is not None:
                p = _fit(p, points, model)
                r = fs._rate(p, window, model, meta)
                r['share'] = _explained(p, points, model)
                found.append(r)
    for r in found:
        r['painted'] = _painted(r['p'], points, model)
    found = sorted((r for r in found if abs(r['people']) <= MAX_PEOPLE_ERROR and r['on_pitch'] >= MIN_FEET_ON_PITCH),
                   key=lambda r: -r['share']*r['painted'])
    for r in found[:5]:
        log(f'후보 일치 {r["share"]:.0%} 선 {r["painted"]:.0%} 키 {r["people"]:+.2f} f {np.exp(r["p"][3]):.0f} 위치 {np.round(r["p"][4:7], 1).tolist()}')
    if not found:
        log('중계 카메라 후보를 믿을 수 없어요: 사람 키가 맞는 후보가 없어요')
        return None
    best = found[0]
    p = best['p']
    lowest = fs.minimum_height(p[4:7], np.exp(p[3]), meta)
    if best['share'] < MIN_PAINT_SHARE or best['painted'] < MIN_PAINTED or p[6] < lowest:
        log(f'중계 카메라 후보를 믿을 수 없어요: 페인트 일치 {best["share"]:.0%}, 선 {best["painted"]:.0%}, 높이 {p[6]:.1f} m, 최소 {lowest:.1f} m')
        return None
    log(f'중계 카메라 ({p[4]:.1f}, {p[5]:.1f}) 높이 {p[6]:.1f} m, 초점거리 {np.exp(p[3]):.0f} px, 페인트 일치 {best["share"]:.0%}')
    H = fs._homography(p)
    return _keys_on_paint(F, path, boxes, frames, base, H, model)


def _keys_on_paint(F, path, boxes, frames, base, H, model):
    """Keys around the base that still sit on paint. The path can break in a whip pan, and what
    lies past a break belongs to another solve."""
    keys = {}
    step = max(1, len(frames)//KEYS_PER_SHOT)
    middle = int(np.searchsorted(frames, base))
    for side in (frames[middle::step], frames[middle::-step]):
        for k in map(int, side):
            view = fs._read_window(F, path, boxes, [k], k)
            points = paint_points(view)
            H_k = pitch.chain(path, base, k, H)
            if len(points) >= 40 and _share_at(H_k, points, model) < MIN_PAINT_SHARE/2:
                break
            keys[k] = H_k
    return keys


def _share_at(H, points, model):
    """Share of paint that lies on the pitch model under camera H."""
    world = warp(points, np.linalg.inv(H))
    return float(np.mean(model['tree'].query(np.nan_to_num(world, nan=1e6))[0] < 1.))
