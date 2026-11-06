"""Experimental calibration from named pitch lines. Not used by the app unless called directly.

Input is, per key frame, a list of {'segment': index into pitch.model_segments, 'points': Nx2,
'confidence': c}. Three fits share the same line residual so that a comparison between them
measures the camera model alone:

  fit_homographies  a free 8-parameter homography per key frame
  fit_shared        one position and lens for the clip, a rotation per key, optional chain ties
  fit_tripod        tilt and roll also shared; only yaw changes, about a vertical axis

Coordinates follow field_solver: X along the near touchline, Y toward the camera, Z down.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import camfit, field_solver as fs, pitch
from .camera import distort, pinhole_motion, undistort, warp

MAX_RMS_PX = 15.         # a fit worse than this has failed
BAD_LINE_PX = 12.        # a named line further off than this is treated as misnamed
CHAIN_WEIGHT = .3        # px of residual per milliradian of disagreement with the camera chain
MIN_LINES = 4
MAX_DROPPED = .3         # share of named lines a fit may discard before it counts as failed
MAX_K1 = .5              # a stronger lens term means the fit is using it to bend something else
LONG_LENS = 3000.        # px; past this a lens term cannot be told from the camera's own pose
ZOOM_DRIFT = .02         # share of focal length a clip must change by to count as zooming
FAR_TOUCHLINE = 2        # index in pitch.model_segments
CURVES = (17, 18, 19)    # centre circle and penalty arcs


class Calibration:
    """Same conversion interface as pitch.Calib, with optional radial distortion."""

    def __init__(self, keys, matrices, cumulative, meta, k1=0., cameras=None, rms=None, notes='', focal=None):
        self.keys, self.H = np.asarray(keys), matrices
        self.CUM, self.meta, self.k1, self.focal = cumulative, meta, k1, focal
        self.cameras, self.rms, self.notes = cameras, rms, notes

    def supports(self, start, end=None):
        return True

    def img_from_pitch(self, frame):
        i = np.searchsorted(self.keys, frame)
        if i < len(self.keys) and self.keys[i] == frame:
            return self.H[int(frame)]
        if i == 0 or i == len(self.keys):
            near = int(self.keys[min(i, len(self.keys)-1)])
            return pitch.chain(self.CUM, near, frame, self.H[near])
        a, b = int(self.keys[i-1]), int(self.keys[i])
        return pitch.blend(pitch.chain(self.CUM, a, frame, self.H[a]),
                           pitch.chain(self.CUM, b, frame, self.H[b]), (frame-a)/(b-a))

    def _focal(self, frame):
        if not self.k1:
            return 1.
        return self.focal or float(camfit.decompose(self.img_from_pitch(frame))[0])

    def to_img(self, frame, points):
        return distort(warp(points, self.img_from_pitch(frame)), self.k1, self._focal(frame))

    def to_pitch(self, frame, points):
        return warp(undistort(np.asarray(points, float), self.k1, self._focal(frame)),
                    np.linalg.inv(self.img_from_pitch(frame)))


def _features(meta):
    out = []
    for feature in pitch.model_segments(meta):
        if feature[0] == 'line':
            n = max(2, int(np.linalg.norm(feature[2]-feature[1])/.25))
            points, tangent = np.linspace(feature[1], feature[2], n), np.tile(feature[2]-feature[1], (n, 1))
        else:
            points, tangent = feature[1], np.gradient(feature[1], axis=0)
        tangent = tangent/np.linalg.norm(tangent, axis=1, keepdims=True)
        out.append((cKDTree(points), points, np.c_[-tangent[:, 1], tangent[:, 0]]))
    return out


def _tie(H, lines, features, k1=0., focal=1.):
    """For each observed point, the world tangent line of the model line it was named as."""
    Hi = np.linalg.inv(H)
    points, world, weight, owner = [], [], [], []
    for n, line in enumerate(lines):
        tree, model, normal = features[line['segment']]
        raw = np.asarray(line['points'], float)
        ground = warp(undistort(raw, k1, focal), Hi)
        ground = np.nan_to_num(ground, nan=1e6, posinf=1e6, neginf=-1e6)
        _, nearest = tree.query(ground)
        points.append(raw)
        world.append(np.c_[normal[nearest], -(normal[nearest]*model[nearest]).sum(1)])
        weight.append(np.full(len(raw), np.sqrt(line.get('confidence', 1.))))
        owner.append(np.full(len(raw), n))
    if not points:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty(0), np.empty(0, int)
    return np.vstack(points), np.vstack(world), np.concatenate(weight), np.concatenate(owner)


def _distance(H, points, world):
    seen = world @ np.linalg.inv(H)
    return ((seen[:, 0]*points[:, 0]+seen[:, 1]*points[:, 1]+seen[:, 2])
            / np.maximum(np.hypot(seen[:, 0], seen[:, 1]), 1e-12))


def _key_error(H, lines, features, k1=0., focal=1.):
    points, world, _, _ = _tie(H, lines, features, k1, focal)
    if not len(points):
        return None
    return float(np.sqrt(np.mean(_distance(H, undistort(points, k1, focal), world)**2)))


def _lines_only(lines, meta):
    # Least-squares line per observed straight class, paired with two world points of that line.
    segments = pitch.model_segments(meta)
    pairs = []
    for line in lines:
        feature = segments[line['segment']]
        points = np.asarray(line['points'], float)
        if feature[0] != 'line' or len(points) < 3 or np.ptp(points, axis=0).max() < 25:
            continue
        centre = points.mean(0)
        direction = np.linalg.svd(points-centre)[2][0]
        normal = np.array([-direction[1], direction[0]])
        pairs.append((np.r_[normal, -normal @ centre], feature[1], feature[2]))
    return pairs


def _dlt(pairs):
    scale = np.diag([1/1000, 1/1000, 1.])
    rows = []
    for image, a, b in pairs:
        l = np.linalg.inv(scale).T @ image
        l = l/np.linalg.norm(l)
        for w in (a, b):
            rows.append(np.kron(l, [w[0]/50, w[1]/50, 1.]))
    _, s, vt = np.linalg.svd(np.array(rows))
    if s[-2] < 1e-6*s[0]:
        return None
    H = np.linalg.inv(scale) @ vt[-1].reshape(3, 3) @ np.diag([1/50, 1/50, 1.])
    return H/H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def homography_from_lines(lines, meta, features):
    """A free homography from named lines, or None when they cannot fix one."""
    pairs = _lines_only(lines, meta)
    if len(pairs) < MIN_LINES:
        return None
    H = _dlt(pairs)
    if H is None or not np.isfinite(H).all():
        return None
    for _ in range(3):
        points, world, weight, _ = _tie(H, lines, features)
        if len(points) < 8:
            return None
        fit = least_squares(lambda v: weight*_distance(np.r_[v, 1.].reshape(3, 3), points, world),
                            (H/H[2, 2]).ravel()[:8], loss='soft_l1', f_scale=3., x_scale='jac', max_nfev=40)
        H = np.r_[fit.x, 1.].reshape(3, 3)
    return H


def fit_homographies(observations, cumulative, meta):
    features = _features(meta)
    solved = {}
    for k, lines in observations.items():
        H = homography_from_lines(lines, meta, features)
        if H is None or (_key_error(H, lines, features) or 1e9) >= MAX_RMS_PX:
            continue
        # A free fit to a few lines can be a matrix no camera could produce.
        if .5 < _pose_of(H)[6] < 80 and np.linalg.cond(H/np.linalg.norm(H)) < 1e9:
            solved[k] = H
    if not solved:
        return None
    return Calibration(sorted(solved), solved, cumulative, meta,
                       rms=float(np.median([_key_error(solved[k], observations[k], features) for k in solved])))


def _pose_of(H):
    focal = float(camfit.decompose(H)[0])
    R, centre = fs._poses((H/H[2, 2])[None], np.array([focal]))
    return fs._parameters(R[0], centre[0], focal)


def _chain_rotation(cumulative, a, b, focal):
    K = np.array([[focal, 0, fs.CENTRE[0]], [0, focal, fs.CENTRE[1]], [0, 0, 1.]])
    M = np.linalg.inv(K) @ cumulative[b] @ np.linalg.inv(cumulative[a]) @ K
    u, _, vt = np.linalg.svd(M)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def _rotation_vector(R):
    angle = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    axis = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    return axis*(.5 if angle < 1e-6 else angle/(2*np.sin(angle)))


def _aimed(position, focal, lines, meta, features):
    # Best rotation for a guessed position: aim at the named lines, then fit the three angles.
    ground = np.vstack([np.mean(features[l['segment']][1], axis=0)[None] for l in lines])
    target = ground.mean(0)
    to = target-position[:2]
    yaw = np.arctan2(to[0], -to[1])
    tilt = np.arctan2(position[2], np.linalg.norm(to))
    p = np.array([yaw, tilt, 0., np.log(focal), *position])
    for _ in range(3):
        H = fs._homography(p)
        points, world, weight, _ = _tie(H, lines, features)
        if len(points) < 6:
            return None, 1e9
        fit = least_squares(lambda v: weight*_distance(fs._homography(np.r_[v, p[3:]]), points, world),
                            p[:3], loss='soft_l1', f_scale=5., x_scale=.02, max_nfev=30)
        p = np.r_[fit.x, p[3:]]
    return p, _key_error(fs._homography(p), lines, features)


def _start(observations, cumulative, meta, features):
    """A first camera per key: from free homographies, else a coarse search over positions."""
    solved, loose = {}, {}
    for k, lines in observations.items():
        H = homography_from_lines(lines, meta, features)
        if H is None:
            continue
        p = _pose_of(H)
        error = _key_error(H, lines, features) or 1e9
        if p[6] > 0 and error < MAX_RMS_PX:
            solved[k] = p
        elif p[6] > 0 and error < 3*MAX_RMS_PX:
            loose[k] = p  # bent lines from lens distortion can still give a usable first guess
    solved = solved or loose
    if solved:
        cameras = np.array(list(solved.values()))
        return np.median(cameras[:, 4:7], axis=0), float(np.exp(np.median(cameras[:, 3]))), solved
    busiest = max(observations, key=lambda k: sum(len(l['points']) for l in observations[k]), default=None)
    if busiest is None or not observations[busiest]:
        return None
    best = (1e9, None)
    half = meta['width']/2
    for x in np.linspace(0, meta['length'], 6):
        for y in (half+4, half+12, half+25):
            for h in (2.5, 5., 10.):
                for focal in (1300., 1900.):
                    p, error = _aimed(np.array([x, y, h]), focal, observations[busiest], meta, features)
                    if error < best[0]:
                        best = error, p
    if best[1] is not None and best[0] < BAD_LINE_PX:
        return best[1][4:7], float(np.exp(best[1][3])), {busiest: best[1]}
    return _long_lens_start(observations, cumulative, meta, features, busiest)


def _settled(start, keys, busiest, observations, cumulative, features):
    """Every key aimed from one camera position and focal length, outward from the busiest key."""
    focal = float(np.exp(start[3]))
    cameras = {busiest: start}
    for k in sorted(keys, key=lambda j: abs(j-busiest)):
        if k in cameras:
            continue
        near = min(cameras, key=lambda j: abs(j-k))
        R = _chain_rotation(cumulative, near, k, focal) @ fs._rotation(*cameras[near][:3])
        p = np.r_[fs._parameters(R, np.zeros(3), focal)[:3], start[3:]]
        for _ in range(2):
            points, world, weight, _ = _tie(fs._homography(p), observations[k], features)
            if len(points) >= 6:
                p[:3] = least_squares(lambda v: weight*_distance(fs._homography(np.r_[v, p[3:]]), points, world),
                                      p[:3], loss='soft_l1', f_scale=15., x_scale=.01, max_nfev=30).x
        cameras[k] = p
    return cameras


def _long_lens_start(observations, cumulative, meta, features, busiest):
    # A broadcast camera sits far back and high with a long lens. One frame cannot tell its
    # distance from its focal length, so candidates are judged on every key.
    half = meta['width']/2
    tried = []
    for x in np.linspace(0, meta['length'], 6):
        for y in (half+10, half+25, half+40):
            for h in (10., 15., 25.):
                for focal in (3000., 4500., 6500.):
                    p, error = _aimed(np.array([x, y, h]), focal, observations[busiest], meta, features)
                    if p is not None:
                        tried.append((error, p))
    if not tried:
        return None
    best = None
    for _, p in sorted(tried, key=lambda t: t[0])[:4]:
        cameras = _settled(p, sorted(observations), busiest, observations, cumulative, features)
        error = np.median([_key_error(fs._homography(cameras[k]), observations[k], features) or 1e9 for k in cameras])
        if best is None or error < best[0]:
            best = error, cameras, p
    return best[2][4:7], float(np.exp(best[2][3])), best[1]


def _spread(solved, keys, cumulative, focal):
    # Angles for keys without a camera of their own, through the chain from the nearest that has.
    angles = {}
    for k in keys:
        if k in solved:
            angles[k] = solved[k][:3]
            continue
        source = min(solved, key=lambda s: abs(s-k))
        R = _chain_rotation(cumulative, source, k, focal) @ fs._rotation(*solved[source][:3])
        p = fs._parameters(R, np.zeros(3), focal)
        angles[k] = p[:3]
    return angles


def _clean(observations, cameras, features, k1=0., focal=1.):
    # Drop lines that sit far from the model line they were named as: most likely misnamed.
    kept, dropped = {}, 0
    for k, lines in observations.items():
        H = fs._homography(cameras[k]) if k in cameras else None
        keep = []
        for line in lines:
            if H is None:
                keep.append(line)
                continue
            points, world, _, _ = _tie(H, [line], features, k1, focal)
            if np.median(np.abs(_distance(H, undistort(points, k1, focal), world))) < BAD_LINE_PX:
                keep.append(line)
            else:
                dropped += 1
        kept[k] = keep
    return kept, dropped


def _result(keys, cameras, cumulative, meta, k1, rms, notes):
    focal = float(np.exp(next(iter(cameras.values()))[3]))
    return Calibration(keys, {k: fs._homography(cameras[k]) for k in keys}, cumulative, meta, k1=k1,
                       cameras=cameras, rms=rms, notes=notes, focal=focal)


def fit_shared(observations, cumulative, meta, temporal=True, distortion=True, free_position=False, zoom=False):
    """One camera position and lens per clip, a rotation per key frame.

    With zoom, the focal length may drift from key to key. A broadcast camera zooms while it pans,
    and a single focal length then turns that zoom into false ground distance.
    """
    features = _features(meta)
    start = _start(observations, cumulative, meta, features)
    if start is None:
        return None
    position, focal, solved = start
    keys = sorted(observations)
    angles = _spread(solved, keys, cumulative, focal)
    first = {k: np.r_[angles[k], np.log(focal), position] for k in keys}
    start_error = np.median([_key_error(fs._homography(first[k]), observations[k], features) or 1e9 for k in keys])
    rough = focal >= LONG_LENS and start_error >= BAD_LINE_PX
    if rough:
        # On a long lens a small change of position moves every line far, so angles solved for
        # each key's own position do not suit the shared one. Start them from the shared camera,
        # or the robust fit reads every line as an outlier.
        busiest = max(keys, key=lambda k: sum(len(l['points']) for l in observations[k]))
        settled = _settled(first[busiest], keys, busiest, observations, cumulative, features)
        angles = {k: settled[k][:3] for k in keys}
    distortion = distortion and focal < LONG_LENS
    k1, dropped = 0., 0
    n = len(keys)
    x = np.r_[position, np.log(focal), k1, np.concatenate([angles[k] for k in keys])]
    if free_position:
        x = np.r_[x, np.tile(position, n)]
    drift_at = len(x)
    if zoom:
        distortion = False
        x = np.r_[x, np.zeros(n)]
    lower, upper = np.full(len(x), -np.inf), np.full(len(x), np.inf)
    lower[3], upper[3] = x[3]-.7, x[3]+.7
    lower[4], upper[4] = -MAX_K1, MAX_K1
    if zoom:
        lower[drift_at:], upper[drift_at:] = -.25, .25
    links = [(a, b) for a, b in zip(keys, keys[1:])]

    def unpack(v):
        focal_, shift = v[3], v[4]
        cameras = {}
        for i, k in enumerate(keys):
            where = v[5+3*n+3*i:5+3*n+3*i+3] if free_position else v[:3]
            drift = v[drift_at+i] if zoom else 0.
            cameras[k] = np.r_[v[5+3*i:8+3*i], focal_+drift, where]
        return cameras, shift

    active = observations
    for limit in range(4):
        cameras, k1 = unpack(x)
        focal = float(np.exp(x[3]))
        tied = {k: _tie(fs._homography(cameras[k]), active[k], features, k1, focal) for k in keys}
        moves = {link: _chain_rotation(cumulative, *link, focal) for link in links} if temporal else {}

        def residual(v):
            cams, shift = unpack(v)
            f = float(np.exp(v[3]))
            out = []
            for k in keys:
                points, world, weight, _ = tied[k]
                if len(points):
                    out.append(weight*_distance(fs._homography(cams[k]), undistort(points, shift, f), world))
            for a, b in moves:
                # rotations are world-to-camera, so the chain turns camera a into camera b
                R = fs._rotation(*cams[b][:3]) @ fs._rotation(*cams[a][:3]).T @ moves[a, b].T
                out.append(1000*CHAIN_WEIGHT*_rotation_vector(R))
            if free_position:
                # A moving camera still moves smoothly, and rarely more than a few metres.
                out.append(.2*(v[5+3*n+3:]-v[5+3*n:-3]))
            if distortion:
                out.append(np.array([3*v[4]]))
            if zoom:
                out.append(40*v[drift_at:])
                out.append(300*np.diff(v[drift_at:]))
            return np.concatenate(out) if out else np.zeros(1)

        free = np.ones(len(x), bool)
        free[4] = distortion
        def only(u):
            v = x.copy()
            v[free] = u
            return residual(v)
        scale = np.r_[[.5, .5, .2, .03, .05], np.full(3*n, .01), np.full(3*n if free_position else 0, .5),
                      np.full(n if zoom else 0, .01)]
        x[free] = least_squares(only, x[free], bounds=(lower[free], upper[free]), loss='soft_l1',
                                f_scale=(20., 8., 3., 3.)[limit] if rough else 3., x_scale=scale[free], max_nfev=30).x
        if limit < 3 and (limit or not rough):
            cameras, k1 = unpack(x)
            active, count = _clean(observations, cameras, features, k1, float(np.exp(x[3])))
            dropped = count
    cameras, k1 = unpack(x)
    focal = float(np.exp(x[3]))
    errors = [e for e in (_key_error(fs._homography(cameras[k]), active[k], features, k1, focal) for k in keys) if e is not None]
    named = sum(len(lines) for lines in observations.values())
    if not errors or np.median(errors) > MAX_RMS_PX or x[2] <= 0 or dropped > MAX_DROPPED*named:
        return None  # a fit that had to discard this much paint explains the clip by ignoring it
    return _result(keys, cameras, cumulative, meta, k1, float(np.median(errors)), f'dropped={dropped}')


def fit_tripod(observations, cumulative, meta, start=None):
    """Yaw about a vertical axis only, with an optional lens offset from that axis.

    C_i = A + Rz(yaw_i) r. Rotating about the world vertical keeps tilt and roll fixed, so the
    clip has one tilt, one roll and a single yaw per key.
    """
    features = _features(meta)
    start = start or fit_shared(observations, cumulative, meta)
    if start is None:
        return None
    keys = sorted(observations)
    first = np.array([start.cameras[k] for k in keys])
    x = np.r_[first[:, 4:7].mean(0), np.median(first[:, 3]), np.median(first[:, 1]), np.median(first[:, 2]), 0., first[:, 0]]
    k1, focal = start.k1, float(np.exp(x[3]))
    turns = np.diff(first[:, 0])  # the temporally tied fit carries keys that show little paint

    def camera(v, i):
        yaw = v[7+i]
        forward = np.array([np.sin(yaw), -np.cos(yaw)])
        return np.r_[yaw, v[4], v[5], v[3], v[:2]+v[6]*forward, v[2]]

    active = observations
    for _ in range(3):
        cameras = {k: camera(x, i) for i, k in enumerate(keys)}
        active = _clean(observations, cameras, features, k1, focal)[0]
        tied = {k: _tie(fs._homography(cameras[k]), active[k], features, k1, focal) for k in keys}

        def residual(v):
            out = [np.array([2*v[6]]), 1000*CHAIN_WEIGHT*(np.diff(v[7:])-turns)]  # small lens offset
            for i, k in enumerate(keys):
                points, world, weight, _ = tied[k]
                if len(points):
                    out.append(weight*_distance(fs._homography(camera(v, i)), undistort(points, k1, focal), world))
            return np.concatenate(out)

        x = least_squares(residual, x, loss='soft_l1', f_scale=3., max_nfev=30,
                          x_scale=np.r_[.5, .5, .2, .03, .01, .01, .05, np.full(len(keys), .01)]).x
    cameras = {k: camera(x, i) for i, k in enumerate(keys)}
    errors = [e for e in (_key_error(fs._homography(cameras[k]), active[k], features, k1, float(np.exp(x[3])))
                          for k in keys) if e is not None]
    if not errors or np.median(errors) > MAX_RMS_PX:
        return None
    return _result(keys, cameras, cumulative, meta, k1, float(np.median(errors)), f'offset={x[6]:.3f}')


def anchored(calibration, observations, min_lines=4):
    """Drop keys too thin to trust, so the lines ease between solid keys instead of wandering.

    Keys with few lines disagree by far more than the camera moved. Between the remaining keys
    the small difference is spread over the whole stretch. A moving or zooming camera keeps every
    key: there the difference is real.
    """
    kept = ('model=moving', 'model=homography', 'model=zoom')
    if calibration is None or any(model in calibration.notes for model in kept):
        return calibration
    keys = [int(k) for k in calibration.keys]
    solid = [k for k in keys if len(observations[k]) >= min_lines]
    if not solid:
        solid = [max(keys, key=lambda k: sum(len(l['points']) for l in observations[k]))]
    calibration.keys, calibration.H = np.array(solid), {k: calibration.H[k] for k in solid}
    return calibration


def named_paint(c, F, boxes, frames, meta, gate=1.2, far_paint=False):
    """Name the paint a working calibration can see, as input for fit_shared.

    Curves never form straight strokes, so the centre circle and penalty arcs are named from raw
    paint pixels. Without them nothing ties the far side of a wide view to the box it was fitted
    on, and the circle slides as the camera turns.
    """
    features = _features(meta)
    named = {}
    for k in frames:
        found = boxes[k] if boxes is not None else np.empty((0, 5))
        strokes, lawn, loose = fs._strokes(F[k], found, False)
        lines = []
        points = fs._sample(strokes, 8)
        if len(points):
            ground = np.nan_to_num(c.to_pitch(k, points), nan=1e6)
            distance = np.stack([f[0].query(ground)[0] for f in features], 1)
            nearest, close = distance.argmin(1), distance.min(1) < 1.
            for segment in np.unique(nearest[close]):
                group = points[close & (nearest == segment)]
                if len(group) >= 6 and (far_paint or segment != FAR_TOUCHLINE):
                    lines.append({'segment': int(segment), 'points': group, 'confidence': 1.})
        y, x = np.nonzero(loose & lawn)
        paint = np.c_[x, y]*2.
        if len(paint):
            paint = paint[::max(1, len(paint)//4000)]
            ground = np.nan_to_num(c.to_pitch(k, paint), nan=1e6)
            for segment in CURVES:
                near = paint[features[segment][0].query(ground)[0] < gate]
                if len(near) >= 25:
                    lines.append({'segment': segment, 'points': near[::max(1, len(near)//120)], 'confidence': .7})
        edge = fs._far_edge(F[k], found)
        if len(edge):
            lines.append({'segment': FAR_TOUCHLINE, 'points': fs._sample(edge, 16), 'confidence': 1.})
        named[k] = lines
    return named


def refine_camera(c, F, cumulative, meta, boxes, rounds=2, step=15):
    """One camera and lens for the clip, from paint named by a working calibration, or None."""
    frames = [k for k in range(0, len(F), step) if c.supports(k)]
    fitted, position = c, None
    for _ in range(rounds):
        fitted = fit_shared(named_paint(fitted, F, boxes, frames, meta), cumulative, meta)
        if fitted is None:
            return None
    fitted.CUM = pinhole_motion(cumulative, fitted.k1, fitted.focal)
    return fitted


def refine_long_lens(c, F, cumulative, meta, boxes, rounds=2, step=10):
    """One camera, with zoom if the clip needs it, for footage on a long lens, or None."""
    frames = [k for k in range(0, len(F), step) if c.supports(k)]
    fitted = c
    for _ in range(rounds):
        # From a gantry the far touchline is plain paint, with no track edge to find it by.
        fitted = calibrate(named_paint(fitted, F, boxes, frames, meta, far_paint=True), cumulative, meta)
        if fitted is None:
            return None
    return fitted


def calibrate(observations, cumulative, meta):
    """The strictest camera model the clip still fits, tied to the ground through one key."""
    return anchored(_choose(observations, cumulative, meta), observations)


def _choose(observations, cumulative, meta):
    shared = fit_shared(observations, cumulative, meta)
    free = fit_homographies(observations, cumulative, meta)
    if shared is None:
        zoomed = fit_shared(observations, cumulative, meta, zoom=True)
        if zoomed is None:
            return free
        zoomed.notes += ' model=zoom'
        return zoomed
    # A free homography per frame cannot be fooled by a camera that moves, so a much better free
    # fit means the one-position model is wrong for this clip.
    if free is not None and shared.rms > 1.5*free.rms+.5:
        moving = fit_shared(observations, cumulative, meta, temporal=False, free_position=True)
        if moving is not None and moving.rms < 1.3*free.rms+.3:
            moving.notes += ' model=moving'
            return moving
        free.notes += ' model=homography'
        return free
    zoomed = fit_shared(observations, cumulative, meta, zoom=True)
    if zoomed is not None and (np.ptp([c[3] for c in zoomed.cameras.values()]) > ZOOM_DRIFT
                               or zoomed.rms < .85*shared.rms):
        zoomed.notes += ' model=zoom'
        return zoomed
    tripod = fit_tripod(observations, cumulative, meta, shared)
    if tripod is not None and tripod.rms <= 1.1*shared.rms+.1:
        tripod.notes += ' model=tripod'
        return tripod
    shared.notes += ' model=shared'
    return shared
